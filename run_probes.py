"""Main entry point: record probe data for one model on the first 100 GSM8K test problems.

Usage (from T3 project root):
    python -m probe_runner.run_probes --model llada
    python -m probe_runner.run_probes --model dream
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch
from datasets import load_dataset

from probe_runner import configs, hooks as hooks_mod, storage


def _resolve_model_dims(model, model_type: str) -> dict:
    """Resolve n_layers / n_heads / d_model / d_head from a loaded model."""
    out = {"model_type": model_type}
    if model_type == "llada":
        cfg = model.config
        out["n_layers"] = int(cfg.n_layers)
        out["n_heads"] = int(cfg.n_heads)
        out["d_model"] = int(cfg.d_model)
        out["d_head"] = int(cfg.d_model) // int(cfg.n_heads)
        out["n_kv_heads"] = int(getattr(cfg, "effective_n_kv_heads", cfg.n_heads))
    elif model_type == "dream":
        cfg = model.config
        out["n_layers"] = int(cfg.num_hidden_layers)
        out["n_heads"] = int(cfg.num_attention_heads)
        out["d_model"] = int(cfg.hidden_size)
        out["d_head"] = int(cfg.hidden_size) // int(cfg.num_attention_heads)
        out["n_kv_heads"] = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    elif model_type == "llada2":
        cfg = model.config
        out["n_layers"] = int(cfg.num_hidden_layers)
        out["n_heads"] = int(cfg.num_attention_heads)
        out["d_model"] = int(cfg.hidden_size)
        kv = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        out["d_head"] = int(getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads))
        out["n_kv_heads"] = int(kv)
    else:
        raise ValueError(model_type)
    return out


def _format_prompt_llada(tokenizer, question: str) -> torch.Tensor:
    """Apply LLaDA's chat template, no few-shot. Returns [1, S] long tensor on CUDA."""
    msg = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(prompt)["input_ids"]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).cuda()


def _format_prompt_dream(tokenizer, question: str) -> torch.Tensor:
    msg = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return ids.cuda()


def _format_prompt_llada2(tokenizer, question: str) -> torch.Tensor:
    """LLaDA-2.0 / DMax chat template, no few-shot. Returns [1, S] long on CUDA."""
    msg = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return ids.cuda()


def _export_llada2_head(model, out_dir: Path) -> Path:
    """Save the final RMSNorm + lm_head weights so the rank / repairability
    analyses can map captured hidden → logits WITHOUT reloading the 16B model.

    Writes `<out_dir>/lm_head.pt` = {lm_head_weight [V,D] f16, norm_weight [D]
    f16, rms_norm_eps, d_model, vocab_size}. ~320 MB (f16). Re-applied offline
    as: logits = (h / rms(h) * norm_weight) @ lm_head_weight.T .
    """
    base = getattr(model, "model", model)
    norm = getattr(base, "norm", None)
    head = getattr(model, "lm_head", None)
    if norm is None or head is None:
        print("[llada2] could not locate norm / lm_head — skipping head export")
        return out_dir / "lm_head.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lm_head.pt"
    torch.save(
        {
            "lm_head_weight": head.weight.detach().to(torch.float16).cpu(),
            "norm_weight": norm.weight.detach().to(torch.float16).cpu(),
            "rms_norm_eps": float(getattr(model.config, "rms_norm_eps", 1e-6)),
            "d_model": int(model.config.hidden_size),
            "vocab_size": int(model.config.vocab_size),
        },
        path,
    )
    print(f"[llada2] exported final-norm + lm_head -> {path}")
    return path


def _check_first_block_sanity(buf_data: dict, prompt_len: int, num_blocks: int, n_layers: int) -> dict:
    """Run §6 sanity checks on block 0 of the first sample. Returns a dict of check results."""
    if 0 not in buf_data:
        return {"ok": False, "reason": "block_0 missing"}
    block_0 = buf_data[0]
    h = block_0.get("h_masked")  # [L+1, num_masked, d_model]
    res = {}

    res["all_blocks_recorded"] = sum(1 for b in range(num_blocks) if b in buf_data)
    res["h_shape"] = list(h.shape) if h is not None else None
    h_ok = h is not None and h.shape[0] == n_layers + 1

    attn = block_0.get("attn")  # None on hidden-only (sdpa) runs
    if attn is None:
        res["attn"] = "not captured (hidden-only run; load eager to capture attn)"
        res["ok"] = bool(h_ok)
        return res

    res["attn_shape"] = list(attn.shape)
    res["S_0"] = attn.shape[-1]
    res["expected_S_0"] = prompt_len + 32

    # Attention rows sum to 1
    row_sums = attn.float().sum(dim=-1)  # [num_masked, L, H]
    res["attn_row_sum_min"] = float(row_sums.min())
    res["attn_row_sum_max"] = float(row_sums.max())

    # Sink detection
    pos0_mass = attn[..., 0].float().mean().item()
    res["mean_attn_to_pos0"] = pos0_mass
    res["likely_sink_pos_0"] = pos0_mass > 0.1

    res["ok"] = bool(
        abs(res["attn_row_sum_min"] - 1.0) < 1e-2
        and abs(res["attn_row_sum_max"] - 1.0) < 1e-2
        and h_ok
    )
    return res


def run_for_model(model_type: str, *, n_samples: int = 100, output_root: Path | None = None,
                  max_prompt_tokens: int = 512, gen_length: int = 256, block_length: int = 32,
                  steps: int = 256, threshold: float = 0.9,
                  fast_dllm_path: str | Path | None = None,
                  intra_block: bool = False,
                  prompt_kv: bool = False,
                  # llada2 / DMax-only:
                  model_path: str | Path | None = None,
                  t3dmax_root: str | Path | None = None,
                  attn_impl: str = "sdpa",
                  decode_mode: str = "soft",
                  commit_threshold: float | None = None,
                  break_threshold: float = 0.9,
                  max_iter_per_block: int | None = None,
                  export_head: bool = True) -> dict:
    output_root = output_root or Path(configs.PROBE_CONFIG["output"]["root"])
    out_dir = output_root / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # Commit threshold default depends on the decode rule: DMax decode_uniform
    # uses 0.3; LLaDA-2.0-mini's native threshold decode uses 0.9 (dInfer default).
    if commit_threshold is None:
        commit_threshold = 0.9 if decode_mode == "threshold" else 0.3

    eos_token_id = None

    # 1. Load model + tokenizer
    print(f"[{model_type}] loading model …")
    if model_type == "llada":
        from probe_runner.llada_runner import load_llada, generate_with_probes
        model, tokenizer = load_llada(fast_dllm_path=fast_dllm_path)
        format_prompt = _format_prompt_llada
        mask_token_id = configs.PROBE_CONFIG["models"]["llada"]["mask_token_id"]
    elif model_type == "llada2":
        if model_path is None:
            model_path = (os.environ.get("T3DMAX_LLADA2_MODEL")
                          or os.environ.get("T3DMAX_DMAX_MODEL"))
        if not model_path:
            raise ValueError(
                "llada2 needs --model_path (or env T3DMAX_LLADA2_MODEL / "
                "T3DMAX_DMAX_MODEL) pointing at the LLaDA-2.0-mini / DMax-Math-16B weights.")
        from probe_runner.llada2_runner import load_llada2, generate_with_probes
        model, tokenizer = load_llada2(model_path, attn_implementation=attn_impl,
                                       t3dmax_root=t3dmax_root)
        format_prompt = _format_prompt_llada2
        m2 = configs.PROBE_CONFIG["models"]["llada2"]
        mask_token_id = m2["mask_token_id"]
        eos_token_id = m2["eos_token_id"]
        if export_head:
            _export_llada2_head(model, out_dir)
    else:
        from probe_runner.dream_runner import load_dream, generate_with_probes
        model, tokenizer = load_dream(fast_dllm_path=fast_dllm_path)
        format_prompt = _format_prompt_dream
        mask_token_id = getattr(model.config, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            raise RuntimeError("Could not infer mask_token_id for Dream")

    dims = _resolve_model_dims(model, model_type)
    print(f"[{model_type}] n_layers={dims['n_layers']} n_heads={dims['n_heads']} d_model={dims['d_model']}")

    # 2. Load dataset
    ds = load_dataset(configs.PROBE_CONFIG["dataset"]["name"],
                      configs.PROBE_CONFIG["dataset"]["config"],
                      split=configs.PROBE_CONFIG["dataset"]["split"]).select(range(n_samples))

    # 3. Run loop
    num_blocks = gen_length // block_length
    record_blocks_cfg = configs.PROBE_CONFIG["probe"]["record_blocks"]
    if record_blocks_cfg == "all":
        record_blocks_set = set(range(num_blocks))
    else:
        record_blocks_set = set(record_blocks_cfg)

    v_norm_blocks_cfg = configs.PROBE_CONFIG["probe"]["v_norm_blocks"]
    if v_norm_blocks_cfg == "all":
        v_norm_blocks_set = set(range(num_blocks))
    else:
        v_norm_blocks_set = set(v_norm_blocks_cfg)

    sanity: dict = {}
    sample_runtimes = []

    for sample_idx, problem in enumerate(ds):
        question = problem["question"]
        gold_answer = problem["answer"]
        out_path = out_dir / f"sample_{sample_idx:04d}.h5"
        if out_path.exists():
            print(f"[{model_type}] skipping sample {sample_idx} (already exists)")
            continue

        # Tokenize prompt; truncate if too long
        prompt_ids = format_prompt(tokenizer, question)
        if prompt_ids.shape[1] > max_prompt_tokens:
            prompt_ids = prompt_ids[:, -max_prompt_tokens:]
        prompt_len = int(prompt_ids.shape[1])

        # Identify special-token positions in the prompt (chat-template tokens, BOS, etc.)
        prompt_ids_list = prompt_ids[0].tolist()
        try:
            special_ids_set = set(tokenizer.all_special_ids or [])
        except Exception:
            special_ids_set = set()
        # Always treat position 0 as a sink (BOS / first-token attention sink heuristic)
        special_token_positions = sorted(
            {0} | {i for i, t in enumerate(prompt_ids_list) if t in special_ids_set}
        )

        # Install probe hooks
        hooks = hooks_mod.ProbeHooks(
            model,
            model_type=model_type,
            n_layers=dims["n_layers"],
            n_heads=dims["n_heads"],
            d_head=dims["d_head"],
            record_v_norm=True,  # we toggle per-block via a wrapper below
            intra_block=intra_block,
            record_prompt_kv=prompt_kv,
        )
        if prompt_kv:
            # Same prompt indices reused across all 8 blocks; pass-0-only
            # capture handled inside the hooks.
            hooks.set_prompt_indices(list(range(prompt_len)))

        def on_block_start(block_idx: int, masked_positions_abs: list[int]):
            if block_idx not in record_blocks_set:
                hooks.armed = False
                return
            hooks.set_block(block_idx, masked_positions_abs)
            hooks.record_v_norm = block_idx in v_norm_blocks_set
            hooks.armed = True

        def on_block_end(block_idx: int):
            hooks.armed = False

        def on_pass_start(block_idx: int, pass_idx: int, token_state, indices_in_forward,
                          committed_tokens=None):
            if block_idx not in record_blocks_set:
                return
            hooks.set_pass(
                block_idx, pass_idx,
                token_state=token_state,
                indices=indices_in_forward,
                committed_tokens=committed_tokens,
            )
            hooks.armed = True

        def on_pass_end(block_idx: int, pass_idx: int, revealed_this_pass):
            if block_idx not in record_blocks_set:
                return
            hooks.finalize_pass(revealed_this_pass=revealed_this_pass)
            hooks.armed = False

        try:
            t0 = time.time()
            with torch.inference_mode():
                gen_kwargs = dict(
                    on_block_start=on_block_start,
                    on_block_end=on_block_end,
                    on_pass_start=on_pass_start,
                    on_pass_end=on_pass_end,
                    intra_block=intra_block,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    threshold=threshold,
                    temperature=0.0,
                )
                if model_type == "dream":
                    output_ids, nfe = generate_with_probes(
                        model, prompt_ids, mask_token_id=mask_token_id, **gen_kwargs,
                    )
                elif model_type == "llada2":
                    # decode_uniform runner; ignores the Fast-dLLM steps/threshold
                    # in gen_kwargs and uses its own commit/break thresholds.
                    output_ids, nfe = generate_with_probes(
                        model, prompt_ids, mask_id=mask_token_id, eos_id=eos_token_id,
                        decode_mode=decode_mode, commit_threshold=commit_threshold,
                        break_threshold=break_threshold, max_iter_per_block=max_iter_per_block,
                        **gen_kwargs,
                    )
                else:
                    output_ids, nfe = generate_with_probes(
                        model, prompt_ids, mask_id=mask_token_id, **gen_kwargs,
                    )
            dt = time.time() - t0
            sample_runtimes.append(dt)

            data_per_block = hooks.collect()
            intra_block_per_block = hooks.collect_intra_block() if intra_block else None
            # Attach each block's converged (final) token ids — the rank / flip
            # reference. Post the last commit, so taken from output_ids (not
            # derivable from the per-pass input tokens).
            if intra_block_per_block:
                for b in list(intra_block_per_block.keys()):
                    bs = prompt_len + block_length * b
                    be = bs + block_length
                    intra_block_per_block[b]["converged_tokens"] = output_ids[0, bs:be].detach().cpu()
            generated_ids = output_ids[0, prompt_ids.shape[1]:].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Find first EOS-like token in generated portion. Multiple ids may count as end-of-answer.
            end_token_ids: set[int] = set()
            if getattr(tokenizer, "eos_token_id", None) is not None:
                end_token_ids.add(int(tokenizer.eos_token_id))
            if getattr(tokenizer, "pad_token_id", None) is not None:
                end_token_ids.add(int(tokenizer.pad_token_id))
            eos_pos_in_generated = len(generated_ids)  # default: never emitted
            for i, t in enumerate(generated_ids):
                if t in end_token_ids:
                    eos_pos_in_generated = i
                    break

            # Build per-block metadata
            block_seq_lens = []
            block_mask_positions = []
            for b in sorted(data_per_block.keys()):
                S_b = prompt_len + block_length * (b + 1)
                block_seq_lens.append(S_b)
                block_mask_positions.append(
                    list(range(prompt_len + block_length * b, prompt_len + block_length * (b + 1))))

            storage.write_h5(
                out_path,
                data_per_block,
                prompt_text=question,
                gold_answer=gold_answer,
                generated_text=generated_text,
                model_name=model_type,
                n_layers=dims["n_layers"],
                n_heads=dims["n_heads"],
                d_model=dims["d_model"],
                prompt_len=prompt_len,
                num_masked=block_length,
                block_seq_lens=block_seq_lens,
                block_mask_positions=block_mask_positions,
                attention_sink_positions=[0],  # default — refined by sanity check
                special_token_positions=special_token_positions,
                eos_pos_in_generated=eos_pos_in_generated,
                intra_block_per_block=intra_block_per_block,
            )

            if sample_idx == 0:
                sanity = _check_first_block_sanity(data_per_block, prompt_len, num_blocks, dims["n_layers"])
                print(f"[{model_type}] sanity sample 0: {json.dumps(sanity, indent=2)}")

            print(f"[{model_type}] sample {sample_idx:03d} done in {dt:.1f}s, nfe={nfe}")
        except Exception as e:
            traceback.print_exc()
            print(f"[{model_type}] sample {sample_idx} FAILED: {e}")
        finally:
            hooks.remove()

    return {
        "model_type": model_type,
        "n_samples_done": len(list(out_dir.glob("sample_*.h5"))),
        "mean_runtime_s": float(sum(sample_runtimes) / max(len(sample_runtimes), 1)),
        "sanity_sample_0": sanity,
        "dims": dims,
    }


def write_meta(model_summaries: dict, output_root: Path):
    """Write probes_out/meta.json — see step1to4 §4.3."""
    import platform
    meta = {
        "config": configs.PROBE_CONFIG,
        "platform": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(),
        },
        "models": model_summaries,
        "attn_implementation": "manual_softmax",
        "dream_adapter_notes": "Dream uses Fast-dLLM v1's `model.modeling_dream.DreamModel` "
                                "(supports dual_cache + replace_position the same way LLaDA does).",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    storage.write_meta(output_root / "meta.json", meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llada", "dream", "llada2", "both"], default="both")
    parser.add_argument("--n_samples", type=int, default=configs.PROBE_CONFIG["dataset"]["n_samples"])
    parser.add_argument("--output_root", type=str, default=configs.PROBE_CONFIG["output"]["root"])
    parser.add_argument(
        "--fast_dllm_path",
        type=str,
        default=None,
        help="Path to Fast-dLLM v1 (the dir that contains llada/ and dream/). "
             "Defaults to env FAST_DLLM_V1_PATH or ./external/Fast-dLLM/v1.",
    )
    parser.add_argument(
        "--intra_block",
        action="store_true",
        help="Record per-pass hidden states within each block (in addition to "
             "the existing pass-0 attn / v_norm / h_masked). Adds ~470 MB per "
             "sample of fp16 h_per_pass; needed for the intra-block drift / "
             "diff-heatmap analyses.",
    )
    parser.add_argument(
        "--prompt_kv",
        action="store_true",
        help="Record per-block per-layer post-RoPE K/V at prompt positions "
             "(pass-0 only). Each block's pass-0 forward freshly recomputes "
             "the prompt-region KV under Fast-dLLM's dual-cache, so this "
             "captures whether that recomputed prompt KV drifts block-to-block. "
             "Adds ~256 MB / sample (LLaDA, no GQA, prompt_len=512) or "
             "~32 MB / sample (Dream, GQA factor 8). Required by "
             "plots.plot_prompt_kv_drift.",
    )
    # ---- llada2 / DMax-Math-16B options ----
    parser.add_argument("--model_path", type=str, default=None,
                        help="LLaDA-2.0-mini / DMax-Math-16B weight dir (llada2 only). "
                             "Falls back to env T3DMAX_LLADA2_MODEL / T3DMAX_DMAX_MODEL.")
    parser.add_argument("--t3dmax_root", type=str, default=None,
                        help="T3-DMax checkout root (for the LLaDA2MoeModelLM source). "
                             "Falls back to env T3DMAX_ROOT or the default relative path.")
    parser.add_argument("--attn_impl", choices=["sdpa", "eager"], default="sdpa",
                        help="llada2 attention impl. 'eager' lets the probe capture "
                             "attn/v_norm; 'sdpa' (default) is faster, hidden-only.")
    parser.add_argument("--decode_mode", choices=["soft", "threshold", "hard"], default="soft",
                        help="llada2 decode rule. 'soft' = DMax decode_uniform "
                             "(contiguous prefix + soft mix, revisable → CR domain); "
                             "'threshold' = LLaDA-2.0-mini NATIVE (global confidence-"
                             "threshold parallel, hard, fixed) — USE FOR THE MINI BASELINE; "
                             "'hard' = contiguous+hard diagnostic (not a native decode).")
    parser.add_argument("--commit_threshold", type=float, default=None,
                        help="Commit threshold. Default: 0.3 for soft/hard (DMax "
                             "decode_uniform), 0.9 for threshold (mini native).")
    parser.add_argument("--break_threshold", type=float, default=0.9)
    parser.add_argument("--max_iter_per_block", type=int, default=None)
    parser.add_argument("--no_export_head", action="store_true",
                        help="Skip exporting lm_head+norm (llada2). The rank / "
                             "repairability analyses then need the full model.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = {}
    targets = ["llada", "dream"] if args.model == "both" else [args.model]

    # Validate Fast-dLLM only if a Fast-dLLM-based model is targeted.
    if any(m in ("llada", "dream") for m in targets):
        fdp = configs.resolve_fast_dllm_path(args.fast_dllm_path)
        print(f"Using Fast-dLLM v1 at: {fdp}")

    for m in targets:
        summaries[m] = run_for_model(
            m,
            n_samples=args.n_samples,
            output_root=output_root,
            fast_dllm_path=args.fast_dllm_path,
            intra_block=args.intra_block,
            prompt_kv=args.prompt_kv,
            model_path=args.model_path,
            t3dmax_root=args.t3dmax_root,
            attn_impl=args.attn_impl,
            decode_mode=args.decode_mode,
            commit_threshold=args.commit_threshold,
            break_threshold=args.break_threshold,
            max_iter_per_block=args.max_iter_per_block,
            export_head=not args.no_export_head,
        )

    write_meta(summaries, output_root)
    print("Done.")
    print(json.dumps(summaries, indent=2, default=str))


if __name__ == "__main__":
    main()
