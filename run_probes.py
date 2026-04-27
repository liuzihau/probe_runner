"""Main entry point: record probe data for one model on the first 100 GSM8K test problems.

Usage (from T3 project root):
    python -m probe_runner.run_probes --model llada
    python -m probe_runner.run_probes --model dream
"""

from __future__ import annotations

import argparse
import json
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


def _check_first_block_sanity(buf_data: dict, prompt_len: int, num_blocks: int, n_layers: int) -> dict:
    """Run §6 sanity checks on block 0 of the first sample. Returns a dict of check results."""
    if 0 not in buf_data:
        return {"ok": False, "reason": "block_0 missing"}
    block_0 = buf_data[0]
    attn = block_0["attn"]      # [num_masked, L, H, S_0]
    h = block_0["h_masked"]      # [L+1, num_masked, d_model]
    res = {}

    res["all_blocks_recorded"] = sum(1 for b in range(num_blocks) if b in buf_data)
    res["attn_shape"] = list(attn.shape)
    res["h_shape"] = list(h.shape)
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

    res["ok"] = (
        attn.shape[-1] == prompt_len + 32
        and abs(res["attn_row_sum_min"] - 1.0) < 1e-2
        and abs(res["attn_row_sum_max"] - 1.0) < 1e-2
        and h.shape[0] == n_layers + 1
    )
    return res


def run_for_model(model_type: str, *, n_samples: int = 100, output_root: Path | None = None,
                  max_prompt_tokens: int = 512, gen_length: int = 256, block_length: int = 32,
                  steps: int = 256, threshold: float = 0.9,
                  fast_dllm_path: str | Path | None = None) -> dict:
    output_root = output_root or Path(configs.PROBE_CONFIG["output"]["root"])
    out_dir = output_root / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model + tokenizer
    print(f"[{model_type}] loading model …")
    if model_type == "llada":
        from probe_runner.llada_runner import load_llada, generate_with_probes
        model, tokenizer = load_llada(fast_dllm_path=fast_dllm_path)
        format_prompt = _format_prompt_llada
        mask_token_id = configs.PROBE_CONFIG["models"]["llada"]["mask_token_id"]
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

        # Install probe hooks
        hooks = hooks_mod.ProbeHooks(
            model,
            model_type=model_type,
            n_layers=dims["n_layers"],
            n_heads=dims["n_heads"],
            d_head=dims["d_head"],
            record_v_norm=True,  # we toggle per-block via a wrapper below
        )

        def on_block_start(block_idx: int, masked_positions_abs: list[int]):
            if block_idx not in record_blocks_set:
                hooks.armed = False
                return
            hooks.set_block(block_idx, masked_positions_abs)
            hooks.record_v_norm = block_idx in v_norm_blocks_set
            hooks.armed = True

        def on_block_end(block_idx: int):
            hooks.armed = False

        try:
            t0 = time.time()
            with torch.inference_mode():
                output_ids, nfe = generate_with_probes(
                    model,
                    prompt_ids,
                    on_block_start=on_block_start,
                    on_block_end=on_block_end,
                    mask_token_id=mask_token_id,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    threshold=threshold,
                    temperature=0.0,
                ) if model_type == "dream" else generate_with_probes(
                    model,
                    prompt_ids,
                    on_block_start=on_block_start,
                    on_block_end=on_block_end,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    threshold=threshold,
                    temperature=0.0,
                    mask_id=mask_token_id,
                )
            dt = time.time() - t0
            sample_runtimes.append(dt)

            data_per_block = hooks.collect()
            generated_ids = output_ids[0, prompt_ids.shape[1]:].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Build per-block metadata
            block_seq_lens = []
            block_mask_positions = []
            for b in sorted(data_per_block.keys()):
                S_b = prompt_len + 32 * (b + 1)
                block_seq_lens.append(S_b)
                block_mask_positions.append(list(range(prompt_len + 32 * b, prompt_len + 32 * (b + 1))))

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
    parser.add_argument("--model", choices=["llada", "dream", "both"], default="both")
    parser.add_argument("--n_samples", type=int, default=configs.PROBE_CONFIG["dataset"]["n_samples"])
    parser.add_argument("--output_root", type=str, default=configs.PROBE_CONFIG["output"]["root"])
    parser.add_argument(
        "--fast_dllm_path",
        type=str,
        default=None,
        help="Path to Fast-dLLM v1 (the dir that contains llada/ and dream/). "
             "Defaults to env FAST_DLLM_V1_PATH or ./external/Fast-dLLM/v1.",
    )
    args = parser.parse_args()

    # Validate Fast-dLLM is reachable BEFORE we start loading large models.
    fdp = configs.resolve_fast_dllm_path(args.fast_dllm_path)
    print(f"Using Fast-dLLM v1 at: {fdp}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = {}
    targets = ["llada", "dream"] if args.model == "both" else [args.model]
    for m in targets:
        summaries[m] = run_for_model(
            m,
            n_samples=args.n_samples,
            output_root=output_root,
            fast_dllm_path=args.fast_dllm_path,
        )

    write_meta(summaries, output_root)
    print("Done.")
    print(json.dumps(summaries, indent=2, default=str))


if __name__ == "__main__":
    main()
