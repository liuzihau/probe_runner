"""B — repairability: can the lightweight talk reproduce the heavy refinement?

The T3-D question, in hidden space. The talk's anchor is the iter-0 think hidden
(think-once); the heavy refinement it must compress is d_heavy = anchor_converged
− anchor_0 at the block positions. We ask whether the talk's one-shot correction
d_talk = delta_head(talk_hidden) points the same way (and far enough):

  B7  talk-delta vs heavy-deviation alignment — cos(d_talk, d_heavy),
      ‖d_talk − d_heavy‖ / ‖d_heavy‖, grouped by iter-0 readiness.
  B8  talk LM-head fix rate — does lm_head(anchor_0 + d_talk) hit the converged
      token (the localized, per-bucket version of val/acc_still_masked).
  B9  light-probe upper bound — fit a fresh linear map anchor_0 → anchor_converged
      on a prompt-disjoint split; the best ANY light map can do (no T3 arch/training).
  B10 unreachable residual — fraction of ‖d_heavy‖ a linear map can't reach.

Grouping = the iter-0 readiness bucket (A7 connection): "easy" (converged token
already top-1 at iter-0), "medium" (top-10), "hard" (deeper). The headline pairing:
  A7 top-K-early BUT B can't pull it to top-1 ⇒ info present, talk can't extract
  (architecture/training); ~chance early ⇒ info absent ⇒ anchor-refresh.

This script LOADS the trained ThinkTalk checkpoint + reuses the existing capture's
prompts + converged tokens. It needs a GPU + the model — the numpy metric cores
(alignment / linear-probe) are unit-tested via `python -m ...plot_repairability --selftest`.

Usage:
    python -m probe_runner.plots.plot_repairability --model llada2 \
        --talk_ckpt /path/to/global_step_XXXX/hf_ckpt --probes_root probes_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc


# ---- numpy metric cores (unit-testable) -------------------------------------

def alignment_metrics(d_talk: np.ndarray, d_heavy: np.ndarray) -> dict:
    """Per-position B7: cosine(d_talk, d_heavy) and relative error.

    d_talk / d_heavy: [N, D]. Returns {cosine [N], rel_err [N], mag_ratio [N]}.
    """
    d_talk = np.asarray(d_talk, dtype=np.float64)
    d_heavy = np.asarray(d_heavy, dtype=np.float64)
    nt = np.linalg.norm(d_talk, axis=-1)
    nh = np.linalg.norm(d_heavy, axis=-1)
    cos = (d_talk * d_heavy).sum(-1) / np.maximum(nt * nh, 1e-12)
    rel = np.linalg.norm(d_talk - d_heavy, axis=-1) / np.maximum(nh, 1e-12)
    ratio = nt / np.maximum(nh, 1e-12)
    return {"cosine": cos, "rel_err": rel, "mag_ratio": ratio}


def fit_linear_map(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    """Ridge least-squares W minimizing ‖X W − Y‖ (with a bias column). X:[N,D],
    Y:[N,D] → W:[D+1, D]. For the B9 light-probe upper bound."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)         # [N, D+1]
    A = Xb.T @ Xb + ridge * np.eye(Xb.shape[1])
    W = np.linalg.solve(A, Xb.T @ Y)
    return W


def apply_linear_map(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return Xb @ W


def probe_recovery(anchor0: np.ndarray, anchor_conv: np.ndarray,
                   W: np.ndarray) -> dict:
    """B9/B10: how well a linear map from anchor_0 reproduces the heavy delta.

    Predicts anchor_conv from anchor0; reports per-position cosine of the
    predicted vs actual delta (B9) and the unreachable residual fraction (B10).
    """
    pred_conv = apply_linear_map(anchor0, W)
    d_pred = pred_conv - anchor0
    d_true = anchor_conv - anchor0
    nt = np.linalg.norm(d_pred, axis=-1)
    nh = np.linalg.norm(d_true, axis=-1)
    cos = (d_pred * d_true).sum(-1) / np.maximum(nt * nh, 1e-12)
    resid = np.linalg.norm(d_pred - d_true, axis=-1) / np.maximum(nh, 1e-12)
    return {"cosine": cos, "residual_frac": resid}


def readiness_bucket(rank0: np.ndarray) -> np.ndarray:
    """A7-readiness bucket from the iter-0 rank of the converged token:
    0=easy(rank1) 1=medium(<=10) 2=hard(>10)."""
    rank0 = np.asarray(rank0)
    out = np.full(rank0.shape, 2, dtype=np.int8)
    out[rank0 <= 10] = 1
    out[rank0 <= 1] = 0
    return out


BUCKETS = ("easy(r=1)", "medium(r<=10)", "hard(r>10)")


# ---- model driver (needs GPU + ThinkTalk checkpoint) ------------------------

def compute_repairability(talk_ckpt: str, samples: list, *, tokenizer=None,
                          block_length: int = 32, n_samples: int | None = None,
                          t3dmax_root=None, device: str = "cuda") -> dict:
    """Re-run think (iter-0 + converged inputs) + talk (iter-0) per block, using
    the capture's prompts + converged tokens. Returns pooled arrays per position:
    d_talk, d_heavy, anchor0, anchor_conv, rank0, talk_fix(bool). Heavy: needs
    the model. See module docstring.
    """
    import torch
    from probe_runner import configs
    from probe_runner.llada2_runner import _build_block_causal_mask

    configs.add_llada2_to_path(t3dmax_root)
    # load_t3d_model lives in the T3-DMax dInfer decoding module.
    from dinfer.decoding.generate_t3d import load_t3d_model  # type: ignore
    model = load_t3d_model(talk_ckpt, device=device)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(talk_ckpt, trust_remote_code=True)
    think = model.model
    emb = think.word_embeddings if hasattr(think, "word_embeddings") else model.get_input_embeddings()
    dtype = next(model.parameters()).dtype

    def _format(q):
        msg = [{"role": "user", "content": q}]
        p = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        return tokenizer(p, return_tensors="pt")["input_ids"].to(device)

    def _anchor(input_ids, block_end):
        mask = _build_block_causal_mask(block_end, Lp, block_length, dtype, device)
        out = think(input_ids=input_ids[:, :block_end], attention_mask=mask,
                    use_cache=False, output_hidden_states=True, return_dict=True)
        return model.anchor_fuser(out.hidden_states)                  # [1, block_end, D]

    acc = {k: [] for k in ("d_talk", "d_heavy", "anchor0", "anchor_conv", "rank0", "fix")}
    done = 0
    for sample in samples:
        attrs = sample["attrs"]
        prompt_ids = _format(attrs["prompt_text"])
        Lp = int(prompt_ids.shape[1])
        # full converged sequence (prompt + every block's converged tokens)
        nb = max(sample["blocks"].keys()) + 1
        full = torch.full((1, Lp + nb * block_length), tc.MASK_ID, dtype=torch.long, device=device)
        full[0, :Lp] = prompt_ids[0, :Lp]
        for b, blk in sample["blocks"].items():
            conv = torch.tensor(tc.converged_tokens(blk), device=device, dtype=torch.long)
            full[0, Lp + b * block_length: Lp + (b + 1) * block_length] = conv

        for b, blk in sorted(sample["blocks"].items()):
            s, e = Lp + b * block_length, Lp + (b + 1) * block_length
            conv = full[0, s:e]
            iter0 = full.clone()
            iter0[0, s:e] = tc.MASK_ID                              # current block masked
            with torch.no_grad():
                anchor0 = _anchor(iter0, e)                          # [1, e, D]
                anchor_conv = _anchor(full, e)
                a0_blk = anchor0[:, s:e]                             # [1, B, D]
                ac_blk = anchor_conv[:, s:e]
                # talk one-shot on the iter-0 state
                talk_embeds = emb(iter0[:, s:e])
                pos_self = torch.arange(s, e, device=device).unsqueeze(0)
                pos_kv = torch.arange(0, e, device=device).unsqueeze(0)
                talk_hidden = model.talk_model(
                    inputs_embeds=talk_embeds, anchor=a0_blk, attention_mask=None,
                    position_ids=pos_self, anchor_kv=anchor0[:, :e],
                    cross_attention_mask=None, cross_position_ids=pos_kv)
                d_talk = model.delta_head(talk_hidden) if model.delta_head is not None \
                    else (talk_hidden - a0_blk)
                fixed = a0_blk + d_talk
                think_logits0 = model.lm_head(a0_blk)                # iter-0 think logits
                fix_tok = model.lm_head(fixed).argmax(-1)            # talk's prediction
            d_heavy = (ac_blk - a0_blk)[0].float().cpu().numpy()
            d_talk_np = d_talk[0].float().cpu().numpy()
            a0_np = a0_blk[0].float().cpu().numpy()
            ac_np = ac_blk[0].float().cpu().numpy()
            conv_np = conv.cpu().numpy()
            # iter-0 rank of converged token (think logits)
            tl = think_logits0[0].float().cpu().numpy()
            rank0 = (tl > tl[np.arange(len(conv_np)), conv_np][:, None]).sum(-1) + 1
            fix = (fix_tok[0].cpu().numpy() == conv_np)
            valid = tc.valid_position_mask(b, e - s, attrs, block_length)
            acc["d_talk"].append(d_talk_np[valid])
            acc["d_heavy"].append(d_heavy[valid])
            acc["anchor0"].append(a0_np[valid])
            acc["anchor_conv"].append(ac_np[valid])
            acc["rank0"].append(rank0[valid])
            acc["fix"].append(fix[valid])
        done += 1
        if n_samples and done >= n_samples:
            break
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in acc.items()}


# ---- report + plot ----------------------------------------------------------

def plot(data: dict, out_dir: Path, model: str) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    d_talk, d_heavy = data["d_talk"], data["d_heavy"]
    rank0 = data["rank0"]
    bucket = readiness_bucket(rank0)
    align = alignment_metrics(d_talk, d_heavy)

    # B9/B10: fit a light probe on half, evaluate on the other half.
    a0, ac = data["anchor0"], data["anchor_conv"]
    n = a0.shape[0]
    half = n // 2
    W = fit_linear_map(a0[:half], ac[:half])
    rec = probe_recovery(a0[half:], ac[half:], W)
    b_test = bucket[half:]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    # B7 alignment by bucket
    ax = axes[0]
    xs = np.arange(len(BUCKETS))
    cos_by = [float(np.nanmean(align["cosine"][bucket == k])) if (bucket == k).any() else np.nan
              for k in range(len(BUCKETS))]
    fix_by = [float(data["fix"][bucket == k].mean()) if (bucket == k).any() else np.nan
              for k in range(len(BUCKETS))]
    ax.bar(xs - 0.2, cos_by, width=0.4, color="tab:blue", label="B7 cos(d_talk,d_heavy)")
    ax.bar(xs + 0.2, fix_by, width=0.4, color="tab:green", label="B8 talk fix-rate")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(BUCKETS, fontsize=8)
    ax.set_title("B7 alignment + B8 fix-rate by iter-0 readiness")
    ax.set_ylabel("cosine / fix-rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # B7 magnitude ratio (under-correction diagnosis)
    ax = axes[1]
    for k in range(len(BUCKETS)):
        sel = bucket == k
        if sel.any():
            ax.hist(np.clip(align["mag_ratio"][sel], 0, 3), bins=30, alpha=0.5,
                    density=True, label=BUCKETS[k])
    ax.axvline(1.0, ls="--", color="black", label="‖d_talk‖=‖d_heavy‖")
    ax.set_title("B7 magnitude ratio ‖d_talk‖/‖d_heavy‖\n(<1 = under-correction)")
    ax.set_xlabel("magnitude ratio")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    # B9/B10 light-probe ceiling vs talk, by bucket
    ax = axes[2]
    probe_cos = [float(np.nanmean(rec["cosine"][b_test == k])) if (b_test == k).any() else np.nan
                 for k in range(len(BUCKETS))]
    ax.bar(xs - 0.2, probe_cos, width=0.4, color="tab:purple", label="B9 light-probe cos")
    talk_cos_test = [float(np.nanmean(alignment_metrics(d_talk[half:], d_heavy[half:])["cosine"][b_test == k]))
                     if (b_test == k).any() else np.nan for k in range(len(BUCKETS))]
    ax.bar(xs + 0.2, talk_cos_test, width=0.4, color="tab:blue", label="talk cos (B7)")
    ax.set_xticks(xs)
    ax.set_xticklabels(BUCKETS, fontsize=8)
    ax.set_title("B9 light-probe upper bound vs talk\n(probe ≫ talk ⇒ talk arch/training-limited)")
    ax.set_ylabel("cosine to d_heavy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"{model}: B repairability  [N positions={d_talk.shape[0]}]")
    fig.tight_layout()
    p = out_dir / "repairability.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")


def _selftest():
    """Unit-test the numpy cores (no model)."""
    rng = np.random.default_rng(0)
    N, D = 200, 16
    d_heavy = rng.standard_normal((N, D))
    # d_talk = 0.6*d_heavy (aligned, under-corrected) + noise
    d_talk = 0.6 * d_heavy + 0.1 * rng.standard_normal((N, D))
    a = alignment_metrics(d_talk, d_heavy)
    assert a["cosine"].mean() > 0.9, a["cosine"].mean()
    assert 0.4 < a["mag_ratio"].mean() < 0.8, a["mag_ratio"].mean()
    # linear map should recover a linear relationship well
    anchor0 = rng.standard_normal((N, D))
    Wtrue = rng.standard_normal((D, D))
    anchor_conv = anchor0 @ Wtrue + 0.01 * rng.standard_normal((N, D))
    W = fit_linear_map(anchor0[:150], anchor_conv[:150])
    rec = probe_recovery(anchor0[150:], anchor_conv[150:], W)
    assert np.nanmean(rec["cosine"]) > 0.95, np.nanmean(rec["cosine"])
    assert readiness_bucket(np.array([1, 5, 50])).tolist() == [0, 1, 2]
    print("plot_repairability selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llada2")
    ap.add_argument("--probes_root", default="probes_out")
    ap.add_argument("--talk_ckpt", default=None,
                    help="ThinkTalk checkpoint dir (global_step_XXXX/hf_ckpt).")
    ap.add_argument("--t3dmax_root", default=None)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.talk_ckpt:
        raise SystemExit("--talk_ckpt is required (the trained ThinkTalk checkpoint).")
    samples = [tc.load_sample(p) for p in tc.iter_sample_paths(args.probes_root, args.model)]
    samples = [s for s in samples if tc.has_committed(s["blocks"])]
    data = compute_repairability(args.talk_ckpt, samples, block_length=args.block_length,
                                 n_samples=args.n_samples, t3dmax_root=args.t3dmax_root)
    plot(data, Path(args.probes_root) / args.model / "plots", args.model)


if __name__ == "__main__":
    main()
