"""Plot logit-lens metrics: how similar each layer's projected distribution is to the final layer.

For each masked position, project the per-layer hidden state through the model's final norm and
LM head:

    p_ℓ = softmax( lm_head( final_norm( h_masked[ℓ] ) ) )

Then compare p_ℓ to p_L using three metrics, all averaged over masked positions / samples / blocks:

  1. **top_k_overlap** (default K=5): |topk(p_ℓ) ∩ topk(p_L)| / K  ∈ [0, 1]
  2. **kl_divergence**: KL(p_ℓ || p_L)  ∈ [0, ∞), 0 = identical
  3. **shared_mass**:  Σ_x min(p_ℓ(x), p_L(x))  ∈ [0, 1], 1 = identical
                      (equivalent to 1 - 0.5 × ||p_ℓ - p_L||₁ — total variation overlap)

The plot answers: "at which layer ℓ is the model's *output distribution* effectively the same as
the final layer's?" Layers above that point are doing only minor tweaks to the prediction and are
strong prune candidates.

Run from the directory containing probe_runner/:
    python -m probe_runner.plots.plot_logit_lens --model llada
    python -m probe_runner.plots.plot_logit_lens --model dream
    python -m probe_runner.plots.plot_logit_lens --model llada --top_k 10

Cost: needs the model's final norm + LM head loaded. ~5–10 min total runtime on a 3090
(model load + 100 samples × 8 blocks × 33 layers of projection).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from probe_runner import storage


# ---------------------------------------------------------------------------
# Locate final norm + LM head in the loaded model
# ---------------------------------------------------------------------------

def _get_llada_final_norm_and_head(model):
    """LLaDA: model.transformer.{ln_f, ff_out (or wte if tied)}."""
    # Some wrappers nest the actual transformer one level deeper
    candidates = [model, getattr(model, "model", None)]
    for c in candidates:
        if c is None:
            continue
        t = getattr(c, "transformer", None)
        if t is None:
            continue
        # transformer is a ModuleDict
        ln_f = t["ln_f"] if isinstance(t, nn.ModuleDict) and "ln_f" in t else getattr(t, "ln_f", None)
        ff_out = t["ff_out"] if isinstance(t, nn.ModuleDict) and "ff_out" in t else getattr(t, "ff_out", None)
        wte = t["wte"] if isinstance(t, nn.ModuleDict) and "wte" in t else getattr(t, "wte", None)
        if ln_f is None or wte is None:
            continue
        if ff_out is None:
            # Tied LM head: use wte as a linear projection
            head = nn.Linear(wte.embedding_dim, wte.num_embeddings, bias=False).to(
                next(model.parameters()).device, dtype=next(model.parameters()).dtype
            )
            head.weight = wte.weight
            return ln_f, head
        return ln_f, ff_out
    raise RuntimeError("Could not locate ln_f / ff_out in LLaDA model.")


def _get_dream_final_norm_and_head(model):
    """Dream: a Llama-style decoder. final_norm at model.model.norm, lm_head at model.lm_head."""
    base = getattr(model, "model", None) or model
    norm = getattr(base, "norm", None) or getattr(model, "norm", None)
    head = getattr(model, "lm_head", None)
    if norm is None or head is None:
        # Tied embedding fallback
        embed = getattr(base, "embed_tokens", None) or getattr(model, "embed_tokens", None)
        if norm is None or embed is None:
            raise RuntimeError("Could not locate norm / lm_head / embed_tokens in Dream model.")
        head = nn.Linear(embed.embedding_dim, embed.num_embeddings, bias=False).to(
            next(model.parameters()).device, dtype=next(model.parameters()).dtype
        )
        head.weight = embed.weight
    return norm, head


def _load_model_components(model_type: str, fast_dllm_path: str | None):
    if model_type == "llada":
        from probe_runner.llada_runner import load_llada
        model, _ = load_llada(fast_dllm_path=fast_dllm_path)
        norm, head = _get_llada_final_norm_and_head(model)
    elif model_type == "dream":
        from probe_runner.dream_runner import load_dream
        model, _ = load_dream(fast_dllm_path=fast_dllm_path)
        norm, head = _get_dream_final_norm_and_head(model)
    else:
        raise ValueError(model_type)
    model.eval()
    return model, norm, head


# ---------------------------------------------------------------------------
# Per-layer metric computation
# ---------------------------------------------------------------------------

def _compute_logit_lens_metrics(
    files: list[Path],
    model_type: str,
    fast_dllm_path: str | None,
    top_k: int = 5,
    apply_eos_cutoff: bool = True,
    apply_special_filter: bool = True,
):
    """Returns:
        per_block: {block_idx: {metric: np.ndarray of shape [L+1]}}
        pooled:    {metric: np.ndarray of shape [L+1]}
        diagnostics dict.
    """
    print(f"[{model_type}] loading model for logit-lens projection (this can take a few minutes) …")
    model, final_norm, lm_head = _load_model_components(model_type, fast_dllm_path)
    device = next(model.parameters()).device
    dtype_proj = next(model.parameters()).dtype  # bf16 typically

    # Lazily import filter helpers (mirrors plot_info_flow_to_prefix behavior)
    from probe_runner.plots.plot_info_flow_to_prefix import (
        _resolve_special_positions, _resolve_eos_pos, _allowed_masked_positions,
    )

    accum_per_block: dict[int, dict[str, list[np.ndarray]]] = {}
    diagnostics = {
        "n_files": len(files),
        "top_k": top_k,
        "samples_with_eos": 0,
        "blocks_kept_per_sample": {b: 0 for b in range(8)},
    }

    with torch.no_grad():
        for f_idx, f in enumerate(files):
            sample = storage.read_h5(f)
            eos_pos = _resolve_eos_pos(sample, model_type) if apply_eos_cutoff else 10**9
            if eos_pos < 256:
                diagnostics["samples_with_eos"] += 1

            for b, blk in sample["blocks"].items():
                num_masked_total = blk["h_masked"].shape[1]
                allowed_m = _allowed_masked_positions(b, num_masked_total, eos_pos)
                if not allowed_m:
                    continue

                h_np = blk["h_masked"][:, allowed_m, :].astype(np.float32)  # [L+1, n_kept, d_model]
                h = torch.from_numpy(h_np).to(device=device, dtype=dtype_proj)
                Lp1 = h.shape[0]
                n_kept = h.shape[1]

                # Final-layer logits (compute once per block)
                logits_final = lm_head(final_norm(h[Lp1 - 1])).float()       # [n_kept, vocab]
                topk_final = torch.topk(logits_final, k=top_k, dim=-1).indices  # [n_kept, K]
                # Use log-softmax for numerical stability when needed
                logp_final = F.log_softmax(logits_final, dim=-1)
                p_final = logp_final.exp()

                metrics = {
                    "top_k_overlap": np.zeros(Lp1, dtype=np.float32),
                    "kl_divergence": np.zeros(Lp1, dtype=np.float32),
                    "shared_mass":   np.zeros(Lp1, dtype=np.float32),
                }

                for ell in range(Lp1):
                    logits_ell = lm_head(final_norm(h[ell])).float()        # [n_kept, vocab]
                    logp_ell = F.log_softmax(logits_ell, dim=-1)
                    p_ell = logp_ell.exp()

                    # Top-K overlap (vectorized via broadcasting set-intersection)
                    topk_ell = torch.topk(logits_ell, k=top_k, dim=-1).indices  # [n_kept, K]
                    # For each row, count how many of topk_ell elements are also in topk_final
                    # by comparing all pairs and summing matches; clamp to K to be safe.
                    matches = (topk_ell.unsqueeze(2) == topk_final.unsqueeze(1)).any(dim=2).sum(dim=1)
                    overlap = matches.float() / top_k
                    metrics["top_k_overlap"][ell] = overlap.mean().item()

                    # KL(p_ell || p_final)  (forward KL — "if I sampled from layer ℓ, how surprising would final's distribution be?")
                    kl = (p_ell * (logp_ell - logp_final)).sum(dim=-1)
                    metrics["kl_divergence"][ell] = kl.mean().item()

                    # Shared mass
                    shared = torch.minimum(p_ell, p_final).sum(dim=-1)
                    metrics["shared_mass"][ell] = shared.mean().item()

                    del logits_ell, logp_ell, p_ell, topk_ell, matches, overlap, kl, shared

                if b not in accum_per_block:
                    accum_per_block[b] = {m: [] for m in metrics}
                for m, vals in metrics.items():
                    accum_per_block[b][m].append(vals)

                diagnostics["blocks_kept_per_sample"][b] = (
                    diagnostics["blocks_kept_per_sample"].get(b, 0) + 1
                )

                del h, logits_final, topk_final, logp_final, p_final
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if (f_idx + 1) % 5 == 0 or (f_idx + 1) == len(files):
                print(f"[{model_type}] processed {f_idx + 1}/{len(files)}")

    # Reduce accumulators
    metric_names = ["top_k_overlap", "kl_divergence", "shared_mass"]
    per_block = {}
    for b, ms in accum_per_block.items():
        per_block[b] = {m: np.stack(arrs).mean(axis=0) for m, arrs in ms.items()}

    pooled = {}
    if per_block:
        for m in metric_names:
            pooled[m] = np.stack([per_block[b][m] for b in per_block]).mean(axis=0)

    return per_block, pooled, diagnostics


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(per_block, pooled, top_k: int, model_name: str, out_path: Path):
    fig = plt.figure(figsize=(22, 7))
    gs = fig.add_gridspec(1, 3, wspace=0.27, top=0.86, bottom=0.13, left=0.05, right=0.98)
    cmap = plt.colormaps["tab10"]

    metrics_meta = [
        ("top_k_overlap",
         f"Top-{top_k} overlap with final layer",
         f"|topk(p_ℓ) ∩ topk(p_L)| / {top_k}",
         True,    # has 0.95 threshold
         (0.0, 1.05)),
        ("kl_divergence",
         "KL divergence vs final layer",
         "KL(p_ℓ || p_L)  (lower = closer to final)",
         False,
         None),
        ("shared_mass",
         "Shared probability mass with final layer",
         "Σ min(p_ℓ, p_L)  (higher = closer to final)",
         True,
         (0.0, 1.05)),
    ]

    for col, (mname, title, ylabel, has_threshold, ylim) in enumerate(metrics_meta):
        ax = fig.add_subplot(gs[0, col])
        if not per_block or mname not in pooled:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        L_plus_1 = len(pooled[mname])
        for i, b in enumerate(sorted(per_block.keys())):
            ax.plot(range(L_plus_1), per_block[b][mname],
                    label=f"block {b}", alpha=0.55, color=cmap(i % 10))
        ax.plot(range(L_plus_1), pooled[mname],
                color="black", linewidth=2.6, label="pooled mean")
        if has_threshold:
            ax.axhline(0.95, linestyle="--", color="red", alpha=0.6, label="0.95")
        ax.set_xlabel("layer ℓ (0 = embedding output, L = final block output)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8, loc="best")

    fig.suptitle(f"{model_name} — logit-lens layer-importance metrics  (top-K = {top_k})",
                 fontsize=14, fontweight="bold", y=0.97)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llada", "dream"], required=True)
    parser.add_argument("--probes_root", type=str, default="probes_out")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--fast_dllm_path", type=str, default=None)
    parser.add_argument("--no_eos_cutoff", action="store_true")
    parser.add_argument("--no_special_filter", action="store_true")
    args = parser.parse_args()

    probes_root = Path(args.probes_root)
    sample_dir = probes_root / args.model
    plots_dir = probes_root / "plots"
    files = sorted(sample_dir.glob("sample_*.h5"))
    if not files:
        raise SystemExit(f"No samples in {sample_dir}")

    apply_eos = not args.no_eos_cutoff
    apply_special = not args.no_special_filter

    per_block, pooled, diag = _compute_logit_lens_metrics(
        files, args.model, args.fast_dllm_path,
        top_k=args.top_k,
        apply_eos_cutoff=apply_eos,
        apply_special_filter=apply_special,
    )

    suffix = []
    if apply_eos:
        suffix.append("eosfilter")
    if apply_special:
        suffix.append("specialfilter")
    suffix_str = "_" + "_".join(suffix) if suffix else ""

    out_path = plots_dir / f"logit_lens_{args.model}_topk{args.top_k}{suffix_str}.png"
    _plot(per_block, pooled, args.top_k, args.model, out_path)
    print(f"[{args.model}] saved {out_path}")

    diag_path = plots_dir / f"diagnostics_logit_lens_{args.model}{suffix_str}.json"
    with open(diag_path, "w") as fp:
        json.dump(diag, fp, indent=2)
    print(f"[{args.model}] saved {diag_path}")


if __name__ == "__main__":
    main()
