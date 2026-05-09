"""Cross-block prompt-region KV drift.

Operates on the per-block per-layer prompt-position K/V captures produced by
`run_probes.py --prompt_kv`. Each sample's HDF5 has, per block, optional
`k_prompt [L, n_kv_heads, prompt_len, d_head]` and `v_prompt` datasets — these
are the post-RoPE pre-GQA-broadcast K/V that the model freshly recomputes for
the prompt region at pass 0 of every block (because Fast-dLLM's dual-cache
re-runs a full forward at each block boundary, the prompt KV is not literally
reused — it gets re-emitted, and may drift as previously-decoded blocks change
the input).

For each (layer ℓ, head h, prompt position p, sample), this script computes the
similarity / drift between block 0's K/V and block N's K/V (N ≥ 1), pooled
across (h, p, sample). The result is a per-(layer, block) curve answering:
"how close is block N's prompt KV to block 0's?"

Three metrics, all per (block_N, layer, head, position):

    --metric cosine_sim     cos( KV_block0 , KV_blockN )      higher = more similar
    --metric l2             ‖KV_block0 − KV_blockN‖₂          raw L2 drift
    --metric l2_normalized  ‖Δ‖ / mean(‖KV_block0‖, ‖KV_blockN‖)  scale-free drift
    --metric all            all three

Two sub-plots (K and V) per metric.

Usage:

    python -m probe_runner.plots.plot_prompt_kv_drift --model llada
    python -m probe_runner.plots.plot_prompt_kv_drift --model llada --metric all
    python -m probe_runner.plots.plot_prompt_kv_drift --model dream \\
        --no_special_filter

Output goes to `probes_out/<model>/plots/`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


METRICS = ("cosine_sim", "l2", "l2_normalized")


# --- Loading ------------------------------------------------------------------

def _iter_sample_paths(probes_root: Path, model: str):
    return sorted((probes_root / model).glob("sample_*.h5"))


def _load_prompt_kv(path: Path) -> dict:
    """Load all blocks' k_prompt / v_prompt + relevant attrs.

    Returns dict with keys:
        blocks: {block_idx: {"k": [L, H_kv, P, d], "v": [L, H_kv, P, d]}}
        special_token_positions: list[int] (may be empty)
        prompt_len: int
    """
    out: dict = {"blocks": {}}
    with h5py.File(path, "r") as f:
        out["prompt_len"] = int(f.attrs.get("prompt_len", 0))
        stp = f.attrs.get("special_token_positions", "[]")
        try:
            out["special_token_positions"] = json.loads(stp) if isinstance(stp, str) else list(stp)
        except Exception:
            out["special_token_positions"] = []
        for grp_name in f.keys():
            if not grp_name.startswith("block_"):
                continue
            grp = f[grp_name]
            if "k_prompt" not in grp or "v_prompt" not in grp:
                continue
            block_idx = int(grp_name.split("_")[1])
            out["blocks"][block_idx] = {
                "k": np.asarray(grp["k_prompt"]),
                "v": np.asarray(grp["v_prompt"]),
            }
    return out


# --- Metrics ------------------------------------------------------------------

def _pairwise_metric(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """Same-shape arrays, reduce along last axis (d_head)."""
    if metric == "l2":
        return np.linalg.norm(a - b, axis=-1)
    if metric == "l2_normalized":
        diff = np.linalg.norm(a - b, axis=-1)
        scale = (np.linalg.norm(a, axis=-1) + np.linalg.norm(b, axis=-1)) * 0.5
        return diff / np.maximum(scale, 1e-8)
    if metric == "cosine_sim":
        dot = (a * b).sum(axis=-1)
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        return dot / np.maximum(na * nb, 1e-8)
    raise ValueError(f"unknown metric: {metric}")


# --- Aggregation --------------------------------------------------------------

def _aggregate_drift(
    probes_root: Path,
    model: str,
    metric: str,
    apply_special_filter: bool = True,
) -> dict:
    """Per-sample mean(over heads, prompt positions) of metric(K[N], K[0]) and
    metric(V[N], V[0]), pooled over samples for mean / 95% CI.

    Returns:
        {
            "k_per_sample": dict[block_idx ≥ 1] → [n_samples, L] float32,
            "v_per_sample": dict[block_idx ≥ 1] → [n_samples, L] float32,
            "n_samples": int (number of samples that contributed),
            "n_layers": int,
            "block_indices": sorted list of block indices ≥ 1 with data,
        }
    """
    k_collect: dict[int, list[np.ndarray]] = {}
    v_collect: dict[int, list[np.ndarray]] = {}
    n_layers_seen = None
    samples_with_data = 0

    for path in _iter_sample_paths(probes_root, model):
        sample = _load_prompt_kv(path)
        blocks = sample["blocks"]
        if 0 not in blocks:
            continue
        # Build per-position keep mask: drop special-token positions in the
        # prompt (chat template, BOS, role markers — structural, not semantic).
        prompt_len = sample["prompt_len"]
        keep = np.ones(prompt_len, dtype=bool)
        if apply_special_filter:
            for pos in sample["special_token_positions"]:
                if 0 <= pos < prompt_len:
                    keep[pos] = False
        if not keep.any():
            continue

        ref_k = blocks[0]["k"].astype(np.float32, copy=False)  # [L, H_kv, P, d]
        ref_v = blocks[0]["v"].astype(np.float32, copy=False)
        ref_k = ref_k[:, :, keep, :]
        ref_v = ref_v[:, :, keep, :]
        L = ref_k.shape[0]
        if n_layers_seen is None:
            n_layers_seen = L

        contributed = False
        for block_idx in sorted(blocks.keys()):
            if block_idx == 0:
                continue
            cur_k = blocks[block_idx]["k"].astype(np.float32, copy=False)[:, :, keep, :]
            cur_v = blocks[block_idx]["v"].astype(np.float32, copy=False)[:, :, keep, :]
            if cur_k.shape != ref_k.shape:
                continue  # skip ragged blocks (shouldn't happen but defensive)
            # [L, H_kv, P_kept] per metric
            mk = _pairwise_metric(ref_k, cur_k, metric)
            mv = _pairwise_metric(ref_v, cur_v, metric)
            # Reduce over (head, position) to a per-layer scalar.
            mk_per_layer = mk.reshape(L, -1).mean(axis=1)  # [L]
            mv_per_layer = mv.reshape(L, -1).mean(axis=1)
            k_collect.setdefault(block_idx, []).append(mk_per_layer)
            v_collect.setdefault(block_idx, []).append(mv_per_layer)
            contributed = True
        if contributed:
            samples_with_data += 1

    block_indices = sorted(k_collect.keys())
    k_per_sample = {b: np.stack(k_collect[b], axis=0) for b in block_indices}
    v_per_sample = {b: np.stack(v_collect[b], axis=0) for b in block_indices}
    return {
        "k_per_sample": k_per_sample,
        "v_per_sample": v_per_sample,
        "n_samples": samples_with_data,
        "n_layers": n_layers_seen or 0,
        "block_indices": block_indices,
    }


# --- Plotting -----------------------------------------------------------------

def _y_label(metric: str) -> str:
    return {
        "cosine_sim":   "cos(KV[N], KV[0])",
        "l2":           "‖KV[N] − KV[0]‖₂",
        "l2_normalized": "‖Δ‖ / mean(‖KV[N]‖, ‖KV[0]‖)",
    }[metric]


def _ci95(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, half_width_95) along axis 0."""
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    if n < 2:
        return mean, np.zeros_like(mean)
    sd = arr.std(axis=0, ddof=1)
    sem = sd / np.sqrt(n)
    return mean, 1.96 * sem


def _plot_one_metric(stats: dict, metric: str, out_path: Path, model: str) -> None:
    import matplotlib.pyplot as plt

    block_indices = stats["block_indices"]
    if not block_indices:
        raise RuntimeError(
            "No prompt-KV data found. Did you run `run_probes --prompt_kv`?"
        )
    n_layers = stats["n_layers"]
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(block_indices) - 1, 1)) for i in range(len(block_indices))]

    for ax, kind, store_key in [
        (axes[0], "K", "k_per_sample"),
        (axes[1], "V", "v_per_sample"),
    ]:
        for c, b in zip(colors, block_indices):
            arr = stats[store_key][b]  # [n_samples, L]
            mean, hw = _ci95(arr)
            ax.plot(layers, mean, color=c, lw=1.6, label=f"block {b} vs 0")
            if hw.any():
                ax.fill_between(layers, mean - hw, mean + hw, color=c, alpha=0.18, lw=0)
        ax.set_title(f"{kind} drift — {metric}")
        ax.set_xlabel("layer ℓ")
        ax.grid(alpha=0.3)
        if metric == "cosine_sim":
            ax.axhline(0.95, ls="--", color="black", alpha=0.45, lw=1.0,
                       label="0.95 (≈equivalent)")
            ax.set_ylim(top=1.02)
        else:
            ax.set_ylim(bottom=0.0)
    axes[0].set_ylabel(_y_label(metric))
    axes[1].legend(fontsize=8, loc="best", ncol=1)

    fig.suptitle(
        f"{model} — prompt-region KV drift across blocks  "
        f"(n={stats['n_samples']} samples, mean ± 95% CI)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[{model}] saved {out_path}")


# --- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llada", "dream"], required=True)
    parser.add_argument("--probes_root", type=str, default="probes_out")
    parser.add_argument(
        "--metric",
        choices=list(METRICS) + ["all"],
        default="cosine_sim",
        help="Drift metric. cos_sim is the natural 'closeness' metric; "
             "l2_normalized is the natural 'drift' metric. Pass 'all' to "
             "render all three.",
    )
    parser.add_argument(
        "--no_special_filter",
        action="store_true",
        help="Don't drop special-token positions (BOS, chat-template markers) "
             "from the per-position average.",
    )
    args = parser.parse_args()

    probes_root = Path(args.probes_root)
    plots_dir = probes_root / args.model / "plots"

    metrics = list(METRICS) if args.metric == "all" else [args.metric]
    suffix = "_nofilter" if args.no_special_filter else ""
    apply_filter = not args.no_special_filter

    for metric in metrics:
        print(f"[{args.model}] aggregating prompt-KV drift, metric={metric} …")
        stats = _aggregate_drift(probes_root, args.model, metric,
                                 apply_special_filter=apply_filter)
        if not stats["block_indices"]:
            print(f"[{args.model}] no samples with k_prompt / v_prompt — skipping {metric}")
            continue
        out_path = plots_dir / f"prompt_kv_drift_{args.model}_{metric}{suffix}.png"
        _plot_one_metric(stats, metric, out_path, args.model)


if __name__ == "__main__":
    main()
