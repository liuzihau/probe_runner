"""Intra-block drift analyses (Analysis A + B).

Operates on the per-pass hidden-state captures produced by
`run_probes.py --intra_block`. Each sample's HDF5 has, per block, a
`h_per_pass [n_passes, L+1, num_masked, d_model]` dataset alongside the
existing pass-0 fields.

Three metrics, all computed per (pass-pair, layer, position) on `h_per_pass`:

    --metric l2                ‖h_i − h_{i+1}‖₂                     (raw L2 drift)
    --metric l2_normalized     ‖h_i − h_{i+1}‖ / mean(‖h_i‖, ‖h_{i+1}‖)
                                                                     (depth-norm-invariant drift)
    --metric cosine_sim        cos(h_i, h_{i+1})                     (alignment; higher = more similar)
    --metric all               all three above

Two analyses (variant flag):

    --variant drift_grouped (Analysis A)
        Group positions by (token_state_pre[i], token_state_pre[i+1]) into
        MM (mask→mask), MC (mask→clean), CC (clean→clean). For each group
        plot mean metric vs. layer, faceted by block. Pooled over samples.

    --variant diff_heatmap (Analysis B)
        For each sample (top --n_samples_heatmap, default 5) build per-block
        [n_passes-1 × num_masked] heatmaps whose cells are the layer-mean
        metric. × marks on cells where the position was revealed during
        that pass-to-pass transition. One figure per (sample, metric).

    --variant all → both variants for the chosen metric(s).

Usage:

    python -m probe_runner.plots.plot_intra_block --model llada
    python -m probe_runner.plots.plot_intra_block --model llada --variant diff_heatmap --metric cosine_sim
    python -m probe_runner.plots.plot_intra_block --model llada --n_samples_heatmap 3

Output goes to `probes_out/<model>/plots/`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


# --- Loading ------------------------------------------------------------------

def _iter_sample_paths(probes_root: Path, model: str):
    return sorted((probes_root / model).glob("sample_*.h5"))


def _load_intra_block(path: Path) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    with h5py.File(path, "r") as f:
        for grp_name in f.keys():
            if not grp_name.startswith("block_"):
                continue
            grp = f[grp_name]
            if "h_per_pass" not in grp:
                continue
            block_idx = int(grp_name.split("_")[1])
            out[block_idx] = {
                "h_per_pass":           np.asarray(grp["h_per_pass"]),
                "token_state_per_pass": np.asarray(grp["token_state_per_pass"]),
                "revealed_per_pass":    np.asarray(grp["revealed_per_pass"]),
                "pass_indices":         np.asarray(grp["pass_indices"]),
            }
    return out


# --- Metrics ------------------------------------------------------------------

METRICS = ("l2", "l2_normalized", "cosine_sim")


def _compute_diffs(h_per_pass: np.ndarray, metric: str) -> np.ndarray:
    """Per (pass-pair, layer, position) drift / similarity.

    Args:
        h_per_pass: [n_passes, L+1, num_masked, d_model] (fp16 ok, cast inside).
        metric: one of METRICS.
    Returns:
        [n_passes-1, L+1, num_masked] float32.
    """
    h = h_per_pass.astype(np.float32, copy=False)
    a = h[:-1]                                # [P-1, L+1, M, D]
    b = h[1:]
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


def _is_drift_metric(metric: str) -> bool:
    """Whether higher = more change. Used to flip color/orientation."""
    return metric in ("l2", "l2_normalized")


# --- Analysis A: token-state-grouped drift ------------------------------------

def _accumulate_grouped(probes_root: Path, model: str, metric: str) -> dict:
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}
    n_layers_p1 = None
    n_samples = 0

    for path in _iter_sample_paths(probes_root, model):
        per_block = _load_intra_block(path)
        if not per_block:
            continue
        n_samples += 1
        for block_idx, data in per_block.items():
            h = data["h_per_pass"]
            ts = data["token_state_per_pass"].astype(np.int8)
            P, Lp1, M, _ = h.shape
            if n_layers_p1 is None:
                n_layers_p1 = Lp1
            if P < 2:
                continue
            diffs = _compute_diffs(h, metric)                 # [P-1, L+1, M]
            key = (ts[:-1].astype(np.int32) << 1) | ts[1:].astype(np.int32)  # [P-1, M]
            sums.setdefault(block_idx, np.zeros((3, Lp1), dtype=np.float64))
            counts.setdefault(block_idx, np.zeros((3, Lp1), dtype=np.int64))
            d_flat = diffs.transpose(1, 0, 2).reshape(Lp1, -1)  # [Lp1, (P-1)*M]
            for slot, k in enumerate((0b00, 0b01, 0b11)):  # MM, MC, CC
                m_flat = (key == k).reshape(-1)
                if m_flat.any():
                    sums[block_idx][slot] += d_flat[:, m_flat].sum(axis=1)
                    counts[block_idx][slot] += int(m_flat.sum())
    return {"sums": sums, "counts": counts, "n_layers_p1": n_layers_p1, "n_samples": n_samples}


def _plot_drift_grouped(stats: dict, out_path: Path, model: str, metric: str) -> None:
    import matplotlib.pyplot as plt
    sums, counts = stats["sums"], stats["counts"]
    n_layers_p1 = stats["n_layers_p1"]
    if not sums:
        raise RuntimeError("No intra-block data found. Did you run with --intra_block?")
    blocks = sorted(sums.keys())
    n_blocks = len(blocks)
    ncols = min(4, n_blocks)
    nrows = int(np.ceil(n_blocks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False, sharey=True)

    # Skip layer 0 (input embedding) — it's a degenerate token-id-only
    # readout that swamps the y-axis for MC and tells us nothing about
    # the model's contextual processing. Plot layers 1..L only, with
    # x-axis labelled by the original layer index.
    plot_layers = np.arange(1, n_layers_p1)

    colors = {"MM": "tab:gray", "MC": "tab:red", "CC": "tab:blue"}
    for ax_i, block_idx in enumerate(blocks):
        ax = axes[ax_i // ncols][ax_i % ncols]
        s = sums[block_idx]; c = counts[block_idx]
        with np.errstate(invalid="ignore"):
            mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        for slot, label in enumerate(("MM", "MC", "CC")):
            ax.plot(plot_layers, mean[slot, 1:], label=label, color=colors[label], lw=1.5)
        # 0.95 reference (CKA-style "effectively equivalent") on cosine only.
        if metric == "cosine_sim":
            ax.axhline(0.95, ls="--", color="black", alpha=0.45, lw=1.0,
                       label="0.95 (≈equivalent)")
        ax.set_title(f"block {block_idx}")
        ax.set_xlabel("Layer k")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(_y_label(metric))
    axes[0][0].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"{model}: hidden-state {_metric_word(metric)} between denoise steps i and i+1, "
        f"grouped by token-state transition  "
        f"[n_samples={stats['n_samples']}, layers 1+]"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def _y_label(metric: str) -> str:
    return {
        "l2":            "L2 distance",
        "l2_normalized": "L2 normalized",
        "cosine_sim":    "Cosine similarity",
    }[metric]


def _metric_word(metric: str) -> str:
    """Direction-aware word for plot titles."""
    return "similarity" if metric == "cosine_sim" else "drift"


# --- Analysis B: per-sample diff heatmaps -------------------------------------

def _collect_samples_for_heatmap(probes_root: Path, model: str, metric: str,
                                  n_samples: int) -> list[tuple[str, dict]]:
    """Eagerly load + pre-compute heatmap cells for the first n_samples that
    have intra-block data. Returns [(sample_id, {block_idx: {cell, revealed,
    n_passes}}), ...]. Done in one pass so we can compute the *shared*
    vmin/vmax across all samples for stable cross-sample colour comparison.

    Cell aggregation: mean over **layers 1..L** (transformer blocks only) —
    layer 0 (input embedding) is omitted. The embed layer is a token-id
    readout that swamps the average for MC positions without telling us
    anything about contextual processing.
    """
    out: list[tuple[str, dict]] = []
    for path in _iter_sample_paths(probes_root, model):
        per_block = _load_intra_block(path)
        if not per_block:
            continue
        sample_data: dict[int, dict] = {}
        for block_idx, data in per_block.items():
            h = data["h_per_pass"]
            rv = data["revealed_per_pass"].astype(bool)
            if h.shape[0] < 2:
                continue
            diffs = _compute_diffs(h, metric)        # [P-1, L+1, M]
            cell = diffs[:, 1:, :].mean(axis=1)      # [P-1, M] — skip layer 0 (embed)
            sample_data[block_idx] = {
                "cell":     cell,
                "revealed": rv[:-1],
                "n_passes": int(h.shape[0]),
            }
        if sample_data:
            out.append((path.stem, sample_data))
        if len(out) >= n_samples:
            break
    return out


def _global_vrange(samples: list[tuple[str, dict]]) -> tuple[float, float]:
    """Min/max across all (sample, block, pass-pair, position) cells. Used
    so all per-sample heatmaps share a colorbar — different samples decode
    in different orders, so without a fixed scale you can't visually
    compare drift magnitudes across samples.
    """
    lo = np.inf
    hi = -np.inf
    for _, blocks in samples:
        for _, d in blocks.items():
            cell = d["cell"]
            lo = min(lo, float(np.nanmin(cell)))
            hi = max(hi, float(np.nanmax(cell)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi == lo:
        hi = lo + 1e-8
    return lo, hi


def _plot_diff_heatmap_one(sample_data: dict, out_path: Path,
                           model: str, metric: str, sample_id: str,
                           vmin: float, vmax: float) -> None:
    import matplotlib.pyplot as plt
    blocks = sorted(sample_data.keys())
    n_blocks = len(blocks)
    ncols = min(2, n_blocks)
    nrows = int(np.ceil(n_blocks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 + 1.5 * nrows),
                             squeeze=False)
    # For drift metrics: dark = low drift, bright = high drift.
    # For cosine sim: reverse so dark = high sim (low change), matching reading.
    cmap = "viridis" if _is_drift_metric(metric) else "viridis_r"
    cbar_label = {
        "l2":            r"mean-over-layers $\|h_i - h_{i+1}\|_2$",
        "l2_normalized": r"mean-over-layers $\|h_i - h_{i+1}\| / \overline{\|h\|}$",
        "cosine_sim":    r"mean-over-layers $\cos(h_i, h_{i+1})$",
    }[metric]

    for ax_i, block_idx in enumerate(blocks):
        ax = axes[ax_i // ncols][ax_i % ncols]
        d = sample_data[block_idx]
        cell, revealed, n_passes = d["cell"], d["revealed"], d["n_passes"]
        im = ax.imshow(cell, aspect="auto", origin="lower",
                       interpolation="nearest", cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ys, xs = np.where(revealed)
        ax.scatter(xs, ys, marker="x", s=14, c="white", linewidths=0.7)
        ax.set_title(f"block {block_idx}  (n_passes={n_passes})")
        ax.set_xlabel("position in block")
        ax.set_ylabel("decode pass i (i → i+1)")
        fig.colorbar(im, ax=ax, label=cbar_label)

    for j in range(n_blocks, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    direction = "× = revealed during transition; higher = more drift" \
                if _is_drift_metric(metric) \
                else "× = revealed; higher = more similar"
    fig.suptitle(
        f"{model} {sample_id}: hidden-state {_metric_word(metric)} between denoise steps i and i+1  "
        f"[layer-mean over transformer blocks 1..L; embed layer omitted]\n"
        f"({direction})  "
        f"shared scale [vmin={vmin:.3g}, vmax={vmax:.3g}] across all sampled plots"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# --- CLI ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=["llada", "dream"], required=True)
    p.add_argument("--probes_root", type=Path, default=Path("probes_out"))
    p.add_argument("--variant", choices=["drift_grouped", "diff_heatmap", "all"],
                   default="all")
    p.add_argument("--metric", choices=("l2", "l2_normalized", "cosine_sim", "all"),
                   default="all", help="Drift metric. 'all' produces one plot per metric.")
    p.add_argument("--n_samples_heatmap", type=int, default=5,
                   help="Number of samples to plot per metric for the diff_heatmap variant.")
    args = p.parse_args()

    out_dir = args.probes_root / args.model / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = METRICS if args.metric == "all" else (args.metric,)

    do_grouped = args.variant in ("drift_grouped", "all")
    do_heatmap = args.variant in ("diff_heatmap", "all")

    for metric in metrics:
        if do_grouped:
            stats = _accumulate_grouped(args.probes_root, args.model, metric)
            if stats["n_samples"] == 0:
                print(f"WARN: no intra_block samples found for drift_grouped/{metric}.")
            else:
                out = out_dir / f"intra_block_drift_grouped_{metric}.png"
                _plot_drift_grouped(stats, out, args.model, metric)

        if do_heatmap:
            samples = _collect_samples_for_heatmap(
                args.probes_root, args.model, metric, args.n_samples_heatmap,
            )
            if not samples:
                print(f"WARN: no intra_block samples found for diff_heatmap/{metric}.")
                continue
            vmin, vmax = _global_vrange(samples)
            for sample_id, per_block in samples:
                out = out_dir / f"intra_block_diff_heatmap_{metric}_{sample_id}.png"
                _plot_diff_heatmap_one(per_block, out, args.model, metric,
                                       sample_id, vmin, vmax)


if __name__ == "__main__":
    main()
