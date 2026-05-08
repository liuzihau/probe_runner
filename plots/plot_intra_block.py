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
    """Per (pass-pair, layer, position) drift / similarity, consecutive pairs.

    Args:
        h_per_pass: [n_passes, L+1, num_masked, d_model] (fp16 ok, cast inside).
        metric: one of METRICS.
    Returns:
        [n_passes-1, L+1, num_masked] float32.
    """
    h = h_per_pass.astype(np.float32, copy=False)
    a = h[:-1]                                # [P-1, L+1, M, D]
    b = h[1:]
    return _pairwise_metric(a, b, metric)


def _compute_diffs_vs_first(h_per_pass: np.ndarray, metric: str) -> np.ndarray:
    """Per (i ≥ 1, layer, position) drift / similarity, each pass vs pass 0.

    Args:
        h_per_pass: [n_passes, L+1, num_masked, d_model].
        metric: one of METRICS.
    Returns:
        [n_passes-1, L+1, num_masked] float32 — index 0 corresponds to
        pass 1 vs pass 0, index 1 to pass 2 vs pass 0, etc.
    """
    h = h_per_pass.astype(np.float32, copy=False)
    a = h[0:1]                                # [1, L+1, M, D]
    b = h[1:]                                  # [P-1, L+1, M, D]
    return _pairwise_metric(a, b, metric)


def _pairwise_metric(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """Apply `metric` to two same-shape (broadcastable) arrays along the last axis."""
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


# --- Analysis C: drift vs pass 0 (each iteration compared to first pass) -----

def _accumulate_vs_pass0(probes_root: Path, model: str, metric: str,
                         blocks_to_use: list[int], n_samples: int,
                         max_iter: int) -> dict:
    """For each (iter i, layer), accumulate per-position metric values into
    sum / sum-of-squares / count, separately for MM and MC groups, where:

    - MM = position still has mask as input at pass i (state[i, p] == 0)
    - MC = position revealed by some earlier pass (state[i, p] == 1)

    At pass 0 every position is mask, so the (state[0, p], state[i, p])
    pair reduces to a single state[i, p] classification — that's why this
    variant has only two groups instead of three.

    The per-iter mean is computed over all (sample, block, position)
    entries belonging to that group. Variance / SEM follow from the
    sum-of-squares track. Blocks outside `blocks_to_use` are skipped
    (default 0, 1 — late blocks have higher EOS-token noise).
    """
    n_layers_p1 = None
    sum_mm = sum_sq_mm = count_mm = None
    sum_mc = sum_sq_mc = count_mc = None
    n_collected = 0

    for path in _iter_sample_paths(probes_root, model):
        if n_collected >= n_samples:
            break
        per_block = _load_intra_block(path)
        if not per_block:
            continue
        n_collected += 1
        for block_idx in blocks_to_use:
            if block_idx not in per_block:
                continue
            data = per_block[block_idx]
            h = data["h_per_pass"]
            ts = data["token_state_per_pass"].astype(np.int8)
            P, Lp1, M, _ = h.shape
            if P < 2:
                continue

            if n_layers_p1 is None:
                n_layers_p1 = Lp1
                shape = (max_iter, Lp1)
                sum_mm = np.zeros(shape, dtype=np.float64)
                sum_sq_mm = np.zeros(shape, dtype=np.float64)
                count_mm = np.zeros(shape, dtype=np.int64)
                sum_mc = np.zeros(shape, dtype=np.float64)
                sum_sq_mc = np.zeros(shape, dtype=np.float64)
                count_mc = np.zeros(shape, dtype=np.int64)

            n_iter = min(P - 1, max_iter)
            diffs = _compute_diffs_vs_first(h, metric)[:n_iter]   # [n_iter, L+1, M]

            state_after = ts[1:1 + n_iter]                         # [n_iter, M]
            mm_pos = (state_after == 0)                             # [n_iter, M]
            mc_pos = (state_after == 1)                             # [n_iter, M]
            mm_mask = mm_pos[:, None, :]                            # [n_iter, 1, M]
            mc_mask = mc_pos[:, None, :]

            d_mm = np.where(mm_mask, diffs, 0.0)
            d_mc = np.where(mc_mask, diffs, 0.0)
            n_per_i_mm = mm_pos.sum(axis=1)                         # [n_iter]
            n_per_i_mc = mc_pos.sum(axis=1)

            sum_mm[:n_iter]    += d_mm.sum(axis=2)
            sum_sq_mm[:n_iter] += (d_mm ** 2).sum(axis=2)
            count_mm[:n_iter]  += n_per_i_mm[:, None]               # broadcasts to [n_iter, Lp1]
            sum_mc[:n_iter]    += d_mc.sum(axis=2)
            sum_sq_mc[:n_iter] += (d_mc ** 2).sum(axis=2)
            count_mc[:n_iter]  += n_per_i_mc[:, None]

    return {
        "sum_mm": sum_mm, "sum_sq_mm": sum_sq_mm, "count_mm": count_mm,
        "sum_mc": sum_mc, "sum_sq_mc": sum_sq_mc, "count_mc": count_mc,
        "n_layers_p1": n_layers_p1,
        "n_samples": n_collected,
        "blocks": list(blocks_to_use),
        "max_iter": max_iter,
    }


def _plot_drift_vs_pass0(stats: dict, out_path: Path, model: str, metric: str) -> None:
    import matplotlib.pyplot as plt
    sum_mm, sum_sq_mm, count_mm = stats["sum_mm"], stats["sum_sq_mm"], stats["count_mm"]
    sum_mc, sum_sq_mc, count_mc = stats["sum_mc"], stats["sum_sq_mc"], stats["count_mc"]
    n_layers_p1 = stats["n_layers_p1"]
    if n_layers_p1 is None:
        raise RuntimeError("No intra-block data found. Did you run with --intra_block?")
    plot_layers = np.arange(1, n_layers_p1)                         # skip layer 0 (embed)
    max_iter = sum_mm.shape[0]

    fig, (ax_mm, ax_mc) = plt.subplots(1, 2, figsize=(12, 4.2),
                                       sharey=(metric == "cosine_sim"))
    cmap = plt.get_cmap("viridis")

    for ax, sum_arr, sumsq_arr, count_arr, group_label in [
        (ax_mm, sum_mm, sum_sq_mm, count_mm, "MM (still mask at input of pass i)"),
        (ax_mc, sum_mc, sum_sq_mc, count_mc, "MC (revealed at some earlier pass)"),
    ]:
        for i_idx in range(max_iter):
            n = count_arr[i_idx, 1:].astype(np.float64)              # skip embed
            if n.sum() == 0:
                continue
            mean = sum_arr[i_idx, 1:] / np.maximum(n, 1)
            var = sumsq_arr[i_idx, 1:] / np.maximum(n, 1) - mean ** 2
            var = np.clip(var, 0.0, None)                            # FP safety
            sem = np.sqrt(var / np.maximum(n, 1))
            color = cmap((i_idx + 0.5) / max_iter)
            ax.plot(plot_layers, mean, color=color,
                    label=f"iter {i_idx + 1}", lw=1.4)
            ax.fill_between(plot_layers, mean - 1.96 * sem, mean + 1.96 * sem,
                            color=color, alpha=0.18, linewidth=0)
        ax.set_title(group_label)
        ax.set_xlabel("Layer k")
        ax.grid(alpha=0.3)
        if metric == "cosine_sim":
            ax.axhline(0.95, ls="--", color="black", alpha=0.45, lw=1.0,
                       label="0.95 (≈equivalent)")
        ax.legend(fontsize=8, loc="best")

    ax_mm.set_ylabel(_y_label(metric))
    fig.suptitle(
        f"{model}: hidden-state {_metric_word(metric)} between pass 0 and pass i  "
        f"(95% CI shaded)  [blocks={stats['blocks']}, n_samples={stats['n_samples']}, "
        f"layers 1+]"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# --- Analysis D: prediction-distribution overlap (speculative-style) ---------
#
# For each (sample, block, pass i), project the LAST-LAYER hidden state at
# pass i and at the reference pass (pass 0 by default, or i-1) through the
# model's final norm + LM head + softmax, then compute
#
#     shared_mass(p, q) = Σ_x min(p(x), q(x))   ∈ [0, 1]
#
# This is the closed-form expected acceptance rate from speculative decoding
# (Leviathan et al. 2023, Chen et al. 2023): if pass-0 (or the previous pass)
# were used as a draft for pass-i as the target, what fraction of mass is
# shared between the two distributions? It's also `1 - 0.5·||p − q||₁` (TV
# overlap). Plotted as iteration → shared_mass with 95% CI shading, MM and
# MC side-by-side.
#
# Costs: requires loading the model (final_norm + lm_head). Reuses
# `plot_logit_lens._load_model_components` to avoid duplication.

def _accumulate_prediction_overlap(
    probes_root: Path,
    model_type: str,
    fast_dllm_path: str | None,
    *,
    blocks_to_use: list[int],
    n_samples: int,
    max_iter: int,
    reference: str,            # "pass0" or "prev"
) -> dict:
    import torch
    import torch.nn.functional as F
    from probe_runner.plots.plot_logit_lens import _load_model_components

    print(f"[prediction_overlap] loading {model_type} model + final_norm + lm_head…")
    model, final_norm, lm_head = _load_model_components(model_type, fast_dllm_path)
    device = next(model.parameters()).device
    dtype_proj = next(model.parameters()).dtype

    sum_mm = sum_sq_mm = count_mm = None
    sum_mc = sum_sq_mc = count_mc = None
    n_collected = 0

    with torch.no_grad():
        for path in _iter_sample_paths(probes_root, model_type):
            if n_collected >= n_samples:
                break
            per_block = _load_intra_block(path)
            if not per_block:
                continue
            n_collected += 1

            for block_idx in blocks_to_use:
                if block_idx not in per_block:
                    continue
                data = per_block[block_idx]
                h_pp = data["h_per_pass"]                       # [P, L+1, M, D]
                ts = data["token_state_per_pass"].astype(np.int8)  # [P, M]
                P, Lp1, M, D = h_pp.shape
                if P < 2:
                    continue

                if sum_mm is None:
                    shape = (max_iter,)
                    sum_mm = np.zeros(shape, dtype=np.float64)
                    sum_sq_mm = np.zeros(shape, dtype=np.float64)
                    count_mm = np.zeros(shape, dtype=np.int64)
                    sum_mc = np.zeros(shape, dtype=np.float64)
                    sum_sq_mc = np.zeros(shape, dtype=np.float64)
                    count_mc = np.zeros(shape, dtype=np.int64)

                n_iter = min(P - 1, max_iter)

                # Project last-layer hidden states across all passes we need
                # in one shot (saves redundant ln_f / lm_head invocations).
                h_last = torch.from_numpy(
                    h_pp[: n_iter + 1, -1, :, :].astype(np.float32)
                ).to(device=device, dtype=dtype_proj)              # [n_iter+1, M, D]
                logits = lm_head(final_norm(h_last)).float()        # [n_iter+1, M, V]
                p_all = F.softmax(logits, dim=-1)                    # [n_iter+1, M, V]

                for i_idx in range(n_iter):
                    pass_i = i_idx + 1
                    ref_idx = 0 if reference == "pass0" else i_idx
                    p_target = p_all[pass_i]                         # [M, V]
                    p_draft = p_all[ref_idx]                         # [M, V]
                    shared = torch.minimum(p_target, p_draft).sum(dim=-1)  # [M]
                    shared_np = shared.cpu().numpy()

                    state_at_i = ts[pass_i]                          # [M]
                    mm = (state_at_i == 0)
                    mc = (state_at_i == 1)

                    if mm.any():
                        sm = shared_np[mm]
                        sum_mm[i_idx] += sm.sum()
                        sum_sq_mm[i_idx] += (sm ** 2).sum()
                        count_mm[i_idx] += int(mm.sum())
                    if mc.any():
                        sm = shared_np[mc]
                        sum_mc[i_idx] += sm.sum()
                        sum_sq_mc[i_idx] += (sm ** 2).sum()
                        count_mc[i_idx] += int(mc.sum())

                del h_last, logits, p_all
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if n_collected % 5 == 0 or n_collected == n_samples:
                print(f"[prediction_overlap] processed {n_collected}/{n_samples}")

    return {
        "sum_mm": sum_mm, "sum_sq_mm": sum_sq_mm, "count_mm": count_mm,
        "sum_mc": sum_mc, "sum_sq_mc": sum_sq_mc, "count_mc": count_mc,
        "n_samples": n_collected,
        "blocks": list(blocks_to_use),
        "max_iter": max_iter,
        "reference": reference,
    }


def _plot_prediction_overlap(stats: dict, out_path: Path, model_name: str) -> None:
    import matplotlib.pyplot as plt
    sum_mm = stats["sum_mm"]; sum_sq_mm = stats["sum_sq_mm"]; count_mm = stats["count_mm"]
    sum_mc = stats["sum_mc"]; sum_sq_mc = stats["sum_sq_mc"]; count_mc = stats["count_mc"]
    if sum_mm is None:
        raise RuntimeError("No intra-block data found. Did you run with --intra_block?")

    max_iter = sum_mm.shape[0]
    iters = np.arange(1, max_iter + 1)
    ref_label = "pass 0" if stats["reference"] == "pass0" else "previous pass (i−1)"

    fig, (ax_mm, ax_mc) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, sum_arr, sumsq_arr, count_arr, group_label, color in [
        (ax_mm, sum_mm, sum_sq_mm, count_mm,
         "MM (still mask at input of pass i)", "tab:gray"),
        (ax_mc, sum_mc, sum_sq_mc, count_mc,
         "MC (revealed at some pass < i)", "tab:red"),
    ]:
        n = count_arr.astype(np.float64)
        valid = n > 0
        if not valid.any():
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            mean = np.where(valid, sum_arr / np.maximum(n, 1), np.nan)
            var = np.where(valid, sumsq_arr / np.maximum(n, 1) - mean ** 2, 0.0)
            var = np.clip(var, 0.0, None)
            sem = np.where(valid, np.sqrt(var / np.maximum(n, 1)), 0.0)
            ax.plot(iters, mean, "-o", color=color, lw=1.6, markersize=5)
            ax.fill_between(iters, mean - 1.96 * sem, mean + 1.96 * sem,
                            color=color, alpha=0.18, linewidth=0)
            # Sample-count annotations help spot iterations with thin support.
            for i, c, m in zip(iters, count_arr, mean):
                if not np.isnan(m):
                    ax.annotate(f"n={int(c)}", (i, m),
                                textcoords="offset points", xytext=(0, 8),
                                ha="center", fontsize=7, color="gray")
        ax.axhline(0.95, ls="--", color="black", alpha=0.45, lw=1.0,
                   label="0.95 (≈identical predictions)")
        ax.set_title(group_label)
        ax.set_xlabel("Iteration i")
        ax.set_xticks(iters)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")

    ax_mm.set_ylabel(r"Shared mass  $\sum_x \min(p_i(x), q(x))$")
    fig.suptitle(
        f"{model_name}: prediction-distribution overlap (target=pass i, draft={ref_label})\n"
        f"[blocks={stats['blocks']}, n_samples={stats['n_samples']}, 95% CI shaded]"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# --- Analysis E: layered per-(sample, block) heatmaps vs pass 0 --------------
#
# Per (sample, block), a 3×2 grid of heatmaps:
#   panels 1..5 — cos_sim(h[i, layer, p], h[0, layer, p]) for selected layers,
#                  default 1, 8, 16, 24, 32 (LLaDA-32 quantiles).
#   panel 6 — shared_mass(p_i, p_0) at the LM head, computed via
#              softmax(lm_head(final_norm(h[*, -1, :, :]))) for each pass.
# y-axis = iteration i (vs pass 0); x-axis = position in block;
# × overlay = position revealed at some pass ≤ i.
#
# Loads the model (for shared mass). Heavy; not part of --variant all.


def _layered_heatmap_payload(
    h_per_pass: np.ndarray,
    ts: np.ndarray,
    layers_to_plot: list[int],
    p_all: torch.Tensor,         # [P, M, V]
) -> dict:
    """Compute the per-(sample, block) data needed for the layered heatmap."""
    P, Lp1, M, _ = h_per_pass.shape
    h = h_per_pass.astype(np.float32, copy=False)
    h0 = h[0:1]                                  # [1, L+1, M, D]
    rest = h[1:]                                 # [P-1, L+1, M, D]
    dot = (rest * h0).sum(axis=-1)               # [P-1, L+1, M]
    nr = np.linalg.norm(rest, axis=-1)
    n0 = np.linalg.norm(h0, axis=-1)
    cos_full = dot / np.maximum(nr * n0, 1e-8)   # [P-1, L+1, M]

    layer_diffs: dict[int, np.ndarray] = {}
    for layer in layers_to_plot:
        if 0 <= layer < Lp1:
            layer_diffs[layer] = cos_full[:, layer, :]   # [P-1, M]

    # Shared mass at LM head: panel 6.
    p0 = p_all[0:1]                              # [1, M, V]
    sm = torch.minimum(p_all[1:], p0).sum(dim=-1)  # [P-1, M]
    shared_mass = sm.cpu().numpy()

    revealed_so_far = (ts[1:].astype(np.int8) == 1)   # [P-1, M] — clean at pass i

    return {
        "layer_diffs":     layer_diffs,
        "shared_mass":     shared_mass,
        "revealed_so_far": revealed_so_far,
        "n_passes":        int(P),
    }


def _accumulate_layered_heatmaps(
    probes_root: Path, model_type: str, fast_dllm_path: str | None,
    *, blocks_to_use: list[int], n_samples: int, layers_to_plot: list[int],
):
    """Generator: yield (sample_id, block_idx, payload) for the first n_samples
    that have intra-block data. Loads model + final_norm + lm_head once."""
    import torch
    import torch.nn.functional as F
    from probe_runner.plots.plot_logit_lens import _load_model_components

    print(f"[layered_heatmap] loading {model_type} model + final_norm + lm_head…")
    model, final_norm, lm_head = _load_model_components(model_type, fast_dllm_path)
    device = next(model.parameters()).device
    dtype_proj = next(model.parameters()).dtype

    n_collected = 0
    with torch.no_grad():
        for path in _iter_sample_paths(probes_root, model_type):
            if n_collected >= n_samples:
                break
            per_block = _load_intra_block(path)
            if not per_block:
                continue
            n_collected += 1

            for block_idx in blocks_to_use:
                if block_idx not in per_block:
                    continue
                data = per_block[block_idx]
                h_pp = data["h_per_pass"]
                ts = data["token_state_per_pass"]
                P = h_pp.shape[0]
                if P < 2:
                    continue

                # Project last-layer h through final_norm + lm_head + softmax for shared mass.
                h_last = torch.from_numpy(
                    h_pp[:, -1, :, :].astype(np.float32),
                ).to(device=device, dtype=dtype_proj)            # [P, M, D]
                logits = lm_head(final_norm(h_last)).float()      # [P, M, V]
                p_all = F.softmax(logits, dim=-1)

                payload = _layered_heatmap_payload(h_pp, ts, layers_to_plot, p_all)
                yield path.stem, block_idx, payload

                del h_last, logits, p_all
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def _plot_layered_heatmap(
    payload: dict, sample_id: str, block_idx: int, model_name: str,
    out_path: Path, layers_to_plot: list[int],
) -> None:
    import matplotlib.pyplot as plt

    panels: list[tuple[str, np.ndarray | None, str]] = []
    for layer in layers_to_plot[:5]:
        cell = payload["layer_diffs"].get(layer)
        title = f"layer {layer}: cos(h_i, h_0)"
        panels.append((title, cell, "Cosine similarity"))
    panels.append((
        "shared mass: Σ min(p_i, p_0)",
        payload["shared_mass"],
        "Shared mass (LM-head distribution overlap)",
    ))

    revealed = payload["revealed_so_far"]
    n_iter = revealed.shape[0]
    M = revealed.shape[1]
    iters = np.arange(1, n_iter + 1)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), squeeze=False)
    cmap = "viridis_r"   # high similarity = dark, low similarity = light

    for idx, (title, cell, cbar_label) in enumerate(panels):
        ax = axes[idx // 2][idx % 2]
        if cell is None:
            ax.text(0.5, 0.5, "no data\n(layer index out of range)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_visible(False)
            continue
        im = ax.imshow(cell, aspect="auto", origin="lower",
                       interpolation="nearest", cmap=cmap,
                       vmin=0.0, vmax=1.0,
                       extent=[-0.5, M - 0.5, 0.5, n_iter + 0.5])
        ys, xs = np.where(revealed)
        if len(ys):
            ax.scatter(xs, ys + 1, marker="x", s=14, c="white", linewidths=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("position in block")
        ax.set_ylabel("iteration i (vs pass 0)")
        ax.set_yticks(iters)
        fig.colorbar(im, ax=ax, label=cbar_label)

    fig.suptitle(
        f"{model_name} {sample_id} block {block_idx} — drift / overlap vs pass 0\n"
        f"(× = position revealed at some pass ≤ i; n_passes={payload['n_passes']})"
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
    p.add_argument("--variant",
                   choices=["drift_grouped", "diff_heatmap", "drift_vs_pass0",
                            "prediction_overlap", "layered_heatmap", "all"],
                   default="all")
    p.add_argument("--metric", choices=("l2", "l2_normalized", "cosine_sim", "all"),
                   default="all", help="Drift metric. 'all' produces one plot per metric.")
    p.add_argument("--n_samples_heatmap", type=int, default=5,
                   help="Number of samples to plot per metric for the diff_heatmap variant.")
    p.add_argument("--vs_pass0_blocks", type=str, default="0,1",
                   help="Comma-separated block indices to aggregate for the drift_vs_pass0 "
                        "variant. Default '0,1' — late blocks have higher EOS-token noise.")
    p.add_argument("--vs_pass0_n_samples", type=int, default=5,
                   help="Sample count for drift_vs_pass0 (5 for fast iteration; bump to 100 "
                        "for paper-quality bands).")
    p.add_argument("--vs_pass0_max_iter", type=int, default=6,
                   help="Number of denoise iterations to plot in drift_vs_pass0. Default 6.")
    p.add_argument("--prediction_overlap_reference", choices=["pass0", "prev"],
                   default="pass0",
                   help="Reference distribution for prediction_overlap variant. "
                        "'pass0' = compare every pass-i to pass-0 (T3 talk_rps reuse story). "
                        "'prev' = consecutive pairs (i-1 → i, speculative cascade story).")
    p.add_argument("--fast_dllm_path", type=str, default=None,
                   help="Path to Fast-dLLM v1 (used by prediction_overlap and layered_heatmap, "
                        "which load the model). Defaults to env FAST_DLLM_V1_PATH or "
                        "./external/Fast-dLLM/v1.")
    p.add_argument("--layered_heatmap_layers", type=str, default="1,8,16,24,32",
                   help="Comma-separated layer indices to plot in the layered_heatmap variant. "
                        "First 5 are used for panels 1-5; panel 6 is always shared_mass. "
                        "Default '1,8,16,24,32' (LLaDA-32 quantiles); for Dream try '1,7,14,21,28'.")
    p.add_argument("--layered_heatmap_n_samples", type=int, default=5,
                   help="Number of samples to plot for the layered_heatmap variant. Default 5.")
    args = p.parse_args()

    out_dir = args.probes_root / args.model / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = METRICS if args.metric == "all" else (args.metric,)

    do_grouped = args.variant in ("drift_grouped", "all")
    do_heatmap = args.variant in ("diff_heatmap", "all")
    do_vs_pass0 = args.variant in ("drift_vs_pass0", "all")
    # `prediction_overlap` and `layered_heatmap` load the model; only run
    # when explicitly asked — not included in --variant all.
    do_prediction_overlap = args.variant == "prediction_overlap"
    do_layered_heatmap = args.variant == "layered_heatmap"
    blocks_to_use = [int(x) for x in args.vs_pass0_blocks.split(",") if x.strip()]
    layered_layers = [int(x) for x in args.layered_heatmap_layers.split(",") if x.strip()]

    if do_prediction_overlap:
        # prediction_overlap is metric-agnostic (the metric is fixed:
        # shared_mass on softmax(lm_head(ln_f(h_last)))). Run it once.
        stats = _accumulate_prediction_overlap(
            args.probes_root, args.model, args.fast_dllm_path,
            blocks_to_use=blocks_to_use,
            n_samples=args.vs_pass0_n_samples,
            max_iter=args.vs_pass0_max_iter,
            reference=args.prediction_overlap_reference,
        )
        if stats["sum_mm"] is None:
            print("WARN: no intra_block samples found for prediction_overlap.")
        else:
            blocks_tag = "-".join(str(b) for b in blocks_to_use)
            out = out_dir / (
                f"intra_block_prediction_overlap_blocks{blocks_tag}"
                f"_n{stats['n_samples']}_ref{args.prediction_overlap_reference}.png"
            )
            _plot_prediction_overlap(stats, out, args.model)
        return

    if do_layered_heatmap:
        # Iterates per (sample, block); writes one figure per pair.
        produced = 0
        for sample_id, block_idx, payload in _accumulate_layered_heatmaps(
            args.probes_root, args.model, args.fast_dllm_path,
            blocks_to_use=blocks_to_use,
            n_samples=args.layered_heatmap_n_samples,
            layers_to_plot=layered_layers,
        ):
            out = out_dir / (
                f"intra_block_layered_heatmap_{sample_id}_block{block_idx}.png"
            )
            _plot_layered_heatmap(
                payload, sample_id, block_idx, args.model, out, layered_layers,
            )
            produced += 1
        if produced == 0:
            print("WARN: no intra_block samples found for layered_heatmap.")
        return

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

        if do_vs_pass0:
            stats = _accumulate_vs_pass0(
                args.probes_root, args.model, metric,
                blocks_to_use=blocks_to_use,
                n_samples=args.vs_pass0_n_samples,
                max_iter=args.vs_pass0_max_iter,
            )
            if stats["n_layers_p1"] is None:
                print(f"WARN: no intra_block samples found for drift_vs_pass0/{metric}.")
            else:
                blocks_tag = "-".join(str(b) for b in blocks_to_use)
                out = out_dir / (
                    f"intra_block_drift_vs_pass0_{metric}"
                    f"_blocks{blocks_tag}_n{stats['n_samples']}.png"
                )
                _plot_drift_vs_pass0(stats, out, args.model, metric)


if __name__ == "__main__":
    main()
