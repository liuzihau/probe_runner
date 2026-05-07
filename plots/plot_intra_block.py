"""Intra-block drift analyses (Analysis A + B from the design notes).

Both analyses operate on the per-pass hidden-state captures produced by
`run_probes.py --intra_block`. Each sample's HDF5 has, per block, a
`h_per_pass [n_passes, L+1, num_masked, d_model]` dataset alongside the
existing pass-0 fields. We compute consecutive-pass L2 differences:

    diff[i, layer, p] = || h[i, layer, p] − h[i+1, layer, p] ||_2     (L2)

Two variants:

    --variant drift_grouped (Analysis A)
        Group positions by (token_state_pre[i], token_state_pre[i+1]) into
        MM (mask→mask), MC (mask→clean), CC (clean→clean). For each group
        plot mean L2 diff vs. layer, faceted by block.

    --variant diff_heatmap (Analysis B)
        For each (sample, block) build a [n_passes-1 × num_masked] heatmap
        whose cells are the layer-mean L2 diff. Cells where the position
        was revealed during that transition get a marker overlay. Saved
        either per-sample or pooled across samples (mean over samples).

Usage:

    python -m probe_runner.plots.plot_intra_block --model llada --variant drift_grouped
    python -m probe_runner.plots.plot_intra_block --model llada --variant diff_heatmap --pooled
    python -m probe_runner.plots.plot_intra_block --model llada --variant all

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
    """Return {block_idx: {h_per_pass, token_state_per_pass, revealed_per_pass, pass_indices}}.
    Empty dict if this sample wasn't recorded with --intra_block.
    """
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


# --- Diff computation ---------------------------------------------------------

def _l2_diffs(h_per_pass: np.ndarray) -> np.ndarray:
    """Consecutive-pass L2 differences.

    Args:
        h_per_pass: [n_passes, L+1, num_masked, d_model] (fp16 ok).
    Returns:
        diffs: [n_passes-1, L+1, num_masked] float32 — L2 norm over d_model.
    """
    h = h_per_pass.astype(np.float32, copy=False)
    delta = h[1:] - h[:-1]                  # [n_passes-1, L+1, num_masked, d_model]
    return np.linalg.norm(delta, axis=-1)   # [n_passes-1, L+1, num_masked]


# --- Analysis A: token-state-grouped drift ------------------------------------

GROUP_LABELS = {0b00: "MM", 0b01: "MC", 0b11: "CC"}


def _accumulate_grouped(probes_root: Path, model: str) -> dict:
    """Mean L2 diff per (block, layer, group) accumulated over samples."""
    n_samples = 0
    sums: dict[int, np.ndarray] = {}    # block -> [3, L+1] sum of mean-over-positions L2
    counts: dict[int, np.ndarray] = {}  # block -> [3, L+1] count for mean
    n_layers_p1 = None

    for path in _iter_sample_paths(probes_root, model):
        per_block = _load_intra_block(path)
        if not per_block:
            continue
        n_samples += 1
        for block_idx, data in per_block.items():
            h = data["h_per_pass"]                          # [P, L+1, M, D]
            ts = data["token_state_per_pass"].astype(np.int8)  # [P, M]  0=mask, 1=clean
            P, Lp1, M, _ = h.shape
            if n_layers_p1 is None:
                n_layers_p1 = Lp1
            if P < 2:
                continue
            diffs = _l2_diffs(h)                             # [P-1, L+1, M]
            # Group key per (i, p): 2*ts[i,p] + ts[i+1,p]  (mask=0, clean=1)
            key = (ts[:-1].astype(np.int32) << 1) | ts[1:].astype(np.int32)  # [P-1, M]
            sums.setdefault(block_idx, np.zeros((3, Lp1), dtype=np.float64))
            counts.setdefault(block_idx, np.zeros((3, Lp1), dtype=np.int64))
            for slot, k in enumerate((0b00, 0b01, 0b11)):  # MM, MC, CC
                mask = (key == k)
                if not mask.any():
                    continue
                # Average over the (i, p) entries that match this group, keep layer dim
                # diffs[i, layer, p] -> mean over (i, p) where mask[i, p]
                d_sel = diffs[mask[:, None, :].repeat(Lp1, axis=1).reshape(-1, Lp1).T
                              if False else None]  # placeholder; actual implementation below
                # Implementation: reshape to [-1, Lp1] indexed by flattened (i,p)
                d_flat = diffs.transpose(1, 0, 2).reshape(Lp1, -1)        # [Lp1, (P-1)*M]
                m_flat = mask.reshape(-1)                                  # [(P-1)*M]
                if m_flat.any():
                    sums[block_idx][slot] += d_flat[:, m_flat].sum(axis=1)
                    counts[block_idx][slot] += int(m_flat.sum())

    return {"sums": sums, "counts": counts, "n_layers_p1": n_layers_p1, "n_samples": n_samples}


def _plot_drift_grouped(stats: dict, out_path: Path, model: str) -> None:
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
    layer_idx = np.arange(n_layers_p1)

    colors = {"MM": "tab:gray", "MC": "tab:red", "CC": "tab:blue"}
    for ax_i, block_idx in enumerate(blocks):
        ax = axes[ax_i // ncols][ax_i % ncols]
        s = sums[block_idx]; c = counts[block_idx]
        with np.errstate(invalid="ignore"):
            mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        for slot, label in enumerate(("MM", "MC", "CC")):
            ax.plot(layer_idx, mean[slot], label=label, color=colors[label], lw=1.5)
        ax.set_title(f"block {block_idx}")
        ax.set_xlabel("layer")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("mean ‖h_i − h_{i+1}‖₂")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{model}: intra-block hidden-state drift, grouped by token-state transition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# --- Analysis B: per-(time, position) diff heatmap -----------------------------

def _accumulate_heatmap(probes_root: Path, model: str, *, pooled: bool) -> dict:
    """Per-block heatmap of (pass_i → i+1) cell = mean over layers of L2 diff.

    If pooled, averages across samples per block (also unioning revealed mask
    via OR — i.e., 'was this position ever revealed during this transition
    in any sample'). Counts are tracked per pass index so late passes (where
    fewer samples have data due to early-exit) aren't biased downward by
    dividing through samples that didn't reach that pass.
    """
    per_block: dict[int, dict] = {}
    for path in _iter_sample_paths(probes_root, model):
        for block_idx, data in _load_intra_block(path).items():
            h = data["h_per_pass"]                      # [P, L+1, M, D]
            rv = data["revealed_per_pass"].astype(bool) # [P, M]
            if h.shape[0] < 2:
                continue
            diffs = _l2_diffs(h)                         # [P-1, L+1, M]
            cell = diffs.mean(axis=1)                    # [P-1, M] — mean over layers
            revealed = rv[:-1]                           # [P-1, M]

            slot = per_block.setdefault(block_idx, {"sum": None,
                                                     "count_per_pass": None,
                                                     "n_samples": 0,
                                                     "revealed_or": None})
            P_minus_1, M = cell.shape
            if slot["sum"] is None:
                slot["sum"] = np.zeros((P_minus_1, M), dtype=np.float64)
                slot["revealed_or"] = np.zeros((P_minus_1, M), dtype=bool)
                slot["count_per_pass"] = np.zeros(P_minus_1, dtype=np.int64)
            # Pad if this sample has more passes than seen so far.
            if P_minus_1 > slot["sum"].shape[0]:
                pad_n = P_minus_1 - slot["sum"].shape[0]
                slot["sum"] = np.concatenate(
                    [slot["sum"], np.zeros((pad_n, M), dtype=np.float64)], axis=0)
                slot["revealed_or"] = np.concatenate(
                    [slot["revealed_or"], np.zeros((pad_n, M), dtype=bool)], axis=0)
                slot["count_per_pass"] = np.concatenate(
                    [slot["count_per_pass"], np.zeros(pad_n, dtype=np.int64)])
            slot["sum"][:P_minus_1] += cell
            slot["revealed_or"][:P_minus_1] |= revealed
            slot["count_per_pass"][:P_minus_1] += 1
            slot["n_samples"] += 1
    return per_block


def _plot_diff_heatmap(per_block: dict, out_path: Path, model: str) -> None:
    import matplotlib.pyplot as plt
    if not per_block:
        raise RuntimeError("No intra-block data found.")
    blocks = sorted(per_block.keys())
    n_blocks = len(blocks)
    ncols = min(2, n_blocks)
    nrows = int(np.ceil(n_blocks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 + 1.5 * nrows),
                             squeeze=False)
    for ax_i, block_idx in enumerate(blocks):
        ax = axes[ax_i // ncols][ax_i % ncols]
        slot = per_block[block_idx]
        # Per-pass mean: divide each row's sum by the per-pass count.
        denom = np.maximum(slot["count_per_pass"][:, None], 1).astype(np.float64)
        mean = slot["sum"] / denom                              # [n_passes-1, M]
        revealed = slot["revealed_or"]                          # [n_passes-1, M]
        im = ax.imshow(mean, aspect="auto", origin="lower",
                       interpolation="nearest", cmap="viridis")
        # Mark revealed cells with a small white ×.
        ys, xs = np.where(revealed)
        ax.scatter(xs, ys, marker="x", s=14, c="white", linewidths=0.7)
        ax.set_title(f"block {block_idx}  (n_samples={slot['n_samples']})")
        ax.set_xlabel("position in block")
        ax.set_ylabel("decode pass i (i → i+1)")
        fig.colorbar(im, ax=ax, label="mean-over-layers L2 diff")
    fig.suptitle(f"{model}: intra-block hidden-state diff heatmap "
                 "(× = revealed during transition)")
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
    p.add_argument("--pooled", action="store_true",
                   help="(diff_heatmap only) pool across samples; default already pools.")
    args = p.parse_args()

    out_dir = args.probes_root / args.model / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.variant in ("drift_grouped", "all"):
        stats = _accumulate_grouped(args.probes_root, args.model)
        if stats["n_samples"] == 0:
            print("WARN: no samples with intra_block data found for drift_grouped.")
        else:
            _plot_drift_grouped(stats, out_dir / "intra_block_drift_grouped.png", args.model)
            with (out_dir / "intra_block_drift_grouped.json").open("w") as f:
                json.dump({"n_samples": stats["n_samples"],
                           "blocks": list(stats["sums"].keys())}, f, indent=2)

    if args.variant in ("diff_heatmap", "all"):
        per_block = _accumulate_heatmap(args.probes_root, args.model, pooled=True)
        if not per_block:
            print("WARN: no samples with intra_block data found for diff_heatmap.")
        else:
            _plot_diff_heatmap(per_block, out_dir / "intra_block_diff_heatmap.png", args.model)


if __name__ == "__main__":
    main()
