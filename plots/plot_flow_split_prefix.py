"""Plot Probe A with prefix split into RECENT vs DISTANT.

For each block k:
  - "recent" prefix = positions in [block_start - W, block_start), excluding sinks/specials.
  - "distant" prefix = positions in [0, block_start - W), excluding sinks/specials.

Where W = --recent_window (default 8 tokens). This separates "local context immediately
before the masked block" from "the rest of the prompt and earlier decoded blocks."

Same EOS-cutoff and special-token-subtraction filters as plot_info_flow_to_prefix.py.

Run:
    python -m probe_runner.plots.plot_flow_split_prefix --model llada
    python -m probe_runner.plots.plot_flow_split_prefix --model llada --variant flow --recent_window 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_runner import storage
from probe_runner.plots.plot_info_flow_to_prefix import (
    _resolve_special_positions,
    _resolve_eos_pos,
    _allowed_masked_positions,
)


def _per_layer_per_block_split_signal(
    files: list[Path],
    variant: str,
    model_name: str,
    recent_window: int,
    apply_eos_cutoff: bool,
    apply_special_filter: bool,
    include_future: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int], dict]:
    """Compute (recent_arr, distant_arr) of shape [num_blocks, L, H] each.

    For block 0 the "distant" partition may be empty (if block_start ≤ recent_window);
    those blocks return NaN for the distant array, which the plotter skips.
    """
    accum_recent: dict[int, np.ndarray] = {}
    accum_distant: dict[int, np.ndarray] = {}
    counts_recent: dict[int, int] = {}
    counts_distant: dict[int, int] = {}
    diagnostics = {
        "n_files": len(files),
        "recent_window": recent_window,
        "samples_with_eos": 0,
        "mean_eos_pos": 0.0,
        "blocks_kept_per_sample": {b: 0 for b in range(8)},
    }
    eos_positions = []

    for f in files:
        sample = storage.read_h5(f)
        prompt_len = int(sample["attrs"]["prompt_len"])
        special_positions = _resolve_special_positions(sample, model_name) if apply_special_filter else {0}
        eos_pos = _resolve_eos_pos(sample, model_name) if apply_eos_cutoff else 10**9
        eos_positions.append(eos_pos)
        if eos_pos < 256:
            diagnostics["samples_with_eos"] += 1

        for b, blk in sample["blocks"].items():
            attn = blk["attn"].astype(np.float32)            # [num_masked, L, H, S_b]
            num_masked, L, H, S_b = attn.shape

            allowed_m = _allowed_masked_positions(b, num_masked, eos_pos)
            if not allowed_m:
                continue
            attn_kept = attn[allowed_m]

            block_start_abs = prompt_len + b * num_masked
            recent_lo = max(0, block_start_abs - recent_window)
            recent_hi = block_start_abs

            recent_idx = [j for j in range(recent_lo, recent_hi) if j not in special_positions]
            distant_idx = [j for j in range(0, recent_lo) if j not in special_positions]

            v_norm = blk.get("v_norm")
            if variant in ("flow", "flow_normalized", "attn_normalized") and v_norm is None and variant != "attn_normalized":
                continue
            if v_norm is not None:
                v_norm = v_norm.astype(np.float32)

            # Denominator index list for the *_normalized variants:
            #   include_future=False  → [0, block_end) excluding sinks/specials
            #                            = old prefix + recent prefix + current block
            #   include_future=True   → [0, S_b) excluding sinks/specials
            #                            = above + future masked blocks
            block_end_abs = block_start_abs + num_masked
            if include_future:
                denom_idx = [j for j in range(S_b) if j not in special_positions]
            else:
                denom_idx = [j for j in range(block_end_abs) if j not in special_positions]

            # Precompute the total once per (sample, block) for normalized variants
            denom_total = None
            if variant == "attn_normalized" and len(denom_idx) > 0:
                denom_total = attn_kept[..., denom_idx].sum(axis=-1)            # [n_kept, L, H]
            elif variant == "flow_normalized" and len(denom_idx) > 0 and v_norm is not None:
                v_d = v_norm[..., denom_idx]
                a_d = attn_kept[..., denom_idx]
                denom_total = (a_d * v_d[None, ...]).sum(axis=-1)               # [n_kept, L, H]

            def _signal_over(idx_list):
                if len(idx_list) == 0:
                    return None
                a = attn_kept[..., idx_list]
                if variant == "attn":
                    return a.sum(axis=-1)
                if variant == "flow":
                    v = v_norm[..., idx_list]
                    return (a * v[None, ...]).sum(axis=-1)
                if variant == "attn_normalized":
                    if denom_total is None:
                        return None
                    return a.sum(axis=-1) / (denom_total + 1e-9)
                if variant == "flow_normalized":
                    if denom_total is None:
                        return None
                    v = v_norm[..., idx_list]
                    flow_part = (a * v[None, ...]).sum(axis=-1)
                    return flow_part / (denom_total + 1e-9)
                raise ValueError(variant)

            sig_recent = _signal_over(recent_idx)
            sig_distant = _signal_over(distant_idx)

            if sig_recent is not None:
                m = sig_recent.mean(axis=0)
                if b not in accum_recent:
                    accum_recent[b] = m
                    counts_recent[b] = 1
                else:
                    accum_recent[b] = accum_recent[b] + m
                    counts_recent[b] = counts_recent[b] + 1
            if sig_distant is not None:
                m = sig_distant.mean(axis=0)
                if b not in accum_distant:
                    accum_distant[b] = m
                    counts_distant[b] = 1
                else:
                    accum_distant[b] = accum_distant[b] + m
                    counts_distant[b] = counts_distant[b] + 1

            diagnostics["blocks_kept_per_sample"][b] = diagnostics["blocks_kept_per_sample"].get(b, 0) + 1

    diagnostics["mean_eos_pos"] = float(np.mean(eos_positions)) if eos_positions else 0.0
    blocks = sorted(set(accum_recent.keys()) | set(accum_distant.keys()))

    def _stack(accum, counts, blocks_):
        out = []
        for b in blocks_:
            if b in accum:
                out.append(accum[b] / counts[b])
            else:
                # Pad with NaN so plotting can mask it out
                shape = next(iter(accum.values())).shape if accum else None
                out.append(np.full(shape, np.nan) if shape else None)
        return np.stack(out, axis=0)

    recent_arr = _stack(accum_recent, counts_recent, blocks)
    distant_arr = _stack(accum_distant, counts_distant, blocks)
    return recent_arr, distant_arr, blocks, diagnostics


def _plot_split(
    recent_arr: np.ndarray,
    distant_arr: np.ndarray,
    blocks: list[int],
    title: str,
    out_path: Path,
    y_label: str,
):
    """One figure with two stacked 8-block heatmap rows + overlaid summary curves."""
    num_blocks, L, H = recent_arr.shape
    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    gs = fig.add_gridspec(5, 4, height_ratios=[3, 3, 3, 3, 3])

    # Joint color scale across both partitions for fair visual comparison
    finite_vals = np.concatenate([
        recent_arr[~np.isnan(recent_arr)].ravel(),
        distant_arr[~np.isnan(distant_arr)].ravel(),
    ])
    vmin, vmax = float(finite_vals.min()), float(finite_vals.max())

    last_im = None

    # Row 1+2: recent heatmaps (8 subplots)
    fig.text(0.5, 0.995, "RECENT prefix (last W tokens before block start)",
             ha="center", fontsize=11, color="darkblue")
    for i, b in enumerate(blocks[:8]):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        if not np.all(np.isnan(recent_arr[i])):
            last_im = ax.imshow(recent_arr[i].T, aspect="auto", origin="lower",
                                vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"block {b} — recent")
        ax.set_xlabel("layer ℓ")
        ax.set_ylabel("head h")

    # Row 3+4: distant heatmaps
    fig.text(0.5, 0.595, "DISTANT prefix (everything earlier)",
             ha="center", fontsize=11, color="darkred")
    for i, b in enumerate(blocks[:8]):
        ax = fig.add_subplot(gs[2 + i // 4, i % 4])
        if not np.all(np.isnan(distant_arr[i])):
            last_im = ax.imshow(distant_arr[i].T, aspect="auto", origin="lower",
                                vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"block {b} — distant")
        ax.set_xlabel("layer ℓ")
        ax.set_ylabel("head h")

    if last_im is not None:
        fig.colorbar(last_im, ax=fig.axes, fraction=0.02, pad=0.02, shrink=0.6)

    # Row 5: summary curves (recent solid, distant dashed)
    ax = fig.add_subplot(gs[4, :])
    recent_curves = np.nanmean(recent_arr, axis=-1)   # [num_blocks, L]
    distant_curves = np.nanmean(distant_arr, axis=-1)  # [num_blocks, L]
    cmap = plt.colormaps["tab10"]
    for i, b in enumerate(blocks):
        c = cmap(i % 10)
        ax.plot(range(L), recent_curves[i], label=f"block {b} recent", alpha=0.7,
                color=c, linestyle="-")
        if not np.all(np.isnan(distant_curves[i])):
            ax.plot(range(L), distant_curves[i], label=f"block {b} distant", alpha=0.7,
                    color=c, linestyle="--")
    ax.plot(range(L), np.nanmean(recent_curves, axis=0), color="black", linewidth=2.5,
            label="mean RECENT")
    ax.plot(range(L), np.nanmean(distant_curves, axis=0), color="dimgray", linewidth=2.5,
            linestyle="--", label="mean DISTANT")
    ax.set_xlabel("layer ℓ")
    ax.set_ylabel(y_label)
    ax.set_title(title + " — per-layer (mean over heads). Solid = RECENT, Dashed = DISTANT.")
    ax.legend(ncol=4, fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llada", "dream"], required=True)
    parser.add_argument("--probes_root", type=str, default="probes_out")
    parser.add_argument(
        "--variant",
        choices=["attn", "flow", "attn_normalized", "flow_normalized", "all"],
        default="all",
    )
    parser.add_argument("--recent_window", type=int, default=8,
                        help="How many tokens before block_start count as 'recent prefix'.")
    parser.add_argument("--no_eos_cutoff", action="store_true")
    parser.add_argument("--no_special_filter", action="store_true")
    parser.add_argument(
        "--include_future",
        action="store_true",
        help="For *_normalized variants: include future masked blocks in the denominator. "
             "Default OFF — denominator = old_prefix + recent_prefix + current_block.",
    )
    args = parser.parse_args()

    probes_root = Path(args.probes_root)
    sample_dir = probes_root / args.model
    plots_dir = probes_root / "plots"
    files = sorted(sample_dir.glob("sample_*.h5"))
    if not files:
        raise SystemExit(f"No samples in {sample_dir}")

    apply_eos = not args.no_eos_cutoff
    apply_special = not args.no_special_filter
    if args.variant == "all":
        variants = ["attn", "flow", "attn_normalized", "flow_normalized"]
    else:
        variants = [args.variant]

    suffix = []
    if apply_eos:
        suffix.append("eosfilter")
    if apply_special:
        suffix.append("specialfilter")
    if args.include_future:
        suffix.append("incfut")
    suffix_str = "_" + "_".join(suffix) if suffix else ""

    denom_label = ("old + recent + current + future"
                   if args.include_future else
                   "old + recent + current  (no future masks)")

    diagnostics_collected = None
    for v in variants:
        print(f"[{args.model}] split-prefix variant={v} W={args.recent_window} "
              f"include_future={args.include_future} on {len(files)} files …")
        recent_arr, distant_arr, blocks, diag = _per_layer_per_block_split_signal(
            files, v, model_name=args.model, recent_window=args.recent_window,
            apply_eos_cutoff=apply_eos, apply_special_filter=apply_special,
            include_future=args.include_future,
        )
        if diagnostics_collected is None:
            diagnostics_collected = diag
        if v == "attn":
            title = f"{args.model} — raw attention to prefix (split, W={args.recent_window})"
            ylabel = "Σ attn ∈ [0, 1]"
        elif v == "flow":
            title = f"{args.model} — info flow (attn × ||W_O·v||) to prefix (split, W={args.recent_window})"
            ylabel = "Σ (attn · v_norm)"
        elif v == "attn_normalized":
            title = (f"{args.model} — normalized raw attention (split, W={args.recent_window})\n"
                     f"denominator: {denom_label}")
            ylabel = "attn_part / attn_total ∈ [0, 1]"
        else:  # flow_normalized
            title = (f"{args.model} — normalized info flow (split, W={args.recent_window})\n"
                     f"denominator: {denom_label}")
            ylabel = "flow_part / flow_total ∈ [0, 1]"
        out_png = plots_dir / f"flow_split_W{args.recent_window}_{args.model}_{v}{suffix_str}.png"
        _plot_split(recent_arr, distant_arr, blocks, title, out_png, y_label=ylabel)
        print(f"[{args.model}] saved {out_png}")

    if diagnostics_collected is not None:
        diag_path = plots_dir / f"diagnostics_split_{args.model}{suffix_str}.json"
        with open(diag_path, "w") as fp:
            json.dump(diagnostics_collected, fp, indent=2)
        print(f"[{args.model}] saved {diag_path}")


if __name__ == "__main__":
    main()
