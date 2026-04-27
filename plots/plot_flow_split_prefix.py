"""Plot Probe A with three partitions: RECENT prefix, DISTANT prefix, CURRENT block.

For each block k:
  - "recent" prefix  = positions in [block_start - W, block_start), excluding sinks/specials.
  - "distant" prefix = positions in [0, block_start - W),            excluding sinks/specials.
  - "current" block  = positions in [block_start, block_start + 32), excluding sinks/specials
                       (essentially the masked block itself; gives a "reverse look" — how
                        much each masked token attends to others within the same block).

Where W = --recent_window (default 8 tokens).

For the *_normalized variants the denominator is:
  * default (--include_future OFF): old + recent + current  (positions [0, block_end))
  * with --include_future:           above + future masked blocks  (positions [0, S_b))

Same EOS-cutoff and special-token-subtraction filters as plot_info_flow_to_prefix.py.

Run:
    python -m probe_runner.plots.plot_flow_split_prefix --model llada
    python -m probe_runner.plots.plot_flow_split_prefix --model llada --variant flow_normalized
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict]:
    """Compute (recent_arr, distant_arr, current_arr) of shape [num_blocks, L, H] each.

    For blocks where the "distant" partition is empty (e.g., very short prompts at block 0),
    that block's slot is NaN; the plotter masks those out.
    """
    accum_recent: dict[int, np.ndarray] = {}
    accum_distant: dict[int, np.ndarray] = {}
    accum_current: dict[int, np.ndarray] = {}
    counts_recent: dict[int, int] = {}
    counts_distant: dict[int, int] = {}
    counts_current: dict[int, int] = {}
    diagnostics = {
        "n_files": len(files),
        "recent_window": recent_window,
        "include_future": include_future,
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
            block_end_abs = block_start_abs + num_masked
            recent_lo = max(0, block_start_abs - recent_window)
            recent_hi = block_start_abs

            recent_idx  = [j for j in range(recent_lo,         recent_hi)     if j not in special_positions]
            distant_idx = [j for j in range(0,                 recent_lo)     if j not in special_positions]
            current_idx = [j for j in range(block_start_abs,   block_end_abs) if j not in special_positions]

            v_norm = blk.get("v_norm")
            if variant in ("flow", "flow_normalized") and v_norm is None:
                continue
            if v_norm is not None:
                v_norm = v_norm.astype(np.float32)

            # Denominator for *_normalized:
            #   include_future=False  → [0, block_end) excluding sinks/specials
            #                            = old + recent + current
            #   include_future=True   → [0, S_b) excluding sinks/specials
            if include_future:
                denom_idx = [j for j in range(S_b) if j not in special_positions]
            else:
                denom_idx = [j for j in range(block_end_abs) if j not in special_positions]

            denom_total = None
            if variant == "attn_normalized" and len(denom_idx) > 0:
                denom_total = attn_kept[..., denom_idx].sum(axis=-1)
            elif variant == "flow_normalized" and len(denom_idx) > 0 and v_norm is not None:
                v_d = v_norm[..., denom_idx]
                a_d = attn_kept[..., denom_idx]
                denom_total = (a_d * v_d[None, ...]).sum(axis=-1)

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

            for accum, counts, idx in (
                (accum_recent,  counts_recent,  recent_idx),
                (accum_distant, counts_distant, distant_idx),
                (accum_current, counts_current, current_idx),
            ):
                sig = _signal_over(idx)
                if sig is None:
                    continue
                m = sig.mean(axis=0)
                # Find which dict by identity to update count
                for b_, _ in [(b, None)]:
                    pass
                if b not in accum:
                    accum[b] = m
                    counts[b] = 1
                else:
                    accum[b] = accum[b] + m
                    counts[b] = counts[b] + 1

            diagnostics["blocks_kept_per_sample"][b] = diagnostics["blocks_kept_per_sample"].get(b, 0) + 1

    diagnostics["mean_eos_pos"] = float(np.mean(eos_positions)) if eos_positions else 0.0
    blocks = sorted(set(accum_recent) | set(accum_distant) | set(accum_current))

    def _stack(accum, counts, blocks_):
        if not accum:
            # Build a NaN array with a plausible shape from any other accum
            return np.full((len(blocks_), 1, 1), np.nan)
        ref_shape = next(iter(accum.values())).shape
        out = []
        for b in blocks_:
            if b in accum:
                out.append(accum[b] / counts[b])
            else:
                out.append(np.full(ref_shape, np.nan))
        return np.stack(out, axis=0)

    recent_arr  = _stack(accum_recent,  counts_recent,  blocks)
    distant_arr = _stack(accum_distant, counts_distant, blocks)
    current_arr = _stack(accum_current, counts_current, blocks)
    return recent_arr, distant_arr, current_arr, blocks, diagnostics


# ---------------------------------------------------------------------------
# Plotting (uses subfigures so section labels never overlap subplot titles)
# ---------------------------------------------------------------------------

def _plot_split_3way(
    recent_arr: np.ndarray,
    distant_arr: np.ndarray,
    current_arr: np.ndarray,
    blocks: list[int],
    title: str,
    out_path: Path,
    y_label: str,
    recent_window: int,
):
    num_blocks = recent_arr.shape[0]
    L = recent_arr.shape[1] if recent_arr.shape[1] > 1 else (
        distant_arr.shape[1] if distant_arr.shape[1] > 1 else current_arr.shape[1]
    )

    # Joint color scale across all three partitions
    finite = []
    for a in (recent_arr, distant_arr, current_arr):
        flat = a[~np.isnan(a)].ravel()
        if flat.size > 0:
            finite.append(flat)
    finite = np.concatenate(finite) if finite else np.array([0.0, 1.0])
    vmin, vmax = float(finite.min()), float(finite.max())

    fig = plt.figure(figsize=(22, 26))
    fig.suptitle(title, fontsize=15, y=0.995, fontweight="bold")

    subfigs = fig.subfigures(
        4, 1,
        height_ratios=[3.0, 3.0, 3.0, 2.5],
        hspace=0.04,
    )

    sections = [
        (subfigs[0], recent_arr,  f"RECENT prefix  [-{recent_window}:]  (last {recent_window} tokens before block start)", "tab:blue"),
        (subfigs[1], distant_arr, f"DISTANT prefix  [0:-{recent_window}]  (everything earlier)",                            "tab:red"),
        (subfigs[2], current_arr,  "CURRENT BLOCK  (the 32 mask positions of block k itself — reverse look)",               "tab:green"),
    ]

    last_im = None
    for sf, arr, label, color in sections:
        sf.suptitle(label, fontsize=13, color=color, fontweight="bold", y=0.99)
        axes = sf.subplots(2, 4)
        for i, b in enumerate(blocks[:8]):
            ax = axes[i // 4, i % 4]
            if not np.all(np.isnan(arr[i])):
                last_im = ax.imshow(
                    arr[i].T, aspect="auto", origin="lower",
                    vmin=vmin, vmax=vmax, cmap="viridis",
                )
            ax.set_title(f"block {b}", fontsize=10)
            ax.set_xlabel("layer ℓ", fontsize=9)
            ax.set_ylabel("head h", fontsize=9)
        # Tighten internal spacing
        sf.subplots_adjust(top=0.86, bottom=0.10, left=0.05, right=0.97, hspace=0.55, wspace=0.30)

    # Shared colorbar at the right of the heatmap rows
    if last_im is not None:
        cbar_ax = fig.add_axes([0.985, 0.30, 0.008, 0.55])
        fig.colorbar(last_im, cax=cbar_ax)

    # Summary subfigure: 3 side-by-side line plots (recent | distant | current_block)
    sf_summary = subfigs[3]
    sf_summary.suptitle("Per-layer summary (mean over heads)", fontsize=12, y=0.97)
    axes_s = sf_summary.subplots(1, 3)
    cmap = plt.colormaps["tab10"]

    for ax_s, arr, label, color in zip(
        axes_s,
        [recent_arr, distant_arr, current_arr],
        ["RECENT", "DISTANT", "CURRENT BLOCK"],
        ["tab:blue", "tab:red", "tab:green"],
    ):
        curves = np.nanmean(arr, axis=-1)  # [num_blocks, L]
        for i, b in enumerate(blocks):
            if np.all(np.isnan(curves[i])):
                continue
            ax_s.plot(range(L), curves[i], label=f"block {b}", alpha=0.7, color=cmap(i % 10))
        if not np.all(np.isnan(curves)):
            ax_s.plot(range(L), np.nanmean(curves, axis=0),
                      color="black", linewidth=2.5, label="mean")
        ax_s.set_xlabel("layer ℓ")
        ax_s.set_ylabel(y_label)
        ax_s.set_title(label, fontsize=12, color=color, fontweight="bold")
        ax_s.legend(ncol=2, fontsize=8, loc="best")
        ax_s.grid(True, alpha=0.3)
    sf_summary.subplots_adjust(top=0.85, bottom=0.18, left=0.05, right=0.97, wspace=0.25)

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
        recent_arr, distant_arr, current_arr, blocks, diag = _per_layer_per_block_split_signal(
            files, v, model_name=args.model, recent_window=args.recent_window,
            apply_eos_cutoff=apply_eos, apply_special_filter=apply_special,
            include_future=args.include_future,
        )
        if diagnostics_collected is None:
            diagnostics_collected = diag

        if v == "attn":
            title = f"{args.model} — raw attention split (W={args.recent_window})"
            ylabel = "Σ attn ∈ [0, 1]"
        elif v == "flow":
            title = f"{args.model} — info flow (attn × ||W_O·v||) split (W={args.recent_window})"
            ylabel = "Σ (attn · v_norm)"
        elif v == "attn_normalized":
            title = (f"{args.model} — normalized raw attention split (W={args.recent_window})\n"
                     f"denominator: {denom_label}")
            ylabel = "attn_part / attn_total ∈ [0, 1]"
        else:  # flow_normalized
            title = (f"{args.model} — normalized info flow split (W={args.recent_window})\n"
                     f"denominator: {denom_label}")
            ylabel = "flow_part / flow_total ∈ [0, 1]"

        out_png = plots_dir / f"flow_split_W{args.recent_window}_{args.model}_{v}{suffix_str}.png"
        _plot_split_3way(
            recent_arr, distant_arr, current_arr, blocks,
            title=title, out_path=out_png, y_label=ylabel, recent_window=args.recent_window,
        )
        print(f"[{args.model}] saved {out_png}")

    if diagnostics_collected is not None:
        diag_path = plots_dir / f"diagnostics_split_{args.model}{suffix_str}.json"
        with open(diag_path, "w") as fp:
            json.dump(diagnostics_collected, fp, indent=2)
        print(f"[{args.model}] saved {diag_path}")


if __name__ == "__main__":
    main()
