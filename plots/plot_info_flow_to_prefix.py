"""Plot Probe A: raw-attention and info-flow to prefix.

Two figures per model: raw-attn variant and flow variant. Each has 8 per-block facets and a summary.

Run from the T3 project root:
    python -m probe_runner.plots.plot_info_flow_to_prefix --model llada
    python -m probe_runner.plots.plot_info_flow_to_prefix --model dream
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_runner import storage


def _per_layer_per_block_signal(
    files: list[Path], variant: str, sink_positions: set[int]
) -> tuple[np.ndarray, list[int]]:
    """Return signal of shape [num_blocks, L, H] averaged across samples and masked positions.

    `variant`: "attn" → sum of raw attention to prefix.
               "flow" → sum of (attn * v_norm) to prefix.
    """
    accumulators: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}

    for f in files:
        sample = storage.read_h5(f)
        for b, blk in sample["blocks"].items():
            attn = blk["attn"].astype(np.float32)            # [num_masked, L, H, S_b]
            num_masked, L, H, S_b = attn.shape
            prefix_len = S_b - num_masked
            # Build prefix mask, excluding sinks
            prefix_idx = [j for j in range(prefix_len) if j not in sink_positions]
            attn_prefix = attn[..., prefix_idx]               # [num_masked, L, H, |prefix|]
            if variant == "attn":
                contrib = attn_prefix.sum(axis=-1)             # [num_masked, L, H]
            elif variant == "flow":
                v_norm = blk.get("v_norm")
                if v_norm is None:
                    continue
                v_norm = v_norm.astype(np.float32)             # [L, H, S_b]
                v_pref = v_norm[..., prefix_idx]               # [L, H, |prefix|]
                # flow per (m, ℓ, h, j) = attn[m,ℓ,h,j] * v_norm[ℓ,h,j]
                contrib = (attn_prefix * v_pref[None, ...]).sum(axis=-1)  # [num_masked, L, H]
            else:
                raise ValueError(variant)
            mean = contrib.mean(axis=0)                        # [L, H]
            if b not in accumulators:
                accumulators[b] = mean
                counts[b] = 1
            else:
                accumulators[b] = accumulators[b] + mean
                counts[b] = counts[b] + 1

    blocks = sorted(accumulators.keys())
    arr = np.stack([accumulators[b] / counts[b] for b in blocks], axis=0)  # [num_blocks, L, H]
    return arr, blocks


def _plot(arr: np.ndarray, blocks: list[int], title: str, out_path: Path):
    """Render the standard 8-block facet grid + summary."""
    num_blocks, L, H = arr.shape
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[3, 3, 2])

    # Per-block heatmaps in a 2×4 grid
    vmin, vmax = float(arr.min()), float(arr.max())
    for i, b in enumerate(blocks[:8]):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        im = ax.imshow(arr[i].T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"block {b}")
        ax.set_xlabel("layer ℓ")
        ax.set_ylabel("head h")
    fig.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.02, shrink=0.6)

    # Summary curves: per-layer mean over heads, one curve per block + overall mean
    ax = fig.add_subplot(gs[2, :])
    layer_curves = arr.mean(axis=-1)  # [num_blocks, L]
    for i, b in enumerate(blocks):
        ax.plot(range(L), layer_curves[i], label=f"block {b}", alpha=0.6)
    ax.plot(range(L), layer_curves.mean(axis=0), label="mean over blocks", color="black", linewidth=2)
    ax.set_xlabel("layer ℓ")
    ax.set_ylabel("signal")
    ax.set_title(title + " — per-layer (mean over heads)")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llada", "dream"], required=True)
    parser.add_argument("--probes_root", type=str, default="probes_out")
    parser.add_argument("--meta_path", type=str, default=None)
    args = parser.parse_args()

    probes_root = Path(args.probes_root)
    sample_dir = probes_root / args.model
    plots_dir = probes_root / "plots"
    files = sorted(sample_dir.glob("sample_*.h5"))
    if not files:
        raise SystemExit(f"No samples in {sample_dir}")

    # Read sink positions from meta if present
    sink_positions = {0}
    meta_path = Path(args.meta_path) if args.meta_path else probes_root / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        for_model = meta.get("models", {}).get(args.model, {})
        sink_positions = set(for_model.get("attention_sink_positions", [0]))

    # Variant 1: raw attention
    print(f"[{args.model}] computing raw-attn variant on {len(files)} samples …")
    attn_arr, blocks = _per_layer_per_block_signal(files, "attn", sink_positions)
    _plot(attn_arr, blocks, f"{args.model} — raw attention to prefix",
          plots_dir / f"info_flow_to_prefix_{args.model}_rawattn.png")

    # Variant 2: info flow
    print(f"[{args.model}] computing flow variant …")
    flow_arr, _ = _per_layer_per_block_signal(files, "flow", sink_positions)
    _plot(flow_arr, blocks, f"{args.model} — info flow (attn × ||W_O·v||) to prefix",
          plots_dir / f"info_flow_to_prefix_{args.model}_flow.png")

    print(f"[{args.model}] saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
