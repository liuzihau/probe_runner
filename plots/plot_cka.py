"""Plot Probe B: layer-to-final CKA over masked positions.

Two views per model: per-block CKA curves + pooled curve.

Run from the T3 project root:
    python -m probe_runner.plots.plot_cka --model llada
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_runner import storage


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two [N, d] matrices.

    CKA(X, Y) = ||Y^T X||_F^2 / ( ||X^T X||_F * ||Y^T Y||_F )
    """
    X = _center(X.astype(np.float64))
    Y = _center(Y.astype(np.float64))
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y
    num = (XtY ** 2).sum()
    den = np.sqrt((XtX ** 2).sum() * (YtY ** 2).sum())
    if den == 0:
        return 0.0
    return float(num / den)


def _per_block_cka_curves(files: list[Path]) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Return (per_block_curves dict, pooled_curve).

    per_block_curves: {block_idx: array of shape [L+1] giving CKA(h_ℓ, h_L)}
    pooled_curve: array of shape [L+1] computed by concatenating masked positions across all blocks.
    """
    # Pool buffers: per layer, accumulate hidden states from all (sample, block, masked_position)
    pooled: dict[int, list[np.ndarray]] = {}    # ℓ → list of [num_masked, d_model] arrays
    per_block_pooled: dict[int, dict[int, list[np.ndarray]]] = {}  # block_idx → ℓ → list

    L_plus_1 = None
    for f in files:
        sample = storage.read_h5(f)
        for b, blk in sample["blocks"].items():
            h = blk["h_masked"].astype(np.float32)  # [L+1, num_masked, d_model]
            if L_plus_1 is None:
                L_plus_1 = h.shape[0]
            for layer in range(h.shape[0]):
                pooled.setdefault(layer, []).append(h[layer])
                per_block_pooled.setdefault(b, {}).setdefault(layer, []).append(h[layer])

    if L_plus_1 is None:
        raise SystemExit("No data found")

    # Pooled CKA
    h_final_all = np.concatenate(pooled[L_plus_1 - 1], axis=0)  # [N_total, d_model]
    pooled_curve = np.zeros(L_plus_1, dtype=np.float64)
    for layer in range(L_plus_1):
        h_layer_all = np.concatenate(pooled[layer], axis=0)
        pooled_curve[layer] = _linear_cka(h_layer_all, h_final_all)

    # Per-block CKA
    per_block_curves: dict[int, np.ndarray] = {}
    for b, layer_dict in per_block_pooled.items():
        h_final = np.concatenate(layer_dict[L_plus_1 - 1], axis=0)
        curve = np.zeros(L_plus_1, dtype=np.float64)
        for layer in range(L_plus_1):
            h_layer = np.concatenate(layer_dict[layer], axis=0)
            curve[layer] = _linear_cka(h_layer, h_final)
        per_block_curves[b] = curve

    return per_block_curves, pooled_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llada", "dream"], required=True)
    parser.add_argument("--probes_root", type=str, default="probes_out")
    args = parser.parse_args()

    probes_root = Path(args.probes_root)
    sample_dir = probes_root / args.model
    plots_dir = probes_root / "plots"
    files = sorted(sample_dir.glob("sample_*.h5"))
    if not files:
        raise SystemExit(f"No samples in {sample_dir}")

    print(f"[{args.model}] computing CKA on {len(files)} samples …")
    per_block, pooled = _per_block_cka_curves(files)

    L_plus_1 = pooled.shape[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    for b in sorted(per_block.keys()):
        ax.plot(range(L_plus_1), per_block[b], label=f"block {b}", alpha=0.5)
    ax.plot(range(L_plus_1), pooled, label="pooled (all blocks)", color="black", linewidth=2.5)
    ax.axhline(0.95, linestyle="--", color="red", alpha=0.6, label="CKA = 0.95")
    ax.set_xlabel("layer ℓ (0 = embedding output, L = final block output)")
    ax.set_ylabel("CKA(h_ℓ, h_L)")
    ax.set_title(f"{args.model} — CKA to final layer over masked positions")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=9, loc="lower right")

    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = plots_dir / f"cka_{args.model}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[{args.model}] saved {out_png}")


if __name__ == "__main__":
    main()
