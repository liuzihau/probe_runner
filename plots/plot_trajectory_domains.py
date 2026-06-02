"""A1/A3-A6 — 4-domain trajectory analysis for an iter-pair (a, b), gap allowed.

Splits decode positions into CU / CR / MC / MM (CR = committed-revised, DMax-only)
between two passes a < b (e.g. 0 vs 2), using committed_tokens_per_pass, and
reports:

  A1 domain populations           — fraction of positions in each domain.
  A3 last-layer hidden deviation  — ‖h_b − h_a‖₂ and cos(h_a, h_b) by domain.
  A5 per-layer hidden divergence  — cos / L2-normalized at every layer, one curve
                                    per domain → the divergence-onset layer (feeds
                                    the partial-depth refresh idea).
  A6 settle-iter                  — first pass after which a position stays at its
                                    converged token (distribution).

Headline (A2 logit-space deviation) lives in plot_rank_trajectory (needs the head).

Usage:
    python -m probe_runner.plots.plot_trajectory_domains --model llada2 --pair 0 1
    python -m probe_runner.plots.plot_trajectory_domains --model llada2 --pair 0 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc

ACTIVE_DOMAINS = ("CU", "CR", "MC", "MM")  # UC (committed->masked) is rare; folded into the report only


# ---- settle-iter (uses the whole trajectory, not a pair) --------------------

def settle_iters(blk: dict, mask_id: int = tc.MASK_ID) -> np.ndarray:
    """Per position: first pass k after which committed_tokens stays == converged.
    n_passes if it never settles (e.g. still masked at the end)."""
    ct = np.asarray(blk["committed_tokens_per_pass"])      # [P, B]
    conv = tc.converged_tokens(blk, mask_id)                # [B]
    P, B = ct.shape
    out = np.full(B, P, dtype=np.int64)
    for p in range(B):
        col = ct[:, p]
        for k in range(P):
            if np.all(col[k:] == conv[p]):
                out[p] = k
                break
    return out


# ---- accumulation -----------------------------------------------------------

def accumulate(probes_root: Path, model: str, a: int, b: int,
               block_length: int) -> dict:
    paths = tc.iter_sample_paths(probes_root, model)
    if not paths:
        raise RuntimeError(f"No samples under {Path(probes_root)/model}")

    ndom = len(tc.DOMAINS)
    pop = np.zeros(ndom, dtype=np.int64)
    l2_last_sum = np.zeros(ndom, dtype=np.float64)
    cos_last_sum = np.zeros(ndom, dtype=np.float64)
    # per-layer accumulators (resized once we know L+1)
    cos_layer_sum = None
    l2n_layer_sum = None
    layer_cnt = None
    settle_all = []
    n_samples = 0
    Lp1 = None

    for path in paths:
        sample = tc.load_sample(path)
        if not tc.has_committed(sample["blocks"]):
            continue
        used = False
        for bi, blk in sample["blocks"].items():
            h = blk["h_per_pass"]                          # [P, L+1, B, D]
            P, this_Lp1, B, _ = h.shape
            ai, bi_ = tc.resolve_pair(P, a, b)
            if ai == bi_:
                continue
            if Lp1 is None:
                Lp1 = this_Lp1
                cos_layer_sum = np.zeros((ndom, Lp1), dtype=np.float64)
                l2n_layer_sum = np.zeros((ndom, Lp1), dtype=np.float64)
                layer_cnt = np.zeros((ndom, Lp1), dtype=np.int64)
            if this_Lp1 != Lp1:
                continue
            tok_a = tc.committed_tokens_at(blk, ai)
            tok_b = tc.committed_tokens_at(blk, bi_)
            codes = tc.classify_domains(tok_a, tok_b)       # [B]
            valid = tc.valid_position_mask(bi, B, sample["attrs"], block_length)

            ha, hb = h[ai], h[bi_]                          # [L+1, B, D]
            cos_layer = tc.pair_deviation(ha, hb, "cosine_sim")     # [L+1, B]
            l2n_layer = tc.pair_deviation(ha, hb, "l2_normalized")  # [L+1, B]
            l2_last = tc.pair_deviation(ha[-1], hb[-1], "l2")       # [B]
            cos_last = cos_layer[-1]                                # [B]

            for code in range(ndom):
                sel = (codes == code) & valid
                n = int(sel.sum())
                if n == 0:
                    continue
                pop[code] += n
                l2_last_sum[code] += float(l2_last[sel].sum())
                cos_last_sum[code] += float(cos_last[sel].sum())
                cos_layer_sum[code] += cos_layer[:, sel].sum(axis=1)
                l2n_layer_sum[code] += l2n_layer[:, sel].sum(axis=1)
                layer_cnt[code] += n
            used = True

            st = settle_iters(blk).astype(np.float64)
            st = st / max(P - 1, 1)                          # normalize to [0,1]
            settle_all.append(st[valid])
        if used:
            n_samples += 1

    settle = np.concatenate(settle_all) if settle_all else np.array([])
    return {
        "pair": (a, b), "pop": pop, "l2_last_sum": l2_last_sum,
        "cos_last_sum": cos_last_sum, "cos_layer_sum": cos_layer_sum,
        "l2n_layer_sum": l2n_layer_sum, "layer_cnt": layer_cnt,
        "settle": settle, "n_samples": n_samples, "Lp1": Lp1,
    }


# ---- plotting ---------------------------------------------------------------

def plot(stats: dict, out_dir: Path, model: str) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b = stats["pair"]
    pop = stats["pop"].astype(np.float64)
    if pop.sum() == 0:
        raise RuntimeError("No domain data — need a llada2 --intra_block capture.")
    ndom = len(tc.DOMAINS)
    names = list(tc.DOMAINS)
    colors = [tc.DOMAIN_COLORS[n] for n in names]

    # ---------- Figure 1: A1 populations + A3 deviation + A6 settle ----------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    # A1 populations
    ax = axes[0]
    frac = pop / pop.sum()
    ax.bar(names, frac, color=colors)
    for i, fr in enumerate(frac):
        ax.text(i, fr, f"{fr:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"A1: domain populations  (pair {a}→{b})")
    ax.set_ylabel("fraction of positions")
    ax.set_ylim(0, max(frac) * 1.18 + 1e-6)

    # A3 last-layer deviation by domain (L2 + cosine)
    ax = axes[1]
    with np.errstate(invalid="ignore"):
        l2_mean = np.where(pop > 0, stats["l2_last_sum"] / np.maximum(pop, 1), np.nan)
        cos_mean = np.where(pop > 0, stats["cos_last_sum"] / np.maximum(pop, 1), np.nan)
    x = np.arange(ndom)
    ax.bar(x - 0.2, l2_mean, width=0.4, color=colors, label="‖Δh‖₂")
    ax.set_ylabel("last-layer ‖h_b − h_a‖₂")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax2 = ax.twinx()
    ax2.plot(x, cos_mean, "k^--", ms=6, label="cos(h_a,h_b)")
    ax2.set_ylabel("cosine(h_a, h_b)")
    ax2.set_ylim(min(0.0, np.nanmin(cos_mean) - 0.05), 1.02)
    ax.set_title("A3: last-layer hidden deviation by domain")
    ax2.legend(loc="lower right", fontsize=8)

    # A6 settle-iter distribution
    ax = axes[2]
    settle = stats["settle"]
    if settle.size:
        ax.hist(settle, bins=20, color="tab:purple", alpha=0.7, density=True)
        ax.axvline(float(np.median(settle)), ls="--", color="black",
                   label=f"median {np.median(settle):.2f}")
        ax.legend(fontsize=8)
    ax.set_title("A6: settle-iter (norm.)  — when a position locks to converged")
    ax.set_xlabel("settle pass / n_passes  (1.0 = never settles)")
    ax.set_ylabel("density")

    fig.suptitle(f"{model}: trajectory domains, pair {a}→{b}  "
                 f"[n_samples={stats['n_samples']}; CR is DMax-only]")
    fig.tight_layout()
    p1 = out_dir / f"trajectory_domains_pair_{a}_{b}.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p1}")

    # ---------- Figure 2: A5 per-layer divergence (onset) ----------
    if stats["Lp1"] is None:
        return
    Lp1 = stats["Lp1"]
    layers = np.arange(Lp1)
    with np.errstate(invalid="ignore"):
        cos_layer = np.where(stats["layer_cnt"] > 0,
                             stats["cos_layer_sum"] / np.maximum(stats["layer_cnt"], 1), np.nan)
        l2n_layer = np.where(stats["layer_cnt"] > 0,
                             stats["l2n_layer_sum"] / np.maximum(stats["layer_cnt"], 1), np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for code, name in enumerate(names):
        if stats["pop"][code] == 0:
            continue
        axes[0].plot(layers, cos_layer[code], marker="o", ms=3,
                     color=tc.DOMAIN_COLORS[name], label=name)
        axes[1].plot(layers, l2n_layer[code], marker="o", ms=3,
                     color=tc.DOMAIN_COLORS[name], label=name)
    axes[0].axhline(0.95, ls="--", color="black", alpha=0.45, label="0.95 (≈equal)")
    # divergence-onset annotation: first layer where the hardest live domain
    # (prefer CR, else MM) drops below 0.95.
    onset_dom = "CR" if stats["pop"][tc.DOMAIN_CODE["CR"]] > 0 else "MM"
    cc = cos_layer[tc.DOMAIN_CODE[onset_dom]]
    below = np.where(cc < 0.95)[0]
    if below.size:
        onset = int(below[0])
        axes[0].axvline(onset, ls=":", color="red", alpha=0.7)
        axes[0].text(onset, 0.96, f"{onset_dom} onset L{onset}", color="red",
                     fontsize=8, rotation=90, va="bottom")
    axes[0].set_title("A5: per-layer cosine(h_a, h_b) by domain")
    axes[0].set_xlabel("layer (0 = embed)")
    axes[0].set_ylabel("cosine similarity")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_title("A5: per-layer L2-normalized drift by domain")
    axes[1].set_xlabel("layer (0 = embed)")
    axes[1].set_ylabel("‖Δh‖ / mean‖h‖")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"{model}: A5 divergence-onset, pair {a}→{b}  "
                 f"(late onset ⇒ partial-depth refresh from that layer)")
    fig.tight_layout()
    p2 = out_dir / f"layerwise_divergence_pair_{a}_{b}.png"
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llada2")
    ap.add_argument("--probes_root", default="probes_out")
    ap.add_argument("--pair", type=int, nargs=2, default=[0, 1],
                    metavar=("A", "B"), help="iter-pair to compare (gap allowed); "
                                             "negative = from end (e.g. 0 -1).")
    ap.add_argument("--block_length", type=int, default=32)
    args = ap.parse_args()
    stats = accumulate(Path(args.probes_root), args.model, args.pair[0], args.pair[1],
                       args.block_length)
    plot(stats, Path(args.probes_root) / args.model / "plots", args.model)


if __name__ == "__main__":
    main()
