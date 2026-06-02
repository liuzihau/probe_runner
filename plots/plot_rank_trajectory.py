"""A7 — converged-token rank trajectory ("does the model already know its future").

For each masked position p and decode pass k, the rank of the position's FINAL
(converged) token in pass-k's logits (rank 1 = already the argmax). Logits come
from the captured last-layer hidden via the exported final-norm + lm_head
(_trajectory_common.RankHead) — no 16B reload.

Three readouts (design doc §2 A7):

  1. rank heatmap  [pass × block-position]  (log10 rank), pooled over samples.
     The descent of the future token toward rank 1, and which positions lag.
  2. top-K hit-rate vs pass: fraction of positions with the converged token in
     top-1/5/10/100 at pass k. top-1 @ pass0 = the old binary flip_recovery;
     split committed-vs-masked (the per-pass state).
  3. pass-0 readiness: rank-at-pass0 distribution + the ORACLE-reranker ceiling
     (top-K hit @ pass0) — the optimistic upper bound that brackets the trained
     probe's ~10% from above. If even oracle top-100 @ pass0 is low, the
     static-anchor premise is dead regardless of probe quality.

Usage:
    python -m probe_runner.plots.plot_rank_trajectory --model llada2
    python -m probe_runner.plots.plot_rank_trajectory --model llada2 --probes_root probes_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc

TOPK = (1, 5, 10, 100)


# ---- per-sample core (pure: head + arrays in, ranks out) --------------------

def sample_rank_data(sample: dict, head: tc.RankHead, block_length: int,
                     mask_id: int = tc.MASK_ID) -> dict[int, dict]:
    """Per block: rank[pass, pos], valid[pos] (EOS filter), committed[pass, pos]
    (token_state), converged[pos]."""
    out: dict[int, dict] = {}
    for b, blk in sample["blocks"].items():
        if "converged_tokens" not in blk and "committed_tokens_per_pass" not in blk:
            continue
        h = blk["h_per_pass"]                          # [P, L+1, B, D]
        P, _, B, _ = h.shape
        conv = tc.converged_tokens(blk, mask_id)        # [B]
        valid = tc.valid_position_mask(b, B, sample["attrs"], block_length)
        ranks = np.full((P, B), -1, dtype=np.int64)
        for k in range(P):
            ranks[k] = head.rank_of(h[k, -1], conv)     # last-layer hidden
        ranks[:, ~valid] = -1                           # drop post-EOS positions
        committed = (blk["token_state_per_pass"].astype(np.int8) == 1) \
            if "token_state_per_pass" in blk else np.zeros((P, B), bool)
        out[b] = {"ranks": ranks, "valid": valid, "committed": committed,
                  "converged": conv, "n_passes": P}
    return out


def accumulate(probes_root: Path, model: str, head: tc.RankHead,
               block_length: int, max_pass: int | None = None) -> dict:
    paths = tc.iter_sample_paths(probes_root, model)
    if not paths:
        raise RuntimeError(f"No samples under {Path(probes_root)/model}")

    # heatmap accumulators [Pmax, B]
    rank_sum = None
    rank_cnt = None
    # top-K hit per pass, overall + by committed/masked
    hit = {k: {"all": None, "committed": None, "masked": None} for k in TOPK}
    cnt = {"all": None, "committed": None, "masked": None}
    pass0_ranks = []         # pooled rank@pass0 (valid)
    n_samples = 0
    Pmax = 0
    Bmax = 0

    def _ensure(arr, P, B):
        if arr is None:
            return np.zeros((P, B), dtype=np.float64)
        if arr.shape[0] < P or arr.shape[1] < B:
            new = np.zeros((max(arr.shape[0], P), max(arr.shape[1], B)), dtype=arr.dtype)
            new[:arr.shape[0], :arr.shape[1]] = arr
            return new
        return arr

    def _ensure1(arr, P):
        if arr is None:
            return np.zeros(P, dtype=np.float64)
        if arr.shape[0] < P:
            new = np.zeros(P, dtype=arr.dtype)
            new[:arr.shape[0]] = arr
            return new
        return arr

    for path in paths:
        sample = tc.load_sample(path)
        if not tc.has_committed(sample["blocks"]):
            continue
        data = sample_rank_data(sample, head, block_length)
        if not data:
            continue
        n_samples += 1
        for b, d in data.items():
            ranks, valid, committed = d["ranks"], d["valid"], d["committed"]
            P, B = ranks.shape
            if max_pass is not None:
                P = min(P, max_pass)
            Pmax, Bmax = max(Pmax, P), max(Bmax, B)
            rank_sum = _ensure(rank_sum, Pmax, Bmax)
            rank_cnt = _ensure(rank_cnt, Pmax, Bmax)
            for k in TOPK:
                for g in ("all", "committed", "masked"):
                    hit[k][g] = _ensure1(hit[k][g], Pmax)
            for g in ("all", "committed", "masked"):
                cnt[g] = _ensure1(cnt[g], Pmax)

            for p in range(P):
                vr = ranks[p]                         # [B]
                vmask = (vr > 0)                       # valid + has rank
                # heatmap
                rank_sum[p, :B][vmask] += np.log10(vr[vmask])
                rank_cnt[p, :B][vmask] += 1
                # groups for top-K
                comm = committed[p] & vmask
                msk = (~committed[p]) & vmask
                cnt["all"][p] += int(vmask.sum())
                cnt["committed"][p] += int(comm.sum())
                cnt["masked"][p] += int(msk.sum())
                for k in TOPK:
                    intopk = vmask & (vr <= k)
                    hit[k]["all"][p] += int(intopk.sum())
                    hit[k]["committed"][p] += int((intopk & committed[p]).sum())
                    hit[k]["masked"][p] += int((intopk & ~committed[p]).sum())
                if p == 0:
                    pass0_ranks.append(vr[vmask])

    pass0 = np.concatenate(pass0_ranks) if pass0_ranks else np.array([], dtype=np.int64)
    return {
        "rank_sum": rank_sum, "rank_cnt": rank_cnt,
        "hit": hit, "cnt": cnt, "pass0": pass0,
        "n_samples": n_samples, "Pmax": Pmax, "Bmax": Bmax,
        "vocab": head.vocab_size,
    }


# ---- plotting ---------------------------------------------------------------

def plot(stats: dict, out_dir: Path, model: str) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_sum, rank_cnt = stats["rank_sum"], stats["rank_cnt"]
    if rank_sum is None:
        raise RuntimeError("No rank data — did you run with --intra_block on a llada2 capture?")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_log_rank = np.where(rank_cnt > 0, rank_sum / np.maximum(rank_cnt, 1), np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))

    # (1) heatmap
    ax = axes[0]
    im = ax.imshow(mean_log_rank, aspect="auto", origin="lower",
                   interpolation="nearest", cmap="viridis_r")
    ax.set_title("converged-token rank (log10) by pass × position")
    ax.set_xlabel("position in block")
    ax.set_ylabel("decode pass")
    fig.colorbar(im, ax=ax, label="mean log10(rank)  (low = model already knows)")

    # (2) top-K hit-rate vs pass (overall)
    ax = axes[1]
    Pmax = stats["Pmax"]
    xs = np.arange(Pmax)
    for k in TOPK:
        h = np.asarray(stats["hit"][k]["all"], dtype=np.float64)
        c = np.asarray(stats["cnt"]["all"], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            rate = np.where(c > 0, h / np.maximum(c, 1), np.nan)
        ax.plot(xs, rate[:Pmax], marker="o", ms=3, label=f"top-{k}")
    ax.set_ylim(0, 1.02)
    ax.set_title("top-K hit-rate vs pass\n(top-1 @ pass0 = flip_recovery)")
    ax.set_xlabel("decode pass")
    ax.set_ylabel("fraction with converged token in top-K")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, title="K")

    # (3) pass-0 readiness + oracle ceiling
    ax = axes[2]
    pass0 = stats["pass0"]
    vocab = stats["vocab"]
    if pass0.size:
        logr = np.log10(np.maximum(pass0, 1))
        ax.hist(logr, bins=30, color="tab:blue", alpha=0.65, density=True)
        ax.axvline(np.log10(vocab / 2), ls="--", color="black", alpha=0.6,
                   label=f"chance (log10 {np.log10(vocab/2):.1f})")
        # oracle ceiling text
        lines = ["oracle top-K @ pass0 (ceiling):"]
        for k in TOPK:
            frac = float((pass0 <= k).mean())
            lines.append(f"  top-{k}: {frac:5.1%}")
        ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes,
                ha="right", va="top", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_title("pass-0 readiness (rank of converged token @ iter 0)")
    ax.set_xlabel("log10(rank) at pass 0")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"{model}: A7 converged-token rank trajectory  "
                 f"[n_samples={stats['n_samples']}, vocab={vocab}]")
    fig.tight_layout()
    path = out_dir / "rank_trajectory.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llada2")
    ap.add_argument("--probes_root", default="probes_out")
    ap.add_argument("--head_path", default=None,
                    help="lm_head.pt/.npz export. Default: <probes_root>/<model>/lm_head.*")
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--max_pass", type=int, default=None)
    args = ap.parse_args()

    head_path = args.head_path or tc.find_head_export(args.probes_root, args.model)
    if head_path is None:
        raise SystemExit(
            "No lm_head export found. Re-run run_probes (llada2) without "
            "--no_export_head, or pass --head_path.")
    head = tc.RankHead.load(head_path)
    stats = accumulate(Path(args.probes_root), args.model, head,
                       args.block_length, args.max_pass)
    plot(stats, Path(args.probes_root) / args.model / "plots", args.model)


if __name__ == "__main__":
    main()
