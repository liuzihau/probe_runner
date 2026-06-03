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
    # per-(pass x pos) committed fraction — generalizes the "decoded boundary" to
    # NON-contiguous decode. DMax decode_uniform commits a left-to-right prefix, so
    # #committed == boundary position; LLaDA-2.0 native threshold decode commits a
    # GLOBAL (scattered) set, so there is no L->R boundary and we draw a frontier.
    cstate_sum = None
    cstate_cnt = None
    contig_hit = 0.0          # passes whose committed set is a left-aligned prefix
    contig_tot = 0.0
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

    # Per-sample curves (for the across-sample confidence band) — ragged [Ps],
    # padded to Pmax at the end.
    sample_rates = {k: [] for k in TOPK}
    sample_commit = []   # avg committed-through-pass (cumulative reveals), per sample
    # pass-0 rank split by think's iter-0 decision: committed by think's first
    # forward (revealed_per_pass[0]) vs LEFT MASKED for the talk (the hard tail).
    pass0_committed, pass0_masked = [], []

    for path in paths:
        sample = tc.load_sample(path)
        if not tc.has_committed(sample["blocks"]):
            continue
        data = sample_rank_data(sample, head, block_length)
        if not data:
            continue
        n_samples += 1
        Ps = max(d["n_passes"] for d in data.values())
        if max_pass is not None:
            Ps = min(Ps, max_pass)
        s_hit = {k: np.zeros(Ps) for k in TOPK}
        s_cnt = np.zeros(Ps)
        s_commit = np.zeros(Ps)
        s_commit_blk = np.zeros(Ps)
        for b, d in data.items():
            ranks, valid, committed = d["ranks"], d["valid"], d["committed"]
            P, B = ranks.shape
            P = min(P, Ps)
            Pmax, Bmax = max(Pmax, P), max(Bmax, B)
            rank_sum = _ensure(rank_sum, Pmax, Bmax)
            rank_cnt = _ensure(rank_cnt, Pmax, Bmax)
            cstate_sum = _ensure(cstate_sum, Pmax, Bmax)
            cstate_cnt = _ensure(cstate_cnt, Pmax, Bmax)
            for k in TOPK:
                for g in ("all", "committed", "masked"):
                    hit[k][g] = _ensure1(hit[k][g], Pmax)
            for g in ("all", "committed", "masked"):
                cnt[g] = _ensure1(cnt[g], Pmax)
            # cumulative reveals → #committed-through-pass (left-to-right contiguous
            # decode ⇒ this is the committed/masked boundary position).
            rev = sample["blocks"][b].get("revealed_per_pass")
            cum = np.cumsum(np.asarray(rev).astype(np.int64), axis=0) if rev is not None else None

            for p in range(P):
                vr = ranks[p]                         # [B]
                vmask = (vr > 0)                       # valid + has rank
                rank_sum[p, :B][vmask] += np.log10(vr[vmask])
                rank_cnt[p, :B][vmask] += 1
                cnt["all"][p] += int(vmask.sum())
                cnt["committed"][p] += int((committed[p] & vmask).sum())
                cnt["masked"][p] += int(((~committed[p]) & vmask).sum())
                s_cnt[p] += int(vmask.sum())
                # committed fraction by (pass, pos) over all EOS-valid positions,
                # plus a contiguity check (is the committed set a left-aligned
                # prefix?) to decide the heatmap overlay (staircase vs frontier).
                cstate_sum[p, :B][valid] += committed[p][valid].astype(np.float64)
                cstate_cnt[p, :B][valid] += 1.0
                cm = committed[p] & valid
                mm = (~committed[p]) & valid
                if cm.any() and mm.any():
                    contig_tot += 1.0
                    if int(np.where(cm)[0].max()) < int(np.where(mm)[0].min()):
                        contig_hit += 1.0
                for k in TOPK:
                    intopk = vmask & (vr <= k)
                    hit[k]["all"][p] += int(intopk.sum())
                    hit[k]["committed"][p] += int((intopk & committed[p]).sum())
                    hit[k]["masked"][p] += int((intopk & ~committed[p]).sum())
                    s_hit[k][p] += int(intopk.sum())
                if cum is not None:
                    s_commit[p] += int(cum[p].sum())
                    s_commit_blk[p] += 1
                if p == 0:
                    pr = vr[vmask]
                    pass0_ranks.append(pr)
                    if rev is not None and np.asarray(rev).shape[0] > 0:
                        rev0 = np.asarray(rev[0]).astype(bool)[vmask]
                        pass0_committed.append(pr[rev0])
                        pass0_masked.append(pr[~rev0])
        with np.errstate(invalid="ignore", divide="ignore"):
            for k in TOPK:
                sample_rates[k].append(np.where(s_cnt > 0, s_hit[k] / np.maximum(s_cnt, 1), np.nan))
            sample_commit.append(np.where(s_commit_blk > 0,
                                          s_commit / np.maximum(s_commit_blk, 1), np.nan))

    def _pad(a, P):
        out = np.full(P, np.nan)
        out[:min(len(a), P)] = a[:P]
        return out

    rates = {k: (np.vstack([_pad(a, Pmax) for a in sample_rates[k]])
                 if sample_rates[k] else np.zeros((0, Pmax))) for k in TOPK}
    commit = (np.vstack([_pad(a, Pmax) for a in sample_commit])
              if sample_commit else None)

    def _cat(lst):
        return np.concatenate(lst) if lst else np.array([], dtype=np.int64)

    with np.errstate(invalid="ignore", divide="ignore"):
        commit_frac = (np.where(cstate_cnt > 0, cstate_sum / np.maximum(cstate_cnt, 1), np.nan)
                       if cstate_sum is not None else None)
    # >=90% of mixed passes left-aligned  => treat as contiguous (DMax). Otherwise
    # the decode is scattered (LLaDA-2.0 native threshold) and the L->R staircase
    # is meaningless, so the heatmap falls back to a committed-fraction frontier.
    is_contiguous = (contig_tot == 0) or (contig_hit / max(contig_tot, 1.0) >= 0.9)

    return {
        "rank_sum": rank_sum, "rank_cnt": rank_cnt,
        "hit": hit, "cnt": cnt, "rates": rates, "commit": commit,
        "commit_frac": commit_frac, "is_contiguous": bool(is_contiguous),
        "pass0": _cat(pass0_ranks),
        "pass0_committed": _cat(pass0_committed),
        "pass0_masked": _cat(pass0_masked),
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

    # (1) heatmap + avg-committed staircase
    ax = axes[0]
    im = ax.imshow(mean_log_rank, aspect="auto", origin="lower",
                   interpolation="nearest", cmap="viridis_r")
    ax.set_title("converged-token rank (log10) by pass × position")
    ax.set_xlabel("position in block")
    ax.set_ylabel("decode pass")
    fig.colorbar(im, ax=ax, label="mean log10(rank)  (low = model already knows)")
    # Overlay the committed/masked frontier. For a LEFT-TO-RIGHT contiguous decode
    # (DMax decode_uniform) #committed is exactly the boundary position, so we draw
    # the staircase (left = decoded). For a NON-contiguous decode (LLaDA-2.0 native
    # threshold: scattered global commits) there is no L->R boundary, so we instead
    # contour the per-(pass,pos) 50%-committed frontier.
    commit = stats.get("commit")
    is_contig = stats.get("is_contiguous", True)
    cfrac = stats.get("commit_frac")
    if is_contig and commit is not None and commit.size:
        with np.errstate(invalid="ignore"):
            avg_commit = np.nanmean(commit, axis=0)        # [Pmax]
        boundary = avg_commit - 0.5                          # right edge of last committed
        xs_c, ys_c = [], []
        for k in range(len(boundary)):
            if not np.isfinite(boundary[k]):
                continue
            xs_c += [boundary[k], boundary[k]]
            ys_c += [k - 0.5, k + 0.5]
        if xs_c:
            ax.plot(xs_c, ys_c, color="red", lw=1.6,
                    label="avg #committed (left = decoded)")
            ax.legend(loc="upper left", fontsize=7, framealpha=0.75)
    elif cfrac is not None and np.isfinite(cfrac).any():
        P_, B_ = cfrac.shape
        ax.contour(np.arange(B_), np.arange(P_), np.nan_to_num(cfrac, nan=0.0),
                   levels=[0.5], colors="red", linewidths=1.4)
        ax.plot([], [], color="red", lw=1.4,
                label="50%-committed frontier (non-contiguous decode)")
        ax.legend(loc="upper left", fontsize=7, framealpha=0.75)

    # (2) top-K hit-rate vs pass, with 95% CI shading across samples
    ax = axes[1]
    Pmax = stats["Pmax"]
    xs = np.arange(Pmax)
    rates = stats["rates"]
    colors = {1: "tab:blue", 5: "tab:orange", 10: "tab:green", 100: "tab:red"}
    band_lows = []
    for k in TOPK:
        R = rates.get(k)
        if R is None or R.size == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nanmean(R, axis=0)
            neff = np.sum(~np.isnan(R), axis=0)
            sem = np.nanstd(R, axis=0, ddof=1) / np.sqrt(np.maximum(neff, 1))
            sem = np.where(neff >= 2, sem, np.nan)
        lo, hi = mean - 1.96 * sem, mean + 1.96 * sem
        c = colors.get(k)
        ax.plot(xs, mean[:Pmax], marker="o", ms=3, color=c, label=f"top-{k}")
        ax.fill_between(xs, lo[:Pmax], hi[:Pmax], color=c, alpha=0.18, linewidth=0)
        if np.isfinite(lo).any():
            band_lows.append(np.nanmin(lo))
    # Crop the y-axis: 0–0.4 is empty, so start just below the lowest band.
    ymin = max(0.0, (min(band_lows) if band_lows else 0.0) - 0.02)
    ax.set_xlim(-0.5, max(Pmax - 0.5, 0.5))
    ax.set_ylim(ymin, 1.01)
    ax.set_title("top-K hit-rate vs pass  (95% CI across samples)\n(top-1 @ pass0 = flip_recovery)")
    ax.set_xlabel("decode pass")
    ax.set_ylabel("fraction with converged token in top-K")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, title="K", loc="lower right")

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


def plot_by_domain(stats: dict, out_dir: Path, model: str) -> None:
    """Rank-by-domain: split the readiness by think's iter-0 decision so we see
    the ceiling on the HARD TAIL (the positions think leaves masked for the talk)
    — the apples-to-apples comparison for the single-step talk's ~0.20.
    """
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    Pmax = stats["Pmax"]
    xs = np.arange(Pmax)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A: top-K vs pass for committed vs still-masked (pooled).
    ax = axes[0]

    def rate(group, K):
        h = np.asarray(stats["hit"][K][group], dtype=np.float64)
        c = np.asarray(stats["cnt"][group], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            return np.where(c > 0, h / np.maximum(c, 1), np.nan)

    for group, col in (("committed", "tab:green"), ("masked", "tab:red")):
        for K, ls, a in ((1, "-", 0.95), (10, "--", 0.55)):
            ax.plot(xs, rate(group, K)[:Pmax], ls=ls, marker="o", ms=3, color=col,
                    alpha=a, label=f"{group} top-{K}")
    ax.set_ylim(0, 1.02)
    ax.set_title("top-K hit-rate vs pass, by state\n"
                 "(masked = undecided at the START of each pass — PRE-commit)")
    ax.set_xlabel("decode pass")
    ax.set_ylabel("fraction with converged token in top-K")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")

    # Recovery readout. The masked group at the start of pass 1 is exactly the
    # right panel's left-for-talk set (the hard tail think left after iter 0). So
    # pass0_masked top-1 (panel B, iter-0 forward) vs masked@pass1 top-1 (iter-1
    # forward) is a same-set comparison: how much one extra THINK pass recovers.
    pm = stats["pass0_masked"]
    m1 = rate("masked", 1)
    if pm.size and m1.size > 1 and np.isfinite(m1[1]):
        a0, a1 = float((pm <= 1).mean()), float(m1[1])
        ax.text(0.97, 0.05,
                "hard tail (left-for-talk) top-1:\n"
                f"  iter0 = {a0:.0%}  ->  iter1 = {a1:.0%}\n"
                "  (recovered via a 2nd THINK pass)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                family="monospace", bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    # Panel B: pass-0 hard-tail ceiling (the headline).
    ax = axes[1]
    pm, pc = stats["pass0_masked"], stats["pass0_committed"]
    vocab = stats["vocab"]
    ntot = pm.size + pc.size
    for arr, lab, col in ((pc, "think-committed @0 (POST-commit)", "tab:green"),
                          (pm, "left-for-talk (undecided AFTER iter 0)", "tab:red")):
        if arr.size:
            ax.hist(np.log10(np.maximum(arr, 1)), bins=30, density=True, alpha=0.5,
                    color=col, label=f"{lab}  (n={arr.size})")
    ax.axvline(np.log10(vocab / 2), ls="--", color="black", alpha=0.5,
               label=f"chance (log10 {np.log10(vocab/2):.1f})")
    if pm.size:
        lines = [f"LEFT-FOR-TALK oracle ceiling",
                 f"(n={pm.size}, {pm.size / max(ntot, 1):.0%} of positions):"]
        for K in TOPK:
            lines.append(f"  top-{K}: {float((pm <= K).mean()):5.1%}")
        ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, family="monospace",
                bbox=dict(boxstyle="round", fc="white", alpha=0.88))
    ax.set_title("pass-0 readiness split: does the ITER-0 ANCHOR\n"
                 "already encode the hard tail's converged token?")
    ax.set_xlabel("log10(rank) of converged token @ pass 0")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"{model}: A7 rank by domain  [n_samples={stats['n_samples']}]\n"
        "LEFT: masked = undecided at START of pass (PRE-commit)   |   "
        "RIGHT: split by think's iter-0 commit (POST-commit hard tail)")
    fig.tight_layout()
    p = out_dir / "rank_by_domain.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")


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
    out_dir = Path(args.probes_root) / args.model / "plots"
    plot(stats, out_dir, args.model)
    plot_by_domain(stats, out_dir, args.model)


if __name__ == "__main__":
    main()
