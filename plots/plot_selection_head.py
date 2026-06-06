"""Step 1 — the §2 cheap precursor: can a light readout *select* the answer?

The decisive-but-cheap test from T3D_RERANKING_STRATEGY.md §2 ("Cheap precursor —
do first"). The rank probe proved the converged token lives in think's iter-0
top-K (top-10 for ~75% of the hard tail, top-100 for ~98%). A full-vocab CE talk
nonetheless realises only ~21% because it collapses to copying think's argmax.
The question this script answers, WITHOUT training the real talk: if we restrict
a learned readout to think's own top-K candidates and ask it only to *select*,
does accuracy move off the 21% argmax-copy floor toward the 75% / 98% ceiling?

The readout is a tuned-lens-style affine probe (Belrose et al.) over the frozen
iter-0 anchor: features z = rmsnorm(h0) (the logit-lens input), learn A,b, score
candidate v as (zA+b)·W_head[v]. With A=I,b=0 this *is* the logit lens, so its
argmax reproduces think's top-1 (the 21% baseline); training A,b lets it re-rank
within the K. Restricted-softmax CE toward the converged token. Because the score
is linear in (A,b), the objective is convex multinomial logistic regression — it
fits deterministically with plain gradient descent, no model/GPU.

Fully offline: needs only the captured iter-0 hidden + the exported lm_head.
Prompt-disjoint train/test split (a fresh readout must generalise across prompts,
not memorise positions). Reported per readiness bucket and per candidate budget K:

    argmax-copy  (think top-1, the floor)   <   selection-head   ≤   ceiling (c∈topK)

If selection-head ≈ ceiling ⇒ extraction confirmed, green-light the reranker run.
If it sticks near argmax-copy ⇒ info is present but not shallowly extractable ⇒
the §3 partial-depth-refresh fallback.

Usage
-----
    python -m probe_runner.plots.plot_selection_head --model llada2 \
        --probes_root probes_out --K 10 100
    python -m probe_runner.plots.plot_selection_head --selftest      # numpy cores
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc

BUCKETS = ("easy(r=1)", "medium(r<=10)", "hard(r>10)")


def readiness_bucket(rank0: np.ndarray) -> np.ndarray:
    rank0 = np.asarray(rank0)
    out = np.full(rank0.shape, 2, dtype=np.int8)
    out[rank0 <= 10] = 1
    out[rank0 <= 1] = 0
    return out


# ---- tuned-lens selection head (pure numpy, convex, unit-testable) ----------

def _softmax(s: np.ndarray) -> np.ndarray:
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-30)


def head_scores(z: np.ndarray, cand_emb: np.ndarray,
                A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Candidate scores s[n,k] = (z_n @ A + b) · cand_emb[n,k].

    z [N,Dz] position features (rmsnorm'd anchor hidden, optionally concatenated
    with a revealed-neighbor feature); cand_emb [N,K,D] unembedding rows of each
    position's K candidates; A [Dz,D], b [D]. Linear in (A,b)."""
    u = z @ A + b                                          # [N, D]
    return np.einsum("nd,nkd->nk", u, cand_emb)            # [N, K]


def _init_affine(Dz: int, D: int) -> np.ndarray:
    """A [Dz,D] with the leading DxD (anchor) block = identity, rest zero, so the
    head starts at the logit lens over the anchor and a no-op on extra features."""
    A = np.zeros((Dz, D))
    m = min(Dz, D)
    A[:m, :m] = np.eye(m)
    return A


def fit_selection_head(z: np.ndarray, cand_emb: np.ndarray, target_idx: np.ndarray,
                       *, epochs: int = 300, lr: float = 0.05, l2: float = 1e-4,
                       batch: int = 4096, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fit A,b minimising restricted-softmax CE toward target_idx over the K
    candidates. Positions with target_idx < 0 (converged token not in top-K) are
    dropped from the loss. Adam, minibatched (cand_emb can be large at K=100).

    Feature dim Dz may exceed the candidate-embedding dim D (neighbor-conditioned
    head). Initialised at the logit lens on the anchor block (_init_affine) so
    training only *re-ranks*.
    """
    z = np.asarray(z, np.float64)
    cand_emb = np.asarray(cand_emb, np.float64)
    target_idx = np.asarray(target_idx)
    keep = target_idx >= 0
    z, cand_emb, t = z[keep], cand_emb[keep], target_idx[keep]
    N, Dz = z.shape
    D = cand_emb.shape[2]
    K = cand_emb.shape[1]
    A = _init_affine(Dz, D)
    b = np.zeros(D)
    mA = np.zeros_like(A); vA = np.zeros_like(A)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed)
    if N == 0:
        return A, b
    step = 0
    for _ in range(epochs):
        order = rng.permutation(N)
        for s0 in range(0, N, batch):
            idx = order[s0:s0 + batch]
            zb, eb, tb = z[idx], cand_emb[idx], t[idx]
            m = zb.shape[0]
            u = zb @ A + b                                 # [m,D]
            scores = np.einsum("nd,nkd->nk", u, eb)        # [m,K]
            p = _softmax(scores)
            onehot = np.zeros((m, K)); onehot[np.arange(m), tb] = 1.0
            dscore = (p - onehot) / m                      # [m,K]
            du = np.einsum("nk,nkd->nd", dscore, eb)       # [m,D]
            gA = zb.T @ du + l2 * A                        # [D,D]
            gb = du.sum(0) + l2 * b                        # [D]
            step += 1
            for prm, g, mm, vv in ((A, gA, mA, vA), (b, gb, mb, vb)):
                mm *= b1; mm += (1 - b1) * g
                vv *= b2; vv += (1 - b2) * (g * g)
                mhat = mm / (1 - b1 ** step)
                vhat = vv / (1 - b2 ** step)
                prm -= lr * mhat / (np.sqrt(vhat) + eps)
    return A, b


def selection_accuracy(scores: np.ndarray, target_idx: np.ndarray) -> float:
    """Top-1 selection accuracy over positions whose converged token is in the
    candidate set (target_idx >= 0)."""
    target_idx = np.asarray(target_idx)
    keep = target_idx >= 0
    if not keep.any():
        return float("nan")
    pred = scores[keep].argmax(axis=-1)
    return float((pred == target_idx[keep]).mean())


def baseline_summary(target_idx: np.ndarray) -> dict:
    """argmax-copy floor = P(converged == think top-1) (candidate slot 0);
    ceiling = P(converged ∈ top-K). Over ALL positions (incl. out-of-set)."""
    t = np.asarray(target_idx)
    n = max(len(t), 1)
    return {
        "n": int(len(t)),
        "argmax_copy": float((t == 0).sum() / n),          # slot 0 = think's top-1
        "ceiling": float((t >= 0).sum() / n),              # converged in top-K at all
    }


# ---- offline data driver (head only, no GPU/model/tokenizer) ----------------

def _topk_ids(logits_row: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k logits, descending. [k]."""
    k = min(k, logits_row.shape[0])
    part = np.argpartition(-logits_row, k - 1)[:k]
    return part[np.argsort(-logits_row[part])]


def compute_selection_inputs(probes_root, model: str, head: tc.RankHead, K: int,
                             block_length: int = 32, n_samples: int | None = None) -> dict:
    """Per valid position gather (z, neigh, cand_emb, target_idx, rank0, prompt_id).

    z      = rmsnorm(iter-0 hidden) — the static anchor feature (Step 1 / R3);
    neigh  = leave-one-out mean of the block's *converged* (revealed) neighbor
             hiddens, rmsnorm'd — the revealed-neighbor context (R4). Excludes the
             position's own hidden, so it cannot leak the target. This is the
             fully-revealed upper bound on neighbor signal: at decode the talk sees
             only the *already-committed* neighbors, so neigh over-states what is
             available mid-block.
    cand_emb = W_head rows of think's iter-0 top-K; target_idx = slot of the
    converged token in that top-K (-1 if absent); prompt_id = sample stem (split).
    """
    z_all, g_all, e_all, t_all, r_all, pid_all = [], [], [], [], [], []
    W = head.W                                             # [vocab, D]
    done = 0
    for path in tc.iter_sample_paths(probes_root, model):
        sample = tc.load_sample(path)
        attrs = sample["attrs"]
        pid = path.stem
        for b, blk in sorted(sample["blocks"].items()):
            if "converged_tokens" not in blk and "committed_tokens_per_pass" not in blk:
                continue
            hpp = np.asarray(blk["h_per_pass"])            # [P, L+1, B, D]
            h0 = hpp[0, -1]                                # [B, D] iter-0 last layer
            h_conv = hpp[-1, -1]                           # [B, D] converged last layer
            conv = np.asarray(tc.converged_tokens(blk))    # [B]
            valid = tc.valid_position_mask(b, h0.shape[0], attrs, block_length)
            valid &= (conv != tc.MASK_ID)
            if not valid.any():
                continue
            h0v, convv = h0[valid], conv[valid]
            logits = head.logits(h0v)                      # [Bv, vocab]
            z = head._rmsnorm(h0v)                         # [Bv, D]
            # revealed-neighbor feature: LOO-mean of converged neighbor vectors.
            zc = head._rmsnorm(h_conv[valid])              # [Bv, D]
            Bv = zc.shape[0]
            neigh = ((zc.sum(0, keepdims=True) - zc) / (Bv - 1)) if Bv > 1 else np.zeros_like(zc)
            r0 = (logits > logits[np.arange(len(convv)), convv][:, None]).sum(-1) + 1
            for i in range(len(convv)):
                ids = _topk_ids(logits[i], K)              # [K]
                hit = np.where(ids == convv[i])[0]
                t_all.append(int(hit[0]) if hit.size else -1)
                e_all.append(W[ids])                       # [K, D]
                z_all.append(z[i])
                g_all.append(neigh[i])
                r_all.append(int(r0[i]))
                pid_all.append(pid)
        done += 1
        if n_samples and done >= n_samples:
            break
    D = head.d_model
    return {
        "z": np.asarray(z_all, np.float64) if z_all else np.zeros((0, D)),
        "neigh": np.asarray(g_all, np.float64) if g_all else np.zeros((0, D)),
        "cand_emb": np.asarray(e_all, np.float64) if e_all else np.zeros((0, K, D)),
        "target_idx": np.asarray(t_all, np.int64),
        "rank0": np.asarray(r_all, np.int64),
        "prompt_id": np.asarray(pid_all),
    }


def _prompt_split(prompt_id: np.ndarray, frac_train: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train, test) masks splitting on UNIQUE prompt ids (disjoint)."""
    uniq = sorted(set(prompt_id.tolist()))
    cut = max(1, int(round(len(uniq) * frac_train)))
    train_ids = set(uniq[:cut])
    tr = np.array([p in train_ids for p in prompt_id])
    return tr, ~tr


def _feature(data: dict, use_neighbors: bool) -> np.ndarray:
    """Position feature: anchor only, or [anchor ; revealed-neighbor]."""
    if not use_neighbors:
        return data["z"]
    return np.concatenate([data["z"], data["neigh"]], axis=1)


def evaluate(data: dict, *, tail_only: bool = False, use_neighbors: bool = False,
             epochs: int = 300, seed: int = 0) -> dict:
    """Fit on train prompts (optionally tail-only / neighbor-conditioned), report
    acc on the held-out test prompts, overall + per bucket + tail aggregate.

    tail_only (R1): exclude easy (r0=1) positions from *training* — the committed
    half is carried through at think's argmax, so the readout should not spend a
    single global affine preserving it. use_neighbors (R4): concatenate the
    revealed-neighbor feature.
    """
    feat = _feature(data, use_neighbors)
    tr, te = _prompt_split(data["prompt_id"])
    train = tr.copy()
    if tail_only:
        train &= (data["rank0"] > 1)
    A, b = fit_selection_head(feat[train], data["cand_emb"][train], data["target_idx"][train],
                              epochs=epochs, seed=seed)
    scores = head_scores(feat[te], data["cand_emb"][te], A, b)
    t_te = data["target_idx"][te]
    bucket = readiness_bucket(data["rank0"][te])
    res = {"overall": {**baseline_summary(t_te), "selection": selection_accuracy(scores, t_te)},
           "by_bucket": {}, "tail": {}, "tail_only": tail_only, "use_neighbors": use_neighbors,
           "n_train": int(train.sum()), "n_test": int(te.sum())}
    for bk in range(3):
        sel = bucket == bk
        if not sel.any():
            continue
        res["by_bucket"][BUCKETS[bk]] = {
            **baseline_summary(t_te[sel]),
            "selection": selection_accuracy(scores[sel], t_te[sel]),
        }
    # left-for-talk aggregate (the number that matters): r0 > 1.
    tail = bucket > 0
    res["tail"] = {**baseline_summary(t_te[tail]),
                   "selection": selection_accuracy(scores[tail], t_te[tail])}
    return res


# variants compared in main(): (label, tail_only, use_neighbors)
VARIANTS = (
    ("static-all", False, False),     # the original Step-1 probe (R1-violating)
    ("tail-only", True, False),       # fix 1(a)
    ("tail+neigh", True, True),       # fix 1(a)+1(b)
)


def run_variants(data: dict, *, epochs: int = 300, seed: int = 0) -> dict:
    return {label: evaluate(data, tail_only=to, use_neighbors=un, epochs=epochs, seed=seed)
            for label, to, un in VARIANTS}


# ---- report + plot ----------------------------------------------------------

def format_report(per_k: dict) -> str:
    """per_k[K][variant_label] = evaluate() result."""
    lines = ["Step-1 selection-head probe (tuned-lens over iter-0 anchor, top-K restricted)",
             "  variants: static-all (R1-violating) | tail-only (1a) | tail+neigh (1a+1b)"]
    for k, variants in per_k.items():
        any_res = next(iter(variants.values()))
        lines.append(f"  K={k}: test positions={any_res['n_test']}  "
                     f"tail ceiling(c∈topK)={any_res['tail']['ceiling']:.1%}")
        for label, res in variants.items():
            t = res["tail"]; med = res["by_bucket"].get(BUCKETS[1], {}); hard = res["by_bucket"].get(BUCKETS[2], {})
            lines.append(
                f"    {label:<11s} tail-selection={t['selection']:.1%}  "
                f"(medium={med.get('selection', float('nan')):.1%}, "
                f"hard={hard.get('selection', float('nan')):.1%})  "
                f"[train n={res['n_train']}]")
    return "\n".join(lines)


def plot(per_k: dict, out_dir: Path, model: str) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = list(per_k.keys())
    labels = [lbl for lbl, _, _ in VARIANTS]
    colors = {"static-all": "tab:gray", "tail-only": "tab:orange", "tail+neigh": "tab:green"}
    fig, axes = plt.subplots(1, len(ks), figsize=(7.5 * len(ks), 4.8), squeeze=False)
    for j, k in enumerate(ks):
        ax = axes[0][j]
        variants = per_k[k]
        groups = ["medium(r≤10)", "hard(r>10)", "tail(r>1)"]
        xs = np.arange(len(groups))
        w = 0.8 / (len(labels) + 1)
        for i, lbl in enumerate(labels):
            res = variants[lbl]
            vals = [res["by_bucket"].get(BUCKETS[1], {}).get("selection", np.nan),
                    res["by_bucket"].get(BUCKETS[2], {}).get("selection", np.nan),
                    res["tail"]["selection"]]
            ax.bar(xs + (i - len(labels) / 2) * w, vals, width=w, color=colors[lbl], label=lbl)
        # ceiling per group
        any_res = next(iter(variants.values()))
        ceil = [any_res["by_bucket"].get(BUCKETS[1], {}).get("ceiling", np.nan),
                any_res["by_bucket"].get(BUCKETS[2], {}).get("ceiling", np.nan),
                any_res["tail"]["ceiling"]]
        ax.bar(xs + (len(labels) - len(labels) / 2) * w, ceil, width=w,
               color="tab:purple", alpha=0.45, label="ceiling c∈topK")
        ax.set_xticks(xs); ax.set_xticklabels(groups, fontsize=8)
        ax.set_ylim(0, 1); ax.set_ylabel("selection accuracy")
        ax.axhline(0.215, ls="--", lw=0.8, color="black")
        ax.text(0.02, 0.225, "static-anchor floor ≈21% (R4)", fontsize=7, transform=ax.get_yaxis_transform())
        ax.set_title(f"K={k}: does tail-only + neighbors lift the tail?")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"{model}: Step-1 selection-head extractability (variant comparison)")
    fig.tight_layout()
    p = out_dir / "selection_head.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")


# ---- selftest ---------------------------------------------------------------

def _selftest() -> None:
    rng = np.random.default_rng(0)
    N, D, K, V, ntr = 2000, 12, 8, 200, 1500
    W = rng.standard_normal((V, D))                        # unembedding rows
    z = rng.standard_normal((N, D))
    # ground-truth readout: a random affine the head must recover.
    Atrue = rng.standard_normal((D, D)) * 0.5 + np.eye(D)
    u_true = z @ Atrue
    cand_ids = np.stack([rng.choice(V, size=K, replace=False) for _ in range(N)])
    cand_emb = W[cand_ids]                                 # [N,K,D]
    true_scores = np.einsum("nd,nkd->nk", u_true, cand_emb)
    target_idx = true_scores.argmax(1)                     # the "converged" slot
    # the fitted head should recover the true ranking on held-out positions;
    # argmax-copy (logit-lens slot 0) is random on this synthetic target (~1/K).
    A, b = fit_selection_head(z[:ntr], cand_emb[:ntr], target_idx[:ntr],
                              epochs=600, lr=0.1, l2=1e-5)
    sc = head_scores(z[ntr:], cand_emb[ntr:], A, b)
    acc = selection_accuracy(sc, target_idx[ntr:])
    assert acc > 0.9, f"selection head should recover the affine readout, got {acc:.2f}"
    # untrained logit-lens (A=I,b=0) must do far worse on this synthetic target
    base = selection_accuracy(head_scores(z[ntr:], cand_emb[ntr:], np.eye(D), np.zeros(D)),
                              target_idx[ntr:])
    assert acc > base + 0.3, f"trained ({acc:.2f}) should beat logit-lens ({base:.2f})"
    # out-of-set targets are excluded from loss + accuracy
    ti = target_idx.copy(); ti[:50] = -1
    assert np.isnan(selection_accuracy(np.zeros((50, K)), ti[:50] * 0 - 1))
    b_ = baseline_summary(ti)
    assert abs(b_["ceiling"] - (N - 50) / N) < 1e-9
    # disjoint split is prompt-exclusive
    pid = np.array(["a"] * 300 + ["b"] * 300)
    tr, te = _prompt_split(pid)
    assert not (set(pid[tr]) & set(pid[te]))

    # neighbor-conditioned (non-square) head: build a target that depends ONLY on
    # a neighbor feature; the anchor block is pure noise. The [anchor;neigh] head
    # (Dz=2D) must recover it, while an anchor-only head (Dz=D) cannot.
    rng2 = np.random.default_rng(1)
    Wn = rng2.standard_normal((V, D))
    anchor = rng2.standard_normal((N, D))                  # uninformative
    g = rng2.standard_normal((N, D))                       # informative neighbor feat
    Ag = rng2.standard_normal((D, D)) * 0.5 + np.eye(D)
    cids = np.stack([rng2.choice(V, size=K, replace=False) for _ in range(N)])
    ce2 = Wn[cids]
    tgt2 = np.einsum("nd,nkd->nk", g @ Ag, ce2).argmax(1)
    feat2 = np.concatenate([anchor, g], axis=1)            # [N, 2D]
    A2, b2 = fit_selection_head(feat2[:ntr], ce2[:ntr], tgt2[:ntr], epochs=600, lr=0.1, l2=1e-5)
    assert A2.shape == (2 * D, D)
    acc_n = selection_accuracy(head_scores(feat2[ntr:], ce2[ntr:], A2, b2), tgt2[ntr:])
    A_anchor, b_anchor = fit_selection_head(anchor[:ntr], ce2[:ntr], tgt2[:ntr], epochs=600, lr=0.1, l2=1e-5)
    acc_a = selection_accuracy(head_scores(anchor[ntr:], ce2[ntr:], A_anchor, b_anchor), tgt2[ntr:])
    assert acc_n > 0.8, f"neighbor head should recover neighbor-driven target, got {acc_n:.2f}"
    assert acc_n > acc_a + 0.3, f"neighbor ({acc_n:.2f}) must beat anchor-only ({acc_a:.2f})"
    print("plot_selection_head selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llada2")
    ap.add_argument("--probes_root", default="probes_out")
    ap.add_argument("--K", type=int, nargs="+", default=[10, 100])
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    head_path = tc.find_head_export(args.probes_root, args.model)
    if head_path is None:
        raise SystemExit(f"no lm_head export under {args.probes_root}/{args.model}")
    head = tc.RankHead.load(head_path)
    per_k = {}
    for k in args.K:
        data = compute_selection_inputs(args.probes_root, args.model, head, k,
                                        block_length=args.block_length, n_samples=args.n_samples)
        per_k[k] = run_variants(data, epochs=args.epochs, seed=args.seed)
    print(format_report(per_k))
    plot(per_k, Path(args.probes_root) / args.model / "plots", args.model)


if __name__ == "__main__":
    main()
