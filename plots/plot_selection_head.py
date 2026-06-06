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

    z [N,D] anchor features (rmsnorm'd hidden); cand_emb [N,K,D] unembedding rows
    of each position's K candidates; A [D,D], b [D]. Linear in (A,b)."""
    u = z @ A + b                                          # [N, D]
    return np.einsum("nd,nkd->nk", u, cand_emb)            # [N, K]


def fit_selection_head(z: np.ndarray, cand_emb: np.ndarray, target_idx: np.ndarray,
                       *, epochs: int = 300, lr: float = 0.05, l2: float = 1e-4,
                       batch: int = 4096, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fit A,b minimising restricted-softmax CE toward target_idx over the K
    candidates. Positions with target_idx < 0 (converged token not in top-K) are
    dropped from the loss. Adam, minibatched (cand_emb can be large at K=100).

    Initialised at A=I, b=0 (the logit lens) so training only *re-ranks*.
    """
    z = np.asarray(z, np.float64)
    cand_emb = np.asarray(cand_emb, np.float64)
    target_idx = np.asarray(target_idx)
    keep = target_idx >= 0
    z, cand_emb, t = z[keep], cand_emb[keep], target_idx[keep]
    N, D = z.shape
    K = cand_emb.shape[1]
    A = np.eye(D)
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
    """Per valid position gather (z, cand_emb, target_idx, rank0, prompt_id).

    z = rmsnorm(iter-0 hidden); cand_emb = W_head rows of think's iter-0 top-K;
    target_idx = slot of the converged token in that top-K (-1 if absent);
    prompt_id = sample-file stem (for the prompt-disjoint split).
    """
    z_all, e_all, t_all, r_all, pid_all = [], [], [], [], []
    W = head.W                                             # [vocab, D]
    done = 0
    for path in tc.iter_sample_paths(probes_root, model):
        sample = tc.load_sample(path)
        attrs = sample["attrs"]
        pid = path.stem
        for b, blk in sorted(sample["blocks"].items()):
            if "converged_tokens" not in blk and "committed_tokens_per_pass" not in blk:
                continue
            h0 = np.asarray(blk["h_per_pass"][0, -1])      # [B, D] iter-0 last layer
            conv = np.asarray(tc.converged_tokens(blk))    # [B]
            valid = tc.valid_position_mask(b, h0.shape[0], attrs, block_length)
            valid &= (conv != tc.MASK_ID)
            if not valid.any():
                continue
            h0v, convv = h0[valid], conv[valid]
            logits = head.logits(h0v)                      # [Bv, vocab]
            z = head._rmsnorm(h0v)                         # [Bv, D]
            r0 = (logits > logits[np.arange(len(convv)), convv][:, None]).sum(-1) + 1
            for i in range(len(convv)):
                ids = _topk_ids(logits[i], K)              # [K]
                hit = np.where(ids == convv[i])[0]
                t_all.append(int(hit[0]) if hit.size else -1)
                e_all.append(W[ids])                       # [K, D]
                z_all.append(z[i])
                r_all.append(int(r0[i]))
                pid_all.append(pid)
        done += 1
        if n_samples and done >= n_samples:
            break
    return {
        "z": np.asarray(z_all, np.float64) if z_all else np.zeros((0, head.d_model)),
        "cand_emb": np.asarray(e_all, np.float64) if e_all else np.zeros((0, K, head.d_model)),
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


def evaluate(data: dict, *, epochs: int = 300, seed: int = 0) -> dict:
    """Fit on train prompts, report acc on test prompts, overall + per bucket."""
    tr, te = _prompt_split(data["prompt_id"])
    A, b = fit_selection_head(data["z"][tr], data["cand_emb"][tr], data["target_idx"][tr],
                              epochs=epochs, seed=seed)
    scores = head_scores(data["z"][te], data["cand_emb"][te], A, b)
    t_te = data["target_idx"][te]
    bucket = readiness_bucket(data["rank0"][te])
    base = baseline_summary(t_te)
    res = {"overall": {**base, "selection": selection_accuracy(scores, t_te)},
           "by_bucket": {}, "n_train": int(tr.sum()), "n_test": int(te.sum())}
    for bk in range(3):
        sel = bucket == bk
        if not sel.any():
            continue
        res["by_bucket"][BUCKETS[bk]] = {
            **baseline_summary(t_te[sel]),
            "selection": selection_accuracy(scores[sel], t_te[sel]),
        }
    return res


# ---- report + plot ----------------------------------------------------------

def format_report(per_k: dict) -> str:
    lines = ["Step-1 selection-head probe (tuned-lens over iter-0 anchor, top-K restricted)"]
    for k, res in per_k.items():
        o = res["overall"]
        lines.append(f"  K={k}: train/test positions={res['n_train']}/{res['n_test']}")
        lines.append(f"    overall   argmax-copy={o['argmax_copy']:.1%}  "
                     f"selection={o['selection']:.1%}  ceiling(c∈topK)={o['ceiling']:.1%}")
        for name, r in res["by_bucket"].items():
            lines.append(f"    {name:<14s} argmax-copy={r['argmax_copy']:.1%}  "
                         f"selection={r['selection']:.1%}  ceiling={r['ceiling']:.1%}  (n={r['n']})")
    return "\n".join(lines)


def plot(per_k: dict, out_dir: Path, model: str) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = list(per_k.keys())
    fig, axes = plt.subplots(1, len(ks), figsize=(7.5 * len(ks), 4.8), squeeze=False)
    for j, k in enumerate(ks):
        ax = axes[0][j]
        res = per_k[k]
        names = ["overall"] + list(res["by_bucket"].keys())
        xs = np.arange(len(names))
        argmax = [res["overall"]["argmax_copy"]] + [res["by_bucket"][n]["argmax_copy"] for n in names[1:]]
        sel = [res["overall"]["selection"]] + [res["by_bucket"][n]["selection"] for n in names[1:]]
        ceil = [res["overall"]["ceiling"]] + [res["by_bucket"][n]["ceiling"] for n in names[1:]]
        ax.bar(xs - 0.25, argmax, width=0.25, color="tab:gray", label="argmax-copy (floor)")
        ax.bar(xs + 0.00, sel, width=0.25, color="tab:green", label="selection-head")
        ax.bar(xs + 0.25, ceil, width=0.25, color="tab:purple", alpha=0.6, label="ceiling c∈topK")
        ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.set_ylim(0, 1); ax.set_ylabel("selection accuracy")
        ax.set_title(f"K={k}: does the readout beat argmax-copy?")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"{model}: Step-1 selection-head extractability")
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
        per_k[k] = evaluate(data, epochs=args.epochs, seed=args.seed)
    print(format_report(per_k))
    plot(per_k, Path(args.probes_root) / args.model / "plots", args.model)


if __name__ == "__main__":
    main()
