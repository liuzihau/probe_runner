"""Step 1, capacity rung — an *attention* selection head over revealed neighbors.

The linear tuned-lens probe (plot_selection_head.py) tests whether the converged
token is *linearly* selectable from the static anchor (+ a pooled neighbor mean).
This module is the next rung: a small cross-attention head that attends from each
masked position's iter-0 anchor over the *individual* revealed-neighbor hiddens of
its block, then selects within think's iter-0 top-K. It is the architecturally
faithful probe — it matches the real talk's cross-attention inductive bias and the
layerwise finding that the 21%→75% nudge is upper-layer attention integrating
revealed neighbors (R4 / \S layerwise). An LSTM's sequential recurrence is the wrong
bias for a bidirectional block with scattered reveals; a 1–2 layer attention block is
right, and it strictly contains the linear probe (zero the attention → the anchor
readout remains), so it cannot do worse than linear given training.

Design / fairness invariants (must hold or the result is meaningless):
  * SELF-MASK. A query position never attends to its own converged hidden — that
    would read the answer. The attention diagonal is −∞; invalid positions are
    key-padding-masked.
  * SAME SCORING. Candidate v scores ⟨u_p, W_head[v]⟩ over think's top-K, identical
    to the linear probe (only the extractor u_p changes), so numbers are comparable.
  * SAME SPLIT/METRICS. Prompt-disjoint split, tail-only training (R1), per-bucket +
    tail aggregate — all reused from plot_selection_head.
  * LOGIT-LENS INIT. The anchor branch is identity-init and the attention branch is
    zero-init, so at step 0 the head == the logit lens (reproduces argmax-copy);
    training only adds neighbor-driven re-ranking.

Fully offline (captured hidden + exported head, no 16B forward), but needs torch +
(ideally) a GPU for the attention training — runs on the H200 in parallel with the
CPU linear 1a+1b run.

Usage
-----
    python -m probe_runner.plots.plot_selection_attn --model llada2-DMAX \
        --probes_root probes_out --K 10 100
    python -m probe_runner.plots.plot_selection_attn --selftest    # CPU, tiny synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc
from probe_runner.plots.plot_selection_head import (
    BUCKETS, readiness_bucket, baseline_summary, selection_accuracy,
    _prompt_split, _topk_ids,
)


# ---- block-structured data driver (offline; head only) ----------------------

def compute_block_inputs(probes_root, model: str, head: tc.RankHead, K: int,
                         block_length: int = 32, n_samples: int | None = None) -> dict:
    """Per block, padded to B=block_length:
        z      [nb,B,D]  rmsnorm(iter-0 anchor hidden)
        zc     [nb,B,D]  rmsnorm(converged hidden) — the revealed-neighbor context
        cand   [nb,B,K]  think iter-0 top-K ids (0-padded)
        target [nb,B]    slot of converged token in top-K (-1 if absent/pad)
        rank0  [nb,B]    iter-0 rank of converged token (big for pad)
        valid  [nb,B]    bool: EOS-filtered, non-mask, real position
        pid    [nb]      sample stem (prompt-disjoint split)
    Context for query i is {zc[j] : j≠i, valid[j]} — enforced at attention time by
    the diagonal self-mask + key-padding mask (NOT here), so zc keeps every slot.
    """
    Z, ZC, CAND, TGT, R0, VAL, PID = [], [], [], [], [], [], []
    B, D = block_length, head.d_model
    done = 0
    for path in tc.iter_sample_paths(probes_root, model):
        sample = tc.load_sample(path)
        attrs = sample["attrs"]
        pid = path.stem
        for b, blk in sorted(sample["blocks"].items()):
            if "converged_tokens" not in blk and "committed_tokens_per_pass" not in blk:
                continue
            hpp = np.asarray(blk["h_per_pass"])            # [P, L+1, Bblk, D]
            Bblk = hpp.shape[2]
            h0 = hpp[0, -1]                                # [Bblk, D]
            h_conv = hpp[-1, -1]                           # [Bblk, D]
            conv = np.asarray(tc.converged_tokens(blk))    # [Bblk]
            valid = tc.valid_position_mask(b, Bblk, attrs, block_length)
            valid = valid & (conv != tc.MASK_ID)
            z = head._rmsnorm(h0)                          # [Bblk, D]
            zc = head._rmsnorm(h_conv)
            logits = head.logits(h0)                       # [Bblk, vocab]
            cand = np.zeros((Bblk, K), dtype=np.int64)
            tgt = np.full(Bblk, -1, dtype=np.int64)
            r0 = np.full(Bblk, head.vocab_size, dtype=np.int64)
            for i in range(Bblk):
                if not valid[i]:
                    continue
                ids = _topk_ids(logits[i], K)
                cand[i] = ids
                hit = np.where(ids == conv[i])[0]
                tgt[i] = int(hit[0]) if hit.size else -1
                r0[i] = int((logits[i] > logits[i, conv[i]]).sum() + 1)
            # pad block up to B
            def _pad(a, fill):
                if a.shape[0] >= B:
                    return a[:B]
                pad = np.full((B - a.shape[0],) + a.shape[1:], fill, dtype=a.dtype)
                return np.concatenate([a, pad], axis=0)
            Z.append(_pad(z.astype(np.float16), 0)); ZC.append(_pad(zc.astype(np.float16), 0))
            CAND.append(_pad(cand, 0)); TGT.append(_pad(tgt, -1))
            R0.append(_pad(r0, head.vocab_size))
            VAL.append(_pad(valid.astype(bool), False)); PID.append(pid)
        done += 1
        if n_samples and done >= n_samples:
            break
    return {
        "z": np.asarray(Z), "zc": np.asarray(ZC), "cand": np.asarray(CAND),
        "target": np.asarray(TGT), "rank0": np.asarray(R0), "valid": np.asarray(VAL),
        "pid": np.asarray(PID), "D": D, "K": K,
    }


# ---- attention selection head (torch) ---------------------------------------

def _sinusoid(n: int, d: int, device):
    import torch
    pos = torch.arange(n, device=device).float().unsqueeze(1)        # [n,1]
    j = torch.arange(d, device=device).float().unsqueeze(0)          # [1,d]
    angle = pos / (10000.0 ** (2.0 * (j // 2) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe                                                        # [n,d]


def _build_model(D: int, d_attn: int, n_heads: int, n_layers: int):
    import torch
    import torch.nn as nn

    class SelectionAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(D, d_attn)
            self.k = nn.Linear(D, d_attn)         # keys: routing (get positional encoding)
            self.v = nn.Linear(D, d_attn)         # values: token content (no positions)
            # learned null key+value (token-free) so every query has ≥1 valid key even
            # with no revealed neighbor — avoids NaN softmax without leaking.
            self.null_k = nn.Parameter(torch.zeros(1, 1, d_attn))
            self.null_v = nn.Parameter(torch.zeros(1, 1, d_attn))
            self.layers = nn.ModuleList([
                nn.MultiheadAttention(d_attn, n_heads, batch_first=True) for _ in range(n_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(d_attn) for _ in range(n_layers)])
            self.out = nn.Linear(d_attn, D)
            self.anchor = nn.Linear(D, D)
            # logit-lens init: anchor == identity, attention contributes nothing.
            nn.init.eye_(self.anchor.weight); nn.init.zeros_(self.anchor.bias)
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

        def forward(self, z, zc, valid):
            # z,zc [b,B,D]; valid [b,B] bool. Returns u [b,B,D] = anchor + attn(neighbors).
            b, Bn, _ = z.shape
            pe = _sinusoid(Bn, self.q.out_features, z.device).unsqueeze(0)  # [1,B,d]
            q = self.q(z) + pe                                             # [b,B,d] (positioned)
            K = torch.cat([self.null_k.expand(b, -1, -1), self.k(zc) + pe], dim=1)  # keys: positioned
            V = torch.cat([self.null_v.expand(b, -1, -1), self.v(zc)], dim=1)       # values: token-only
            # column 0 = null (never masked); cols 1.. = neighbors (position j → col j+1).
            # self-mask: query i cannot attend its own column (i+1). Bool masks (same
            # type as key_padding_mask): True = disallow.
            attn_mask = torch.zeros(Bn, Bn + 1, dtype=torch.bool, device=z.device)
            attn_mask[torch.arange(Bn), torch.arange(Bn) + 1] = True
            key_pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=z.device),
                                 ~valid], dim=1)                           # [b,1+B] True = ignore
            h = q
            for attn, norm in zip(self.layers, self.norms):
                a, _ = attn(h, K, V, attn_mask=attn_mask, key_padding_mask=key_pad,
                            need_weights=False)
                h = norm(h + torch.nan_to_num(a))
            return self.anchor(z) + self.out(h)            # [b,B,D]

    return SelectionAttn()


def _gather_cand_emb(W, cand):
    """W [V,D] (torch), cand [b,B,K] (torch long) → [b,B,K,D]."""
    return W[cand]


def train_attn(data: dict, head: tc.RankHead, *, tail_only: bool = True,
               zero_neighbors: bool = False,
               d_attn: int = 256, n_heads: int = 4, n_layers: int = 2,
               epochs: int = 40, lr: float = 1e-3, batch_blocks: int = 16,
               device: str = "cuda", seed: int = 0) -> dict:
    """Fit on train prompts, evaluate selection accuracy on held-out prompts.

    zero_neighbors: ablation — feed zeroed neighbor hiddens so the value branch
    carries no token content (only the null key + the anchor branch remain). If the
    full-neighbor run equals this ablation, the neighbor pathway adds nothing —
    the decisive check for "neighbors don't help" vs "attention not training".
    """
    import torch
    torch.manual_seed(seed)
    D, K = data["D"], data["K"]
    tr_pid, te_pid = _prompt_split(data["pid"])
    W = torch.tensor(head.W, dtype=torch.float32, device=device)   # [V,D]
    model = _build_model(D, d_attn, n_heads, n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    zc_scale = 0.0 if zero_neighbors else 1.0

    def _batched(idx):
        for s in range(0, len(idx), batch_blocks):
            yield idx[s:s + batch_blocks]

    def _to(t, dt=torch.float32):
        return torch.tensor(t, dtype=dt, device=device)

    tr_idx = np.where(tr_pid)[0]
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = tr_idx[rng.permutation(len(tr_idx))]
        for bi in _batched(order):
            z = _to(data["z"][bi]); zc = _to(data["zc"][bi]) * zc_scale
            valid = torch.tensor(data["valid"][bi], device=device)
            cand = torch.tensor(data["cand"][bi], dtype=torch.long, device=device)
            tgt = torch.tensor(data["target"][bi], dtype=torch.long, device=device)
            r0 = torch.tensor(data["rank0"][bi], dtype=torch.long, device=device)
            u = model(z, zc, valid)                         # [b,B,D]
            cand_emb = _gather_cand_emb(W, cand)            # [b,B,K,D]
            scores = (u.unsqueeze(2) * cand_emb).sum(-1)    # [b,B,K]
            train_pos = valid & (tgt >= 0)
            if tail_only:
                train_pos = train_pos & (r0 > 1)
            if train_pos.sum() == 0:
                continue
            logp = torch.log_softmax(scores[train_pos], dim=-1)
            loss = -logp[torch.arange(logp.shape[0]), tgt[train_pos]].mean()
            opt.zero_grad(); loss.backward(); opt.step()

    # eval on held-out prompts
    model.eval()
    sc_all, tgt_all, r0_all = [], [], []
    te_idx = np.where(te_pid)[0]
    with torch.no_grad():
        for bi in _batched(te_idx):
            z = _to(data["z"][bi]); zc = _to(data["zc"][bi]) * zc_scale
            valid = torch.tensor(data["valid"][bi], device=device)
            cand = torch.tensor(data["cand"][bi], dtype=torch.long, device=device)
            u = model(z, zc, valid)
            cand_emb = _gather_cand_emb(W, cand)
            scores = (u.unsqueeze(2) * cand_emb).sum(-1)    # [b,B,K]
            v = valid.cpu().numpy().reshape(-1)
            sc_all.append(scores.cpu().numpy().reshape(-1, K)[v])
            tgt_all.append(data["target"][bi].reshape(-1)[v])
            r0_all.append(data["rank0"][bi].reshape(-1)[v])
    scores = np.concatenate(sc_all); t_te = np.concatenate(tgt_all); r0_te = np.concatenate(r0_all)
    bucket = readiness_bucket(r0_te)
    res = {"overall": {**baseline_summary(t_te), "selection": selection_accuracy(scores, t_te)},
           "by_bucket": {}, "tail": {}, "tail_only": tail_only,
           "n_train": int(tr_pid.sum()), "n_test": int(len(t_te)),
           "arch": f"attn(d={d_attn},h={n_heads},L={n_layers})"}
    for bk in range(3):
        sel = bucket == bk
        if sel.any():
            res["by_bucket"][BUCKETS[bk]] = {**baseline_summary(t_te[sel]),
                                             "selection": selection_accuracy(scores[sel], t_te[sel])}
    tail = bucket > 0
    res["tail"] = {**baseline_summary(t_te[tail]), "selection": selection_accuracy(scores[tail], t_te[tail])}
    return res


# ---- report + plot ----------------------------------------------------------

def format_report(per_k: dict) -> str:
    lines = ["Step-1 ATTENTION selection head (cross-attn over revealed neighbors, self-masked)"]
    for k, res in per_k.items():
        t = res["tail"]; med = res["by_bucket"].get(BUCKETS[1], {}); hard = res["by_bucket"].get(BUCKETS[2], {})
        lines.append(f"  K={k} [{res['arch']}, tail_only={res['tail_only']}, "
                     f"train/test blocks/pos={res['n_train']}/{res['n_test']}]")
        lines.append(f"    tail-selection={t['selection']:.1%}  "
                     f"(medium={med.get('selection', float('nan')):.1%}, "
                     f"hard={hard.get('selection', float('nan')):.1%})  "
                     f"tail ceiling={t['ceiling']:.1%}")
    return "\n".join(lines)


def plot(per_k: dict, out_dir: Path, model: str) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = list(per_k.keys())
    fig, axes = plt.subplots(1, len(ks), figsize=(7.5 * len(ks), 4.8), squeeze=False)
    for j, k in enumerate(ks):
        ax = axes[0][j]; res = per_k[k]
        groups = ["medium(r≤10)", "hard(r>10)", "tail(r>1)"]
        xs = np.arange(len(groups))
        sel = [res["by_bucket"].get(BUCKETS[1], {}).get("selection", np.nan),
               res["by_bucket"].get(BUCKETS[2], {}).get("selection", np.nan),
               res["tail"]["selection"]]
        ceil = [res["by_bucket"].get(BUCKETS[1], {}).get("ceiling", np.nan),
                res["by_bucket"].get(BUCKETS[2], {}).get("ceiling", np.nan),
                res["tail"]["ceiling"]]
        ax.bar(xs - 0.2, sel, width=0.4, color="tab:blue", label="attn selection")
        ax.bar(xs + 0.2, ceil, width=0.4, color="tab:purple", alpha=0.45, label="ceiling c∈topK")
        ax.axhline(0.215, ls="--", lw=0.8, color="black")
        ax.text(0.02, 0.225, "static-anchor floor ≈21%", fontsize=7, transform=ax.get_yaxis_transform())
        ax.set_xticks(xs); ax.set_xticklabels(groups, fontsize=8); ax.set_ylim(0, 1)
        ax.set_ylabel("selection accuracy"); ax.set_title(f"K={k}: {res['arch']}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"{model}: attention selection head extractability")
    fig.tight_layout()
    p = out_dir / "selection_attn.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")


# ---- selftest (CPU, tiny synthetic) -----------------------------------------

def _selftest() -> None:
    """Each query's correct candidate is encoded by a SPECIFIC neighbor's vector;
    only a self-masked attention over neighbors can solve it. Verifies forward/
    backward, the self-mask (no leakage), and that it learns the neighbor signal."""
    import torch
    rng = np.random.default_rng(0)
    nb, B, D, K, V = 80, 6, 16, 4, 40
    W = rng.standard_normal((V, D)).astype(np.float32)
    Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-8)
    z = np.zeros((nb, B, D), np.float16)                         # anchor: deliberately empty,
    zc = np.zeros((nb, B, D), np.float16)                        # so ALL signal is in neighbors
    cand = np.zeros((nb, B, K), np.int64); tgt = np.full((nb, B), -1, np.int64)
    r0 = np.full((nb, B), 5, np.int64); valid = np.ones((nb, B), bool)
    for s in range(nb):
        toks = rng.integers(0, V, size=B)
        for i in range(B):
            ids = rng.choice(V, size=K, replace=False)
            # the answer for i is the token carried by its right neighbor (i+1)%B
            nb_tok = toks[(i + 1) % B]
            ids[0] = nb_tok                                       # ensure in candidate set
            cand[s, i] = ids
            tgt[s, i] = 0
            zc[s, i] = Wn[toks[i]]                                # neighbor i broadcasts its token
        # so query i must attend to neighbor (i+1) to read its token → select it
    data = {"z": z, "zc": zc, "cand": cand, "target": tgt, "rank0": r0,
            "valid": valid, "pid": np.array([f"p{s%8}" for s in range(nb)]), "D": D, "K": K}
    head = tc.RankHead(W, np.ones(D, np.float32))               # only .W used for gather here
    res = train_attn(data, head, tail_only=False, d_attn=32, n_heads=2, n_layers=2,
                     epochs=200, lr=5e-3, batch_blocks=16, device="cpu", seed=0)
    acc = res["tail"]["selection"]
    assert acc > 0.7, f"attn head should learn the neighbor-encoded target, got {acc:.2f}"
    print(f"plot_selection_attn selftest OK (neighbor-target acc={acc:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llada2")
    ap.add_argument("--probes_root", default="probes_out")
    ap.add_argument("--K", type=int, nargs="+", default=[10, 100])
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--d_attn", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_blocks", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_tail_only", action="store_true")
    ap.add_argument("--no_neighbors", action="store_true",
                    help="ablation: zero neighbor content (only null+anchor) — if equal to "
                         "the full run, the neighbor pathway adds nothing.")
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
        data = compute_block_inputs(args.probes_root, args.model, head, k,
                                    block_length=args.block_length, n_samples=args.n_samples)
        per_k[k] = train_attn(data, head, tail_only=not args.no_tail_only,
                              zero_neighbors=args.no_neighbors,
                              d_attn=args.d_attn, n_heads=args.n_heads, n_layers=args.n_layers,
                              epochs=args.epochs, lr=args.lr, batch_blocks=args.batch_blocks,
                              device=args.device, seed=args.seed)
    print(format_report(per_k))
    plot(per_k, Path(args.probes_root) / args.model / "plots", args.model)


if __name__ == "__main__":
    main()
