"""B1 — Selectivity signal: can a CHEAP signal tell which positions are safe to draft?

The selective drafter (brainstorm.md §8.6) must, at draft time, separate positions it will get
RIGHT (commit) from ones it will get WRONG (abstain — a wrong draft is taxed: it costs a heavy
correction round and delays the hard tail). E1's teaser showed the heavy's think-pass confidence
FAILS this (AUC < 0.5). B1 tests stronger signals.

Population: positions NOT committed in the think pass (the drafter's actual candidates) that the
decode eventually commits.

Labels (both cheap — no oracle leave-one-out needed):
  * draft_correct : heavy think-pass argmax == final committed token  (the abstention target)
  * lingering     : position takes >= 2 passes to commit (~ E1 'easy-uncertain'; hard_tail ~1%)

Signals tested (predict each label):
  * conf   : think-pass top-1 probability
  * entropy: think-pass entropy
  * margin : top1 - top2 probability
  * probe@shallow : trained logistic/MLP probe on the think-pass hidden at a SHALLOW layer
                    (~ what a depth-truncated lightweight would actually see)
  * probe@deep    : same on the last-layer hidden (upper bound on available signal)

GATE: some cheap signal reaches test AUC > 0.7 on draft_correct. FAIL => no abstention signal =>
the selective drafter is not viable (contamination tax uncontrollable).

Run at the OPERATING point where the drafter is viable (thr 0.5/0.6), not 0.3.

Usage
-----
    python -m probe_runner.exp_b1_selectivity \
        --model_path $DMAX_MATH_PATH --t3dmax_root $T3DMAX_ROOT \
        --gsm8k_n 100 --commit_threshold 0.5 --feat_layer_shallow 9 \
        --out_dir fork_bounded_surrogate/results/B1_thr0.5
    python -m probe_runner.exp_b1_selectivity --selftest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc

MASK_ID = tc.MASK_ID
DEFAULT_EOS = 156892


# ---- pure-python cores (torch-free, unit-testable) --------------------------

def auc_mann_whitney(scores_pos, scores_neg) -> float:
    pos = np.asarray(scores_pos, float); neg = np.asarray(scores_neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_from_scores(scores, labels) -> float:
    s = np.asarray(scores, float); y = np.asarray(labels).astype(bool)
    return auc_mann_whitney(s[y], s[~y])


def standardize(X_tr, X_te):
    mu = X_tr.mean(0, keepdims=True); sd = X_tr.std(0, keepdims=True) + 1e-6
    return (X_tr - mu) / sd, (X_te - mu) / sd


# ---- decode + think-pass feature capture (torch) ----------------------------

def decode_and_capture(model, prompt, *, mask_id, gen_length, block_length,
                       commit_threshold, break_threshold, shallow, deep, max_iter=None):
    """Soft DMax decode; at the THINK pass capture per-position features. Returns dict of
    np arrays over generated positions [gen]: reveal_pass, conf, entropy, margin, draft_tok,
    final_tok, and hidden_shallow/deep [gen, D] (think-pass hidden at those layers)."""
    import torch
    import torch.nn.functional as F
    from probe_runner.llada2_runner import (
        _build_block_causal_mask, _commit_uniform, _build_block_soft_embeds, _embedding_layer)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prompt = prompt.to(device)
    Lp = int(prompt.shape[1])
    num_blocks = gen_length // block_length
    if max_iter is None:
        max_iter = block_length
    emb = _embedding_layer(model)

    x = F.pad(prompt, (0, gen_length), value=mask_id)
    reveal_pass = np.full(gen_length, -1, np.int64)
    conf = np.full(gen_length, np.nan); ent = np.full(gen_length, np.nan)
    margin = np.full(gen_length, np.nan); draft = np.full(gen_length, -1, np.int64)
    h_shallow = None; h_deep = None  # lazily sized [gen, D]

    with torch.no_grad():
        for nb in range(num_blocks):
            s, e = Lp + nb * block_length, Lp + (nb + 1) * block_length
            g0 = nb * block_length
            attn = _build_block_causal_mask(e, Lp, block_length, dtype, device)
            pos = torch.arange(e, device=device).unsqueeze(0)
            block_soft = None
            for i in range(max_iter):
                inputs_embeds = emb(x[:, :e])
                if block_soft is not None:
                    inputs_embeds[:, s:e, :] = block_soft
                want_hidden = (i == 0)
                out = model(inputs_embeds=inputs_embeds, attention_mask=attn, position_ids=pos,
                            use_cache=False, output_hidden_states=want_hidden, return_dict=True)
                block_logits = out.logits[:, s:e, :]
                block_x = x[:, s:e]
                mask_index = (block_x == mask_id)
                x0, commit_index, max_probs = _commit_uniform(
                    block_logits, block_x, mask_id, commit_threshold)
                if i == 0:
                    p = torch.softmax(block_logits.float(), dim=-1)[0]      # [B, V]
                    top2 = p.topk(2, dim=-1).values                         # [B, 2]
                    conf[g0:g0 + block_length] = top2[:, 0].cpu().numpy()
                    margin[g0:g0 + block_length] = (top2[:, 0] - top2[:, 1]).cpu().numpy()
                    ent[g0:g0 + block_length] = (-(p * torch.clamp(p, min=1e-12).log()).sum(-1)).cpu().numpy()
                    draft[g0:g0 + block_length] = block_logits[0].argmax(-1).cpu().numpy()
                    hs = out.hidden_states                                  # tuple len depth+1
                    D = hs[0].shape[-1]
                    if h_shallow is None:
                        h_shallow = np.zeros((gen_length, D), np.float16)
                        h_deep = np.zeros((gen_length, D), np.float16)
                    h_shallow[g0:g0 + block_length] = hs[shallow][0, s:e, :].float().cpu().numpy()
                    h_deep[g0:g0 + block_length] = hs[deep][0, s:e, :].float().cpu().numpy()

                update_mask = commit_index | (~mask_index)
                changed = update_mask & (x0 != block_x)
                new_block = block_x.clone(); new_block[update_mask] = x0[update_mask]
                x[:, s:e] = new_block
                revealed = (mask_index & commit_index)[0]
                for loc in torch.where(revealed)[0].cpu().numpy():
                    if reveal_pass[g0 + loc] < 0:
                        reveal_pass[g0 + loc] = i
                if not bool(changed.any()) or bool((max_probs >= break_threshold).all()):
                    break
                soft_cond = (x[0, s:e] != mask_id)
                block_soft = _build_block_soft_embeds(x[:, s:e], max_probs, x0, soft_cond, emb, mask_id)

    final = x[0, Lp:].cpu().numpy()
    return {"reveal_pass": reveal_pass, "conf": conf, "entropy": ent, "margin": margin,
            "draft": draft, "final": final, "h_shallow": h_shallow, "h_deep": h_deep}


# ---- probe (torch logistic / 1-hidden-layer MLP) ----------------------------

def train_probe_auc(Xtr, ytr, Xte, yte, *, mlp=False, epochs=300, lr=1e-2, wd=1e-3, device="cpu"):
    import torch
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xte = torch.tensor(Xte, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    d = Xtr.shape[1]
    if mlp:
        h = 64
        net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, 1)).to(device)
    else:
        net = torch.nn.Linear(d, 1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xtr).squeeze(-1), ytr_t)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        scores = net(Xte).squeeze(-1).cpu().numpy()
    return auc_from_scores(scores, yte)


def run(args) -> None:
    import torch
    from probe_runner.llada2_runner import load_llada2
    from datasets import load_dataset

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_llada2(args.model_path, attn_implementation="sdpa",
                                   t3dmax_root=args.t3dmax_root, device=args.device)
    depth = sum(1 for _ in model.model.layers)
    deep = args.feat_layer_deep if args.feat_layer_deep >= 0 else depth
    mask_id = MASK_ID
    eos_id = args.eos_id if args.eos_id is not None else (tokenizer.eos_token_id or DEFAULT_EOS)
    print(f"[B1] depth={depth} shallow_layer={args.feat_layer_shallow} deep_layer={deep} "
          f"thr={args.commit_threshold}")

    ds = load_dataset("gsm8k", "main", split="test").select(range(args.gsm8k_n))

    feats = {"conf": [], "entropy": [], "margin": []}
    Hs, Hd, y_correct, y_linger, prob_id = [], [], [], [], []
    for qi, ex in enumerate(ds):
        msg = [{"role": "user", "content": ex["question"]}]
        p = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        prompt = tokenizer(p, return_tensors="pt")["input_ids"].to(args.device)
        cap = decode_and_capture(
            model, prompt, mask_id=mask_id, gen_length=args.gen_length, block_length=args.block_length,
            commit_threshold=args.commit_threshold, break_threshold=args.break_threshold,
            shallow=args.feat_layer_shallow, deep=deep)
        final = cap["final"]
        eos_hits = np.where(final == eos_id)[0]
        gen_end = int(eos_hits[0]) + 1 if len(eos_hits) else args.gen_length
        for g in range(gen_end):
            rp = int(cap["reveal_pass"][g])
            if rp == 0 or rp == -1:        # committed at think (not a candidate) / never committed
                continue
            if final[g] == mask_id:
                continue
            feats["conf"].append(float(cap["conf"][g]))
            feats["entropy"].append(float(cap["entropy"][g]))
            feats["margin"].append(float(cap["margin"][g]))
            Hs.append(cap["h_shallow"][g]); Hd.append(cap["h_deep"][g])
            y_correct.append(bool(cap["draft"][g] == final[g]))
            y_linger.append(bool(rp >= 2))
            prob_id.append(qi)
        if (qi + 1) % 10 == 0:
            print(f"[B1] {qi+1}/{len(ds)}  candidates so far={len(y_correct)}")

    prob_id = np.array(prob_id); n = len(y_correct)
    yc = np.array(y_correct); yl = np.array(y_linger)
    conf = np.array(feats["conf"]); entropy = np.array(feats["entropy"]); margin = np.array(feats["margin"])
    Hs = np.asarray(Hs, np.float32); Hd = np.asarray(Hd, np.float32)
    print(f"[B1] candidates n={n}  draft_correct base_rate={yc.mean():.3f}  lingering base_rate={yl.mean():.3f}")

    # split by problem id (no leakage)
    rng = np.random.default_rng(0)
    probs = np.unique(prob_id); rng.shuffle(probs)
    n_te = max(1, int(0.2 * len(probs)))
    te_probs = set(probs[:n_te].tolist())
    te = np.array([pid in te_probs for pid in prob_id]); tr = ~te

    results = {}
    for label_name, y in [("draft_correct", yc), ("lingering", yl)]:
        row = {"base_rate": float(y.mean())}
        # scalar signals: AUC directly (sign-agnostic: report max(auc, 1-auc) is misleading; keep raw)
        row["conf"] = auc_from_scores(conf, y)
        row["entropy_neg"] = auc_from_scores(-entropy, y)        # lower entropy -> positive?
        row["margin"] = auc_from_scores(margin, y)
        # trained probes on hidden
        Xs_tr, Xs_te = standardize(Hs[tr], Hs[te])
        Xd_tr, Xd_te = standardize(Hd[tr], Hd[te])
        row["probe_shallow_logreg"] = train_probe_auc(Xs_tr, y[tr], Xs_te, y[te], mlp=False, device=args.device)
        row["probe_deep_logreg"] = train_probe_auc(Xd_tr, y[tr], Xd_te, y[te], mlp=False, device=args.device)
        row["probe_shallow_mlp"] = train_probe_auc(Xs_tr, y[tr], Xs_te, y[te], mlp=True, device=args.device)
        results[label_name] = row

    gate_signals = ["conf", "entropy_neg", "margin", "probe_shallow_logreg", "probe_shallow_mlp", "probe_deep_logreg"]
    best_correct = max(abs(results["draft_correct"][s] - 0.5) for s in gate_signals) + 0.5
    gate_pass = best_correct > args.gate_auc

    summary = {"model_path": args.model_path, "commit_threshold": args.commit_threshold,
               "n_candidates": n, "depth": depth, "shallow_layer": args.feat_layer_shallow,
               "deep_layer": deep, "gate_auc": args.gate_auc, "results": results,
               "best_draft_correct_auc": best_correct, "GATE_PASS": bool(gate_pass)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [f"B1 — selectivity (thr={args.commit_threshold}, n_candidates={n})",
             f"  draft_correct base_rate={yc.mean():.3f}  lingering base_rate={yl.mean():.3f}",
             "  test AUC (>0.5 useful; need >%.2f to PASS):" % args.gate_auc]
    for label_name in ("draft_correct", "lingering"):
        lines.append(f"  [{label_name}]")
        for sig in gate_signals:
            lines.append(f"      {sig:<22s} {results[label_name][sig]:.3f}")
    lines.append("")
    lines.append(f"  GATE (best draft_correct AUC > {args.gate_auc}): best={best_correct:.3f} => "
                 f"{'PASS — a usable selectivity signal exists' if gate_pass else 'FAIL — no cheap signal separates safe-to-draft from not'}")
    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "report.txt").write_text(report + "\n")
    print(f"[B1] wrote summary.json, report.txt -> {out_dir}")


def _selftest() -> None:
    assert abs(auc_mann_whitney([3, 4, 5], [0, 1, 2]) - 1.0) < 1e-9
    assert abs(auc_from_scores([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0]) - 1.0) < 1e-9
    a, b = standardize(np.array([[0.0], [2.0]]), np.array([[1.0]]))
    assert abs(a.mean()) < 1e-6 and abs(b[0, 0]) < 1e-6
    print("exp_b1 selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="B1: selectivity signal for the drafter")
    ap.add_argument("--model_path")
    ap.add_argument("--t3dmax_root", default=None)
    ap.add_argument("--gsm8k_n", type=int, default=100)
    ap.add_argument("--gen_length", type=int, default=512)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--commit_threshold", type=float, default=0.5)
    ap.add_argument("--break_threshold", type=float, default=0.9)
    ap.add_argument("--feat_layer_shallow", type=int, default=9, help="hidden-state layer ~ a truncated lightweight")
    ap.add_argument("--feat_layer_deep", type=int, default=-1, help="-1 => last layer")
    ap.add_argument("--gate_auc", type=float, default=0.7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eos_id", type=int, default=None)
    ap.add_argument("--out_dir", default="fork_bounded_surrogate/results/B1")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if not args.model_path:
        raise SystemExit("--model_path is required (or pass --selftest).")
    run(args)


if __name__ == "__main__":
    main()
