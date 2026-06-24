"""E1 — Cannibalization / easy-uncertain mass.  THE decider (brainstorm.md §9, §5.0).

Measures whether the proposed lightweight drafter has a job at all. Of the positions
still masked after DMax's THINK pass, how many are:

  * REDUNDANT     - the heavy commits them in the very next pass anyway (no win),
  * EASY-UNCERTAIN- low-confidence at think, but HIGH-confidence once their neighbors
                    are resolved (the lightweight's target: blocked only by neighbor
                    uncertainty, cheap to resolve, but the heavy spends real rounds on
                    them), or
  * HARD-TAIL     - low-confidence even given perfectly-resolved neighbors (genuinely
                    heavy-capacity; not offloadable).

If EASY-UNCERTAIN mass is small, or it overlaps what the heavy commits within a pass or
two anyway, the lightweight is re-harvesting DMax's own redundancy => kill it (ship
Mechanism-B truncated-heavy only).

Method (zero training, reuses DMax decode primitives from llada2_runner):
  1. Instrumented DMax soft decode (mirrors generate_with_probes soft mode) recording,
     per generated position: reveal_pass (in-block pass index at first commit; -1 never),
     pass0_conf / pass0_entropy (THINK-pass signals), and the final token.
  2. Oracle-neighbor leave-one-out: for each candidate position p, reveal ALL other
     positions to their final (oracle) tokens, mask only p, one forward, read p's top-1
     confidence `oracle_neighbor_conf` and whether argmax recovers the final token.
  3. Classify + aggregate; print PASS/FAIL against the gate.

Usage
-----
    python -m probe_runner.exp_e1_easy_uncertain \
        --model_path $DMAX_MATH_PATH --t3dmax_root $T3DMAX_ROOT \
        --gsm8k_n 100 --gen_length 512 --hi 0.9 --loo_batch 8 \
        --out_dir fork_bounded_surrogate/results/E1
    python -m probe_runner.exp_e1_easy_uncertain --selftest   # pure-python cores
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc

DEFAULT_EOS = 156892
MASK_ID = tc.MASK_ID


# ---- pure-python cores (unit-testable, no torch) ----------------------------

def classify(reveal_pass: int, oracle_conf: float, hi: float) -> str:
    """Three-way (+context) class for one generated position.

    reveal_pass: in-block pass index of first commit; -1 if never committed.
    oracle_conf: top-1 confidence with neighbors revealed (NaN/None if not probed).
    """
    if reveal_pass == 0:
        return "think"          # heavy resolves it in the think pass -> context, not the LW's job
    if reveal_pass == 1:
        return "redundant"      # heavy commits it the very next pass anyway
    # reveal_pass >= 2 or -1 (never): the candidate set
    if oracle_conf is not None and oracle_conf == oracle_conf and oracle_conf >= hi:
        return "easy_uncertain"
    return "hard_tail"


def auc_mann_whitney(scores_pos, scores_neg) -> float:
    """AUC that `score` ranks positives above negatives (0.5 = no separation)."""
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ---- instrumented DMax soft decode (torch) ----------------------------------

def decode_and_record(model, prompt, *, mask_id, eos_id, gen_length, block_length,
                      commit_threshold=0.3, break_threshold=0.9, max_iter=None):
    """DMax `decode_uniform` soft decode, recording per-position signals.

    Faithful to llada2_runner.generate_with_probes (soft mode) but also captures the
    logit-derived signals the callbacks don't expose. Returns:
        x          [1, Lp+gen] final ids
        reveal_pass  int   [gen]  in-block pass index at first commit (-1 never)
        pass0_conf   float [gen]  THINK-pass top-1 confidence
        pass0_ent    float [gen]  THINK-pass entropy (nats)
        block_id     int   [gen]
    """
    import torch
    import torch.nn.functional as F
    from probe_runner.llada2_runner import (
        _build_block_causal_mask, _commit_uniform, _build_block_soft_embeds, _embedding_layer)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prompt = prompt.to(device)
    Lp = int(prompt.shape[1])
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    if max_iter is None:
        max_iter = block_length
    emb = _embedding_layer(model)

    x = F.pad(prompt, (0, gen_length), value=mask_id)
    reveal_pass = np.full(gen_length, -1, dtype=np.int64)
    pass0_conf = np.full(gen_length, np.nan, dtype=np.float64)
    pass0_ent = np.full(gen_length, np.nan, dtype=np.float64)
    block_id = np.arange(gen_length) // block_length

    with torch.no_grad():
        for nb in range(num_blocks):
            s, e = Lp + nb * block_length, Lp + (nb + 1) * block_length
            g0 = nb * block_length                       # global gen index of block start
            attn = _build_block_causal_mask(e, Lp, block_length, dtype, device)
            pos = torch.arange(e, device=device).unsqueeze(0)
            block_soft = None
            for i in range(max_iter):
                inputs_embeds = emb(x[:, :e])
                if block_soft is not None:
                    inputs_embeds[:, s:e, :] = block_soft
                out = model(inputs_embeds=inputs_embeds, attention_mask=attn,
                            position_ids=pos, use_cache=False,
                            output_hidden_states=False, return_dict=True)
                block_logits = out.logits[:, s:e, :]
                block_x = x[:, s:e]
                mask_index = (block_x == mask_id)
                x0, commit_index, max_probs = _commit_uniform(
                    block_logits, block_x, mask_id, commit_threshold)
                if i == 0:
                    p = torch.softmax(block_logits.float(), dim=-1)
                    ent = -(p * torch.clamp(p, min=1e-12).log()).sum(-1)[0]  # [B]
                    pass0_conf[g0:g0 + block_length] = max_probs[0].float().cpu().numpy()
                    pass0_ent[g0:g0 + block_length] = ent.float().cpu().numpy()

                update_mask = commit_index | (~mask_index)               # soft: re-argmax committed
                changed = update_mask & (x0 != block_x)
                new_block = block_x.clone()
                new_block[update_mask] = x0[update_mask]
                x[:, s:e] = new_block

                revealed = (mask_index & commit_index)[0]                # mask->committed this pass
                idx = torch.where(revealed)[0].cpu().numpy()
                for loc in idx:
                    if reveal_pass[g0 + loc] < 0:
                        reveal_pass[g0 + loc] = i

                no_change = not bool(changed.any())
                if bool((max_probs >= break_threshold).all()) or no_change:
                    break
                soft_cond = (x[0, s:e] != mask_id)
                block_soft = _build_block_soft_embeds(
                    x[:, s:e], max_probs, x0, soft_cond, emb, mask_id)
    return x, reveal_pass, pass0_conf, pass0_ent, block_id


def oracle_neighbor_conf(model, x_final, Lp, *, block_length, targets, mask_id, loo_batch):
    """Leave-one-out: reveal all neighbors to their final tokens, mask one target, read
    its top-1 confidence + whether argmax recovers the final token.

    targets: list of global gen indices (g) to probe. Returns {g: (conf, recovered_bool)}.
    """
    import torch
    from probe_runner.llada2_runner import _build_block_causal_mask, _embedding_layer

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    emb = _embedding_layer(model)
    res = {}
    # group targets by block so the forward length e is shared within a batch
    by_block = {}
    for g in targets:
        by_block.setdefault(g // block_length, []).append(g)

    with torch.no_grad():
        for nb, gs in by_block.items():
            s, e = Lp + nb * block_length, Lp + (nb + 1) * block_length
            base = x_final[:, :e].clone()                      # [1, e] oracle prefix+block
            attn = _build_block_causal_mask(e, Lp, block_length, dtype, device)
            pos = torch.arange(e, device=device).unsqueeze(0)
            for c0 in range(0, len(gs), loo_batch):
                chunk = gs[c0:c0 + loo_batch]
                ids = base.repeat(len(chunk), 1)               # [bs, e]
                abs_pos = []
                for k, g in enumerate(chunk):
                    ap = Lp + g                                # absolute seq position
                    abs_pos.append(ap)
                    ids[k, ap] = mask_id
                # this model asserts the attn mask / position_ids batch-match the inputs
                # (no broadcast over batch), so expand both to the chunk size.
                bs = len(chunk)
                out = model(inputs_embeds=emb(ids),
                            attention_mask=attn.expand(bs, -1, -1, -1),
                            position_ids=pos.expand(bs, -1), use_cache=False,
                            output_hidden_states=False, return_dict=True)
                for k, g in enumerate(chunk):
                    lg = out.logits[k, abs_pos[k], :].float()
                    p = torch.softmax(lg, dim=-1)
                    conf, arg = float(p.max()), int(p.argmax())
                    recovered = (arg == int(x_final[0, abs_pos[k]]))
                    res[g] = (conf, recovered)
    return res


# ---- driver -----------------------------------------------------------------

def run(args) -> None:
    import torch
    from probe_runner.llada2_runner import load_llada2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_f = (out_dir / "per_position.jsonl").open("w")

    model, tokenizer = load_llada2(args.model_path, attn_implementation="sdpa",
                                   t3dmax_root=args.t3dmax_root, device=args.device)
    mask_id = MASK_ID
    eos_id = args.eos_id if args.eos_id is not None else (tokenizer.eos_token_id or DEFAULT_EOS)
    print(f"[E1] mask_id={mask_id} eos_id={eos_id} gen_length={args.gen_length} "
          f"block_length={args.block_length} hi={args.hi}")

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test").select(range(args.gsm8k_n))

    counts = Counter()
    # per-class accumulators for distributions / AUC
    easy_reveal, easy_pass0c, easy_oraclec = [], [], []
    hard_pass0c, hard_oraclec = [], []
    easy_ent, hard_ent = [], []
    n_generated_total = 0

    def _format(q):
        msg = [{"role": "user", "content": q}]
        pp = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        return tokenizer(pp, return_tensors="pt")["input_ids"].to(args.device)

    for qi, ex in enumerate(ds):
        prompt = _format(ex["question"])
        Lp = int(prompt.shape[1])
        x, reveal_pass, pass0_conf, pass0_ent, block_id = decode_and_record(
            model, prompt, mask_id=mask_id, eos_id=eos_id,
            gen_length=args.gen_length, block_length=args.block_length)

        # truncate the accounting at EOS: positions after the first EOS are padding, not work
        gen = x[0, Lp:]
        eos_hits = torch.where(gen == eos_id)[0]
        gen_end = int(eos_hits[0]) + 1 if len(eos_hits) else args.gen_length
        n_generated_total += gen_end

        # candidates needing the oracle probe: reveal_pass not in {0,1} within [0,gen_end)
        targets = [g for g in range(gen_end) if reveal_pass[g] not in (0, 1)]
        oracle = oracle_neighbor_conf(
            model, x, Lp, block_length=args.block_length, targets=targets,
            mask_id=mask_id, loo_batch=args.loo_batch) if targets else {}

        for g in range(gen_end):
            rp = int(reveal_pass[g])
            oc, recov = oracle.get(g, (None, None))
            cls = classify(rp, oc, args.hi)
            counts[cls] += 1
            if cls == "easy_uncertain":
                easy_reveal.append(rp if rp >= 0 else args.block_length)
                easy_pass0c.append(float(pass0_conf[g]))
                easy_oraclec.append(float(oc))
                easy_ent.append(float(pass0_ent[g]))
            elif cls == "hard_tail":
                hard_pass0c.append(float(pass0_conf[g]))
                if oc is not None:
                    hard_oraclec.append(float(oc))
                hard_ent.append(float(pass0_ent[g]))
            pos_f.write(json.dumps({
                "problem": qi, "g": g, "block": int(block_id[g]), "reveal_pass": rp,
                "pass0_conf": _nan_to_none(pass0_conf[g]), "pass0_entropy": _nan_to_none(pass0_ent[g]),
                "oracle_neighbor_conf": oc, "oracle_recovered": recov, "class": cls,
            }) + "\n")
        pos_f.flush()
        if (qi + 1) % 5 == 0:
            print(f"[E1] {qi+1}/{len(ds)}  classes={dict(counts)}")

    pos_f.close()

    # ---- aggregate metrics ----
    gen_classified = sum(counts.values())
    easy = counts["easy_uncertain"]
    easy_frac = easy / max(gen_classified, 1)
    # cannibalization: of easy-uncertain, fraction the heavy would commit within <=2 passes
    near_term = sum(1 for r in easy_reveal if r <= 2)
    cannib_overlap = near_term / max(easy, 1)
    high_value = sum(1 for r in easy_reveal if r >= 4)
    high_value_frac = high_value / max(easy, 1)
    conf_gap = (float(np.mean(easy_oraclec)) - float(np.mean(easy_pass0c))) if easy else float("nan")
    # E2 teaser: can a cheap think-pass signal separate easy from hard?
    auc_conf = auc_mann_whitney([-c for c in easy_pass0c], [-c for c in hard_pass0c])  # lower conf -> easy?
    auc_ent = auc_mann_whitney(easy_ent, hard_ent)                                     # higher entropy -> easy?

    gate_pass = (easy_frac > args.gate_easy_frac) and (cannib_overlap < args.gate_overlap)

    summary = {
        "model_path": args.model_path, "n_problems": len(ds),
        "gen_length": args.gen_length, "block_length": args.block_length, "hi": args.hi,
        "n_generated_classified": gen_classified,
        "class_fractions": {k: counts[k] / max(gen_classified, 1)
                            for k in ("think", "redundant", "easy_uncertain", "hard_tail")},
        "class_counts": dict(counts),
        "easy_uncertain_frac": easy_frac,
        "cannibalization_overlap_reveal_le2": cannib_overlap,
        "high_value_frac_reveal_ge4": high_value_frac,
        "easy_reveal_pass_mean": float(np.mean(easy_reveal)) if easy else None,
        "easy_reveal_pass_hist": dict(Counter(easy_reveal)),
        "conf_gap_oracle_minus_pass0": conf_gap,
        "easy_pass0_conf_mean": float(np.mean(easy_pass0c)) if easy else None,
        "easy_oracle_conf_mean": float(np.mean(easy_oraclec)) if easy else None,
        "e2_teaser_auc_pass0conf": auc_conf,
        "e2_teaser_auc_pass0entropy": auc_ent,
        "GATE_easy_frac_threshold": args.gate_easy_frac,
        "GATE_overlap_threshold": args.gate_overlap,
        "GATE_PASS": bool(gate_pass),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "E1 — easy-uncertain mass / cannibalization",
        f"  problems={len(ds)}  generated(classified)={gen_classified}",
        "  class fractions:",
    ]
    for k in ("think", "redundant", "easy_uncertain", "hard_tail"):
        lines.append(f"    {k:<14s} {counts[k]/max(gen_classified,1):6.1%}  (n={counts[k]})")
    lines += [
        f"  easy_uncertain_frac          = {easy_frac:.1%}",
        f"  cannibalization_overlap(<=2) = {cannib_overlap:.1%}   (lower is better)",
        f"  high_value_frac (reveal>=4)  = {high_value_frac:.1%}   (heavy spent real rounds)",
        f"  easy reveal_pass mean        = {summary['easy_reveal_pass_mean']}",
        f"  conf gap (oracle - pass0)    = {conf_gap:.3f}   (large => real opportunity)",
        f"  E2 teaser AUC pass0_conf     = {auc_conf:.3f}",
        f"  E2 teaser AUC pass0_entropy  = {auc_ent:.3f}",
        "",
        f"  GATE: easy_frac>{args.gate_easy_frac} AND overlap<{args.gate_overlap}  =>  "
        f"{'PASS — train the lightweight' if gate_pass else 'FAIL — re-harvesting; ship Mechanism-B truncated-heavy only'}",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "report.txt").write_text(report + "\n")
    print(f"[E1] wrote per_position.jsonl, summary.json, report.txt -> {out_dir}")


def _nan_to_none(v):
    v = float(v)
    return None if v != v else v


# ---- selftest (torch-free) --------------------------------------------------

def _selftest() -> None:
    assert classify(0, None, 0.9) == "think"
    assert classify(1, None, 0.9) == "redundant"
    assert classify(5, 0.95, 0.9) == "easy_uncertain"
    assert classify(5, 0.40, 0.9) == "hard_tail"
    assert classify(-1, 0.10, 0.9) == "hard_tail"          # never committed, hard even w/ neighbors
    assert classify(3, float("nan"), 0.9) == "hard_tail"   # unprobed candidate
    # AUC: perfectly separable -> 1.0; reversed -> 0.0; identical -> 0.5
    assert abs(auc_mann_whitney([3, 4, 5], [0, 1, 2]) - 1.0) < 1e-9
    assert abs(auc_mann_whitney([0, 1, 2], [3, 4, 5]) - 0.0) < 1e-9
    assert abs(auc_mann_whitney([1, 2, 3], [1, 2, 3]) - 0.5) < 1e-9
    assert _nan_to_none(float("nan")) is None and _nan_to_none(0.5) == 0.5
    print("exp_e1 selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="E1: easy-uncertain mass / cannibalization (the decider)")
    ap.add_argument("--model_path", help="DMax-Math-16B weight dir")
    ap.add_argument("--t3dmax_root", default=None)
    ap.add_argument("--gsm8k_n", type=int, default=100)
    ap.add_argument("--gen_length", type=int, default=512)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--hi", type=float, default=0.9,
                    help="oracle-neighbor confidence above which a position is 'easy-given-neighbors'")
    ap.add_argument("--loo_batch", type=int, default=8, help="leave-one-out probe batch size")
    ap.add_argument("--gate_easy_frac", type=float, default=0.25)
    ap.add_argument("--gate_overlap", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eos_id", type=int, default=None)
    ap.add_argument("--out_dir", default="fork_bounded_surrogate/results/E1")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.model_path:
        raise SystemExit("--model_path is required (or pass --selftest).")
    run(args)


if __name__ == "__main__":
    main()
