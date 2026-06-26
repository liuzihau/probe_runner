"""B2 — Draft-then-correct economics: does a drafter beat partial-depth ALONE?

The decider for the drafter direction. partial-depth (Track A) is already a free ~1.2x
(saves ~21% of full-decode cost at thr 0.5). A trained drafter is only worth it if
`drafter + heavy` beats `partial-depth alone` at iso-accuracy.

We measure a real draft-then-correct decode (LLaDA native global-threshold rule):
  per round, per block:
    DRAFT   : propose tokens for the masked positions (cost = draft_cut/depth)
    FILL    : place the drafter's top `draft_fill_frac` (by draft confidence) as hard tokens
    CORRECT : one full heavy forward (cost = (depth-corrector_cut)/depth)
    COMMIT  : a masked position is committed if the heavy is confident (>=thr) AND, if it was
              a drafted position, the heavy agrees with the draft (agreement-accept);
              non-drafted masked positions follow the plain confidence rule.
  Cost is counted in full-pass-equivalents (draft + correct per round).

Three drafter modes:
  * none   : no drafter — plain heavy threshold decode (the in-experiment baseline).
  * trunc  : drafter = bottom `draft_cut` layers + shared norm/head (zero-training, weak by
             construction — the logit-lens drafter; the realistic free option).
  * oracle : drafter = the FINAL decoded tokens (perfect drafts) — the UPPER BOUND. If even
             oracle drafting can't beat partial-depth-alone cost, the direction is dead.

DECISION: compare `cost(mode)` at iso-accuracy. The drafter must save MORE than partial-depth
alone (~21% vs full at thr 0.5; pass `--partial_depth_saving 0.21`). If oracle can't, stop.

Usage
-----
    python -m probe_runner.exp_b2_selfspec \
        --model_path $DMAX_MATH_PATH --t3dmax_root $T3DMAX_ROOT \
        --modes none trunc oracle --draft_cut 9 --commit_threshold 0.5 \
        --gsm8k_n 100 --out_dir fork_bounded_surrogate/results/B2_thr0.5
    python -m probe_runner.exp_b2_selfspec --selftest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc
from probe_runner.eval_decode_compute import is_correct, gold_number, _last_number

MASK_ID = tc.MASK_ID
DEFAULT_EOS = 156892


def _model_parts(model):
    from probe_runner.llada2_runner import _embedding_layer
    base = getattr(model, "model", model)
    lm_head = getattr(model, "lm_head", None) or model.get_output_embeddings()
    return _embedding_layer(model), base.layers, base.norm, base.rotary_emb, lm_head


def _run_layers(h, layers, *, attn, pos, rope):
    for layer in layers:
        h = layer(h, attention_mask=attn, position_ids=pos, past_key_value=None,
                  output_attentions=False, output_router_logits=False, use_cache=False,
                  position_embeddings=rope)[0]
    return h


def _block_logits(emb, layers, norm, lm_head, ids, *, s, e, attn, pos, rope, upto=None):
    """Logits over [s:e]. upto=None -> all layers (full); upto=k -> bottom k layers (trunc)."""
    h = emb(ids[:, :e])
    h = _run_layers(h, layers[:upto] if upto is not None else layers, attn=attn, pos=pos, rope=rope)
    return lm_head(norm(h)[:, s:e, :])


def decode_b2(model, prompt, *, mode, mask_id, eos_id, gen_length, block_length, depth,
              draft_cut, corrector_cut, accept_thr, draft_fill_frac, oracle_final=None,
              max_rounds=None):
    """Draft-then-correct decode. Returns (x, cost, rounds)."""
    import torch
    import torch.nn.functional as F
    from probe_runner.llada2_runner import _build_block_causal_mask

    emb, layers, norm, rotary, lm_head = _model_parts(model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prompt = prompt.to(device); Lp = int(prompt.shape[1])
    if max_rounds is None:
        max_rounds = block_length
    x = F.pad(prompt, (0, gen_length), value=mask_id)
    draft_cost = draft_cut / depth
    correct_cost = (depth - corrector_cut) / depth
    cost = 0.0; rounds = 0
    num_blocks = gen_length // block_length

    with torch.no_grad():
        for nb in range(num_blocks):
            s, e = Lp + nb * block_length, Lp + (nb + 1) * block_length
            attn = _build_block_causal_mask(e, Lp, block_length, dtype, device)
            pos = torch.arange(e, device=device).unsqueeze(0)
            rope = rotary(emb(x[:, :e]), pos)
            for _ in range(max_rounds):
                blk = x[0, s:e]
                masked = (blk == mask_id)
                if not bool(masked.any()):
                    break
                rounds += 1
                # ---- DRAFT ----
                fill = torch.zeros(block_length, dtype=torch.bool, device=device)
                draft_tok = None
                if mode != "none":
                    if mode == "oracle":
                        draft_tok = oracle_final[s - Lp:e - Lp].to(device)   # oracle_final is gen-relative
                        draft_conf = torch.where(masked, torch.ones(block_length, device=device),
                                                 torch.zeros(block_length, device=device))
                    else:  # trunc
                        dl = _block_logits(emb, layers, norm, lm_head, x, s=s, e=e,
                                           attn=attn, pos=pos, rope=rope, upto=draft_cut)[0]
                        dp = torch.softmax(dl.float(), -1)
                        draft_conf, draft_tok = dp.max(-1)
                        draft_conf = torch.where(masked, draft_conf, torch.zeros_like(draft_conf))
                    cost += draft_cost
                    # choose fill set = top frac of masked positions by draft confidence
                    n_fill = int(round(draft_fill_frac * int(masked.sum())))
                    if n_fill > 0:
                        order = torch.argsort(draft_conf, descending=True)
                        chosen = order[:n_fill]
                        fill[chosen] = True
                        fill = fill & masked
                # ---- FILL + CORRECT ----
                x_in = x.clone()
                if mode != "none" and bool(fill.any()):
                    x_in[0, s:e][fill] = draft_tok[fill]
                cl = _block_logits(emb, layers, norm, lm_head, x_in, s=s, e=e,
                                   attn=attn, pos=pos, rope=rope, upto=(depth - corrector_cut)
                                   if corrector_cut else None)[0]
                cp = torch.softmax(cl.float(), -1)
                corr_conf, corr_tok = cp.max(-1)
                cost += correct_cost
                # ---- COMMIT ----
                accept = masked & (corr_conf >= accept_thr)
                if mode != "none":
                    agree_ok = (~fill) | (corr_tok == draft_tok)
                    accept = accept & agree_ok
                if not bool(accept.any()):
                    # guarantee progress: commit the single highest-confidence masked position
                    cand = torch.where(masked, corr_conf, torch.full_like(corr_conf, -1.0))
                    accept[int(cand.argmax())] = True
                new = blk.clone(); new[accept] = corr_tok[accept]
                x[0, s:e] = new
    return x, cost, rounds


def run(args) -> None:
    import torch
    from probe_runner.llada2_runner import load_llada2
    from datasets import load_dataset

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_llada2(args.model_path, attn_implementation="sdpa",
                                   t3dmax_root=args.t3dmax_root, device=args.device)
    _, layers, _, _, _ = _model_parts(model); depth = len(layers)
    mask_id = MASK_ID
    eos_id = args.eos_id if args.eos_id is not None else (tokenizer.eos_token_id or DEFAULT_EOS)
    print(f"[B2] depth={depth} modes={args.modes} draft_cut={args.draft_cut} "
          f"corrector_cut={args.corrector_cut} thr={args.commit_threshold} fill={args.draft_fill_frac}")

    ds = load_dataset("gsm8k", "main", split="test").select(range(args.gsm8k_n))
    agg = {m: {"correct": 0, "total": 0, "cost": 0.0, "rounds": 0} for m in args.modes}

    def _format(q):
        msg = [{"role": "user", "content": q}]
        p = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        return tokenizer(p, return_tensors="pt")["input_ids"].to(args.device)

    for qi, ex in enumerate(ds):
        prompt = _format(ex["question"]); Lp = int(prompt.shape[1])
        oracle_final = None
        if "oracle" in args.modes:
            xo, _, _ = decode_b2(model, prompt, mode="none", mask_id=mask_id, eos_id=eos_id,
                                 gen_length=args.gen_length, block_length=args.block_length, depth=depth,
                                 draft_cut=args.draft_cut, corrector_cut=0, accept_thr=args.commit_threshold,
                                 draft_fill_frac=args.draft_fill_frac)
            oracle_final = xo[0, Lp:Lp + args.gen_length].clone()
        for m in args.modes:
            x, cost, rnds = decode_b2(
                model, prompt, mode=m, mask_id=mask_id, eos_id=eos_id, gen_length=args.gen_length,
                block_length=args.block_length, depth=depth, draft_cut=args.draft_cut,
                corrector_cut=args.corrector_cut, accept_thr=args.commit_threshold,
                draft_fill_frac=args.draft_fill_frac, oracle_final=oracle_final)
            gen = x[0, Lp:]
            if bool((gen == eos_id).any()):
                gen = gen[: int(torch.where(gen == eos_id)[0][0])]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            ok = is_correct(text, ex["answer"])
            a = agg[m]; a["correct"] += int(ok); a["total"] += 1; a["cost"] += cost; a["rounds"] += rnds
        if (qi + 1) % 10 == 0:
            print(f"[B2] {qi+1}/{len(ds)} " + " ".join(
                f"{m}:acc={agg[m]['correct']/agg[m]['total']:.2f},cost={agg[m]['cost']/agg[m]['total']:.1f}"
                for m in args.modes))

    base = agg.get("none")
    base_cost = base["cost"] / max(base["total"], 1) if base else None
    summary = {"depth": depth, "draft_cut": args.draft_cut, "corrector_cut": args.corrector_cut,
               "commit_threshold": args.commit_threshold, "draft_fill_frac": args.draft_fill_frac,
               "partial_depth_saving_bar": args.partial_depth_saving, "per_mode": {}}
    lines = [f"B2 — draft-then-correct (thr={args.commit_threshold}, draft_cut=L{args.draft_cut}, "
             f"corrector_cut=L{args.corrector_cut}, fill={args.draft_fill_frac})",
             f"  partial-depth-alone saves {args.partial_depth_saving:.0%} of full — a drafter must beat that.",
             f"  {'mode':<8s} {'acc':>7s} {'mean_cost':>10s} {'mean_rounds':>12s} {'saving_vs_none':>15s}"]
    for m in args.modes:
        a = agg[m]; n = max(a["total"], 1)
        acc = a["correct"] / n; mc = a["cost"] / n; mr = a["rounds"] / n
        saving = (1 - mc / base_cost) if (base_cost and m != "none") else 0.0
        summary["per_mode"][m] = {"acc": acc, "mean_cost": mc, "mean_rounds": mr,
                                  "saving_vs_none": saving}
        lines.append(f"  {m:<8s} {acc:7.1%} {mc:10.2f} {mr:12.1f} {saving:14.1%}")
    # verdict
    best_draft = max((summary["per_mode"][m]["saving_vs_none"] for m in args.modes if m != "none"), default=0.0)
    oracle_saving = summary["per_mode"].get("oracle", {}).get("saving_vs_none", None)
    verdict = ("oracle (upper bound) cannot beat partial-depth-alone => DRAFTER DEAD, ship partial-depth"
               if (oracle_saving is not None and oracle_saving <= args.partial_depth_saving)
               else "a drafter mode beats the partial-depth bar — worth pursuing (check iso-accuracy)"
               if best_draft > args.partial_depth_saving else
               "inconclusive vs bar — inspect per-mode")
    lines += ["", f"  VERDICT: {verdict}"]
    summary["verdict"] = verdict
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report = "\n".join(lines); print("\n" + report)
    (out_dir / "report.txt").write_text(report + "\n")
    print(f"[B2] wrote summary.json, report.txt -> {out_dir}")


def _selftest() -> None:
    assert is_correct("answer 42", "#### 42")
    # cost arithmetic: draft L9 + full correct over 1 round = 0.45 + 1.0
    assert abs(9 / 20 + (20 - 0) / 20 - 1.45) < 1e-9
    print("exp_b2 selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="B2: draft-then-correct economics vs partial-depth")
    ap.add_argument("--model_path")
    ap.add_argument("--t3dmax_root", default=None)
    ap.add_argument("--modes", nargs="+", default=["none", "trunc", "oracle"])
    ap.add_argument("--draft_cut", type=int, default=9, help="drafter = bottom this-many layers + head")
    ap.add_argument("--corrector_cut", type=int, default=0,
                    help="0=full corrector. >0 = SHALLOW corrector (bottom depth-cut layers + head); "
                         "NOT the cache-reuse partial-depth (that needs A1). Leave 0 for the decisive test.")
    ap.add_argument("--commit_threshold", type=float, default=0.5)
    ap.add_argument("--draft_fill_frac", type=float, default=1.0)
    ap.add_argument("--partial_depth_saving", type=float, default=0.21,
                    help="cost fraction partial-depth-alone saves vs full (the bar to beat); thr0.5≈0.21")
    ap.add_argument("--gsm8k_n", type=int, default=100)
    ap.add_argument("--gen_length", type=int, default=512)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eos_id", type=int, default=None)
    ap.add_argument("--out_dir", default="fork_bounded_surrogate/results/B2")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if not args.model_path:
        raise SystemExit("--model_path is required (or pass --selftest).")
    run(args)


if __name__ == "__main__":
    main()
