"""E0 — re-establish the L9 partial-depth baseline, WITH per-problem logging.

`brainstorm.md` §3 claims partial-depth refresh (L9M4) gets ~1.4x at 82-84% GSM8K,
but the support log is lost (§8.4). This re-measures the accuracy-vs-compute frontier
against the DMax full-pass baseline and — unlike `eval_decode_compute.run_eval`, which
only aggregates — writes an auditable per-problem record so the number can never go
missing again.

It is a thin logging wrapper: the decode (`decode_with_refresh`), scoring (`is_correct`),
compute accounting (`pass_cost`), config parsing, and report/plot are all imported from
`eval_decode_compute` verbatim. The `full` config IS the DMax `decode_uniform` baseline
(cut=0 => every pass full-depth).

Usage
-----
    python -m probe_runner.exp_e0_baseline \
        --model_path $DMAX_MATH_PATH --t3dmax_root $T3DMAX_ROOT \
        --configs full L9M4 L9 L6M4 L12M4 \
        --gsm8k_n 200 --gen_length 512 --out_dir fork_bounded_surrogate/results/E0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from probe_runner.plots import _trajectory_common as tc
from probe_runner.eval_decode_compute import (
    decode_with_refresh, _decoder_layers, is_correct, gold_number, _last_number,
    distinct_n, text_similarity, parse_config, format_report, plot,
)

DEFAULT_EOS = 156892  # LLaDA-2.0 family; tokenizer.eos is sometimes wrong (see eval_decode_compute)


def load_gsm8k(n: int):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test").select(range(n))
    return [(p["question"], p["answer"]) for p in ds]


def run(args) -> None:
    import torch
    from probe_runner.llada2_runner import load_llada2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "records.jsonl"
    rec_f = rec_path.open("w")

    model, tokenizer = load_llada2(
        args.model_path, attn_implementation="sdpa",
        t3dmax_root=args.t3dmax_root, device=args.device)
    layers = _decoder_layers(model)
    depth = len(layers)
    mask_id = tc.MASK_ID
    eos_id = args.eos_id if args.eos_id is not None else (tokenizer.eos_token_id or DEFAULT_EOS)
    cfgs = [parse_config(c) for c in args.configs]
    print(f"[E0] depth={depth} mask_id={mask_id} eos_id={eos_id} "
          f"gen_length={args.gen_length} block_length={args.block_length}")

    def _format(q):
        msg = [{"role": "user", "content": q}]
        p = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        return tokenizer(p, return_tensors="pt")["input_ids"].to(args.device)

    problems = load_gsm8k(args.gsm8k_n)[: args.gsm8k_n]
    res = {lbl: {"correct": 0, "total": 0, "cost": 0.0, "nfe": 0, "truncated": 0,
                 "distinct4": 0.0, "sim_full": 0.0, "cut": L, "rethink": M}
           for lbl, L, M in cfgs}

    t0 = time.time()
    for qi, (q, gold) in enumerate(problems):
        prompt = _format(q)
        Lp = int(prompt.shape[1])
        texts = {}
        for lbl, L, M in cfgs:
            x, cost, nfe = decode_with_refresh(
                model, layers, prompt, cut_layer=L, rethink_every=M, mask_id=mask_id,
                eos_id=eos_id, gen_length=args.gen_length, block_length=args.block_length,
                commit_threshold=args.commit_threshold, break_threshold=args.break_threshold)
            gen = x[0, Lp:]
            hit_eos = bool((gen == eos_id).any())
            if hit_eos:
                gen = gen[: int(torch.where(gen == eos_id)[0][0])]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            texts[lbl] = text
            ok = is_correct(text, gold)
            n_mask = int((x[0, Lp:] == mask_id).sum())
            r = res[lbl]
            r["correct"] += int(ok); r["total"] += 1
            r["cost"] += cost; r["nfe"] += nfe
            r["truncated"] += int(not hit_eos)
            r["distinct4"] += distinct_n(text, 4)
            rec_f.write(json.dumps({
                "problem": qi, "config": lbl, "cut": L, "rethink": (M if M < 10**8 else None),
                "correct": bool(ok), "pred": _last_number(text), "gold": gold_number(gold),
                "cost": float(cost), "nfe": int(nfe), "hit_eos": hit_eos,
                "undecoded_masks": n_mask, "gen_tokens": int((x[0, Lp:] != mask_id).sum()),
                "text": text[:2000],
            }) + "\n")
        rec_f.flush()
        if "full" in texts:
            for lbl in texts:
                res[lbl]["sim_full"] += text_similarity(texts[lbl], texts["full"])
        if (qi + 1) % 10 == 0:
            acc = {l: res[l]["correct"] / max(res[l]["total"], 1) for l in res}
            print(f"[E0] {qi+1}/{len(problems)}  {time.time()-t0:.0f}s  acc={acc}")

    rec_f.close()
    out = {"depth": depth, "per_config": res}
    report = format_report(out)
    print("\n" + report)
    (out_dir / "report.txt").write_text(report + "\n")

    summary = {"depth": depth, "n_problems": len(problems), "gen_length": args.gen_length,
               "block_length": args.block_length, "model_path": args.model_path, "per_config": {}}
    base = res.get("full")
    base_cost = (base["cost"] / max(base["total"], 1)) if base else None
    for lbl, r in res.items():
        n = max(r["total"], 1)
        mc = r["cost"] / n
        summary["per_config"][lbl] = {
            "cut": r["cut"], "rethink": (r["rethink"] if r["rethink"] < 10**8 else None),
            "acc": r["correct"] / n, "correct": r["correct"], "total": r["total"],
            "mean_cost_thinkpass_equiv": mc,
            "frac_of_full": (mc / base_cost) if base_cost else None,
            "speedup_vs_full": (base_cost / mc) if (base_cost and mc) else None,
            "no_eos": r["truncated"], "distinct4": r["distinct4"] / n, "sim_full": r["sim_full"] / n,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[E0] wrote {rec_path}, summary.json, report.txt")
    try:
        plot(out, out_dir, "E0")
    except Exception as exc:  # plotting is non-essential
        print(f"[E0] plot skipped ({type(exc).__name__}: {exc})")


def main() -> None:
    ap = argparse.ArgumentParser(description="E0: partial-depth refresh accuracy-vs-compute, logged")
    ap.add_argument("--model_path", required=True, help="DMax-Math-16B weight dir")
    ap.add_argument("--t3dmax_root", default=None, help="root with dFactory models on path")
    ap.add_argument("--configs", nargs="+", default=["full", "L9M4", "L9", "L6M4", "L12M4"])
    ap.add_argument("--gsm8k_n", type=int, default=200)
    ap.add_argument("--gen_length", type=int, default=512)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--commit_threshold", type=float, default=0.3,
                    help="DMax soft decode commit threshold (0.3=aggressive default; try 0.5/0.6 for quality)")
    ap.add_argument("--break_threshold", type=float, default=0.9,
                    help="DMax soft decode per-block stop threshold")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eos_id", type=int, default=None)
    ap.add_argument("--out_dir", default="fork_bounded_surrogate/results/E0")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
