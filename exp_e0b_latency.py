"""E0b — realized per-pass latency of a partial-depth (layer-skip) talk pass.

E0 showed the "1.3x" partial-depth speedup is a `pass_cost` PROJECTION (talk pass counted
at (depth-cut)/depth) that was never realized — the probe computes all layers then overwrites
the low ones. This microbench measures the *realized* per-pass wall-clock factor of actually
skipping layers 0..cut-1, to see how far it lands from the 0.55 projection.

It times two real forwards at several sequence lengths e, apples-to-apples (same mask, same
shared rope, same final norm + lm_head over the scored block), differing ONLY in:
  (a) FULL : embed(token_ids) -> layers[0:depth] -> norm -> lm_head[:, s:e]
  (b) SKIP : H (cached layer cut-1 hidden) -> layers[cut:depth] -> norm -> lm_head[:, s:e]

`time(b)/time(a)` is the realized per-talk-pass factor. The skip is along DEPTH only: all e
positions still flow through layers[cut:]. (b) is the OPTIMISTIC bound for the realized
L9M4 — it treats layers 0..cut-1 as entirely free (i.e. it is the cost of the aggressive
"embed -> layer cut" Design B; the faithful Design A that recomputes revealed tokens through
0..cut-1 costs slightly more.)

Usage
-----
    python -m probe_runner.exp_e0b_latency \
        --model_path $DMAX_MATH_PATH --t3dmax_root $T3DMAX_ROOT \
        --cuts 9 6 12 --block_length 32 --warmup 10 --iters 30 \
        --out_dir fork_bounded_surrogate/results/E0b
    python -m probe_runner.exp_e0b_latency --selftest
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from probe_runner.plots import _trajectory_common as tc

MASK_ID = tc.MASK_ID
DEFAULT_EOS = 156892


def _median(xs):
    return statistics.median(xs)


def _model_parts(model):
    """(embed, layers ModuleList, norm, rotary, lm_head) across the LLaDA2 wrapper."""
    from probe_runner.llada2_runner import _embedding_layer
    base = getattr(model, "model", model)
    embed = _embedding_layer(model)
    lm_head = getattr(model, "lm_head", None) or model.get_output_embeddings()
    return embed, base.layers, base.norm, base.rotary_emb, lm_head


def _run_layers(h, layers, *, attn, pos, rope):
    """Push hidden h through the given decoder layers (faithful to model.forward's loop)."""
    for layer in layers:
        h = layer(h, attention_mask=attn, position_ids=pos, past_key_value=None,
                  output_attentions=False, output_router_logits=False, use_cache=False,
                  position_embeddings=rope)[0]
    return h


def time_fn(fn, warmup, iters):
    import torch
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return _median(ts), min(ts), max(ts)


def run(args) -> None:
    import torch
    from probe_runner.llada2_runner import load_llada2, _build_block_causal_mask

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_llada2(args.model_path, attn_implementation="sdpa",
                                   t3dmax_root=args.t3dmax_root, device=args.device)
    embed, layers, norm, rotary, lm_head = _model_parts(model)
    depth = len(layers)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    B = args.block_length
    print(f"[E0b] depth={depth} block_length={B} cuts={args.cuts} "
          f"warmup={args.warmup} iters={args.iters}")

    # one real prompt -> a realistic decode state (prompt + masks) for each e
    from datasets import load_dataset
    q = load_dataset("gsm8k", "main", split="test")[0]["question"]
    msg = [{"role": "user", "content": q}]
    p = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    prompt = tokenizer(p, return_tensors="pt")["input_ids"].to(device)
    Lp = int(prompt.shape[1])
    e_list = [Lp + n * B for n in args.blocks]      # sequence length at block boundaries
    print(f"[E0b] Lp={Lp}  e_list={e_list}")

    results = []
    for e in e_list:
        x = torch.nn.functional.pad(prompt, (0, e - Lp), value=MASK_ID)   # [1, e]
        attn = _build_block_causal_mask(e, Lp, B, dtype, device)          # [1,1,e,e]
        pos = torch.arange(e, device=device).unsqueeze(0)                 # [1, e]
        s = e - B                                                         # scored block [s:e]
        with torch.no_grad():
            emb_x = embed(x[:, :e])                                       # [1, e, D]
            rope = rotary(emb_x, pos)                                     # shared (cos, sin)

        def full_pass():
            h = _run_layers(emb_x, layers, attn=attn, pos=pos, rope=rope)
            return lm_head(norm(h)[:, s:e, :])

        ta, ta_lo, ta_hi = time_fn(full_pass, args.warmup, args.iters)

        for cut in args.cuts:
            with torch.no_grad():
                H = _run_layers(emb_x, layers[:cut], attn=attn, pos=pos, rope=rope)  # layer cut-1 out

            def skip_pass(_H=H, _cut=cut):
                h = _run_layers(_H, layers[_cut:], attn=attn, pos=pos, rope=rope)
                return lm_head(norm(h)[:, s:e, :])

            tb, tb_lo, tb_hi = time_fn(skip_pass, args.warmup, args.iters)
            proj = (depth - cut) / depth
            row = {"e": e, "cut": cut, "full_ms": ta * 1e3, "skip_ms": tb * 1e3,
                   "realized_factor": tb / ta, "projected_factor": proj,
                   "overhead_vs_projection": (tb / ta) / proj}
            results.append(row)
            print(f"  e={e:5d} cut=L{cut:<2d}  full={ta*1e3:7.2f}ms  skip={tb*1e3:7.2f}ms  "
                  f"realized={tb/ta:.3f}  projected={proj:.3f}  "
                  f"(realized is {tb/ta/proj:.2f}x the projection)")
        del emb_x, rope
        torch.cuda.empty_cache()

    summary = {"depth": depth, "block_length": B, "Lp": Lp, "model_path": args.model_path,
               "warmup": args.warmup, "iters": args.iters, "rows": results}
    (out_dir / "latency.json").write_text(json.dumps(summary, indent=2))

    # headline per cut, averaged over e
    lines = ["E0b — realized per-talk-pass latency factor (time(skip)/time(full))",
             f"  depth={depth} block={B}  (skip is depth-only; all e positions still run layers[cut:])"]
    for cut in args.cuts:
        rs = [r for r in results if r["cut"] == cut]
        mean_rf = sum(r["realized_factor"] for r in rs) / len(rs)
        proj = (depth - cut) / depth
        lines.append(f"  L{cut}: realized≈{mean_rf:.3f} vs projected {proj:.3f}  "
                     f"=> realized saving {1-mean_rf:.0%} (projection claimed {1-proj:.0%})")
    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "report.txt").write_text(report + "\n")
    print(f"[E0b] wrote latency.json, report.txt -> {out_dir}")


def _selftest() -> None:
    assert _median([3, 1, 2]) == 2
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5
    # realized/projected arithmetic
    rf, proj = 0.72, (20 - 9) / 20
    assert abs(proj - 0.55) < 1e-9 and abs(rf / proj - 1.309) < 1e-2
    print("exp_e0b selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="E0b: realized per-pass latency of a layer-skip talk pass")
    ap.add_argument("--model_path")
    ap.add_argument("--t3dmax_root", default=None)
    ap.add_argument("--cuts", nargs="+", type=int, default=[9, 6, 12])
    ap.add_argument("--blocks", nargs="+", type=int, default=[1, 4, 8, 16],
                    help="block boundaries to time at: e = Lp + n*block_length for each n")
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="fork_bounded_surrogate/results/E0b")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if not args.model_path:
        raise SystemExit("--model_path is required (or pass --selftest).")
    run(args)


if __name__ == "__main__":
    main()
