"""A2 — Skip-variant accuracy: does the AGGRESSIVE layer-skip (Design B) hold accuracy?

E0/E0b established that partial-depth (reuse layers 0..cut-1) is a real ~1.2x. There are two ways
to realize it; they have the SAME speed (E0b) but DIFFERENT accuracy:

  * Design A (faithful, = what E0 measured): on a talk pass, revealed (newly-committed) positions
    are RECOMPUTED through all of layers 0..cut-1; settled positions reuse cache.
  * Design B (aggressive): revealed positions SKIP layers 0..cut-1 entirely — their committed-token
    EMBEDDING is injected directly at the cut layer. Cheapest, but the embedding is off-distribution
    for the cut layer and neighbors see stale low-layer KV.

A2 runs Design B and scores GSM8K, to compare against E0's Design-A accuracy at the same threshold.
If Design B holds accuracy => take the cheaper path. If it tanks => Design A is required.

Reuses the scoring / config / cost cores from eval_decode_compute; only the talk-pass hook differs.

Usage
-----
    python -m probe_runner.exp_a2_skip_variant \
        --model_path $DMAX_MATH_PATH --t3dmax_root $T3DMAX_ROOT \
        --cut 9 --commit_threshold 0.5 --gsm8k_n 200 \
        --out_dir fork_bounded_surrogate/results/A2_thr0.5
    python -m probe_runner.exp_a2_skip_variant --selftest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe_runner.plots import _trajectory_common as tc
from probe_runner.eval_decode_compute import (
    is_correct, gold_number, _last_number, distinct_n, pass_cost, _decoder_layers)

DEFAULT_EOS = 156892


def _mk_hook_designB(idx, cache, state, emb_layer):
    """Talk-pass hook: layers idx<cut reuse cache for everyone; at idx==cut-1 the REVEALED
    positions are replaced by their committed-token EMBEDDING (injected at the cut layer)."""
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if state["record"]:
            cache[idx] = h.detach()
            return None
        if state["cut"] is not None and idx < state["cut"]:
            h_new = cache[idx].clone()
            if idx == state["cut"] - 1 and state["reveal_emb"] is not None:
                rev = state["reveal"]
                h_new[:, rev] = state["reveal_emb"]            # embed(committed) at cut-layer input
            return (h_new,) + out[1:] if isinstance(out, tuple) else h_new
        return None
    return hook


def decode_designB(model, layers, prompt, *, cut_layer, rethink_every, mask_id, eos_id=None,
                   gen_length=512, block_length=32, commit_threshold=0.5, break_threshold=0.9):
    """DMax soft decode with the Design-B aggressive skip on talk passes. Returns (x, cost, nfe)."""
    import torch
    import torch.nn.functional as F
    from probe_runner.llada2_runner import (
        _build_block_causal_mask, _commit_uniform, _build_block_soft_embeds, _embedding_layer)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prompt = prompt.to(device); Lp = int(prompt.shape[1]); depth = len(layers)
    emb = _embedding_layer(model)
    cache: dict = {}
    state = {"cut": None, "record": False, "reveal": None, "reveal_emb": None}
    handles = [layer.register_forward_hook(_mk_hook_designB(i, cache, state, emb))
               for i, layer in enumerate(layers)]
    x = F.pad(prompt, (0, gen_length), value=mask_id)
    nfe, cost = 0, 0.0
    num_blocks = gen_length // block_length
    try:
        with torch.no_grad():
            for nb in range(num_blocks):
                s, e = Lp + nb * block_length, Lp + (nb + 1) * block_length
                attn = _build_block_causal_mask(e, Lp, block_length, dtype, device)
                pos = torch.arange(e, device=device).unsqueeze(0)
                block_soft = None; think_tokens = None
                for i in range(block_length):
                    is_think = (i == 0) or (cut_layer <= 0) or (i % rethink_every == 0)
                    inputs_embeds = emb(x[:, :e])
                    if block_soft is not None:
                        inputs_embeds[:, s:e, :] = block_soft
                    if is_think:
                        state["record"], state["cut"], state["reveal"], state["reveal_emb"] = True, None, None, None
                    else:
                        changed = (x[0, s:e] != think_tokens) & (x[0, s:e] != mask_id)
                        reveal = torch.zeros(e, dtype=torch.bool, device=device)
                        rev_idx = s + torch.where(changed)[0]
                        reveal[rev_idx] = True
                        state["record"], state["cut"], state["reveal"] = False, cut_layer, reveal
                        # committed-token embedding at the revealed absolute positions
                        state["reveal_emb"] = emb(x[:, rev_idx]).to(dtype) if rev_idx.numel() else None
                    out = model(inputs_embeds=inputs_embeds, attention_mask=attn, position_ids=pos,
                                use_cache=False, output_hidden_states=False, return_dict=True)
                    if is_think:
                        state["record"] = False; think_tokens = x[0, s:e].clone()
                    nfe += 1; cost += pass_cost(is_think, cut_layer, depth)
                    block_logits = out.logits[:, s:e, :]; block_x = x[:, s:e]
                    mask_index = (block_x == mask_id)
                    x0, commit_index, max_probs = _commit_uniform(block_logits, block_x, mask_id, commit_threshold)
                    update_mask = commit_index | (~mask_index)
                    changed_m = update_mask & (x0 != block_x)
                    new_block = block_x.clone(); new_block[update_mask] = x0[update_mask]
                    x[:, s:e] = new_block
                    if not bool(changed_m.any()) or bool((max_probs >= break_threshold).all()):
                        break
                    soft_cond = (x[0, s:e] != mask_id)
                    block_soft = _build_block_soft_embeds(x[:, s:e], max_probs, x0, soft_cond, emb, mask_id)
    finally:
        for h in handles:
            h.remove()
    return x, cost, nfe


def run(args) -> None:
    import torch
    from probe_runner.llada2_runner import load_llada2
    from datasets import load_dataset

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rec = (out_dir / "records.jsonl").open("w")
    model, tokenizer = load_llada2(args.model_path, attn_implementation="sdpa",
                                   t3dmax_root=args.t3dmax_root, device=args.device)
    layers = _decoder_layers(model); depth = len(layers); mask_id = tc.MASK_ID
    eos_id = args.eos_id if args.eos_id is not None else (tokenizer.eos_token_id or DEFAULT_EOS)
    M = args.rethink_every if args.rethink_every > 0 else 10 ** 9
    print(f"[A2] Design-B skip  cut=L{args.cut} rethink={M if M<10**8 else '∞'} thr={args.commit_threshold} depth={depth}")

    ds = load_dataset("gsm8k", "main", split="test").select(range(args.gsm8k_n))
    correct = total = 0; cost_sum = nfe_sum = 0.0; trunc = 0
    for qi, ex in enumerate(ds):
        msg = [{"role": "user", "content": ex["question"]}]
        p = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        prompt = tokenizer(p, return_tensors="pt")["input_ids"].to(args.device)
        Lp = int(prompt.shape[1])
        x, cost, nfe = decode_designB(model, layers, prompt, cut_layer=args.cut, rethink_every=M,
                                      mask_id=mask_id, eos_id=eos_id, gen_length=args.gen_length,
                                      block_length=args.block_length, commit_threshold=args.commit_threshold)
        gen = x[0, Lp:]
        hit = bool((gen == eos_id).any())
        if hit:
            gen = gen[: int(torch.where(gen == eos_id)[0][0])]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        ok = is_correct(text, ex["answer"])
        correct += int(ok); total += 1; cost_sum += cost; nfe_sum += nfe; trunc += int(not hit)
        rec.write(json.dumps({"problem": qi, "correct": bool(ok), "pred": _last_number(text),
                              "gold": gold_number(ex["answer"]), "cost": cost, "nfe": nfe,
                              "hit_eos": hit, "text": text[:1500]}) + "\n")
        if (qi + 1) % 20 == 0:
            print(f"[A2] {qi+1}/{len(ds)} acc={correct/total:.3f}")
    rec.close()
    acc = correct / max(total, 1)
    summary = {"variant": "design_B_aggressive_skip", "cut": args.cut, "rethink": (M if M < 10**8 else None),
               "commit_threshold": args.commit_threshold, "acc": acc, "correct": correct, "total": total,
               "mean_cost": cost_sum / max(total, 1), "mean_nfe": nfe_sum / max(total, 1), "no_eos": trunc}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report = (f"A2 — Design-B aggressive skip (embed->cut layer)\n"
              f"  cut=L{args.cut} thr={args.commit_threshold}  acc={acc:.1%} ({correct}/{total})  "
              f"mean_cost={cost_sum/max(total,1):.2f}  mean_nfe={nfe_sum/max(total,1):.1f}  no-EOS={trunc}/{total}\n"
              f"  COMPARE to E0 Design-A (faithful) at the same thr/cut: if acc holds => take Design B (cheaper).")
    print("\n" + report); (out_dir / "report.txt").write_text(report + "\n")
    print(f"[A2] wrote records.jsonl, summary.json, report.txt -> {out_dir}")


def _selftest() -> None:
    assert pass_cost(False, 9, 20) == 0.55 and pass_cost(True, 9, 20) == 1.0
    assert is_correct("the answer is 42", "#### 42")
    print("exp_a2 selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="A2: Design-B aggressive layer-skip accuracy")
    ap.add_argument("--model_path")
    ap.add_argument("--t3dmax_root", default=None)
    ap.add_argument("--cut", type=int, default=9)
    ap.add_argument("--rethink_every", type=int, default=0, help="0 => one think per block (∞)")
    ap.add_argument("--commit_threshold", type=float, default=0.5)
    ap.add_argument("--gsm8k_n", type=int, default=200)
    ap.add_argument("--gen_length", type=int, default=512)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eos_id", type=int, default=None)
    ap.add_argument("--out_dir", default="fork_bounded_surrogate/results/A2")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if not args.model_path:
        raise SystemExit("--model_path is required (or pass --selftest).")
    run(args)


if __name__ == "__main__":
    main()
