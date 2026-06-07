"""Layer-merge compression trial: collapse a redundant middle block into a few
trainable layers, keep the load-bearing ends frozen, and heal by fine-tuning.

Motivation (from the Stage-1 layer-subset search on DMax-Math-16B): the best
4-layer drafts ALWAYS keep layer 12 and layer 19, usually an early layer (1–6);
layers 7–11 and 13–18 are the redundant span. So the principled compressed model
keeps the load-bearing anchors CLEAN (frozen, untouched) and collapses the gaps:

    keep   [0..5]   (frozen, clean — early feature extraction)
    merge  [6..11]  → trainable layer(s)   (redundant flank 1)
    keep   [12]     (frozen, CLEAN — the critical middle layer)
    merge  [13..18] → trainable layer(s)   (redundant flank 2)
    keep   [19]     (frozen, clean — readout-facing)

`--keep 0-5,12,19 --n_merged_per_block 1` → 10 layers. Tighten by widening the gaps
(e.g. `--keep 0-3,12,19`) or raise n_merged_per_block; find WHERE accuracy breaks.

Training: full fine-tune ONLY the merged layers to reproduce the FULL model's
iteration-1 argmax on the still-masked tail (the same fidelity the search measured;
teacher targets come from `eval_layer_subset.precompute_blocks` run on the full
model BEFORE surgery). Prompt-disjoint train/val, save-best + early-stop.

Judge by GSM8K ACCURACY vs compute (`--eval_gsm8k`), not fidelity — Step 0 showed
tail divergence is on filler, so a modest-fidelity compressed model can still hold
task accuracy. Compute is real here: the model genuinely runs `new_depth` layers.

Needs the model + GPU. Pure helpers are unit-tested via `--selftest`; `--dry_run`
does the surgery + a forward sanity check and stops (verify before training).

Usage
-----
    # 0) verify the model surgery + a forward pass (no training)
    python -m probe_runner.merge_layers --model llada2-DMAX \
        --model_path T3-DMax/DMax-Math-16B-moe-merge/ \
        --keep 0-5,12,19 --n_merged_per_block 1 --dry_run
    # 1) train the 10-layer compressed model, then GSM8K accuracy-vs-compute
    python -m probe_runner.merge_layers --model llada2-DMAX \
        --model_path T3-DMax/DMax-Math-16B-moe-merge/ \
        --keep 0-5,12,19 --n_merged_per_block 1 \
        --n_train_samples 80 --n_val_samples 20 --epochs 8 --lr 1e-4 \
        --eval_gsm8k 100 --gen_length 512 --out_dir merged_k1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc
from probe_runner.eval_layer_subset import precompute_blocks


# ---- pure helpers (unit-testable) -------------------------------------------

def partition(a: int, b: int, k: int) -> list[list[int]]:
    """Split inclusive [a, b] into k contiguous index groups. Each group seeds one
    merged trainable layer: 'representative' init uses the group's midpoint layer,
    'average' init means the group's layers elementwise. e.g. partition(6,11,1) =
    [[6,7,8,9,10,11]]; partition(6,11,2) = [[6,7,8],[9,10,11]]."""
    if b < a:
        raise ValueError(f"empty merge range [{a},{b}]")
    idxs = list(range(a, b + 1))
    n = len(idxs)
    k = max(1, min(k, n))
    return [idxs[i * n // k:(i + 1) * n // k] for i in range(k)]


def parse_keep(s: str, depth: int) -> list[int]:
    """'0-5,12,19' → sorted [0,1,2,3,4,5,12,19] — the frozen (clean) anchor layers.
    Everything NOT listed is a gap that gets collapsed into trainable layers."""
    keep = set()
    for part in s.replace(" ", "").split(","):
        if part == "":
            continue
        if "-" in part:
            lo, hi = part.split("-")
            keep.update(range(int(lo), int(hi) + 1))
        else:
            keep.add(int(part))
    return sorted(k for k in keep if 0 <= k < depth)


def plan_compressed(keep, depth: int, n_merged_per_block: int = 1):
    """The new-stack plan: a list of ('keep', idx) [frozen, untouched] and
    ('merge', a, b, groups) [gap a..b collapsed into len(groups) trainable layers,
    each seeded from its group of source layers]. Pure — no model needed."""
    keep_set = set(keep)
    plan, i = [], 0
    while i < depth:
        if i in keep_set:
            plan.append(("keep", i)); i += 1
        else:
            a = i
            while i < depth and i not in keep_set:
                i += 1
            b = i - 1
            plan.append(("merge", a, b, partition(a, b, n_merged_per_block)))
    return plan


def new_depth_of(plan) -> int:
    return sum(1 if e[0] == "keep" else len(e[3]) for e in plan)


def describe_plan(plan, init: str = "representative") -> str:
    parts = []
    for e in plan:
        if e[0] == "keep":
            parts.append(f"[{e[1]}]")
        else:
            for g in e[3]:
                parts.append(f"avg({g[0]}-{g[-1]})" if init == "average"
                             else f"rep({g[len(g) // 2]})")
    return " ".join(parts)


def compute_fraction(new_depth: int, full_depth: int) -> float:
    return new_depth / full_depth


def split_samples(samples, n_train: int, n_val: int):
    return samples[:n_train], samples[n_train:n_train + n_val]


# ---- model surgery ----------------------------------------------------------

def _average_layers(layers):
    """A new layer whose weights are the elementwise MEAN of `layers` (structure
    deep-copied from the first). NOTE: averaging MoE experts blends layer-specific
    specialists — a weak init; full-FT must heal it. Use --init representative for
    a coherent single-layer init instead."""
    import copy
    import torch
    base = copy.deepcopy(layers[0])
    sds = [l.state_dict() for l in layers]
    avg = base.state_dict()
    for key in avg:
        avg[key] = torch.stack([sd[key].float() for sd in sds], 0).mean(0).to(avg[key].dtype)
    base.load_state_dict(avg)
    return base


def build_compressed(model, keep, n_merged_per_block: int = 1, init: str = "representative"):
    """Rebuild the decoder stack from `plan_compressed`: kept anchors stay frozen
    and UNTOUCHED (clean); each gap collapses into trainable layers seeded either
    from the group's midpoint layer (init='representative') or the group's
    elementwise weight average (init='average'). Returns (model, merged, new_depth, plan)."""
    import copy
    import torch.nn as nn
    base = getattr(model, "model", model)
    orig = list(base.layers)
    depth = len(orig)
    plan = plan_compressed(keep, depth, n_merged_per_block)

    new_layers, merged = [], []
    for e in plan:
        if e[0] == "keep":
            layer = orig[e[1]]
            for p in layer.parameters():
                p.requires_grad_(False)          # frozen, clean
            new_layers.append(layer)
        else:
            _, a, b, groups = e
            for g in groups:
                if init == "average":
                    m = _average_layers([orig[j] for j in g])
                else:                            # representative: the group midpoint
                    m = copy.deepcopy(orig[g[len(g) // 2]])
                for p in m.parameters():
                    p.requires_grad_(True)        # trainable merged layer
                merged.append(m); new_layers.append(m)
    base.layers = nn.ModuleList(new_layers)
    new_depth = len(new_layers)
    for i, layer in enumerate(base.layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = i
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "layer_idx"):
            attn.layer_idx = i
    for cfg in (getattr(base, "config", None), getattr(model, "config", None)):
        if cfg is not None and hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = new_depth
    return model, merged, new_depth, plan


def _layers(model):
    return list(getattr(model, "model", model).layers)


# ---- fidelity + training ----------------------------------------------------

def model_fidelity(model, blocks, *, block_length: int, device: str) -> float:
    """Mean tail agreement of the compressed model's iter-1 argmax with the full
    model's (stored true_pred)."""
    import torch
    from probe_runner.llada2_runner import _build_block_causal_mask, _embedding_layer
    emb = _embedding_layer(model)
    dtype = next(model.parameters()).dtype
    accs = []
    model.eval()
    with torch.no_grad():
        for blk in blocks:
            attn = _build_block_causal_mask(blk["e"], blk["Lp"], block_length, dtype, device)
            pos = torch.arange(blk["e"], device=device).unsqueeze(0)
            logits = model(inputs_embeds=emb(blk["iter1"]), attention_mask=attn,
                           position_ids=pos, use_cache=False, return_dict=True).logits
            draft = logits[0, blk["s"]:blk["e"]].argmax(-1)
            t = blk["tail"]
            accs.append(float(((draft == blk["true_pred"]) & t).float().sum() / t.float().sum()))
    return float(np.mean(accs)) if accs else float("nan")


def save_compressed(merged, spec, out_dir):
    import torch
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"spec": spec, "merged": [m.state_dict() for m in merged]},
               os.path.join(out_dir, "compressed.pt"))


def train(model, merged, train_blocks, val_blocks, *, block_length, device, epochs,
          lr, batch_blocks, weight_decay, patience, out_dir, spec) -> dict:
    import torch
    import torch.nn.functional as F
    from probe_runner.llada2_runner import _build_block_causal_mask, _embedding_layer
    emb = _embedding_layer(model)
    dtype = next(model.parameters()).dtype
    params = [p for m in merged for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(0)

    f0 = model_fidelity(model, val_blocks, block_length=block_length, device=device)
    print(f"[merge] val tail-fidelity before training (untrained merge floor): {f0:.1%}")
    best, best_ep, since, history = f0, 0, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_blocks))
        opt.zero_grad(); loss_sum, n, accum = 0.0, 0, 0
        for blk in tc.progress_iter([train_blocks[j] for j in order],
                                    total=len(order), desc=f"merge epoch {ep}/{epochs}"):
            tail = blk["tail"]
            if tail.float().sum() == 0:
                continue
            attn = _build_block_causal_mask(blk["e"], blk["Lp"], block_length, dtype, device)
            pos = torch.arange(blk["e"], device=device).unsqueeze(0)
            logits = model(inputs_embeds=emb(blk["iter1"]), attention_mask=attn,
                           position_ids=pos, use_cache=False, return_dict=True
                           ).logits[0, blk["s"]:blk["e"]]
            loss = F.cross_entropy(logits[tail], blk["true_pred"][tail])
            (loss / batch_blocks).backward()
            loss_sum += float(loss); n += 1; accum += 1
            if accum % batch_blocks == 0:
                opt.step(); opt.zero_grad()
        if accum % batch_blocks != 0:
            opt.step(); opt.zero_grad()
        vf = model_fidelity(model, val_blocks, block_length=block_length, device=device)
        tr = loss_sum / max(n, 1)
        history.append({"epoch": ep, "train_ce": tr, "val_fidelity": vf})
        tag = ""
        if vf > best:
            best, best_ep, since = vf, ep, 0
            if out_dir:
                save_compressed(merged, spec, out_dir); tag = "  ↑ new best (saved)"
        else:
            since += 1
        print(f"[merge] epoch {ep}: train_ce={tr:.3f}  val tail-fidelity={vf:.1%}{tag}")
        if patience and since >= patience:
            print(f"[merge] early stop: no val improvement for {patience} epochs"); break
    print(f"[merge] best val tail-fidelity={best:.1%} at epoch {best_ep} (floor {f0:.1%})")
    return {"floor": f0, "best": best, "best_epoch": best_ep, "history": history}


# ---- GSM8K accuracy vs compute (compressed model) ---------------------------

def eval_gsm8k(model, tokenizer, problems, *, n_problems, gen_length, block_length,
               device, eos_id, full_depth: int) -> dict:
    import torch
    from probe_runner.eval_decode_compute import decode_with_refresh, is_correct
    layers = _layers(model)
    new_depth = len(layers)

    def _format(q):
        msg = [{"role": "user", "content": q}]
        p = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        return tokenizer(p, return_tensors="pt")["input_ids"].to(device)

    correct, total, passes, trunc = 0, 0, 0, 0
    for q, gold in tc.progress_iter(problems[:n_problems], total=min(n_problems, len(problems)),
                                    desc="GSM8K (compressed)"):
        prompt = _format(q); Lp = int(prompt.shape[1])
        x, _, nfe = decode_with_refresh(model, layers, prompt, cut_layer=0, rethink_every=1,
                                        mask_id=tc.MASK_ID, eos_id=eos_id, gen_length=gen_length,
                                        block_length=block_length)
        gen = x[0, Lp:]
        hit = eos_id is not None and bool((gen == eos_id).any())
        if hit:
            gen = gen[: int(torch.where(gen == eos_id)[0][0])]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        correct += int(is_correct(text, gold)); total += 1; passes += nfe; trunc += int(not hit)
    mean_passes = passes / max(total, 1)
    return {"acc": correct / max(total, 1), "n": total, "mean_passes": mean_passes,
            "new_depth": new_depth,
            "eff_compute_passes": mean_passes * new_depth / full_depth, "no_eos": trunc}


# ---- selftest (torch-free) --------------------------------------------------

def _selftest() -> None:
    assert partition(6, 11, 1) == [[6, 7, 8, 9, 10, 11]]
    assert partition(6, 11, 2) == [[6, 7, 8], [9, 10, 11]]
    assert partition(13, 18, 1) == [[13, 14, 15, 16, 17, 18]]
    assert parse_keep("0-5,12,19", 20) == [0, 1, 2, 3, 4, 5, 12, 19]
    # the canonical plan: 0-5 | merge(6-11) | 12 (clean) | merge(13-18) | 19
    plan = plan_compressed([0, 1, 2, 3, 4, 5, 12, 19], 20, 1)
    assert new_depth_of(plan) == 10
    keeps = [e[1] for e in plan if e[0] == "keep"]
    merges = [(e[1], e[2]) for e in plan if e[0] == "merge"]
    assert 12 in keeps and 0 in keeps and 19 in keeps      # anchors kept CLEAN
    assert merges == [(6, 11), (13, 18)]                    # the two redundant flanks
    assert [e[0] for e in plan] == (["keep"] * 6 + ["merge"] + ["keep"] + ["merge"] + ["keep"])
    # midpoint reps for representative-init: 6-11→9, 13-18→16
    assert describe_plan(plan) == "[0] [1] [2] [3] [4] [5] rep(9) [12] rep(16) [19]"
    assert describe_plan(plan, "average") == "[0] [1] [2] [3] [4] [5] avg(6-11) [12] avg(13-18) [19]"
    assert abs(compute_fraction(10, 20) - 0.5) < 1e-9
    tr, va = split_samples(list(range(10)), 6, 3)
    assert tr == [0, 1, 2, 3, 4, 5] and va == [6, 7, 8]
    print("merge_layers selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llada2")
    ap.add_argument("--probes_root", default="probes_out")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--t3dmax_root", default=None)
    ap.add_argument("--keep", default="0-5,12,19",
                    help="frozen CLEAN anchor layers (importance-map: 0-5 early, 12 critical, "
                         "19 last); the gaps between them collapse into trainable layers")
    ap.add_argument("--n_merged_per_block", type=int, default=1,
                    help="trainable layers each gap collapses into (1 → 10 layers for the default keep)")
    ap.add_argument("--init", choices=["representative", "average"], default="representative",
                    help="merged-layer init: 'representative' = the group's midpoint layer "
                         "(coherent; recommended for MoE); 'average' = elementwise mean of the "
                         "group (blends MoE experts — weak init, relies on full-FT to heal)")
    ap.add_argument("--n_train_samples", type=int, default=80)
    ap.add_argument("--n_val_samples", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--batch_blocks", type=int, default=8)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_gsm8k", type=int, default=0, help="if >0, GSM8K eval on N problems after training")
    ap.add_argument("--gen_length", type=int, default=512)
    ap.add_argument("--eos_id", type=int, default=156892)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry_run", action="store_true", help="build compressed model + test a forward, then stop")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if not args.model_path:
        raise SystemExit("--model_path is required.")

    from probe_runner.llada2_runner import load_llada2
    model, tokenizer = load_llada2(args.model_path, attn_implementation="sdpa",
                                   t3dmax_root=args.t3dmax_root, device=args.device)
    full_depth = len(_layers(model))
    keep = parse_keep(args.keep, full_depth)

    if args.dry_run:
        import torch
        model, merged, new_depth, plan = build_compressed(model, keep, args.n_merged_per_block, args.init)
        n_tr = sum(p.numel() for m in merged for p in m.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"[merge] plan ({args.init} init): {describe_plan(plan, args.init)}")
        print(f"[merge] compressed depth {full_depth}→{new_depth} "
              f"({compute_fraction(new_depth, full_depth):.0%} layers); "
              f"trainable {n_tr:,} / {n_all:,} ({100*n_tr/max(n_all,1):.2f}%)")
        try:
            ids = torch.randint(0, 1000, (1, 48), device=args.device)
            from probe_runner.llada2_runner import _build_block_causal_mask, _embedding_layer
            emb = _embedding_layer(model)
            attn = _build_block_causal_mask(48, 16, args.block_length,
                                            next(model.parameters()).dtype, args.device)
            pos = torch.arange(48, device=args.device).unsqueeze(0)
            with torch.no_grad():
                out = model(inputs_embeds=emb(ids), attention_mask=attn, position_ids=pos,
                            use_cache=False, return_dict=True)
            print(f"[merge] forward OK — logits {tuple(out.logits.shape)}. --dry_run done.")
        except Exception as e:
            print(f"[merge] FORWARD FAILED after surgery: {type(e).__name__}: {e}\n"
                  f"        the model's forward may assume a fixed layer count / layer_idx; "
                  f"inspect LLaDA2MoeModel.forward.")
        return

    # teacher targets from the FULL model (before surgery), prompt-disjoint
    state = {"keep": None}
    samples = [tc.load_sample(p) for p in tc.iter_sample_paths(args.probes_root, args.model)]
    tr_s, va_s = split_samples(samples, args.n_train_samples, args.n_val_samples)
    train_blocks = precompute_blocks(model, tokenizer, tr_s, block_length=args.block_length,
                                     n_samples=args.n_train_samples, device=args.device, state=state)
    val_blocks = precompute_blocks(model, tokenizer, va_s, block_length=args.block_length,
                                   n_samples=args.n_val_samples, device=args.device, state=state)
    print(f"[merge] teacher: train blocks={len(train_blocks)}  val blocks={len(val_blocks)}")

    # surgery → compressed model
    model, merged, new_depth, plan = build_compressed(model, keep, args.n_merged_per_block, args.init)
    n_tr = sum(p.numel() for m in merged for p in m.parameters() if p.requires_grad)
    print(f"[merge] plan ({args.init} init): {describe_plan(plan, args.init)}")
    print(f"[merge] compressed depth {full_depth}→{new_depth} "
          f"({compute_fraction(new_depth, full_depth):.0%} layers); trainable {n_tr:,} params")
    spec = {"keep": keep, "n_merged_per_block": args.n_merged_per_block, "init": args.init,
            "new_depth": new_depth, "full_depth": full_depth}

    res = train(model, merged, train_blocks, val_blocks, block_length=args.block_length,
                device=args.device, epochs=args.epochs, lr=args.lr, batch_blocks=args.batch_blocks,
                weight_decay=args.weight_decay, patience=args.patience, out_dir=args.out_dir, spec=spec)

    if args.eval_gsm8k > 0:
        problems = []
        for path in tc.iter_sample_paths(args.probes_root, args.model):
            attrs = tc.load_sample(path)["attrs"]
            q, g = attrs.get("prompt_text"), attrs.get("gold_answer")
            q = q.decode() if isinstance(q, bytes) else q
            g = g.decode() if isinstance(g, bytes) else g
            if q and g:
                problems.append((q, g))
        ev = eval_gsm8k(model, tokenizer, problems, n_problems=args.eval_gsm8k,
                        gen_length=args.gen_length, block_length=args.block_length,
                        device=args.device, eos_id=args.eos_id, full_depth=full_depth)
        print(f"[merge] GSM8K acc={ev['acc']:.1%} ({ev['acc']*ev['n']:.0f}/{ev['n']})  "
              f"depth={ev['new_depth']}/{full_depth} ({ev['new_depth']/full_depth:.0%} layers)  "
              f"mean-passes={ev['mean_passes']:.1f}  eff-compute={ev['eff_compute_passes']:.1f} "
              f"full-pass-equiv  no-EOS={ev['no_eos']}/{ev['n']}")
        print(f"[merge] compare vs full (84% @ 100% layers, ~49 cost in the gen512 sweep).")


if __name__ == "__main__":
    main()
