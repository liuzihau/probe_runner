"""Stage 1 — Draft-&-Verify layer-subset search for the iter>0 draft network.

The plan this serves: at inference, iter 0 is a FULL DMax think pass; iter>0 runs
a cheap DRAFT that keeps only a sparse subset of layers (the rest skipped =
identity on the residual stream), optionally LoRA-compensated. This module is
Stage 1: find the best layer subset to keep, before any training.

Why a *subset* (not the contiguous C15 cut): the layers that matter aren't
contiguous, so picking k layers freely can beat keeping the top-k. We force the
first and last layer (embedding-readout and final-norm-facing layers are
load-bearing) and search for the remaining `extra` middle layers.

Metric (cheap proxy, same family as C15): for each block we take the iter-0 →
iter-1 reveal (commit the iter-0 high-conf prefix) and measure how often the
DRAFT (full iter-1 input, but only the kept layers applied; others = identity)
reproduces the FULL model's iter-1 argmax on the still-masked tail. The full
passes are computed ONCE per block and reused across all candidate subsets.

Untrained layer-skip is usually brutal — read the resulting fidelity as a FLOOR.
Stage 2 (LoRA on the kept layers, fine-tuned for iter>0) is what is expected to
lift it; this search picks *which* layers LoRA should sit on.

Offline (prompt+gold/converged from captures) but needs the model + GPU. The
enumeration / aggregation cores are unit-tested via `--selftest`.

Usage
-----
    python -m probe_runner.eval_layer_subset --model llada2-DMAX \
        --probes_root probes_out --model_path T3-DMax/DMax-Math-16B-moe-merge/ \
        --extra 2 --n_samples 20
    python -m probe_runner.eval_layer_subset --selftest
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc


# ---- subset enumeration (pure python, unit-testable) ------------------------

def forced_layers(depth: int, keep_first_last: bool = True) -> tuple[int, ...]:
    return (0, depth - 1) if keep_first_last else ()


def enumerate_candidates(depth: int, extra: int, *, keep_first_last: bool = True):
    """All keep-sets = forced ∪ (extra middle layers). Returns sorted tuples."""
    forced = set(forced_layers(depth, keep_first_last))
    middle = [l for l in range(depth) if l not in forced]
    return [tuple(sorted(forced | set(c))) for c in itertools.combinations(middle, extra)]


def contiguous_top(depth: int, n_total: int, *, keep_first_last: bool = True) -> tuple[int, ...]:
    """The C15-style baseline at EQUAL budget: keep exactly `n_total` layers
    contiguously from the top. If keep_first_last, force layer 0 and fill the
    remaining n_total-1 from the top (the last layer is naturally in that block),
    so the count matches a flexible subset of the same size — not one fewer."""
    n_total = max(1, min(n_total, depth))
    if keep_first_last:
        keep = {0} | set(range(depth - (n_total - 1), depth))
    else:
        keep = set(range(depth - n_total, depth))
    return tuple(sorted(keep))


def compute_fraction(keep, depth: int) -> float:
    """Fraction of layers actually run (the draft's per-pass compute vs full)."""
    return len(set(keep)) / depth


# ---- skip-identity hooks (torch; robust to args/kwargs) ---------------------

def _install_skip_hooks(layers, state):
    """Make layers not in state['keep'] act as identity on the residual stream.
    state['keep']=None ⇒ full model (no skipping). Returns handles to remove."""
    cached_in: dict = {}
    handles = []
    for i, layer in enumerate(layers):
        def pre(mod, args, kwargs, idx=i):
            cached_in[idx] = args[0] if args else kwargs.get("hidden_states")
            return None
        def post(mod, inp, out, idx=i):
            keep = state["keep"]
            if keep is not None and idx not in keep:
                h = cached_in[idx]                      # identity: pass input through
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return None
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(layer.register_forward_hook(post))
    return handles


# ---- precompute the per-block full passes (once) ----------------------------

def precompute_blocks(model, tokenizer, samples, *, block_length: int, n_samples: int,
                      device: str, state) -> list:
    """Per block: {iter1 ids [1,e], s, e, Lp, true_pred [B], tail [B]}. The full
    iter-0 (→reveal) and full iter-1 (→true_pred) passes are done here once."""
    import torch
    from probe_runner.llada2_runner import (
        _build_block_causal_mask, _commit_uniform, _embedding_layer)

    emb = _embedding_layer(model)
    dtype = next(model.parameters()).dtype

    def _format(q):
        msg = [{"role": "user", "content": q}]
        p = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        return tokenizer(p, return_tensors="pt")["input_ids"].to(device)

    blocks = []
    state["keep"] = None                                    # full model
    done = 0
    for sample in samples:
        attrs = sample["attrs"]
        q = attrs.get("prompt_text")
        q = q.decode() if isinstance(q, bytes) else q
        if q is None:
            continue
        prompt_ids = _format(q)
        Lp = int(prompt_ids.shape[1])
        for b, blk in sorted(sample["blocks"].items()):
            if "converged_tokens" not in blk and "committed_tokens_per_pass" not in blk:
                continue
            s, e = Lp + b * block_length, Lp + (b + 1) * block_length
            attn = _build_block_causal_mask(e, Lp, block_length, dtype, device)
            pos = torch.arange(e, device=device).unsqueeze(0)
            iter0 = torch.full((1, e), tc.MASK_ID, dtype=torch.long, device=device)
            iter0[0, :Lp] = prompt_ids[0, :Lp]
            for j in range(b):
                iter0[0, Lp + j * block_length: Lp + (j + 1) * block_length] = \
                    torch.tensor(tc.converged_tokens(sample["blocks"][j]), device=device)
            with torch.no_grad():
                logits0 = model(inputs_embeds=emb(iter0), attention_mask=attn,
                                position_ids=pos, use_cache=False, return_dict=True).logits[:, s:e]
                x0, hc, _ = _commit_uniform(logits0, iter0[:, s:e], tc.MASK_ID, 0.3)
                iter1 = iter0.clone()
                iter1[0, s:e][hc[0]] = x0[0][hc[0]]
                true_pred = model(inputs_embeds=emb(iter1), attention_mask=attn,
                                  position_ids=pos, use_cache=False, return_dict=True
                                  ).logits[:, s:e].argmax(-1)[0]
            valid = torch.tensor(tc.valid_position_mask(b, e - s, attrs, block_length), device=device)
            tail = valid & ~hc[0]
            if tail.float().sum() == 0:
                continue
            blocks.append({"iter1": iter1, "s": s, "e": e, "Lp": Lp,
                           "true_pred": true_pred, "tail": tail})
        done += 1
        if n_samples and done >= n_samples:
            break
    return blocks


def subset_fidelity(model, blocks, keep, *, block_length: int, device: str, state) -> float:
    """Mean tail agreement of the keep-subset draft with the full iter-1 prediction."""
    import torch
    from probe_runner.llada2_runner import _build_block_causal_mask, _embedding_layer
    emb = _embedding_layer(model)
    dtype = next(model.parameters()).dtype
    keep_set = set(keep)
    state["keep"] = keep_set
    accs = []
    with torch.no_grad():
        for blk in blocks:
            attn = _build_block_causal_mask(blk["e"], blk["Lp"], block_length, dtype, device)
            pos = torch.arange(blk["e"], device=device).unsqueeze(0)
            draft = model(inputs_embeds=emb(blk["iter1"]), attention_mask=attn,
                          position_ids=pos, use_cache=False, return_dict=True
                          ).logits[:, blk["s"]:blk["e"]].argmax(-1)[0]
            t = blk["tail"]
            accs.append(float(((draft == blk["true_pred"]) & t).float().sum() / t.float().sum()))
    state["keep"] = None
    return float(np.mean(accs)) if accs else float("nan")


# ---- search -----------------------------------------------------------------

def greedy_search(model, blocks, depth, extra, *, keep_first_last, block_length, device, state):
    keep = set(forced_layers(depth, keep_first_last))
    middle = [l for l in range(depth) if l not in keep]
    history = []
    for _ in range(extra):
        scored = [(c, subset_fidelity(model, blocks, keep | {c}, block_length=block_length,
                                      device=device, state=state)) for c in middle if c not in keep]
        c_best, f_best = max(scored, key=lambda kv: kv[1])
        keep.add(c_best)
        history.append({"added": c_best, "keep": tuple(sorted(keep)), "fidelity": f_best})
    return history


def exhaustive_search(model, blocks, depth, extra, *, keep_first_last, block_length, device, state):
    cands = enumerate_candidates(depth, extra, keep_first_last=keep_first_last)
    scored = [{"keep": k, "fidelity": subset_fidelity(model, blocks, k, block_length=block_length,
                                                       device=device, state=state)} for k in cands]
    scored.sort(key=lambda d: d["fidelity"], reverse=True)
    return scored


def one_swaps(keep, depth: int, forced=()):
    """Neighbor subsets: replace one (non-forced) kept layer with one dropped
    layer. The fixed-budget analogue of a 1-bit flip — the local-search move."""
    kept = set(keep)
    forced = set(forced)
    out = []
    for r in kept - forced:                     # never swap out a forced layer
        for a in range(depth):
            if a not in kept:
                out.append(tuple(sorted((kept - {r}) | {a})))
    return out


def local_search_refine(model, blocks, seed_keep, depth, *, forced=(), block_length,
                        device, state, max_rounds: int = 12):
    """Hill-climb from `seed_keep`: at each round adopt the single best 1-swap that
    improves fidelity; stop at a local optimum (the cheap stand-in for D&V's
    Bayesian optimization over the fixed-budget subset space)."""
    cur = tuple(sorted(seed_keep))
    cur_f = subset_fidelity(model, blocks, cur, block_length=block_length, device=device, state=state)
    history = [{"keep": cur, "fidelity": cur_f, "round": 0}]
    for rnd in range(1, max_rounds + 1):
        best, best_f = cur, cur_f
        for cand in one_swaps(cur, depth, forced):
            f = subset_fidelity(model, blocks, cand, block_length=block_length, device=device, state=state)
            if f > best_f:
                best, best_f = cand, f
        if best == cur:
            break                               # local optimum
        cur, cur_f = best, best_f
        history.append({"keep": cur, "fidelity": cur_f, "round": rnd})
    return history


def layer_importance(scored, depth, top_frac=0.1):
    """How often each layer appears among the top fraction of subsets."""
    n = max(1, int(len(scored) * top_frac))
    freq = np.zeros(depth)
    for d in scored[:n]:
        for l in d["keep"]:
            freq[l] += 1
    return freq / n


# ---- report + plot ----------------------------------------------------------

def format_report(scored, refs, depth, n_total) -> str:
    lines = [f"Layer-subset search (depth={depth}, keep {n_total} layers; "
             f"draft compute = {n_total}/{depth} = {n_total/depth:.0%})"]
    lines.append("  references:")
    for name, k, f in refs:
        lines.append(f"    {name:<16s} keep={k}  fidelity={f:.1%}")
    lines.append("  top subsets:")
    for d in scored[:8]:
        lines.append(f"    keep={d['keep']}  fidelity={d['fidelity']:.1%}")
    return "\n".join(lines)


def plot(scored, refs, depth, out_dir: Path, model: str) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    # top subsets vs references
    ax = axes[0]
    top = scored[:10]
    ys = [d["fidelity"] for d in top]
    ax.bar(range(len(top)), ys, color="tab:blue")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([",".join(map(str, d["keep"])) for d in top], rotation=40, fontsize=7, ha="right")
    for name, _, f in refs:
        ax.axhline(f, ls="--", alpha=0.6, label=f"{name} ({f:.0%})")
    ax.set_ylabel("tail fidelity to full iter-1"); ax.set_ylim(0, 1)
    ax.set_title("top keep-subsets vs references"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    # layer importance
    ax = axes[1]
    imp = layer_importance(scored, depth)
    ax.bar(range(depth), imp, color="tab:green")
    ax.set_xlabel("layer"); ax.set_ylabel("freq in top-10% subsets")
    ax.set_title("layer importance (which layers the best drafts keep)")
    ax.grid(alpha=0.3)
    fig.suptitle(f"{model}: Stage-1 layer-subset search")
    fig.tight_layout()
    p = out_dir / "layer_subset_search.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")


# ---- selftest (torch-free combinatorics) ------------------------------------

def _selftest() -> None:
    import math
    depth, extra = 20, 2
    cands = enumerate_candidates(depth, extra)
    assert len(cands) == math.comb(depth - 2, extra), len(cands)
    for k in cands:
        assert 0 in k and depth - 1 in k and len(k) == extra + 2
        assert list(k) == sorted(k)
    # no forced
    assert len(enumerate_candidates(depth, 2, keep_first_last=False)) == math.comb(depth, 2)
    # equal-budget contiguous: n_total layers exactly, first forced + top block
    assert contiguous_top(20, 4) == (0, 17, 18, 19), contiguous_top(20, 4)
    assert len(contiguous_top(20, 4)) == 4 and len(enumerate_candidates(20, 2)[0]) == 4
    assert contiguous_top(20, 4, keep_first_last=False) == (16, 17, 18, 19)
    assert abs(compute_fraction((0, 7, 13, 19), 20) - 0.2) < 1e-9
    # layer_importance picks the always-present layers
    scored = [{"keep": (0, 5, 19), "fidelity": 0.9}, {"keep": (0, 6, 19), "fidelity": 0.8},
              {"keep": (0, 7, 19), "fidelity": 0.1}]
    imp = layer_importance(scored, 20, top_frac=0.67)        # top 2 of 3
    assert imp[0] == 1.0 and imp[19] == 1.0 and imp[5] == 0.5 and imp[7] == 0.0
    # 1-swap neighbors: fixed size, each is a valid swap, forced layers never dropped
    sw = one_swaps((0, 4, 18, 19), 20, forced=(0, 19))
    assert all(len(s) == 4 and 0 in s and 19 in s for s in sw)
    assert (0, 5, 18, 19) in sw and (0, 4, 18, 19) not in sw   # neighbors, not self
    # swappable layers = {4,18}, targets = 20-4=16 each → 32 neighbors
    assert len(sw) == 2 * (20 - 4), len(sw)
    # no forced: all 4 are swappable
    assert len(one_swaps((1, 4, 9, 15), 20)) == 4 * (20 - 4)
    print("eval_layer_subset selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llada2")
    ap.add_argument("--probes_root", default="probes_out")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--t3dmax_root", default=None)
    ap.add_argument("--extra", type=int, default=2, help="middle layers to choose (+ forced first/last)")
    ap.add_argument("--no_force_first_last", action="store_true")
    ap.add_argument("--greedy", action="store_true", help="forward-greedy instead of exhaustive")
    ap.add_argument("--refine", action="store_true",
                    help="hill-climb (1-swap local search) from the best subset — the cheap "
                         "stand-in for Draft&Verify's Bayesian optimization; ~150-250 evals")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.model_path:
        raise SystemExit("--model_path is required for the search.")
    from probe_runner.llada2_runner import load_llada2
    model, tokenizer = load_llada2(args.model_path, attn_implementation="sdpa",
                                   t3dmax_root=args.t3dmax_root, device=args.device)
    base = getattr(model, "model", model)
    layers = [m for m in base.modules() if type(m).__name__ == "LLaDA2MoeDecoderLayer"]
    depth = len(layers)
    keep_fl = not args.no_force_first_last
    state = {"keep": None}
    handles = _install_skip_hooks(layers, state)
    try:
        samples = [tc.load_sample(p) for p in tc.iter_sample_paths(args.probes_root, args.model)]
        blocks = precompute_blocks(model, tokenizer, samples, block_length=args.block_length,
                                   n_samples=args.n_samples, device=args.device, state=state)
        print(f"[layer-subset] {len(blocks)} blocks; depth={depth}; "
              f"{'greedy' if args.greedy else 'exhaustive'} over choose-{args.extra}")
        if args.greedy:
            hist = greedy_search(model, blocks, depth, args.extra, keep_first_last=keep_fl,
                                 block_length=args.block_length, device=args.device, state=state)
            scored = [{"keep": h["keep"], "fidelity": h["fidelity"]} for h in hist][::-1]
        else:
            scored = exhaustive_search(model, blocks, depth, args.extra, keep_first_last=keep_fl,
                                       block_length=args.block_length, device=args.device, state=state)
        if args.refine:
            forced = forced_layers(depth, keep_fl)
            seed = scored[0]["keep"]
            print(f"[layer-subset] refining (1-swap hill-climb) from seed {seed} "
                  f"f={scored[0]['fidelity']:.1%}")
            ref_hist = local_search_refine(model, blocks, seed, depth, forced=forced,
                                           block_length=args.block_length, device=args.device, state=state)
            for h in ref_hist[1:]:
                print(f"    round {h['round']}: keep={h['keep']}  fidelity={h['fidelity']:.1%}")
            # merge refined path into scored and re-rank (dedup by keep-set)
            merged = {d["keep"]: d["fidelity"] for d in scored}
            for h in ref_hist:
                merged[h["keep"]] = max(merged.get(h["keep"], 0.0), h["fidelity"])
            scored = sorted(({"keep": k, "fidelity": f} for k, f in merged.items()),
                            key=lambda d: d["fidelity"], reverse=True)
        # references at EQUAL budget (same #layers as the flexible subset)
        n_keep_total = args.extra + (2 if keep_fl else 0)
        ctop = contiguous_top(depth, n_keep_total, keep_first_last=keep_fl)
        refs = [
            ("full(all layers)", tuple(range(depth)),
             subset_fidelity(model, blocks, tuple(range(depth)), block_length=args.block_length,
                             device=args.device, state=state)),
            (f"contiguous-top({len(ctop)}L)", ctop,
             subset_fidelity(model, blocks, ctop, block_length=args.block_length,
                             device=args.device, state=state)),
        ]
        print(format_report(scored, refs, depth, n_keep_total))
        plot(scored, refs, depth, Path(args.probes_root) / args.model / "plots", args.model)
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
