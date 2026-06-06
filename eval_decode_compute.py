"""End-to-end GSM8K accuracy vs compute for partial-depth / cadence refresh decode.

The probes settled the proxies (extraction is dead; refresh-fidelity has no
free-lunch depth) but proxies keep disagreeing because the hard tail is *filler*
(Step 0): a partial-depth refresh that diverges from a full think pass on tail
tokens may still produce the right GSM8K answer. This harness measures the only
thing that settles H1 — **task accuracy vs compute** — for the real block-diffusion
decode under a refresh policy, against the DMax full-pass baseline.

Decode policy (per block):
  * iteration 0 is always a THINK pass — full forward, all layers; its per-layer
    hidden is cached.
  * a re-THINK (full pass, cache refreshed) fires every `rethink_every` iterations.
  * every other iteration is a TALK pass — reuse the cached hidden for layers
    `< cut_layer`, recompute `cut_layer..top`, with positions committed since the
    last think kept FRESH so their tokens propagate (the fixed C15 mechanism).
  * cut_layer=0 ⇒ every pass is a think ⇒ exactly the DMax soft `decode_uniform`
    baseline (`llada2_runner.generate_with_probes`, soft mode).

Compute is counted in full-think-pass equivalents: a think pass costs 1.0, a talk
pass costs (depth−cut_layer)/depth (the layers < cut_layer are reused, not run).
Accuracy is GSM8K exact-match on the extracted final number.

Offline source of (prompt, gold): the existing capture attrs (`prompt_text`,
`gold_answer`) — same problems the probes analyzed — or `--gsm8k_n` via `datasets`.
The decode itself needs the model + a GPU; the answer-extraction / scoring /
compute-accounting cores are unit-tested via `--selftest`.

Usage
-----
    python -m probe_runner.eval_decode_compute --probes_root probes_out \
        --model llada2-DMAX --model_path T3-DMax/DMax-Math-16B-moe-merge/ \
        --configs full L9M4 L6M4 L12M4 --n_problems 100
    python -m probe_runner.eval_decode_compute --selftest
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc


# ---- GSM8K answer extraction + scoring (pure python, unit-testable) ----------

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(text: str):
    """Last numeric literal in `text` as float (commas stripped); None if none."""
    if text is None:
        return None
    nums = _NUM.findall(text.replace("$", " "))
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def gold_number(gold_answer: str):
    """GSM8K gold: the number after '####' if present, else the last number."""
    if gold_answer is None:
        return None
    tail = gold_answer.split("####")[-1] if "####" in gold_answer else gold_answer
    return _last_number(tail)


def is_correct(pred_text: str, gold_answer: str, tol: float = 1e-4) -> bool:
    """Exact-match on the extracted final numbers (GSM8K convention)."""
    p, g = _last_number(pred_text), gold_number(gold_answer)
    if p is None or g is None:
        return False
    return abs(p - g) <= tol * max(1.0, abs(g))


# ---- compute accounting (pure, unit-testable) -------------------------------

def pass_cost(is_think: bool, cut_layer: int, depth: int) -> float:
    """Full-think-pass-equivalent cost of one forward pass."""
    if is_think or cut_layer <= 0:
        return 1.0
    return max(depth - cut_layer, 0) / depth


def parse_config(label: str) -> tuple[str, int, int]:
    """'full' → (label,0,1); 'L9M4' → (label, cut_layer=9, rethink_every=4);
    'L9' → rethink_every=∞ (one think per block)."""
    if label.lower() == "full":
        return label, 0, 1
    m = re.fullmatch(r"[Ll](\d+)([Mm](\d+))?", label)
    if not m:
        raise ValueError(f"bad config '{label}' (use 'full', 'L9', or 'L9M4')")
    L = int(m.group(1))
    M = int(m.group(3)) if m.group(3) else 10 ** 9          # ∞ = one think per block
    return label, L, M


# ---- model-driven decode with refresh policy (torch) ------------------------

def _decoder_layers(model):
    base = getattr(model, "model", model)
    return [m for m in base.modules() if type(m).__name__ == "LLaDA2MoeDecoderLayer"]


def _mk_hook(idx, cache, state):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if state["record"]:
            cache[idx] = h.detach()
        elif state["cut"] is not None and idx < state["cut"]:
            h_new = cache[idx].clone()
            rev = state["reveal"]
            if rev is not None:
                h_new[:, rev] = h[:, rev]
            return (h_new,) + out[1:] if isinstance(out, tuple) else h_new
        return None
    return hook


def decode_with_refresh(model, layers, prompt, *, cut_layer: int, rethink_every: int,
                        mask_id: int, eos_id=None, gen_length: int = 256,
                        block_length: int = 32, commit_threshold: float = 0.3,
                        break_threshold: float = 0.9, max_iter_per_block=None):
    """Soft DMax decode with a partial-depth/cadence refresh policy. Returns
    (x, cost, nfe): x [1, Lp+gen] ids, cost in full-pass-equivalents, nfe raw passes."""
    import torch
    import torch.nn.functional as F
    from probe_runner.llada2_runner import (
        _build_block_causal_mask, _commit_uniform, _build_block_soft_embeds,
        _embedding_layer)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prompt = prompt.to(device)
    Lp = int(prompt.shape[1])
    depth = len(layers)
    if max_iter_per_block is None:
        max_iter_per_block = block_length
    emb = _embedding_layer(model)

    cache: dict = {}
    state = {"cut": None, "record": False, "reveal": None}
    handles = [layer.register_forward_hook(_mk_hook(i, cache, state)) for i, layer in enumerate(layers)]

    x = F.pad(prompt, (0, gen_length), value=mask_id)
    nfe, cost = 0, 0.0
    num_blocks = gen_length // block_length
    try:
        with torch.no_grad():
            for nb in range(num_blocks):
                s, e = Lp + nb * block_length, Lp + (nb + 1) * block_length
                attn_mask = _build_block_causal_mask(e, Lp, block_length, dtype, device)
                pos_ids = torch.arange(e, device=device).unsqueeze(0)
                block_soft = None
                think_tokens = None
                for i in range(max_iter_per_block):
                    is_think = (i == 0) or (cut_layer <= 0) or (i % rethink_every == 0)
                    inputs_embeds = emb(x[:, :e])
                    if block_soft is not None:
                        inputs_embeds[:, s:e, :] = block_soft
                    if is_think:
                        state["record"], state["cut"], state["reveal"] = True, None, None
                    else:
                        changed = (x[0, s:e] != think_tokens) & (x[0, s:e] != mask_id)
                        reveal = torch.zeros(e, dtype=torch.bool, device=device)
                        reveal[s + torch.where(changed)[0]] = True
                        state["record"], state["cut"], state["reveal"] = False, cut_layer, reveal
                    out = model(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                                position_ids=pos_ids, use_cache=False,
                                output_hidden_states=False, return_dict=True)
                    if is_think:
                        state["record"] = False
                        think_tokens = x[0, s:e].clone()
                    nfe += 1
                    cost += pass_cost(is_think, cut_layer, depth)

                    block_logits = out.logits[:, s:e, :]
                    block_x = x[:, s:e]
                    mask_index = (block_x == mask_id)
                    x0, commit_index, max_probs = _commit_uniform(
                        block_logits, block_x, mask_id, commit_threshold)
                    update_mask = commit_index | (~mask_index)          # soft: re-argmax committed
                    changed = update_mask & (x0 != block_x)
                    new_block = block_x.clone()
                    new_block[update_mask] = x0[update_mask]
                    x[:, s:e] = new_block

                    no_change = not bool(changed.any())
                    stop = bool((max_probs >= break_threshold).all()) or no_change
                    if stop:
                        break
                    soft_cond = (x[0, s:e] != mask_id)
                    block_soft = _build_block_soft_embeds(
                        x[:, s:e], max_probs, x0, soft_cond, emb, mask_id)
    finally:
        for h in handles:
            h.remove()
        state["cut"], state["reveal"], state["record"] = None, None, False
    return x, cost, nfe


# ---- evaluation driver ------------------------------------------------------

def load_problems_from_captures(probes_root, model: str):
    """[(question, gold_answer)] from the capture attrs."""
    out = []
    for path in tc.iter_sample_paths(probes_root, model):
        sample = tc.load_sample(path)
        attrs = sample["attrs"]
        q = attrs.get("prompt_text"); g = attrs.get("gold_answer")
        q = q.decode() if isinstance(q, bytes) else q
        g = g.decode() if isinstance(g, bytes) else g
        if q is not None and g is not None:
            out.append((q, g))
    return out


def run_eval(model_path: str, problems: list, config_labels: list, *,
             n_problems: int, gen_length: int = 256, block_length: int = 32,
             t3dmax_root=None, device: str = "cuda") -> dict:
    import torch
    from probe_runner.llada2_runner import load_llada2

    model, tokenizer = load_llada2(model_path, attn_implementation="sdpa",
                                   t3dmax_root=t3dmax_root, device=device)
    layers = _decoder_layers(model)
    depth = len(layers)
    mask_id = tc.MASK_ID
    eos_id = tokenizer.eos_token_id
    cfgs = [parse_config(c) for c in config_labels]

    def _format(q):
        msg = [{"role": "user", "content": q}]
        p = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        return tokenizer(p, return_tensors="pt")["input_ids"].to(device)

    res = {lbl: {"correct": 0, "total": 0, "cost": 0.0, "nfe": 0, "cut": L, "rethink": M}
           for lbl, L, M in cfgs}
    for q, gold in problems[:n_problems]:
        prompt = _format(q)
        Lp = int(prompt.shape[1])
        for lbl, L, M in cfgs:
            x, cost, nfe = decode_with_refresh(
                model, layers, prompt, cut_layer=L, rethink_every=M, mask_id=mask_id,
                eos_id=eos_id, gen_length=gen_length, block_length=block_length)
            gen = x[0, Lp:]
            if eos_id is not None and (gen == eos_id).any():
                gen = gen[: int(torch.where(gen == eos_id)[0][0])]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            r = res[lbl]
            r["correct"] += int(is_correct(text, gold))
            r["total"] += 1
            r["cost"] += cost
            r["nfe"] += nfe
    return {"depth": depth, "per_config": res}


def format_report(out: dict) -> str:
    lines = [f"GSM8K accuracy vs compute  (depth={out['depth']} layers)"]
    base = out["per_config"].get("full")
    base_cost = (base["cost"] / max(base["total"], 1)) if base else None
    for lbl, r in out["per_config"].items():
        acc = r["correct"] / max(r["total"], 1)
        mc = r["cost"] / max(r["total"], 1)
        rel = f"  ({mc / base_cost:.0%} of full)" if base_cost else ""
        lines.append(f"  {lbl:<8s} cut=L{r['cut']} rethink={r['rethink'] if r['rethink']<10**8 else '∞'}  "
                     f"acc={acc:.1%} ({r['correct']}/{r['total']})  "
                     f"mean-cost={mc:.2f} think-pass-equiv{rel}")
    return "\n".join(lines)


def plot(out: dict, out_dir: Path, model: str) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for lbl, r in out["per_config"].items():
        acc = r["correct"] / max(r["total"], 1)
        mc = r["cost"] / max(r["total"], 1)
        ax.scatter(mc, acc, s=70)
        ax.annotate(lbl, (mc, acc), textcoords="offset points", xytext=(6, 4), fontsize=9)
    base = out["per_config"].get("full")
    if base:
        ax.axhline(base["correct"] / max(base["total"], 1), ls="--", color="gray",
                   alpha=0.6, label="full (DMax) accuracy")
    ax.set_xlabel("mean compute per problem (full-think-pass equivalents)")
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title(f"{model}: accuracy vs compute (partial-depth / cadence refresh)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "decode_accuracy_vs_compute.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")


# ---- selftest (torch-free cores) --------------------------------------------

def _selftest() -> None:
    assert _last_number("The answer is 18.") == 18.0
    assert _last_number("so we get 1,234 apples") == 1234.0
    assert _last_number("x = 3.5 kg") == 3.5
    assert _last_number("no digits here") is None
    assert gold_number("Natalia sold ... #### 72") == 72.0
    assert gold_number("just 5 then 9") == 9.0
    assert is_correct("the total is 42", "#### 42")
    assert is_correct("answer: 1,000", "#### 1000")
    assert not is_correct("we get 41", "#### 42")
    assert not is_correct("no number", "#### 42")
    # compute accounting
    assert pass_cost(True, 9, 20) == 1.0
    assert pass_cost(False, 10, 20) == 0.5
    assert pass_cost(False, 0, 20) == 1.0          # cut=0 ⇒ always full
    assert parse_config("full") == ("full", 0, 1)
    assert parse_config("L9M4") == ("L9M4", 9, 4)
    assert parse_config("L12")[1] == 12 and parse_config("L12")[2] > 10 ** 8
    # a block of 1 think + 3 talks at L=10/depth=20 costs 1 + 3*0.5 = 2.5
    total = pass_cost(True, 10, 20) + sum(pass_cost(False, 10, 20) for _ in range(3))
    assert abs(total - 2.5) < 1e-9
    print("eval_decode_compute selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llada2")
    ap.add_argument("--probes_root", default="probes_out")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--t3dmax_root", default=None)
    ap.add_argument("--configs", nargs="+", default=["full", "L9M4", "L6M4", "L12M4"],
                    help="e.g. full L9M4 L6 L12M2  (L<cut>M<rethink>; no M ⇒ one think/block)")
    ap.add_argument("--n_problems", type=int, default=100)
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--gsm8k_n", type=int, default=0,
                    help="if >0, load this many GSM8K test problems via `datasets` "
                         "instead of reading prompts from the captures.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.model_path:
        raise SystemExit("--model_path is required for the decode.")
    if args.gsm8k_n > 0:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test").select(range(args.gsm8k_n))
        problems = [(p["question"], p["answer"]) for p in ds]
    else:
        problems = load_problems_from_captures(args.probes_root, args.model)
        if not problems:
            raise SystemExit(f"no (prompt,gold) in captures under {args.probes_root}/{args.model}; "
                             f"use --gsm8k_n N instead.")
    out = run_eval(args.model_path, problems, args.configs, n_problems=args.n_problems,
                   gen_length=args.gen_length, block_length=args.block_length,
                   t3dmax_root=args.t3dmax_root, device=args.device)
    print(format_report(out))
    plot(out, Path(args.probes_root) / args.model / "plots", args.model)


if __name__ == "__main__":
    main()
