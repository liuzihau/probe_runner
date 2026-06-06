"""Stage 2 — LoRA finetune of the kept-layer draft (Draft-&-Verify + adaptation).

The scheme:
  * INFERENCE: iteration 0 of each block is a FULL DMax think pass (all layers, no
    LoRA, adapters disabled). Iterations >0 run a cheap DRAFT that keeps only the
    chosen subset of layers (the rest skipped = identity on the residual stream)
    WITH LoRA adapters on the kept layers.
  * STAGE 1 (eval_layer_subset.py) picked the subset; its untrained tail fidelity is
    a FLOOR. STAGE 2 (this file) trains LoRA on exactly those layers to lift it.

Training objective: the draft (kept layers + LoRA), run on the iteration-1 input
(the heavy model's iter-0 high-confidence prefix revealed), should reproduce the
FULL model's iteration-1 argmax on the still-masked tail. That is exactly the
fidelity the subset search measured, so training maximizes the searched metric.
Teacher targets (the full model's iter-1 argmax, `true_pred`) and the iter-1 inputs
are produced once by `eval_layer_subset.precompute_blocks` with adapters absent;
prompt-disjoint train/val by splitting samples before precompute.

LoRA uses HuggingFace PEFT with `layers_to_transform=<subset>` so adapters land
ONLY on the kept layers; `--dora` / `--rslora` enable the SOTA variants. The
skip-identity hooks compose with PEFT (they fire on the unchanged decoder-layer
modules; LoRA wraps the inner Linears of the kept layers).

Needs the model + GPU + `peft`. The torch-free helpers are unit-tested via
`--selftest`; `--dry_run` wraps the model and prints the adapted modules /
trainable params so targeting can be verified before a real run.

Usage
-----
    # 1) verify targeting lands on the chosen layers only
    python -m probe_runner.train_subset_lora --model llada2-DMAX \
        --model_path T3-DMax/DMax-Math-16B-moe-merge/ --subset 0,4,18,19 --dry_run
    # 2) train
    python -m probe_runner.train_subset_lora --model llada2-DMAX \
        --model_path T3-DMax/DMax-Math-16B-moe-merge/ --subset 0,4,18,19 \
        --rank 16 --rslora --n_train_samples 80 --n_val_samples 20 --epochs 5 \
        --out_dir lora_subset_0_4_18_19
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc
from probe_runner.eval_layer_subset import (
    precompute_blocks, subset_fidelity, _install_skip_hooks, compute_fraction)


# ---- helpers (pure python, unit-testable) -----------------------------------

def parse_subset(s: str) -> tuple[int, ...]:
    """'0,4,18,19' -> (0,4,18,19) (sorted, de-duplicated)."""
    out = sorted({int(x) for x in s.replace(" ", "").split(",") if x != ""})
    if not out:
        raise ValueError(f"empty subset: {s!r}")
    return tuple(out)


def split_samples(samples, n_train: int, n_val: int):
    """Prompt-disjoint split: first n_train samples for training, the next n_val
    (disjoint) for validation."""
    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    return train, val


# ---- LoRA model construction ------------------------------------------------

def make_lora_model(model, subset, *, rank: int, alpha: int, dropout: float,
                    target_modules, use_dora: bool, use_rslora: bool):
    """Wrap `model` with PEFT LoRA on ONLY the `subset` layers. Returns the
    PeftModel. `target_modules` may be 'all-linear' or a list of name substrings."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules,
        layers_to_transform=list(subset),     # adapters ONLY on the kept layers
        layers_pattern="layers",              # the decoder-layer module-name segment
        bias="none", task_type="CAUSAL_LM",
        use_dora=use_dora, use_rslora=use_rslora,
    )
    return get_peft_model(model, cfg)


def adapted_layer_report(peft_model) -> dict:
    """Which layer indices actually received a LoRA adapter (sanity check that
    targeting matched the subset), + trainable parameter count."""
    import re
    layers, n_train, n_tot = set(), 0, 0
    for name, p in peft_model.named_parameters():
        n_tot += p.numel()
        if p.requires_grad:
            n_train += p.numel()
            m = re.search(r"layers\.(\d+)\.", name)
            if m and ("lora" in name.lower()):
                layers.add(int(m.group(1)))
    return {"adapted_layers": tuple(sorted(layers)), "trainable": n_train, "total": n_tot}


# ---- training ---------------------------------------------------------------

def _embed(model):
    base = getattr(model, "get_input_embeddings", None)
    if base is not None:
        return model.get_input_embeddings()
    from probe_runner.llada2_runner import _embedding_layer
    return _embedding_layer(model)


def _block_logits(model, emb, blk, *, block_length, dtype, device, state, keep):
    import torch
    from probe_runner.llada2_runner import _build_block_causal_mask
    state["keep"] = keep
    attn = _build_block_causal_mask(blk["e"], blk["Lp"], block_length, dtype, device)
    pos = torch.arange(blk["e"], device=device).unsqueeze(0)
    out = model(inputs_embeds=emb(blk["iter1"]), attention_mask=attn, position_ids=pos,
                use_cache=False, return_dict=True)
    return out.logits[0, blk["s"]:blk["e"], :]            # [B, V]


def train(model, subset, train_blocks, val_blocks, *, block_length, device, state,
          epochs: int, lr: float, batch_blocks: int, out_dir: str | None) -> dict:
    import torch
    import torch.nn.functional as F
    emb = _embed(model)
    dtype = next(model.parameters()).dtype
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    keep = tuple(subset)
    rng = np.random.default_rng(0)
    history = []

    def _val():
        model.eval()
        with torch.no_grad():
            f = subset_fidelity(model, val_blocks, keep, block_length=block_length,
                                device=device, state=state)
        return f

    f0 = _val()
    print(f"[lora] val tail-fidelity before training (untrained subset floor): {f0:.1%}")
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_blocks))
        opt.zero_grad()
        loss_sum, n_steps, accum = 0.0, 0, 0
        for i in tc.progress_iter([train_blocks[j] for j in order],
                                  total=len(order), desc=f"lora epoch {ep}/{epochs}"):
            tail = i["tail"]
            if tail.float().sum() == 0:
                continue
            lg = _block_logits(model, emb, i, block_length=block_length, dtype=dtype,
                               device=device, state=state, keep=keep)
            loss = F.cross_entropy(lg[tail], i["true_pred"][tail])
            (loss / batch_blocks).backward()
            loss_sum += float(loss); n_steps += 1; accum += 1
            if accum % batch_blocks == 0:
                opt.step(); opt.zero_grad()
        if accum % batch_blocks != 0:
            opt.step(); opt.zero_grad()
        vf = _val()
        tr = loss_sum / max(n_steps, 1)
        history.append({"epoch": ep, "train_ce": tr, "val_fidelity": vf})
        print(f"[lora] epoch {ep}: train_ce={tr:.3f}  val tail-fidelity={vf:.1%}")
    if out_dir:
        model.save_pretrained(out_dir)
        print(f"[lora] saved adapter to {out_dir}")
    return {"floor": f0, "history": history}


# ---- selftest (torch-free helpers) ------------------------------------------

def _selftest() -> None:
    assert parse_subset("0,4,18,19") == (0, 4, 18, 19)
    assert parse_subset(" 19, 0 ,4, 4 ") == (0, 4, 19)          # sorted + dedup
    try:
        parse_subset(",")
        assert False, "empty subset should raise"
    except ValueError:
        pass
    tr, va = split_samples(list(range(10)), 6, 3)
    assert tr == [0, 1, 2, 3, 4, 5] and va == [6, 7, 8] and not (set(tr) & set(va))
    assert abs(compute_fraction((0, 4, 18, 19), 20) - 0.2) < 1e-9   # 4/20 draft compute
    print("train_subset_lora selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llada2")
    ap.add_argument("--probes_root", default="probes_out")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--t3dmax_root", default=None)
    ap.add_argument("--subset", default=None, help="kept layers, e.g. 0,4,18,19 (from Stage-1 search)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", default="all-linear",
                    help="'all-linear' or comma-separated module-name substrings "
                         "(e.g. q_proj,k_proj,v_proj,dense). Use --dry_run to inspect.")
    ap.add_argument("--dora", action="store_true", help="weight-decomposed LoRA (DoRA)")
    ap.add_argument("--rslora", action="store_true", help="rank-stabilized LoRA scaling")
    ap.add_argument("--n_train_samples", type=int, default=80)
    ap.add_argument("--n_val_samples", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_blocks", type=int, default=8)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry_run", action="store_true",
                    help="wrap LoRA + print adapted layers / trainable params, then stop")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.model_path or not args.subset:
        raise SystemExit("--model_path and --subset are required.")
    subset = parse_subset(args.subset)
    targets = args.lora_targets if args.lora_targets == "all-linear" \
        else [t for t in args.lora_targets.split(",") if t]

    from probe_runner.llada2_runner import load_llada2
    model, tokenizer = load_llada2(args.model_path, attn_implementation="sdpa",
                                   t3dmax_root=args.t3dmax_root, device=args.device)
    base = getattr(model, "model", model)
    layers = [m for m in base.modules() if type(m).__name__ == "LLaDA2MoeDecoderLayer"]
    depth = len(layers)
    state = {"keep": None}
    handles = _install_skip_hooks(layers, state)
    print(f"[lora] subset={subset} ({len(subset)}/{depth} layers = "
          f"{compute_fraction(subset, depth):.0%} draft compute); "
          f"variant={'DoRA' if args.dora else ('rsLoRA' if args.rslora else 'LoRA')} r={args.rank}")
    try:
        # 1) teacher targets + iter-1 inputs from the FULL model (no adapters yet)
        samples = [tc.load_sample(p) for p in tc.iter_sample_paths(args.probes_root, args.model)]
        tr_s, va_s = split_samples(samples, args.n_train_samples, args.n_val_samples)
        train_blocks = precompute_blocks(model, tokenizer, tr_s, block_length=args.block_length,
                                         n_samples=args.n_train_samples, device=args.device, state=state)
        val_blocks = precompute_blocks(model, tokenizer, va_s, block_length=args.block_length,
                                       n_samples=args.n_val_samples, device=args.device, state=state)
        print(f"[lora] train blocks={len(train_blocks)}  val blocks={len(val_blocks)}")
        # 2) wrap LoRA on the subset layers
        model = make_lora_model(model, subset, rank=args.rank, alpha=args.alpha,
                                dropout=args.dropout, target_modules=targets,
                                use_dora=args.dora, use_rslora=args.rslora)
        rep = adapted_layer_report(model)
        print(f"[lora] adapted layers={rep['adapted_layers']}  "
              f"trainable={rep['trainable']:,} / {rep['total']:,} "
              f"({100*rep['trainable']/max(rep['total'],1):.3f}%)")
        if set(rep["adapted_layers"]) != set(subset):
            print(f"[lora] WARNING: adapted layers {rep['adapted_layers']} != subset {subset} "
                  f"— check --lora_targets / layers_pattern.")
        if args.dry_run:
            print("[lora] --dry_run: stopping before training.")
            return
        # 3) train
        train(model, subset, train_blocks, val_blocks, block_length=args.block_length,
              device=args.device, state=state, epochs=args.epochs, lr=args.lr,
              batch_blocks=args.batch_blocks, out_dir=args.out_dir)
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
