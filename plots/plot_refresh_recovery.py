"""C — refresh cost: how few & how shallow think-refreshes recover the hard tail.

Two parts (design doc §2 C):

  C11-C13 (capture-only, parameter-free):  the heavy model's per-pass recovery
      curve toward the converged answer.
        * think-pred recovery(k)  = fraction whose argmax(think logits @ pass k)
                                    == converged  (rank==1)  — "if we refreshed
                                    the anchor at pass k, would it be right now".
        * committed recovery(k)   = fraction already committed to the converged
                                    token at pass k.
        * by domain (committed vs still-masked at pass k).
      The KNEE of recovery(k) is the anchor-refresh cadence M: if think nails the
      answer by ~pass 2-3, refreshing every few iters suffices; if it crawls,
      there's no compute saving.

  C15 (model-driven):  partial-depth refresh fidelity. For a cut-layer L, reuse
      iter-0's hidden for layers < L and recompute only L..top with the iter-1
      reveals; measure recovery of the iter-1 prediction vs L. Needs the model +
      output_hidden_states (the captured hidden is block-position-only, so this
      runs its own forwards). A late knee ⇒ a shallow refresh ~ (depth−L)/depth
      of a full think pass.

The capture-only cores are unit-tested via `--selftest`; C15 needs a GPU + model.

Usage:
    python -m probe_runner.plots.plot_refresh_recovery --model llada2
    python -m probe_runner.plots.plot_refresh_recovery --model llada2 \
        --partial_depth --model_path /path/to/DMax-Math-16B
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc


# ---- C11-C13: capture-only recovery curve -----------------------------------

def sample_recovery(sample: dict, head: tc.RankHead, block_length: int,
                    mask_id: int = tc.MASK_ID) -> dict[int, dict]:
    """Per block: think_correct[pass, pos] (bool), committed_correct[pass, pos],
    committed[pass, pos], valid[pos]."""
    out: dict[int, dict] = {}
    for b, blk in sample["blocks"].items():
        if "committed_tokens_per_pass" not in blk:
            continue
        h = blk["h_per_pass"]                          # [P, L+1, B, D]
        P, _, B, _ = h.shape
        conv = tc.converged_tokens(blk, mask_id)
        ct = np.asarray(blk["committed_tokens_per_pass"])           # [P, B]
        valid = tc.valid_position_mask(b, B, sample["attrs"], block_length)
        think_correct = np.zeros((P, B), bool)
        for k in range(P):
            think_correct[k] = head.rank_of(h[k, -1], conv) == 1     # argmax hit
        committed_correct = (ct == conv[None, :])
        committed = (ct != mask_id)
        out[b] = {"think_correct": think_correct, "committed_correct": committed_correct,
                  "committed": committed, "valid": valid, "n_passes": P}
    return out


def accumulate_recovery(probes_root: Path, model: str, head: tc.RankHead,
                        block_length: int) -> dict:
    paths = tc.iter_sample_paths(probes_root, model)
    if not paths:
        raise RuntimeError(f"No samples under {Path(probes_root)/model}")
    Pmax = 0
    # sums over (pass): think hits, committed hits, counts; + by-domain think hits
    acc = {"think": None, "comm": None, "cnt": None,
           "think_oncommitted": None, "cnt_committed": None,
           "think_onmasked": None, "cnt_masked": None}
    n_samples = 0

    def _g(a, P):
        if a is None:
            return np.zeros(P)
        if a.shape[0] < P:
            n = np.zeros(P); n[:a.shape[0]] = a; return n
        return a

    for path in paths:
        sample = tc.load_sample(path)
        if not tc.has_committed(sample["blocks"]):
            continue
        data = sample_recovery(sample, head, block_length)
        if not data:
            continue
        n_samples += 1
        for b, d in data.items():
            P = d["n_passes"]
            Pmax = max(Pmax, P)
            for key in acc:
                acc[key] = _g(acc[key], Pmax)
            valid = d["valid"]
            for k in range(P):
                tc_k = d["think_correct"][k] & valid
                comm_k = d["committed"][k] & valid
                cc_k = d["committed_correct"][k] & valid
                acc["think"][k] += int(tc_k.sum())
                acc["comm"][k] += int(cc_k.sum())
                acc["cnt"][k] += int(valid.sum())
                acc["think_oncommitted"][k] += int((tc_k & comm_k).sum())
                acc["cnt_committed"][k] += int(comm_k.sum())
                acc["think_onmasked"][k] += int((tc_k & ~comm_k & valid).sum())
                acc["cnt_masked"][k] += int(((~d["committed"][k]) & valid).sum())
    return {"acc": acc, "Pmax": Pmax, "n_samples": n_samples}


def plot_recovery(stats: dict, out_dir: Path, model: str) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    acc, Pmax = stats["acc"], stats["Pmax"]
    if not Pmax:
        raise RuntimeError("No recovery data — need a llada2 --intra_block capture.")
    xs = np.arange(Pmax)

    def rate(num, den):
        num = np.asarray(num, float); den = np.asarray(den, float)
        with np.errstate(invalid="ignore"):
            return np.where(den > 0, num / np.maximum(den, 1), np.nan)

    think = rate(acc["think"], acc["cnt"])
    comm = rate(acc["comm"], acc["cnt"])
    think_c = rate(acc["think_oncommitted"], acc["cnt_committed"])
    think_m = rate(acc["think_onmasked"], acc["cnt_masked"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.plot(xs, think, "o-", color="tab:red", label="think argmax == converged")
    ax.plot(xs, comm, "s-", color="tab:blue", label="committed == converged")
    ax.set_ylim(0, 1.02)
    ax.set_title("C11/C12: recovery vs pass\n(knee = anchor-refresh cadence M)")
    ax.set_xlabel("decode pass (≈ #think refreshes)")
    ax.set_ylabel("fraction recovered")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(xs, think_c, "o-", color="tab:green", label="committed positions")
    ax.plot(xs, think_m, "o-", color="tab:gray", label="still-masked positions")
    ax.set_ylim(0, 1.02)
    ax.set_title("C13: think recovery by domain")
    ax.set_xlabel("decode pass")
    ax.set_ylabel("fraction think-correct")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"{model}: C refresh recovery  [n_samples={stats['n_samples']}]")
    fig.tight_layout()
    p = out_dir / "refresh_recovery.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")


# ---- C15: partial-depth refresh fidelity (model-driven) ---------------------

def run_partial_depth(model_path: str, samples: list, *, block_length: int = 32,
                      n_samples: int = 10, t3dmax_root=None, device: str = "cuda") -> dict:
    """For each cut-layer L: reuse iter-0's per-layer hidden for layers < L,
    recompute L..top with the iter-1 reveals, and measure how well the resulting
    top prediction matches the TRUE iter-1 think prediction (full recompute).

    Implemented with forward hooks that, when active, REPLACE a decoder layer's
    output with the cached iter-0 hidden (for layers < L). Returns
    {fidelity[L]} over L = 0..depth (L=0 = full refresh = 1.0 by construction).
    Needs the model — validated on the cloud.
    """
    import torch
    from probe_runner import configs
    from probe_runner.llada2_runner import (
        load_llada2, _build_block_causal_mask, _commit_uniform)

    model, tokenizer = load_llada2(model_path, attn_implementation="sdpa",
                                   t3dmax_root=t3dmax_root, device=device)
    base = getattr(model, "model", model)
    layers = [m for m in base.modules() if type(m).__name__ == "LLaDA2MoeDecoderLayer"]
    depth = len(layers)
    emb = base.word_embeddings if hasattr(base, "word_embeddings") else model.get_input_embeddings()
    dtype = next(model.parameters()).dtype

    # hook state: cache[layer_idx] = iter-0 hidden to inject; cut_layer = L;
    # reveal = bool mask [seq] of positions whose iter-1 input changed (the new
    # commits) — these must keep their FRESH compute so the reveal propagates.
    cache: dict[int, "torch.Tensor"] = {}
    state = {"cut": None, "record": False, "reveal": None}
    handles = []

    def mk_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if state["record"]:
                cache[idx] = h.detach()
            elif state["cut"] is not None and idx < state["cut"]:
                # reuse iter-0 hidden for unchanged positions, but PRESERVE the
                # freshly-computed hidden at revealed positions so their new tokens
                # propagate up the stack (else the cut layer changes nothing).
                h_new = cache[idx].clone()
                rev = state["reveal"]
                if rev is not None:
                    h_new[:, rev] = h[:, rev]
                return (h_new,) + out[1:] if isinstance(out, tuple) else h_new
            return None
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(mk_hook(i)))

    def _format(q):
        msg = [{"role": "user", "content": q}]
        p = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        return tokenizer(p, return_tensors="pt")["input_ids"].to(device)

    def _think_pred(input_ids, block_end, s, e):
        mask = _build_block_causal_mask(block_end, Lp, block_length, dtype, device)
        pos = torch.arange(block_end, device=device).unsqueeze(0)
        out = model(inputs_embeds=emb(input_ids[:, :block_end]), attention_mask=mask,
                    position_ids=pos, use_cache=False, return_dict=True)
        return out.logits[:, s:e].argmax(-1)[0]                     # [B]

    fid_sum = np.zeros(depth + 1)
    fid_cnt = 0
    for sample in samples[:n_samples]:
        prompt_ids = _format(sample["attrs"]["prompt_text"])
        Lp = int(prompt_ids.shape[1])
        for b, blk in sorted(sample["blocks"].items()):
            s, e = Lp + b * block_length, Lp + (b + 1) * block_length
            conv = torch.tensor(tc.converged_tokens(blk), device=device, dtype=torch.long)
            iter0 = torch.full((1, e), tc.MASK_ID, dtype=torch.long, device=device)
            iter0[0, :Lp] = prompt_ids[0, :Lp]
            for j in range(b):
                iter0[0, Lp + j * block_length: Lp + (j + 1) * block_length] = \
                    torch.tensor(tc.converged_tokens(sample["blocks"][j]), device=device)
            # iter-0: record the per-layer hidden
            with torch.no_grad():
                state["record"], state["cut"] = True, None
                _ = _think_pred(iter0, e, s, e)
                state["record"] = False
                # iter-1: commit the iter-0 high-conf prefix as the "reveal"
                mask_blk = _build_block_causal_mask(e, Lp, block_length, dtype, device)
                pos = torch.arange(e, device=device).unsqueeze(0)
                logits0 = model(inputs_embeds=emb(iter0), attention_mask=mask_blk,
                                position_ids=pos, use_cache=False, return_dict=True).logits[:, s:e]
                x0, hc, _ = _commit_uniform(logits0, iter0[:, s:e], tc.MASK_ID, 0.3)
                iter1 = iter0.clone()
                iter1[0, s:e][hc[0]] = x0[0][hc[0]]
                # measure recovery on the STILL-MASKED tail only: the committed
                # (revealed) positions keep fresh compute and trivially match at
                # every L, which would flatten/inflate the curve. The tail is the
                # population a refresh is FOR.
                valid = torch.tensor(tc.valid_position_mask(b, e - s, sample["attrs"],
                                                            block_length), device=device)
                tail = valid & ~hc[0]
                if tail.float().sum() == 0:
                    continue
                # revealed positions (absolute) = the freshly-committed block tokens
                reveal = torch.zeros(e, dtype=torch.bool, device=device)
                reveal[s + torch.where(hc[0])[0]] = True
                state["reveal"] = reveal
                # true iter-1 prediction (full recompute)
                state["cut"] = None
                true_pred = _think_pred(iter1, e, s, e)
                # partial-depth at each cut L: reuse iter-0 for masked tail < L,
                # recompute L..top; reveals propagate via the preserved-fresh hook.
                for L in range(depth + 1):
                    state["cut"] = L
                    pred = _think_pred(iter1, e, s, e)
                    match = ((pred == true_pred) & tail).float().sum() / tail.float().sum()
                    fid_sum[L] += float(match)
                state["cut"], state["reveal"] = None, None
            fid_cnt += 1
    for h in handles:
        h.remove()
    return {"fidelity": fid_sum / max(fid_cnt, 1), "depth": depth, "n_blocks": fid_cnt}


def plot_partial_depth(pd: dict, out_dir: Path, model: str) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    fid = pd["fidelity"]
    L = np.arange(len(fid))
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(L, fid, "o-", color="tab:blue")
    ax.axhline(0.95, ls="--", color="black", alpha=0.5, label="0.95 fidelity")
    below = np.where(fid >= 0.95)[0]
    if below.size:
        knee = int(below.max())   # deepest cut still >=0.95 fidelity
        ax.axvline(knee, ls=":", color="red",
                   label=f"cut L={knee} → refresh ~{(pd['depth']-knee)/pd['depth']:.0%} of think")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{model}: C15 partial-depth refresh fidelity\n"
                 f"reuse iter-0 layers <L, recompute L..top with iter-1 reveals")
    ax.set_xlabel("cut layer L")
    ax.set_ylabel("fidelity to full-recompute iter-1 prediction")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "partial_depth_refresh.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")


def _selftest():
    rng = np.random.default_rng(0)
    # synthetic: a sample where think gets steadily more correct.
    V, D = 40, 8
    W = rng.standard_normal((V, D)).astype(np.float32)
    head = tc.RankHead(W, np.ones(D, np.float32))
    Wn = W / np.linalg.norm(W, axis=-1, keepdims=True)
    P, B = 5, 6
    conv = rng.integers(0, V, B)
    h = np.zeros((P, 2, B, D), np.float32)   # L+1 = 2 (embed + 1 layer); use last
    for k in range(P):
        strength = 0.2 + 1.5 * k
        h[k, -1] = strength * Wn[conv] + 0.1 * rng.standard_normal((B, D))
    ct = np.tile(np.where(np.arange(B) < 0, conv, tc.MASK_ID), (P, 1))
    sample = {"attrs": {"eos_pos_in_generated": 999},
              "blocks": {0: {"h_per_pass": h, "committed_tokens_per_pass": ct,
                             "converged_tokens": conv}}}
    data = sample_recovery(sample, head, block_length=B)
    tc_acc = data[0]["think_correct"].mean(axis=1)
    assert tc_acc[-1] > tc_acc[0], tc_acc           # recovery rises
    print("plot_refresh_recovery selftest OK  think_correct/pass:", np.round(tc_acc, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llada2")
    ap.add_argument("--probes_root", default="probes_out")
    ap.add_argument("--head_path", default=None)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--partial_depth", action="store_true",
                    help="Also run the C15 partial-depth fidelity (needs --model_path + GPU).")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--t3dmax_root", default=None)
    ap.add_argument("--n_pd_samples", type=int, default=10)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    out_dir = Path(args.probes_root) / args.model / "plots"
    head_path = args.head_path or tc.find_head_export(args.probes_root, args.model)
    if head_path is None:
        raise SystemExit("No lm_head export found; pass --head_path or re-run capture.")
    head = tc.RankHead.load(head_path)
    stats = accumulate_recovery(Path(args.probes_root), args.model, head, args.block_length)
    plot_recovery(stats, out_dir, args.model)

    if args.partial_depth:
        if not args.model_path:
            raise SystemExit("--partial_depth needs --model_path (the LLaDA-2.0/DMax weights).")
        samples = [tc.load_sample(p) for p in tc.iter_sample_paths(args.probes_root, args.model)]
        samples = [s for s in samples if tc.has_committed(s["blocks"])]
        pd = run_partial_depth(args.model_path, samples, block_length=args.block_length,
                               n_samples=args.n_pd_samples, t3dmax_root=args.t3dmax_root)
        plot_partial_depth(pd, out_dir, args.model)


if __name__ == "__main__":
    main()
