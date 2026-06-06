"""Step 0 — is the un-extractable tail *load-bearing* or filler?

The validity gate from T3D_RERANKING_STRATEGY.md §6 / R5, in its cheap, read-only
form. The rank probe says think's iter-0 top-K carries the converged token for
~75% of the hard tail at K=10 (98% at K=100). The reranker can therefore *never*
recover the complementary tail — the positions whose converged token sits at
rank > K in the iter-0 logits. This script asks the only question that decides
whether that miss matters: are those un-reachable tokens **answer digits / math
operators** (load-bearing — losing them wrecks GSM8K accuracy) or **filler**
("the", "so", connectives — losing them is cosmetic)?

It needs NEITHER gold answers NOR a trained talk NOR the 16B model: just the
captured iter-0 hidden + the exported `lm_head` head (for the rank) + the
tokenizer (to decode each converged id to text for typing). Everything is the
same offline substrate the rank trajectory (A7) already uses.

Definitions
-----------
* iter-0 rank r0(p) = rank of the converged token in pass-0 last-layer logits
  (RankHead) — the same quantity readiness_bucket() splits on.
* miss@K = {p : r0(p) > K}  — the reranker's unreachable set at candidate budget K.
* token type of p = classify_token_type(decode(converged_id(p))) ∈
  {digit, operator, word, punct, other}; load-bearing = {digit, operator}.

Headline numbers (printed + plotted)
------------------------------------
* composition of miss@10 / miss@100 by token type vs the overall composition;
* load-bearing miss-rate: of all digit/operator tokens, what fraction land in
  miss@K — i.e. are the answer-carrying tokens *disproportionately* unreachable?
  (the scary number: high ⇒ extraction alone won't buy accuracy, §3 fallback.)

Usage
-----
    python -m probe_runner.plots.plot_miss_types --model llada2 \
        --probes_root probes_out --tokenizer /path/to/DMax/ckpt
    python -m probe_runner.plots.plot_miss_types --selftest      # numpy/py cores
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from probe_runner.plots import _trajectory_common as tc


# ---- token typing (pure python, unit-testable) ------------------------------

# Symbols that carry arithmetic meaning in a GSM8K solution.
_OPERATOR_CHARS = set("+-*/=<>%^×÷·±≤≥≠√")
# Leading sub-word markers (sentencepiece ▁, GPT-BPE Ġ) and whitespace to strip.
_LEAD_STRIP = "▁Ġ \t\n\r"

TOKEN_TYPES = ("digit", "operator", "word", "punct", "other")
TYPE_CODE = {t: i for i, t in enumerate(TOKEN_TYPES)}
LOAD_BEARING = ("digit", "operator")
LOAD_BEARING_CODES = tuple(TYPE_CODE[t] for t in LOAD_BEARING)


def classify_token_type(text: str) -> str:
    """Coarse type of a decoded token string.

    digit    — contains any 0-9 (numbers and number-fragments are answer-carrying)
    operator — non-empty and every non-space char is an arithmetic/relational symbol
    word     — contains an alphabetic character
    punct    — contains punctuation but no digit/letter/operator
    other    — whitespace-only / empty / unprintable
    """
    s = text.lstrip(_LEAD_STRIP).rstrip(" \t\n\r")
    if s == "":
        return "other"
    if any(ch.isdigit() for ch in s):
        return "digit"
    if all(ch in _OPERATOR_CHARS for ch in s):
        return "operator"
    if any(ch.isalpha() for ch in s):
        return "word"
    if any((not ch.isspace()) for ch in s):
        return "punct"
    return "other"


def type_codes(texts) -> np.ndarray:
    """[N] int type codes from an iterable of decoded token strings."""
    return np.array([TYPE_CODE[classify_token_type(t)] for t in texts], dtype=np.int8)


# ---- miss-type statistics (pure numpy, unit-testable) -----------------------

def readiness_bucket(rank0: np.ndarray) -> np.ndarray:
    """0=easy(r=1) 1=medium(r<=10) 2=hard(r>10). Mirrors plot_repairability."""
    rank0 = np.asarray(rank0)
    out = np.full(rank0.shape, 2, dtype=np.int8)
    out[rank0 <= 10] = 1
    out[rank0 <= 1] = 0
    return out


def _composition(codes: np.ndarray) -> np.ndarray:
    """Fraction of each TOKEN_TYPES code in `codes` ([len(TOKEN_TYPES)])."""
    n = max(len(codes), 1)
    return np.array([(codes == c).sum() / n for c in range(len(TOKEN_TYPES))])


def miss_type_stats(rank0: np.ndarray, codes: np.ndarray,
                    ceilings=(10, 100)) -> dict:
    """The Step-0 headline table.

    rank0 [N] iter-0 rank of converged token; codes [N] token-type codes.
    Returns per-K composition of miss@K, the overall composition, and the
    load-bearing miss-rate (P(r0>K | load-bearing)).
    """
    rank0 = np.asarray(rank0)
    codes = np.asarray(codes)
    n = int(rank0.shape[0])
    lb = np.isin(codes, LOAD_BEARING_CODES)            # [N] load-bearing positions
    out = {
        "n": n,
        "overall_comp": _composition(codes),
        "load_bearing_share": float(lb.mean()) if n else float("nan"),
        "per_k": {},
    }
    for k in ceilings:
        miss = rank0 > k                                # unreachable at budget k
        miss_codes = codes[miss]
        out["per_k"][k] = {
            "miss_count": int(miss.sum()),
            "miss_rate": float(miss.mean()) if n else float("nan"),
            "miss_comp": _composition(miss_codes),
            # of all load-bearing tokens, fraction that are unreachable at k:
            "lb_miss_rate": float((miss & lb).sum() / max(lb.sum(), 1)),
            # of all filler tokens, fraction unreachable (for contrast):
            "filler_miss_rate": float((miss & ~lb).sum() / max((~lb).sum(), 1)),
            # composition of the miss tail collapsed to load-bearing vs filler:
            "miss_load_bearing_frac": float(
                np.isin(miss_codes, LOAD_BEARING_CODES).mean()) if miss.any() else float("nan"),
        }
    return out


def bucket_type_table(rank0: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """[3 buckets x len(TOKEN_TYPES)] count cross-tab (easy/medium/hard x type)."""
    bucket = readiness_bucket(rank0)
    tbl = np.zeros((3, len(TOKEN_TYPES)), dtype=np.int64)
    for bk in range(3):
        for c in range(len(TOKEN_TYPES)):
            tbl[bk, c] = int(((bucket == bk) & (codes == c)).sum())
    return tbl


# ---- offline data driver (head + tokenizer, no GPU/model) -------------------

def compute_miss_types(probes_root, model: str, head: tc.RankHead, decode_fn,
                       block_length: int = 32, n_samples: int | None = None) -> dict:
    """Accumulate (rank0, type_code) over every valid block position.

    decode_fn(token_id:int)->str maps a converged id to text (tokenizer.decode in
    real runs; an injected map in tests). Returns {"rank0", "codes"} arrays.
    """
    rank0_all: list[np.ndarray] = []
    code_all: list[np.ndarray] = []
    done = 0
    for path in tc.iter_sample_paths(probes_root, model):
        sample = tc.load_sample(path)
        attrs = sample["attrs"]
        for b, blk in sorted(sample["blocks"].items()):
            if "converged_tokens" not in blk and "committed_tokens_per_pass" not in blk:
                continue
            h = blk["h_per_pass"]                          # [P, L+1, B, D]
            B = h.shape[2]
            conv = tc.converged_tokens(blk)                # [B]
            r0 = head.rank_of(h[0, -1], conv)              # iter-0 (pass 0, last layer)
            valid = tc.valid_position_mask(b, B, attrs, block_length)
            valid &= (np.asarray(conv) != tc.MASK_ID)
            if not valid.any():
                continue
            codes = type_codes(decode_fn(int(t)) for t in np.asarray(conv)[valid])
            rank0_all.append(np.asarray(r0)[valid])
            code_all.append(codes)
        done += 1
        if n_samples and done >= n_samples:
            break
    return {
        "rank0": np.concatenate(rank0_all) if rank0_all else np.array([], dtype=np.int64),
        "codes": np.concatenate(code_all) if code_all else np.array([], dtype=np.int8),
    }


# ---- report + plot ----------------------------------------------------------

def format_report(stats: dict) -> str:
    lines = [f"Step-0 miss-type audit  [N={stats['n']} positions, "
             f"load-bearing share={stats['load_bearing_share']:.1%}]"]
    overall = stats["overall_comp"]
    lines.append("  overall composition: " +
                 ", ".join(f"{t}={overall[i]:.1%}" for i, t in enumerate(TOKEN_TYPES)))
    for k, d in stats["per_k"].items():
        lines.append(
            f"  miss@{k:<3d} (r0>{k}): {d['miss_rate']:.1%} of positions | "
            f"of that tail {d['miss_load_bearing_frac']:.1%} load-bearing")
        lines.append(
            f"            load-bearing miss-rate P(r0>{k}|digit/op)={d['lb_miss_rate']:.1%}  "
            f"vs filler {d['filler_miss_rate']:.1%}  "
            f"→ {'⚠ tail is answer-carrying' if d['lb_miss_rate'] > d['filler_miss_rate'] else '✓ tail is mostly filler'}")
    return "\n".join(lines)


def plot(stats: dict, table: np.ndarray, out_dir: Path, model: str) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    xs = np.arange(len(TOKEN_TYPES))

    # 1) composition: overall vs miss@K
    ax = axes[0]
    ax.bar(xs - 0.25, stats["overall_comp"], width=0.25, label="overall", color="tab:gray")
    for off, (k, d) in zip((0.0, 0.25), stats["per_k"].items()):
        ax.bar(xs + off, d["miss_comp"], width=0.25, label=f"miss@{k}")
    ax.set_xticks(xs); ax.set_xticklabels(TOKEN_TYPES, fontsize=8, rotation=20)
    ax.set_ylabel("share"); ax.set_title("token-type composition:\noverall vs unreachable tail")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2) load-bearing vs filler miss-rate per K (the scary number)
    ax = axes[1]
    ks = list(stats["per_k"].keys())
    lb = [stats["per_k"][k]["lb_miss_rate"] for k in ks]
    fl = [stats["per_k"][k]["filler_miss_rate"] for k in ks]
    xk = np.arange(len(ks))
    ax.bar(xk - 0.2, lb, width=0.4, color="tab:red", label="load-bearing (digit/op)")
    ax.bar(xk + 0.2, fl, width=0.4, color="tab:blue", label="filler")
    ax.set_xticks(xk); ax.set_xticklabels([f"miss@{k}" for k in ks])
    ax.set_ylabel("P(unreachable | type)")
    ax.set_title("miss-rate by token class\n(red ≫ blue ⇒ extraction won't buy accuracy)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3) readiness-bucket x type cross-tab (row-normalised)
    ax = axes[2]
    row = table / np.maximum(table.sum(1, keepdims=True), 1)
    bottom = np.zeros(3)
    for c, t in enumerate(TOKEN_TYPES):
        ax.bar(np.arange(3), row[:, c], bottom=bottom, label=t)
        bottom += row[:, c]
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(["easy r=1", "med r≤10", "hard r>10"], fontsize=8)
    ax.set_ylabel("share within bucket")
    ax.set_title("type mix per iter-0 readiness bucket")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"{model}: Step-0 miss-type audit  [N={stats['n']}]")
    fig.tight_layout()
    p = out_dir / "miss_types.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")


# ---- selftest ---------------------------------------------------------------

def _selftest() -> None:
    # token typing
    assert classify_token_type("▁42") == "digit"
    assert classify_token_type(" 3.14") == "digit"
    assert classify_token_type("=") == "operator"
    assert classify_token_type(" +") == "operator"
    assert classify_token_type("apples") == "word"
    assert classify_token_type(" Ġthe") == "word"
    assert classify_token_type(".") == "punct"
    assert classify_token_type("   ") == "other"

    # construct a population where load-bearing tokens are DELIBERATELY hard:
    # 200 filler at rank 1, 100 digits at rank 50 (unreachable at K=10).
    rank0 = np.concatenate([np.ones(200, int), np.full(100, 50)])
    codes = np.concatenate([np.full(200, TYPE_CODE["word"], np.int8),
                            np.full(100, TYPE_CODE["digit"], np.int8)])
    st = miss_type_stats(rank0, codes, ceilings=(10, 100))
    d10 = st["per_k"][10]
    assert d10["miss_count"] == 100
    assert abs(d10["miss_load_bearing_frac"] - 1.0) < 1e-9      # tail is all digits
    assert d10["lb_miss_rate"] > 0.99 and d10["filler_miss_rate"] < 1e-9
    assert st["per_k"][100]["miss_count"] == 0                  # all within top-100
    tbl = bucket_type_table(rank0, codes)
    assert tbl[0, TYPE_CODE["word"]] == 200 and tbl[2, TYPE_CODE["digit"]] == 100

    # mirror-image safe case: digits all easy, filler in the tail
    rank0b = np.concatenate([np.ones(100, int), np.full(200, 50)])
    codesb = np.concatenate([np.full(100, TYPE_CODE["digit"], np.int8),
                             np.full(200, TYPE_CODE["word"], np.int8)])
    sb = miss_type_stats(rank0b, codesb, ceilings=(10,))
    assert sb["per_k"][10]["lb_miss_rate"] < 1e-9               # digits all reachable
    assert sb["per_k"][10]["miss_load_bearing_frac"] < 1e-9
    print("plot_miss_types selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llada2")
    ap.add_argument("--probes_root", default="probes_out")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer dir/name to decode converged ids (e.g. the DMax ckpt).")
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--ceilings", type=int, nargs="+", default=[10, 100])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    head_path = tc.find_head_export(args.probes_root, args.model)
    if head_path is None:
        raise SystemExit(f"no lm_head export under {args.probes_root}/{args.model}")
    head = tc.RankHead.load(head_path)
    if not args.tokenizer:
        raise SystemExit("--tokenizer is required to decode converged ids into token text.")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    decode_fn = lambda tid: tok.decode([tid])
    data = compute_miss_types(args.probes_root, args.model, head, decode_fn,
                              block_length=args.block_length, n_samples=args.n_samples)
    stats = miss_type_stats(data["rank0"], data["codes"], ceilings=tuple(args.ceilings))
    table = bucket_type_table(data["rank0"], data["codes"])
    print(format_report(stats))
    plot(stats, table, Path(args.probes_root) / args.model / "plots", args.model)


if __name__ == "__main__":
    main()
