"""Shared helpers for the T3-D refresh/repair trajectory analyses.

Implements the design doc (T3D_LLADA2_REFRESH_REPAIR_DESIGN.md) substrate used
by plot_rank_trajectory (A7), plot_trajectory_domains (A1,A3-A6),
plot_repairability (B) and plot_refresh_recovery (C):

  * load the per-pass intra-block capture incl. `committed_tokens_per_pass`
    and `converged_tokens`;
  * the 4-domain taxonomy (CU / CR / MC / MM) for an arbitrary iter-pair (a,b)
    with a gap allowed (e.g. 0 vs 2);
  * EOS filtering (drop positions at/after the answer's first EOS);
  * a numpy `RankHead` (final RMSNorm + lm_head) that maps captured last-layer
    hidden -> logits/ranks WITHOUT reloading the 16B model (uses the
    `lm_head.pt` exported by run_probes).

The cores are pure numpy (no torch at import time — torch is only touched
inside RankHead.from_export), so the logic is unit-testable on synthetic HDF5.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

# LLaDA-2.0 family mask id (see T3-DMax handoff §5).
MASK_ID = 156895

# ---- 4-domain taxonomy ------------------------------------------------------
# CU committed-unchanged | CR committed-revised (DMax-only) |
# MC newly-committed | MM still-masked | UC committed->masked (rare; re-masking off)
DOMAINS = ("CU", "CR", "MC", "MM", "UC")
DOMAIN_CODE = {name: i for i, name in enumerate(DOMAINS)}
DOMAIN_COLORS = {
    "CU": "tab:green", "CR": "tab:orange", "MC": "tab:blue",
    "MM": "tab:gray", "UC": "tab:purple",
}


# ---- loading ----------------------------------------------------------------

def iter_sample_paths(probes_root: Path, model: str):
    return sorted((Path(probes_root) / model).glob("sample_*.h5"))


def load_sample(path: Path) -> dict:
    """Read one sample's intra-block capture + the attrs the analyses need.

    Returns {"attrs": {...}, "blocks": {block_idx: {h_per_pass,
    token_state_per_pass, revealed_per_pass, committed_tokens_per_pass,
    converged_tokens, pass_indices}}}. Blocks without `h_per_pass` (no
    --intra_block) are skipped.
    """
    out: dict = {"attrs": {}, "blocks": {}}
    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            if k in {"block_seq_lens", "block_mask_positions",
                     "attention_sink_positions", "special_token_positions"}:
                try:
                    out["attrs"][k] = json.loads(v)
                except Exception:
                    out["attrs"][k] = v
            else:
                out["attrs"][k] = v
        for grp_name in f.keys():
            if not grp_name.startswith("block_"):
                continue
            grp = f[grp_name]
            if "h_per_pass" not in grp:
                continue
            b = int(grp_name.split("_")[1])
            blk = {"h_per_pass": np.asarray(grp["h_per_pass"])}
            for name in ("token_state_per_pass", "revealed_per_pass",
                         "committed_tokens_per_pass", "converged_tokens", "pass_indices"):
                if name in grp:
                    blk[name] = np.asarray(grp[name])
            out["blocks"][b] = blk
    return out


def has_committed(blocks: dict) -> bool:
    """Whether committed_tokens_per_pass was captured (llada2 runs)."""
    return any("committed_tokens_per_pass" in d for d in blocks.values())


# ---- EOS filtering ----------------------------------------------------------

def valid_position_mask(block_idx: int, num_pos: int, attrs: dict,
                        block_length: int) -> np.ndarray:
    """Bool [num_pos]: positions strictly before the answer's first EOS.

    Generated index of block-position j = block_idx*block_length + j (gen
    starts at prompt_len). `eos_pos_in_generated` is in generated coords; if
    absent, all positions are valid.
    """
    eos = attrs.get("eos_pos_in_generated", None)
    base = block_idx * block_length
    gen_idx = base + np.arange(num_pos)
    if eos is None:
        return np.ones(num_pos, dtype=bool)
    return gen_idx < int(eos)


# ---- 4-domain classification -----------------------------------------------

def classify_domains(tok_a: np.ndarray, tok_b: np.ndarray,
                     mask_id: int = MASK_ID) -> np.ndarray:
    """Per-position domain code for iter-pair (a, b) from committed token ids.

    tok_a / tok_b are [num_pos] int arrays (committed token id, or mask_id
    where still masked). Codes per DOMAIN_CODE: CU/CR/MC/MM/UC.
    """
    tok_a = np.asarray(tok_a)
    tok_b = np.asarray(tok_b)
    ca = tok_a != mask_id          # committed at a
    cb = tok_b != mask_id          # committed at b
    same = tok_a == tok_b
    codes = np.full(tok_a.shape, DOMAIN_CODE["MM"], dtype=np.int8)
    codes[ca & cb & same] = DOMAIN_CODE["CU"]
    codes[ca & cb & (~same)] = DOMAIN_CODE["CR"]
    codes[(~ca) & cb] = DOMAIN_CODE["MC"]
    codes[(~ca) & (~cb)] = DOMAIN_CODE["MM"]
    codes[ca & (~cb)] = DOMAIN_CODE["UC"]
    return codes


def committed_tokens_at(blk: dict, pass_idx: int) -> np.ndarray:
    """committed_tokens_per_pass[pass_idx] (the tokens fed INTO that pass)."""
    return np.asarray(blk["committed_tokens_per_pass"][pass_idx])


def converged_tokens(blk: dict, mask_id: int = MASK_ID) -> np.ndarray:
    """Final decode [num_pos]. Prefer the stored converged_tokens; fall back to
    the last pass's input tokens (imperfect for last-pass commits)."""
    if "converged_tokens" in blk:
        return np.asarray(blk["converged_tokens"])
    ct = blk.get("committed_tokens_per_pass")
    if ct is not None:
        return np.asarray(ct[-1])
    raise KeyError("no converged_tokens or committed_tokens_per_pass in block")


def resolve_pair(n_passes: int, a: int, b: int) -> tuple[int, int]:
    """Clamp an iter-pair (a, b) to [0, n_passes-1]; supports negative indices
    (e.g. b=-1 = last pass)."""
    if a < 0:
        a = n_passes + a
    if b < 0:
        b = n_passes + b
    a = max(0, min(a, n_passes - 1))
    b = max(0, min(b, n_passes - 1))
    return a, b


# ---- hidden-state deviation -------------------------------------------------

def pair_deviation(ha: np.ndarray, hb: np.ndarray, metric: str) -> np.ndarray:
    """Per-position deviation between two same-shape hidden arrays (last axis =
    d_model). metric in {l2, l2_normalized, cosine_sim}. Mirrors
    plot_intra_block._pairwise_metric."""
    a = ha.astype(np.float32, copy=False)
    b = hb.astype(np.float32, copy=False)
    if metric == "l2":
        return np.linalg.norm(a - b, axis=-1)
    if metric == "l2_normalized":
        diff = np.linalg.norm(a - b, axis=-1)
        scale = 0.5 * (np.linalg.norm(a, axis=-1) + np.linalg.norm(b, axis=-1))
        return diff / np.maximum(scale, 1e-8)
    if metric == "cosine_sim":
        dot = (a * b).sum(axis=-1)
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        return dot / np.maximum(na * nb, 1e-8)
    raise ValueError(f"unknown metric: {metric}")


# ---- RankHead: hidden -> logits/ranks without reloading the model -----------

class RankHead:
    """Final RMSNorm + lm_head, as numpy, for the rank trajectory (A7) and the
    logit-space deviation (A2). Built from run_probes' `lm_head.pt` export, or
    directly from arrays (tests). logits = rmsnorm(h) @ Wᵀ.
    """

    def __init__(self, lm_head_weight: np.ndarray, norm_weight: np.ndarray,
                 eps: float = 1e-6):
        self.W = np.asarray(lm_head_weight, dtype=np.float32)   # [vocab, d_model]
        self.norm_weight = np.asarray(norm_weight, dtype=np.float32)  # [d_model]
        self.eps = float(eps)
        self.vocab_size = int(self.W.shape[0])
        self.d_model = int(self.W.shape[1])

    @classmethod
    def load(cls, path: str | Path) -> "RankHead":
        """Load an exported head. `.pt` (run_probes' torch export) or `.npz`
        (numpy; used by tests / a torch-free environment)."""
        path = Path(path)
        if path.suffix == ".npz":
            d = np.load(path)
            return cls(d["lm_head_weight"], d["norm_weight"],
                       float(d["rms_norm_eps"]) if "rms_norm_eps" in d else 1e-6)
        import torch  # local import — only needed for the .pt path
        d = torch.load(str(path), map_location="cpu")
        return cls(
            d["lm_head_weight"].float().numpy(),
            d["norm_weight"].float().numpy(),
            float(d.get("rms_norm_eps", 1e-6)),
        )

    # Back-compat alias.
    from_export = load

    def _rmsnorm(self, h: np.ndarray) -> np.ndarray:
        h = h.astype(np.float32, copy=False)
        var = np.mean(h * h, axis=-1, keepdims=True)
        return (h / np.sqrt(var + self.eps)) * self.norm_weight

    def logits(self, h: np.ndarray) -> np.ndarray:
        """[..., d_model] -> [..., vocab]. Materializes the full vocab — call
        per-pass (small) rather than on the whole block at once."""
        hn = self._rmsnorm(h)
        return hn @ self.W.T

    def rank_of(self, h: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
        """Rank (1 = argmax) of each position's `target` token in its logits.

        h: [num_pos, d_model]; target_ids: [num_pos] int. Returns [num_pos] int
        rank = 1 + #{v : logit[v] > logit[target]}. Invalid targets (<0) -> -1.
        """
        target_ids = np.asarray(target_ids)
        logits = self.logits(h)                                # [num_pos, vocab]
        num_pos = logits.shape[0]
        rows = np.arange(num_pos)
        tgt = np.clip(target_ids, 0, self.vocab_size - 1)
        tgt_logit = logits[rows, tgt][:, None]                 # [num_pos, 1]
        rank = (logits > tgt_logit).sum(axis=-1) + 1           # [num_pos]
        rank = np.where(target_ids < 0, -1, rank)
        return rank.astype(np.int64)

    def topk_hit(self, h: np.ndarray, target_ids: np.ndarray, k: int) -> np.ndarray:
        """Bool [num_pos]: is the target token in the top-k of its logits."""
        return (self.rank_of(h, target_ids) <= k) & (np.asarray(target_ids) >= 0)


def find_head_export(probes_root: Path, model: str) -> Path | None:
    """Locate the exported head for a model run (.pt preferred, .npz fallback)."""
    base = Path(probes_root) / model
    for name in ("lm_head.pt", "lm_head.npz"):
        p = base / name
        if p.exists():
            return p
    return None


def progress_iter(iterable, total=None, desc="", min_interval=5.0):
    """Yield items from `iterable`, showing progress + elapsed + ETA so long
    model-driven loops report how far they are. Uses `tqdm` if importable; else
    falls back to printing one line at most every `min_interval` seconds. Pure
    Python (no torch); safe to wrap any loop."""
    import time
    try:
        from tqdm import tqdm                       # nice single-line bar if available
        yield from tqdm(iterable, total=total, desc=desc)
        return
    except Exception:
        pass
    if total is None:
        try:
            total = len(iterable)
        except Exception:
            total = None
    t0 = time.time()
    last = 0.0
    n = 0
    for x in iterable:
        yield x
        n += 1
        now = time.time()
        if now - last >= min_interval or (total and n >= total):
            last = now
            el = now - t0
            rate = n / el if el > 0 else 0.0
            eta = (total - n) / rate if (total and rate > 0) else float("nan")
            tot = total if total is not None else "?"
            print(f"  [{desc}] {n}/{tot}  elapsed={el:.0f}s  eta={eta:.0f}s  ({rate:.2f}/s)",
                  flush=True)
