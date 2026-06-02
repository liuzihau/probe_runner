"""Generate a small, STRUCTURED synthetic probe capture for testing the
trajectory analyses without torch / a GPU / the 16B model.

Writes, under `<root>/llada2/`:
  * sample_0000.h5 … — the same intra-block layout llada2_runner produces
    (h_per_pass, token_state_per_pass, revealed_per_pass,
    committed_tokens_per_pass, converged_tokens, pass_indices + attrs);
  * lm_head.npz — a numpy RankHead export (W, norm_weight, eps) that the rank
    analyses load via _trajectory_common.RankHead.load.

The synthetic decode is built so the analyses have something real to find:
  * positions commit left-to-right over passes (MC progression);
  * one position is committed WRONG then REVISED (a CR event);
  * the last-layer hidden leans toward the token the model "predicts" each
    pass (converged for committed positions, a distractor for masked) with
    confidence growing over passes -> the converged-token rank DESCENDS, and
    committed positions rank lower than still-masked ones;
  * lower-layer hidden drifts more at later layers across passes -> a
    divergence-onset profile for A5.

Run:  python -m probe_runner.tests.make_synthetic_probes /tmp/probes_synth
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

MASK_ID = 156895


def _unit_rows(W: np.ndarray) -> np.ndarray:
    return W / np.maximum(np.linalg.norm(W, axis=-1, keepdims=True), 1e-8)


def build(root: str | Path, *, n_samples: int = 4, num_blocks: int = 2,
          block_length: int = 8, n_layers: int = 4, d_model: int = 16,
          vocab: int = 64, prompt_len: int = 6, n_passes: int = 5,
          seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    out_dir = root / "llada2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared lm_head + final-norm (the RankHead export).
    W = rng.standard_normal((vocab, d_model)).astype(np.float32)
    norm_weight = np.ones(d_model, dtype=np.float32)
    eps = 1e-6
    np.savez(out_dir / "lm_head.npz", lm_head_weight=W.astype(np.float16),
             norm_weight=norm_weight, rms_norm_eps=np.float32(eps))
    Wn = _unit_rows(W)

    B, L, D, P = block_length, n_layers, d_model, n_passes

    for si in range(n_samples):
        path = out_dir / f"sample_{si:04d}.h5"
        block_seq_lens, block_mask_positions = [], []
        with h5py.File(path, "w") as f:
            for b in range(num_blocks):
                converged = rng.integers(0, vocab, size=B)            # final tokens
                distractor = rng.integers(0, vocab, size=B)           # what masked positions lean to

                # commit schedule, left-to-right. schedule[k] = #committed at the
                # INPUT of pass k (schedule[0]=0 → pass 0 sees all-mask; schedule[P]=B).
                # revealed_per_pass[k] = positions committed DURING pass k (matches
                # the runner), so pass 0 commits a real prefix.
                schedule = np.linspace(0, B, P + 1).astype(int)       # [P+1]
                idx = np.arange(B)
                # CR event: position 0 committed WRONG while masked-free, revised at pass 3.
                wrong_tok = (converged[0] + 7) % vocab

                ct_per_pass = np.full((P, B), MASK_ID, dtype=np.int64)
                ts_per_pass = np.zeros((P, B), dtype=np.int8)
                rv_per_pass = np.zeros((P, B), dtype=bool)
                h_per_pass = np.zeros((P, L + 1, B, D), dtype=np.float32)

                for k in range(P):
                    committed = idx < schedule[k]                     # input state at pass k
                    revealed_now = (idx >= schedule[k]) & (idx < schedule[k + 1])
                    toks = np.where(committed, converged, MASK_ID).astype(np.int64)
                    if committed[0] and k < 3:
                        toks[0] = wrong_tok                            # CR: wrong until pass 3
                    ct_per_pass[k] = toks
                    ts_per_pass[k] = committed.astype(np.int8)
                    rv_per_pass[k] = revealed_now

                    # ---- hidden ----
                    # what the model "leans toward" at this pass, per position
                    lean = np.where(committed, converged, distractor)
                    if committed[0] and k < 3:
                        lean[0] = wrong_tok
                    strength = 1.0 + 2.5 * (k / max(P - 1, 1))         # confidence grows
                    last = strength * Wn[lean] + 0.25 * rng.standard_normal((B, D))
                    h_per_pass[k, L] = last
                    # lower layers: base + pass-drift that grows with layer depth
                    for ell in range(L):
                        base = rng.standard_normal((B, D)) * 0.5
                        drift = (ell / max(L - 1, 1)) * k * 0.4 * rng.standard_normal((B, D))
                        h_per_pass[k, ell] = base + drift

                grp = f.create_group(f"block_{b}")
                grp.create_dataset("h_per_pass", data=h_per_pass.astype(np.float16))
                grp.create_dataset("token_state_per_pass", data=ts_per_pass)
                grp.create_dataset("revealed_per_pass", data=rv_per_pass)
                grp.create_dataset("committed_tokens_per_pass", data=ct_per_pass)
                grp.create_dataset("converged_tokens", data=converged.astype(np.int64))
                grp.create_dataset("pass_indices", data=np.arange(P, dtype=np.int32))

                s = prompt_len + block_length * b
                block_seq_lens.append(prompt_len + block_length * (b + 1))
                block_mask_positions.append(list(range(s, s + block_length)))

            f.attrs["prompt_len"] = prompt_len
            f.attrs["num_masked"] = block_length
            f.attrs["num_blocks"] = num_blocks
            f.attrs["block_seq_lens"] = json.dumps(block_seq_lens)
            f.attrs["block_mask_positions"] = json.dumps(block_mask_positions)
            f.attrs["model_name"] = "llada2"
            f.attrs["n_layers"] = n_layers
            f.attrs["n_heads"] = 2
            f.attrs["d_model"] = d_model
            # EOS near the end of block 1 so a couple of trailing positions get filtered.
            f.attrs["eos_pos_in_generated"] = num_blocks * block_length - 2
    print(f"wrote {n_samples} synthetic samples + lm_head.npz to {out_dir}")
    return out_dir


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/tmp/probes_synth"
    build(target)
