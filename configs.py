"""Single source of truth for probe-run hyperparameters.

See ../T3_pruning_probe_step1to4.md §8 for the spec this mirrors.
"""

from __future__ import annotations

import os
from pathlib import Path

PROBE_CONFIG = {
    "models": {
        "llada": {
            "hf_name": "GSAI-ML/LLaDA-8B-Instruct",
            # Resolved at load time and written into meta.json:
            "n_layers": 32,
            "n_heads": 32,
            "d_model": 4096,
            "d_head": 128,
            "mask_token_id": 126336,
        },
        "dream": {
            "hf_name": "Dream-org/Dream-v0-Instruct-7B",
            # Resolved at load time:
            "n_layers": None,
            "n_heads": None,
            "d_model": None,
            "d_head": None,
            "mask_token_id": None,
        },
    },
    "dataset": {
        "name": "gsm8k",
        "config": "main",
        "split": "test",
        "n_samples": 100,
    },
    "generation": {
        "gen_length": 256,
        "block_length": 32,
        "steps": 256,            # Fast-dLLM v1 convention: total steps; per-block = steps / num_blocks.
                                  # With block_length=32 and num_blocks=8, steps_per_block=32.
                                  # Parallel decoding may finish a block in <32 forwards via confidence threshold.
        "use_prefix_cache": True,
        "max_prompt_tokens": 512,
        "temperature": 0.0,
        "remasking": "low_confidence",
        # "threshold" enables Fast-dLLM v1 parallel decoding when set; None = sequential.
        "threshold": 0.9,
    },
    "probe": {
        # "all" → record all 8 blocks; or list[int] like [0, 3, 7] to subsample.
        "record_blocks": "all",
        # "all" → save v_norm per block; or [0] to save only block 0's v_norm.
        "v_norm_blocks": "all",
        "attn_dtype": "float16",
        "v_norm_dtype": "float32",
        "h_masked_dtype": "float16",
        # "eager" or "manual_softmax" — both expose attention weights.
        "attn_implementation": "manual_softmax",
    },
    "output": {
        # Relative to the cwd (where you launch python -m probe_runner.run_probes).
        "root": "probes_out",
    },
}


def derived(num_blocks: int = 8) -> dict:
    """Convenience: derived values from PROBE_CONFIG."""
    g = PROBE_CONFIG["generation"]
    return {
        "num_blocks": num_blocks,
        "gen_length": g["gen_length"],
        "block_length": g["block_length"],
        "steps_per_block": g["steps"] // num_blocks,
    }


# ----------------------------------------------------------------------
# Fast-dLLM v1 path resolution
#
# probe_runner depends on Fast-dLLM v1's `model.modeling_llada` and `model.modeling_dream` Python
# modules. These are NOT bundled in this repo. The user clones Fast-dLLM separately (see ../README.md),
# and we look it up via (in order of priority):
#   1. explicit path passed at the call site
#   2. environment variable FAST_DLLM_V1_PATH
#   3. default: ./external/Fast-dLLM/v1   (relative to cwd)
# ----------------------------------------------------------------------

DEFAULT_FAST_DLLM_RELATIVE = Path("external") / "Fast-dLLM" / "v1"


def resolve_fast_dllm_path(explicit: str | os.PathLike | None = None) -> Path:
    """Find Fast-dLLM v1 root (the directory containing `llada/` and `dream/` subdirs)."""
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
    elif os.environ.get("FAST_DLLM_V1_PATH"):
        candidate = Path(os.environ["FAST_DLLM_V1_PATH"]).expanduser().resolve()
    else:
        candidate = (Path.cwd() / DEFAULT_FAST_DLLM_RELATIVE).resolve()

    if not (candidate / "llada" / "model" / "modeling_llada.py").exists():
        raise FileNotFoundError(
            f"Fast-dLLM v1 not found at {candidate}.\n"
            f"Expected file: {candidate / 'llada' / 'model' / 'modeling_llada.py'}\n\n"
            f"Fix one of:\n"
            f"  1. Run `bash setup.sh` from the directory that holds probe_runner/.\n"
            f"  2. Pass --fast_dllm_path /your/path/to/Fast-dLLM/v1 .\n"
            f"  3. Export FAST_DLLM_V1_PATH=/your/path/to/Fast-dLLM/v1 .\n"
        )
    return candidate
