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
        # LLaDA-2.0-mini-MoE and its finetune DMax-Math-16B share ONE arch
        # (DMax-Math is a finetune of mini) — same model_type, the weight path
        # selects which. All dims resolved from model.config at load time.
        # IDs are fixed for the LLaDA-2.0 family (see T3-DMax handoff §5):
        #   MASK = 156895 ; EOS = PAD = 156892 ; vocab = 157184.
        "llada2": {
            # Weight dir on disk (HF-style). No default — pass --model_path or
            # set env T3DMAX_LLADA2_MODEL (mini) / T3DMAX_DMAX_MODEL (DMax).
            "model_path": None,
            "n_layers": None,
            "n_heads": None,
            "d_model": None,
            "d_head": None,
            "mask_token_id": 156895,
            "eos_token_id": 156892,
            "pad_token_id": 156892,
            # decode rule: "soft" = DMax decode_uniform (soft-embedding mix,
            # committed positions re-argmax'd each pass → CR domain live);
            # "hard" = LLaDA-2.0-mini threshold decode (hard commit, no CR).
            "decode_mode": "soft",
            "commit_threshold": 0.3,   # DMax decode_uniform commit threshold
            "break_threshold": 0.9,    # early-stop when all active conf >= this
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


# ----------------------------------------------------------------------
# T3-DMax source resolution (LLaDA-2.0-mini / DMax-Math-16B + ThinkTalk)
#
# The LLaDA-2.0 runner imports the model code from the T3-DMax checkout:
#   - LLaDA2MoeModelLM       <- T3-DMax/dInfer/python/dinfer/model/
#   - ThinkTalkLLaDA2ForCausalLM (repairability probe) <- T3-DMax/dFactory/
# We add those two dirs to sys.path at load time. The checkout root is found
# via (in order): explicit arg -> env T3DMAX_ROOT -> default relative path
# (../T3/T3-D/T3-DMax from the dir that holds probe_runner/).
# ----------------------------------------------------------------------

DEFAULT_T3DMAX_RELATIVE = Path("T3") / "T3-D" / "T3-DMax"


def resolve_t3dmax_root(explicit: str | os.PathLike | None = None) -> Path:
    """Find the T3-DMax checkout root (the dir containing dInfer/ and dFactory/)."""
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
    elif os.environ.get("T3DMAX_ROOT"):
        candidate = Path(os.environ["T3DMAX_ROOT"]).expanduser().resolve()
    else:
        # probe_runner/ -> its parent (peft_project) -> T3/T3-D/T3-DMax
        here = Path(__file__).resolve().parent.parent
        candidate = (here / DEFAULT_T3DMAX_RELATIVE).resolve()

    if not (candidate / "dInfer" / "python" / "dinfer" / "model").is_dir():
        raise FileNotFoundError(
            f"T3-DMax checkout not found at {candidate}.\n"
            f"Expected dir: {candidate / 'dInfer' / 'python' / 'dinfer' / 'model'}\n\n"
            f"Fix one of:\n"
            f"  1. Pass --t3dmax_root /path/to/T3-DMax .\n"
            f"  2. Export T3DMAX_ROOT=/path/to/T3-DMax .\n"
        )
    return candidate


def add_llada2_to_path(t3dmax_root: str | os.PathLike | None = None) -> Path:
    """Put T3-DMax's dInfer + dFactory on sys.path so the model classes import.

    `from dinfer.model import LLaDA2MoeModelLM` needs dInfer/python on the path;
    `from models.think_talk_llada2... import ...` (the load_t3d_model convention)
    needs dFactory on the path. Returns the resolved root.
    """
    import sys

    root = resolve_t3dmax_root(t3dmax_root)
    for sub in (root / "dInfer" / "python", root / "dFactory"):
        s = str(sub)
        if sub.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
    return root
