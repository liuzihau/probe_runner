"""Single source of truth for probe-run hyperparameters.

See ../T3_pruning_probe_step1to4.md §8 for the spec this mirrors.
"""

from __future__ import annotations

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
        # Relative to T3 project root.
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
