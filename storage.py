"""HDF5 readers and writers for probe data.

File layout (per sample, see ../T3_pruning_probe_step1to4.md §4.1):

    sample_NNNN.h5
    ├── block_0/
    │   ├── attn        [num_masked, L, H, S_0]   float16
    │   ├── v_norm      [L, H, S_0]               float32   (optional per config)
    │   └── h_masked    [L+1, num_masked, d_model] float16
    ├── block_1/
    │   └── ...
    ├── ...
    └── attrs:
        prompt_len, num_masked, num_blocks, block_seq_lens (json),
        block_mask_positions (json), prompt_text, gold_answer,
        generated_text, model_name, n_layers, n_heads, d_model,
        attention_sink_positions (json)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def write_h5(
    path: str | Path,
    data_per_block: dict[int, dict[str, torch.Tensor]],
    *,
    prompt_text: str,
    gold_answer: str,
    generated_text: str,
    model_name: str,
    n_layers: int,
    n_heads: int,
    d_model: int,
    prompt_len: int,
    num_masked: int,
    block_seq_lens: list[int],
    block_mask_positions: list[list[int]],
    attention_sink_positions: list[int] | None = None,
) -> None:
    """Write one sample's probe data to HDF5.

    `data_per_block` maps block_idx → {"attn": Tensor, "v_norm": Tensor | None, "h_masked": Tensor}.
    `v_norm` may be None for blocks where it was not recorded (per `v_norm_blocks` config).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        for block_idx, tensors in sorted(data_per_block.items()):
            grp = f.create_group(f"block_{block_idx}")
            grp.create_dataset("attn", data=_to_numpy(tensors["attn"]), compression="gzip", compression_opts=4)
            if tensors.get("v_norm") is not None:
                grp.create_dataset("v_norm", data=_to_numpy(tensors["v_norm"]), compression="gzip", compression_opts=4)
            grp.create_dataset("h_masked", data=_to_numpy(tensors["h_masked"]), compression="gzip", compression_opts=4)

        f.attrs["prompt_len"] = int(prompt_len)
        f.attrs["num_masked"] = int(num_masked)
        f.attrs["num_blocks"] = int(len(data_per_block))
        f.attrs["block_seq_lens"] = json.dumps(block_seq_lens)
        f.attrs["block_mask_positions"] = json.dumps(block_mask_positions)
        f.attrs["prompt_text"] = prompt_text
        f.attrs["gold_answer"] = gold_answer
        f.attrs["generated_text"] = generated_text
        f.attrs["model_name"] = model_name
        f.attrs["n_layers"] = int(n_layers)
        f.attrs["n_heads"] = int(n_heads)
        f.attrs["d_model"] = int(d_model)
        if attention_sink_positions is not None:
            f.attrs["attention_sink_positions"] = json.dumps(attention_sink_positions)


def read_h5(path: str | Path) -> dict[str, Any]:
    """Read a sample's HDF5 back into Python dicts (for plotting)."""
    path = Path(path)
    out: dict[str, Any] = {"blocks": {}, "attrs": {}}
    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            if k in {"block_seq_lens", "block_mask_positions", "attention_sink_positions"}:
                out["attrs"][k] = json.loads(v)
            else:
                out["attrs"][k] = v
        for grp_name in sorted(f.keys()):
            if not grp_name.startswith("block_"):
                continue
            block_idx = int(grp_name.split("_")[1])
            grp = f[grp_name]
            block_data = {"attn": np.asarray(grp["attn"])}
            if "v_norm" in grp:
                block_data["v_norm"] = np.asarray(grp["v_norm"])
            block_data["h_masked"] = np.asarray(grp["h_masked"])
            out["blocks"][block_idx] = block_data
    return out


def write_meta(path: str | Path, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
