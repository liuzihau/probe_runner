"""Plot Probe A: raw-attention, info-flow, and normalized info-flow to prefix.

Three figures per model: rawattn, flow, flow_normalized. Each has 8 per-block facets
plus a per-layer summary.

Filters applied per sample:
  1. Masked positions at-or-after the first EOS token are dropped from the count.
  2. Special-token positions in the prompt (BOS, chat-template markers, ...) are
     subtracted from the prefix.
  3. Position 0 ("attention sink") is always treated as a sink.

These two filters can be backfilled at plot time from the existing HDF5 files: if
`special_token_positions` / `eos_pos_in_generated` are missing (older data), we
recompute via the model's tokenizer on the saved `prompt_text` / `generated_text`.

Run from the directory that contains probe_runner/:
    python -m probe_runner.plots.plot_info_flow_to_prefix --model llada
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_runner import storage, configs


# ---------------------------------------------------------------------------
# Filter resolution (saved-attrs-first, recompute fallback)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _load_tokenizer_for(model_name: str):
    """Lazily load the model's tokenizer if we need to recompute filters from text."""
    from transformers import AutoTokenizer  # noqa: WPS433
    hf_name = configs.PROBE_CONFIG["models"][model_name]["hf_name"]
    return AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)


def _resolve_special_positions(sample: dict, model_name: str) -> set[int]:
    """Positions in the prompt that should be subtracted from prefix (sink + special tokens)."""
    attrs = sample["attrs"]
    if "special_token_positions" in attrs:
        return set(attrs["special_token_positions"])
    # Recompute from prompt_text via tokenizer
    tok = _load_tokenizer_for(model_name)
    msg = [{"role": "user", "content": attrs["prompt_text"]}]
    prompt_str = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    ids = tok(prompt_str)["input_ids"]
    try:
        special_ids = set(tok.all_special_ids or [])
    except Exception:
        special_ids = set()
    out = {0} | {i for i, t in enumerate(ids) if t in special_ids}
    # Truncate to actual recorded prompt length (recording may have left-truncated long prompts)
    prompt_len = int(attrs.get("prompt_len", len(ids)))
    drop = len(ids) - prompt_len
    if drop > 0:
        out = {p - drop for p in out if p - drop >= 0}
    return out


def _resolve_eos_pos(sample: dict, model_name: str) -> int:
    """Position (within generated portion) where first EOS-like token appeared.

    Returns gen_length when the model never emitted EOS (i.e., keep all blocks).
    """
    attrs = sample["attrs"]
    if "eos_pos_in_generated" in attrs:
        return int(attrs["eos_pos_in_generated"])
    # Recompute approximately: re-tokenize generated_text. This is approximate because
    # the recorded text was decoded with skip_special_tokens=True.
    tok = _load_tokenizer_for(model_name)
    text = attrs.get("generated_text", "")
    if not text:
        return 0
    retokens = tok(text, add_special_tokens=False)["input_ids"]
    return len(retokens)  # the EOS would have been at this position in the original IDs


def _allowed_masked_positions(block_idx: int, num_masked: int, eos_pos_in_generated: int) -> list[int]:
    """Which masked positions (0..num_masked-1) of block `block_idx` are PRE-EOS."""
    block_start = block_idx * num_masked
    block_end = block_start + num_masked
    if eos_pos_in_generated <= block_start:
        return []
    if eos_pos_in_generated >= block_end:
        return list(range(num_masked))
    return list(range(eos_pos_in_generated - block_start))


# ---------------------------------------------------------------------------
# Per-(layer × head) signal accumulation
# ---------------------------------------------------------------------------

def _per_layer_per_block_signal(
    files: list[Path],
    variant: str,
    model_name: str,
    apply_eos_cutoff: bool,
    apply_special_filter: bool,
) -> tuple[np.ndarray, list[int], dict]:
    """Compute mean signal of shape [num_blocks, L, H], per `variant`.

    `variant`:
      - "attn"            → Σ_{j ∈ prefix \\ sinks} attn[m, ℓ, h, j]
      - "flow"            → Σ_{j ∈ prefix \\ sinks} attn[m, ℓ, h, j] * v_norm[ℓ, h, j]
      - "flow_normalized" → flow[ℓ, h, prefix] / flow[ℓ, h, ALL non-sink]
    """
    accumulators: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    diagnostics = {
        "n_files": len(files),
        "samples_with_eos": 0,
        "mean_eos_pos": 0.0,
        "blocks_kept_per_sample": {b: 0 for b in range(8)},
    }
    eos_positions = []

    for f in files:
        sample = storage.read_h5(f)
        prompt_len = int(sample["attrs"]["prompt_len"])
        special_positions = _resolve_special_positions(sample, model_name) if apply_special_filter else {0}
        eos_pos = _resolve_eos_pos(sample, model_name) if apply_eos_cutoff else 10**9
        eos_positions.append(eos_pos)
        if eos_pos < 256:
            diagnostics["samples_with_eos"] += 1

        for b, blk in sample["blocks"].items():
            attn = blk["attn"].astype(np.float32)            # [num_masked, L, H, S_b]
            num_masked, L, H, S_b = attn.shape

            # Filter masked positions to pre-EOS
            allowed_m = _allowed_masked_positions(b, num_masked, eos_pos)
            if not allowed_m:
                continue
            attn_kept = attn[allowed_m]                       # [n_kept, L, H, S_b]

            # Build prefix index: everything before the current block, minus sinks/specials
            block_start_abs = prompt_len + b * num_masked
            prefix_all = list(range(block_start_abs))
            prefix_idx = [j for j in prefix_all if j not in special_positions]

            # Build "all non-sink" index for the normalized variant
            all_idx = [j for j in range(S_b) if j not in special_positions]

            attn_prefix = attn_kept[..., prefix_idx]          # [n_kept, L, H, |prefix|]

            if variant == "attn":
                contrib = attn_prefix.sum(axis=-1)            # [n_kept, L, H]
            elif variant == "flow":
                v_norm = blk.get("v_norm")
                if v_norm is None:
                    continue
                v_norm = v_norm.astype(np.float32)            # [L, H, S_b]
                v_pref = v_norm[..., prefix_idx]              # [L, H, |prefix|]
                contrib = (attn_prefix * v_pref[None, ...]).sum(axis=-1)
            elif variant == "flow_normalized":
                v_norm = blk.get("v_norm")
                if v_norm is None:
                    continue
                v_norm = v_norm.astype(np.float32)
                v_pref = v_norm[..., prefix_idx]
                v_all = v_norm[..., all_idx]
                attn_all = attn_kept[..., all_idx]
                flow_pref = (attn_prefix * v_pref[None, ...]).sum(axis=-1)
                flow_total = (attn_all * v_all[None, ...]).sum(axis=-1)
                contrib = flow_pref / (flow_total + 1e-9)
            else:
                raise ValueError(variant)

            mean = contrib.mean(axis=0)                       # [L, H], averaged over kept masked positions
            if b not in accumulators:
                accumulators[b] = mean
                counts[b] = 1
            else:
                accumulators[b] = accumulators[b] + mean
                counts[b] = counts[b] + 1
            diagnostics["blocks_kept_per_sample"][b] = diagnostics["blocks_kept_per_sample"].get(b, 0) + 1

    diagnostics["mean_eos_pos"] = float(np.mean(eos_positions)) if eos_positions else 0.0
    blocks = sorted(accumulators.keys())
    if not blocks:
        raise RuntimeError("No blocks survived filtering — all samples gated out.")
    arr = np.stack([accumulators[b] / counts[b] for b in blocks], axis=0)
    return arr, blocks, diagnostics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(arr: np.ndarray, blocks: list[int], title: str, out_path: Path,
          y_label: str = "signal"):
    num_blocks, L, H = arr.shape
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[3, 3, 2])

    vmin, vmax = float(arr.min()), float(arr.max())
    last_im = None
    for i, b in enumerate(blocks[:8]):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        last_im = ax.imshow(arr[i].T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"block {b}")
        ax.set_xlabel("layer ℓ (0=early, L-1=late)")
        ax.set_ylabel("head h")
    if last_im is not None:
        fig.colorbar(last_im, ax=fig.axes, fraction=0.02, pad=0.02, shrink=0.6)

    ax = fig.add_subplot(gs[2, :])
    layer_curves = arr.mean(axis=-1)  # [num_blocks, L]
    for i, b in enumerate(blocks):
        ax.plot(range(L), layer_curves[i], label=f"block {b}", alpha=0.6)
    ax.plot(range(L), layer_curves.mean(axis=0), label="mean over blocks", color="black", linewidth=2)
    ax.set_xlabel("layer ℓ")
    ax.set_ylabel(y_label)
    ax.set_title(title + " — per-layer (mean over heads)")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llada", "dream"], required=True)
    parser.add_argument("--probes_root", type=str, default="probes_out")
    parser.add_argument("--variant", choices=["attn", "flow", "flow_normalized", "all"],
                        default="all")
    parser.add_argument("--no_eos_cutoff", action="store_true",
                        help="Disable per-sample EOS-cutoff filter on masked positions.")
    parser.add_argument("--no_special_filter", action="store_true",
                        help="Disable per-sample special-token subtraction (use only pos-0 sink).")
    args = parser.parse_args()

    probes_root = Path(args.probes_root)
    sample_dir = probes_root / args.model
    plots_dir = probes_root / "plots"
    files = sorted(sample_dir.glob("sample_*.h5"))
    if not files:
        raise SystemExit(f"No samples in {sample_dir}")

    apply_eos = not args.no_eos_cutoff
    apply_special = not args.no_special_filter

    variants_to_plot = ["attn", "flow", "flow_normalized"] if args.variant == "all" else [args.variant]

    suffix = []
    if apply_eos:
        suffix.append("eosfilter")
    if apply_special:
        suffix.append("specialfilter")
    suffix_str = "_" + "_".join(suffix) if suffix else ""

    diagnostics_collected = None
    for v in variants_to_plot:
        print(f"[{args.model}] computing variant={v} on {len(files)} samples …")
        arr, blocks, diag = _per_layer_per_block_signal(
            files, v, model_name=args.model,
            apply_eos_cutoff=apply_eos, apply_special_filter=apply_special,
        )
        if diagnostics_collected is None:
            diagnostics_collected = diag
        if v == "attn":
            title = f"{args.model} — raw attention to prefix"
            ylabel = "Σ attn over prefix (∈ [0, 1])"
        elif v == "flow":
            title = f"{args.model} — info flow (attn × ||W_O·v||) to prefix"
            ylabel = "Σ (attn · v_norm) over prefix"
        else:
            title = f"{args.model} — normalized info flow (prefix / total)"
            ylabel = "flow_prefix / flow_total ∈ [0, 1]"
        out_png = plots_dir / f"info_flow_to_prefix_{args.model}_{v}{suffix_str}.png"
        _plot(arr, blocks, title, out_png, y_label=ylabel)
        print(f"[{args.model}] saved {out_png}")

    if diagnostics_collected is not None:
        diag_path = plots_dir / f"diagnostics_{args.model}{suffix_str}.json"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, "w") as fp:
            json.dump(diagnostics_collected, fp, indent=2)
        print(f"[{args.model}] saved {diag_path}")
        print(f"  samples with EOS: {diagnostics_collected['samples_with_eos']}/{diagnostics_collected['n_files']}")
        print(f"  mean EOS position: {diagnostics_collected['mean_eos_pos']:.1f}")
        print(f"  blocks kept (samples per block): {diagnostics_collected['blocks_kept_per_sample']}")


if __name__ == "__main__":
    main()
