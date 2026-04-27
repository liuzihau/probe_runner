# probe_runner

Implementation of `T3_pruning_probe_step1to4.md`. Records per-layer / per-head attention weights, value-projection norms, and hidden states at masked positions for LLaDA-8B-Instruct and Dream-7B-Instruct on the first 100 GSM8K test problems. Used to decide how many tail layers to prune from the T3 think backbone.

## What it does

For each (model, sample) pair:

1. Generates 256 tokens (32 per block × 8 blocks) using Fast-dLLM v1's prefix-cache + parallel-decoding protocol.
2. At the **first forward pass of each block** (when that block's 32 positions are still fully masked) installs hooks that record:
   - `attn` — per-layer / per-head attention weights at every masked position, full sequence granularity (no aggregation).
   - `v_norm` — per-layer / per-head per-position `||W_O · v_j||₂` (so info flow can be computed offline).
   - `h_masked` — per-layer hidden states at every masked position (for CKA).
3. Saves each sample as one HDF5 file with 8 block groups: `block_0/` through `block_7/`.

## Quick start

From the **T3 project root** (the directory containing `Think-Then-Talk/`, `related_repos/`, and the `T3_*.md` docs):

```bash
# Probe LLaDA only
python -m probe_runner.run_probes --model llada

# Probe Dream only
python -m probe_runner.run_probes --model dream

# Both in sequence
python -m probe_runner.run_probes --model both

# Smaller smoke test (first 5 samples)
python -m probe_runner.run_probes --model llada --n_samples 5

# Plot
python -m probe_runner.plots.plot_info_flow_to_prefix --model llada
python -m probe_runner.plots.plot_cka --model llada
```

Output goes to `probes_out/`:

```
probes_out/
├── llada/sample_{0000..0099}.h5
├── dream/sample_{0000..0099}.h5
├── plots/
│   ├── info_flow_to_prefix_<model>_rawattn.{png,pdf}
│   ├── info_flow_to_prefix_<model>_flow.{png,pdf}
│   └── cka_<model>.{png,pdf}
└── meta.json
```

## Dependencies

```
torch >= 2.1
transformers >= 4.40
datasets
h5py
numpy
matplotlib
```

Plus Fast-dLLM v1 dependencies (already in `related_repos/Fast-dLLM/v1/requirements.txt`).

## File map

```
probe_runner/
├── configs.py            # PROBE_CONFIG — single source of truth for hyperparameters
├── storage.py            # HDF5 read/write
├── hooks.py              # ProbeHooks: monkey-patches attention; per-block accumulator
├── llada_runner.py       # LLaDA generate_with_probes (mirrors Fast-dLLM v1 dual-cache)
├── dream_runner.py       # Dream generate_with_probes (mirrors Fast-dLLM v1 dual-cache)
├── run_probes.py         # CLI entry: load model + dataset, drive the loop, write HDF5 + meta.json
└── plots/
    ├── plot_info_flow_to_prefix.py   # Probe A: raw-attn + info-flow variants, faceted by block
    └── plot_cka.py                    # Probe B: per-block CKA + pooled CKA
```

## Knobs (in `configs.py`)

- `record_blocks`: `"all"` (default, 8 blocks) or `[0, 3, 7]` to subsample.
- `v_norm_blocks`: `"all"` or `[0]` if disk is tight.
- `threshold`: parallel-decoding confidence threshold (default 0.9, Fast-dLLM v1 default).
- `gen_length`: 256.
- `block_length`: 32.
- `max_prompt_tokens`: 512.
- `attn_implementation`: `"manual_softmax"` — required because SDPA/flash hide attention weights.

## Sanity checks

`run_probes.py` prints a JSON summary after sample 0 of each model. Verify:

- `all_blocks_recorded == 8`.
- `attn_row_sum_min` and `_max` are within `1e-2` of `1.0` (attention rows sum to 1).
- `mean_attn_to_pos0` — flag in `meta.json` if > 0.1 so plotting can subtract.
- `attn_shape[-1] == prompt_len + 32`.
- `h_shape[0] == n_layers + 1` (embedding + per-block outputs).

## Storage budget

Per sample, all 8 blocks recorded with `prompt_len ≤ 512`:

- LLaDA: ~430 MB / sample → ~43 GB / 100 samples.
- Dream: smaller (28 layers × 28 heads vs LLaDA's 32 × 32) → ~25 GB / 100 samples.

If disk is tight, set `v_norm_blocks = [0]` (~5% saving) or `record_blocks = [0, 3, 7]` (~62% saving).

## Known caveats

- **Eager attention only.** SDPA/flash do not expose attention weights. The patched attention runs manual softmax for all forward passes during the probe run. This roughly halves throughput vs flash, but it only matters during the 100-sample probe run, not during downstream T3 training.
- **Single sample per batch.** The hooks assume `B=1`. Batching is possible but adds index-juggling; not implemented.
- **Dream's KV-cache format.** Fast-dLLM v1's `model.modeling_dream.DreamModel` provides the `dual_cache + replace_position` interface. If you load Dream from the upstream HF id without trust_remote_code, the cache interface differs and `dream_runner.py` will fail. We try the local Fast-dLLM v1 class first and fall back to AutoModel.
- **Attention-sink subtraction is heuristic.** Default `attention_sink_positions=[0]`. If the sanity check shows pos 0 mass < 0.1 the heuristic is wrong; manually edit `meta.json`.

See the parent `T3_pruning_probe_step1to4.md` for the full spec including failure modes (§10) and acceptance criteria (§9).
