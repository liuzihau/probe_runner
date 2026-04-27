# probe_runner

Implementation of `T3_pruning_probe_step1to4.md`. Records per-layer / per-head attention weights, value-projection norms, and hidden states at masked positions for **LLaDA-8B-Instruct** and **Dream-7B-Instruct** on the first 100 GSM8K test problems. The output is used to decide how many tail layers to prune from the T3 think backbone.

This repo is **self-contained**. The only external dependency is **Fast-dLLM v1**, fetched by `setup.sh`. No assumption is made about Think-Then-Talk being checked out anywhere.

---

## Quick start

```bash
# 1. Get the code (either clone or copy probe_runner/ into your workspace)
#    After this, you should have a ./probe_runner/ directory.

# 2. Run setup.sh from the directory that CONTAINS probe_runner/ (not from inside it).
#    This clones Fast-dLLM into ./external/Fast-dLLM/ and installs Python deps.
bash probe_runner/setup.sh

# 3. (Optional) sanity smoke — 2 samples per model, ~5 minutes on a 3090
python -m probe_runner.run_probes --model llada --n_samples 2

# 4. Full run (100 samples, both models)
python -m probe_runner.run_probes --model both

# 5. Plot
python -m probe_runner.plots.plot_info_flow_to_prefix --model llada
python -m probe_runner.plots.plot_info_flow_to_prefix --model dream
python -m probe_runner.plots.plot_cka --model llada
python -m probe_runner.plots.plot_cka --model dream
```

All commands assume the **current directory** is the parent of `probe_runner/`. Output is written to `./probes_out/`.

---

## Where Fast-dLLM lives

You have three ways to point this code at Fast-dLLM v1, in order of priority:

1. **Default (recommended).** `bash probe_runner/setup.sh` clones Fast-dLLM into `./external/Fast-dLLM/`. The runner then auto-detects it at `./external/Fast-dLLM/v1`. No configuration needed.

2. **Environment variable.** If you already have Fast-dLLM cloned somewhere else:
   ```bash
   export FAST_DLLM_V1_PATH=/abs/path/to/Fast-dLLM/v1
   python -m probe_runner.run_probes --model llada
   ```

3. **CLI flag.**
   ```bash
   python -m probe_runner.run_probes --model llada --fast_dllm_path /abs/path/to/Fast-dLLM/v1
   ```

The path you provide must contain `llada/model/modeling_llada.py` and `dream/model/modeling_dream.py`. The runner verifies this at startup and prints a clear error if it can't find them.

If the auto-clone fails (no internet, etc.), do this manually:

```bash
mkdir -p external
git clone https://github.com/NVlabs/Fast-dLLM.git external/Fast-dLLM
```

---

## What the experiment does

For each (model, sample) pair:

1. Generates 256 tokens (32 per block × 8 blocks) using Fast-dLLM v1's prefix-cache + parallel-decoding protocol (the same protocol the eventual T3 inference comparison will use).
2. At the **first forward pass of each block** (when that block's 32 positions are still fully masked) installs hooks that record:
   - `attn` — per-layer / per-head attention weights at every masked position, full sequence granularity (no aggregation).
   - `v_norm` — per-layer / per-head per-position `||W_O · v_j||₂` (so info flow can be computed offline).
   - `h_masked` — per-layer hidden states at every masked position (for CKA).
3. Saves each sample as one HDF5 file with 8 block groups: `block_0/` through `block_7/`.

The hook fires **8 times per sample**, once per block at step 0. Subsequent decoding steps within each block run with hooks disarmed.

Output:

```
probes_out/
├── llada/sample_{0000..0099}.h5      # 100 files, ~430 MB each (LLaDA)
├── dream/sample_{0000..0099}.h5      # 100 files
├── plots/
│   ├── info_flow_to_prefix_<model>_rawattn.{png,pdf}
│   ├── info_flow_to_prefix_<model>_flow.{png,pdf}
│   └── cka_<model>.{png,pdf}
└── meta.json
```

---

## Repo layout

```
probe_runner/
├── README.md                           this file
├── setup.sh                            clones Fast-dLLM v1 + installs Python deps
├── requirements.txt
├── __init__.py
├── configs.py                          PROBE_CONFIG; resolves FAST_DLLM_V1_PATH
├── storage.py                          HDF5 read/write with per-block groups
├── hooks.py                            ProbeHooks: monkey-patches LLaDA + Dream attention
├── llada_runner.py                     generate_with_probes mirroring Fast-dLLM v1 dual-cache
├── dream_runner.py                     same protocol on Dream
├── run_probes.py                       CLI entry: load model + GSM8K, drive 8-block firings, write HDF5 + meta.json
└── plots/
    ├── __init__.py
    ├── plot_info_flow_to_prefix.py     Probe A: raw-attn + flow variants, 8-block facets
    └── plot_cka.py                     Probe B: per-block + pooled CKA curves
```

The cloned external code lives outside this directory:

```
external/Fast-dLLM/                     fetched by setup.sh, NOT bundled
├── v1/
│   ├── llada/model/modeling_llada.py   imported by probe_runner/llada_runner.py
│   └── dream/model/modeling_dream.py   imported by probe_runner/dream_runner.py
└── ...
```

---

## CLI reference

```
python -m probe_runner.run_probes --model {llada,dream,both}
                                  [--n_samples N]
                                  [--output_root probes_out]
                                  [--fast_dllm_path /path/to/Fast-dLLM/v1]
```

```
python -m probe_runner.plots.plot_info_flow_to_prefix --model {llada,dream}
                                                       [--probes_root probes_out]
```

```
python -m probe_runner.plots.plot_cka --model {llada,dream}
                                       [--probes_root probes_out]
```

---

## Tunable knobs

All in `probe_runner/configs.py`:

- `record_blocks`: `"all"` (default — 8 blocks) or `[0, 3, 7]` to subsample.
- `v_norm_blocks`: `"all"` or `[0]` if disk is tight.
- `threshold`: parallel-decoding confidence threshold (default 0.9, Fast-dLLM v1 default).
- `gen_length`: 256.
- `block_length`: 32.
- `max_prompt_tokens`: 512 (prompts longer than this are truncated to the last 512 tokens).
- `attn_implementation`: `"manual_softmax"` — required because SDPA/flash hide attention weights.

---

## Sanity checks

`run_probes.py` prints a JSON summary after sample 0 of each model. Verify:

- `all_blocks_recorded == 8`.
- `attn_row_sum_min` and `attn_row_sum_max` are within `1e-2` of `1.0` (attention rows sum to 1).
- `mean_attn_to_pos0` — flagged in `meta.json` if > 0.1 so plotting can subtract it as a sink.
- `attn_shape[-1] == prompt_len + 32`.
- `h_shape[0] == n_layers + 1` (embedding + per-block outputs).

---

## Storage budget (per sample, all 8 blocks recorded, prompt_len ≤ 512)

| Component | LLaDA | Dream |
|---|---:|---:|
| `attn` (8 blocks, fp16) | ~340 MB | ~210 MB |
| `v_norm` (8 blocks, fp32) | ~21 MB | ~13 MB |
| `h_masked` (8 blocks, fp16) | ~70 MB | ~53 MB |
| **Total per sample** | **~430 MB** | **~280 MB** |
| × 100 samples | **~43 GB** | **~28 GB** |

If disk is tight: set `v_norm_blocks = [0]` (~5% saving) or `record_blocks = [0, 3, 7]` (~62% saving).

---

## Known caveats

- **Eager / manual-softmax attention required.** SDPA/flash do not expose attention weights. The patched attention runs manual softmax during the probe-hooked forwards. Roughly halves throughput vs flash, but it only matters during the 100-sample probe run, not during downstream T3 training.
- **Single sample per batch.** The hooks assume `B=1`. Batching is possible but adds index-juggling; not implemented.
- **Dream's KV-cache format.** Fast-dLLM v1's local `model/modeling_dream.py` provides the `dual_cache + replace_position` interface. If Fast-dLLM upstream changes that interface, `dream_runner.py` may need a small alignment patch — see `T3_pruning_probe_step1to4.md` §10 for the failure-mode table.
- **Attention-sink subtraction is heuristic.** Default `attention_sink_positions=[0]`. If sample-0 sanity shows `mean_attn_to_pos0 < 0.1`, the heuristic is wrong; manually edit `meta.json` to update the sink list before plotting.

For the full failure-mode table and acceptance criteria, see `T3_pruning_probe_step1to4.md` §9 and §10.

---

## Cross-references

- `T3_pruning_probe_step1to4.md` — full spec this repo implements.
- `T3_pruning_probe_step5to8.md` — what to do with the resulting plots: composite scoring, prune-count selection, retraining.
- `T3_overview.md`, `T3_drawbacks.md`, `T3_next_trial.md` — main project context (sit alongside, not required for running this repo).
