# probe_runner

Diagnostic capture + analysis pipeline for **block-diffusion language
models** (LLaDA-8B-Instruct and Dream-7B-Instruct) on the first 100
GSM8K test problems. Records per-layer attention, value-projection
norms, and hidden states at masked positions for each block of the
generation, and provides a family of plotting scripts that distil those
captures into the kind of figures used to drive layer-pruning,
KV-reuse, and feature-staleness decisions for the T3 project.

This repo is **self-contained**. The only external dependency is
**Fast-dLLM v1**, fetched by `setup.sh`. No assumption is made about
Think-Then-Talk being checked out anywhere.

---

## What it produces

Two layers of output.

### Captures (raw HDF5, one file per sample)

```
probes_out/
├── llada/sample_0000.h5  …  sample_0099.h5         # ~430 MB each (legacy mode)
├── dream/sample_0000.h5  …                         # ~280 MB each
└── meta.json
```

Each `sample_NNNN.h5` has 8 block groups (`block_0/` … `block_7/`),
plus optional intra-block per-pass tensors when `--intra_block` is on.
See **Storage layout** further down.

### Plots (PNG / PDF, derived from captures)

```
probes_out/<model>/plots/
├── info_flow_to_prefix_<model>_attn.png             # variant=attn
├── info_flow_to_prefix_<model>_flow.png             # variant=flow
├── info_flow_to_prefix_<model>_flow_normalized.png  # variant=flow_normalized
├── flow_split_prefix_<model>_attn.png               # 3-way split, attn
├── flow_split_prefix_<model>_flow_normalized.png    # 3-way split, normalised flow
├── cka_<model>.png
├── logit_lens_<model>_topk5_eosfilter_specialfilter.png
├── intra_block_drift_grouped_<metric>.png           # MM / MC / CC drift vs layer
├── intra_block_diff_heatmap_<metric>_sample_NNNN.png
├── intra_block_drift_vs_pass0_<metric>_blocks0-1_n5.png
└── intra_block_prediction_overlap_blocks0-1_n5_refpass0.png
```

Each plot is described in its own subsection below.

---

## Quick start

```bash
# 1. Get the code (clone or copy probe_runner/ into your workspace).
#    After this, you should have a ./probe_runner/ directory.

# 2. Run setup.sh from the directory that CONTAINS probe_runner/ (not from inside).
#    Clones Fast-dLLM into ./external/Fast-dLLM/ and installs Python deps.
bash probe_runner/setup.sh

# 3. (Optional) sanity smoke — 2 samples per model, ~5 minutes on a 3090.
python -m probe_runner.run_probes --model llada --n_samples 2

# 4. Full legacy capture (100 samples, both models).
python -m probe_runner.run_probes --model both

# 5. Intra-block per-pass capture. Adds ~470 MB / sample of fp16 h_per_pass;
#    needed for the intra-block analyses. Supported for both LLaDA and Dream.
python -m probe_runner.run_probes --model llada --intra_block
python -m probe_runner.run_probes --model dream --intra_block

# 6. Plots — see "Analyses" section below for what each command produces.
python -m probe_runner.plots.plot_info_flow_to_prefix --model llada
python -m probe_runner.plots.plot_flow_split_prefix --model llada
python -m probe_runner.plots.plot_cka --model llada
python -m probe_runner.plots.plot_logit_lens --model llada
python -m probe_runner.plots.plot_intra_block --model llada                # H5-only variants
python -m probe_runner.plots.plot_intra_block --model llada \              # heavier: loads model
    --variant prediction_overlap
```

All commands assume the **current directory** is the parent of
`probe_runner/`. Output is written to `./probes_out/`.

---

## Where Fast-dLLM lives

Three ways to point this code at Fast-dLLM v1, in order of priority:

1. **Default (recommended).** `bash probe_runner/setup.sh` clones
   Fast-dLLM into `./external/Fast-dLLM/`. The runner auto-detects it
   at `./external/Fast-dLLM/v1`. No configuration needed.

2. **Environment variable.** If you already have Fast-dLLM cloned
   somewhere else:
   ```bash
   export FAST_DLLM_V1_PATH=/abs/path/to/Fast-dLLM/v1
   python -m probe_runner.run_probes --model llada
   ```

3. **CLI flag.**
   ```bash
   python -m probe_runner.run_probes --model llada \
       --fast_dllm_path /abs/path/to/Fast-dLLM/v1
   ```

The path you provide must contain `llada/model/modeling_llada.py` and
`dream/model/modeling_dream.py`. The runner verifies this at startup
and prints a clear error if it can't find them.

If the auto-clone fails (no internet, etc.):

```bash
mkdir -p external
git clone https://github.com/NVlabs/Fast-dLLM.git external/Fast-dLLM
```

---

## Capture: what `run_probes.py` does

For each (model, sample) pair:

1. Generates 256 tokens (32 per block × 8 blocks) using **Fast-dLLM
   v1's prefix-cache + parallel-decoding protocol** — the same protocol
   the eventual T3 inference comparison will use.
2. **Legacy mode (default):** at the *first* forward of each block
   (when that block's 32 positions are still all mask) installs hooks
   that record:
   - `attn` — per-layer / per-head attention weights at every masked
     position, full sequence granularity.
   - `v_norm` — per-layer / per-head per-position `||W_O · v_j||₂`
     (so info flow can be computed offline).
   - `h_masked` — per-layer hidden states at every masked position.
   The hook fires **8 times per sample**, once per block at step 0.
   Subsequent decoding steps within each block run with hooks disarmed.
3. **Intra-block mode (`--intra_block`):** keeps hooks armed across
   *every* parallel-decode pass within a block. Records additional
   per-pass tensors:
   - `h_per_pass` — `[n_passes, L+1, num_masked, d_model]` fp16.
   - `token_state_per_pass` — `[n_passes, num_masked]` int8 (0=mask,
     1=clean, captured before each pass's forward).
   - `revealed_per_pass` — `[n_passes, num_masked]` bool, set True at
     the pass that flipped the position from mask → clean.
   - `pass_indices` — `[n_passes]` int32, the original pass numbers.
   Pass-0 attn / v_norm capture is preserved (uses manual softmax);
   passes ≥ 1 fall back to the original SDPA so the inner decode loop
   doesn't pay the manual-softmax cost.

### CLI

```
python -m probe_runner.run_probes
    --model {llada, dream, both}        # which model(s) to run
    [--n_samples N]                     # default 100 (set in configs.py)
    [--output_root probes_out]
    [--fast_dllm_path /path/to/Fast-dLLM/v1]
    [--intra_block]                     # adds per-pass capture (both LLaDA and Dream)
```

Existing samples in the output dir are skipped (so you can resume
interrupted runs by re-invoking the same command).

### Storage layout

```
sample_NNNN.h5
├── block_0/
│   ├── attn               [num_masked, L, H, S_0]    float16  gzip
│   ├── v_norm             [L, H, S_0]                float32  gzip   (optional per config)
│   ├── h_masked           [L+1, num_masked, d_model] float16  gzip
│   ├── h_per_pass         [n_passes, L+1, num_masked, d_model]    float16 gzip   ← intra_block
│   ├── token_state_per_pass [n_passes, num_masked]   int8     gzip   ← intra_block
│   ├── revealed_per_pass  [n_passes, num_masked]     bool     gzip   ← intra_block
│   └── pass_indices       [n_passes]                 int32           ← intra_block
├── block_1/  …  block_7/
└── attrs:
    prompt_len, num_masked, num_blocks, block_seq_lens (json),
    block_mask_positions (json), prompt_text, gold_answer,
    generated_text, model_name, n_layers, n_heads, d_model,
    attention_sink_positions (json), special_token_positions (json),
    eos_pos_in_generated
```

Block 7 of most samples is post-EOS noise; the plot scripts filter it
out by default (see "Filters applied to all plots").

---

## Analyses

Six analysis scripts, one per concern. All are H5-only and run from
captures except `plot_logit_lens` and `plot_intra_block --variant
prediction_overlap`, which load the model to project hidden states
through the LM head.

### 1. Raw attention to prefix — *where* attention points

```bash
python -m probe_runner.plots.plot_info_flow_to_prefix \
    --model {llada, dream} \
    [--probes_root probes_out] \
    [--variant {attn, flow, flow_normalized, all}] \
    [--no_eos_cutoff] \
    [--no_special_filter]
```

For each (layer ℓ, head h, sample, masked position m), sums
`attn[m, ℓ, h, j]` over `j ∈ prefix \ sinks`. Output is a fraction in
`[0, 1]`: "what share of attention from masked tokens lands on prefix
positions vs. on the current block / future masks." Faceted by block.

**Use:** quick first-look at whether late layers shift their
attention away from prefix.

### 2. Info flow to prefix — *how much information* arrives

`--variant flow` on the same script. Multiplies the attention by
per-position value-projection norm:
`Σ attn[m, ℓ, h, j] × ||W_O · v_j||₂`. Captures the *magnitude* of
signal flowing from prefix to masked positions, not just where
attention points.

**Use:** distinguishes "model attends to prefix but values are tiny"
from "model attends to prefix and values are large."

**Caveat:** value-projection norms grow naturally with depth; the next
variant strips this out.

### 3. Normalised info flow — *what fraction* comes from prefix

`--variant flow_normalized`. `flow_to_prefix / flow_to_all_non_sinks`,
a per-(ℓ, h) ratio in `[0, 1]`. Strips depth-induced norm growth.

**Use:** the cleanest "where does the model do its prefix processing?"
plot. If prefix flow ratio drops sharply at some layer ℓ, that layer
onward is doing token-formation, not prefix absorption — strong prune
candidate.

### 4. Split normalised flow — *where in prefix* the flow comes from

```bash
python -m probe_runner.plots.plot_flow_split_prefix \
    --model {llada, dream} \
    [--probes_root probes_out] \
    [--variant {attn, flow, attn_normalized, flow_normalized, all}] \
    [--recent_window 8] \
    [--include_future] \
    [--no_eos_cutoff] \
    [--no_special_filter]
```

Splits the input into **three partitions** instead of treating prefix
as one block:

- **RECENT prefix** = positions `[block_start − recent_window,
  block_start)` (last 8 tokens before current block, by default).
- **DISTANT prefix** = positions `[0, block_start − recent_window)`.
- **CURRENT BLOCK** = positions `[block_start, block_start + 32)`
  (intra-block "reverse look").

For each partition, plots the normalised flow ratio with the same
denominator. The three partitions sum to 1 per (ℓ, h), so you get a
clean three-way breakdown. Pass `--include_future` to also factor in
the four-way split with future masks.

**Use:** distinguishes "late layers shift to local context" (recent
rises, distant falls) from "late layers shift to intra-block
reasoning" (current_block rises). Different patterns suggest different
prune strategies.

### 5. CKA — *which layers stop changing* the representation

```bash
python -m probe_runner.plots.plot_cka \
    --model {llada, dream} \
    [--probes_root probes_out]
```

Plots `CKA(h_ℓ, h_L)` against ℓ — the centred-kernel-alignment
similarity between each layer's masked-position hidden state and the
final layer's. CKA ∈ `[0, 1]`, invariant to orthogonal rotation and
isotropic scaling. The **knee** at ~0.95 is the conventional
saturation threshold (Kornblith et al. 2019).

**Use:** representation-level prune-cut diagnostic. Layers above the
knee are doing rotation / fine-tuning, not new representational work.

### 6. Logit lens — *which layers stop changing the predicted distribution*

```bash
python -m probe_runner.plots.plot_logit_lens \
    --model {llada, dream} \
    [--probes_root probes_out] \
    [--top_k 5] \
    [--fast_dllm_path /path/to/Fast-dLLM/v1] \
    [--no_eos_cutoff] \
    [--no_special_filter]
```

Loads the model and projects each layer's masked-position hidden state
through `final_norm` + `lm_head`, then compares to the final layer's
distribution via three metrics, all averaged across positions / samples
/ blocks:

| Metric | Range | Higher = | Saturation |
|---|---:|---|---|
| `top_k_overlap` (default K=5) | `[0, 1]` | layer agrees with final on top-K | red dashed at 0.95 |
| `kl_divergence`  KL(p_ℓ ‖ p_L) | `[0, ∞)` | layer disagrees with final | none — interpret near zero |
| `shared_mass`  Σ min(p_ℓ, p_L) | `[0, 1]` | layer's distribution overlaps final | red dashed at 0.95 |

All three are plotted side-by-side in one figure.

**Use:** the layer at which `top_k_overlap` ≥ 0.95 and `shared_mass`
≥ 0.95 (and KL ≈ 0) is the layer at which the model has effectively
decided what to emit. Subsequent layers do not reshape *which tokens
get probability mass* — direct prune signal.

Cost: ~5–10 min on a 3090 (full model load + projections per
layer/sample/block).

### 7. Intra-block — drift across denoise passes within a block

This family operates on the per-pass capture (`--intra_block` from
`run_probes`). All variants live in one script with a `--variant`
flag.

```bash
python -m probe_runner.plots.plot_intra_block \
    --model {llada, dream} \
    [--probes_root probes_out] \
    [--variant {drift_grouped, diff_heatmap, drift_vs_pass0, prediction_overlap, all}] \
    [--metric {l2, l2_normalized, cosine_sim, all}] \
    [--n_samples_heatmap N]                # for diff_heatmap (default 5)
    [--vs_pass0_blocks "0,1"]              # for drift_vs_pass0 / prediction_overlap (default "0,1")
    [--vs_pass0_n_samples N]               # default 5; bump to ~100 for paper bands
    [--vs_pass0_max_iter N]                # default 6
    [--prediction_overlap_reference {pass0, prev}]    # default pass0
    [--fast_dllm_path /path/to/Fast-dLLM/v1]          # only used by prediction_overlap
```

Note: `--variant all` runs `drift_grouped + diff_heatmap +
drift_vs_pass0` for the chosen metrics. It does **not** run
`prediction_overlap` (which loads the model and is heavier). Pass
`--variant prediction_overlap` explicitly to run it.

Token-state classification used by all variants:

- **MM** = position is mask at input of pass i AND mask at input of
  pass i+1 (or pass i, for the vs-pass-0 variant). Token id at this
  position did not change between the two captures.
- **MC** = position is mask at input of pass i, clean at input of
  pass i+1 (or i). Position got revealed during the comparison
  interval — token id flipped.
- **CC** = position is clean at both. Already revealed before the
  interval. Only meaningful for the consecutive-pair variant
  (drift_grouped); for vs-pass-0 it's structurally absent because
  pass 0 is always all-mask.

#### 7a. `drift_grouped` — per-layer cos_sim / drift, grouped by token-state transition

For consecutive pairs `(i, i+1)`, computes the chosen metric per layer
per position, then averages over positions (within each MM/MC/CC
group), pooled across all samples and blocks. One subplot per block,
one line per group on the layer x-axis.

**Use:** confirm that already-decided positions (CC) have stable
representations across passes; that still-mask positions (MM) have
stable-ish representations early; and that mask→clean transitions
(MC) carry the largest drop. The 0.95 reference line on the
`cosine_sim` variant gives a CKA-style "≈equivalent" marker.

#### 7b. `diff_heatmap` — per-sample (pass × position) heatmap

For each of `--n_samples_heatmap` samples, builds a per-block
`[n_passes-1 × num_masked]` heatmap whose cells are the layer-mean
metric (mean over layers 1..L; embed layer omitted). Cells where the
position was revealed during that pass-to-pass transition get a `×`
marker overlay.

A **shared `vmin`/`vmax`** is computed across all sampled plots per
metric, so cross-sample comparisons are visually meaningful.

**Use:** see actual decode trajectories. Where do reveals happen?
How does drift correlate with the reveal timing?

#### 7c. `drift_vs_pass0` — each iteration vs pass 0

Same drift computation as 7a, but the reference is **pass 0** (the
moment talk_rps gets cached in T3) instead of `i+1`. Two-panel
figure: MM | MC. Multiple lines, one per iteration `i = 1..max_iter`,
**95% CI shaded**. Aggregated across samples and the chosen blocks
(default `0,1` — late blocks have higher EOS-token noise).

**Use:** how stale does pass-0's representation get as decoding
progresses within a block? Maps directly to "is T3's pass-0 talk_rps
reuse OK?"

#### 7d. `prediction_overlap` — speculative-decoding-style shared-mass

Loads the model. For each (sample, block, pass `i`), projects the
last-layer hidden state at pass `i` and at the reference pass through
`final_norm + lm_head + softmax`, then computes

```
shared_mass(p, q) = Σ_x min(p(x), q(x))   ∈ [0, 1]
```

This is the closed-form expected acceptance rate from speculative
decoding (Leviathan et al. 2023, Chen et al. 2023): if the reference
distribution were used as a draft and pass-`i` as the target, what
fraction of probability mass is shared?

Two-panel plot, x-axis = iteration, y-axis = shared mass, 95% CI
shaded, with `n=…` annotations per iteration. Reference is `pass0`
(default, T3 talk_rps reuse story) or `prev` (consecutive cascade).
Heavier than the H5-only variants — model load adds ~30 s on a 3090.

**Use:** translate the cos_sim plot into a directly interpretable
prediction-overlap number. cos_sim is a proxy; `shared_mass` is what
actually controls speculative acceptance and (closely) what the talk
model would "see" if it consumed stale rps.

---

## Filters applied to all plots

All plot scripts apply two per-sample filters by default:

- **EOS cutoff.** For each sample, masked positions at-or-after the
  first EOS token in the generated 256 are dropped. (Block 7 of most
  samples is post-EOS noise.) Disable with `--no_eos_cutoff`.
- **Special-token subtraction.** Positions in the prompt that hold
  special tokens (BOS, chat-template markers like `<|im_start|>`,
  `<|im_end|>`, role labels) are excluded from "prefix" — these
  tokens carry structural rather than semantic information. Disable
  with `--no_special_filter`.

For older HDF5 files that don't contain `eos_pos_in_generated` and
`special_token_positions` attrs (recorded by the latest version of
`run_probes.py`), the plot scripts recompute them on-the-fly from the
saved `prompt_text` / `generated_text` using the model's tokenizer.
Special-token recovery is exact; EOS recovery is approximate (off by
1–2 tokens occasionally).

---

## Tunable knobs (all in `probe_runner/configs.py`)

- `record_blocks`: `"all"` (default — 8 blocks) or `[0, 3, 7]` to
  subsample.
- `v_norm_blocks`: `"all"` or `[0]` if disk is tight.
- `threshold`: parallel-decoding confidence threshold (default 0.9,
  Fast-dLLM v1 default).
- `gen_length`: 256.
- `block_length`: 32.
- `max_prompt_tokens`: 512 (prompts longer than this are truncated to
  the last 512 tokens).
- `attn_implementation`: `"manual_softmax"` — required because
  SDPA / flash hide attention weights.

---

## Sanity checks

`run_probes.py` prints a JSON summary after sample 0 of each model.
Verify:

- `all_blocks_recorded == 8`.
- `attn_row_sum_min` and `attn_row_sum_max` within `1e-2` of `1.0`
  (attention rows sum to 1).
- `mean_attn_to_pos0` — flagged in `meta.json` if > 0.1 so plotting
  can subtract it as a sink.
- `attn_shape[-1] == prompt_len + 32`.
- `h_shape[0] == n_layers + 1` (embedding + per-block outputs).

The split-prefix and main flow plotters also write
`diagnostics_<model>_<suffix>.json` next to their figures. Useful
values:

- `samples_with_eos`: how many of 100 samples emitted EOS within the
  256-token budget. If much less than ~80, your tokenizer's EOS
  handling differs from what we assume — widen `end_token_ids` in
  `run_probes.py`.
- `mean_eos_pos`: typical answer length. For GSM8K should be 100–200.
- `blocks_kept_per_sample[b]`: how many samples contribute to block
  b's average after the EOS filter. Block 7 typically has very few;
  block 0 should have all 100.

---

## Storage budget

Per sample at `prompt_len ≤ 512`, all 8 blocks recorded, after gzip:

| Component | LLaDA | Dream |
|---|---:|---:|
| `attn` (8 blocks, fp16) | ~340 MB | ~210 MB |
| `v_norm` (8 blocks, fp32) | ~21 MB | ~13 MB |
| `h_masked` (8 blocks, fp16) | ~70 MB | ~53 MB |
| **Subtotal (legacy)** | **~430 MB** | **~280 MB** |
| `h_per_pass` (8 blocks, ~7 passes, fp16) | ~470 MB | n/a (not implemented) |
| `token_state_per_pass` + `revealed_per_pass` + `pass_indices` | ~negligible | n/a |
| **Total with `--intra_block`** | **~900 MB** | **(legacy only)** |
| × 100 samples (legacy)         | **~43 GB** | **~28 GB** |
| × 100 samples (intra_block)    | **~90 GB** | n/a |

If disk is tight: set `v_norm_blocks = [0]` (~5% saving) or
`record_blocks = [0, 3, 7]` (~62% saving) in `configs.py`.

---

## Repo layout

```
probe_runner/
├── README.md                             this file
├── setup.sh                              clones Fast-dLLM v1 + installs Python deps
├── requirements.txt
├── __init__.py
├── configs.py                            PROBE_CONFIG; resolves FAST_DLLM_V1_PATH
├── storage.py                            HDF5 read/write with per-block groups
├── hooks.py                              ProbeHooks: monkey-patches attention; supports
│                                         legacy (pass-0 only) and intra_block (every pass)
├── llada_runner.py                       LLaDA generation with Fast-dLLM v1 dual-cache;
│                                         on_block_start / on_pass_start callbacks
├── dream_runner.py                       same protocol on Dream (intra_block supported;
│                                         pass-0 reveals only position s, passes ≥ 1 use
│                                         dual_cache + replace_position)
├── run_probes.py                         CLI entry: drives capture, writes HDF5 + meta.json
└── plots/
    ├── __init__.py
    ├── plot_info_flow_to_prefix.py       attn / flow / flow_normalized
    ├── plot_flow_split_prefix.py         3-way split (recent / distant / current_block)
    ├── plot_cka.py                       CKA(h_ℓ, h_L) per-block + pooled
    ├── plot_logit_lens.py                top-K overlap, KL, shared-mass vs final layer
    │                                     (loads model)
    └── plot_intra_block.py               drift_grouped / diff_heatmap / drift_vs_pass0 /
                                          prediction_overlap (last variant loads model)
```

External code lives outside this directory:

```
external/Fast-dLLM/                       fetched by setup.sh, NOT bundled
├── v1/
│   ├── llada/model/modeling_llada.py     imported by probe_runner/llada_runner.py
│   └── dream/model/modeling_dream.py     imported by probe_runner/dream_runner.py
└── …
```

---

## Known caveats

- **Eager / manual-softmax attention required for pass-0 captures.**
  SDPA / flash do not expose attention weights. The patched attention
  runs manual softmax during the probe-hooked forwards. Roughly
  halves throughput vs. flash, but only matters during the probe run,
  not during downstream training.
- **Logit-lens and `prediction_overlap` need the model loaded.**
  Other plots run from H5 alone in a couple of minutes; these two
  re-load the full model (LLaDA-8B or Dream-7B) so they can project
  hidden states through the LM head. Need ≥16 GB GPU memory; cost is
  ~5–10 min total on a 3090.
- **Single sample per batch.** The hooks assume `B=1`. Batching is
  possible but adds index-juggling; not implemented.
- **Dream's KV-cache format.** Fast-dLLM v1's local
  `model/modeling_dream.py` provides the `dual_cache + replace_position`
  interface. If Fast-dLLM upstream changes that interface,
  `dream_runner.py` may need a small alignment patch.
- **Dream `--intra_block` is implemented but lightly tested.** The
  per-pass loop mirrors the LLaDA one with two adaptations: pass 0
  only samples position `s` (block-start) — Dream's convention — so
  the recorded `revealed_per_pass[0]` for a Dream block has at most
  one True entry (at local index 0). Passes ≥ 1 are the standard
  parallel-confidence-thresholded reveal under `dual_cache=True` +
  `replace_position`. Hidden states at every pass are captured at
  all 32 block positions, same as LLaDA. Cross-check sanity-sample-0
  output before relying on bulk Dream intra-block runs.
- **Attention-sink subtraction is heuristic.** Default
  `attention_sink_positions=[0]`. If sample-0 sanity shows
  `mean_attn_to_pos0 < 0.1`, manually edit `meta.json` to update the
  sink list before plotting.
- **EOS-cutoff approximation on legacy files.** HDF5 files recorded
  before the `eos_pos_in_generated` attr was added recover EOS by
  re-tokenising `generated_text`. Off by 1–2 tokens occasionally;
  sufficient for filtering block-7 noise.

---

## How the analyses fit together

Reading order when interpreting a fresh capture:

1. **Logit lens** (analysis 6) — most direct prune signal. The layer
   at which `top_k_overlap` ≥ 0.95 and `shared_mass` ≥ 0.95 is the
   layer at which the model has effectively decided what to emit.
   Strong prune candidate from there onward.
2. **CKA** (analysis 5) — sanity check on (1). Should saturate at
   roughly the same layer as logit lens. Disagreements are
   diagnostic.
3. **Normalised flow** (analysis 3) — checks that prefix-flow ratio
   also stops changing where logit lens / CKA saturate.
4. **Split normalised flow** (analysis 4) — *why* did flow change?
   Recent / distant / current_block share-of-flow tells you whether
   the late layers shift to local context, intra-block reasoning, or
   plateau.
5. **Intra-block drift_grouped** (analysis 7a) — within a block, do
   already-decoded positions stay stable across denoise passes? CC
   ≫ MM ≫ MC by design.
6. **Intra-block drift_vs_pass0** (analysis 7c) — does pass-0
   talk_rps stay close enough to later-pass representations to reuse?
   The MM curve is the load-bearing signal for the T3 talk-model
   architecture.
7. **Intra-block prediction_overlap** (analysis 7d) — translate the
   cos_sim drift into a speculative-decoding-style acceptance rate.
   Hits ≈ 0.95 → "draft is essentially equivalent to target."
8. **Intra-block diff_heatmap** (analysis 7b) — qualitative single-
   sample inspection. Useful when something looks weird in the
   aggregated curves.
9. **Raw attn / flow** (analyses 1, 2) — sanity check only. These are
   dominated by sink mass and depth-induced norm growth respectively;
   don't drive the prune decision off them.

---

## Cross-references

- `T3_pruning_probe_step1to4.md` — full spec this repo implements
  (legacy capture + analyses 1–6).
- `T3_pruning_probe_step5to8.md` — what to do with the resulting
  plots: composite scoring, prune-count selection, retraining.
- `T3_overview.md`, `T3_drawbacks.md`, `T3_next_trial.md` — main
  project context (sit alongside, not required for running this
  repo).
- `T3_project/intra_block_drift_findings.md` — current takeaway from
  the intra-block analyses on the LLaDA-8B-Instruct probe data.
