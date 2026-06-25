#!/usr/bin/env bash
# Sequential commit-threshold sweep for E1 (the decider) + E0 (the bar).
# E1 at thr=0.3 already ran (results/E1); this re-tests the higher-quality operating
# points 0.5/0.6 to check whether the E1 GATE FAIL is a threshold artifact.
#
# Run from anywhere (it cd's to the dir containing probe_runner/):
#     DMAX_MATH_PATH=./T3-DMax/DMax-Math-16B-moe-merge \
#     T3DMAX_ROOT=./T3-DMax \
#     bash probe_runner/run_threshold_sweep.sh 2>&1 | tee sweep.log
#
# Override any of: THRESHOLDS, E1_N, E0_N, GEN_LENGTH, BLOCK_LENGTH, E0_CONFIGS, OUT_ROOT
set -euo pipefail

: "${DMAX_MATH_PATH:?set DMAX_MATH_PATH to the DMax-Math-16B weight dir}"
: "${T3DMAX_ROOT:?set T3DMAX_ROOT to the T3-DMax checkout}"
THRESHOLDS="${THRESHOLDS:-0.5 0.6}"
E1_N="${E1_N:-100}"
E0_N="${E0_N:-200}"
GEN_LENGTH="${GEN_LENGTH:-512}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
E0_CONFIGS="${E0_CONFIGS:-full L9M4 L9}"
OUT_ROOT="${OUT_ROOT:-fork_bounded_surrogate/results}"

# cd to the parent of probe_runner/ so `python -m probe_runner...` imports and the
# relative --out_dir matches the existing results layout.
cd "$(dirname "$0")/.."

echo "=================================================="
echo " commit-threshold sweep:  $THRESHOLDS"
echo " DMAX_MATH_PATH = $DMAX_MATH_PATH"
echo " T3DMAX_ROOT    = $T3DMAX_ROOT"
echo " cwd            = $(pwd)"
echo " E1_N=$E1_N  E0_N=$E0_N  gen=$GEN_LENGTH  block=$BLOCK_LENGTH"
echo "=================================================="
date

for thr in $THRESHOLDS; do
  echo; echo "######################## threshold = $thr ########################"

  echo "--- E1 (decider) @ thr=$thr ---"
  python -m probe_runner.exp_e1_easy_uncertain \
      --model_path "$DMAX_MATH_PATH" --t3dmax_root "$T3DMAX_ROOT" \
      --gsm8k_n "$E1_N" --gen_length "$GEN_LENGTH" --block_length "$BLOCK_LENGTH" \
      --hi 0.9 --loo_batch 8 --commit_threshold "$thr" \
      --out_dir "$OUT_ROOT/E1_thr$thr"

  echo "--- E0 (bar) @ thr=$thr ---"
  python -m probe_runner.exp_e0_baseline \
      --model_path "$DMAX_MATH_PATH" --t3dmax_root "$T3DMAX_ROOT" \
      --configs $E0_CONFIGS --gsm8k_n "$E0_N" --gen_length "$GEN_LENGTH" --block_length "$BLOCK_LENGTH" \
      --commit_threshold "$thr" \
      --out_dir "$OUT_ROOT/E0_thr$thr"
done

echo; echo "==================== E1 gate verdicts ===================="
for thr in $THRESHOLDS; do
  echo "-- thr=$thr --"
  grep -E "easy_uncertain_frac|cannibalization_overlap|high_value_frac|GATE" \
      "$OUT_ROOT/E1_thr$thr/report.txt" 2>/dev/null || echo "  (no report — run may have failed)"
done

echo; echo "=== sweep done ==="; date
echo "results under: $OUT_ROOT/{E1,E0}_thr{${THRESHOLDS// /,}}"
