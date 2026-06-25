#!/usr/bin/env bash
# Today's H200 batch: B1 (selectivity — the drafter's weakest link) + A2 (Design-B skip accuracy),
# at the realistic operating thresholds 0.5 and 0.6. Sequential, self-contained.
#
#     DMAX_MATH_PATH=./T3-DMax/DMax-Math-16B-moe-merge \
#     T3DMAX_ROOT=./T3-DMax \
#     bash probe_runner/run_today_batch.sh 2>&1 | tee today_batch.log
set -euo pipefail
: "${DMAX_MATH_PATH:?set DMAX_MATH_PATH}"
: "${T3DMAX_ROOT:?set T3DMAX_ROOT}"
THRESHOLDS="${THRESHOLDS:-0.5 0.6}"
B1_N="${B1_N:-100}"
A2_N="${A2_N:-200}"
CUT="${CUT:-9}"
OUT_ROOT="${OUT_ROOT:-fork_bounded_surrogate/results}"
cd "$(dirname "$0")/.."

echo "=== today batch: B1 + A2 @ thr {$THRESHOLDS} ==="; date
for thr in $THRESHOLDS; do
  echo; echo "######## B1 selectivity @ thr=$thr ########"
  python -m probe_runner.exp_b1_selectivity \
      --model_path "$DMAX_MATH_PATH" --t3dmax_root "$T3DMAX_ROOT" \
      --gsm8k_n "$B1_N" --commit_threshold "$thr" --feat_layer_shallow "$CUT" \
      --out_dir "$OUT_ROOT/B1_thr$thr"

  echo; echo "######## A2 Design-B skip @ thr=$thr (cut=L$CUT) ########"
  python -m probe_runner.exp_a2_skip_variant \
      --model_path "$DMAX_MATH_PATH" --t3dmax_root "$T3DMAX_ROOT" \
      --cut "$CUT" --commit_threshold "$thr" --gsm8k_n "$A2_N" \
      --out_dir "$OUT_ROOT/A2_thr$thr"
done

echo; echo "==================== verdicts ===================="
for thr in $THRESHOLDS; do
  echo "-- B1 thr=$thr --"; grep -E "GATE|draft_correct|probe_shallow" "$OUT_ROOT/B1_thr$thr/report.txt" 2>/dev/null | head -8 || echo "  (missing)"
  echo "-- A2 thr=$thr --"; grep -E "acc=" "$OUT_ROOT/A2_thr$thr/report.txt" 2>/dev/null || echo "  (missing)"
done
echo; echo "=== done ==="; date
