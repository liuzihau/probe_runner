#!/usr/bin/env bash
# B2 sweep: draft-then-correct economics across the meaningful conditions.
#   * threshold        : 0.5, 0.6 (operating points)
#   * draft_fill_frac  : 1.0 (fill all masked) and 0.5 (the contamination hedge — fill only the
#                        top-confidence half, per B1's finding that confidence predicts correctness)
# Corrector is FULL (corrector_cut=0) — the decisive test is "does oracle-draft + full-corrector beat
# partial-depth ALONE?", i.e. oracle saving_vs_none vs the per-threshold partial-depth bar.
# (The true drafter+partial-depth-corrector combo needs A1's cache mechanism; deferred.)
#
#     DMAX_MATH_PATH=./T3-DMax/DMax-Math-16B-moe-merge \
#     T3DMAX_ROOT=./T3-DMax \
#     bash probe_runner/run_b2_sweep.sh 2>&1 | tee b2_sweep.log
#
# Override: THRESHOLDS, FILLS, DRAFT_CUT, N, MODES, OUT_ROOT
set -euo pipefail
: "${DMAX_MATH_PATH:?set DMAX_MATH_PATH}"
: "${T3DMAX_ROOT:?set T3DMAX_ROOT}"
THRESHOLDS="${THRESHOLDS:-0.5 0.6}"
FILLS="${FILLS:-1.0 0.5}"
DRAFT_CUT="${DRAFT_CUT:-9}"
N="${N:-100}"
MODES="${MODES:-none trunc oracle}"
OUT_ROOT="${OUT_ROOT:-fork_bounded_surrogate/results}"
cd "$(dirname "$0")/.."

# partial-depth-alone saving (vs full) is the bar a drafter must beat — it differs by threshold
# (from E0: L9 vs full -> thr0.5 = 1-42.0/53.0 ≈ 0.21 ; thr0.6 = 1-46.0/55.6 ≈ 0.17)
bar_for_thr() { case "$1" in 0.5) echo 0.21;; 0.6) echo 0.17;; *) echo 0.20;; esac; }

echo "=== B2 sweep: thr={$THRESHOLDS} x fill={$FILLS}  draft_cut=L$DRAFT_CUT  N=$N ==="; date
for thr in $THRESHOLDS; do
  bar=$(bar_for_thr "$thr")
  for fill in $FILLS; do
    tag="thr${thr}_fill${fill}"
    echo; echo "######## B2 $tag  (partial-depth bar=$bar) ########"
    python -m probe_runner.exp_b2_selfspec \
        --model_path "$DMAX_MATH_PATH" --t3dmax_root "$T3DMAX_ROOT" \
        --modes $MODES --draft_cut "$DRAFT_CUT" --corrector_cut 0 \
        --commit_threshold "$thr" --draft_fill_frac "$fill" \
        --partial_depth_saving "$bar" --gsm8k_n "$N" \
        --out_dir "$OUT_ROOT/B2_$tag"
  done
done

echo; echo "==================== B2 verdicts ===================="
for thr in $THRESHOLDS; do
  for fill in $FILLS; do
    echo "-- thr=$thr fill=$fill --"
    grep -E "VERDICT|none|trunc|oracle|must beat" "$OUT_ROOT/B2_thr${thr}_fill${fill}/report.txt" 2>/dev/null \
      || echo "  (missing)"
  done
done
echo; echo "=== done ==="; date
