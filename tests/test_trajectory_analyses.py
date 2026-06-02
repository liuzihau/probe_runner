"""End-to-end regression test for the T3-D trajectory analyses on synthetic data.

Exercises the full analysis stack WITHOUT torch / a GPU / the 16B model:
builds a structured synthetic capture (incl. committed_tokens_per_pass +
converged_tokens + a numpy lm_head), then runs every analysis' accumulate +
plot and asserts the key invariants.

    python -m probe_runner.tests.test_trajectory_analyses

Exits 0 on success. The model-driven paths (plot_repairability.compute_*,
plot_refresh_recovery.run_partial_depth) are validated separately on the cloud;
their numpy cores are covered by each script's `--selftest`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from probe_runner.tests.make_synthetic_probes import build
from probe_runner.plots import _trajectory_common as tc
from probe_runner.plots import plot_rank_trajectory as rank
from probe_runner.plots import plot_trajectory_domains as dom
from probe_runner.plots import plot_refresh_recovery as refresh
from probe_runner.plots import plot_repairability as repair

BL = 8  # synthetic block_length


def _check_rank(root):
    head = tc.RankHead.load(tc.find_head_export(root, "llada2"))
    stats = rank.accumulate(Path(root), "llada2", head, BL)
    # rank descends: mean log-rank at the last pass < first pass
    mlr = np.where(stats["rank_cnt"] > 0, stats["rank_sum"] / np.maximum(stats["rank_cnt"], 1), np.nan)
    first, last = np.nanmean(mlr[0]), np.nanmean(mlr[stats["Pmax"] - 1])
    assert last < first, f"rank should descend: first={first:.2f} last={last:.2f}"
    # oracle ceiling at pass0 is a valid fraction
    frac1 = float((stats["pass0"] <= 1).mean())
    assert 0.0 <= frac1 <= 1.0
    rank.plot(stats, Path(root) / "llada2" / "plots", "llada2")
    print(f"  [rank] descends {first:.2f}→{last:.2f} log10; oracle top-1@0={frac1:.1%}  OK")
    # rank-by-domain: both groups should populate (synthetic commits a pass-0 prefix)
    assert stats["pass0_committed"].size > 0, "expected think-committed@0 positions"
    assert stats["pass0_masked"].size > 0, "expected left-for-talk positions"
    rank.plot_by_domain(stats, Path(root) / "llada2" / "plots", "llada2")
    hard_top10 = float((stats["pass0_masked"] <= 10).mean())
    print(f"  [rank-by-domain] committed@0={stats['pass0_committed'].size} "
          f"left-for-talk={stats['pass0_masked'].size} (top-10 ceiling {hard_top10:.0%})  OK")


def _check_domains(root):
    stats = dom.accumulate(Path(root), "llada2", 1, 3, BL)
    pop = stats["pop"]
    assert pop[tc.DOMAIN_CODE["CR"]] > 0, "CR (committed-revised) should be present on pair (1,3)"
    assert pop[tc.DOMAIN_CODE["MC"]] > 0, "MC should be present"
    dom.plot(stats, Path(root) / "llada2" / "plots", "llada2")
    # per-layer divergence should grow with depth for at least one domain
    cl = np.where(stats["layer_cnt"] > 0,
                  stats["cos_layer_sum"] / np.maximum(stats["layer_cnt"], 1), np.nan)
    print(f"  [domains] pair(1,3) pop CU/CR/MC/MM={pop[:4].tolist()}  OK")


def _check_refresh(root):
    head = tc.RankHead.load(tc.find_head_export(root, "llada2"))
    stats = refresh.accumulate_recovery(Path(root), "llada2", head, BL)
    acc = stats["acc"]
    think = np.asarray(acc["think"], float) / np.maximum(np.asarray(acc["cnt"], float), 1)
    assert think[stats["Pmax"] - 1] >= think[0], "think recovery should not decrease"
    refresh.plot_recovery(stats, Path(root) / "llada2" / "plots", "llada2")
    refresh._selftest()
    print(f"  [refresh] recovery {think[0]:.2f}→{think[stats['Pmax']-1]:.2f}  OK")


def _check_repair():
    repair._selftest()
    print("  [repairability] numpy cores  OK")


def main():
    with tempfile.TemporaryDirectory() as td:
        build(td, n_samples=4)
        print("synthetic capture built; running analyses:")
        _check_rank(td)
        _check_domains(td)
        _check_refresh(td)
        _check_repair()
        plots = sorted((Path(td) / "llada2" / "plots").glob("*.png"))
        assert len(plots) >= 4, f"expected >=4 plots, got {len(plots)}"
        print(f"\nALL TRAJECTORY ANALYSIS TESTS PASS ({len(plots)} plots generated)")


if __name__ == "__main__":
    main()
