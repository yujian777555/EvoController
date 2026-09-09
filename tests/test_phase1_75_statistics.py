"""Tests for :mod:`experiments.analyze_phase1_75` (Phase 1.75 statistics).

Unit tests use synthetic arrays only (no dependency on a real
``results.json``); the end-to-end test drives ``main`` on a hand-written
fake results file under ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from scipy.stats import wilcoxon

from experiments.analyze_phase1_75 import (
    CONTROL_ARM,
    PRIMARY_COMPARISONS,
    failure_rate,
    holm_correction,
    main,
    paired_bootstrap_ci,
    paired_wilcoxon,
)

# ---------------------------------------------------------------------------
# Unit tests: statistical primitives
# ---------------------------------------------------------------------------


def test_holm_correction_hand_example():
    """Holm-Bonferroni on hand-verified sequences.

    For [0.01, 0.03, 0.04] (m=3, sorted): adj(1)=3*0.01=0.03,
    adj(2)=max(0.03, 2*0.03)=0.06, adj(3)=max(0.06, 1*0.04)=0.06 —
    exactly the spec's worked example [0.01, 0.04, 0.03] -> [0.03,
    0.06, 0.06].
    """
    adjusted = holm_correction({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    # Input key order must not matter.
    assert holm_correction({"b": 0.04, "c": 0.03, "a": 0.01}) == pytest.approx(
        adjusted
    )
    # No-monotonicity-fix case: 3*0.01=0.03, 2*0.02=0.04, 1*0.05=0.05.
    assert holm_correction({"x": 0.01, "y": 0.02, "z": 0.05}) == pytest.approx(
        {"x": 0.03, "y": 0.04, "z": 0.05}
    )
    # Single-test family is unchanged.
    assert holm_correction({"only": 0.07}) == pytest.approx({"only": 0.07})
    # Adjusted values are capped at 1.0.
    assert holm_correction({"x": 0.9, "y": 0.8}) == pytest.approx(
        {"x": 1.0, "y": 1.0}
    )
    assert holm_correction({}) == {}
    with pytest.raises(ValueError):
        holm_correction({"bad": 1.5})
    with pytest.raises(ValueError):
        holm_correction({"bad": -0.1})


def test_paired_bootstrap_ci_deterministic():
    """Same seed -> bit-identical CI; the CI brackets the point estimate."""
    a = [5.1, 4.7, 6.3, 5.9, 4.8, 6.1]
    b = [4.2, 4.9, 5.1, 4.4, 5.3, 4.6]
    first = paired_bootstrap_ci(a, b, n_boot=2000, seed=7)
    second = paired_bootstrap_ci(a, b, n_boot=2000, seed=7)
    assert first == second  # exact, bit-for-bit
    assert first[1] <= first[0] <= first[2]


def test_paired_bootstrap_ci_large_effect_excludes_zero():
    """A large paired effect yields a CI strictly above zero, and a
    symmetric zero-effect case yields a CI straddling zero."""
    _, lo, hi = paired_bootstrap_ci(
        [10, 11, 12, 13, 20], [1, 2, 3, 4, 5], n_boot=2000, seed=0
    )
    assert lo > 0.0
    assert hi > 0.0
    median_diff, lo, hi = paired_bootstrap_ci(
        [3, 4, 1, 2], [1, 2, 3, 4], n_boot=2000, seed=0
    )
    assert median_diff == pytest.approx(0.0, abs=1e-12)
    assert lo <= 0.0 <= hi


def test_paired_bootstrap_ci_input_validation():
    """Length mismatch, empty input, and non-positive n_boot are rejected."""
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], [])
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0, 2.0], [1.0, 2.0], n_boot=0)


def test_paired_wilcoxon_direction():
    """`alternative="greater"` on a consistent positive shift gives a
    small p; `"less"` gives a large p; the wrapper matches a direct
    scipy call with the fixed conventions."""
    a = [10, 12, 14, 16, 18, 20, 22, 24]
    b = [value - 3 for value in a]
    p_greater = paired_wilcoxon(a, b, alternative="greater")
    p_less = paired_wilcoxon(a, b, alternative="less")
    assert p_greater == pytest.approx(0.5**8)  # 8/8 positive shifts, exact
    assert p_greater < 0.05
    assert p_less > 0.9
    assert p_greater == float(
        wilcoxon(a, b, zero_method="zsplit", alternative="greater").pvalue
    )
    with pytest.raises(ValueError):
        paired_wilcoxon([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_wilcoxon([], [])


def test_failure_rate_boundaries():
    """failure_rate edges: all True -> 1.0, mixed -> fraction, empty -> 0.0."""
    assert failure_rate([True, True, True, True]) == 1.0
    assert failure_rate([False, False]) == 0.0
    assert failure_rate([True, False, True]) == pytest.approx(2.0 / 3.0)
    assert failure_rate([]) == 0.0
    assert failure_rate(np.array([True, True])) == 1.0


# ---------------------------------------------------------------------------
# End-to-end test on a hand-written fake results file
# ---------------------------------------------------------------------------

_FAKE_PROBLEMS = ["zdt1", "zdt2", "zdt3"]
_FAKE_SEEDS = [10, 11, 12, 13, 14]
#: final_hv = base[arm] + slope[arm] * i + 0.05 * problem_index
_HV_BASE = {
    CONTROL_ARM: 0.80,
    "static_full_global": 0.70,
    "open_loop_global": 0.60,
    "generation_only_mlp": 0.75,
    "state_scrambled_mlp": 0.83,
    "fixed_nsga2": 0.65,
}
_HV_SLOPE = {
    CONTROL_ARM: 0.01,
    "static_full_global": 0.01,
    "open_loop_global": 0.005,
    "generation_only_mlp": 0.02,
    "state_scrambled_mlp": 0.002,
    "fixed_nsga2": 0.01,
}


def _fake_results() -> Dict[str, Any]:
    """Build a small deterministic results dictionary.

    ``static_full_global`` fails on the first two zdt1 seeds and
    ``open_loop_global`` fails on every zdt3 seed; everything else
    succeeds.
    """
    arms = ["fixed_nsga2", *PRIMARY_COMPARISONS, CONTROL_ARM]
    runs: Dict[str, Any] = {}
    for arm in arms:
        for p_index, problem in enumerate(_FAKE_PROBLEMS):
            for i, seed in enumerate(_FAKE_SEEDS):
                hv = _HV_BASE[arm] + _HV_SLOPE[arm] * i + 0.05 * p_index
                failed = (
                    arm == "static_full_global"
                    and problem == "zdt1"
                    and i < 2
                ) or (arm == "open_loop_global" and problem == "zdt3")
                runs[f"{arm}|{problem}|{seed}"] = {
                    "final_hv": hv,
                    "final_igd": 1.0 - hv,
                    "auc_hv": hv / 2.0,
                    "runtime_sec": 1.0,
                    "failed": failed,
                    "trajectory_file": "unused.json",
                }
    return {
        "config": {"note": "synthetic fixture"},
        "failure_thresholds": {problem: 0.5 for problem in _FAKE_PROBLEMS},
        "arms": arms,
        "problems": _FAKE_PROBLEMS,
        "eval_seeds": _FAKE_SEEDS,
        "runs": runs,
    }


def test_end_to_end_fake_results(tmp_path: Path):
    """main() on a fake results.json produces the documented outputs with
    hand-verifiable numbers and byte-identical reruns."""
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_fake_results()), encoding="utf-8")
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    assert main(["--results", str(results_path), "--out-dir", str(out_a)]) == 0
    assert main(["--results", str(results_path), "--out-dir", str(out_b)]) == 0

    for name in ("stats.json", "stats.md"):
        assert (out_a / name).is_file()
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()

    report = json.loads((out_a / "stats.json").read_text(encoding="utf-8"))

    # Descriptives of the control arm on zdt1 / final_hv:
    # values [0.80, 0.81, 0.82, 0.83, 0.84].
    control = report["per_problem"]["zdt1"]["final_hv"]["arms"][CONTROL_ARM]
    assert control["n"] == 5
    assert control["mean"] == pytest.approx(0.82, rel=1e-9)
    assert control["median"] == pytest.approx(0.82, rel=1e-9)
    assert control["std"] == pytest.approx(0.0158113883008419, rel=1e-9)
    assert "fixed_nsga2" in report["per_problem"]["zdt1"]["final_hv"]["arms"]

    # static_full_global has a constant +0.10 paired difference on zdt1.
    static = report["per_problem"]["zdt1"]["final_hv"]["comparisons"][
        "static_full_global"
    ]
    assert static["median_diff"] == pytest.approx(0.10, abs=1e-12)
    assert static["ci_lo"] <= static["median_diff"] <= static["ci_hi"]
    assert static["ci_lo"] > 0.09
    assert static["wilcoxon_p"] == pytest.approx(0.5**5)
    assert static["holm_p"] >= static["wilcoxon_p"] - 1e-12

    # state_scrambled_mlp beats the control on zdt1 (negative diff).
    scrambled = report["per_problem"]["zdt1"]["final_hv"]["comparisons"][
        "state_scrambled_mlp"
    ]
    assert scrambled["median_diff"] < -0.01

    # Holm wiring: stored adjusted p equals holm_correction over the
    # per-problem raw p-values of the same (metric, comparison) family.
    family = {
        problem: report["per_problem"][problem]["final_hv"]["comparisons"][
            "static_full_global"
        ]["wilcoxon_p"]
        for problem in _FAKE_PROBLEMS
    }
    expected = holm_correction(family)
    for problem in _FAKE_PROBLEMS:
        stored = report["per_problem"][problem]["final_hv"]["comparisons"][
            "static_full_global"
        ]["holm_p"]
        assert stored == pytest.approx(expected[problem])

    # Failure-rate aggregation per arm x problem.
    rates = report["failure_rates"]
    assert rates["static_full_global"]["zdt1"] == pytest.approx(0.4)
    assert rates["open_loop_global"]["zdt3"] == 1.0
    assert rates["open_loop_global"]["overall"] == pytest.approx(1.0 / 3.0)
    assert rates[CONTROL_ARM]["overall"] == 0.0

    # Markdown tables cover every problem and comparison.
    markdown = (out_a / "stats.md").read_text(encoding="utf-8")
    assert "## final_hv" in markdown
    assert "## auc_hv" in markdown
    assert "## Failure rates by arm x problem" in markdown
    for comparison in PRIMARY_COMPARISONS:
        assert f"vs {comparison}: p_Holm" in markdown
    for problem in _FAKE_PROBLEMS:
        assert f"| {problem} |" in markdown
    assert "| static_full_global |" in markdown
