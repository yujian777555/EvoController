"""Paper-grade statistical analysis for the Phase 1.75 experiment.

This module post-processes ``results/phase1_75/results.json`` — the
evaluation dump of the Phase 1.75 arm grid — into publication-ready
statistics. The ``runs`` dictionary of that file maps
``"<arm>|<problem>|<seed>"`` to per-run outcomes (``final_hv``,
``final_igd``, ``auc_hv``, ``runtime_sec``, ``failed``).

Orientation. Every comparison is control-minus-comparison:
``median_diff = median(control - comparison)`` computed per problem over
seed-aligned value pairs, tested with a one-sided Wilcoxon signed-rank
test (``alternative="greater"``, ``zero_method="zsplit"``) of the
hypothesis that the control arm ``mlp2_closed_loop_normalized`` exceeds
the comparison arm. Both primary metrics (``final_hv``, ``auc_hv``) are
maximized, so a positive ``median_diff`` (and a small p-value) favours
the control.

Multiple testing. For every ``(metric, comparison)`` pair the per-problem
Wilcoxon p-values form one family and are Holm-Bonferroni corrected
across problems.

Conventions. ``std`` is the sample standard deviation (ddof=1); the
bootstrap CI of the paired median difference is the percentile interval
(2.5 / 97.5) over ``n_boot`` paired resamples of the seed-aligned
differences, drawn with ``numpy.random.Generator(PCG64(seed))`` and
therefore fully deterministic; failed runs stay in the metric statistics
and failure rates are reported separately per arm x problem.

Outputs. ``stats.json`` (every number) and ``stats.md`` (per metric one
row per problem and one column group per primary comparison, including
Holm-adjusted p-values, plus an arm x problem failure-rate table) are
written to ``--out-dir``.

Example:
    python experiments/analyze_phase1_75.py \
        --results results/phase1_75/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results" / "phase1_75" / "results.json"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "phase1_75"

#: Fixed control arm of all primary comparisons.
CONTROL_ARM = "mlp2_closed_loop_normalized"

#: Comparison arms contrasted against :data:`CONTROL_ARM`.
PRIMARY_COMPARISONS = (
    "static_full_global",
    "open_loop_global",
    "generation_only_mlp",
    "state_scrambled_mlp",
)

#: Maximized metrics reported per problem.
PRIMARY_METRICS = ("final_hv", "auc_hv")

#: Bootstrap resample count and seed for the paired median CI.
DEFAULT_N_BOOT = 10_000
BOOTSTRAP_SEED = 0

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Statistical primitives (exact contract signatures)
# ---------------------------------------------------------------------------


def holm_correction(p_values: dict[str, float]) -> dict[str, float]:
    """Apply the Holm-Bonferroni step-down correction.

    The p-values are sorted ascending; the i-th smallest is compared
    against ``alpha / (m - i + 1)`` in the underlying test procedure,
    and its adjusted value is ``(m - i + 1) * p_(i)`` enforced to be
    monotone non-decreasing in i and capped at 1.

    Args:
        p_values: Mapping from comparison name (e.g. ``"zdt1"``) to its
            raw p-value in ``[0, 1]``.

    Returns:
        Mapping from the same names to the Holm-adjusted p-values.

    Raises:
        ValueError: If any p-value lies outside ``[0, 1]`` or is NaN.
    """
    for name, p in p_values.items():
        if np.isnan(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"invalid p-value for {name!r}: {p!r}")
    m = len(p_values)
    if m == 0:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p) in enumerate(ordered, start=1):
        running_max = max(running_max, (m - rank + 1) * p)
        adjusted[name] = min(running_max, 1.0)
    return adjusted


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired median difference (``a - b``) with a 95% bootstrap CI.

    The differences are resampled with replacement ``n_boot`` times; the
    CI is the percentile interval (2.5th / 97.5th) of the resampled
    medians. Sampling uses ``numpy.random.Generator(PCG64(seed))``, so
    identical arguments always yield bit-identical results.

    Args:
        a: First sample (e.g. the control arm's per-seed values).
        b: Second sample, paired element-wise with ``a``.
        n_boot: Number of bootstrap resamples.
        seed: Seed of the PCG64 generator.

    Returns:
        ``(median_diff, ci_lo, ci_hi)`` where ``median_diff`` is the
        median of ``a - b`` and ``[ci_lo, ci_hi]`` its 95% CI.

    Raises:
        ValueError: If the inputs have different lengths, are empty, or
            ``n_boot`` is not positive.
    """
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = diffs.size
    if np.asarray(a).size != np.asarray(b).size:
        raise ValueError("a and b must have the same length")
    if n == 0:
        raise ValueError("a and b must be non-empty")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    median_diff = float(np.median(diffs))
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_medians = np.median(diffs[indices], axis=1)
    ci_lo, ci_hi = np.percentile(boot_medians, (2.5, 97.5))
    return median_diff, float(ci_lo), float(ci_hi)


def paired_wilcoxon(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "greater",
) -> float:
    """One-sided paired Wilcoxon signed-rank p-value for ``a > b``.

    Thin, convention-fixing wrapper around
    :func:`scipy.stats.wilcoxon` with ``zero_method="zsplit"`` (zero
    differences receive random split ranks, keeping the test exact for
    data with ties at zero).

    Args:
        a: First sample (e.g. the control arm), paired with ``b``.
        b: Second sample.
        alternative: One of ``"greater"`` (tests ``a > b``),
            ``"less"``, or ``"two-sided"``.

    Returns:
        The Wilcoxon p-value.

    Raises:
        ValueError: If the inputs have different lengths or are empty
            (propagated from scipy for degenerate inputs).
    """
    a_array = np.asarray(a, dtype=float)
    b_array = np.asarray(b, dtype=float)
    if a_array.size != b_array.size:
        raise ValueError("a and b must have the same length")
    if a_array.size == 0:
        raise ValueError("a and b must be non-empty")
    result = wilcoxon(
        a_array, b_array, zero_method="zsplit", alternative=alternative
    )
    return float(result.pvalue)


def failure_rate(flags: Sequence[bool]) -> float:
    """Fraction of failed runs.

    Args:
        flags: Per-run ``failed`` flags (any boolean sequence).

    Returns:
        ``sum(flags) / len(flags)``, defined as ``0.0`` for an empty
        sequence.
    """
    if len(flags) == 0:
        return 0.0
    return float(np.mean(np.asarray(flags, dtype=bool)))


# ---------------------------------------------------------------------------
# Analysis driver
# ---------------------------------------------------------------------------


def _run_key(arm: str, problem: str, seed: int) -> str:
    """Compose the ``"<arm>|<problem>|<seed>"`` runs-dictionary key."""
    return f"{arm}|{problem}|{seed}"


def _metric_vector(
    runs: Dict[str, Dict[str, Any]],
    arm: str,
    problem: str,
    metric: str,
    eval_seeds: Sequence[int],
) -> np.ndarray:
    """Collect a metric vector for one arm x problem, ordered by seed.

    Args:
        runs: The ``runs`` mapping of the results file.
        arm: Arm name.
        problem: Problem name.
        metric: Metric key, e.g. ``"final_hv"``.
        eval_seeds: Evaluation seeds; their order defines the pairing.

    Returns:
        Array of per-seed metric values.

    Raises:
        ValueError: If a required run entry is missing.
    """
    values = []
    for seed in eval_seeds:
        key = _run_key(arm, problem, seed)
        try:
            values.append(float(runs[key][metric]))
        except KeyError as exc:
            raise ValueError(f"missing run entry or metric for {key!r}") from exc
    return np.asarray(values, dtype=float)


def analyze(
    results: Dict[str, Any],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Compute the full statistics report from a results dictionary.

    Args:
        results: Parsed content of ``results.json`` (must contain
            ``arms``, ``problems``, ``eval_seeds``, ``runs``).
        n_boot: Bootstrap resample count for the paired median CIs.
        seed: PCG64 seed of the bootstrap.

    Returns:
        The report dictionary written to ``stats.json``.

    Raises:
        ValueError: If required arms or run entries are missing.
    """
    arms = list(results["arms"])
    problems = list(results["problems"])
    eval_seeds = list(results["eval_seeds"])
    runs = results["runs"]

    required_arms = [CONTROL_ARM, *PRIMARY_COMPARISONS]
    missing = [arm for arm in required_arms if arm not in arms]
    if missing:
        raise ValueError(f"results.json is missing required arms: {missing}")

    per_problem: Dict[str, Any] = {}
    for problem in problems:
        per_problem[problem] = {}
        for metric in PRIMARY_METRICS:
            arm_stats: Dict[str, Any] = {}
            for arm in arms:
                values = _metric_vector(runs, arm, problem, metric, eval_seeds)
                flags = [bool(runs[_run_key(arm, problem, s)]["failed"]) for s in eval_seeds]
                arm_stats[arm] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "n": int(values.size),
                    "failure_rate": failure_rate(flags),
                }
            control_values = _metric_vector(
                runs, CONTROL_ARM, problem, metric, eval_seeds
            )
            comparisons: Dict[str, Any] = {}
            for comparison in PRIMARY_COMPARISONS:
                comp_values = _metric_vector(
                    runs, comparison, problem, metric, eval_seeds
                )
                median_diff, ci_lo, ci_hi = paired_bootstrap_ci(
                    control_values, comp_values, n_boot=n_boot, seed=seed
                )
                comparisons[comparison] = {
                    "median_diff": median_diff,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "wilcoxon_p": paired_wilcoxon(
                        control_values, comp_values, alternative="greater"
                    ),
                }
            per_problem[problem][metric] = {
                "arms": arm_stats,
                "comparisons": comparisons,
            }

    # Holm-Bonferroni per (metric, comparison) family across problems.
    for metric in PRIMARY_METRICS:
        for comparison in PRIMARY_COMPARISONS:
            family = {
                problem: per_problem[problem][metric]["comparisons"][
                    comparison
                ]["wilcoxon_p"]
                for problem in problems
            }
            adjusted = holm_correction(family)
            for problem in problems:
                per_problem[problem][metric]["comparisons"][comparison][
                    "holm_p"
                ] = adjusted[problem]

    failure_rates: Dict[str, Dict[str, float]] = {}
    for arm in arms:
        per_arm: Dict[str, float] = {}
        all_flags: List[bool] = []
        for problem in problems:
            flags = [
                bool(runs[_run_key(arm, problem, s)]["failed"])
                for s in eval_seeds
            ]
            all_flags.extend(flags)
            per_arm[problem] = failure_rate(flags)
        per_arm["overall"] = failure_rate(all_flags)
        failure_rates[arm] = per_arm

    return {
        "meta": {
            "control_arm": CONTROL_ARM,
            "primary_comparisons": list(PRIMARY_COMPARISONS),
            "primary_metrics": list(PRIMARY_METRICS),
            "n_problems": len(problems),
            "n_eval_seeds": len(eval_seeds),
            "arms": arms,
            "conventions": {
                "diff_orientation": (
                    "median(control - comparison); positive favours the "
                    "control because both metrics are maximized"
                ),
                "wilcoxon": "one-sided greater (control > comparison), zero_method=zsplit",
                "std_ddof": 1,
                "bootstrap": {
                    "n_boot": n_boot,
                    "seed": seed,
                    "rng": "numpy Generator(PCG64)",
                    "ci": "percentile 2.5/97.5 of resampled paired medians",
                },
                "holm_family": "per (metric, comparison) across problems",
                "failed_runs": (
                    "included in metric statistics; failure rates "
                    "reported separately"
                ),
            },
        },
        "per_problem": per_problem,
        "failure_rates": failure_rates,
    }


def _fmt(value: float) -> str:
    """Format a number for the markdown tables (4 significant digits)."""
    return f"{value:.4g}"


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the report as the ``stats.md`` document.

    Args:
        report: Output of :func:`analyze`.

    Returns:
        The full markdown text.
    """
    meta = report["meta"]
    lines: List[str] = []
    lines.append("# Phase 1.75 statistical report")
    lines.append("")
    lines.append(
        f"- Control arm: `{meta['control_arm']}`; "
        f"{meta['n_eval_seeds']} evaluation seeds per arm x problem."
    )
    lines.append(
        "- Delta columns are `median(control - comparison)` with a "
        "percentile bootstrap 95% CI; positive favours the control "
        "(both metrics are maximized)."
    )
    lines.append(
        "- `p` is the one-sided Wilcoxon signed-rank test "
        '(`greater`, `zero_method=zsplit`); `p_Holm` is the '
        "Holm-Bonferroni adjustment across the "
        f"{meta['n_problems']} problems within each (metric, comparison) "
        "family."
    )
    lines.append("")

    problems = list(report["per_problem"].keys())
    for metric in report["meta"]["primary_metrics"]:
        lines.append(f"## {metric} — primary comparisons")
        lines.append("")
        header = ["problem"]
        for comparison in report["meta"]["primary_comparisons"]:
            header.extend(
                [
                    f"vs {comparison}: delta_med",
                    f"vs {comparison}: 95% CI",
                    f"vs {comparison}: p",
                    f"vs {comparison}: p_Holm",
                ]
            )
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for problem in problems:
            comparisons = report["per_problem"][problem][metric][
                "comparisons"
            ]
            row = [problem]
            for comparison in report["meta"]["primary_comparisons"]:
                stats = comparisons[comparison]
                row.extend(
                    [
                        _fmt(stats["median_diff"]),
                        f"[{_fmt(stats['ci_lo'])}, {_fmt(stats['ci_hi'])}]",
                        _fmt(stats["wilcoxon_p"]),
                        _fmt(stats["holm_p"]),
                    ]
                )
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Failure rates by arm x problem")
    lines.append("")
    header = ["arm", *problems, "overall"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for arm, per_arm in report["failure_rates"].items():
        row = [arm, *[_fmt(per_arm[p]) for p in problems], _fmt(per_arm["overall"])]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Paper-grade statistics for the Phase 1.75 results "
            "(paired comparisons vs the closed-loop control, Holm "
            "correction, bootstrap CIs, failure rates)."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"results.json to analyse (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory for stats.json / stats.md (default: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the analysis and write ``stats.json`` and ``stats.md``.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    with Path(args.results).open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    report = analyze(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats_json = args.out_dir / "stats.json"
    with stats_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    stats_md = args.out_dir / "stats.md"
    stats_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"wrote {stats_json}")
    print(f"wrote {stats_md}")
    print(
        f"control={CONTROL_ARM}; comparisons={len(PRIMARY_COMPARISONS)}; "
        f"problems={report['meta']['n_problems']}; "
        f"seeds={report['meta']['n_eval_seeds']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
