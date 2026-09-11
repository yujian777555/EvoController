from __future__ import annotations

"""Phase 2.75D Task 5: dynamics diagnosis of the difficult landscapes.

Phase 2.75 failed on ZDT4 although the advantage ranking there is *good*
(Spearman 0.65-0.74 on all 30 states) and the action-effect SNR is the highest
of the five problems (6.21). Ranking ability and signal strength are therefore
excluded as explanations, and ``docs/PHASE2_75D_PLAN.md`` asks for a dynamics
view instead: diversity recovery, local-optimum escape, exploration duration
and the action-selection trajectory.

For one problem (the sharding unit) this script branches every candidate action
from every harvested snapshot and records **every generation** of the branch:

* quality metrics — ``hypervolume``, ``igd`` and ``diversity_spread`` of the
  nondominated front (``metrics.indicators``), i.e. the curve, not just its end;
* exploration proxies — the applied ``mutation_probability``/operator, the mean
  per-variable standard deviation of the population (the real "how much is
  still being explored" signal), the mean per-variable span and the mean
  absolute population movement ``|x_t - x_{t-1}|`` between generations;
* escape dynamics — the longest run of stagnating generations (HV gain below
  ``--stagnation-epsilon``) and the first generation whose HV reaches 10% and
  50% of the problem's reference hypervolume.

Candidate layout follows the Phase-2.75D protocol: index 0 is the controller
action, index 1 the NSGA-II default action (``kind="default"``, disable with
``--no-include-default-action``), then the sampled alternatives — the same
construction ``evaluate-horizon`` uses, reused from
``experiments/counterfactual_actions.py``. The branch RNG is this script's own
deterministic scheme, ``Generator(PCG64([snapshot_hash_seed, candidate_index,
rep]))``; it differs from ``evaluate-horizon``'s ``crc32``-derived branch seeds,
so these traces are reproducible but **not** numerically comparable to the
intervention files.

Output (``--out``, default ``results/phase2_75d/hard_landscape_diagnosis.json``)::

    {"problem": ..., "config": {...},
     "per_candidate": [{"index", "kind", "action", "hv_trace", "diversity_trace",
                        "population_std_trace", "final_hv", "final_hv_mean",
                        "stagnation_gens", "stagnation_gens_total",
                        "escape_gen_10pct", "escape_gen_50pct",
                        "exploration_duration", ...}],
     "summary": {"best_action_by_final_hv", "planner_action_rank_by_final_hv",
                 "mean_escape_gen_by_action_kind", "final_hv_snr",
                 "stagnation_by_action_kind", "exploration_duration_by_action_kind",
                 "hv_reference", "per_state": [...]}}

``hv_trace``/``diversity_trace``/``population_std_trace`` are replicate means;
``final_hv`` keeps the per-replicate values so the between-action SNR can be
recomputed downstream (``--store-per-rep`` stores the full per-replicate
traces).

``--report`` mode reads such a file and answers the four candidate causes of
``docs/PHASE2_75D_PLAN.md`` (wrong advantage target / insufficient intervention
data / inadequate action space / horizon mismatch) one by one, each with the
numbers behind the verdict, and says ``indistinguishable`` where the data
cannot separate two causes instead of guessing.

Cost (measured on the Phase-2.75 host, pop 100, ~0.47 s per NSGA-II
generation incl. the per-generation metrics): one state costs
``(2 + n_alternatives) * n_reps * generations`` generations, so the default
``--max-states 30 --generations 20 --n-alternatives 10 --n-reps 3`` grid costs
30 * 12 * 3 * 20 = 21,600 generations ~ 2.8 h per problem.

Example:
    ``python experiments/diagnose_hard_landscape.py --problem zdt4``
    ``python experiments/diagnose_hard_landscape.py --report``
"""

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    # Allow ``python experiments/diagnose_hard_landscape.py`` from the repo
    # root: the script directory (not the repo root) is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.nsga2 import NSGAII, OperatorConfig
from benchmarks import get_problem
from experiments.counterfactual_actions import (
    _build_candidate_actions,
    _load_controller_and_encoder,
    _select_snapshot_files,
    _snapshot_hash_seed,
)
from metrics.indicators import diversity_spread, hypervolume, igd

#: Output of the diagnosis and of ``--report``.
DEFAULT_OUT = "results/phase2_75d/hard_landscape_diagnosis.json"
DEFAULT_REPORT_OUT = "results/phase2_75d/hard_landscape_report.json"
#: Snapshot corpus.
DEFAULT_SNAPSHOTS_DIR = "results/phase1_75/snapshots"
#: Controller artefacts of the index-0 candidate.
DEFAULT_CONTROLLER = "results/phase2_outcome/planning_controller.json"
DEFAULT_PREDICTOR = "results/phase2_outcome/predictor.pt"
DEFAULT_ENCODER = "results/phase2_outcome/encoder.json"
#: Diagnosis grid.
DEFAULT_GENERATIONS = 20
DEFAULT_MAX_STATES = 30
DEFAULT_N_ALTERNATIVES = 10
DEFAULT_N_REPS = 3
#: Absolute HV gain below which a generation counts as stagnating.
DEFAULT_STAGNATION_EPSILON = 1e-4
#: Escape thresholds as fractions of the problem's reference hypervolume.
ESCAPE_THRESHOLDS: tuple[float, float] = (0.10, 0.50)
#: Population spread below this fraction of its first branched generation is
#: considered to have stopped exploring.
DEFAULT_EXPLORATION_STD_FRACTION = 0.5
#: Deployment pm bounds (Phase-1.5 contract).
DEFAULT_PM_MULT_RANGE: tuple[float, float] = (0.25, 8.0)
#: Default hypervolume reference point.
DEFAULT_REF_POINT: tuple[float, float] = (1.1, 1.1)
DEFAULT_N_REFERENCE_POINTS = 200
#: Verdict vocabulary of ``--report``.
VERDICT_SUPPORTED = "supported"
VERDICT_NOT_SUPPORTED = "not_supported"
VERDICT_INDISTINGUISHABLE = "indistinguishable"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments of the landscape diagnosis."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2.75D Task 5: per-generation dynamics diagnosis of difficult "
            "landscapes (diversity, saturation, escape, exploration duration) "
            "plus a data-backed cause report."
        )
    )
    parser.add_argument("--problem", type=str, default=None,
                        help="Problem to diagnose (sharding unit; required "
                        "unless --report is used).")
    parser.add_argument("--snapshots-dir", type=str, default=DEFAULT_SNAPSHOTS_DIR,
                        help="Snapshot directory (default: %(default)s).")
    parser.add_argument("--max-states", type=int, default=DEFAULT_MAX_STATES,
                        help="Snapshots per problem (default: %(default)s).")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS,
                        help="Generations per branch (default: %(default)s).")
    parser.add_argument("--n-alternatives", type=int, default=DEFAULT_N_ALTERNATIVES,
                        help="Sampled alternatives per state (default: %(default)s).")
    parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS,
                        help="Replicate branches per candidate (default: %(default)s).")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT,
                        help="Diagnosis JSON (default: %(default)s).")
    parser.add_argument("--controller", type=str, default=DEFAULT_CONTROLLER,
                        help="Controller config of the index-0 candidate "
                        "(default: %(default)s).")
    parser.add_argument("--controller-type", choices=["planning", "multihead"],
                        default="planning",
                        help="Controller family of the index-0 candidate "
                        "(default: %(default)s).")
    parser.add_argument("--predictor", type=str, default=DEFAULT_PREDICTOR,
                        help="Outcome predictor of a planning controller "
                        "(default: %(default)s).")
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER,
                        help="Fitted encoder (default: %(default)s).")
    parser.add_argument(
        "--include-default-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Insert the NSGA-II default action as candidate 1, tagged "
        "kind='default' (default: %(default)s; mirrors the Phase-2.75D "
        "evaluate-horizon protocol).",
    )
    parser.add_argument("--stagnation-epsilon", type=float,
                        default=DEFAULT_STAGNATION_EPSILON,
                        help="Absolute HV gain below which a generation counts "
                        "as stagnating (default: %(default)s).")
    parser.add_argument("--exploration-std-fraction", type=float,
                        default=DEFAULT_EXPLORATION_STD_FRACTION,
                        help="Population-spread fraction below which exploration "
                        "is considered finished (default: %(default)s).")
    parser.add_argument("--store-per-rep", action="store_true",
                        help="Also store the per-replicate traces (larger file).")
    parser.add_argument("--pm-mult-range", nargs=2, type=float,
                        default=list(DEFAULT_PM_MULT_RANGE),
                        metavar=("PM_MULT_LO", "PM_MULT_HI"),
                        help="Deployment pm bounds (default: %(default)s).")
    parser.add_argument("--ref-point", nargs=2, type=float,
                        default=list(DEFAULT_REF_POINT),
                        metavar=("REF_F1", "REF_F2"),
                        help="Hypervolume reference point (default: %(default)s).")
    parser.add_argument("--n-reference-points", type=int,
                        default=DEFAULT_N_REFERENCE_POINTS,
                        help="Reference-front samples for IGD (default: %(default)s).")
    # --- report mode --------------------------------------------------------
    parser.add_argument("--report", action="store_true",
                        help="Read the diagnosis JSON from --out and judge the "
                        "four candidate failure causes instead of running.")
    parser.add_argument("--report-out", type=str, default=DEFAULT_REPORT_OUT,
                        help="Report JSON of --report mode (default: %(default)s).")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# pure per-trace statistics (synthetic-data testable)
# ---------------------------------------------------------------------------


def stagnation_and_escape(
    hv_trace: Sequence[float],
    *,
    epsilon: float = DEFAULT_STAGNATION_EPSILON,
    hv_reference: float | None = None,
    thresholds: Sequence[float] = ESCAPE_THRESHOLDS,
) -> dict[str, Any]:
    """Stagnation run lengths and first escape generations of one HV trace.

    Definitions:

    * a generation ``i >= 1`` *stagnates* when ``hv_trace[i] - hv_trace[i-1] <
      epsilon`` (generation 0 is the branch start and never counts);
    * ``stagnation_gens`` is the **longest consecutive run** of stagnating
      generations, ``stagnation_gens_total`` their total count;
    * ``escape_gen_Xpct`` is the smallest index whose HV is ``>= X% * hv_reference``
      (``0`` when the snapshot already exceeds it, ``None`` when the horizon
      never reaches it).

    Args:
        hv_trace: Hypervolume per branch generation, index 0 = branch start.
        epsilon: Absolute HV gain below which a generation stagnates.
        hv_reference: Reference hypervolume for the escape thresholds; when
            ``None`` the escape fields are ``None``.
        thresholds: Escape fractions of ``hv_reference`` (default 10% / 50%).

    Returns:
        ``{"stagnation_gens", "stagnation_gens_total", "stagnation_epsilon",
        "escape_gen_10pct", "escape_gen_50pct", "escape_thresholds",
        "final_hv", "hv_gain_total"}``.

    Raises:
        ValueError: If ``hv_trace`` is empty.
    """
    trace = [float(value) for value in hv_trace]
    if not trace:
        raise ValueError("hv_trace must not be empty")
    gains = [trace[index] - trace[index - 1] for index in range(1, len(trace))]
    longest = 0
    current = 0
    total = 0
    for gain in gains:
        if gain < float(epsilon):
            current += 1
            total += 1
            longest = max(longest, current)
        else:
            current = 0
    escape: dict[str, int | None] = {}
    threshold_values: dict[str, float | None] = {}
    for fraction in thresholds:
        key = f"escape_gen_{int(round(fraction * 100))}pct"
        if hv_reference is None:
            escape[key] = None
            threshold_values[key] = None
            continue
        value = float(fraction) * float(hv_reference)
        threshold_values[key] = value
        escape[key] = next(
            (index for index, hv in enumerate(trace) if hv >= value), None
        )
    return {
        "stagnation_gens": int(longest),
        "stagnation_gens_total": int(total),
        "stagnation_epsilon": float(epsilon),
        "escape_gen_10pct": escape.get("escape_gen_10pct"),
        "escape_gen_50pct": escape.get("escape_gen_50pct"),
        "escape_thresholds": threshold_values,
        "final_hv": float(trace[-1]),
        "hv_gain_total": float(trace[-1] - trace[0]),
    }


def exploration_duration(
    std_trace: Sequence[float], *, fraction: float = DEFAULT_EXPLORATION_STD_FRACTION
) -> int:
    """Generations the population kept at least ``fraction`` of its initial spread.

    Args:
        std_trace: Mean per-variable population standard deviation per branch
            generation, index 0 = branch start.
        fraction: Fraction of the first branched generation's spread.

    Returns:
        Number of generations (including the branch start) whose spread is
        ``>= fraction * std_trace[1]`` — the first branched generation is the
        reference because generation 0 is the untouched snapshot. ``0`` when
        the trace is shorter than two entries.

    Raises:
        ValueError: If ``fraction`` is not in ``(0, 1]``.
    """
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction}")
    values = [float(value) for value in std_trace]
    if len(values) < 2:
        return 0
    reference = values[1]
    if reference <= 0.0:
        return 0
    limit = float(fraction) * reference
    return int(sum(1 for value in values if value >= limit))


def action_effect_snr(final_hv_by_candidate: Sequence[Sequence[float]]) -> float | None:
    """Between-action variance over within-action replicate variance.

    Same statistic as the Phase-2B D4 diagnostic, applied to the branch-end HV
    of one state: ``var(per-candidate mean, ddof=1) / mean(per-candidate
    var(ddof=1))``.

    Args:
        final_hv_by_candidate: Per-candidate replicate values.

    Returns:
        The SNR, or ``None`` when it is undefined (fewer than two candidates,
        fewer than two replicates, or zero noise).
    """
    rows = [np.asarray(list(row), dtype=np.float64) for row in final_hv_by_candidate]
    if len(rows) < 2 or any(row.size < 2 for row in rows):
        return None
    means = np.asarray([float(row.mean()) for row in rows])
    noise = float(np.mean([float(row.var(ddof=1)) for row in rows]))
    if noise <= 0.0:
        return None
    return float(means.var(ddof=1) / noise)


def _mean_trace(traces: Sequence[Sequence[float]]) -> list[float]:
    """Element-wise mean of equal-length traces."""
    arrays = [np.asarray(list(trace), dtype=np.float64) for trace in traces]
    if not arrays:
        return []
    stacked = np.vstack(arrays)
    return [float(value) for value in stacked.mean(axis=0)]


def _median_or_none(values: Sequence[float | None]) -> float | None:
    """Median of the defined values, or ``None``."""
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None
    return float(np.median(defined))


def _fraction_defined(values: Sequence[float | None]) -> float | None:
    """Fraction of entries that are not ``None``."""
    if not values:
        return None
    return float(sum(1 for value in values if value is not None) / len(values))


# ---------------------------------------------------------------------------
# diagnosis
# ---------------------------------------------------------------------------


def _population_statistics(
    population_x: np.ndarray, previous_x: np.ndarray | None
) -> dict[str, float]:
    """Spread and movement of one population snapshot."""
    values = np.asarray(population_x, dtype=np.float64)
    per_variable_std = values.std(axis=0, ddof=0)
    per_variable_span = values.max(axis=0) - values.min(axis=0)
    movement = (
        float(np.abs(values - previous_x).mean())
        if previous_x is not None
        else 0.0
    )
    return {
        "population_std_mean": float(per_variable_std.mean()),
        "population_span_mean": float(per_variable_span.mean()),
        "population_abs_change_mean": movement,
    }


def diagnose_state(
    payload: dict[str, Any],
    *,
    problem: Any,
    reference_front: np.ndarray,
    ref_point: np.ndarray,
    hv_reference: float,
    controller_action: dict[str, Any],
    generations: int,
    n_alternatives: int,
    n_reps: int,
    include_default_action: bool,
    pm_mult_range: tuple[float, float],
    stagnation_epsilon: float,
    exploration_std_fraction: float,
    store_per_rep: bool,
    verbose: bool = True,
) -> dict[str, Any]:
    """Branch every candidate for ``generations`` steps, recording each one.

    Returns:
        Per-candidate records (traces + statistics) for this snapshot.
    """
    problem_name = str(payload["problem"])
    n_vars = int(problem.n_vars)
    snapshot = payload["state"]
    hash_seed = _snapshot_hash_seed(
        problem_name, int(payload["seed"]), int(payload["generation"])
    )
    actions = _build_candidate_actions(
        controller_action=controller_action,
        hash_seed=hash_seed,
        n_alternatives=int(n_alternatives),
        base_pm=1.0 / n_vars,
        pm_mult_range=pm_mult_range,
        include_default_action=bool(include_default_action),
    )
    default_index = 1 if include_default_action else None
    records: list[dict[str, Any]] = []
    for k, action in enumerate(actions):
        per_rep: list[dict[str, Any]] = []
        for rep in range(int(n_reps)):
            algorithm = NSGAII(
                problem,
                pop_size=int(snapshot["config"]["pop_size"]),
                operators=OperatorConfig(),
                seed=0,
            )
            algorithm.restore_state(snapshot)
            algorithm.rng = np.random.Generator(
                np.random.PCG64([hash_seed, k, rep])
            )
            previous_x = algorithm.population_x
            hv_values = [
                float(hypervolume(algorithm.nondominated_front(), ref_point))
            ]
            igd_values = [
                float(igd(algorithm.nondominated_front(), reference_front))
            ]
            diversity_values = [
                float(diversity_spread(algorithm.nondominated_front()))
            ]
            spread = _population_statistics(previous_x, None)
            std_values = [spread["population_std_mean"]]
            span_values = [spread["population_span_mean"]]
            movement_values = [0.0]
            pm_values = [float(action["mutation_probability"])]
            operator_values = [str(action["mutation_operator"])]
            exploration_values = [float(action["exploration_strength"])]
            for _ in range(int(generations)):
                previous_x = algorithm.population_x
                algorithm.step(
                    mutation_prob=action["mutation_probability"],
                    mutation_operator=action["mutation_operator"],
                    exploration_strength=action["exploration_strength"],
                )
                front = algorithm.nondominated_front()
                hv_values.append(float(hypervolume(front, ref_point)))
                igd_values.append(float(igd(front, reference_front)))
                diversity_values.append(float(diversity_spread(front)))
                stats = _population_statistics(algorithm.population_x, previous_x)
                std_values.append(stats["population_std_mean"])
                span_values.append(stats["population_span_mean"])
                movement_values.append(stats["population_abs_change_mean"])
                pm_values.append(float(action["mutation_probability"]))
                operator_values.append(str(action["mutation_operator"]))
                exploration_values.append(float(action["exploration_strength"]))
            per_rep.append(
                {
                    "hv_trace": hv_values,
                    "igd_trace": igd_values,
                    "diversity_trace": diversity_values,
                    "population_std_trace": std_values,
                    "population_span_trace": span_values,
                    "population_abs_change_trace": movement_values,
                    "mutation_probability_trace": pm_values,
                    "mutation_operator_trace": operator_values,
                    "exploration_strength_trace": exploration_values,
                }
            )
        hv_traces = [entry["hv_trace"] for entry in per_rep]
        statistics = [
            stagnation_and_escape(
                trace,
                epsilon=stagnation_epsilon,
                hv_reference=hv_reference,
            )
            for trace in hv_traces
        ]
        exploration = [
            exploration_duration(
                entry["population_std_trace"], fraction=exploration_std_fraction
            )
            for entry in per_rep
        ]
        record: dict[str, Any] = {
            "index": int(k),
            "kind": (
                "controller"
                if k == 0
                else "default"
                if k == default_index
                else "alternative"
            ),
            "action": {
                "mutation_operator": str(action["mutation_operator"]),
                "mutation_probability": float(action["mutation_probability"]),
                "exploration_strength": float(action["exploration_strength"]),
            },
            "hv_trace": _mean_trace(hv_traces),
            "igd_trace": _mean_trace([entry["igd_trace"] for entry in per_rep]),
            "diversity_trace": _mean_trace(
                [entry["diversity_trace"] for entry in per_rep]
            ),
            "population_std_trace": _mean_trace(
                [entry["population_std_trace"] for entry in per_rep]
            ),
            "population_span_trace": _mean_trace(
                [entry["population_span_trace"] for entry in per_rep]
            ),
            "population_abs_change_trace": _mean_trace(
                [entry["population_abs_change_trace"] for entry in per_rep]
            ),
            "mutation_probability_trace": _mean_trace(
                [entry["mutation_probability_trace"] for entry in per_rep]
            ),
            "mutation_operator_trace": list(
                per_rep[0]["mutation_operator_trace"]
            ),
            "exploration_strength_trace": _mean_trace(
                [entry["exploration_strength_trace"] for entry in per_rep]
            ),
            "final_hv": [float(entry["hv_trace"][-1]) for entry in per_rep],
            "final_hv_mean": float(np.mean([e["hv_trace"][-1] for e in per_rep])),
            "final_hv_std": (
                float(np.std([e["hv_trace"][-1] for e in per_rep], ddof=1))
                if len(per_rep) > 1
                else 0.0
            ),
            "stagnation_gens": float(
                np.mean([entry["stagnation_gens"] for entry in statistics])
            ),
            "stagnation_gens_total": float(
                np.mean([entry["stagnation_gens_total"] for entry in statistics])
            ),
            "stagnation_epsilon": float(stagnation_epsilon),
            "escape_gen_10pct": _median_or_none(
                [entry["escape_gen_10pct"] for entry in statistics]
            ),
            "escape_gen_50pct": _median_or_none(
                [entry["escape_gen_50pct"] for entry in statistics]
            ),
            "escape_fraction_10pct": _fraction_defined(
                [entry["escape_gen_10pct"] for entry in statistics]
            ),
            "escape_fraction_50pct": _fraction_defined(
                [entry["escape_gen_50pct"] for entry in statistics]
            ),
            "exploration_duration": float(np.mean(exploration)),
            "exploration_std_fraction": float(exploration_std_fraction),
            "n_reps": int(n_reps),
            "generations": int(generations),
        }
        if store_per_rep:
            record["per_rep"] = per_rep
        records.append(record)
        if verbose:
            print(
                f"[state {problem_name}|seed{payload['seed']}|gen{payload['generation']}] "
                f"cand{k}({record['kind']}): final_hv={record['final_hv_mean']:.5f} "
                f"stagnation={record['stagnation_gens']:.1f} "
                f"escape50={record['escape_gen_50pct']} "
                f"exploration={record['exploration_duration']:.1f}"
            )
    return {
        "state": f"{problem_name}|seed{int(payload['seed'])}|gen{int(payload['generation'])}",
        "seed": int(payload["seed"]),
        "generation": int(payload["generation"]),
        "hv_before": float(hv_traces[0][0]),
        "candidates": records,
    }


def summarize_states(
    states: Sequence[dict[str, Any]], *, hv_reference: float
) -> dict[str, Any]:
    """Aggregate the per-state records into the artifact's summary block.

    Returns:
        Summary with ``best_action_by_final_hv``,
        ``planner_action_rank_by_final_hv``, ``mean_escape_gen_by_action_kind``,
        the per-kind stagnation/exploration means, the branch-end SNR and the
        per-state table.
    """
    n_candidates = len(states[0]["candidates"]) if states else 0
    planner_ranks: list[float] = []
    per_state: list[dict[str, Any]] = []
    best_records: list[dict[str, Any]] = []
    snr_values: list[float] = []
    by_kind: dict[str, dict[str, list[float]]] = {}
    escape_by_kind: dict[str, dict[str, list[float | None]]] = {}
    for state in states:
        candidates = state["candidates"]
        means = [float(entry["final_hv_mean"]) for entry in candidates]
        order = sorted(range(len(means)), key=lambda index: -means[index])
        best_index = int(order[0])
        planner_rank = int(order.index(0)) + 1  # 1 = best
        planner_ranks.append(
            1.0 - (planner_rank - 1) / max(len(candidates) - 1, 1)
        )
        snr = action_effect_snr([entry["final_hv"] for entry in candidates])
        if snr is not None:
            snr_values.append(snr)
        for entry in candidates:
            kind = str(entry["kind"])
            bucket = by_kind.setdefault(
                kind,
                {"stagnation_gens": [], "exploration_duration": [], "final_hv": []},
            )
            bucket["stagnation_gens"].append(float(entry["stagnation_gens"]))
            bucket["exploration_duration"].append(
                float(entry["exploration_duration"])
            )
            bucket["final_hv"].append(float(entry["final_hv_mean"]))
            escape_bucket = escape_by_kind.setdefault(
                kind,
                {"escape_gen_10pct": [], "escape_gen_50pct": []},
            )
            escape_bucket["escape_gen_10pct"].append(entry["escape_gen_10pct"])
            escape_bucket["escape_gen_50pct"].append(entry["escape_gen_50pct"])
        per_state.append(
            {
                "state": state["state"],
                "hv_before": state["hv_before"],
                "best_candidate_index": best_index,
                "best_kind": str(candidates[best_index]["kind"]),
                "best_final_hv": means[best_index],
                "planner_rank": planner_rank,
                "planner_final_hv": means[0],
                "planner_rank_percentile": 1.0
                - (planner_rank - 1) / max(len(candidates) - 1, 1),
                "best_minus_planner": means[best_index] - means[0],
                "mean_final_hv": float(np.mean(means)),
                "final_hv_snr": snr,
                "n_candidates": len(candidates),
            }
        )
        best_records.append(
            {
                "state": state["state"],
                "candidate_index": best_index,
                "kind": str(candidates[best_index]["kind"]),
                "final_hv_mean": means[best_index],
                "planner_is_best": bool(best_index == 0),
            }
        )
    escape_summary = {
        kind: {
            "n_candidates": len(bucket["escape_gen_50pct"]),
            "mean_escape_gen_10pct": _median_or_none(bucket["escape_gen_10pct"]),
            "mean_escape_gen_50pct": _median_or_none(bucket["escape_gen_50pct"]),
            "fraction_escaping_10pct": _fraction_defined(bucket["escape_gen_10pct"]),
            "fraction_escaping_50pct": _fraction_defined(bucket["escape_gen_50pct"]),
        }
        for kind, bucket in sorted(escape_by_kind.items())
    }
    kind_summary = {
        kind: {
            "n": len(bucket["final_hv"]),
            "mean_stagnation_gens": float(np.mean(bucket["stagnation_gens"])),
            "mean_exploration_duration": float(
                np.mean(bucket["exploration_duration"])
            ),
            "mean_final_hv": float(np.mean(bucket["final_hv"])),
        }
        for kind, bucket in sorted(by_kind.items())
    }
    return {
        "n_states": len(states),
        "n_candidates": int(n_candidates),
        "hv_reference": float(hv_reference),
        "best_action_by_final_hv": {
            "n_states": len(best_records),
            "planner_is_best_fraction": (
                float(np.mean([entry["planner_is_best"] for entry in best_records]))
                if best_records
                else None
            ),
            "kind_counts": {
                kind: sum(1 for entry in best_records if entry["kind"] == kind)
                for kind in sorted({entry["kind"] for entry in best_records})
            },
            "per_state": best_records,
        },
        "planner_action_rank_by_final_hv": {
            "mean_percentile": (
                float(np.mean(planner_ranks)) if planner_ranks else None
            ),
            "median_rank": (
                float(np.median([entry["planner_rank"] for entry in per_state]))
                if per_state
                else None
            ),
            "mean_rank": (
                float(np.mean([entry["planner_rank"] for entry in per_state]))
                if per_state
                else None
            ),
            "n_states": len(per_state),
        },
        "mean_escape_gen_by_action_kind": escape_summary,
        "stagnation_by_action_kind": {
            kind: entry["mean_stagnation_gens"] for kind, entry in kind_summary.items()
        },
        "exploration_duration_by_action_kind": {
            kind: entry["mean_exploration_duration"]
            for kind, entry in kind_summary.items()
        },
        "final_hv_by_action_kind": {
            kind: entry["mean_final_hv"] for kind, entry in kind_summary.items()
        },
        "final_hv_snr": {
            "mean": float(np.mean(snr_values)) if snr_values else None,
            "median": float(np.median(snr_values)) if snr_values else None,
            "n_states": len(snr_values),
        },
        "per_state": per_state,
    }


def run_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    """Run the per-generation diagnosis for one problem.

    Returns:
        The payload written to ``--out``.

    Raises:
        ValueError: If ``--problem`` is missing or the grid is invalid.
        FileNotFoundError: If a controller/encoder artefact is missing.
    """
    if not args.problem:
        raise ValueError("--problem is required unless --report is used")
    if int(args.generations) < 1:
        raise ValueError(f"generations must be >= 1, got {args.generations}")
    if int(args.n_alternatives) < 1 or int(args.n_reps) < 1:
        raise ValueError("n_alternatives and n_reps must be >= 1")
    started = time.perf_counter()
    problem = get_problem(args.problem)
    controller, encoder, _predictor = _load_controller_and_encoder(args)
    pm_mult_range = (float(args.pm_mult_range[0]), float(args.pm_mult_range[1]))
    ref_point = np.asarray(args.ref_point, dtype=float)
    reference_front = problem.reference_front(n_points=args.n_reference_points)
    hv_reference = float(hypervolume(reference_front, ref_point))
    files = _select_snapshot_files(
        Path(args.snapshots_dir), problem.name, int(args.max_states)
    )
    per_state: list[dict[str, Any]] = []
    for position, path in enumerate(files, start=1):
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        controller_action = controller.predict_action(
            list(payload["history"]),
            encoder,
            pm_mult_range[0] / problem.n_vars,
            pm_mult_range[1] / problem.n_vars,
            n_vars=problem.n_vars,
        )
        print(f"[diagnose] state {position}/{len(files)}: {path.name}")
        per_state.append(
            diagnose_state(
                payload,
                problem=problem,
                reference_front=reference_front,
                ref_point=ref_point,
                hv_reference=hv_reference,
                controller_action=controller_action,
                generations=int(args.generations),
                n_alternatives=int(args.n_alternatives),
                n_reps=int(args.n_reps),
                include_default_action=bool(args.include_default_action),
                pm_mult_range=pm_mult_range,
                stagnation_epsilon=float(args.stagnation_epsilon),
                exploration_std_fraction=float(args.exploration_std_fraction),
                store_per_rep=bool(args.store_per_rep),
                verbose=False,
            )
        )
    summary = summarize_states(per_state, hv_reference=hv_reference)
    payload = {
        "problem": problem.name,
        "config": {
            "problem": problem.name,
            "snapshots_dir": str(args.snapshots_dir),
            "max_states": int(args.max_states),
            "n_states_evaluated": len(per_state),
            "generations": int(args.generations),
            "n_alternatives": int(args.n_alternatives),
            "n_reps": int(args.n_reps),
            "include_default_action": bool(args.include_default_action),
            "controller": str(args.controller),
            "controller_type": str(args.controller_type),
            "predictor": str(args.predictor) if args.predictor else None,
            "encoder": str(args.encoder),
            "pm_mult_range": [float(pm_mult_range[0]), float(pm_mult_range[1])],
            "ref_point": [float(ref_point[0]), float(ref_point[1])],
            "n_reference_points": int(args.n_reference_points),
            "stagnation_epsilon": float(args.stagnation_epsilon),
            "exploration_std_fraction": float(args.exploration_std_fraction),
            "escape_thresholds": [float(value) for value in ESCAPE_THRESHOLDS],
            "store_per_rep": bool(args.store_per_rep),
            "trace_length": int(args.generations) + 1,
            "protocol": (
                "per-generation branch records: index 0 is the snapshot, then one "
                "entry per branched generation; HV/IGD/diversity of the "
                "nondominated front plus the population spread and movement"
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "per_candidate": [candidate for state in per_state for candidate in state["candidates"]],
        "per_state": per_state,
        "summary": summary,
        "wall_time_sec": float(time.perf_counter() - started),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file():
        try:
            with out_path.open("r", encoding="utf-8") as fh:
                existing = json.load(fh)
            other = str(existing.get("problem", ""))
            if other and other != problem.name:
                print(
                    f"[warn] {out_path} currently holds problem {other!r}; "
                    f"shard the diagnosis by giving each problem its own --out"
                )
        except (json.JSONDecodeError, OSError):
            pass
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(
        f"[done] {len(files)} states, {summary['n_candidates']} candidates, "
        f"{args.generations} generations -> {out_path} "
        f"({time.perf_counter() - started:.1f}s)"
    )
    return payload


# ---------------------------------------------------------------------------
# report mode
# ---------------------------------------------------------------------------


def _verdict(cause: str, verdict: str, evidence: str, numbers: dict[str, Any]) -> dict[str, Any]:
    """One cause/verdict/evidence record of the report."""
    return {
        "cause": cause,
        "verdict": verdict,
        "evidence": evidence,
        "numbers": numbers,
    }


def assess_causes(payload: dict[str, Any]) -> dict[str, Any]:
    """Judge the four candidate failure causes from one diagnosis payload.

    The rules are fixed and every verdict carries the numbers behind it:

    * **wrong_target** — supported when the planner's action ranks below the
      candidate median by branch-end HV; not supported when it ranks at or
      above the 60th percentile; otherwise indistinguishable. Because the
      branch horizon is finite, the question "is a 20-generation advantage
      decoupled from *true* convergence?" is only answerable when the branch
      is long (>= 50 generations) — below that the verdict records that limit
      explicitly.
    * **insufficient_data** — driven by the branch-end between-action SNR and
      the state count: SNR < 1 or fewer than 10 states means the effect cannot
      be separated from noise; SNR >= 3 with enough states means it can.
    * **inadequate_action_space** — supported when *no* candidate escapes
      50% of the reference hypervolume in most states; not supported when some
      candidates do escape (then the space is adequate and the failure is in
      selection); otherwise indistinguishable.
    * **horizon_mismatch** — supported when the median escape generation is
      beyond ~80% of the branch horizon or a large share of candidates never
      escape within it; not supported when escapes cluster early.

    Args:
        payload: Diagnosis payload as written by :func:`run_diagnosis`.

    Returns:
        ``{"problem", "config", "verdicts": [...], "caveats": [...]}``.
    """
    config = payload.get("config", {})
    summary = payload.get("summary", {})
    states = payload.get("per_state", [])
    generations = int(config.get("generations", 0))
    n_states = int(summary.get("n_states", 0))
    rank = summary.get("planner_action_rank_by_final_hv", {}) or {}
    mean_percentile = rank.get("mean_percentile")
    snr = (summary.get("final_hv_snr") or {}).get("mean")
    escape = summary.get("mean_escape_gen_by_action_kind", {}) or {}
    all_escape_50 = [
        entry.get("mean_escape_gen_50pct")
        for entry in escape.values()
    ]
    escape_fractions = [
        entry.get("fraction_escaping_50pct")
        for entry in escape.values()
        if entry.get("fraction_escaping_50pct") is not None
    ]
    any_escape = any(value is not None for value in all_escape_50)
    best_escape_fraction = max(escape_fractions) if escape_fractions else 0.0
    per_state_gap = [
        float(entry.get("best_minus_planner", 0.0)) for entry in states
    ]
    verdicts: list[dict[str, Any]] = []

    # --- 1. wrong advantage target -----------------------------------------
    numbers = {
        "planner_mean_percentile_by_final_hv": mean_percentile,
        "planner_median_rank": rank.get("median_rank"),
        "n_candidates": summary.get("n_candidates"),
        "generations": generations,
        "best_minus_planner_mean": (
            float(np.mean(per_state_gap)) if per_state_gap else None
        ),
    }
    if mean_percentile is None:
        verdicts.append(
            _verdict("wrong_target", VERDICT_INDISTINGUISHABLE,
                     "no state produced a usable planner rank", numbers)
        )
    elif mean_percentile < 0.5:
        verdicts.append(
            _verdict(
                "wrong_target", VERDICT_SUPPORTED,
                f"the planner's action ranks at the {mean_percentile:.3f} "
                f"percentile of the candidate field by branch-end HV, i.e. "
                f"below the median its own objective should improve",
                numbers,
            )
        )
    elif mean_percentile >= 0.6:
        verdicts.append(
            _verdict(
                "wrong_target", VERDICT_NOT_SUPPORTED,
                f"the planner's action still ranks at the {mean_percentile:.3f} "
                f"percentile by branch-end HV, so its target is not obviously "
                f"misaligned with the branch outcome",
                numbers,
            )
        )
    else:
        verdicts.append(
            _verdict(
                "wrong_target", VERDICT_INDISTINGUISHABLE,
                f"the planner's mean percentile ({mean_percentile:.3f}) is too "
                f"close to the median to separate a misaligned target from noise",
                numbers,
            )
        )
    if generations < 50:
        verdicts.append(
            _verdict(
                "wrong_target_long_horizon_decoupling",
                VERDICT_INDISTINGUISHABLE,
                f"the branch is only {generations} generations long, so the "
                f"question 'is a short-window advantage decoupled from true "
                f"convergence?' cannot be answered from this data; re-run with "
                f"--generations >= 50 to make it observable",
                numbers,
            )
        )

    # --- 2. insufficient intervention data ---------------------------------
    numbers = {
        "n_states": n_states,
        "n_reps": config.get("n_reps"),
        "n_candidates": summary.get("n_candidates"),
        "final_hv_snr_mean": snr,
    }
    if snr is None:
        verdicts.append(
            _verdict("insufficient_data", VERDICT_INDISTINGUISHABLE,
                     "the branch-end SNR is undefined (no replicate noise)", numbers)
        )
    elif snr < 1.0 or n_states < 10:
        verdicts.append(
            _verdict(
                "insufficient_data", VERDICT_SUPPORTED,
                f"branch-end action SNR {snr:.2f} and {n_states} states: the "
                f"action effect is at or below replicate noise, so no amount of "
                f"modelling can rank these candidates reliably",
                numbers,
            )
        )
    elif snr >= 3.0:
        verdicts.append(
            _verdict(
                "insufficient_data", VERDICT_NOT_SUPPORTED,
                f"branch-end SNR {snr:.2f} over {n_states} states is well above "
                f"noise, so the signal is measurable with the current data",
                numbers,
            )
        )
    else:
        verdicts.append(
            _verdict(
                "insufficient_data", VERDICT_INDISTINGUISHABLE,
                f"branch-end SNR {snr:.2f} sits between noise and a clear "
                f"signal; more replicates or states would be needed to decide",
                numbers,
            )
        )

    # --- 3. inadequate action space ----------------------------------------
    numbers = {
        "escape_fraction_50pct_by_kind": escape_fractions,
        "any_candidate_escapes_50pct": any_escape,
        "best_kind_escape_fraction": best_escape_fraction,
    }
    if not any_escape:
        verdicts.append(
            _verdict(
                "inadequate_action_space", VERDICT_INDISTINGUISHABLE,
                "no candidate reaches 50% of the reference hypervolume inside "
                "the branch, so 'the action space cannot escape' and 'the branch "
                "is too short to see the escape' are not separable here",
                numbers,
            )
        )
    elif best_escape_fraction < 0.5:
        verdicts.append(
            _verdict(
                "inadequate_action_space", VERDICT_SUPPORTED,
                f"only {best_escape_fraction:.2f} of candidate branches escape "
                f"within the horizon; the sampled action space rarely contains "
                f"an escaping action",
                numbers,
            )
        )
    else:
        verdicts.append(
            _verdict(
                "inadequate_action_space", VERDICT_NOT_SUPPORTED,
                f"at least one action kind escapes in "
                f"{best_escape_fraction:.2f} of its branches, so escaping "
                f"actions exist in the space; the failure is in selecting them",
                numbers,
            )
        )

    # --- 4. horizon mismatch -----------------------------------------------
    escape_gens = [value for value in all_escape_50 if value is not None]
    median_escape = float(np.median(escape_gens)) if escape_gens else None
    numbers = {
        "generations": generations,
        "median_escape_gen_50pct": median_escape,
        "escape_gen_by_kind": {kind: entry.get("mean_escape_gen_50pct")
                               for kind, entry in escape.items()},
        "never_escaping_kinds": [
            kind for kind, entry in escape.items()
            if entry.get("mean_escape_gen_50pct") is None
        ],
    }
    if median_escape is None:
        verdicts.append(
            _verdict(
                "horizon_mismatch", VERDICT_SUPPORTED,
                f"no candidate kind reaches 50% of the reference hypervolume "
                f"within {generations} generations, which is the signature of a "
                f"branch that is too short to observe convergence",
                numbers,
            )
        )
    elif generations > 0 and median_escape > 0.8 * generations:
        verdicts.append(
            _verdict(
                "horizon_mismatch", VERDICT_SUPPORTED,
                f"the median escape happens at generation {median_escape:.0f} of "
                f"{generations}, i.e. at the very end of the branch",
                numbers,
            )
        )
    elif generations > 0 and median_escape <= 0.5 * generations:
        verdicts.append(
            _verdict(
                "horizon_mismatch", VERDICT_NOT_SUPPORTED,
                f"the median escape happens at generation {median_escape:.0f} of "
                f"{generations}, comfortably inside the branch",
                numbers,
            )
        )
    else:
        verdicts.append(
            _verdict(
                "horizon_mismatch", VERDICT_INDISTINGUISHABLE,
                f"the median escape generation ({median_escape:.0f}) lies in the "
                f"late middle of the {generations}-generation branch; a longer "
                f"branch would be needed to see whether convergence completes",
                numbers,
            )
        )

    caveats = [
        "'final HV' in this report means the HV at the end of the branch "
        f"({generations} generations), not the HV of a fully converged run",
        "escape thresholds are fractions of the reference-front hypervolume, so "
        "they measure progress toward the known Pareto front, not toward the "
        "best achievable run",
        "the diagnosis branches the same candidate actions as "
        "evaluate-horizon; it cannot see actions outside the sampled space",
    ]
    return {
        "problem": payload.get("problem"),
        "config": config,
        "verdicts": verdicts,
        "caveats": caveats,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def print_report(report: dict[str, Any]) -> None:
    """Print the report as a compact text table."""
    print(f"=== hard-landscape report: {report.get('problem')} ===")
    for entry in report["verdicts"]:
        print(f"\n[{entry['cause']}] {entry['verdict'].upper()}")
        print(f"  {entry['evidence']}")
        print(f"  numbers: {json.dumps(entry['numbers'], sort_keys=True)}")
    print("\ncaveats:")
    for caveat in report["caveats"]:
        print(f"  - {caveat}")


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    """Run ``--report`` mode on an existing diagnosis file.

    Returns:
        The report payload written to ``--report-out``.

    Raises:
        FileNotFoundError: If ``--out`` does not exist.
    """
    path = Path(args.out)
    if not path.is_file():
        raise FileNotFoundError(
            f"diagnosis file not found: {path}; run the diagnosis first"
        )
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    report = assess_causes(payload)
    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print_report(report)
    print(f"\n[done] wrote {out_path}")
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point: diagnose or report."""
    args = parse_args(argv)
    if args.report:
        return run_report(args)
    return run_diagnosis(args)


if __name__ == "__main__":
    main()
