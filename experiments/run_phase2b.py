from __future__ import annotations

"""Phase 2 Experiment B: paired evaluation of the planning controller.

This runner implements the evaluation half of
``docs/PHASE2_EXPERIMENT_B_PLAN.md`` (B1, candidate action planning). The
:class:`controller.planning_controller.PlanningController` — argmax over
predicted long-horizon HV of sampled candidate actions, scored by the
Experiment-A :class:`controller.outcome_predictor.OutcomePredictor` — is
deployed closed-loop on the Phase-1.75 held-out ``(problem, seed)`` grid
and compared against every Phase-1.75 arm stored in
``results/phase1_75/results.json``.

Deployment semantics (identical to Phase 1.75, so runs are directly
comparable): generation 0 is recorded with the algorithm defaults; the
action for the step *into* generation ``t >= 1`` is chosen from the merged
state+reward dicts of all generations recorded so far (``[0, t)``); the
recorder stores the action actually used (``NSGAII.current_action()``
after the step). Mutation-probability bounds are the multiplier range
``[0.25, 8.0]`` mapped to absolute ``[0.25 / n_vars, 8.0 / n_vars]``.
Failure thresholds are reused from the Phase-1.75 results (10th
percentile of fixed-policy training final HV — the training distribution
only, derived before any held-out evaluation).

Stages (``--stage``):

* ``eval`` — run the (problem, seed) grid with the planning controller
  and write one JSON per run to
  ``{out_dir}/runs/planning_predictor__{problem}__seed{seed}.json``.
* ``aggregate`` — merge ``runs/*.json`` into ``{out_dir}/results.json``
  and write the paired comparison against the Phase-1.75 arms to
  ``{out_dir}/comparison.json`` (mean/std of final HV and AUC-HV plus a
  paired Wilcoxon signed-rank test, ``zero_method='zsplit'``,
  ``alternative='greater'`` = planning outperforms the baseline arm).
* ``all`` — eval -> aggregate.

Example:
    ``python experiments/run_phase2b.py --stage all``
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

if __package__ in (None, ""):
    # Allow ``python experiments/run_phase2b.py`` from the repo root: the
    # script directory (not the repo root) is on sys.path in that mode.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms import NSGAII, OperatorConfig
from benchmarks import get_problem
from controller.dataset import merge_state_reward
from controller.outcome_predictor import OutcomePredictor
from controller.planning_controller import PlanningController
from controller.state_encoder import StateEncoder
from experiments.analyze_phase1_75 import holm_correction, paired_bootstrap_ci
from experiments.run_phase1 import _auc_hv
from experiments.run_phase1_75 import _sanitize_action
from trajectory import EvolutionRecorder

#: Arm identifier of the planning controller (used in run keys/files).
ARM_PLANNING = "planning_predictor"

#: Bootstrap resamples for the paired median-difference CI (deterministic seed 0).
BOOTSTRAP_RESAMPLES = 10_000
#: Default evaluation problem grid (all Phase-0 ZDT benchmarks).
DEFAULT_PROBLEMS: tuple[str, ...] = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")
#: Held-out evaluation seeds; identical to Phase 1.75 for paired statistics.
DEFAULT_EVAL_SEEDS: tuple[int, ...] = tuple(range(1000, 1020))
#: Default Experiment-A artifact directory (trained outcome predictor).
DEFAULT_PREDICTOR_DIR = "results/phase2_outcome"
#: Default Phase-1.75 aggregate results used for comparison arms and
#: failure thresholds.
DEFAULT_PHASE1_75_RESULTS = "results/phase1_75/results.json"
#: Default output directory for Phase-2B results.
DEFAULT_OUT_DIR = "results/phase2b"
#: Deployment pm bounds as multipliers on the natural ``1 / n_vars`` scale.
DEFAULT_PM_MULT_RANGE: tuple[float, float] = (0.25, 8.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Phase-2B experiment.

    Args:
        argv: Argument list to parse; ``None`` reads ``sys.argv``.

    Returns:
        Parsed namespace with stage, predictor, evaluation, and output
        settings.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Phase-2 Experiment B: evaluate the predictor-based planning "
            "controller on the Phase-1.75 held-out (problem, seed) grid and "
            "compare it against every Phase-1.75 arm with paired statistics."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["eval", "aggregate", "all"],
        default="all",
        help="Pipeline stage to run; 'all' = eval -> aggregate (default: %(default)s).",
    )
    parser.add_argument(
        "--predictor-dir",
        type=str,
        default=DEFAULT_PREDICTOR_DIR,
        help=(
            "Directory with the trained outcome predictor artifacts "
            "(predictor.pt, encoder.json, training_meta.json). "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--phase1-75-results",
        type=str,
        default=DEFAULT_PHASE1_75_RESULTS,
        help=(
            "Phase-1.75 aggregate results.json (comparison arms and failure "
            "thresholds). Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=list(DEFAULT_PROBLEMS),
        help="Problem subset to evaluate (sharding; default: %(default)s).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_EVAL_SEEDS),
        help="Held-out evaluation seed subset (sharding; default: 1000..1019).",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=100,
        help="NSGA-II population size for evaluation runs (default: %(default)s).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="NSGA-II generations per evaluation run (default: %(default)s).",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=16,
        help="Candidate actions sampled and scored per generation (default: %(default)s).",
    )
    parser.add_argument(
        "--candidate-seed",
        type=int,
        default=0,
        help="Seed of the planner's candidate generator (default: %(default)s).",
    )
    parser.add_argument(
        "--horizon-weights",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Score weight per predictor horizon (one entry per horizon; "
            "default: the planner's long-horizon default)."
        ),
    )
    parser.add_argument(
        "--n-reference-points",
        type=int,
        default=200,
        help="Points sampled from the true Pareto front for IGD (default: %(default)s).",
    )
    parser.add_argument(
        "--ref-point",
        nargs=2,
        type=float,
        default=[1.1, 1.1],
        metavar=("REF_F1", "REF_F2"),
        help="Hypervolume reference point (default: %(default)s).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_phase1_75_results(path: str | Path) -> dict[str, Any]:
    """Load the Phase-1.75 aggregate results used for comparison.

    Args:
        path: Path to the Phase-1.75 ``results.json``.

    Returns:
        The parsed payload.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the payload has no ``"runs"`` mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase-1.75 results not found: {path}; run "
            "experiments/run_phase1_75.py --stage all first"
        )
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload.get("runs"), dict):
        raise ValueError(f"{path} has no 'runs' mapping")
    return payload


def _failure_thresholds(phase1_75_payload: dict[str, Any]) -> dict[str, float]:
    """Extract per-problem failure thresholds from a Phase-1.75 payload.

    Accepts both the full ``failure_thresholds.json`` payload (thresholds
    nested under ``"thresholds"``) and a flat ``problem -> threshold``
    mapping.

    Args:
        phase1_75_payload: Parsed Phase-1.75 ``results.json``.

    Returns:
        Dict mapping problem name to its final-HV failure threshold.
    """
    raw = phase1_75_payload.get("failure_thresholds", {})
    if isinstance(raw, dict) and isinstance(raw.get("thresholds"), dict):
        raw = raw["thresholds"]
    thresholds: dict[str, float] = {}
    if isinstance(raw, dict):
        for problem, value in raw.items():
            try:
                thresholds[str(problem)] = float(value)
            except (TypeError, ValueError):
                continue
    return thresholds


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    """Mean and sample standard deviation (ddof=1; 0.0 for n < 2)."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std}


def _wilcoxon_greater(
    planning_values: Sequence[float], arm_values: Sequence[float]
) -> dict[str, float | None]:
    """Paired Wilcoxon signed-rank test, planning arm greater than baseline.

    Uses ``zero_method='zsplit'`` (zero differences counted, split between
    signs) and ``alternative='greater'``. The test is undefined for fewer
    than two pairs or all-zero differences; ``None`` fields are returned in
    those (and any degenerate scipy) cases instead of raising.

    Args:
        planning_values: Metric values of the planning controller.
        arm_values: Paired metric values of the comparison arm (same
            ``(problem, seed)`` order as ``planning_values``).

    Returns:
        ``{"statistic": ..., "p_value": ...}`` with float or ``None``
        values.
    """
    x = np.asarray(list(planning_values), dtype=float)
    y = np.asarray(list(arm_values), dtype=float)
    undefined: dict[str, float | None] = {"statistic": None, "p_value": None}
    if x.size < 2 or x.size != y.size or np.all(x == y):
        return undefined
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = stats.wilcoxon(
                x, y, zero_method="zsplit", alternative="greater"
            )
        except ValueError:
            return undefined
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        return undefined
    return {"statistic": statistic, "p_value": p_value}


# ---------------------------------------------------------------------------
# Eval stage
# ---------------------------------------------------------------------------


def _build_run_config(
    *,
    problem_name: str,
    n_vars: int,
    seed: int,
    pop_size: int,
    generations: int,
    controller: PlanningController,
    pm_mult_range: tuple[float, float],
    failure_threshold: float | None,
    ref_point: np.ndarray,
    n_reference_points: int,
    predictor_dir: str | Path,
) -> dict[str, Any]:
    """Configuration stored inside each per-run JSON (fully determines the run)."""
    return {
        "phase": "2B",
        "problem": problem_name,
        "n_vars": int(n_vars),
        "algorithm": "nsga2",
        "arm": ARM_PLANNING,
        "seed": int(seed),
        "pop_size": int(pop_size),
        "generations": int(generations),
        "action_space": "full",
        "controller": {
            "name": ARM_PLANNING,
            "n_candidates": int(controller.n_candidates),
            "candidate_seed": int(controller.candidate_seed),
            "horizon_weights": [float(w) for w in controller.horizon_weights],
            "pm_mult_range": [
                float(controller.pm_mult_range[0]),
                float(controller.pm_mult_range[1]),
            ],
            "predictor_dir": str(predictor_dir),
        },
        "pm_mult_range": [float(pm_mult_range[0]), float(pm_mult_range[1])],
        "pm_absolute_range": [
            float(pm_mult_range[0]) / n_vars,
            float(pm_mult_range[1]) / n_vars,
        ],
        "failure_threshold": (
            None if failure_threshold is None else float(failure_threshold)
        ),
        "protocol": (
            "paired closed-loop evaluation identical to Phase 1.75; the "
            "action for the step into generation t is selected by the "
            "planning controller from the merged history of generations "
            "[0, t); the recorded action is the actual action used "
            "(current_action() after the step)"
        ),
        "ref_point": [float(ref_point[0]), float(ref_point[1])],
        "n_reference_points": int(n_reference_points),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_run(
    problem_name: str,
    seed: int,
    *,
    generations: int,
    pop_size: int,
    controller: PlanningController,
    encoder: StateEncoder,
    n_reference_points: int = 200,
    ref_point: np.ndarray | None = None,
    pm_mult_range: tuple[float, float] = DEFAULT_PM_MULT_RANGE,
    failure_threshold: float | None = None,
    runs_dir: str | Path | None = None,
    predictor_dir: str | Path = "",
) -> dict[str, Any]:
    """Run one planning-controller evaluation on one (problem, seed) pair.

    Generation 0 uses the algorithm defaults; for every step into
    generation ``t >= 1`` the merged recorded history so far is encoded
    and the controller's argmax-predicted candidate action is applied.

    Args:
        problem_name: Benchmark identifier accepted by ``get_problem``.
        seed: Evaluation random seed.
        generations: Number of NSGA-II generations to execute.
        pop_size: Population size.
        controller: The planning controller (predictor already loaded).
        encoder: Fitted state encoder matching the predictor's training
            encoder.
        n_reference_points: Points sampled from the true Pareto front for
            IGD.
        ref_point: Hypervolume reference point; defaults to ``(1.1, 1.1)``.
        pm_mult_range: Deployment multiplier bounds on ``1 / n_vars``.
        failure_threshold: Per-problem failure threshold on final HV;
            ``None`` marks the run not-failed (thresholds unavailable).
        runs_dir: If given, the per-run JSON is written to
            ``{runs_dir}/planning_predictor__{problem}__seed{seed}.json``.
        predictor_dir: Recorded in the run config for provenance.

    Returns:
        Per-run metrics dict with ``arm``, ``problem``, ``seed``,
        ``final_hv``, ``final_igd``, ``auc_hv``, ``runtime_sec``,
        ``failed``, ``trajectory_file``.
    """
    if ref_point is None:
        ref_point = np.asarray([1.1, 1.1], dtype=float)
    problem = get_problem(problem_name)
    n_vars = int(problem.n_vars)
    algorithm = NSGAII(problem, pop_size=pop_size, operators=OperatorConfig(), seed=seed)
    recorder = EvolutionRecorder(
        problem_name=problem.name,
        reference_front=problem.reference_front(n_points=n_reference_points),
        ref_point=ref_point,
    )
    pm_min = float(pm_mult_range[0]) / n_vars
    pm_max = float(pm_mult_range[1]) / n_vars

    start = time.perf_counter()
    algorithm.initialize()
    recorder.record(
        algorithm.generation, algorithm.nondominated_front(), algorithm.current_action()
    )
    for _ in range(generations):
        history = [merge_state_reward(tr) for tr in recorder.transitions()]
        action = controller.predict_action(
            history, encoder, pm_min, pm_max, n_vars=n_vars
        )
        action = _sanitize_action(
            action, n_vars=n_vars, pm_mult_range=pm_mult_range
        )
        algorithm.step(
            mutation_prob=action["mutation_probability"],
            mutation_operator=action["mutation_operator"],
            exploration_strength=action["exploration_strength"],
        )
        recorder.record(
            algorithm.generation,
            algorithm.nondominated_front(),
            algorithm.current_action(),
        )
    runtime_sec = time.perf_counter() - start

    transitions = recorder.transitions()
    final_hv = float(transitions[-1]["state"]["hv"])
    failed = (
        bool(final_hv < float(failure_threshold))
        if failure_threshold is not None
        else False
    )
    metrics: dict[str, Any] = {
        "arm": ARM_PLANNING,
        "problem": problem.name,
        "seed": int(seed),
        "final_hv": final_hv,
        "final_igd": float(transitions[-1]["state"]["igd"]),
        "auc_hv": _auc_hv(transitions, generations),
        "runtime_sec": float(runtime_sec),
        "failed": failed,
        "trajectory_file": None,
    }

    if runs_dir is not None:
        config = _build_run_config(
            problem_name=problem.name,
            n_vars=n_vars,
            seed=seed,
            pop_size=pop_size,
            generations=generations,
            controller=controller,
            pm_mult_range=pm_mult_range,
            failure_threshold=failure_threshold,
            ref_point=ref_point,
            n_reference_points=n_reference_points,
            predictor_dir=predictor_dir,
        )
        out_path = Path(runs_dir) / f"{ARM_PLANNING}__{problem.name}__seed{seed}.json"
        recorder.save(out_path, config=config, seed=seed, runtime_sec=runtime_sec)
        with out_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["metrics"] = {
            key: metrics[key]
            for key in ("final_hv", "final_igd", "auc_hv", "runtime_sec", "failed")
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        metrics["trajectory_file"] = f"runs/{out_path.name}"
    return metrics


def _load_planning_context(
    args: argparse.Namespace,
) -> tuple[PlanningController, StateEncoder]:
    """Load the trained outcome predictor, its encoder, and build the planner.

    Args:
        args: Parsed arguments (``predictor_dir``, ``n_candidates``,
            ``candidate_seed``, ``horizon_weights``).

    Returns:
        The ``(planning_controller, state_encoder)`` pair.

    Raises:
        FileNotFoundError: If ``predictor.pt`` or ``encoder.json`` is
            missing under ``args.predictor_dir``.
    """
    predictor_dir = Path(args.predictor_dir)
    predictor_path = predictor_dir / "predictor.pt"
    encoder_path = predictor_dir / "encoder.json"
    for path in (predictor_path, encoder_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"required predictor artifact missing: {path}; train the "
                "outcome predictor first (experiments/train_outcome_predictor.py)"
            )
    predictor = OutcomePredictor.load(predictor_path)
    encoder = StateEncoder.load(encoder_path)
    controller = PlanningController(
        predictor,
        n_candidates=int(args.n_candidates),
        candidate_seed=int(args.candidate_seed),
        horizon_weights=args.horizon_weights,
    )
    controller.predictor_path = str(predictor_path)
    return controller, encoder


def run_eval_stage(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Evaluate the requested (problem, seed) subset, writing per-run JSONs."""
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    problems = [get_problem(name).name for name in args.problems]
    seeds = [int(s) for s in args.seeds]
    ref_point = np.asarray(args.ref_point, dtype=float)

    phase1_75 = _load_phase1_75_results(args.phase1_75_results)
    thresholds = _failure_thresholds(phase1_75)
    if not thresholds:
        print(
            f"[warn] no failure thresholds in {args.phase1_75_results}; "
            "runs are marked failed=false"
        )

    controller, encoder = _load_planning_context(args)
    metrics: list[dict[str, Any]] = []
    for problem_name in problems:
        for seed in seeds:
            summary = evaluate_run(
                problem_name,
                seed,
                generations=int(args.generations),
                pop_size=int(args.pop_size),
                controller=controller,
                encoder=encoder,
                n_reference_points=int(args.n_reference_points),
                ref_point=ref_point,
                failure_threshold=thresholds.get(problem_name),
                runs_dir=runs_dir,
                predictor_dir=args.predictor_dir,
            )
            metrics.append(summary)
            print(
                f"[eval] {ARM_PLANNING} {problem_name} seed={seed} "
                f"hv={summary['final_hv']:.6f} igd={summary['final_igd']:.6f} "
                f"failed={summary['failed']} ({summary['runtime_sec']:.2f}s)"
            )
    print(
        f"[eval] {len(metrics)} runs in {time.perf_counter() - started:.2f}s "
        f"-> {runs_dir}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Aggregate stage
# ---------------------------------------------------------------------------


def _merge_runs(runs_dir: Path) -> tuple[dict[str, Any], float]:
    """Merge per-run JSONs into the ``results.json`` run mapping.

    Args:
        runs_dir: Directory with ``planning_predictor__*.json`` run files.

    Returns:
        ``(runs, eval_runtime_sum)`` where ``runs`` maps
        ``"planning_predictor|<problem>|<seed>"`` to the metric entry.
    """
    runs: dict[str, Any] = {}
    eval_runtime_sum = 0.0
    if not runs_dir.is_dir():
        return runs, eval_runtime_sum
    for path in sorted(runs_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        config = payload.get("config", {})
        arm = str(config.get("arm", path.name.split("__")[0]))
        problem = str(config.get("problem", ""))
        seed = int(config.get("seed", payload.get("seed", 0)))
        metrics = payload.get("metrics")
        transitions = payload.get("transitions", [])
        if metrics is None and transitions:
            generations = int(config.get("generations", transitions[-1]["generation"]))
            metrics = {
                "final_hv": float(transitions[-1]["state"]["hv"]),
                "final_igd": float(transitions[-1]["state"]["igd"]),
                "auc_hv": _auc_hv(transitions, generations),
                "runtime_sec": float(payload.get("runtime_sec", 0.0)),
                "failed": False,
            }
        if metrics is None:
            print(f"[warn] run file without metrics or transitions: {path}; skipped")
            continue
        eval_runtime_sum += float(metrics["runtime_sec"])
        runs[f"{arm}|{problem}|{seed}"] = {
            "final_hv": float(metrics["final_hv"]),
            "final_igd": float(metrics["final_igd"]),
            "auc_hv": float(metrics["auc_hv"]),
            "runtime_sec": float(metrics["runtime_sec"]),
            "failed": bool(metrics["failed"]),
            "trajectory_file": f"runs/{path.name}",
        }
    return runs, eval_runtime_sum


def _paired_metric_series(
    planning_runs: dict[str, Any],
    arm_runs: dict[str, Any],
    problem: str,
    metric: str,
) -> tuple[list[float], list[float]]:
    """Aligned ``(planning, arm)`` metric values over shared eval seeds."""
    planning: dict[int, float] = {}
    for key, entry in planning_runs.items():
        parts = key.split("|")
        if len(parts) == 3 and parts[1] == problem:
            planning[int(parts[2])] = float(entry[metric])
    arm: dict[int, float] = {}
    for key, entry in arm_runs.items():
        parts = key.split("|")
        if len(parts) == 3 and parts[1] == problem:
            arm[int(parts[2])] = float(entry[metric])
    shared = sorted(set(planning) & set(arm))
    return [planning[s] for s in shared], [arm[s] for s in shared]


def build_comparison(
    planning_runs: dict[str, Any],
    phase1_75_payload: dict[str, Any],
    problems: Sequence[str],
) -> dict[str, Any]:
    """Build the paired comparison of the planning arm vs Phase-1.75 arms.

    For every problem and every Phase-1.75 arm, final HV and AUC-HV are
    paired by eval seed; the planning arm is summarized next to each
    baseline and tested with a paired Wilcoxon signed-rank test
    (``zero_method='zsplit'``, ``alternative='greater'``).

    Args:
        planning_runs: Merged Phase-2B run mapping.
        phase1_75_payload: Parsed Phase-1.75 ``results.json``.
        problems: Problems to include (skipped silently when absent from
            both mappings).

    Returns:
        The comparison payload as written to ``comparison.json``.
    """
    baseline_runs = phase1_75_payload.get("runs", {})
    arm_names = phase1_75_payload.get("arms")
    if not arm_names:
        arm_names = sorted({str(k).split("|")[0] for k in baseline_runs})
    arm_names = [str(a) for a in arm_names if str(a) != ARM_PLANNING]

    metrics = ("final_hv", "auc_hv")
    problem_reports: dict[str, Any] = {}
    # Raw p-values collected per (metric, arm) family so the Holm step-down
    # correction can be applied across the five problems afterwards.
    families: dict[tuple[str, str], dict[str, float]] = {
        (metric, arm): {} for metric in metrics for arm in arm_names
    }
    paired_series: dict[tuple[str, str, str], tuple[list[float], list[float]]] = {}

    for problem in problems:
        planning_hv = [
            float(entry["final_hv"])
            for key, entry in sorted(planning_runs.items())
            if key.split("|")[1] == problem
        ]
        planning_auc = [
            float(entry["auc_hv"])
            for key, entry in sorted(planning_runs.items())
            if key.split("|")[1] == problem
        ]
        n_failed = sum(
            1
            for key, entry in planning_runs.items()
            if key.split("|")[1] == problem and entry.get("failed")
        )
        comparisons: dict[str, Any] = {}
        for arm in arm_names:
            arm_keyed = {
                k: v for k, v in baseline_runs.items() if k.startswith(f"{arm}|")
            }
            paired_hv = _paired_metric_series(
                planning_runs, arm_keyed, problem, "final_hv"
            )
            paired_auc = _paired_metric_series(
                planning_runs, arm_keyed, problem, "auc_hv"
            )
            paired_series[(problem, arm, "final_hv")] = paired_hv
            paired_series[(problem, arm, "auc_hv")] = paired_auc
            arm_hv = [
                float(entry["final_hv"])
                for key, entry in sorted(baseline_runs.items())
                if key.startswith(f"{arm}|{problem}|")
            ]
            arm_auc = [
                float(entry["auc_hv"])
                for key, entry in sorted(baseline_runs.items())
                if key.startswith(f"{arm}|{problem}|")
            ]
            wilcoxon: dict[str, Any] = {}
            for metric, paired in (("final_hv", paired_hv), ("auc_hv", paired_auc)):
                result = _wilcoxon_greater(*paired)
                wilcoxon[metric] = result
                if result.get("p_value") is not None:
                    families[(metric, arm)][problem] = float(result["p_value"])
            comparisons[arm] = {
                "n_runs": len(arm_hv),
                "n_paired": len(paired_hv[0]),
                "final_hv": _mean_std(arm_hv),
                "auc_hv": _mean_std(arm_auc),
                "wilcoxon": wilcoxon,
            }
        problem_reports[problem] = {
            ARM_PLANNING: {
                "n_runs": len(planning_hv),
                "n_failed": int(n_failed),
                "failure_rate": float(n_failed / len(planning_hv))
                if planning_hv
                else 0.0,
                "final_hv": _mean_std(planning_hv),
                "auc_hv": _mean_std(planning_auc),
            },
            "comparisons": comparisons,
        }

    # Holm correction across the five problems within each
    # (metric, comparison-arm) family, plus the paired median difference and
    # its 95% bootstrap CI. Raw p-values are preserved alongside.
    holm: dict[tuple[str, str], dict[str, float]] = {
        key: holm_correction(values) for key, values in families.items() if values
    }
    for problem, report in problem_reports.items():
        for arm, comparison in report["comparisons"].items():
            for metric in metrics:
                p_holm = holm.get((metric, arm), {}).get(problem)
                comparison["wilcoxon"][metric] = dict(
                    comparison["wilcoxon"][metric], p_holm=p_holm
                )
                paired = paired_series.get((problem, arm, metric))
                if paired is None or len(paired[0]) < 2:
                    comparison[f"delta_{metric}"] = {
                        "median": None,
                        "ci95": None,
                    }
                    continue
                median_diff, lo, hi = paired_bootstrap_ci(
                    paired[0], paired[1], n_boot=BOOTSTRAP_RESAMPLES, seed=0
                )
                comparison[f"delta_{metric}"] = {
                    "median": float(median_diff),
                    "ci95": [float(lo), float(hi)],
                }

    failure_rates: dict[str, Any] = {}
    planning_flags = [
        bool(entry.get("failed")) for entry in planning_runs.values()
    ]
    failure_rates[ARM_PLANNING] = {
        problem: float(
            sum(
                1
                for key, entry in planning_runs.items()
                if key.split("|")[1] == problem and entry.get("failed")
            )
            / max(
                1,
                sum(
                    1
                    for key in planning_runs
                    if key.split("|")[1] == problem
                ),
            )
        )
        for problem in problems
    }
    failure_rates[ARM_PLANNING]["overall"] = (
        float(sum(planning_flags) / len(planning_flags)) if planning_flags else 0.0
    )
    for arm in arm_names:
        entry: dict[str, float] = {}
        arm_flags: list[bool] = []
        for problem in problems:
            flags = [
                bool(v.get("failed"))
                for k, v in baseline_runs.items()
                if k.startswith(f"{arm}|{problem}|")
            ]
            entry[problem] = float(sum(flags) / len(flags)) if flags else 0.0
            arm_flags.extend(flags)
        entry["overall"] = (
            float(sum(arm_flags) / len(arm_flags)) if arm_flags else 0.0
        )
        failure_rates[arm] = entry

    return {
        "config": {
            "planning_arm": ARM_PLANNING,
            "baseline_arms": list(arm_names),
            "test": (
                "paired Wilcoxon signed-rank (zero_method='zsplit', "
                "alternative='greater': planning_predictor metric > arm "
                "metric), paired by (problem, seed)"
            ),
            "holm_families": (
                "Holm-Bonferroni step-down across the five problems, applied "
                "separately within each (metric, comparison-arm) family "
                "(10 families: 2 metrics x 5+ comparison arms)"
            ),
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": 0,
                "interval": "percentile 2.5/97.5 of paired median difference",
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "problems": problem_reports,
        "failure_rates": failure_rates,
    }


def run_aggregate_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Merge ``runs/*.json`` into ``results.json`` and write ``comparison.json``.

    Returns:
        The results payload as written to ``{out_dir}/results.json``.
    """
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    problems = [get_problem(name).name for name in args.problems]
    eval_seeds = [int(s) for s in args.seeds]

    runs, eval_runtime_sum = _merge_runs(runs_dir)
    phase1_75 = _load_phase1_75_results(args.phase1_75_results)
    failure_thresholds = phase1_75.get("failure_thresholds", {})

    aggregate_wall = time.perf_counter() - started
    payload: dict[str, Any] = {
        "config": {
            "phase": "2B",
            "arm": ARM_PLANNING,
            "predictor_dir": str(args.predictor_dir),
            "phase1_75_results": str(args.phase1_75_results),
            "problems": problems,
            "eval_seeds": eval_seeds,
            "pop_size": int(args.pop_size),
            "generations": int(args.generations),
            "n_candidates": int(args.n_candidates),
            "candidate_seed": int(args.candidate_seed),
            "horizon_weights": (
                None
                if args.horizon_weights is None
                else [float(w) for w in args.horizon_weights]
            ),
            "pm_mult_range": list(DEFAULT_PM_MULT_RANGE),
            "n_reference_points": int(args.n_reference_points),
            "ref_point": [float(v) for v in args.ref_point],
            "wall_time_sec": float(eval_runtime_sum + aggregate_wall),
            "wall_time_breakdown": {
                "eval_runtime_sum_sec": float(eval_runtime_sum),
                "aggregate_sec": float(aggregate_wall),
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "runs": runs,
        "failure_thresholds": failure_thresholds,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[aggregate] merged {len(runs)} runs -> {results_path}")

    comparison = build_comparison(runs, phase1_75, problems)
    comparison_path = out_dir / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, ensure_ascii=False)
    print(f"[aggregate] wrote comparison -> {comparison_path}")
    _print_comparison_table(comparison, problems)
    return payload


def _print_comparison_table(
    comparison: dict[str, Any], problems: Sequence[str]
) -> None:
    """Print the Holm-corrected planning comparison as a compact table.

    Args:
        comparison: Payload returned by :func:`build_comparison`.
        problems: Problem names in report order.
    """
    print(
        "\n[planning vs baseline] final_hv: raw p / Holm p / median delta "
        "[95% CI] / failure rate"
    )
    header = f"{'problem':<8}{'arm':<28}{'raw_p':>10}{'holm_p':>10}"
    header += f"{'delta_med':>12}{'ci95':>22}{'fail':>8}"
    print(header)
    for problem in problems:
        report = comparison["problems"].get(problem)
        if report is None:
            continue
        for arm, comp in report["comparisons"].items():
            wilcoxon = comp["wilcoxon"]["final_hv"]
            delta = comp.get("delta_final_hv", {})
            ci = delta.get("ci95")
            ci_text = (
                f"[{ci[0]:.4f},{ci[1]:.4f}]"
                if isinstance(ci, list) and len(ci) == 2 and None not in ci
                else "n/a"
            )
            raw_p = wilcoxon.get("p_value")
            holm_p = wilcoxon.get("p_holm")
            rate = comparison["failure_rates"].get(arm, {}).get(problem, 0.0)
            median = delta.get("median")
            raw_text = f"{raw_p:.4g}" if raw_p is not None else "n/a"
            holm_text = f"{holm_p:.4g}" if holm_p is not None else "n/a"
            median_text = f"{median:.4f}" if median is not None else "n/a"
            print(
                f"{problem:<8}{arm:<28}{raw_text:>10}{holm_text:>10}"
                f"{median_text:>12}{ci_text:>22}{rate:>8.2f}"
            )
        print()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the requested Phase-2B stage(s).

    Args:
        args: Parsed arguments as produced by :func:`parse_args`.

    Returns:
        For ``aggregate``/``all``: the results payload written to
        ``results.json``. For ``eval``: a summary dict with the per-run
        metrics under ``"runs"``.
    """
    stage = str(args.stage)
    if stage == "eval":
        return {"runs": run_eval_stage(args)}
    if stage == "aggregate":
        return run_aggregate_stage(args)
    run_eval_stage(args)
    return run_aggregate_stage(args)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point: parse arguments and run the requested stage(s).

    Args:
        argv: Optional argument list; ``None`` reads ``sys.argv``.

    Returns:
        The payload of the last executed stage.
    """
    return run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
