from __future__ import annotations

"""Phase 2.75, Task 5: evaluate the advantage planner against Phase-1.75/2B arms.

Five arms are deployed on the same held-out ``(problem, seed)`` grid with the
protocol of Phase 1.75 / 2B (generation 0 uses the algorithm defaults; the
action for the step into generation ``t >= 1`` is chosen from the merged
history so far):

``advantage_planner``
    :class:`controller.advantage_planner_controller.AdvantagePlannerController`
    over the predictor trained by ``experiments/train_advantage_predictor.py``
    (``{model_dir}/model_contrastive.pt`` by default, paired with the
    ``encoder.json`` that script writes). **Control arm.**
``phase2b_planner``
    Phase-2B :class:`~controller.planning_controller.PlanningController` over
    the Phase-2A outcome predictor (``results/phase2_outcome``).
``fixed_nsga2``
    Plain NSGA-II with the ``OperatorConfig`` defaults (no action injection).
``static_full_global``
    The tuned global full-action tuple of Phase 1.75
    (``results/phase1_75/controllers/static_full_global.json``).
``generation_only_mlp``
    The open-loop, state-free schedule controller of Phase 1.75
    (``generation_only_mlp.pt`` + ``train_info.json``).

The three baselines reuse the Phase-1.75 artifacts as-is; nothing is retrained.
Stages mirror ``experiments/run_phase2b.py``:

* ``eval`` — run the ``(arm, problem, seed)`` grid and write one JSON per run
  (with the recorded action trajectory) to ``{out_dir}/runs/``; the grid can be
  split with ``--shard/--num-shards`` for parallel workers;
* ``aggregate`` — merge the runs into ``{out_dir}/results.json`` and build
  ``{out_dir}/comparison.json``: paired one-sided Wilcoxon tests of the
  advantage planner against every other arm, **Holm-Bonferroni** adjustment
  per ``(metric, comparison arm)`` family across problems (reusing
  ``experiments/analyze_phase1_75.py``), paired bootstrap 95% CIs of the median
  difference, and failure rates next to every mean;
* ``all`` — eval then aggregate.

Every artifact stays inside ``--out-dir`` (default
``results/phase2_75/eval``); Phase-1.75 and Phase-2B directories are only read.

Example:
    ``python experiments/run_phase2_75.py --stage all``
    ``python experiments/run_phase2_75.py --stage eval --shard 0 --num-shards 5``
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    # Allow ``python experiments/run_phase2_75.py`` from the repo root: the
    # script directory (not the repo root) is on sys.path then.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms import NSGAII, OperatorConfig
from benchmarks import get_problem
from controller.advantage_planner_controller import AdvantagePlannerController
from controller.advantage_predictor import AdvantagePredictor
from controller.dataset import merge_state_reward
from controller.multihead_controller import MultiHeadController
from controller.outcome_predictor import OutcomePredictor
from controller.planning_controller import PlanningController
from controller.state_encoder import StateEncoder
from controller.static_full_controller import StaticFullController
from experiments.analyze_phase1_75 import (
    failure_rate,
    holm_correction,
    paired_bootstrap_ci,
    paired_wilcoxon,
)
from experiments.run_phase1 import _auc_hv
from experiments.run_phase1_75 import GenerationOnlyPolicy, _sanitize_action
from trajectory import EvolutionRecorder

#: Arm identifiers (the advantage planner is the control of the comparison).
ARM_ADVANTAGE = "advantage_planner"
ARM_PHASE2B = "phase2b_planner"
ARM_FIXED = "fixed_nsga2"
ARM_STATIC_GLOBAL = "static_full_global"
ARM_GENERATION_ONLY = "generation_only_mlp"
ARMS: tuple[str, ...] = (
    ARM_ADVANTAGE,
    ARM_PHASE2B,
    ARM_FIXED,
    ARM_STATIC_GLOBAL,
    ARM_GENERATION_ONLY,
)
#: Arms whose artefacts live in the Phase-1.75 controller directory.
PHASE1_75_ARMS: tuple[str, ...] = (ARM_STATIC_GLOBAL, ARM_GENERATION_ONLY)
#: Metrics compared between arms.
COMPARISON_METRICS: tuple[str, ...] = ("final_hv", "auc_hv")

DEFAULT_PROBLEMS: tuple[str, ...] = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(1000, 1020))
#: Trained advantage model directory (written by train_advantage_predictor.py).
DEFAULT_MODEL_DIR = "results/phase2_75/model"
#: Isolated output directory of this phase.
DEFAULT_OUT_DIR = "results/phase2_75/eval"
#: Phase-1.75 controller artefacts (baseline arms; read-only).
DEFAULT_PHASE1_75_DIR = "results/phase1_75"
#: Phase-2B artefacts (comparison planner; read-only).
DEFAULT_PHASE2B_DIR = "results/phase2_outcome"
#: Predictor checkpoint used by the advantage arm.
DEFAULT_PREDICTOR_FILE = "model_contrastive.pt"
#: Deployment pm bounds as multipliers of ``1 / n_vars``.
DEFAULT_PM_MULT_RANGE: tuple[float, float] = (0.25, 8.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments of the Phase-2.75 evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2.75: paired evaluation of the advantage planner against "
            "the Phase-2B planner, fixed NSGA-II, the static full-action "
            "baseline and the generation-only schedule (Holm-corrected)."
        )
    )
    parser.add_argument("--stage", choices=["eval", "aggregate", "all"],
                        default="all", help="Pipeline stage (default: %(default)s).")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Advantage model directory (default: %(default)s).")
    parser.add_argument("--predictor-file", type=str, default=DEFAULT_PREDICTOR_FILE,
                        help="Predictor checkpoint inside --model-dir "
                        "(default: %(default)s).")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Output directory (default: %(default)s).")
    parser.add_argument("--phase1-75-dir", type=str, default=DEFAULT_PHASE1_75_DIR,
                        help="Phase-1.75 artefact root, read-only (default: %(default)s).")
    parser.add_argument("--phase2b-dir", type=str, default=DEFAULT_PHASE2B_DIR,
                        help="Phase-2B artefact root, read-only (default: %(default)s).")
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS),
                        help="Problem subset (sharding; default: %(default)s).")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                        help="Held-out seeds (sharding; default: 1000..1019).")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS),
                        help="Arms to run (default: %(default)s).")
    parser.add_argument("--pop-size", type=int, default=100,
                        help="NSGA-II population size (default: %(default)s).")
    parser.add_argument("--generations", type=int, default=100,
                        help="NSGA-II generations per run (default: %(default)s).")
    parser.add_argument("--n-candidates", type=int, default=16,
                        help="Candidates sampled per generation by the advantage "
                        "planner (default: %(default)s).")
    parser.add_argument("--shard", type=int, default=0,
                        help="Shard index over the flattened (arm, problem, seed) "
                        "grid (default: %(default)s).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Number of shards (default: %(default)s).")
    parser.add_argument("--pm-mult-range", nargs=2, type=float,
                        default=list(DEFAULT_PM_MULT_RANGE),
                        metavar=("PM_MULT_LO", "PM_MULT_HI"),
                        help="Multipliers on 1/n_vars bounding pm (default: %(default)s).")
    parser.add_argument("--n-reference-points", type=int, default=200,
                        help="Points sampled from the true front for IGD "
                        "(default: %(default)s).")
    parser.add_argument("--ref-point", nargs=2, type=float, default=[1.1, 1.1],
                        metavar=("REF_F1", "REF_F2"),
                        help="Hypervolume reference point (default: %(default)s).")
    parser.add_argument("--n-resamples", type=int, default=10000,
                        help="Bootstrap resamples for the paired CI "
                        "(default: %(default)s).")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# context (controller artefacts)
# ---------------------------------------------------------------------------


def _require(path: Path, what: str) -> Path:
    """Return ``path`` or raise a helpful FileNotFoundError."""
    if not path.is_file():
        raise FileNotFoundError(f"required {what} not found: {path}")
    return path


def load_context(
    args: argparse.Namespace, problems: Sequence[str]
) -> dict[str, Any]:
    """Load the controller artefacts of the requested arms.

    Args:
        args: Parsed arguments (directories, seed of the model file).
        problems: Problem names of the evaluation (unused by the single
            global baselines, kept for interface symmetry).

    Returns:
        ``{arm: context}``; the learned arms map to ``(controller, encoder)``
        tuples, ``static_full_global`` to the controller alone and
        ``fixed_nsga2`` to ``None``.

    Raises:
        FileNotFoundError: If an artefact of a requested arm is missing.
    """
    arms = set(args.arms)
    context: dict[str, Any] = {}
    if ARM_ADVANTAGE in arms:
        model_dir = Path(args.model_dir)
        predictor = AdvantagePredictor.load(
            _require(model_dir / args.predictor_file, "advantage predictor")
        )
        encoder = StateEncoder.load(_require(model_dir / "encoder.json", "encoder"))
        controller = AdvantagePlannerController(
            predictor, n_candidates=int(args.n_candidates)
        )
        controller.predictor_path = str(model_dir / args.predictor_file)
        context[ARM_ADVANTAGE] = (controller, encoder)
    if ARM_PHASE2B in arms:
        phase2b = Path(args.phase2b_dir)
        predictor = OutcomePredictor.load(
            _require(phase2b / "predictor.pt", "Phase-2A predictor")
        )
        encoder = StateEncoder.load(_require(phase2b / "encoder.json", "encoder"))
        controller = PlanningController.load(
            _require(phase2b / "planning_controller.json", "planner config"),
            predictor,
        )
        controller.predictor_path = str(phase2b / "predictor.pt")
        context[ARM_PHASE2B] = (controller, encoder)
    if ARM_FIXED in arms:
        context[ARM_FIXED] = None
    controllers_dir = Path(args.phase1_75_dir) / "controllers"
    if ARM_STATIC_GLOBAL in arms:
        context[ARM_STATIC_GLOBAL] = StaticFullController.load(
            _require(controllers_dir / f"{ARM_STATIC_GLOBAL}.json", "static controller")
        )
    if ARM_GENERATION_ONLY in arms:
        controller = MultiHeadController.load(
            _require(controllers_dir / f"{ARM_GENERATION_ONLY}.pt", "schedule controller")
        )
        with _require(controllers_dir / "train_info.json", "train_info").open(
            "r", encoding="utf-8"
        ) as fh:
            info = json.load(fh)
        context[ARM_GENERATION_ONLY] = GenerationOnlyPolicy(
            controller,
            info["generation_only"]["problem_vectors_zscored"],
            mutation_target=info["generation_only"].get("mutation_target", "multiplier"),
        )
    return context


def action_for_arm(
    arm: str,
    context: dict[str, Any],
    problem: Any,
    *,
    generation: int,
    generations: int,
    history: list[dict[str, Any]],
    pm_mult_range: tuple[float, float],
) -> dict[str, Any] | None:
    """Action the arm wants for the step into ``generation``.

    Args:
        arm: Arm identifier.
        context: Output of :func:`load_context`.
        problem: Benchmark instance (for ``n_vars``/name).
        generation: Generation the action applies to (``t >= 1``).
        generations: Total generations of the run (normalizer of the
            generation-only arm).
        history: Merged state+reward dicts of generations ``[0, t)``.
        pm_mult_range: Deployment multiplier bounds.

    Returns:
        A 3-key action dict, or ``None`` for ``fixed_nsga2`` (no injection).
    """
    n_vars = int(problem.n_vars)
    if arm == ARM_FIXED:
        return None
    if arm == ARM_STATIC_GLOBAL:
        return context[arm].predict_action(None, n_vars=n_vars)
    if arm == ARM_GENERATION_ONLY:
        return context[arm].predict_action(
            generation,
            generations,
            problem.name,
            n_vars=n_vars,
            pm_mult_range=pm_mult_range,
        )
    controller, encoder = context[arm]
    pm_min = float(pm_mult_range[0]) / n_vars
    pm_max = float(pm_mult_range[1]) / n_vars
    return controller.predict_action(history, encoder, pm_min, pm_max, n_vars=n_vars)


# ---------------------------------------------------------------------------
# eval stage
# ---------------------------------------------------------------------------


def _run_config(
    *,
    arm: str,
    problem_name: str,
    n_vars: int,
    seed: int,
    pop_size: int,
    generations: int,
    pm_mult_range: tuple[float, float],
    failure_threshold: float | None,
    ref_point: np.ndarray,
    n_reference_points: int,
    controller_note: str,
) -> dict[str, Any]:
    """Configuration block stored inside each per-run JSON."""
    return {
        "phase": "2.75",
        "arm": arm,
        "problem": problem_name,
        "n_vars": int(n_vars),
        "algorithm": "nsga2",
        "seed": int(seed),
        "pop_size": int(pop_size),
        "generations": int(generations),
        "action_space": "full",
        "controller": controller_note,
        "pm_mult_range": [float(pm_mult_range[0]), float(pm_mult_range[1])],
        "failure_threshold": (
            None if failure_threshold is None else float(failure_threshold)
        ),
        "protocol": (
            "generation 0 uses the algorithm defaults; the action for the step "
            "into generation t >= 1 is chosen from the merged history of "
            "generations [0, t); the recorded action is the action actually used"
        ),
        "ref_point": [float(ref_point[0]), float(ref_point[1])],
        "n_reference_points": int(n_reference_points),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_run(
    arm: str,
    problem_name: str,
    seed: int,
    *,
    context: dict[str, Any],
    generations: int,
    pop_size: int,
    pm_mult_range: tuple[float, float] = DEFAULT_PM_MULT_RANGE,
    n_reference_points: int = 200,
    ref_point: np.ndarray | None = None,
    failure_threshold: float | None = None,
    runs_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one ``(arm, problem, seed)`` evaluation.

    Returns:
        Metrics dict with ``arm``, ``problem``, ``seed``, ``final_hv``,
        ``final_igd``, ``auc_hv``, ``runtime_sec``, ``failed`` and
        ``trajectory_file``.
    """
    if ref_point is None:
        ref_point = np.asarray([1.1, 1.1], dtype=float)
    problem = get_problem(problem_name)
    n_vars = int(problem.n_vars)
    algorithm = NSGAII(
        problem, pop_size=pop_size, operators=OperatorConfig(), seed=seed
    )
    recorder = EvolutionRecorder(
        problem_name=problem.name,
        reference_front=problem.reference_front(n_points=n_reference_points),
        ref_point=ref_point,
    )
    start = time.perf_counter()
    algorithm.initialize()
    recorder.record(
        algorithm.generation, algorithm.nondominated_front(), algorithm.current_action()
    )
    for _ in range(int(generations)):
        history = [merge_state_reward(transition) for transition in recorder.transitions()]
        action = action_for_arm(
            arm,
            context,
            problem,
            generation=algorithm.generation + 1,
            generations=int(generations),
            history=history,
            pm_mult_range=pm_mult_range,
        )
        if action is None:
            algorithm.step()
        else:
            sanitized = _sanitize_action(
                action, n_vars=n_vars, pm_mult_range=pm_mult_range
            )
            algorithm.step(
                mutation_prob=sanitized["mutation_probability"],
                mutation_operator=sanitized["mutation_operator"],
                exploration_strength=sanitized["exploration_strength"],
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
        "arm": arm,
        "problem": problem.name,
        "seed": int(seed),
        "final_hv": final_hv,
        "final_igd": float(transitions[-1]["state"]["igd"]),
        "auc_hv": _auc_hv(transitions, int(generations)),
        "runtime_sec": float(runtime_sec),
        "failed": failed,
        "trajectory_file": None,
    }
    if runs_dir is not None:
        config = _run_config(
            arm=arm,
            problem_name=problem.name,
            n_vars=n_vars,
            seed=seed,
            pop_size=pop_size,
            generations=int(generations),
            pm_mult_range=pm_mult_range,
            failure_threshold=failure_threshold,
            ref_point=ref_point,
            n_reference_points=n_reference_points,
            controller_note=controller_note(arm, context),
        )
        out_path = Path(runs_dir) / f"{arm}__{problem.name}__seed{seed}.json"
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
    if verbose:
        print(
            f"[eval] {arm} {problem.name} seed={seed} hv={final_hv:.6f} "
            f"failed={failed} ({runtime_sec:.2f}s)"
        )
    return metrics


def controller_note(arm: str, context: dict[str, Any]) -> str:
    """Human-readable provenance of the arm's controller artifact."""
    if arm == ARM_FIXED:
        return "none (OperatorConfig defaults)"
    if arm == ARM_STATIC_GLOBAL:
        return "StaticFullController.load(static_full_global.json)"
    if arm == ARM_GENERATION_ONLY:
        return "GenerationOnlyPolicy(MultiHeadController.load(generation_only_mlp.pt))"
    if arm == ARM_PHASE2B:
        return "PlanningController.load(phase2_outcome/planning_controller.json)"
    controller, _encoder = context[arm]
    return f"{type(controller).__name__}({controller.predictor_path})"


def _failure_thresholds(phase1_75_dir: str | Path) -> dict[str, float]:
    """Per-problem failure thresholds of Phase 1.75 (empty when absent)."""
    path = Path(phase1_75_dir) / "failure_thresholds.json"
    if not path.is_file():
        print(f"[warn] {path} missing; runs are marked failed=false")
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    raw = payload.get("thresholds", payload)
    return {str(name): float(value) for name, value in raw.items()}


def shard_grid(
    arms: Sequence[str],
    problems: Sequence[str],
    seeds: Sequence[int],
    shard: int,
    num_shards: int,
) -> list[tuple[str, str, int]]:
    """Deterministic shard of the flattened ``(arm, problem, seed)`` grid.

    Raises:
        ValueError: If the shard indices are inconsistent.
    """
    if int(num_shards) < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= int(shard) < int(num_shards):
        raise ValueError(
            f"shard must lie in [0, {int(num_shards)}), got {shard}"
        )
    grid = [
        (str(arm), str(problem), int(seed))
        for arm in arms
        for problem in problems
        for seed in seeds
    ]
    if int(num_shards) == 1:
        return grid
    return grid[int(shard) :: int(num_shards)]


def run_eval_stage(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Evaluate the requested grid slice and write per-run JSONs."""
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    problems = [get_problem(name).name for name in args.problems]
    seeds = [int(s) for s in args.seeds]
    pm_mult_range = (float(args.pm_mult_range[0]), float(args.pm_mult_range[1]))
    ref_point = np.asarray(args.ref_point, dtype=float)
    thresholds = _failure_thresholds(args.phase1_75_dir)
    context = load_context(args, problems)
    grid = shard_grid(args.arms, problems, seeds, args.shard, args.num_shards)
    print(
        f"[eval] {len(grid)} runs (arms={list(args.arms)}, problems={problems}, "
        f"seeds={seeds[0]}..{seeds[-1]}, shard {args.shard}/{args.num_shards})"
    )
    metrics: list[dict[str, Any]] = []
    for arm, problem_name, seed in grid:
        metrics.append(
            evaluate_run(
                arm,
                problem_name,
                seed,
                context=context,
                generations=int(args.generations),
                pop_size=int(args.pop_size),
                pm_mult_range=pm_mult_range,
                n_reference_points=int(args.n_reference_points),
                ref_point=ref_point,
                failure_threshold=thresholds.get(problem_name),
                runs_dir=runs_dir,
            )
        )
    print(f"[eval] wrote {len(metrics)} runs -> {runs_dir}")
    return metrics


# ---------------------------------------------------------------------------
# aggregate stage
# ---------------------------------------------------------------------------


def merge_runs(runs_dir: str | Path) -> dict[str, Any]:
    """Merge per-run JSONs into an ``{"arm|problem|seed": metrics}`` mapping."""
    directory = Path(runs_dir)
    runs: dict[str, Any] = {}
    if not directory.is_dir():
        return runs
    for path in sorted(directory.glob("*.json")):
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
            print(f"[warn] {path} has neither metrics nor transitions; skipped")
            continue
        runs[f"{arm}|{problem}|{seed}"] = {
            "final_hv": float(metrics["final_hv"]),
            "final_igd": float(metrics["final_igd"]),
            "auc_hv": float(metrics["auc_hv"]),
            "runtime_sec": float(metrics["runtime_sec"]),
            "failed": bool(metrics["failed"]),
            "trajectory_file": f"runs/{path.name}",
        }
    return runs


def _series(
    runs: dict[str, Any], arm: str, problem: str, metric: str
) -> dict[int, float]:
    """``{seed: value}`` of one arm/problem/metric."""
    values: dict[int, float] = {}
    for key, entry in runs.items():
        parts = key.split("|")
        if len(parts) == 3 and parts[0] == arm and parts[1] == problem:
            values[int(parts[2])] = float(entry[metric])
    return values


def _failed_series(runs: dict[str, Any], arm: str, problem: str) -> dict[int, bool]:
    """``{seed: failed}`` of one arm/problem."""
    flags: dict[int, bool] = {}
    for key, entry in runs.items():
        parts = key.split("|")
        if len(parts) == 3 and parts[0] == arm and parts[1] == problem:
            flags[int(parts[2])] = bool(entry.get("failed", False))
    return flags


def _safe_wilcoxon(a: Sequence[float], b: Sequence[float]) -> float | None:
    """One-sided paired Wilcoxon p-value, ``None`` when undefined."""
    if len(a) < 2 or np.all(np.asarray(a, dtype=float) == np.asarray(b, dtype=float)):
        return None
    try:
        return paired_wilcoxon(a, b, alternative="greater")
    except ValueError:
        return None


def build_comparison(
    runs: dict[str, Any],
    problems: Sequence[str],
    *,
    control_arm: str = ARM_ADVANTAGE,
    arms: Sequence[str] = ARMS,
    metrics: Sequence[str] = COMPARISON_METRICS,
    n_resamples: int = 10000,
) -> dict[str, Any]:
    """Paired comparison of the control arm against every other arm.

    For each ``(problem, metric, arm)`` the runs are paired by seed and
    summarised as mean±std and failure rate, with a one-sided paired Wilcoxon
    test (control greater) and a paired bootstrap CI of the median difference.
    Raw p-values are then Holm-adjusted inside each ``(metric, arm)`` family
    across problems, exactly like Phase 1.75 / 2B.

    Args:
        runs: Merged run mapping.
        problems: Problems to compare.
        control_arm: Arm the others are compared against.
        arms: Arms present in the run mapping.
        metrics: Metrics to compare.
        n_resamples: Bootstrap resamples of the CI.

    Returns:
        The comparison payload written to ``comparison.json``.
    """
    other_arms = [arm for arm in arms if arm != control_arm]
    reports: dict[str, Any] = {}
    family_p_values: dict[tuple[str, str], dict[str, float]] = {}
    for problem in problems:
        entry: dict[str, Any] = {
            "control": {
                "arm": control_arm,
                "n_runs": len(_series(runs, control_arm, problem, "final_hv")),
                "failure_rate": failure_rate(
                    list(_failed_series(runs, control_arm, problem).values())
                ),
            },
            "comparisons": {},
        }
        for arm in other_arms:
            comparisons: dict[str, Any] = {}
            for metric in metrics:
                control_series = _series(runs, control_arm, problem, metric)
                arm_series = _series(runs, arm, problem, metric)
                shared = sorted(set(control_series) & set(arm_series))
                if not shared:
                    comparisons[metric] = {
                        "n_paired": 0,
                        "wilcoxon_p": None,
                        "holm_p": None,
                        "median_diff": None,
                        "bootstrap_ci_95": None,
                    }
                    continue
                control_values = [control_series[seed] for seed in shared]
                arm_values = [arm_series[seed] for seed in shared]
                p_value = _safe_wilcoxon(control_values, arm_values)
                median_diff, ci_lo, ci_hi = paired_bootstrap_ci(
                    control_values, arm_values, n_boot=int(n_resamples)
                )
                comparisons[metric] = {
                    "n_paired": len(shared),
                    "control_mean": float(np.mean(control_values)),
                    "arm_mean": float(np.mean(arm_values)),
                    "control_std": (
                        float(np.std(control_values, ddof=1))
                        if len(control_values) > 1
                        else 0.0
                    ),
                    "arm_std": (
                        float(np.std(arm_values, ddof=1))
                        if len(arm_values) > 1
                        else 0.0
                    ),
                    "control_failure_rate": failure_rate(
                        [_failed_series(runs, control_arm, problem)[s] for s in shared]
                    ),
                    "arm_failure_rate": failure_rate(
                        [_failed_series(runs, arm, problem)[s] for s in shared]
                    ),
                    "wilcoxon_p": p_value,
                    "median_diff": float(median_diff),
                    "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
                    "holm_p": None,
                }
                if p_value is not None:
                    family_p_values.setdefault((metric, arm), {})[problem] = p_value
            entry["comparisons"][arm] = comparisons
        reports[problem] = entry

    holm_report: dict[str, Any] = {}
    for (metric, arm), p_values in sorted(family_p_values.items()):
        adjusted = holm_correction(p_values) if len(p_values) > 1 else dict(p_values)
        for problem, value in adjusted.items():
            reports[problem]["comparisons"][arm][metric]["holm_p"] = float(value)
        holm_report[f"{metric}|{arm}"] = {
            "n_tests": len(p_values),
            "raw_p": {k: float(v) for k, v in sorted(p_values.items())},
            "holm_p": {k: float(v) for k, v in sorted(adjusted.items())},
        }

    return {
        "config": {
            "control_arm": control_arm,
            "comparison_arms": other_arms,
            "metrics": list(metrics),
            "test": (
                "paired one-sided Wilcoxon (zero_method='zsplit', "
                "alternative='greater': control > arm) + paired bootstrap 95% "
                "CI of the median difference + failure rates; Holm-Bonferroni "
                "per (metric, comparison arm) family across problems"
            ),
            "n_resamples": int(n_resamples),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "problems": reports,
        "holm_families": holm_report,
    }


def run_aggregate_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Merge runs and write ``results.json`` plus ``comparison.json``."""
    out_dir = Path(args.out_dir)
    runs = merge_runs(out_dir / "runs")
    problems = [get_problem(name).name for name in args.problems]
    payload: dict[str, Any] = {
        "config": {
            "phase": "2.75",
            "arms": list(args.arms),
            "control_arm": ARM_ADVANTAGE,
            "model_dir": str(args.model_dir),
            "predictor_file": str(args.predictor_file),
            "problems": problems,
            "eval_seeds": [int(s) for s in args.seeds],
            "pop_size": int(args.pop_size),
            "generations": int(args.generations),
            "n_candidates": int(args.n_candidates),
            "pm_mult_range": [float(v) for v in args.pm_mult_range],
            "n_reference_points": int(args.n_reference_points),
            "ref_point": [float(v) for v in args.ref_point],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "runs": runs,
        "failure_thresholds": _failure_thresholds(args.phase1_75_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[aggregate] merged {len(runs)} runs -> {results_path}")
    comparison = build_comparison(
        runs,
        problems,
        control_arm=ARM_ADVANTAGE,
        arms=list(args.arms),
        n_resamples=int(args.n_resamples),
    )
    comparison_path = out_dir / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, ensure_ascii=False)
    print(f"[aggregate] wrote comparison -> {comparison_path}")
    for family, entry in comparison["holm_families"].items():
        print(
            f"[holm] {family}: n={entry['n_tests']} "
            f"raw={entry['raw_p']} holm={entry['holm_p']}"
        )
    return payload


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the requested stage(s)."""
    stage = str(args.stage)
    if stage == "eval":
        return {"runs": run_eval_stage(args)}
    if stage == "aggregate":
        return run_aggregate_stage(args)
    run_eval_stage(args)
    return run_aggregate_stage(args)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point."""
    return run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
