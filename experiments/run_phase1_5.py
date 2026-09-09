from __future__ import annotations

"""Phase 1.5: state-dependent evolution decisions vs a better global constant.

This runner implements the Phase-1.5 experiment of the EvoController roadmap
(``docs/PHASE1_5_PLAN.md``). Phase 1 showed that evolution trajectories
contain learnable signals for the mutation probability; Phase 1.5 asks
whether a controller learns *state-dependent* evolution decisions or merely
a better global constant. To that end the action space is expanded to the
triple ``(mutation_operator, mutation_probability, exploration_strength)``
and the controller input optionally includes problem-aware features.

New arms (both trained on the union of ``--train-dirs``):

* ``mlp2`` — :class:`controller.MultiHeadController` over per-problem
  :class:`controller.ProblemAwareEncoder` encodings (one encoder per
  problem, all fitted over ALL training trajectories and ALL problems so
  the state and problem z-score statistics are shared; each training
  trajectory contributes samples encoded with its own problem's encoder).
* ``mlp2_nopf`` — the same multi-head controller over the plain
  :class:`controller.StateEncoder` (no problem features; ablation).

Reused arms (no re-evaluation): ``fixed``, ``constant``, and ``mlp_w10``
per-seed metrics are read from the Phase-1 ``results.json``
(``--phase1-results``), restricted to ``--eval-seeds``.

Deployment semantics (shared with the dataset builder and Phase 1): at
generation ``t >= 1`` the controller observes the merged state dicts of
generations ``[t - window, t)`` and emits the full action triple used for
the step *into* generation ``t``; generation 0 is recorded with the
algorithm's default action. The deployment mutation-probability bounds are
``(1 / n_vars) * [0.25, 8.0]`` and the exploration-strength bounds are the
operator-specific ranges of the Phase-1.5 dataset generator
(``eta_m in [2, 50]`` polynomial, ``sigma in [0.02, 0.3]`` Gaussian).

Outputs (under ``--out-dir``):

* ``trajectories/{arm}_{problem}_seed{seed}.json`` — one trajectory per
  (new arm, problem, eval seed), in the recorder schema (three-key actions).
* ``results.json`` — per arm x problem aggregates (mean/std of final HV,
  final IGD, anytime AUC-HV, runtime, per-seed raw values) for the two new
  and three reused arms; Wilcoxon signed-rank p-values of each arm vs
  ``fixed`` AND vs ``constant`` on final HV; an action-dynamics summary per
  new arm x problem (mean within-run pm std, mean operator switch rate,
  mean pm); training losses; the full configuration; and the wall time.

Example:
    ``python experiments/run_phase1_5.py``
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
    # Allow ``python experiments/run_phase1_5.py`` from the repo root: the
    # script directory (not the repo root) is on sys.path in that mode.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms import NSGAII, OperatorConfig
from benchmarks import get_problem
from controller import (
    MultiHeadController,
    ProblemAwareEncoder,
    StateEncoder,
    build_multihead_samples,
    merge_state_reward,
    train_val_split,
)
from controller.multihead_controller import (
    GAUSSIAN_EXPLORATION_RANGE,
    POLYNOMIAL_EXPLORATION_RANGE,
)
from experiments.run_phase1 import _aggregate_runs, _auc_hv, _wilcoxon_greater
from trajectory import EvolutionRecorder

#: Default evaluation problem grid (all Phase-0 ZDT benchmarks).
DEFAULT_PROBLEMS: tuple[str, ...] = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")
#: Default evaluation seeds; match the Phase-1 grid so reused arms pair up.
DEFAULT_EVAL_SEEDS: tuple[int, ...] = (100, 101, 102, 103, 104)
#: Default training trajectory directories (union is used).
DEFAULT_TRAIN_DIRS: tuple[str, ...] = (
    "results/trajectory",
    "results/trajectory_random",
    "results/trajectory_full",
)
#: Default output directory for Phase-1.5 results.
DEFAULT_OUT_DIR = "results/phase1_5"
#: Default Phase-1 results file supplying the reused arms.
DEFAULT_PHASE1_RESULTS = "results/phase1/results.json"
#: Deployment pm bounds: ``(1 / n_vars) * pm_mult_range``.
DEFAULT_PM_MULT_RANGE: tuple[float, float] = (0.25, 8.0)

#: Newly evaluated arm identifiers, in report order.
ARM_MLP2 = "mlp2"
ARM_MLP2_NOPF = "mlp2_nopf"
NEW_ARMS: tuple[str, ...] = (ARM_MLP2, ARM_MLP2_NOPF)
#: Arms reused from the Phase-1 results (not re-evaluated here).
REUSED_ARMS: tuple[str, ...] = ("fixed", "constant", "mlp_w10")
#: All reported arms, in report order (new arms first).
ARMS: tuple[str, ...] = NEW_ARMS + REUSED_ARMS

#: Baselines of the one-sided Wilcoxon comparisons.
BASELINE_FIXED = "fixed"
BASELINE_CONSTANT = "constant"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Phase-1.5 experiment.

    Args:
        argv: Argument list to parse; ``None`` reads ``sys.argv``.

    Returns:
        Parsed namespace with training, evaluation, and output settings.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Phase-1.5 experiment: train multi-head (operator, mutation "
            "probability, exploration strength) evolution controllers and "
            "evaluate them against the reused Phase-1 arms on ZDT benchmarks."
        )
    )
    parser.add_argument(
        "--train-dirs",
        nargs="+",
        default=list(DEFAULT_TRAIN_DIRS),
        help=(
            "Directories with training trajectory JSONs; the union is used "
            "(missing directories are skipped with a warning). Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for results.json and eval trajectories (default: %(default)s).",
    )
    parser.add_argument(
        "--phase1-results",
        type=str,
        default=DEFAULT_PHASE1_RESULTS,
        help=(
            "Phase-1 results.json supplying the reused arms "
            f"{REUSED_ARMS} (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=list(DEFAULT_PROBLEMS),
        help="Evaluation benchmark problems (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_EVAL_SEEDS),
        help="Evaluation random seeds (default: %(default)s).",
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
        "--epochs",
        type=int,
        default=200,
        help="Fixed number of controller training epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="History window of the controller input (default: %(default)s).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of trajectories held out for validation (default: %(default)s).",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=0,
        help="Seed for controller init/training and the train/val split (default: %(default)s).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate for the controllers (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size for the controllers (default: %(default)s).",
    )
    parser.add_argument(
        "--pm-mult-range",
        nargs=2,
        type=float,
        default=list(DEFAULT_PM_MULT_RANGE),
        metavar=("PM_MULT_LO", "PM_MULT_HI"),
        help=(
            "Multipliers on the default 1/n_vars mutation probability defining the "
            "deployment pm range [lo/n_vars, hi/n_vars] (default: %(default)s)."
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print controller training progress.",
    )
    return parser.parse_args(argv)


def _load_training_trajectories(
    train_dirs: Sequence[str | Path],
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """Load the union of training trajectories together with problem names.

    Mirrors :func:`controller.dataset.load_trajectories` (sorted filenames,
    ``index.json`` skipped, transitions sorted by generation) but also
    extracts ``config.problem`` of each run, which the problem-aware arm
    needs to pick its encoder. Missing directories are skipped with a
    warning so partially generated datasets remain usable.

    Args:
        train_dirs: Directories with trajectory JSON files as written by
            ``EvolutionRecorder.save``.

    Returns:
        ``(trajectories, problem_names)``: the loaded trajectories and, in
        parallel order, the problem name of each trajectory.

    Raises:
        ValueError: If no trajectories were found at all, a file has no
            ``transitions`` list, or a trajectory config lacks ``problem``.
    """
    trajectories: list[list[dict[str, Any]]] = []
    problem_names: list[str] = []
    for directory in train_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            print(f"[warn] training directory missing, skipped: {directory}")
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name == "index.json":
                continue
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            transitions = payload.get("transitions")
            if transitions is None:
                raise ValueError(f"{path} has no 'transitions' list")
            problem_name = payload.get("config", {}).get("problem")
            if problem_name is None:
                raise ValueError(
                    f"{path} has no 'config.problem'; the problem-aware encoder "
                    "requires the problem name of each training trajectory"
                )
            trajectories.append(
                sorted(transitions, key=lambda t: int(t["generation"]))
            )
            problem_names.append(str(problem_name))
    if not trajectories:
        raise ValueError(f"no training trajectories found in {list(map(str, train_dirs))}")
    return trajectories, problem_names


def train_multihead_controller(
    trajectories: list[list[dict[str, Any]]],
    traj_problems: list[str],
    *,
    window: int,
    name: str,
    problem_aware: bool,
    eval_problem_names: Sequence[str],
    epochs: int = 200,
    val_fraction: float = 0.2,
    seed: int = 0,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train one multi-head controller on the union training set.

    For ``problem_aware=True`` one :class:`ProblemAwareEncoder` per problem
    (the union of ``eval_problem_names`` and the training trajectory
    problems) is fitted over ALL training trajectories and ALL problems, so
    the state and problem z-score statistics are shared across encoders and
    only the constant problem-descriptor block differs. Each training
    trajectory then contributes samples encoded with its own problem's
    encoder, and the per-problem sample blocks are concatenated (trajectory
    ids are offset to stay globally unique for the split). For
    ``problem_aware=False`` a single plain :class:`StateEncoder` is fitted
    and used for every trajectory.

    The three targets are stacked into one matrix so the trajectory-disjoint
    :func:`controller.dataset.train_val_split` applies unchanged.

    Args:
        trajectories: Training trajectories (generation-sorted).
        traj_problems: Problem name of each trajectory, parallel to
            ``trajectories``.
        window: History window in generations.
        name: Arm/controller identifier (``"mlp2"`` / ``"mlp2_nopf"``).
        problem_aware: Whether to use problem-aware encodings.
        eval_problem_names: Problems the controller will be evaluated on.
        epochs: Fixed number of training epochs.
        val_fraction: Fraction of trajectories held out for validation.
        seed: Seed for the controller init, shuffling, and the split.
        lr: Adam learning rate.
        batch_size: Training batch size.
        verbose: Whether the controller prints training progress.

    Returns:
        Dict with ``name``, ``problem_aware``, ``window``, ``input_dim``,
        the fitted ``controller``, ``encoders`` (per-problem dict) or
        ``encoder`` (single), the training ``history``, the data ``split``,
        and ``n_samples``.

    Raises:
        ValueError: If the dataset yields no usable samples.
    """
    encoders: dict[str, ProblemAwareEncoder] | None = None
    encoder: StateEncoder | None = None
    if problem_aware:
        problem_names = sorted(set(eval_problem_names) | set(traj_problems))
        problem_objs = {n: get_problem(n) for n in problem_names}
        all_problems = [problem_objs[n] for n in problem_names]
        encoders = {
            n: ProblemAwareEncoder(window, problem_objs[n]).fit(trajectories, all_problems)
            for n in problem_names
        }
        x_parts: list[np.ndarray] = []
        yop_parts: list[np.ndarray] = []
        ypm_parts: list[np.ndarray] = []
        yexpl_parts: list[np.ndarray] = []
        w_parts: list[np.ndarray] = []
        id_parts: list[np.ndarray] = []
        offset = 0
        for n in problem_names:
            idx = [i for i, p in enumerate(traj_problems) if p == n]
            if not idx:
                continue
            group = [trajectories[i] for i in idx]
            X_g, yop_g, ypm_g, yexpl_g, w_g, ids_g = build_multihead_samples(
                group, encoders[n], window
            )
            x_parts.append(X_g)
            yop_parts.append(yop_g)
            ypm_parts.append(ypm_g)
            yexpl_parts.append(yexpl_g)
            w_parts.append(w_g)
            id_parts.append(ids_g + offset)
            offset += len(group)
        X = np.vstack(x_parts)
        y_op = np.concatenate(yop_parts)
        y_logpm = np.concatenate(ypm_parts)
        y_logexpl = np.concatenate(yexpl_parts)
        w = np.concatenate(w_parts)
        traj_ids = np.concatenate(id_parts)
        input_dim = encoders[problem_names[0]].dim
    else:
        encoder = StateEncoder(window).fit(trajectories)
        X, y_op, y_logpm, y_logexpl, w, traj_ids = build_multihead_samples(
            trajectories, encoder, window
        )
        input_dim = encoder.dim

    if X.shape[0] == 0:
        raise ValueError("training trajectories produced zero multi-head samples")

    targets = np.column_stack([y_op.astype(float), y_logpm, y_logexpl])
    split = train_val_split(X, targets, w, traj_ids, val_fraction=val_fraction, seed=seed)
    has_val = split["X_val"].shape[0] > 0

    controller = MultiHeadController(input_dim=input_dim, seed=seed, lr=lr, name=name)
    history = controller.fit(
        np.asarray(split["X_train"], dtype=float),
        np.asarray(split["y_train"][:, 0], dtype=np.int64),
        np.asarray(split["y_train"][:, 1], dtype=float),
        np.asarray(split["y_train"][:, 2], dtype=float),
        sample_weight=np.asarray(split["w_train"], dtype=float),
        epochs=epochs,
        batch_size=batch_size,
        X_val=np.asarray(split["X_val"], dtype=float) if has_val else None,
        yop_val=np.asarray(split["y_val"][:, 0], dtype=np.int64) if has_val else None,
        ypm_val=np.asarray(split["y_val"][:, 1], dtype=float) if has_val else None,
        yexpl_val=np.asarray(split["y_val"][:, 2], dtype=float) if has_val else None,
        verbose=verbose,
    )
    return {
        "name": name,
        "problem_aware": bool(problem_aware),
        "window": int(window),
        "input_dim": int(input_dim),
        "controller": controller,
        "encoders": encoders,
        "encoder": encoder,
        "history": history,
        "split": split,
        "n_samples": int(X.shape[0]),
    }


def _action_dynamics(transitions: list[dict[str, Any]]) -> dict[str, float]:
    """Action-dynamics statistics of one recorded run.

    Computed over the controller-chosen actions only (transitions ``t >= 1``;
    generation 0 is the algorithm default, not a controller decision).

    * ``pm_std``: population standard deviation of the mutation probability
      within the run (0.0 for fewer than two actions).
    * ``operator_switch_rate``: fraction of consecutive generation pairs
      whose mutation operator differs (0.0 for fewer than two actions).
    * ``pm_mean``: mean mutation probability within the run.

    Args:
        transitions: Recorded transitions of one run, ordered by generation.

    Returns:
        Dict with ``pm_std``, ``operator_switch_rate``, and ``pm_mean``.
    """
    actions = [t["action"] for t in transitions[1:]]
    pms = np.asarray([float(a["mutation_probability"]) for a in actions], dtype=float)
    operators = [str(a["mutation_operator"]) for a in actions]
    pm_std = float(pms.std()) if pms.size > 1 else 0.0
    pm_mean = float(pms.mean()) if pms.size else 0.0
    switches = sum(
        1 for prev, cur in zip(operators, operators[1:]) if prev != cur
    )
    switch_rate = float(switches) / (len(operators) - 1) if len(operators) > 1 else 0.0
    return {
        "pm_std": pm_std,
        "operator_switch_rate": switch_rate,
        "pm_mean": pm_mean,
    }


def _build_eval_config(
    *,
    arm: str,
    problem_name: str,
    n_vars: int,
    pop_size: int,
    generations: int,
    operators: OperatorConfig,
    pm_range: tuple[float, float],
    controller_name: str,
    window: int,
    ref_point: np.ndarray,
    n_reference_points: int,
) -> dict[str, Any]:
    """Build the configuration dict stored inside each eval trajectory JSON.

    The config fully determines the run (problem, arm, resolved operator
    defaults, controller identity/window, deployment pm and exploration
    ranges, reference settings) and carries a UTC timestamp, satisfying the
    EvoController experiment-recording rule (AGENTS.md Rule 2).
    """
    return {
        "problem": problem_name,
        "n_vars": int(n_vars),
        "algorithm": "nsga2",
        "arm": arm,
        "controller": controller_name,
        "window": int(window),
        "action_space": "full",
        "pop_size": int(pop_size),
        "generations": int(generations),
        "operators": {
            "crossover_operator": operators.crossover_operator,
            "crossover_prob": float(operators.crossover_prob),
            "mutation_operator": operators.mutation_operator,
            "mutation_probability": 1.0 / float(n_vars),
            "eta_c": float(operators.eta_c),
            "eta_m": float(operators.eta_m),
            "gaussian_sigma": float(operators.gaussian_sigma),
        },
        "pm_min": float(pm_range[0]),
        "pm_max": float(pm_range[1]),
        "exploration_ranges": {
            "polynomial_eta_m": [
                float(POLYNOMIAL_EXPLORATION_RANGE[0]),
                float(POLYNOMIAL_EXPLORATION_RANGE[1]),
            ],
            "gaussian_sigma": [
                float(GAUSSIAN_EXPLORATION_RANGE[0]),
                float(GAUSSIAN_EXPLORATION_RANGE[1]),
            ],
        },
        "ref_point": [float(ref_point[0]), float(ref_point[1])],
        "n_reference_points": int(n_reference_points),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_multihead_arm(
    arm: str,
    problem_name: str,
    seed: int,
    *,
    generations: int,
    pop_size: int,
    n_reference_points: int = 200,
    ref_point: np.ndarray | None = None,
    pm_mult_range: tuple[float, float] = DEFAULT_PM_MULT_RANGE,
    controller: MultiHeadController,
    encoder: StateEncoder | ProblemAwareEncoder,
    window: int,
    trajectories_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one evaluation trajectory of a multi-head controller arm.

    At each generation ``t >= 1`` the controller observes the merged state
    dicts of generations ``[t - window, t)`` and predicts the full action
    triple via ``predict_action(history, encoder, pm_min, pm_max)`` with
    ``pm_min/pm_max = pm_mult_range / n_vars``; the triple is injected via
    ``NSGAII.step(mutation_prob=..., mutation_operator=...,
    exploration_strength=...)``. The recorded action is the value actually
    used (``algorithm.current_action()`` after the step).

    Args:
        arm: Arm identifier (``"mlp2"`` or ``"mlp2_nopf"``).
        problem_name: Benchmark identifier accepted by ``get_problem``.
        seed: Evaluation random seed.
        generations: Number of NSGA-II generations to execute.
        pop_size: Population size.
        n_reference_points: Points sampled from the true Pareto front for IGD.
        ref_point: Hypervolume reference point, shape ``(2,)``; defaults to
            ``(1.1, 1.1)``.
        pm_mult_range: Multipliers on ``1 / n_vars`` bounding the controller
            mutation-probability output.
        controller: Fitted multi-head controller.
        encoder: Fitted encoder matching the controller's input (per-problem
            :class:`ProblemAwareEncoder` for ``mlp2``, shared
            :class:`StateEncoder` for ``mlp2_nopf``).
        window: History window used to build controller inputs.
        trajectories_dir: If given, the recorded trajectory JSON is written
            to ``{trajectories_dir}/{arm}_{problem}_seed{seed}.json``.

    Returns:
        Per-run summary dict with ``arm``, ``problem``, ``seed``,
        ``final_hv``, ``final_igd``, ``auc_hv``, ``runtime_sec``,
        ``trajectory_file`` (file name or ``None``), and the action-dynamics
        statistics ``pm_std``, ``operator_switch_rate``, ``pm_mean``.
    """
    if ref_point is None:
        ref_point = np.asarray([1.1, 1.1], dtype=float)
    problem = get_problem(problem_name)
    operators = OperatorConfig()
    algorithm = NSGAII(problem, pop_size=pop_size, operators=operators, seed=seed)
    recorder = EvolutionRecorder(
        problem_name=problem.name,
        reference_front=problem.reference_front(n_points=n_reference_points),
        ref_point=ref_point,
    )
    pm_range = (
        float(pm_mult_range[0]) / problem.n_vars,
        float(pm_mult_range[1]) / problem.n_vars,
    )

    start = time.perf_counter()
    algorithm.initialize()
    recorder.record(algorithm.generation, algorithm.nondominated_front(), algorithm.current_action())
    for _ in range(generations):
        history = [merge_state_reward(tr) for tr in recorder.transitions()[-window:]]
        action = controller.predict_action(history, encoder, pm_range[0], pm_range[1])
        algorithm.step(
            mutation_prob=action["mutation_probability"],
            mutation_operator=action["mutation_operator"],
            exploration_strength=action["exploration_strength"],
        )
        recorder.record(algorithm.generation, algorithm.nondominated_front(), algorithm.current_action())
    runtime_sec = time.perf_counter() - start

    transitions = recorder.transitions()
    dynamics = _action_dynamics(transitions)
    trajectory_file: str | None = None
    if trajectories_dir is not None:
        config = _build_eval_config(
            arm=arm,
            problem_name=problem.name,
            n_vars=problem.n_vars,
            pop_size=pop_size,
            generations=generations,
            operators=operators,
            pm_range=pm_range,
            controller_name=str(getattr(controller, "name", arm)),
            window=window,
            ref_point=ref_point,
            n_reference_points=n_reference_points,
        )
        out_path = Path(trajectories_dir) / f"{arm}_{problem.name}_seed{seed}.json"
        recorder.save(out_path, config=config, seed=seed, runtime_sec=runtime_sec)
        trajectory_file = out_path.name

    return {
        "arm": arm,
        "problem": problem.name,
        "seed": int(seed),
        "final_hv": float(transitions[-1]["state"]["hv"]),
        "final_igd": float(transitions[-1]["state"]["igd"]),
        "auc_hv": _auc_hv(transitions, generations),
        "runtime_sec": float(runtime_sec),
        "trajectory_file": trajectory_file,
        "pm_std": dynamics["pm_std"],
        "operator_switch_rate": dynamics["operator_switch_rate"],
        "pm_mean": dynamics["pm_mean"],
    }


def _load_phase1_runs(
    phase1_results_path: str | Path,
    problems: Sequence[str],
    eval_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Extract per-seed metrics of the reused Phase-1 arms.

    Reads ``results[arm][problem]["seeds"][str(seed)]`` of the Phase-1
    ``results.json`` for every arm in :data:`REUSED_ARMS` and converts each
    record into the same per-run summary shape the freshly evaluated arms
    produce (``trajectory_file`` refers to the Phase-1 output directory and
    is kept for provenance only).

    Args:
        phase1_results_path: Path to the Phase-1 ``results.json``.
        problems: Normalized problem names to extract.
        eval_seeds: Evaluation seeds; only these seeds are extracted so the
            reused arms pair seed-wise with the newly evaluated arms.

    Returns:
        List of per-run summary dicts for the reused arms.

    Raises:
        FileNotFoundError: If the Phase-1 results file does not exist.
    """
    path = Path(phase1_results_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase-1 results not found: {path}; required for the reused arms "
            f"{REUSED_ARMS}"
        )
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    results = payload.get("results", {})
    runs: list[dict[str, Any]] = []
    for arm in REUSED_ARMS:
        for problem in problems:
            entry = results.get(arm, {}).get(problem)
            if entry is None:
                print(f"[warn] Phase-1 results lack arm={arm} problem={problem}; skipped")
                continue
            seeds_map = entry.get("seeds", {})
            for seed in eval_seeds:
                record = seeds_map.get(str(seed))
                if record is None:
                    print(
                        f"[warn] Phase-1 results lack arm={arm} problem={problem} "
                        f"seed={seed}; skipped"
                    )
                    continue
                runs.append(
                    {
                        "arm": arm,
                        "problem": problem,
                        "seed": int(seed),
                        "final_hv": float(record["final_hv"]),
                        "final_igd": float(record["final_igd"]),
                        "auc_hv": float(record["auc_hv"]),
                        "runtime_sec": float(record.get("runtime_sec", 0.0)),
                        "trajectory_file": record.get("trajectory_file"),
                    }
                )
    return runs


def _wilcoxon_report_vs(
    runs: list[dict[str, Any]],
    problems: Sequence[str],
    eval_seeds: Sequence[int],
    baseline: str,
) -> dict[str, Any]:
    """One-sided Wilcoxon signed-rank tests of every arm vs ``baseline``.

    Tests ``final_hv(arm) > final_hv(baseline)`` per problem on the seeds
    available for BOTH arms (paired, ordered by ``eval_seeds``), using
    ``zero_method="zsplit"`` and ``alternative="greater"`` (see
    :func:`experiments.run_phase1._wilcoxon_greater`).

    Args:
        runs: Per-run summaries of all arms (new and reused).
        problems: Normalized problem names to report.
        eval_seeds: Evaluation seeds, defining the paired ordering.
        baseline: Baseline arm identifier (``"fixed"`` or ``"constant"``).

    Returns:
        Nested dict ``wilcoxon[arm][problem]`` with ``p_value``,
        ``statistic``, ``n_seeds``, and the test settings, for every arm
        except the baseline itself.
    """
    report: dict[str, Any] = {}
    for arm in ARMS:
        if arm == baseline:
            continue
        report[arm] = {}
        for problem in problems:
            arm_hv = {
                r["seed"]: r["final_hv"]
                for r in runs
                if r["arm"] == arm and r["problem"] == problem
            }
            base_hv = {
                r["seed"]: r["final_hv"]
                for r in runs
                if r["arm"] == baseline and r["problem"] == problem
            }
            paired = [s for s in eval_seeds if s in arm_hv and s in base_hv]
            p_value, statistic = _wilcoxon_greater(
                [arm_hv[s] for s in paired], [base_hv[s] for s in paired]
            )
            report[arm][problem] = {
                "p_value": p_value,
                "statistic": statistic,
                "n_seeds": len(paired),
                "metric": "final_hv",
                "vs": baseline,
                "zero_method": "zsplit",
                "alternative": "greater",
            }
    return report


def _action_dynamics_report(
    runs: list[dict[str, Any]],
    problems: Sequence[str],
    eval_seeds: Sequence[int],
) -> dict[str, Any]:
    """Aggregate action-dynamics statistics per new arm x problem.

    Cross-problem and cross-generation evidence for the Phase-1.5 research
    question (state-dependent decisions vs a global constant): the mean
    within-run pm standard deviation, the mean within-run operator switch
    rate, and the mean pm (per problem, so behavioral differences across
    problems are visible). Per-seed raw values are retained.
    """
    report: dict[str, Any] = {}
    for arm in NEW_ARMS:
        report[arm] = {}
        for problem in problems:
            by_seed = {r["seed"]: r for r in runs if r["arm"] == arm and r["problem"] == problem}
            ordered = [by_seed[s] for s in eval_seeds if s in by_seed]
            pm_std_mean = (
                float(np.mean([r["pm_std"] for r in ordered])) if ordered else 0.0
            )
            switch_mean = (
                float(np.mean([r["operator_switch_rate"] for r in ordered]))
                if ordered
                else 0.0
            )
            pm_mean = float(np.mean([r["pm_mean"] for r in ordered])) if ordered else 0.0
            report[arm][problem] = {
                "n_seeds": len(ordered),
                "mean_within_run_pm_std": pm_std_mean,
                "mean_operator_switch_rate": switch_mean,
                "mean_pm": pm_mean,
                "seeds": {
                    str(r["seed"]): {
                        "pm_std": float(r["pm_std"]),
                        "operator_switch_rate": float(r["operator_switch_rate"]),
                        "pm_mean": float(r["pm_mean"]),
                    }
                    for r in ordered
                },
            }
    return report


def _training_entry(trained: dict[str, Any], epochs: int) -> dict[str, Any]:
    """Assemble one arm's training summary stored in ``results.json``."""
    history = trained["history"]
    train_loss = [float(v) for v in history.get("train_loss", [])]
    val_loss = [float(v) for v in history.get("val_loss", [])]
    split = trained["split"]
    return {
        "kind": "multihead_mlp",
        "problem_aware": bool(trained["problem_aware"]),
        "window": int(trained["window"]),
        "input_dim": int(trained["input_dim"]),
        "n_train_samples": int(len(split["X_train"])),
        "n_val_samples": int(len(split["X_val"])),
        "epochs": int(epochs),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "final_train_loss": train_loss[-1] if train_loss else None,
        "final_val_loss": val_loss[-1] if val_loss else None,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full Phase-1.5 experiment and write ``results.json``.

    Trains the ``mlp2`` (problem-aware) and ``mlp2_nopf`` (no problem
    features) multi-head controllers on the union of ``args.train_dirs``,
    evaluates them on the (problem, eval seed) grid with trajectories
    recorded, reuses the Phase-1 arms ``fixed``/``constant``/``mlp_w10``
    from ``args.phase1_results``, aggregates the metrics, computes Wilcoxon
    p-values vs both baselines, summarizes the controller action dynamics,
    and writes the payload to ``{args.out_dir}/results.json``.

    Args:
        args: Parsed arguments as produced by :func:`parse_args`.

    Returns:
        The results payload exactly as written to ``results.json``.
    """
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    trajectories_dir = out_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    ref_point = np.asarray(args.ref_point, dtype=float)
    pm_mult_range = (float(args.pm_mult_range[0]), float(args.pm_mult_range[1]))
    problems = [get_problem(name).name for name in args.problems]
    eval_seeds = [int(s) for s in args.eval_seeds]
    window = int(args.window)

    print(
        f"Phase-1.5 experiment: train_dirs={list(map(str, args.train_dirs))}, "
        f"problems={problems}, eval_seeds={eval_seeds}, pop={args.pop_size}, "
        f"gens={args.generations}, epochs={args.epochs}, window={window}; "
        f"out_dir={out_dir}"
    )

    # --- (1) load the union training set (with per-trajectory problems) ---
    trajectories, traj_problems = _load_training_trajectories(args.train_dirs)
    per_problem_counts = {
        name: traj_problems.count(name) for name in sorted(set(traj_problems))
    }
    print(f"[data] {len(trajectories)} training trajectories: {per_problem_counts}")

    # --- (2) train the two new arms ---
    trained: dict[str, dict[str, Any]] = {}
    for arm, problem_aware in ((ARM_MLP2, True), (ARM_MLP2_NOPF, False)):
        trained[arm] = train_multihead_controller(
            trajectories,
            traj_problems,
            window=window,
            name=arm,
            problem_aware=problem_aware,
            eval_problem_names=problems,
            epochs=args.epochs,
            val_fraction=args.val_fraction,
            seed=args.train_seed,
            lr=args.lr,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        print(
            f"[trained] {arm}: {trained[arm]['n_samples']} samples, "
            f"input_dim={trained[arm]['input_dim']}, "
            f"final_train_loss={trained[arm]['history']['train_loss'][-1]:.6f}"
        )

    # --- (3) evaluate the new arms on the (problem, seed) grid ---
    runs: list[dict[str, Any]] = []
    for arm in NEW_ARMS:
        spec = trained[arm]
        for problem_name in problems:
            encoder: StateEncoder | ProblemAwareEncoder
            if spec["problem_aware"]:
                assert spec["encoders"] is not None
                encoder = spec["encoders"][problem_name]
            else:
                assert spec["encoder"] is not None
                encoder = spec["encoder"]
            for seed in eval_seeds:
                summary = evaluate_multihead_arm(
                    arm,
                    problem_name,
                    seed,
                    generations=args.generations,
                    pop_size=args.pop_size,
                    n_reference_points=args.n_reference_points,
                    ref_point=ref_point,
                    pm_mult_range=pm_mult_range,
                    controller=spec["controller"],
                    encoder=encoder,
                    window=window,
                    trajectories_dir=trajectories_dir,
                )
                runs.append(summary)
                print(
                    f"[eval] {arm} {problem_name} seed={seed} "
                    f"hv={summary['final_hv']:.6f} igd={summary['final_igd']:.6f} "
                    f"({summary['runtime_sec']:.2f}s)"
                )

    # --- (4) reuse the Phase-1 arms, aggregate, test, and persist ---
    phase1_runs = _load_phase1_runs(args.phase1_results, problems, eval_seeds)
    all_runs = runs + phase1_runs
    results = _aggregate_runs(all_runs, problems, list(ARMS), eval_seeds)
    wilcoxon_vs_fixed = _wilcoxon_report_vs(all_runs, problems, eval_seeds, BASELINE_FIXED)
    wilcoxon_vs_constant = _wilcoxon_report_vs(
        all_runs, problems, eval_seeds, BASELINE_CONSTANT
    )
    action_dynamics = _action_dynamics_report(runs, problems, eval_seeds)
    wall_time_sec = time.perf_counter() - started

    payload: dict[str, Any] = {
        "config": {
            "phase": 1.5,
            "train_dirs": [str(d) for d in args.train_dirs],
            "phase1_results": str(args.phase1_results),
            "out_dir": str(args.out_dir),
            "problems": problems,
            "eval_seeds": eval_seeds,
            "pop_size": int(args.pop_size),
            "generations": int(args.generations),
            "window": window,
            "action_space": "full",
            "pm_mult_range": [float(pm_mult_range[0]), float(pm_mult_range[1])],
            "exploration_ranges": {
                "polynomial_eta_m": [
                    float(POLYNOMIAL_EXPLORATION_RANGE[0]),
                    float(POLYNOMIAL_EXPLORATION_RANGE[1]),
                ],
                "gaussian_sigma": [
                    float(GAUSSIAN_EXPLORATION_RANGE[0]),
                    float(GAUSSIAN_EXPLORATION_RANGE[1]),
                ],
            },
            "ref_point": [float(ref_point[0]), float(ref_point[1])],
            "n_reference_points": int(args.n_reference_points),
            "training": {
                "epochs": int(args.epochs),
                "val_fraction": float(args.val_fraction),
                "train_seed": int(args.train_seed),
                "lr": float(args.lr),
                "batch_size": int(args.batch_size),
                "hidden_dims": [64, 64],
                "n_train_trajectories": len(trajectories),
                "train_trajectories_per_problem": per_problem_counts,
                "n_samples": {arm: int(trained[arm]["n_samples"]) for arm in NEW_ARMS},
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "arms": list(ARMS),
        "new_arms": list(NEW_ARMS),
        "reused_arms": list(REUSED_ARMS),
        "training": {arm: _training_entry(trained[arm], args.epochs) for arm in NEW_ARMS},
        "results": results,
        "wilcoxon_vs_fixed": wilcoxon_vs_fixed,
        "wilcoxon_vs_constant": wilcoxon_vs_constant,
        "action_dynamics": action_dynamics,
        "wall_time_sec": float(wall_time_sec),
    }

    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[done] wrote results -> {results_path} ({wall_time_sec:.2f}s wall)")
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point: parse arguments and run the Phase-1.5 experiment.

    Args:
        argv: Optional argument list; ``None`` reads ``sys.argv``.

    Returns:
        The results payload as written to ``results.json``.
    """
    return run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
