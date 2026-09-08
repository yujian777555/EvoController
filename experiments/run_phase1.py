"""Phase 1: train MLP evolution controllers and benchmark them against fixed NSGA-II.

This runner implements the Phase-1 experiment of the EvoController roadmap
(``docs/PHASE1_PLAN.md``): it verifies whether evolution trajectories contain
learnable information by training MLP controllers that map evolution history
to a per-generation mutation probability, and comparing them against the
Phase-0 fixed-policy baseline.

Arms compared:

* ``fixed`` — Phase-0 NSGA-II baseline (constant ``1 / n_vars`` mutation
  probability, plain ``step()``).
* ``mlp_w10`` — MLP controller over a history window of 10 transitions.
* ``mlp_w1`` — MLP controller over a history window of 1 (current-state-only
  ablation).
* ``constant`` — :class:`controller.ConstantController`, the no-history
  ablation: a single reward-weighted average mutation probability.

Deployment semantics (shared with the dataset builder): at generation
``t >= 1`` the controller observes the merged state dicts of generations
``[t - window, t)`` (i.e. up to and including generation ``t - 1``) and emits
the mutation probability used for the step *into* generation ``t``. Generation
0 is recorded with the algorithm's default action, exactly as in Phase 0.

Outputs (under ``--out-dir``):

* ``trajectories/{arm}_{problem}_seed{seed}.json`` — one trajectory per
  (arm, problem, eval seed), in the Phase-0 recorder schema.
* ``results.json`` — per arm x problem aggregates (mean/std of final HV,
  final IGD, anytime AUC-HV, runtime), per-seed raw values, Wilcoxon
  signed-rank p-values of each controller arm vs ``fixed`` on final HV,
  training losses, the full configuration, and the wall time.

Example:
    ``python experiments/run_phase1.py --train-dir results/trajectory_random``
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import wilcoxon

if __package__ in (None, ""):
    # Allow ``python experiments/run_phase1.py`` from the repo root: the script
    # directory (not the repo root) is on sys.path in that mode.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms import NSGAII, OperatorConfig
from benchmarks import get_problem
from controller import (
    ConstantController,
    MLPController,
    StateEncoder,
    build_supervised_samples,
    load_trajectories,
    train_val_split,
)
from controller.dataset import merge_state_reward
from trajectory import EvolutionRecorder

#: Default evaluation problem grid (all Phase-0 ZDT benchmarks).
DEFAULT_PROBLEMS: tuple[str, ...] = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")
#: Default evaluation seeds; disjoint from the Phase-0 training seeds {0..4}.
DEFAULT_EVAL_SEEDS: tuple[int, ...] = (100, 101, 102, 103, 104)
#: Default directory with the random-policy training trajectories.
DEFAULT_TRAIN_DIR = "results/trajectory_random"
#: Default output directory for Phase-1 results.
DEFAULT_OUT_DIR = "results/phase1"

#: Arm identifiers, in report order.
ARM_FIXED = "fixed"
ARM_MLP_W10 = "mlp_w10"
ARM_MLP_W1 = "mlp_w1"
ARM_CONSTANT = "constant"
ARMS: tuple[str, ...] = (ARM_FIXED, ARM_MLP_W10, ARM_MLP_W1, ARM_CONSTANT)

#: History windows of the MLP arms.
WINDOW_MLP_W10 = 10
WINDOW_MLP_W1 = 1

#: True when the installed NSGA-II already implements the Phase-1 contract
#: ``step(mutation_prob=...)``; Phase-0 builds lack the keyword.
_STEP_SUPPORTS_MUTATION_PROB = "mutation_prob" in inspect.signature(NSGAII.step).parameters


def _step_with_mutation_prob(algorithm: NSGAII, mutation_prob: float) -> None:
    """Advance one generation using an explicit per-variable mutation probability.

    The Phase-1 deployment contract is ``NSGAII.step(mutation_prob=pm)``.
    Phase-0 ``NSGAII`` predates that keyword; when it is missing, the resolved
    probability held by the instance (the same value ``step()`` reads during
    polynomial mutation and ``current_action()`` reports) is overridden
    directly, which is semantically identical. The capability is detected at
    import time, so an upgraded ``NSGAII`` is used automatically.

    Args:
        algorithm: Initialized NSGA-II instance.
        mutation_prob: Effective per-variable mutation probability for this
            single step (already clipped to the deployment range).
    """
    if _STEP_SUPPORTS_MUTATION_PROB:
        algorithm.step(mutation_prob=float(mutation_prob))  # type: ignore[call-arg]
    else:
        algorithm._mutation_prob = float(mutation_prob)
        algorithm.step()


def _controller_history(transitions: list[dict], window: int) -> list[dict]:
    """Build the merged-state history for the next controller action.

    Takes the last ``window`` recorded transitions (oldest first) and merges
    each transition's state and reward dicts exactly as the training dataset
    builder does, so deployment inputs match training inputs.

    Args:
        transitions: Recorded transitions so far (generations ``0..t-1``).
        window: Maximum history length; shorter histories are zero-padded by
            the :class:`controller.StateEncoder`.

    Returns:
        Merged state dicts with the six ``STATE_FEATURES`` keys, oldest first.
    """
    return [merge_state_reward(tr) for tr in transitions[-window:]]


def _auc_hv(transitions: list[dict], generations: int) -> float:
    """Anytime performance: trapezoidal integral of HV over generations.

    The integral over the recorded generation axis is normalized by the
    number of generations, yielding the mean hypervolume maintained during
    the run (generation 0 included).

    Args:
        transitions: Recorded transitions of one run, ordered by generation.
        generations: Number of NSGA-II generations executed (the divisor).

    Returns:
        Normalized AUC-HV; equals the final HV for degenerate one-transition
        runs.
    """
    hv_series = np.asarray([tr["state"]["hv"] for tr in transitions], dtype=float)
    gen_series = np.asarray([tr["generation"] for tr in transitions], dtype=float)
    if len(hv_series) < 2 or generations <= 0:
        return float(hv_series[-1]) if len(hv_series) else 0.0
    integral = float(np.sum(0.5 * (hv_series[:-1] + hv_series[1:]) * np.diff(gen_series)))
    return integral / float(generations)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Phase-1 experiment.

    Args:
        argv: Argument list to parse; ``None`` reads ``sys.argv``.

    Returns:
        Parsed namespace with training, evaluation, and output settings.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Phase-1 experiment: train MLP evolution controllers on random-policy "
            "trajectories and evaluate them against fixed NSGA-II on ZDT benchmarks."
        )
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=DEFAULT_TRAIN_DIR,
        help="Directory with random-policy training trajectory JSONs (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for results.json and eval trajectories (default: %(default)s).",
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
        default=300,
        help="Fixed number of controller training epochs (default: %(default)s).",
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
        help="Adam learning rate for the MLP controllers (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size for the MLP controllers (default: %(default)s).",
    )
    parser.add_argument(
        "--pm-mult-range",
        nargs=2,
        type=float,
        default=[0.5, 5.0],
        metavar=("PM_MULT_LO", "PM_MULT_HI"),
        help=(
            "Multipliers on the default 1/n_vars mutation probability defining the "
            "controller output range [lo/n_vars, hi/n_vars] (default: %(default)s)."
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


def train_controllers(
    train_dir: str | Path,
    window: int,
    *,
    epochs: int = 300,
    val_fraction: float = 0.2,
    seed: int = 0,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train one MLP controller on the random-policy trajectory dataset.

    Loads the training trajectories, fits a z-score :class:`StateEncoder` for
    the given history window, builds the supervised sample set, splits it by
    trajectory id (no leakage between train and validation), and trains an
    :class:`MLPController` with weighted MSE for a fixed number of epochs.

    Args:
        train_dir: Directory with training trajectory JSONs.
        window: History window length (number of past transitions) of the
            controller input.
        epochs: Fixed number of training epochs.
        val_fraction: Fraction of trajectories held out for validation.
        seed: Seed for the controller init, shuffling, and the split.
        lr: Adam learning rate.
        batch_size: Training batch size.
        verbose: Whether the controller prints training progress.

    Returns:
        Dict with ``window``, the fitted ``controller`` and ``encoder``, the
        training ``history`` (``{"train_loss", "val_loss"}``), the data
        ``split``, ``n_trajectories``, and ``n_samples``.
    """
    trajectories = load_trajectories(train_dir)
    if not trajectories:
        raise ValueError(f"no training trajectories found in {train_dir!s}")

    encoder = StateEncoder(window).fit(trajectories)
    X, y, w, traj_ids = build_supervised_samples(trajectories, encoder, window)
    split = train_val_split(X, y, w, traj_ids, val_fraction=val_fraction, seed=seed)

    X_val = np.asarray(split["X_val"], dtype=float)
    y_val = np.asarray(split["y_val"], dtype=float)
    has_val = X_val.shape[0] > 0

    controller = MLPController(input_dim=encoder.dim, seed=seed, lr=lr)
    history = controller.fit(
        np.asarray(split["X_train"], dtype=float),
        np.asarray(split["y_train"], dtype=float),
        sample_weight=np.asarray(split["w_train"], dtype=float),
        epochs=epochs,
        batch_size=batch_size,
        X_val=X_val if has_val else None,
        y_val=y_val if has_val else None,
        verbose=verbose,
    )
    return {
        "window": int(window),
        "controller": controller,
        "encoder": encoder,
        "history": history,
        "split": split,
        "n_trajectories": len(trajectories),
        "n_samples": int(X.shape[0]),
    }


def _build_eval_config(
    *,
    arm: str,
    problem_name: str,
    n_vars: int,
    pop_size: int,
    generations: int,
    operators: OperatorConfig,
    mutation_probability: float,
    pm_range: tuple[float, float] | None,
    controller_name: str | None,
    window: int | None,
    ref_point: np.ndarray,
    n_reference_points: int,
) -> dict[str, Any]:
    """Build the configuration dict stored inside each eval trajectory JSON.

    The config fully determines the run (problem, arm, resolved operators,
    controller identity/window, deployment pm range, reference settings) and
    carries a UTC timestamp, satisfying the EvoController experiment-recording
    rule (AGENTS.md Rule 2).
    """
    return {
        "problem": problem_name,
        "n_vars": int(n_vars),
        "algorithm": "nsga2",
        "arm": arm,
        "controller": controller_name,
        "window": window,
        "pop_size": int(pop_size),
        "generations": int(generations),
        "operators": {
            "crossover_operator": operators.crossover_operator,
            "crossover_prob": float(operators.crossover_prob),
            "mutation_operator": operators.mutation_operator,
            "mutation_probability": float(mutation_probability),
            "eta_c": float(operators.eta_c),
            "eta_m": float(operators.eta_m),
        },
        "pm_min": None if pm_range is None else float(pm_range[0]),
        "pm_max": None if pm_range is None else float(pm_range[1]),
        "ref_point": [float(ref_point[0]), float(ref_point[1])],
        "n_reference_points": int(n_reference_points),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_arm(
    arm: str,
    problem_name: str,
    seed: int,
    *,
    generations: int,
    pop_size: int,
    n_reference_points: int = 200,
    ref_point: np.ndarray | None = None,
    pm_mult_range: tuple[float, float] = (0.5, 5.0),
    controller: MLPController | ConstantController | None = None,
    encoder: StateEncoder | None = None,
    window: int = 1,
    trajectories_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one evaluation trajectory of a single arm on one problem/seed.

    For ``arm == "fixed"`` (``controller=None``) the run is the Phase-0
    baseline: constant default mutation probability, plain ``step()``. For
    controller arms, at each generation ``t >= 1`` the controller observes the
    merged state dicts of generations ``[t - window, t)`` and sets the mutation
    probability for the step into generation ``t`` via
    ``predict_action(history, encoder, pm_min, pm_max)``, where
    ``pm_min/pm_max = pm_mult_range / n_vars``.

    Args:
        arm: Arm identifier (``"fixed"``, ``"mlp_w10"``, ``"mlp_w1"``,
            ``"constant"``).
        problem_name: Benchmark identifier accepted by ``get_problem``.
        seed: Evaluation random seed.
        generations: Number of NSGA-II generations to execute.
        pop_size: Population size.
        n_reference_points: Points sampled from the true Pareto front for IGD.
        ref_point: Hypervolume reference point, shape ``(2,)``; defaults to
            ``(1.1, 1.1)``.
        pm_mult_range: Multipliers on ``1 / n_vars`` bounding the controller
            output range.
        controller: Fitted controller for non-fixed arms; ``None`` for
            ``"fixed"``.
        encoder: Fitted :class:`StateEncoder` matching the controller's
            window; ``None`` for ``"fixed"``.
        window: History window used to build controller inputs.
        trajectories_dir: If given, the recorded trajectory JSON is written to
            ``{trajectories_dir}/{arm}_{problem}_seed{seed}.json``.

    Returns:
        Per-run summary dict with ``arm``, ``problem``, ``seed``,
        ``final_hv``, ``final_igd``, ``auc_hv``, ``runtime_sec``, and
        ``trajectory_file`` (file name or ``None`` when not saved).
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

    pm_base = 1.0 / problem.n_vars
    pm_range: tuple[float, float] | None = None
    if controller is not None:
        pm_range = (float(pm_mult_range[0]) * pm_base, float(pm_mult_range[1]) * pm_base)

    start = time.perf_counter()
    algorithm.initialize()
    recorder.record(algorithm.generation, algorithm.nondominated_front(), algorithm.current_action())
    for _ in range(generations):
        if controller is not None:
            assert pm_range is not None and encoder is not None
            history = _controller_history(recorder.transitions(), window)
            pm = float(controller.predict_action(history, encoder, pm_range[0], pm_range[1]))
            _step_with_mutation_prob(algorithm, pm)
        else:
            algorithm.step()
        recorder.record(algorithm.generation, algorithm.nondominated_front(), algorithm.current_action())
    runtime_sec = time.perf_counter() - start

    transitions = recorder.transitions()
    trajectory_file: str | None = None
    if trajectories_dir is not None:
        config = _build_eval_config(
            arm=arm,
            problem_name=problem.name,
            n_vars=problem.n_vars,
            pop_size=pop_size,
            generations=generations,
            operators=operators,
            mutation_probability=algorithm.current_action()["mutation_probability"],
            pm_range=pm_range,
            controller_name=None if controller is None else str(getattr(controller, "name", arm)),
            window=None if controller is None else int(window),
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
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, sample std) of ``values``; std is 0.0 for n < 2."""
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean()) if len(arr) else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def _aggregate_runs(
    runs: list[dict[str, Any]],
    problems: list[str],
    arm_names: Sequence[str],
    eval_seeds: Sequence[int],
) -> dict[str, Any]:
    """Aggregate per-run summaries into per arm x problem statistics.

    Produces mean/std over seeds of ``final_hv``, ``final_igd``, and
    ``auc_hv``, the mean runtime, and the raw per-seed values required for
    later analysis.

    Args:
        runs: Per-run summaries as returned by :func:`evaluate_arm`.
        problems: Normalized problem names to report.
        arm_names: Arm identifiers to report, in order.
        eval_seeds: Evaluation seeds, defining the per-seed ordering.

    Returns:
        Nested dict ``results[arm][problem]`` of aggregate statistics plus a
        ``seeds`` mapping with the raw per-seed records.
    """
    results: dict[str, Any] = {}
    for arm in arm_names:
        results[arm] = {}
        for problem in problems:
            by_seed = {r["seed"]: r for r in runs if r["arm"] == arm and r["problem"] == problem}
            ordered = [by_seed[s] for s in eval_seeds if s in by_seed]
            hv_mean, hv_std = _mean_std([r["final_hv"] for r in ordered])
            igd_mean, igd_std = _mean_std([r["final_igd"] for r in ordered])
            auc_mean, auc_std = _mean_std([r["auc_hv"] for r in ordered])
            runtime_mean, _ = _mean_std([r["runtime_sec"] for r in ordered])
            results[arm][problem] = {
                "n_seeds": len(ordered),
                "final_hv_mean": hv_mean,
                "final_hv_std": hv_std,
                "final_igd_mean": igd_mean,
                "final_igd_std": igd_std,
                "auc_hv_mean": auc_mean,
                "auc_hv_std": auc_std,
                "mean_runtime_sec": runtime_mean,
                "seeds": {
                    str(r["seed"]): {
                        "final_hv": float(r["final_hv"]),
                        "final_igd": float(r["final_igd"]),
                        "auc_hv": float(r["auc_hv"]),
                        "runtime_sec": float(r["runtime_sec"]),
                        "trajectory_file": r["trajectory_file"],
                    }
                    for r in ordered
                },
            }
    return results


def _wilcoxon_greater(
    controller_values: list[float], fixed_values: list[float]
) -> tuple[float | None, float | None]:
    """One-sided Wilcoxon signed-rank test: controller HV > fixed HV.

    Uses ``zero_method="zsplit"`` (zero differences are split between the
    positive and negative ranks) and ``alternative="greater"``, so a small
    p-value indicates the controller arm outperforms the fixed baseline on
    final hypervolume.

    Args:
        controller_values: Final HV of the controller arm, aligned by seed.
        fixed_values: Final HV of the fixed arm, aligned by seed.

    Returns:
        ``(p_value, statistic)``, or ``(None, None)`` when the test is not
        computable (e.g. degenerate samples) or non-finite.
    """
    x = np.asarray(controller_values, dtype=float)
    y = np.asarray(fixed_values, dtype=float)
    if x.shape != y.shape or x.size == 0:
        return None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = wilcoxon(x, y, zero_method="zsplit", alternative="greater")
        p_value = float(result.pvalue)
        statistic = float(result.statistic)
    except Exception:
        return None, None
    if not (math.isfinite(p_value) and math.isfinite(statistic)):
        return None, None
    return p_value, statistic


def _wilcoxon_report(
    runs: list[dict[str, Any]],
    problems: list[str],
    eval_seeds: Sequence[int],
) -> dict[str, Any]:
    """Wilcoxon signed-rank p-values of each controller arm vs ``fixed``.

    Args:
        runs: Per-run summaries as returned by :func:`evaluate_arm`.
        problems: Normalized problem names to report.
        eval_seeds: Evaluation seeds, defining the paired ordering.

    Returns:
        Nested dict ``wilcoxon[arm][problem]`` with ``p_value``, ``statistic``,
        ``n_seeds``, and the test settings, for every non-fixed arm.
    """
    report: dict[str, Any] = {}
    for arm in ARMS:
        if arm == ARM_FIXED:
            continue
        report[arm] = {}
        for problem in problems:
            controller_hv = [
                r["final_hv"]
                for r in sorted(runs, key=lambda r: r["seed"])
                if r["arm"] == arm and r["problem"] == problem
            ]
            fixed_hv = [
                r["final_hv"]
                for r in sorted(runs, key=lambda r: r["seed"])
                if r["arm"] == ARM_FIXED and r["problem"] == problem
            ]
            p_value, statistic = _wilcoxon_greater(controller_hv, fixed_hv)
            report[arm][problem] = {
                "p_value": p_value,
                "statistic": statistic,
                "n_seeds": min(len(controller_hv), len(fixed_hv)),
                "metric": "final_hv",
                "vs": ARM_FIXED,
                "zero_method": "zsplit",
                "alternative": "greater",
            }
    return report


def _training_report(
    trained_w10: dict[str, Any],
    trained_w1: dict[str, Any],
    constant_controller: ConstantController,
    epochs: int,
) -> dict[str, Any]:
    """Assemble the per-arm training summary stored in ``results.json``.

    Includes sample counts, full train/validation loss curves, and their final
    values; the constant arm reports its learned log-space constant instead.
    """

    def _mlp_entry(trained: dict[str, Any]) -> dict[str, Any]:
        history = trained["history"]
        train_loss = [float(v) for v in history.get("train_loss", [])]
        val_loss = [float(v) for v in history.get("val_loss", [])]
        split = trained["split"]
        return {
            "kind": "mlp",
            "window": trained["window"],
            "input_dim": int(trained["encoder"].dim),
            "n_train_samples": int(len(split["X_train"])),
            "n_val_samples": int(len(split["X_val"])),
            "epochs": int(epochs),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "final_train_loss": train_loss[-1] if train_loss else None,
            "final_val_loss": val_loss[-1] if val_loss else None,
        }

    try:
        log_pm_constant = float(np.asarray(constant_controller.predict(np.zeros((1, 1))))[0])
    except Exception:
        log_pm_constant = None
    return {
        ARM_MLP_W10: _mlp_entry(trained_w10),
        ARM_MLP_W1: _mlp_entry(trained_w1),
        ARM_CONSTANT: {
            "kind": "constant",
            "window": None,
            "log_pm_constant": log_pm_constant,
        },
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full Phase-1 experiment and write ``results.json``.

    Trains the three controllers on ``args.train_dir``, evaluates all arms on
    the (problem, eval seed) grid with trajectories recorded, aggregates the
    metrics, computes Wilcoxon p-values vs the fixed baseline, and writes the
    summary payload to ``{args.out_dir}/results.json``.

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

    print(
        f"Phase-1 experiment: train_dir={args.train_dir}, problems={problems}, "
        f"eval_seeds={eval_seeds}, pop={args.pop_size}, gens={args.generations}, "
        f"epochs={args.epochs}; out_dir={out_dir}"
    )

    # --- (1)-(3) fit encoders, build samples, train the three controllers ---
    trained_w10 = train_controllers(
        args.train_dir,
        WINDOW_MLP_W10,
        epochs=args.epochs,
        val_fraction=args.val_fraction,
        seed=args.train_seed,
        lr=args.lr,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    trained_w1 = train_controllers(
        args.train_dir,
        WINDOW_MLP_W1,
        epochs=args.epochs,
        val_fraction=args.val_fraction,
        seed=args.train_seed,
        lr=args.lr,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    split_w1 = trained_w1["split"]
    constant_controller = ConstantController().fit(
        np.asarray(split_w1["y_train"], dtype=float),
        sample_weight=np.asarray(split_w1["w_train"], dtype=float),
    )
    print(
        f"[trained] {trained_w10['n_trajectories']} trajectories, "
        f"{trained_w10['n_samples']} samples per window"
    )

    arms: dict[str, dict[str, Any]] = {
        ARM_FIXED: {"controller": None, "encoder": None, "window": 0},
        ARM_MLP_W10: {
            "controller": trained_w10["controller"],
            "encoder": trained_w10["encoder"],
            "window": WINDOW_MLP_W10,
        },
        ARM_MLP_W1: {
            "controller": trained_w1["controller"],
            "encoder": trained_w1["encoder"],
            "window": WINDOW_MLP_W1,
        },
        ARM_CONSTANT: {
            "controller": constant_controller,
            "encoder": trained_w1["encoder"],
            "window": WINDOW_MLP_W1,
        },
    }

    # --- (4) evaluate all arms on the (problem, seed) grid ---
    runs: list[dict[str, Any]] = []
    for arm_name, spec in arms.items():
        for problem_name in problems:
            for seed in eval_seeds:
                summary = evaluate_arm(
                    arm_name,
                    problem_name,
                    seed,
                    generations=args.generations,
                    pop_size=args.pop_size,
                    n_reference_points=args.n_reference_points,
                    ref_point=ref_point,
                    pm_mult_range=pm_mult_range,
                    controller=spec["controller"],
                    encoder=spec["encoder"],
                    window=spec["window"],
                    trajectories_dir=trajectories_dir,
                )
                runs.append(summary)
                print(
                    f"[eval] {arm_name} {problem_name} seed={seed} "
                    f"hv={summary['final_hv']:.6f} igd={summary['final_igd']:.6f} "
                    f"({summary['runtime_sec']:.2f}s)"
                )

    # --- aggregate, test, and persist ---
    results = _aggregate_runs(runs, problems, list(arms), eval_seeds)
    wilcoxon_report = _wilcoxon_report(runs, problems, eval_seeds)
    wall_time_sec = time.perf_counter() - started

    payload: dict[str, Any] = {
        "config": {
            "phase": 1,
            "train_dir": str(args.train_dir),
            "out_dir": str(args.out_dir),
            "problems": problems,
            "eval_seeds": eval_seeds,
            "pop_size": int(args.pop_size),
            "generations": int(args.generations),
            "pm_mult_range": [float(pm_mult_range[0]), float(pm_mult_range[1])],
            "ref_point": [float(ref_point[0]), float(ref_point[1])],
            "n_reference_points": int(args.n_reference_points),
            "training": {
                "epochs": int(args.epochs),
                "val_fraction": float(args.val_fraction),
                "train_seed": int(args.train_seed),
                "lr": float(args.lr),
                "batch_size": int(args.batch_size),
                "hidden_dims": [64, 64],
                "windows": {ARM_MLP_W10: WINDOW_MLP_W10, ARM_MLP_W1: WINDOW_MLP_W1},
                "n_train_trajectories": int(trained_w10["n_trajectories"]),
                "n_samples_per_window": int(trained_w10["n_samples"]),
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "arms": list(arms),
        "training": _training_report(trained_w10, trained_w1, constant_controller, args.epochs),
        "results": results,
        "wilcoxon_vs_fixed": wilcoxon_report,
        "wall_time_sec": float(wall_time_sec),
    }

    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[done] wrote results -> {results_path} ({wall_time_sec:.2f}s wall)")
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point: parse arguments and run the Phase-1 experiment.

    Args:
        argv: Optional argument list; ``None`` reads ``sys.argv``.

    Returns:
        The results payload as written to ``results.json``.
    """
    return run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
