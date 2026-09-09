from __future__ import annotations

"""Phase 1.75: fair closed-loop evaluation with paired held-out seeds.

This runner implements Task 6 of ``docs/PHASE1_75_PLAN.md``: the 9-arm paired
evaluation that tests whether EvoController's gains come from closed-loop,
state-dependent decisions rather than action-space, schedule, scale, or
test-luck confounds.

Arms (all evaluated on identical ``(problem, seed)`` pairs):

1. ``fixed_nsga2`` — Phase-0 fixed policy (plain ``NSGAII.step()``).
2. ``static_full_global`` — one tuned full-action tuple for all problems.
3. ``static_full_per_problem`` — per-problem tuned full-action tuple
   (oracle-style baseline).
4. ``open_loop_global`` — generation-only schedule fitted on training
   trajectories, no population state, no problem identity.
5. ``open_loop_per_problem`` — same, conditioned on problem identity.
6. ``mlp2_closed_loop_absolute`` — multi-head MLP, absolute log-pm target.
7. ``mlp2_closed_loop_normalized`` — multi-head MLP, log mutation-multiplier
   target (``multiplier = pm * n_vars``).
8. ``generation_only_mlp`` — same budget, input ``[gen_norm] + 9 problem
   features`` (no population state).
9. ``state_scrambled_mlp`` — same as the normalized arm, but training history
   rows are shuffled across trajectories within the train split (breaks
   state->action causality, keeps marginals). Deployment still feeds real
   histories.

Deployment semantics (shared with Phase 1/1.5): at generation ``t >= 1`` the
arm observes the merged state+reward dicts of generations ``[t - window, t)``
and chooses the action for the step *into* generation ``t``; generation 0 is
recorded with the algorithm defaults; the recorder stores the action actually
used (``NSGAII.current_action()`` after the step). Deployment mutation
probability bounds are ``[0.25, 8.0] * (1 / n_vars)`` and exploration bounds
are ``eta_m in [2, 50]`` (polynomial) / ``sigma in [0.02, 0.3]`` (Gaussian).

Stages (``--stage``):

* ``thresholds`` — derive per-problem failure thresholds (10th percentile of
  final HV over the fixed-policy training trajectories in ``--fixed-dir``)
  and write ``{out_dir}/failure_thresholds.json``. Uses the training
  distribution only and must run BEFORE ``eval``.
* ``train`` — fit per-problem ``ProblemAwareEncoder`` encoders over ALL
  training trajectories and problems, train the four learned arms with
  identical budget, fit the static and open-loop arms, and persist everything
  under ``{out_dir}/controllers/``.
* ``eval`` — run the (arm, problem, seed) grid (shardable via ``--arms``,
  ``--problems``, ``--seeds``) and write one JSON per run to
  ``{out_dir}/runs/{arm}__{problem}__seed{seed}.json``.
* ``aggregate`` — merge ``runs/*.json`` into ``{out_dir}/results.json``.
* ``all`` — thresholds -> train -> eval -> aggregate.

Example:
    ``python experiments/run_phase1_75.py --stage all``
"""

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    # Allow ``python experiments/run_phase1_75.py`` from the repo root: the
    # script directory (not the repo root) is on sys.path in that mode.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms import NSGAII, OperatorConfig
from benchmarks import get_problem
from controller import (
    MultiHeadController,
    ProblemAwareEncoder,
    build_multihead_samples,
    merge_state_reward,
    train_val_split,
)
from controller.dataset import OPERATOR_DEFAULT_EXPLORATION, OPERATOR_TO_INDEX
from controller.multihead_controller import (
    GAUSSIAN_EXPLORATION_RANGE,
    OPERATOR_CLASSES,
    POLYNOMIAL_EXPLORATION_RANGE,
)
from controller.problem_features import problem_feature_vector
from experiments.run_phase1 import _auc_hv
from trajectory import EvolutionRecorder

# --- Contract imports from parallel Task 1-3 work ---------------------------
# ``load_trajectory_records`` is added to ``controller.dataset`` by a parallel
# task; ``StaticFullController`` / ``OpenLoopScheduleController`` are new
# controller modules. When a module is not available yet, a local fallback
# implementing exactly the documented contract is used, so this runner works
# against both the old and the new controller package.

try:  # pragma: no cover - exercised indirectly once the contract lands
    from controller.dataset import load_trajectory_records as _load_records_impl
except ImportError:  # local fallback, identical contract

    def _load_records_impl(directory: str | Path) -> list[dict[str, Any]]:
        """Fallback for ``controller.dataset.load_trajectory_records``.

        Reads each ``*.json`` (except ``index.json``) and returns one record
        per file: ``{'problem', 'n_vars', 'seed', 'runtime_sec', 'config',
        'transitions'}`` with transitions sorted by generation.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"not a directory: {directory}")
        records: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            if path.name == "index.json":
                continue
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            transitions = payload.get("transitions")
            if transitions is None:
                raise ValueError(f"{path} has no 'transitions' list")
            config = payload.get("config", {})
            records.append(
                {
                    "problem": str(config.get("problem", "")),
                    "n_vars": int(config.get("n_vars", 0)),
                    "seed": int(payload.get("seed", 0)),
                    "runtime_sec": float(payload.get("runtime_sec", 0.0)),
                    "config": config,
                    "transitions": sorted(
                        transitions, key=lambda t: int(t["generation"])
                    ),
                }
            )
        return records


try:  # pragma: no cover - exercised indirectly once the contract lands
    from controller.static_full_controller import StaticFullController
except ImportError:  # local fallback, identical contract

    class StaticFullController:  # type: ignore[no-redef]
        """Fallback matched full-action static controller.

        Contract of ``controller.static_full_controller.StaticFullController``:
        constructed with ``(operator, multiplier, exploration_strength)``;
        ``predict_action(history=None, *, n_vars)`` returns the constant
        three-key action dict with ``mutation_probability = multiplier /
        n_vars``; ``save``/``load`` persist to JSON.
        """

        def __init__(
            self, operator: str, multiplier: float, exploration_strength: float
        ) -> None:
            if operator not in OPERATOR_TO_INDEX:
                raise ValueError(
                    f"unsupported operator {operator!r}; "
                    f"expected one of {sorted(OPERATOR_TO_INDEX)}"
                )
            if not multiplier > 0.0:
                raise ValueError(f"multiplier must be > 0, got {multiplier}")
            if not exploration_strength > 0.0:
                raise ValueError(
                    f"exploration_strength must be > 0, got {exploration_strength}"
                )
            self.operator = str(operator)
            self.multiplier = float(multiplier)
            self.exploration_strength = float(exploration_strength)

        def predict_action(
            self, history: Any = None, *, n_vars: int
        ) -> dict[str, Any]:
            """Return the constant full action (history ignored)."""
            return {
                "mutation_operator": self.operator,
                "mutation_probability": self.multiplier / float(n_vars),
                "exploration_strength": self.exploration_strength,
            }

        def save(self, path: str | Path) -> None:
            """Serialize to JSON; parent directories are created."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "operator": self.operator,
                        "multiplier": self.multiplier,
                        "exploration_strength": self.exploration_strength,
                    },
                    fh,
                    indent=2,
                )

        @classmethod
        def load(cls, path: str | Path) -> "StaticFullController":
            """Load a controller saved with :meth:`save`."""
            with Path(path).open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return cls(
                operator=str(payload["operator"]),
                multiplier=float(payload["multiplier"]),
                exploration_strength=float(payload["exploration_strength"]),
            )


try:  # pragma: no cover - exercised indirectly once the contract lands
    from controller.open_loop_controller import OpenLoopScheduleController
except ImportError:  # local fallback, identical contract

    class OpenLoopScheduleController:  # type: ignore[no-redef]
        """Fallback generation-only open-loop schedule controller.

        Contract of ``controller.open_loop_controller``
        ``OpenLoopScheduleController(per_problem, n_bins=10)``: fitted on
        records ``[{'problem', 'n_vars', 'transitions'}]``; per normalized
        generation bin the schedule stores the majority operator, the median
        mutation multiplier, and the median exploration strength of the
        majority operator. ``predict_action(generation, max_generations,
        problem_name=None)`` returns the three-key action dict of the bin,
        with ``mutation_probability = multiplier / n_vars`` (per-problem
        ``n_vars`` when known, else the median training ``n_vars``).
        """

        def __init__(self, per_problem: bool, n_bins: int = 10) -> None:
            self.per_problem = bool(per_problem)
            self.n_bins = int(n_bins)
            self._schedules: dict[str, list[dict[str, Any]]] = {}
            self._n_vars_by_problem: dict[str, int] = {}
            self._default_n_vars = 1

        @staticmethod
        def _bin(generation: int, max_generations: int, n_bins: int) -> int:
            frac = float(generation) / max(1, int(max_generations))
            return min(int(frac * n_bins), n_bins - 1)

        def fit(self, records: list[dict[str, Any]]) -> "OpenLoopScheduleController":
            """Estimate the binned schedule from full-action trajectories."""
            grouped: dict[str, list[dict[str, Any]]] = {}
            for record in records:
                key = str(record["problem"]) if self.per_problem else "__global__"
                grouped.setdefault(key, []).append(record)
                n_vars = int(record.get("n_vars", 0))
                if n_vars > 0:
                    self._n_vars_by_problem[str(record["problem"])] = n_vars
            if self._n_vars_by_problem:
                self._default_n_vars = int(
                    np.median(sorted(self._n_vars_by_problem.values()))
                )
            for key, group in grouped.items():
                bins: list[dict[str, list[Any]]] = [
                    {"operator": [], "multiplier": [], "exploration": []}
                    for _ in range(self.n_bins)
                ]
                for record in group:
                    transitions = record["transitions"]
                    n_vars = max(int(record.get("n_vars", 0)), 1)
                    max_gen = int(
                        record.get("config", {}).get(
                            "generations", max(t["generation"] for t in transitions)
                        )
                    )
                    for transition in transitions[1:]:
                        action = transition["action"]
                        operator = str(action["mutation_operator"])
                        b = self._bin(
                            int(transition["generation"]), max_gen, self.n_bins
                        )
                        bins[b]["operator"].append(operator)
                        bins[b]["multiplier"].append(
                            float(action["mutation_probability"]) * n_vars
                        )
                        bins[b]["exploration"].append(
                            float(
                                action.get(
                                    "exploration_strength",
                                    OPERATOR_DEFAULT_EXPLORATION[operator],
                                )
                            )
                        )
                schedule: list[dict[str, Any]] = []
                for b in range(self.n_bins):
                    operators = bins[b]["operator"]
                    if operators:
                        operator = max(
                            sorted(set(operators)), key=operators.count
                        )
                        multipliers = bins[b]["multiplier"]
                        explorations = [
                            e
                            for e, o in zip(bins[b]["exploration"], operators)
                            if o == operator
                        ]
                        multiplier = float(np.median(multipliers))
                        exploration = float(np.median(explorations))
                    else:  # empty bin: neutral defaults
                        operator = "polynomial"
                        multiplier = 1.0
                        exploration = OPERATOR_DEFAULT_EXPLORATION[operator]
                    schedule.append(
                        {
                            "mutation_operator": operator,
                            "multiplier": multiplier,
                            "exploration_strength": exploration,
                        }
                    )
                self._schedules[key] = schedule
            return self

        def predict_action(
            self,
            generation: int,
            max_generations: int,
            problem_name: str | None = None,
        ) -> dict[str, Any]:
            """Return the scheduled action of the bin covering ``generation``."""
            if self.per_problem and problem_name is not None:
                schedule = self._schedules.get(
                    problem_name, self._schedules.get("__global__")
                )
            else:
                schedule = self._schedules.get("__global__")
                if schedule is None and self._schedules:
                    schedule = next(iter(self._schedules.values()))
            if schedule is None:
                raise RuntimeError(
                    "OpenLoopScheduleController must be fitted before predict_action()"
                )
            entry = schedule[self._bin(generation, max_generations, self.n_bins)]
            n_vars = (
                self._n_vars_by_problem.get(problem_name, self._default_n_vars)
                if problem_name is not None
                else self._default_n_vars
            )
            return {
                "mutation_operator": entry["mutation_operator"],
                "mutation_probability": entry["multiplier"] / float(max(n_vars, 1)),
                "exploration_strength": entry["exploration_strength"],
            }

        def save(self, path: str | Path) -> None:
            """Serialize to JSON; parent directories are created."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "per_problem": self.per_problem,
                        "n_bins": self.n_bins,
                        "schedules": self._schedules,
                        "n_vars_by_problem": self._n_vars_by_problem,
                        "default_n_vars": self._default_n_vars,
                    },
                    fh,
                    indent=2,
                )

        @classmethod
        def load(cls, path: str | Path) -> "OpenLoopScheduleController":
            """Load a controller saved with :meth:`save`."""
            with Path(path).open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            controller = cls(
                per_problem=bool(payload["per_problem"]),
                n_bins=int(payload["n_bins"]),
            )
            controller._schedules = {
                str(k): list(v) for k, v in payload["schedules"].items()
            }
            controller._n_vars_by_problem = {
                str(k): int(v) for k, v in payload["n_vars_by_problem"].items()
            }
            controller._default_n_vars = int(payload["default_n_vars"])
            return controller


#: Default evaluation problem grid (all Phase-0 ZDT benchmarks).
DEFAULT_PROBLEMS: tuple[str, ...] = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")
#: Held-out evaluation seeds; identical across arms for paired statistics.
DEFAULT_EVAL_SEEDS: tuple[int, ...] = tuple(range(1000, 1020))
#: Default Phase-1.75 500-trajectory training corpus.
DEFAULT_TRAIN_DIRS: tuple[str, ...] = ("results/trajectory_phase1_75",)
#: Default fixed-policy trajectories used for failure thresholds (Phase-0).
DEFAULT_FIXED_DIR = "results/trajectory"
#: Default output directory for Phase-1.75 results.
DEFAULT_OUT_DIR = "results/phase1_75"
#: Default static full-action tuning artifact (Task 2).
DEFAULT_STATIC_TUNING = "results/phase1_75/static_full_tuning.json"
#: Deployment pm bounds as multipliers on the natural ``1 / n_vars`` scale.
DEFAULT_PM_MULT_RANGE: tuple[float, float] = (0.25, 8.0)
#: Failure threshold percentile over the fixed-policy training final HV.
FAILURE_PERCENTILE = 10.0

#: Arm identifiers, in report order.
ARM_FIXED = "fixed_nsga2"
ARM_STATIC_GLOBAL = "static_full_global"
ARM_STATIC_PER_PROBLEM = "static_full_per_problem"
ARM_OPEN_LOOP_GLOBAL = "open_loop_global"
ARM_OPEN_LOOP_PER_PROBLEM = "open_loop_per_problem"
ARM_MLP_ABSOLUTE = "mlp2_closed_loop_absolute"
ARM_MLP_NORMALIZED = "mlp2_closed_loop_normalized"
ARM_GENERATION_ONLY = "generation_only_mlp"
ARM_STATE_SCRAMBLED = "state_scrambled_mlp"
ARMS: tuple[str, ...] = (
    ARM_FIXED,
    ARM_STATIC_GLOBAL,
    ARM_STATIC_PER_PROBLEM,
    ARM_OPEN_LOOP_GLOBAL,
    ARM_OPEN_LOOP_PER_PROBLEM,
    ARM_MLP_ABSOLUTE,
    ARM_MLP_NORMALIZED,
    ARM_GENERATION_ONLY,
    ARM_STATE_SCRAMBLED,
)
#: Arms with a trained multi-head MLP (identical training budget).
LEARNED_ARMS: tuple[str, ...] = (
    ARM_MLP_ABSOLUTE,
    ARM_MLP_NORMALIZED,
    ARM_GENERATION_ONLY,
    ARM_STATE_SCRAMBLED,
)

#: Contract-adaptation flags for the parallel Task-1 controller upgrade.
_CTOR_SUPPORTS_TARGET = (
    "mutation_target" in inspect.signature(MultiHeadController.__init__).parameters
)
_PREDICT_SUPPORTS_NVARS = (
    "n_vars" in inspect.signature(MultiHeadController.predict_action).parameters
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Phase-1.75 experiment.

    Args:
        argv: Argument list to parse; ``None`` reads ``sys.argv``.

    Returns:
        Parsed namespace with stage, training, evaluation, and output
        settings.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Phase-1.75 experiment: fair 9-arm paired evaluation of closed-loop "
            "evolution control against matched static/open-loop/scale/causality "
            "baselines on held-out seeds."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["train", "eval", "aggregate", "thresholds", "all"],
        default="all",
        help=(
            "Pipeline stage to run; 'all' = thresholds -> train -> eval -> "
            "aggregate (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--train-dirs",
        nargs="+",
        default=list(DEFAULT_TRAIN_DIRS),
        help=(
            "Directories with training trajectory JSONs (union is used; the "
            "Phase-1.75 500-trajectory corpus by default). Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--fixed-dir",
        type=str,
        default=DEFAULT_FIXED_DIR,
        help=(
            "Directory with fixed-policy training trajectories used to derive "
            "failure thresholds (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--static-tuning",
        type=str,
        default=DEFAULT_STATIC_TUNING,
        help="Static full-action tuning artifact JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(ARMS),
        choices=list(ARMS),
        help="Arm subset to evaluate (sharding; default: all 9 arms).",
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
        "--epochs",
        type=int,
        default=200,
        help="Fixed number of training epochs for every learned arm (default: %(default)s).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="History window of the closed-loop controller input (default: %(default)s).",
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
        help="Adam learning rate for the learned arms (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size for the learned arms (default: %(default)s).",
    )
    parser.add_argument(
        "--pm-mult-range",
        nargs=2,
        type=float,
        default=list(DEFAULT_PM_MULT_RANGE),
        metavar=("PM_MULT_LO", "PM_MULT_HI"),
        help=(
            "Multipliers on 1/n_vars defining the deployment pm range "
            "[lo/n_vars, hi/n_vars] (default: %(default)s)."
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sha256_array(array: np.ndarray) -> str:
    """SHA-256 hex digest of an array's bytes (shape/dtype included)."""
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256(arr.tobytes())
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    return digest.hexdigest()


def _exploration_range(operator: str) -> tuple[float, float]:
    """Deployment exploration-strength range of a mutation operator."""
    if operator == "polynomial":
        return POLYNOMIAL_EXPLORATION_RANGE
    return GAUSSIAN_EXPLORATION_RANGE


def _sanitize_action(
    action: dict[str, Any],
    *,
    n_vars: int,
    pm_mult_range: tuple[float, float],
) -> dict[str, Any]:
    """Clip a three-key action to the deployment ranges.

    Mutation probability is clipped to ``pm_mult_range / n_vars`` and to
    ``[1e-12, 1.0]``; exploration strength to the operator-specific range.
    Actions produced inside this runner already lie within these ranges, so
    the clip is a defensive no-op for well-formed arms.
    """
    operator = str(action["mutation_operator"])
    pm_lo = min(float(pm_mult_range[0]) / n_vars, 1.0)
    pm_hi = min(float(pm_mult_range[1]) / n_vars, 1.0)
    pm = min(max(float(action["mutation_probability"]), pm_lo), pm_hi)
    pm = min(max(pm, 1e-12), 1.0)
    lo, hi = _exploration_range(operator)
    exploration = min(max(float(action["exploration_strength"]), lo), hi)
    return {
        "mutation_operator": operator,
        "mutation_probability": float(pm),
        "exploration_strength": float(exploration),
    }


def _action_from_heads(
    controller: MultiHeadController,
    x_row: np.ndarray,
    *,
    mutation_target: str,
    n_vars: int,
    pm_mult_range: tuple[float, float],
) -> dict[str, Any]:
    """Map one feature row through a multi-head controller to an action dict.

    Replicates the deployment mapping of
    ``MultiHeadController.predict_action``: argmax operator, ``exp`` of the
    clipped log-prediction for pm (multiplier space when
    ``mutation_target == 'multiplier'``, absolute otherwise), and ``exp`` of
    the log-exploration clipped to the operator-specific range. Used for the
    generation-only arm (whose input is not an encoded history) and as the
    fallback for the normalized arm when the installed controller predates
    the ``mutation_target='multiplier'``/``n_vars`` contract.
    """
    op_probs, log_pm, log_expl = controller.predict(np.asarray(x_row).reshape(1, -1))
    operator = OPERATOR_CLASSES[int(np.argmax(op_probs[0]))]
    lo, hi = float(pm_mult_range[0]), float(pm_mult_range[1])
    if mutation_target == "multiplier":
        log_pm = float(np.clip(log_pm[0], math.log(lo), math.log(hi)))
        pm = math.exp(log_pm) / float(n_vars)
    else:
        log_pm = float(np.clip(log_pm[0], math.log(lo / n_vars), math.log(hi / n_vars)))
        pm = math.exp(log_pm)
    expl_lo, expl_hi = _exploration_range(operator)
    log_expl_val = float(
        np.clip(log_expl[0], math.log(expl_lo), math.log(expl_hi))
    )
    return {
        "mutation_operator": operator,
        "mutation_probability": float(pm),
        "exploration_strength": float(math.exp(log_expl_val)),
    }


def _closed_loop_action(
    controller: MultiHeadController,
    encoder: ProblemAwareEncoder,
    history: list[dict[str, Any]],
    *,
    mutation_target: str,
    n_vars: int,
    pm_mult_range: tuple[float, float],
) -> dict[str, Any]:
    """Closed-loop action of a learned arm for the step into the next generation.

    Prefers the upgraded ``predict_action(history, encoder, pm_min, pm_max,
    n_vars=...)`` contract (multiplier bounds + ``n_vars`` for the normalized
    arm); falls back to :func:`_action_from_heads` when the installed
    controller predates that contract.
    """
    lo, hi = float(pm_mult_range[0]), float(pm_mult_range[1])
    if mutation_target == "multiplier":
        if _PREDICT_SUPPORTS_NVARS:
            return controller.predict_action(history, encoder, lo, hi, n_vars=n_vars)
        return _action_from_heads(
            controller,
            encoder.transform(history),
            mutation_target="multiplier",
            n_vars=n_vars,
            pm_mult_range=pm_mult_range,
        )
    return controller.predict_action(history, encoder, lo / n_vars, hi / n_vars)


class GenerationOnlyPolicy:
    """Deployment adapter for the ``generation_only_mlp`` arm.

    The controller input is ``[gen_norm] + z-scored 9-dim problem features``
    only; population state/history is structurally unavailable.
    ``predict_action`` accepts (and ignores) an optional ``history`` argument
    so callers and tests can verify state-independence.
    """

    def __init__(
        self,
        controller: MultiHeadController,
        problem_vectors: dict[str, Sequence[float]],
        mutation_target: str = "multiplier",
    ) -> None:
        """Initialize the adapter.

        Args:
            controller: Fitted multi-head controller with input dim 10.
            problem_vectors: Z-scored 9-dim problem feature vector per
                problem name.
            mutation_target: Log-space pm target of the controller
                (``'multiplier'`` for this arm).
        """
        self.controller = controller
        self.problem_vectors = {
            str(k): np.asarray(v, dtype=float) for k, v in problem_vectors.items()
        }
        self.mutation_target = str(mutation_target)

    def predict_action(
        self,
        generation: int,
        max_generations: int,
        problem_name: str,
        *,
        n_vars: int,
        pm_mult_range: tuple[float, float],
        history: Any = None,
    ) -> dict[str, Any]:
        """Predict the action for the step into ``generation``.

        Args:
            generation: Generation the action applies to (``t >= 1``).
            max_generations: Total generations of the run (normalizer).
            problem_name: Problem identifier (selects the feature block).
            n_vars: Decision-variable count of the problem.
            pm_mult_range: Deployment multiplier bounds.
            history: Ignored; accepted to prove state-independence.

        Returns:
            Three-key action dict.
        """
        gen_norm = float(generation) / max(1, int(max_generations))
        x_row = np.concatenate(
            [[gen_norm], self.problem_vectors[str(problem_name)]]
        )
        return _action_from_heads(
            self.controller,
            x_row,
            mutation_target=self.mutation_target,
            n_vars=n_vars,
            pm_mult_range=pm_mult_range,
        )


# ---------------------------------------------------------------------------
# Thresholds stage
# ---------------------------------------------------------------------------


def run_thresholds_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Derive per-problem failure thresholds from fixed-policy training runs.

    The threshold is the 10th percentile of final hypervolume over the
    fixed-policy trajectories of ``args.fixed_dir`` (records whose
    ``config.policy`` is ``'fixed'``; a missing policy key is treated as the
    Phase-0 fixed policy). The thresholds derive from the TRAINING
    distribution only and must be computed before any held-out evaluation.

    Args:
        args: Parsed arguments.

    Returns:
        The thresholds payload as written to
        ``{out_dir}/failure_thresholds.json``.

    Raises:
        FileNotFoundError: If the fixed-policy directory does not exist.
        ValueError: If no fixed-policy trajectories are found.
    """
    fixed_dir = Path(args.fixed_dir)
    if not fixed_dir.is_dir():
        raise FileNotFoundError(
            f"fixed-policy trajectory directory not found: {fixed_dir}; "
            "failure thresholds require the Phase-0 fixed-policy training runs"
        )
    records = _load_records_impl(fixed_dir)
    fixed = [
        r for r in records if str(r["config"].get("policy", "fixed")) == "fixed"
    ]
    if not fixed:
        raise ValueError(f"no fixed-policy trajectories found in {fixed_dir}")
    by_problem: dict[str, list[float]] = {}
    for record in fixed:
        problem = str(record["config"].get("problem", record["problem"]))
        final_hv = float(record["transitions"][-1]["state"]["hv"])
        by_problem.setdefault(problem, []).append(final_hv)
    thresholds = {
        problem: float(np.percentile(values, FAILURE_PERCENTILE))
        for problem, values in sorted(by_problem.items())
    }
    payload: dict[str, Any] = {
        "method": (
            f"{FAILURE_PERCENTILE:.0f}th percentile of final hypervolume over "
            "fixed-policy training trajectories"
        ),
        "source": (
            "training distribution only (fixed-policy runs); derived before "
            "any held-out test-seed evaluation"
        ),
        "fixed_dir": str(fixed_dir),
        "percentile": float(FAILURE_PERCENTILE),
        "thresholds": thresholds,
        "n_trajectories": {
            problem: len(values) for problem, values in sorted(by_problem.items())
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "failure_thresholds.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[thresholds] wrote {path}: {thresholds}")
    return payload


# ---------------------------------------------------------------------------
# Train stage
# ---------------------------------------------------------------------------


def _load_train_records(train_dirs: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load the union of training trajectory records.

    Args:
        train_dirs: Directories with trajectory JSONs; missing directories
            are skipped with a warning.

    Returns:
        Loaded records (``{'problem', 'n_vars', 'seed', 'runtime_sec',
        'config', 'transitions'}``).

    Raises:
        FileNotFoundError: If none of the directories exists.
        ValueError: If no trajectories were found.
    """
    records: list[dict[str, Any]] = []
    any_found = False
    for directory in train_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            print(f"[warn] training directory missing, skipped: {directory}")
            continue
        any_found = True
        records.extend(_load_records_impl(directory))
    if not any_found:
        raise FileNotFoundError(
            f"no training directory found among {list(map(str, train_dirs))}; "
            "generate the Phase-1.75 corpus first "
            "(experiments/generate_dataset.py --action-space full)"
        )
    if not records:
        raise ValueError(
            f"no training trajectories found in {list(map(str, train_dirs))}"
        )
    return records


def _fit_encoders(
    records: list[dict[str, Any]],
    eval_problem_names: Sequence[str],
    window: int,
) -> dict[str, ProblemAwareEncoder]:
    """Fit one ProblemAwareEncoder per problem over ALL training data.

    Mirrors the Phase-1.5 ``mlp2`` arm: every encoder shares the same state
    and problem z-score statistics (fitted over all training trajectories
    and all problems), only the constant problem-descriptor block differs.

    Args:
        records: Training records.
        eval_problem_names: Evaluation problems (encoders are also built
            for problems without training trajectories).
        window: History window in generations.

    Returns:
        Dict mapping problem name to its fitted encoder.
    """
    problem_names = sorted(
        set(eval_problem_names) | {str(r["problem"]) for r in records}
    )
    problem_objs = {name: get_problem(name) for name in problem_names}
    trajectories = [r["transitions"] for r in records]
    all_problems = [problem_objs[name] for name in problem_names]
    return {
        name: ProblemAwareEncoder(window, problem_objs[name]).fit(
            trajectories, all_problems
        )
        for name in problem_names
    }


def _build_closed_loop_samples(
    records: list[dict[str, Any]],
    encoders: dict[str, ProblemAwareEncoder],
    window: int,
    mutation_target: str,
) -> dict[str, np.ndarray]:
    """Build multi-head samples grouped per problem with its own encoder.

    The log-pm target is absolute (``log pm``) or normalized
    (``log(pm * n_vars)``) depending on ``mutation_target``. The normalized
    target is derived from the absolute one per problem group
    (``log pm + log n_vars``), which is exactly the Task-1
    ``mutation_target='multiplier'`` semantics; this keeps the builder
    compatible with both the old and the upgraded ``controller.dataset``.

    Returns:
        Dict with ``X``, ``y_op``, ``y_logpm``, ``y_logexpl``, ``w``,
        ``traj_ids`` (globally unique ids).
    """
    problem_names = sorted({str(r["problem"]) for r in records})
    x_parts: list[np.ndarray] = []
    yop_parts: list[np.ndarray] = []
    ypm_parts: list[np.ndarray] = []
    yexpl_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    offset = 0
    for name in problem_names:
        group = [r for r in records if str(r["problem"]) == name]
        n_vars = int(group[0]["n_vars"]) or get_problem(name).n_vars
        X_g, yop_g, ypm_g, yexpl_g, w_g, ids_g = build_multihead_samples(
            [r["transitions"] for r in group], encoders[name], window
        )
        if mutation_target == "multiplier":
            ypm_g = ypm_g + math.log(float(n_vars))
        x_parts.append(X_g)
        yop_parts.append(yop_g)
        ypm_parts.append(ypm_g)
        yexpl_parts.append(yexpl_g)
        w_parts.append(w_g)
        id_parts.append(ids_g + offset)
        offset += len(group)
    return {
        "X": np.vstack(x_parts),
        "y_op": np.concatenate(yop_parts),
        "y_logpm": np.concatenate(ypm_parts),
        "y_logexpl": np.concatenate(yexpl_parts),
        "w": np.concatenate(w_parts),
        "traj_ids": np.concatenate(id_parts),
    }


def _zscored_problem_vectors(
    problem_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Z-scored 9-dim problem feature vectors over the given problem set."""
    vectors = {
        name: problem_feature_vector(get_problem(name)) for name in problem_names
    }
    stacked = np.vstack([vectors[name] for name in problem_names])
    std = stacked.std(axis=0)
    mean = stacked.mean(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    return {name: (vectors[name] - mean) / std for name in problem_names}


def _build_generation_only_samples(
    records: list[dict[str, Any]],
    problem_vectors: dict[str, np.ndarray],
    mutation_target: str,
) -> dict[str, np.ndarray]:
    """Build samples with input ``[gen_norm] + 9 problem features`` only.

    Targets and advantage weights replicate
    ``controller.dataset.build_multihead_samples`` exactly (operator class
    index, clipped log targets, exploration imputation for legacy actions,
    weight ``max(r_t - mean(r_traj), 0) + 1e-6`` with
    ``r = delta_hv + delta_igd``); only the input features differ — no
    population state reaches this arm.
    """
    x_rows: list[list[float]] = []
    y_op_vals: list[int] = []
    y_logpm_vals: list[float] = []
    y_logexpl_vals: list[float] = []
    w_vals: list[float] = []
    id_vals: list[int] = []
    for j, record in enumerate(records):
        trajectory = record["transitions"]
        if len(trajectory) < 2:
            continue
        problem = str(record["problem"])
        n_vars = int(record["n_vars"]) or get_problem(problem).n_vars
        max_gen = int(
            record["config"].get("generations", trajectory[-1]["generation"])
        )
        merged = [merge_state_reward(t) for t in trajectory]
        rewards = np.asarray(
            [m["delta_hv"] + m["delta_igd"] for m in merged], dtype=float
        )
        mean_reward = float(rewards.mean())
        for t in range(1, len(trajectory)):
            action = trajectory[t]["action"]
            operator = str(action["mutation_operator"])
            pm = float(action["mutation_probability"])
            if "exploration_strength" in action:
                exploration = float(action["exploration_strength"])
            else:
                exploration = OPERATOR_DEFAULT_EXPLORATION[operator]
            gen_norm = float(trajectory[t]["generation"]) / max(1, max_gen)
            x_rows.append(
                [gen_norm] + [float(v) for v in problem_vectors[problem]]
            )
            y_op_vals.append(OPERATOR_TO_INDEX[operator])
            target_pm = pm * n_vars if mutation_target == "multiplier" else pm
            y_logpm_vals.append(math.log(max(target_pm, 1e-12)))
            y_logexpl_vals.append(math.log(max(exploration, 1e-12)))
            w_vals.append(max(float(rewards[t]) - mean_reward, 0.0) + 1e-6)
            id_vals.append(j)
    return {
        "X": np.asarray(x_rows, dtype=float),
        "y_op": np.asarray(y_op_vals, dtype=int),
        "y_logpm": np.asarray(y_logpm_vals, dtype=float),
        "y_logexpl": np.asarray(y_logexpl_vals, dtype=float),
        "w": np.asarray(w_vals, dtype=float),
        "traj_ids": np.asarray(id_vals, dtype=int),
    }


def _new_controller(
    input_dim: int, *, seed: int, lr: float, name: str, mutation_target: str
) -> MultiHeadController:
    """Construct a MultiHeadController, passing ``mutation_target`` when supported."""
    kwargs: dict[str, Any] = {}
    if _CTOR_SUPPORTS_TARGET:
        kwargs["mutation_target"] = mutation_target
    return MultiHeadController(
        input_dim=input_dim,
        hidden_dims=(64, 64),
        seed=seed,
        lr=lr,
        name=name,
        **kwargs,
    )


def _train_learned_arm(
    name: str,
    samples: dict[str, np.ndarray],
    *,
    mutation_target: str,
    epochs: int,
    val_fraction: float,
    seed: int,
    lr: float,
    batch_size: int,
    shuffle_train_histories: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train one learned arm with the shared budget and split protocol.

    When ``shuffle_train_histories`` is True (``state_scrambled_mlp``), the
    feature rows of the TRAIN split are permuted across trajectories with
    ``np.random.Generator(PCG64(seed + 1))`` while targets/weights stay
    aligned to the original rows, breaking state->action causality while
    preserving input marginals. SHA-256 checksums of the train features
    before/after the shuffle are recorded as the shuffle proof artifact.
    """
    X = np.asarray(samples["X"], dtype=float)
    y_op = np.asarray(samples["y_op"], dtype=np.int64)
    y_logpm = np.asarray(samples["y_logpm"], dtype=float)
    y_logexpl = np.asarray(samples["y_logexpl"], dtype=float)
    w = np.asarray(samples["w"], dtype=float)
    traj_ids = np.asarray(samples["traj_ids"], dtype=int)
    if X.shape[0] == 0:
        raise ValueError(f"arm {name}: zero training samples")

    targets = np.column_stack([y_op.astype(float), y_logpm, y_logexpl])
    split = train_val_split(
        X, targets, w, traj_ids, val_fraction=val_fraction, seed=seed
    )
    X_train = np.asarray(split["X_train"], dtype=float)
    y_train = np.asarray(split["y_train"], dtype=float)
    X_val = np.asarray(split["X_val"], dtype=float)
    y_val = np.asarray(split["y_val"], dtype=float)
    has_val = X_val.shape[0] > 0

    checksum_unshuffled: str | None = None
    checksum_shuffled: str | None = None
    if shuffle_train_histories:
        checksum_unshuffled = _sha256_array(X_train)
        permutation = np.random.Generator(np.random.PCG64(seed + 1)).permutation(
            X_train.shape[0]
        )
        X_train = X_train[permutation]
        checksum_shuffled = _sha256_array(X_train)

    controller = _new_controller(
        X.shape[1], seed=seed, lr=lr, name=name, mutation_target=mutation_target
    )
    history = controller.fit(
        X_train,
        y_train[:, 0].astype(np.int64),
        y_train[:, 1],
        y_train[:, 2],
        sample_weight=np.asarray(split["w_train"], dtype=float),
        epochs=epochs,
        batch_size=batch_size,
        X_val=X_val if has_val else None,
        yop_val=y_val[:, 0].astype(np.int64) if has_val else None,
        ypm_val=y_val[:, 1] if has_val else None,
        yexpl_val=y_val[:, 2] if has_val else None,
        verbose=verbose,
    )
    return {
        "name": name,
        "mutation_target": mutation_target,
        "input_dim": int(X.shape[1]),
        "controller": controller,
        "history": history,
        "n_train_samples": int(X_train.shape[0]),
        "n_val_samples": int(X_val.shape[0]),
        "train_history_shuffled": bool(shuffle_train_histories),
        "x_train_checksum_unshuffled": checksum_unshuffled,
        "x_train_checksum_shuffled": checksum_shuffled,
    }


def _load_static_tuning(path: str | Path) -> dict[str, Any]:
    """Load the static full-action tuning artifact.

    Accepted schema (``results/phase1_75/static_full_tuning.json``, Task 2)::

        {"global": {"operator", "multiplier", "exploration_strength"},
         "per_problem": {"<problem>": {"operator", "multiplier",
                                       "exploration_strength"}, ...}}

    Tolerances: top-level ``"static_full_global"``/``"static_full_per_problem"``
    aliases and the entry key ``"mutation_multiplier"`` for ``"multiplier"``.
    A flat single-tuple file is interpreted as the global entry.

    Raises:
        FileNotFoundError: If the tuning artifact does not exist.
        ValueError: If no usable tuple is found.
    """

    def _entry(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "operator": str(raw["operator"]),
            "multiplier": float(raw.get("multiplier", raw.get("mutation_multiplier"))),
            "exploration_strength": float(raw["exploration_strength"]),
        }

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"static tuning artifact not found: {path}; run "
            "experiments/tune_static_full.py (Task 2) first"
        )
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    result: dict[str, Any] = {"global": None, "per_problem": {}}
    global_raw = payload.get("global", payload.get("static_full_global"))
    if global_raw is None and "operator" in payload:
        global_raw = payload
    if global_raw is not None:
        result["global"] = _entry(global_raw)
    per_problem_raw = payload.get(
        "per_problem", payload.get("static_full_per_problem", {})
    )
    for problem, raw in per_problem_raw.items():
        result["per_problem"][str(problem)] = _entry(raw)
    if result["global"] is None and not result["per_problem"]:
        raise ValueError(f"no static full-action tuples found in {path}")
    return result


def _training_info_entry(trained: dict[str, Any], epochs: int) -> dict[str, Any]:
    """Serializable training summary of one learned arm."""
    history = trained["history"]
    train_loss = [float(v) for v in history.get("train_loss", [])]
    val_loss = [float(v) for v in history.get("val_loss", [])]
    return {
        "kind": "multihead_mlp",
        "mutation_target": trained["mutation_target"],
        "input_dim": int(trained["input_dim"]),
        "hidden_dims": [64, 64],
        "n_train_samples": int(trained["n_train_samples"]),
        "n_val_samples": int(trained["n_val_samples"]),
        "epochs": int(epochs),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "final_train_loss": train_loss[-1] if train_loss else None,
        "final_val_loss": val_loss[-1] if val_loss else None,
        "train_history_shuffled": bool(trained["train_history_shuffled"]),
        "x_train_checksum_unshuffled": trained["x_train_checksum_unshuffled"],
        "x_train_checksum_shuffled": trained["x_train_checksum_shuffled"],
    }


def run_train_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Fit all arms and persist controllers/encoders under ``out_dir/controllers``.

    Returns:
        The training info payload as written to
        ``{out_dir}/controllers/train_info.json``.
    """
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    controllers_dir = out_dir / "controllers"
    controllers_dir.mkdir(parents=True, exist_ok=True)
    window = int(args.window)
    problems = [get_problem(name).name for name in args.problems]

    records = _load_train_records(args.train_dirs)
    per_problem_counts = {
        name: sum(1 for r in records if str(r["problem"]) == name)
        for name in sorted({str(r["problem"]) for r in records})
    }
    print(f"[train] {len(records)} training trajectories: {per_problem_counts}")

    # --- encoders (shared by the closed-loop learned arms) ---
    encoders = _fit_encoders(records, problems, window)
    for name, encoder in encoders.items():
        encoder.save(controllers_dir / "encoders" / f"{name}.json")

    # --- learned arms (identical budget) ---
    learned: dict[str, dict[str, Any]] = {}
    common = dict(
        epochs=int(args.epochs),
        val_fraction=float(args.val_fraction),
        seed=int(args.train_seed),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        verbose=bool(args.verbose),
    )
    samples_abs = _build_closed_loop_samples(records, encoders, window, "absolute")
    learned[ARM_MLP_ABSOLUTE] = _train_learned_arm(
        ARM_MLP_ABSOLUTE, samples_abs, mutation_target="absolute", **common
    )
    samples_mult = _build_closed_loop_samples(records, encoders, window, "multiplier")
    learned[ARM_MLP_NORMALIZED] = _train_learned_arm(
        ARM_MLP_NORMALIZED, samples_mult, mutation_target="multiplier", **common
    )
    learned[ARM_STATE_SCRAMBLED] = _train_learned_arm(
        ARM_STATE_SCRAMBLED,
        samples_mult,
        mutation_target="multiplier",
        shuffle_train_histories=True,
        **common,
    )
    problem_names = sorted(
        set(problems) | {str(r["problem"]) for r in records}
    )
    problem_vectors = _zscored_problem_vectors(problem_names)
    samples_gen = _build_generation_only_samples(records, problem_vectors, "multiplier")
    learned[ARM_GENERATION_ONLY] = _train_learned_arm(
        ARM_GENERATION_ONLY, samples_gen, mutation_target="multiplier", **common
    )
    for arm in LEARNED_ARMS:
        learned[arm]["controller"].save(controllers_dir / f"{arm}.pt")
        print(
            f"[trained] {arm}: {learned[arm]['n_train_samples']} train / "
            f"{learned[arm]['n_val_samples']} val samples, "
            f"final_train_loss={learned[arm]['history']['train_loss'][-1]:.6f}"
        )

    # --- static full-action arms (from the Task-2 tuning artifact) ---
    tuning = _load_static_tuning(args.static_tuning)
    if tuning["global"] is not None:
        StaticFullController(**tuning["global"]).save(
            controllers_dir / f"{ARM_STATIC_GLOBAL}.json"
        )
    for problem, entry in tuning["per_problem"].items():
        StaticFullController(**entry).save(
            controllers_dir / ARM_STATIC_PER_PROBLEM / f"{problem}.json"
        )

    # --- open-loop schedule arms (full-action trajectories only) ---
    full_records = [
        r for r in records if str(r["config"].get("action_space", "")) == "full"
    ]
    if not full_records:
        raise ValueError(
            "no full-action training trajectories found; the open-loop arms "
            "require trajectories with config.action_space == 'full'"
        )
    for arm, per_problem in (
        (ARM_OPEN_LOOP_GLOBAL, False),
        (ARM_OPEN_LOOP_PER_PROBLEM, True),
    ):
        OpenLoopScheduleController(per_problem=per_problem).fit(full_records).save(
            controllers_dir / f"{arm}.json"
        )
    print(
        f"[train] open-loop schedules fitted on {len(full_records)} "
        "full-action trajectories"
    )

    wall_time_sec = time.perf_counter() - started
    info: dict[str, Any] = {
        "wall_time_sec": float(wall_time_sec),
        "window": window,
        "train_dirs": [str(d) for d in args.train_dirs],
        "n_train_trajectories": len(records),
        "train_trajectories_per_problem": per_problem_counts,
        "n_full_action_trajectories": len(full_records),
        "static_tuning": str(args.static_tuning),
        "encoders": {
            "type": "ProblemAwareEncoder",
            "window": window,
            "problems": sorted(encoders),
        },
        "generation_only": {
            "mutation_target": "multiplier",
            "problem_vectors_zscored": {
                name: [float(v) for v in vector]
                for name, vector in problem_vectors.items()
            },
        },
        "arms": {
            arm: _training_info_entry(learned[arm], int(args.epochs))
            for arm in LEARNED_ARMS
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    info_path = controllers_dir / "train_info.json"
    with info_path.open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    print(f"[train] wrote {info_path} ({wall_time_sec:.2f}s)")
    return info


# ---------------------------------------------------------------------------
# Eval stage
# ---------------------------------------------------------------------------


def _build_run_config(
    *,
    arm: str,
    problem_name: str,
    n_vars: int,
    seed: int,
    pop_size: int,
    generations: int,
    window: int,
    pm_mult_range: tuple[float, float],
    failure_threshold: float | None,
    ref_point: np.ndarray,
    n_reference_points: int,
) -> dict[str, Any]:
    """Configuration stored inside each per-run JSON (fully determines the run)."""
    return {
        "phase": 1.75,
        "problem": problem_name,
        "n_vars": int(n_vars),
        "algorithm": "nsga2",
        "arm": arm,
        "seed": int(seed),
        "pop_size": int(pop_size),
        "generations": int(generations),
        "window": int(window),
        "action_space": "full",
        "pm_mult_range": [float(pm_mult_range[0]), float(pm_mult_range[1])],
        "pm_absolute_range": [
            float(pm_mult_range[0]) / n_vars,
            float(pm_mult_range[1]) / n_vars,
        ],
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
        "failure_threshold": (
            None if failure_threshold is None else float(failure_threshold)
        ),
        "protocol": (
            "paired closed-loop evaluation; the action for the step into "
            "generation t uses only merged history up to generation t-1; the "
            "recorded action is the actual action used (current_action() "
            "after the step)"
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
    generations: int,
    pop_size: int,
    n_reference_points: int = 200,
    ref_point: np.ndarray | None = None,
    pm_mult_range: tuple[float, float] = DEFAULT_PM_MULT_RANGE,
    context: dict[str, Any],
    window: int = 10,
    failure_threshold: float | None = None,
    runs_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one evaluation trajectory of one arm on one (problem, seed) pair.

    Args:
        arm: Arm identifier (one of :data:`ARMS`).
        problem_name: Benchmark identifier accepted by ``get_problem``.
        seed: Evaluation random seed.
        generations: Number of NSGA-II generations to execute.
        pop_size: Population size.
        n_reference_points: Points sampled from the true Pareto front for IGD.
        ref_point: Hypervolume reference point; defaults to ``(1.1, 1.1)``.
        pm_mult_range: Deployment multiplier bounds on ``1 / n_vars``.
        context: Loaded controllers/encoders (see :func:`_load_eval_context`).
        window: History window of the closed-loop learned arms.
        failure_threshold: Per-problem failure threshold on final HV;
            ``None`` marks the run not-failed (thresholds unavailable).
        runs_dir: If given, the per-run JSON is written to
            ``{runs_dir}/{arm}__{problem}__seed{seed}.json``.

    Returns:
        Per-run metrics dict with ``arm``, ``problem``, ``seed``,
        ``final_hv``, ``final_igd``, ``auc_hv``, ``runtime_sec``, ``failed``,
        ``trajectory_file``.
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

    start = time.perf_counter()
    algorithm.initialize()
    recorder.record(
        algorithm.generation, algorithm.nondominated_front(), algorithm.current_action()
    )
    for _ in range(generations):
        if arm == ARM_FIXED:
            algorithm.step()
        else:
            generation = algorithm.generation + 1
            if arm == ARM_STATIC_GLOBAL:
                action = context[ARM_STATIC_GLOBAL].predict_action(n_vars=n_vars)
            elif arm == ARM_STATIC_PER_PROBLEM:
                action = context[ARM_STATIC_PER_PROBLEM][problem.name].predict_action(
                    n_vars=n_vars
                )
            elif arm in (ARM_OPEN_LOOP_GLOBAL, ARM_OPEN_LOOP_PER_PROBLEM):
                action = context[arm].predict_action(
                    generation, generations, problem_name=problem.name
                )
            elif arm == ARM_GENERATION_ONLY:
                action = context[ARM_GENERATION_ONLY].predict_action(
                    generation,
                    generations,
                    problem.name,
                    n_vars=n_vars,
                    pm_mult_range=pm_mult_range,
                )
            elif arm in (ARM_MLP_ABSOLUTE, ARM_MLP_NORMALIZED, ARM_STATE_SCRAMBLED):
                history = [
                    merge_state_reward(tr)
                    for tr in recorder.transitions()[-int(window) :]
                ]
                mutation_target = (
                    "absolute" if arm == ARM_MLP_ABSOLUTE else "multiplier"
                )
                action = _closed_loop_action(
                    context[arm],
                    context["encoders"][problem.name],
                    history,
                    mutation_target=mutation_target,
                    n_vars=n_vars,
                    pm_mult_range=pm_mult_range,
                )
            else:
                raise ValueError(f"unknown arm {arm!r}")
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
        "arm": arm,
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
            arm=arm,
            problem_name=problem.name,
            n_vars=n_vars,
            seed=seed,
            pop_size=pop_size,
            generations=generations,
            window=window,
            pm_mult_range=pm_mult_range,
            failure_threshold=failure_threshold,
            ref_point=ref_point,
            n_reference_points=n_reference_points,
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
    return metrics


def _load_eval_context(
    args: argparse.Namespace, problems: Sequence[str]
) -> dict[str, Any]:
    """Load fitted controllers/encoders for the requested arm subset.

    Raises:
        FileNotFoundError: If a required artifact under
            ``{out_dir}/controllers`` is missing (train stage not run).
    """
    controllers_dir = Path(args.out_dir) / "controllers"
    if not controllers_dir.is_dir():
        raise FileNotFoundError(
            f"controllers directory not found: {controllers_dir}; "
            "run --stage train first"
        )

    def _require(path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(
                f"required controller artifact missing: {path}; "
                "run --stage train first"
            )
        return path

    context: dict[str, Any] = {}
    arms = set(args.arms)
    learned = arms & set(LEARNED_ARMS)
    if learned:
        context["encoders"] = {
            name: ProblemAwareEncoder.load(_require(controllers_dir / "encoders" / f"{name}.json"))
            for name in problems
        }
        for arm in sorted(learned):
            context[arm] = MultiHeadController.load(
                _require(controllers_dir / f"{arm}.pt")
            )
    if ARM_GENERATION_ONLY in arms:
        with _require(controllers_dir / "train_info.json").open(
            "r", encoding="utf-8"
        ) as fh:
            info = json.load(fh)
        context[ARM_GENERATION_ONLY] = GenerationOnlyPolicy(
            context[ARM_GENERATION_ONLY],
            info["generation_only"]["problem_vectors_zscored"],
            mutation_target=info["generation_only"].get(
                "mutation_target", "multiplier"
            ),
        )
    if ARM_STATIC_GLOBAL in arms:
        context[ARM_STATIC_GLOBAL] = StaticFullController.load(
            _require(controllers_dir / f"{ARM_STATIC_GLOBAL}.json")
        )
    if ARM_STATIC_PER_PROBLEM in arms:
        context[ARM_STATIC_PER_PROBLEM] = {
            name: StaticFullController.load(
                _require(controllers_dir / ARM_STATIC_PER_PROBLEM / f"{name}.json")
            )
            for name in problems
        }
    for arm in (ARM_OPEN_LOOP_GLOBAL, ARM_OPEN_LOOP_PER_PROBLEM):
        if arm in arms:
            context[arm] = OpenLoopScheduleController.load(
                _require(controllers_dir / f"{arm}.json")
            )
    return context


def run_eval_stage(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Evaluate the requested (arm, problem, seed) subset, writing per-run JSONs."""
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    problems = [get_problem(name).name for name in args.problems]
    seeds = [int(s) for s in args.seeds]
    pm_mult_range = (float(args.pm_mult_range[0]), float(args.pm_mult_range[1]))
    ref_point = np.asarray(args.ref_point, dtype=float)

    thresholds_path = out_dir / "failure_thresholds.json"
    thresholds: dict[str, float] = {}
    if thresholds_path.is_file():
        with thresholds_path.open("r", encoding="utf-8") as fh:
            thresholds = {
                str(k): float(v)
                for k, v in json.load(fh).get("thresholds", {}).items()
            }
    else:
        print(
            f"[warn] {thresholds_path} missing; runs are marked failed=false. "
            "Run --stage thresholds BEFORE eval for failure detection."
        )

    context = _load_eval_context(args, problems)
    metrics: list[dict[str, Any]] = []
    for arm in args.arms:
        for problem_name in problems:
            for seed in seeds:
                summary = evaluate_run(
                    arm,
                    problem_name,
                    seed,
                    generations=int(args.generations),
                    pop_size=int(args.pop_size),
                    n_reference_points=int(args.n_reference_points),
                    ref_point=ref_point,
                    pm_mult_range=pm_mult_range,
                    context=context,
                    window=int(args.window),
                    failure_threshold=thresholds.get(problem_name),
                    runs_dir=runs_dir,
                )
                metrics.append(summary)
                print(
                    f"[eval] {arm} {problem_name} seed={seed} "
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


def run_aggregate_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Merge ``runs/*.json`` into ``{out_dir}/results.json`` (exact schema)."""
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    problems = [get_problem(name).name for name in args.problems]
    eval_seeds = [int(s) for s in args.seeds]

    runs: dict[str, Any] = {}
    eval_runtime_sum = 0.0
    if runs_dir.is_dir():
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

    thresholds: dict[str, Any] = {}
    thresholds_path = out_dir / "failure_thresholds.json"
    if thresholds_path.is_file():
        with thresholds_path.open("r", encoding="utf-8") as fh:
            thresholds = json.load(fh)

    training: dict[str, Any] = {}
    train_wall = 0.0
    info_path = out_dir / "controllers" / "train_info.json"
    if info_path.is_file():
        with info_path.open("r", encoding="utf-8") as fh:
            info = json.load(fh)
        train_wall = float(info.get("wall_time_sec", 0.0))
        training = {arm: info["arms"][arm] for arm in LEARNED_ARMS if arm in info.get("arms", {})}

    aggregate_wall = time.perf_counter() - started
    wall_time_sec = train_wall + eval_runtime_sum + aggregate_wall

    payload: dict[str, Any] = {
        "config": {
            "phase": 1.75,
            "arms": list(ARMS),
            "problems": problems,
            "eval_seeds": eval_seeds,
            "train_dirs": [str(d) for d in args.train_dirs],
            "fixed_dir": str(args.fixed_dir),
            "static_tuning": str(args.static_tuning),
            "out_dir": str(args.out_dir),
            "pop_size": int(args.pop_size),
            "generations": int(args.generations),
            "epochs": int(args.epochs),
            "window": int(args.window),
            "val_fraction": float(args.val_fraction),
            "train_seed": int(args.train_seed),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "pm_mult_range": [
                float(args.pm_mult_range[0]),
                float(args.pm_mult_range[1]),
            ],
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
            "wall_time_sec": float(wall_time_sec),
            "wall_time_breakdown": {
                "train_sec": float(train_wall),
                "eval_runtime_sum_sec": float(eval_runtime_sum),
                "aggregate_sec": float(aggregate_wall),
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "failure_thresholds": thresholds,
        "arms": list(ARMS),
        "problems": problems,
        "eval_seeds": eval_seeds,
        "runs": runs,
        "training": training,
    }
    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(
        f"[aggregate] merged {len(runs)} runs -> {results_path} "
        f"({aggregate_wall:.2f}s)"
    )
    return payload


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the requested Phase-1.75 stage(s).

    ``--stage all`` runs thresholds -> train -> eval -> aggregate in this
    order, guaranteeing failure thresholds exist before eval uses them.

    Args:
        args: Parsed arguments as produced by :func:`parse_args`.

    Returns:
        For ``aggregate``/``all``: the results payload written to
        ``results.json``. For ``thresholds``: the thresholds payload. For
        ``train``: the training info payload. For ``eval``: a summary dict
        with the per-run metrics under ``"runs"``.
    """
    stage = str(args.stage)
    if stage == "thresholds":
        return run_thresholds_stage(args)
    if stage == "train":
        return run_train_stage(args)
    if stage == "eval":
        return {"runs": run_eval_stage(args)}
    if stage == "aggregate":
        return run_aggregate_stage(args)
    run_thresholds_stage(args)
    run_train_stage(args)
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
