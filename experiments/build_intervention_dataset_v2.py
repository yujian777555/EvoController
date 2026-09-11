from __future__ import annotations

"""Phase 2.75C, Tasks 1-2: problem-aware, baseline-controlled intervention data.

Phase 2.75's dataset (``experiments/build_intervention_dataset.py``, kept
untouched for reproducibility) is ``X = [state 60] + [action 4]`` with a single
advantage definition: the candidate's outcome minus the mean outcome of the
same state's candidates. Phase 2.75C asks for two changes.

**Task 1 — problem context.** The feature matrix becomes 76 columns::

    [ state encoding 60 ] [ problem features 9 ] [ runtime context 3 ] [ action 4 ]

* *problem features*: :func:`controller.problem_features.problem_feature_vector`
  (the canonical 9-dim descriptor built from ``Problem.describe()``), z-scored
  over the problems present in the build; the statistics are recorded in the
  meta so the transform is invertible.
* *runtime context* — three quantities computable from the intervention data
  itself: ``hv_before`` (hypervolume at the snapshot), ``generation /
  max_generation`` (progress through the run) and ``hv_before /
  hv_reference`` (the same hypervolume normalized by the hypervolume of the
  problem's sampled reference front under the file's own reference point).

**Task 2 — advantage definition.** ``--advantage-baseline`` selects which
per-``(state, horizon)`` baseline defines the target; every definition is
written to its own file so they can be trained against each other:

``state_mean``
    ``mean_reward(candidate) - mean over all candidates`` (the Phase-2.75
    definition; numerically identical to v1).
``default_action``
    ``mean_reward(candidate) - mean_reward(NSGA-II default action)``, i.e.
    improvement over the standard strategy rather than over the field. The
    default action is ``polynomial`` / multiplier ``1.0`` / ``eta_m 20.0``;
    since the candidate set is sampled, the stand-in is selected in this order
    (the rule actually used is recorded per state):

    1. ``exact`` — a candidate whose action equals the default tuple;
    2. ``controller`` — the candidate with ``kind == "controller"`` (the action
       the deployed controller takes, i.e. the practical status quo);
    3. ``nearest`` — the candidate closest to the default in the normalized
       action space (``hypot`` of the log-multiplier and eta_m gaps, plus a
       unit penalty when the operator differs).

``final_hv``
    the diagnostic of ``docs/PHASE2_75_RESULTS.md``: the target should reflect
    where the branch *ends up*, not a fixed 20-generation window. The corpus
    only branches 5/10/20 generations, so the largest requested horizon is used
    as a **proxy** (``proxy_horizon`` plus a ``limitation`` string are written
    to the meta), the baseline stays ``state_mean`` and every horizon row of a
    candidate receives the proxy-horizon advantage.

Outputs (``--out-dir``, default ``results/phase2_75c``): one directory per
advantage definition, ``{out_dir}/{baseline}/intervention_dataset_{problem}.npz``
plus ``{out_dir}/{baseline}/intervention_meta.json`` (a v1-shaped meta so the
existing ``experiments/train_advantage_predictor.py`` can consume a definition
directory directly), and one combined ``intervention_meta_v2.json`` that keeps
every v1 meta field and adds the feature layout, the problem-feature
statistics, the per-baseline advantage distributions and the default-action
selection counts.

Example:
    ``python experiments/build_intervention_dataset_v2.py``
    ``python experiments/build_intervention_dataset_v2.py --advantage-baseline default_action``
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    # Allow ``python experiments/build_intervention_dataset_v2.py`` from the
    # repo root: the script directory (not the repo root) is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import get_problem
from controller.problem_features import (
    PROBLEM_FEATURE_NAMES,
    problem_feature_vector,
)
from controller.state_encoder import StateEncoder
from experiments.build_intervention_dataset import (
    action_features,
    load_snapshot_history,
    summarize_groups,
)
from metrics.indicators import hypervolume

#: Directory with ``counterfactual_horizon_{problem}.json`` files.
DEFAULT_INPUT_DIR = "results/phase2b/counterfactual"
#: Directory with the harvested snapshot pickles (``history`` source).
DEFAULT_SNAPSHOTS_DIR = "results/phase1_75/snapshots"
#: Fitted encoder of the Phase-2A/2B/2.75 models.
DEFAULT_ENCODER = "results/phase2_outcome/encoder.json"
#: Output directory (Phase-2.75C isolated).
DEFAULT_OUT_DIR = "results/phase2_75c"
#: Advantage horizons extracted from the intervention files.
DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)
#: Advantage definitions supported by ``--advantage-baseline``.
ADVANTAGE_BASELINES: tuple[str, ...] = ("state_mean", "default_action", "final_hv")
#: NSGA-II default action expressed as (operator, multiplier, eta_m).
DEFAULT_ACTION: dict[str, Any] = {
    "mutation_operator": "polynomial",
    "multiplier": 1.0,
    "exploration_strength": 20.0,
}
#: Feature block widths of the 76-column layout.
STATE_BLOCK = 60
PROBLEM_BLOCK = len(PROBLEM_FEATURE_NAMES)
RUNTIME_BLOCK = 3
ACTION_BLOCK = 4
FEATURE_DIM = STATE_BLOCK + PROBLEM_BLOCK + RUNTIME_BLOCK + ACTION_BLOCK
#: Runtime-context column names, in order.
RUNTIME_FEATURE_NAMES: tuple[str, ...] = (
    "hv_before",
    "generation_progress",
    "hv_relative_to_reference",
)
#: Normalized action-space ranges used by the nearest-default search.
MULTIPLIER_RANGE: tuple[float, float] = (0.25, 8.0)
ETA_M_RANGE: tuple[float, float] = (2.0, 50.0)
#: Distance penalty when a candidate's operator differs from the default's.
OPERATOR_PENALTY = 1.0
#: Column names carried through to the npz artifacts.
_COLUMN_NAMES: tuple[str, ...] = (
    "X",
    "problem",
    "seed",
    "generation",
    "horizon",
    "candidate_index",
    "candidate_kind",
    "mean_reward",
    "hv_before",
    "mutation_operator",
    "mutation_probability",
    "exploration_strength",
    "mutation_multiplier",
    "n_reps",
    "baseline_value",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments of the v2 intervention dataset builder."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2.75C: build problem-aware intervention datasets "
            "(76-column X) under three advantage definitions."
        )
    )
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR,
                        help="Directory with counterfactual_horizon_{problem}.json "
                        "(default: %(default)s).")
    parser.add_argument("--snapshots-dir", type=str, default=DEFAULT_SNAPSHOTS_DIR,
                        help="Snapshot directory holding history (default: %(default)s).")
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER,
                        help="Fitted encoder JSON for the state block (default: %(default)s).")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Output directory (default: %(default)s).")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS),
                        help="Horizons to extract (default: %(default)s).")
    parser.add_argument(
        "--advantage-baseline", nargs="+", default=["all"],
        choices=["all", *ADVANTAGE_BASELINES],
        help="Advantage definition(s) to write, one npz each (default: %(default)s).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Task 2: default-action stand-in and advantage definitions
# ---------------------------------------------------------------------------


def default_action_distance(action: dict[str, Any], n_vars: int) -> tuple[float, bool]:
    """Normalized distance from a candidate action to the NSGA-II default.

    Args:
        action: Candidate action with ``mutation_operator``,
            ``mutation_probability`` and ``exploration_strength``.
        n_vars: Decision-variable count (multiplier = pm * n_vars).

    Returns:
        ``(distance, same_operator)`` with
        ``hypot(normalized log-multiplier gap, normalized eta_m gap)`` plus
        :data:`OPERATOR_PENALTY` when the operators differ.
    """
    operator = str(action["mutation_operator"])
    multiplier = float(action["mutation_probability"]) * int(n_vars)
    if multiplier <= 0.0:
        multiplier = 1e-12
    delta_multiplier = abs(
        np.log(multiplier / float(DEFAULT_ACTION["multiplier"]))
    ) / np.log(MULTIPLIER_RANGE[1] / MULTIPLIER_RANGE[0])
    delta_exploration = abs(
        float(action["exploration_strength"])
        - float(DEFAULT_ACTION["exploration_strength"])
    ) / (ETA_M_RANGE[1] - ETA_M_RANGE[0])
    same_operator = operator == str(DEFAULT_ACTION["mutation_operator"])
    distance = float(np.hypot(delta_multiplier, delta_exploration))
    if not same_operator:
        distance += OPERATOR_PENALTY
    return distance, same_operator


def select_default_candidate(
    candidates: Sequence[dict[str, Any]], n_vars: int
) -> tuple[int, str]:
    """Index of the candidate standing in for the NSGA-II default action.

    Rules, first match wins (the rule is recorded in the artifact):

    1. ``exact`` — action tuple equal to :data:`DEFAULT_ACTION`;
    2. ``controller`` — the candidate with ``kind == "controller"``;
    3. ``nearest`` — smallest :func:`default_action_distance`.

    Args:
        candidates: Candidate records of one state.
        n_vars: Decision-variable count.

    Returns:
        ``(index, rule)``.

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("cannot select a default candidate from an empty list")
    for index, candidate in enumerate(candidates):
        action = candidate["action"]
        operator = str(action["mutation_operator"])
        multiplier = float(action["mutation_probability"]) * int(n_vars)
        exploration = float(action["exploration_strength"])
        if (
            operator == str(DEFAULT_ACTION["mutation_operator"])
            and abs(multiplier - float(DEFAULT_ACTION["multiplier"])) <= 1e-9
            and abs(exploration - float(DEFAULT_ACTION["exploration_strength"])) <= 1e-9
        ):
            return index, "exact"
    for index, candidate in enumerate(candidates):
        if str(candidate.get("kind", "alternative")) == "controller":
            return index, "controller"
    distances = [
        default_action_distance(candidate["action"], n_vars)[0]
        for candidate in candidates
    ]
    return int(np.argmin(distances)), "nearest"


def proxy_horizon(horizons: Sequence[int]) -> int:
    """Largest requested horizon, used as the ``final_hv`` proxy.

    Raises:
        ValueError: If ``horizons`` is empty.
    """
    values = [int(h) for h in horizons]
    if not values:
        raise ValueError("horizons must not be empty")
    return max(values)


def advantage_targets(
    means_by_horizon: dict[int, list[float]],
    *,
    baseline: str,
    horizons: Sequence[int],
    default_index: int,
) -> tuple[dict[int, list[float]], dict[int, float]]:
    """Advantage of every candidate at every horizon for one definition.

    Args:
        means_by_horizon: ``{horizon: [mean_reward per candidate]}``.
        baseline: One of :data:`ADVANTAGE_BASELINES`.
        horizons: Horizons being extracted.
        default_index: Candidate standing in for the default action.

    Returns:
        ``(targets, baselines)``: ``targets[h][candidate]`` is the advantage
        and ``baselines[h]`` the baseline value behind it. ``final_hv`` maps
        every horizon onto the proxy horizon's target and baseline.

    Raises:
        ValueError: If ``baseline`` is unknown or a horizon is missing.
    """
    if baseline not in ADVANTAGE_BASELINES:
        raise ValueError(
            f"unknown advantage baseline {baseline!r}; expected one of "
            f"{list(ADVANTAGE_BASELINES)}"
        )
    horizon_list = [int(h) for h in horizons]
    missing = [h for h in horizon_list if h not in means_by_horizon]
    if missing:
        raise ValueError(f"horizons {missing} are missing from the state record")
    if baseline == "final_hv":
        proxy = proxy_horizon(horizon_list)
        proxy_means = means_by_horizon[proxy]
        proxy_baseline = float(np.mean(proxy_means))
        return (
            {
                horizon: [value - proxy_baseline for value in proxy_means]
                for horizon in horizon_list
            },
            {horizon: proxy_baseline for horizon in horizon_list},
        )
    targets: dict[int, list[float]] = {}
    baselines: dict[int, float] = {}
    for horizon in horizon_list:
        means = means_by_horizon[horizon]
        value = (
            float(np.mean(means))
            if baseline == "state_mean"
            else float(means[default_index])
        )
        baselines[horizon] = value
        targets[horizon] = [mean - value for mean in means]
    return targets, baselines


# ---------------------------------------------------------------------------
# Task 1: problem and runtime context blocks
# ---------------------------------------------------------------------------


def problem_context_block(
    problem: Any, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Z-scored 9-dim problem descriptor of one benchmark."""
    vector = np.asarray(problem_feature_vector(problem), dtype=np.float64)
    return (vector - mean) / std


def runtime_context_block(
    *,
    hv_before: float,
    generation: int,
    max_generation: int,
    hv_reference: float,
) -> np.ndarray:
    """Three runtime context features of one snapshot.

    ``[hv_before, generation / max_generation, hv_before / hv_reference]``.

    Raises:
        ValueError: If ``max_generation`` < 1 or ``hv_reference`` <= 0.
    """
    if int(max_generation) < 1:
        raise ValueError(f"max_generation must be >= 1, got {max_generation}")
    if not float(hv_reference) > 0.0:
        raise ValueError(f"hv_reference must be positive, got {hv_reference}")
    return np.asarray(
        [
            float(hv_before),
            float(generation) / float(max_generation),
            float(hv_before) / float(hv_reference),
        ],
        dtype=np.float64,
    )


def feature_layout() -> list[dict[str, Any]]:
    """Machine-readable column layout of the 76-column feature matrix."""
    return [
        {"name": "state", "start": 0, "stop": STATE_BLOCK,
         "detail": "StateEncoder.transform(history): window x 6 state features"},
        {"name": "problem", "start": STATE_BLOCK,
         "stop": STATE_BLOCK + PROBLEM_BLOCK,
         "columns": list(PROBLEM_FEATURE_NAMES),
         "detail": "problem_feature_vector, z-scored over the built problems"},
        {"name": "runtime", "start": STATE_BLOCK + PROBLEM_BLOCK,
         "stop": STATE_BLOCK + PROBLEM_BLOCK + RUNTIME_BLOCK,
         "columns": list(RUNTIME_FEATURE_NAMES),
         "detail": "hv_before, generation/max_generation, hv_before/hv_reference"},
        {"name": "action", "start": STATE_BLOCK + PROBLEM_BLOCK + RUNTIME_BLOCK,
         "stop": FEATURE_DIM,
         "columns": ["mutation_multiplier", "exploration_strength",
                     "onehot_polynomial", "onehot_gaussian"],
         "detail": "identical layout to build_outcome_samples"},
    ]


def _reference_hypervolume(payload: dict[str, Any], problem: Any) -> float:
    """Reference-front hypervolume under the intervention file's own ref point."""
    config = payload.get("config", {})
    ref_point = np.asarray(config.get("ref_point", [1.1, 1.1]), dtype=float)
    n_points = int(config.get("n_reference_points", 200))
    return float(hypervolume(problem.reference_front(n_points=n_points), ref_point))


def extract_state_rows(
    payload: dict[str, Any],
    encoder: Any,
    *,
    horizons: Sequence[int],
    snapshots_dir: str | Path,
    problem_mean: np.ndarray,
    problem_std: np.ndarray,
) -> dict[str, Any]:
    """Extract every ``(state, horizon, candidate)`` row of one intervention file.

    Args:
        payload: Parsed ``counterfactual_horizon_{problem}.json``.
        encoder: Fitted :class:`~controller.state_encoder.StateEncoder`.
        horizons: Requested horizons.
        snapshots_dir: Snapshot directory used for the ``history`` join.
        problem_mean: Mean of the problem descriptors (z-scoring).
        problem_std: Std of the problem descriptors (z-scoring).

    Returns:
        Dict with ``columns`` (76-dim ``X`` plus the meta columns),
        ``targets`` (``{baseline: {horizon: [advantage per row]}}`` in the same
        row order as ``columns``), ``baselines`` (baseline value per
        baseline/horizon), ``default_rules`` (selection rule per state),
        ``horizons``, ``hv_reference`` and ``max_generation``.

    Raises:
        ValueError: If no requested horizon is present in the file.
        FileNotFoundError: If a snapshot history is missing.
    """
    problem_name = str(payload["problem"])
    problem = get_problem(problem_name)
    n_vars = int(problem.n_vars)
    config = payload.get("config", {})
    file_horizons = {int(h) for h in config.get("horizons", [])}
    used_horizons = [int(h) for h in horizons if int(h) in file_horizons]
    if not used_horizons:
        raise ValueError(
            f"{problem_name}: file horizons {sorted(file_horizons)} contain none "
            f"of the requested horizons {[int(h) for h in horizons]}"
        )
    max_generation = int(config.get("generations", 0))
    hv_reference = _reference_hypervolume(payload, problem)
    problem_block = problem_context_block(problem, problem_mean, problem_std)

    columns: dict[str, list[Any]] = {name: [] for name in _COLUMN_NAMES}
    columns["rewards"] = []
    targets: dict[str, dict[int, list[float]]] = {
        baseline: {horizon: [] for horizon in used_horizons}
        for baseline in ADVANTAGE_BASELINES
    }
    baselines: dict[str, dict[int, list[float]]] = {
        baseline: {horizon: [] for horizon in used_horizons}
        for baseline in ADVANTAGE_BASELINES
    }
    default_rules: list[dict[str, Any]] = []
    for state in payload["states"]:
        seed = int(state["seed"])
        generation = int(state["generation"])
        history = load_snapshot_history(snapshots_dir, problem_name, seed, generation)
        state_block = np.asarray(encoder.transform(history), dtype=np.float64)
        hv_before = float(state.get("state_metrics", {}).get("hv", float("nan")))
        effective_max = max_generation if max_generation > 0 else max(generation, 1)
        runtime_block = runtime_context_block(
            hv_before=hv_before,
            generation=generation,
            max_generation=effective_max,
            hv_reference=hv_reference,
        )
        candidates = list(state["candidates"])
        default_index, rule = select_default_candidate(candidates, n_vars)
        default_rules.append(
            {
                "state": f"{problem_name}|seed{seed}|gen{generation}",
                "rule": rule,
                "index": int(default_index),
            }
        )
        means_by_horizon = {
            horizon: [float(c["mean_reward"][str(horizon)]) for c in candidates]
            for horizon in used_horizons
        }
        per_baseline = {
            baseline: advantage_targets(
                means_by_horizon,
                baseline=baseline,
                horizons=used_horizons,
                default_index=default_index,
            )
            for baseline in ADVANTAGE_BASELINES
        }
        for horizon in used_horizons:
            for candidate_index, candidate in enumerate(candidates):
                action = candidate["action"]
                features = action_features(action, n_vars)
                columns["X"].append(
                    np.concatenate(
                        [state_block, problem_block, runtime_block, features]
                    )
                )
                columns["problem"].append(problem_name)
                columns["seed"].append(seed)
                columns["generation"].append(generation)
                columns["horizon"].append(int(horizon))
                columns["candidate_index"].append(
                    int(candidate.get("index", candidate_index))
                )
                columns["candidate_kind"].append(
                    str(candidate.get("kind", "alternative"))
                )
                columns["mean_reward"].append(
                    means_by_horizon[horizon][candidate_index]
                )
                columns["hv_before"].append(hv_before)
                columns["mutation_operator"].append(str(action["mutation_operator"]))
                columns["mutation_probability"].append(
                    float(action["mutation_probability"])
                )
                columns["exploration_strength"].append(
                    float(action["exploration_strength"])
                )
                columns["mutation_multiplier"].append(float(features[0]))
                per_rep = [float(v) for v in candidate["reward"][str(horizon)]]
                columns["n_reps"].append(len(per_rep))
                columns["rewards"].append(per_rep)
                for baseline in ADVANTAGE_BASELINES:
                    target_map, baseline_map = per_baseline[baseline]
                    targets[baseline][horizon].append(
                        float(target_map[horizon][candidate_index])
                    )
                    baselines[baseline][horizon].append(float(baseline_map[horizon]))
                    if baseline == "state_mean":
                        # v1 parity: the baseline behind the primary target
                        columns["baseline_value"].append(float(baseline_map[horizon]))
    return {
        "columns": columns,
        "targets": targets,
        "baselines": baselines,
        "default_rules": default_rules,
        "horizons": used_horizons,
        "hv_reference": hv_reference,
        "max_generation": max_generation,
    }


# ---------------------------------------------------------------------------
# artifact assembly
# ---------------------------------------------------------------------------


def _pad_rewards(rewards: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Ragged replicate rewards -> padded matrix plus lengths."""
    lengths = np.asarray([len(row) for row in rewards], dtype=np.int64)
    if lengths.size == 0:
        return np.zeros((0, 0), dtype=np.float64), lengths
    width = int(lengths.max())
    matrix = np.full((len(rewards), width), np.nan, dtype=np.float64)
    for index, row in enumerate(rewards):
        matrix[index, : len(row)] = np.asarray(row, dtype=np.float64)
    return matrix, lengths


def _arrays(
    columns: dict[str, list[Any]], y_adv: Sequence[float]
) -> dict[str, np.ndarray]:
    """npz array set of one baseline file (v1 keys plus the runtime block)."""
    n_samples = len(y_adv)
    rewards, lengths = _pad_rewards(columns["rewards"])
    matrix = np.asarray(columns["X"], dtype=np.float64).reshape(n_samples, -1)
    return {
        "X": matrix,
        "y_adv": np.asarray(y_adv, dtype=np.float64),
        "problem": np.asarray(columns["problem"], dtype="U16"),
        "seed": np.asarray(columns["seed"], dtype=np.int64),
        "generation": np.asarray(columns["generation"], dtype=np.int64),
        "horizon": np.asarray(columns["horizon"], dtype=np.int64),
        "candidate_index": np.asarray(columns["candidate_index"], dtype=np.int64),
        "candidate_kind": np.asarray(columns["candidate_kind"], dtype="U16"),
        "mean_reward": np.asarray(columns["mean_reward"], dtype=np.float64),
        "hv_before": np.asarray(columns["hv_before"], dtype=np.float64),
        "mutation_operator": np.asarray(columns["mutation_operator"], dtype="U16"),
        "mutation_probability": np.asarray(
            columns["mutation_probability"], dtype=np.float64
        ),
        "exploration_strength": np.asarray(
            columns["exploration_strength"], dtype=np.float64
        ),
        "mutation_multiplier": np.asarray(
            columns["mutation_multiplier"], dtype=np.float64
        ),
        "rewards": rewards,
        "n_reps": lengths,
        "baseline_value": np.asarray(columns["baseline_value"], dtype=np.float64),
        "runtime": matrix[
            :, STATE_BLOCK + PROBLEM_BLOCK : STATE_BLOCK + PROBLEM_BLOCK + RUNTIME_BLOCK
        ],
        "problem_features": matrix[:, STATE_BLOCK : STATE_BLOCK + PROBLEM_BLOCK],
        "state_block": matrix[:, :STATE_BLOCK],
        "action_features": matrix[:, STATE_BLOCK + PROBLEM_BLOCK + RUNTIME_BLOCK :],
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    """Mean/std/min/max of one advantage column."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _rule_summary(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count how often each default-action selection rule fired."""
    summary: dict[str, int] = {}
    for record in records:
        key = str(record["rule"])
        summary[key] = summary.get(key, 0) + 1
    return dict(sorted(summary.items()))


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    """Build the problem-aware datasets and write the artifacts.

    Args:
        args: Parsed arguments from :func:`parse_args`.

    Returns:
        The meta payload written to ``intervention_meta_v2.json``.

    Raises:
        FileNotFoundError: If the encoder or an input file is missing.
        ValueError: If no sample can be built for the requested horizons.
    """
    input_dir = Path(args.input_dir)
    snapshots_dir = Path(args.snapshots_dir)
    out_dir = Path(args.out_dir)
    encoder_path = Path(args.encoder)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    if not encoder_path.is_file():
        raise FileNotFoundError(f"encoder not found: {encoder_path}")
    encoder = StateEncoder.load(encoder_path)
    if int(getattr(encoder, "window", 0)) * 6 != STATE_BLOCK:
        raise ValueError(
            f"encoder window {getattr(encoder, 'window', None)} does not produce "
            f"the {STATE_BLOCK}-column state block this builder declares"
        )
    files = [
        path
        for path in sorted(input_dir.glob("counterfactual_horizon_*.json"))
        if path.name != "counterfactual_horizon.json"
    ]
    if not files:
        raise FileNotFoundError(
            f"no counterfactual_horizon_*.json in {input_dir}; run "
            f"experiments/counterfactual_actions.py evaluate-horizon first"
        )
    baselines = (
        list(ADVANTAGE_BASELINES)
        if "all" in args.advantage_baseline
        else [b for b in ADVANTAGE_BASELINES if b in set(args.advantage_baseline)]
    )
    #: The v1-style ``per_horizon`` summary reports the advantage spread of the
    #: first selected definition (its SNR components are target-independent).
    primary_baseline = baselines[0]

    payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as fh:
            payloads.append((path, json.load(fh)))
    problems = [get_problem(str(payload["problem"])) for _, payload in payloads]
    descriptors = np.vstack(
        [
            np.asarray(problem_feature_vector(problem), dtype=np.float64)
            for problem in problems
        ]
    )
    problem_std = descriptors.std(axis=0)
    problem_std = np.where(problem_std > 0.0, problem_std, 1.0)
    problem_mean = descriptors.mean(axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_problem: dict[str, Any] = {}
    columns_by_problem: dict[str, dict[str, list[Any]]] = {}
    targets_by_problem: dict[str, dict[str, list[float]]] = {
        baseline: {} for baseline in baselines
    }
    baseline_report: dict[str, Any] = {
        baseline: {"n_samples": 0, "per_horizon": {}, "per_problem": {}}
        for baseline in baselines
    }
    rule_counts: dict[str, int] = {}
    used_files: list[str] = []
    skipped: list[dict[str, str]] = []

    for (path, payload), problem in zip(payloads, problems):
        problem_name = str(payload["problem"])
        try:
            extracted = extract_state_rows(
                payload,
                encoder,
                horizons=args.horizons,
                snapshots_dir=snapshots_dir,
                problem_mean=problem_mean,
                problem_std=problem_std,
            )
        except ValueError as exc:
            skipped.append({"file": path.name, "reason": str(exc)})
            print(f"[skip] {path.name}: {exc}")
            continue
        columns = extracted["columns"]
        horizons = extracted["horizons"]
        columns_by_problem[problem_name] = columns
        used_files.append(path.name)
        for record in extracted["default_rules"]:
            key = f"{problem_name}:{record['rule']}"
            rule_counts[key] = rule_counts.get(key, 0) + 1

        # v1-compatible per-horizon statistics of the realized rewards, on the
        # state_mean target (target-independent SNR + that baseline's spread).
        state_columns = dict(columns)
        state_columns["y_adv"] = [
            value
            for horizon in horizons
            for value in extracted["targets"][primary_baseline][horizon]
        ]
        per_horizon = summarize_groups(state_columns)
        for entry in per_horizon.values():
            entry["hv_reference"] = extracted["hv_reference"]
        per_problem[problem_name] = {
            "n_samples": len(columns["problem"]),
            "n_states": len(extracted["default_rules"]),
            "horizons": list(horizons),
            "hv_reference": float(extracted["hv_reference"]),
            "max_generation": int(extracted["max_generation"]),
            "default_action_rules": _rule_summary(extracted["default_rules"]),
            "per_horizon": per_horizon,
        }
        for baseline in baselines:
            flat = [
                value
                for horizon in horizons
                for value in extracted["targets"][baseline][horizon]
            ]
            targets_by_problem[baseline][problem_name] = flat
            baseline_report[baseline]["n_samples"] += len(flat)
            baseline_report[baseline]["per_problem"][problem_name] = _distribution(flat)
            for horizon in horizons:
                baseline_report[baseline]["per_horizon"].setdefault(
                    str(horizon), []
                ).extend(extracted["targets"][baseline][horizon])
        print(
            f"[build] {problem_name}: {len(columns['problem'])} samples, "
            f"{len(extracted['default_rules'])} states, horizons={horizons}, "
            f"rules={_rule_summary(extracted['default_rules'])}"
        )

    if not columns_by_problem:
        raise ValueError(
            f"no samples built from {input_dir} (skipped: "
            f"{[entry['reason'] for entry in skipped]})"
        )
    ordered_problems = sorted(columns_by_problem)
    total_samples = sum(len(columns_by_problem[name]["problem"]) for name in ordered_problems)

    # pooled columns for the v1-style meta summary (per_problem files are
    # written from the per-problem columns, never from this pool)
    pooled: dict[str, list[Any]] = {name: [] for name in _COLUMN_NAMES}
    pooled["rewards"] = []
    for name in ordered_problems:
        for key in _COLUMN_NAMES:
            pooled[key].extend(columns_by_problem[name][key])
        pooled["rewards"].extend(columns_by_problem[name]["rewards"])
    pooled["y_adv"] = [
        value
        for name in ordered_problems
        for value in targets_by_problem[primary_baseline][name]
    ]
    per_horizon_block = summarize_groups(pooled)

    written: dict[str, list[str]] = {baseline: [] for baseline in baselines}
    per_baseline_meta: dict[str, dict[str, Any]] = {}
    for baseline in baselines:
        baseline_dir = out_dir / baseline
        baseline_dir.mkdir(parents=True, exist_ok=True)
        for problem_name in ordered_problems:
            arrays = _arrays(
                columns_by_problem[problem_name],
                targets_by_problem[baseline][problem_name],
            )
            # one directory per definition, so the Phase-2.75 trainer
            # (glob: intervention_dataset_*.npz) can never mix definitions
            out_path = baseline_dir / f"intervention_dataset_{problem_name}.npz"
            np.savez_compressed(out_path, **arrays)
            written[baseline].append(out_path.relative_to(out_dir).as_posix())
        per_baseline_meta[baseline] = {
            "config": {
                "advantage_baseline": baseline,
                "encoder": str(encoder_path),
                "horizons": [int(h) for h in args.horizons],
                "feature_dim": FEATURE_DIM,
                "input_dir": str(input_dir),
                "snapshots_dir": str(snapshots_dir),
                "window": int(getattr(encoder, "window", 0)),
            },
            "n_samples": int(total_samples),
            "n_states": int(sum(entry["n_states"] for entry in per_problem.values())),
            "feature_dim": FEATURE_DIM,
            "files_used": used_files,
            "files_skipped": skipped,
            "per_problem": per_problem,
            "per_horizon": per_horizon_block,
            "per_horizon_baseline": primary_baseline,
            "advantage_distribution": baseline_report[baseline],
            "feature_layout": feature_layout(),
        }
        with (baseline_dir / "intervention_meta.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(per_baseline_meta[baseline], fh, indent=2, ensure_ascii=False)

    for baseline in baselines:
        for horizon, values in list(baseline_report[baseline]["per_horizon"].items()):
            baseline_report[baseline]["per_horizon"][horizon] = _distribution(values)

    meta: dict[str, Any] = {
        "config": {
            "input_dir": str(input_dir),
            "snapshots_dir": str(snapshots_dir),
            "encoder": str(encoder_path),
            "out_dir": str(out_dir),
            "horizons": [int(h) for h in args.horizons],
            "advantage_baselines": baselines,
            "advantage_baseline_flag": list(args.advantage_baseline),
            "window": int(getattr(encoder, "window", 0)),
            "feature_dim": FEATURE_DIM,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "n_samples": int(total_samples),
        "n_states": int(sum(entry["n_states"] for entry in per_problem.values())),
        "feature_dim": FEATURE_DIM,
        "files_used": used_files,
        "files_skipped": skipped,
        "per_problem": per_problem,
        "per_horizon": per_horizon_block,
        "per_horizon_baseline": primary_baseline,
        "feature_layout": feature_layout(),
        "problem_feature_names": list(PROBLEM_FEATURE_NAMES),
        "problem_feature_mean": problem_mean.tolist(),
        "problem_feature_std": problem_std.tolist(),
        "runtime_feature_names": list(RUNTIME_FEATURE_NAMES),
        "advantage_baselines": baseline_report,
        "default_action": {
            "definition": DEFAULT_ACTION,
            "selection_order": ["exact", "controller", "nearest"],
            "rule_counts": dict(sorted(rule_counts.items())),
            "nearest_metric": (
                "hypot(normalized log-multiplier gap, normalized eta_m gap) + "
                f"{OPERATOR_PENALTY} when the operator differs"
            ),
        },
        "proxy_horizon": (
            proxy_horizon(args.horizons) if "final_hv" in baselines else None
        ),
        "limitation": (
            "final_hv uses the largest requested horizon as a proxy for the "
            "branch end state: the corpus only branches 5/10/20 generations, so "
            "the target is not a true final-generation HV"
            if "final_hv" in baselines
            else None
        ),
        "artifacts": written,
        "artifact_layout": (
            "{baseline}/intervention_dataset_{problem}.npz plus "
            "{baseline}/intervention_meta.json (one directory per advantage "
            "definition, so the Phase-2.75 trainer cannot mix them); the "
            "combined summary is this file"
        ),
    }
    meta_path = out_dir / "intervention_meta_v2.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(
        f"[done] {total_samples} samples, dim={FEATURE_DIM}, "
        f"baselines={baselines} -> {meta_path}"
    )
    for baseline in baselines:
        pooled_values = [
            value
            for name in ordered_problems
            for value in targets_by_problem[baseline][name]
        ]
        overall = _distribution(pooled_values)
        print(
            f"[baseline] {baseline}: n={overall['n']} mean={overall['mean']} "
            f"std={overall['std']} min={overall['min']} max={overall['max']}"
        )
    return meta


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point."""
    return run_build(parse_args(argv))


if __name__ == "__main__":
    main()
