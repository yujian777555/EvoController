from __future__ import annotations

"""Phase 2.75, Task 2: turn same-snapshot interventions into a dataset.

Phase 2B's counterfactual evaluator (``evaluate-horizon``) branches one
population snapshot into several candidate actions with independent,
reproducible rollouts. That is exactly the intervention data Phase 2.75
needs, but the artifact stores only what the evaluator itself needed: the
candidate actions, their per-horizon rewards and the snapshot identity —
**not** the encoded state history.

This script joins both sides:

* ``{input_dir}/counterfactual_horizon_{problem}.json`` — per-state
  ``candidates[]`` with ``action``, ``reward{h: [per-rep]}`` and
  ``mean_reward{h}`` (the intervention rollout);
* ``{snapshots_dir}/{problem}__seed{seed}__gen{generation}.pkl`` — the
  harvested ``history`` (merged state+reward dicts) of the same state,
  which :meth:`controller.state_encoder.StateEncoder.transform` consumes
  directly.

and produces, per problem:

* ``y_adv`` — the **advantage** of each candidate at each horizon, i.e. its
  ``mean_reward`` minus a per-``(state, horizon)`` baseline. The default
  ``--baseline state_mean`` uses the mean over all candidates of that state
  (so the advantages are centred and the state-level difficulty is removed
  from the target — the change of objective Phase 2.75 asks for);
  ``oracle_best``, ``worst`` and ``controller`` are available as
  alternatives (see :func:`resolve_baseline`);
* ``X`` — ``encoder.transform(history)`` concatenated with the four action
  features ``[mutation_multiplier, exploration_strength, onehot_polynomial,
  onehot_gaussian]`` where ``mutation_multiplier = mutation_probability *
  n_vars``. The layout is bit-identical to
  :func:`controller.dataset.build_outcome_samples`, which is what both
  predictors are trained on;
* ``meta`` — problem/seed/generation/horizon/candidate index and kind, the
  raw action, ``mean_reward``, the per-replicate rewards (for SNR
  analysis), ``hv_before`` and the baseline value used.

Outputs (``--out-dir``, default ``results/phase2_75``):
``intervention_dataset_{problem}.npz`` (one per problem, arrays listed in
:func:`run_dataset`) and a single ``intervention_meta.json`` holding the
sample counts, per-problem statistics, and the per-horizon action-effect
SNR (between-candidate variance over within-candidate replicate noise).

Example:
    ``python experiments/build_intervention_dataset.py``
    ``python experiments/build_intervention_dataset.py --baseline controller``
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    # Allow ``python experiments/build_intervention_dataset.py`` from the
    # repo root: the script directory (not the repo root) is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import get_problem
from controller.dataset import OPERATOR_TO_INDEX

#: Directory with ``counterfactual_horizon_{problem}.json`` files.
DEFAULT_INPUT_DIR = "results/phase2b/counterfactual"
#: Directory with the harvested snapshot pickles (``history`` source).
DEFAULT_SNAPSHOTS_DIR = "results/phase1_75/snapshots"
#: Fitted encoder of the Phase-2A/2B model.
DEFAULT_ENCODER = "results/phase2_outcome/encoder.json"
#: Output directory (Phase-2.75 isolated).
DEFAULT_OUT_DIR = "results/phase2_75"
#: Advantage horizons to extract (the planner's long-horizon grid).
DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)
#: Baseline definitions of ``advantage = mean_reward - baseline``.
BASELINE_CHOICES: tuple[str, ...] = (
    "state_mean",
    "oracle_best",
    "worst",
    "controller",
)
#: Number of trailing feature columns that carry the action.
N_ACTION_FEATURES = 4


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments of the intervention dataset builder."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2.75: join long-horizon counterfactual interventions with "
            "snapshot histories into an (X, advantage) dataset."
        )
    )
    parser.add_argument(
        "--input-dir", type=str, default=DEFAULT_INPUT_DIR,
        help="Directory with counterfactual_horizon_{problem}.json files "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--snapshots-dir", type=str, default=DEFAULT_SNAPSHOTS_DIR,
        help="Snapshot directory holding history (default: %(default)s).",
    )
    parser.add_argument(
        "--encoder", type=str, default=DEFAULT_ENCODER,
        help="Fitted encoder JSON used for the state block (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=DEFAULT_OUT_DIR,
        help="Output directory for the npz/meta artifacts (default: %(default)s).",
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS),
        help="Advantage horizons to extract from each file (default: %(default)s).",
    )
    parser.add_argument(
        "--baseline", choices=list(BASELINE_CHOICES), default="state_mean",
        help="Per-(state, horizon) baseline of the advantage target: "
        "'state_mean' = mean over all candidates of that state (leave-one-in), "
        "'oracle_best' = best candidate, 'worst' = worst candidate, "
        "'controller' = the controller-kind candidate (default: %(default)s).",
    )
    return parser.parse_args(argv)


def action_features(action: dict[str, Any], n_vars: int) -> np.ndarray:
    """Four action features in the :func:`build_outcome_samples` layout.

    ``[mutation_multiplier, exploration_strength, onehot_polynomial,
    onehot_gaussian]`` with ``mutation_multiplier = mutation_probability *
    n_vars`` (the normalized scale the Phase-2A corpus stores).

    Args:
        action: Action dict with ``mutation_operator``,
            ``mutation_probability`` and ``exploration_strength``.
        n_vars: Decision-variable count of the problem.

    Returns:
        Array of shape ``(4,)``.

    Raises:
        ValueError: If the action's operator is not a supported mutation
            operator.
    """
    operator = str(action["mutation_operator"])
    if operator not in OPERATOR_TO_INDEX:
        raise ValueError(
            f"unsupported mutation_operator {operator!r}; expected one of "
            f"{sorted(OPERATOR_TO_INDEX)}"
        )
    one_hot = [0.0, 0.0]
    one_hot[OPERATOR_TO_INDEX[operator]] = 1.0
    return np.asarray(
        [
            float(action["mutation_probability"]) * int(n_vars),
            float(action["exploration_strength"]),
            *one_hot,
        ],
        dtype=np.float64,
    )


def resolve_baseline(
    mean_rewards: Sequence[float],
    kinds: Sequence[str],
    option: str,
) -> float:
    """Per-``(state, horizon)`` baseline value of the advantage target.

    Args:
        mean_rewards: ``mean_reward`` of every candidate of that state.
        kinds: Candidate kinds (``"controller"`` / ``"alternative"``),
            aligned with ``mean_rewards``.
        option: One of :data:`BASELINE_CHOICES`.

    Returns:
        The baseline value.

    Raises:
        ValueError: If ``mean_rewards`` is empty, ``option`` is unknown, or
            ``"controller"`` is requested without a controller candidate.
    """
    values = np.asarray(list(mean_rewards), dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot resolve a baseline from zero candidates")
    if option == "state_mean":
        return float(values.mean())
    if option == "oracle_best":
        return float(values.max())
    if option == "worst":
        return float(values.min())
    if option == "controller":
        controller_indices = [
            index for index, kind in enumerate(kinds) if str(kind) == "controller"
        ]
        if not controller_indices:
            raise ValueError(
                "baseline 'controller' requested but this state has no "
                "candidate of kind 'controller'"
            )
        return float(values[controller_indices[0]])
    raise ValueError(f"unknown baseline option {option!r}")


def snapshot_path(
    snapshots_dir: str | Path, problem: str, seed: int, generation: int
) -> Path:
    """Path of the snapshot pickle backing one counterfactual state."""
    return (
        Path(snapshots_dir) / f"{problem}__seed{int(seed)}__gen{int(generation)}.pkl"
    )


def load_snapshot_history(
    snapshots_dir: str | Path, problem: str, seed: int, generation: int
) -> list[dict[str, Any]]:
    """Load the merged ``history`` of one snapshot.

    Args:
        snapshots_dir: Directory with harvest pickles.
        problem: Benchmark name.
        seed: Run seed of the snapshot.
        generation: Snapshot generation.

    Returns:
        ``history`` — a list of merged state+reward dicts, oldest first,
        exactly what :meth:`StateEncoder.transform` expects.

    Raises:
        FileNotFoundError: If the snapshot pickle (or its ``history``) is
            missing; the counterfactual file cannot be joined without it.
    """
    import pickle

    path = snapshot_path(snapshots_dir, problem, seed, generation)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing snapshot for state {problem} seed={seed} gen={generation}: "
            f"{path}; the counterfactual file cannot be joined without its "
            f"history (check --snapshots-dir)"
        )
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    history = payload.get("history")
    if history is None:
        raise FileNotFoundError(f"{path} carries no 'history' field")
    return list(history)


def extract_problem_samples(
    payload: dict[str, Any],
    encoder: Any,
    *,
    horizons: Sequence[int],
    baseline: str,
    snapshots_dir: str | Path,
) -> dict[str, list[Any]]:
    """Extract ``(X, y_adv, meta)`` columns from one counterfactual file.

    Args:
        payload: Parsed ``counterfactual_horizon_{problem}.json``.
        encoder: Fitted :class:`controller.state_encoder.StateEncoder`.
        horizons: Requested horizons; those absent from the file are
            skipped.
        baseline: Baseline option (see :func:`resolve_baseline`).
        snapshots_dir: Snapshot directory used for the history join.

    Returns:
        Column lists keyed by array name (see :func:`run_dataset`).

    Raises:
        ValueError: If no requested horizon is present in the file, a
            candidate lacks the requested horizon, or the baseline is
            invalid for a state.
        FileNotFoundError: If a snapshot history is missing.
    """
    problem = str(payload["problem"])
    n_vars = int(get_problem(problem).n_vars)
    file_horizons = {int(h) for h in payload.get("config", {}).get("horizons", [])}
    used_horizons = [int(h) for h in horizons if int(h) in file_horizons]
    if not used_horizons:
        raise ValueError(
            f"{problem}: file horizons {sorted(file_horizons)} contain none of "
            f"the requested horizons {[int(h) for h in horizons]}"
        )
    if baseline not in BASELINE_CHOICES:
        raise ValueError(f"unknown baseline option {baseline!r}")

    columns: dict[str, list[Any]] = {
        key: []
        for key in (
            "X",
            "y_adv",
            "problem",
            "seed",
            "generation",
            "horizon",
            "candidate_index",
            "candidate_kind",
            "mean_reward",
            "hv_before",
            "baseline_value",
            "mutation_operator",
            "mutation_probability",
            "exploration_strength",
            "mutation_multiplier",
            "n_reps",
        )
    }
    rewards_rows: list[list[float]] = []
    for state in payload["states"]:
        seed = int(state["seed"])
        generation = int(state["generation"])
        history = load_snapshot_history(snapshots_dir, problem, seed, generation)
        state_block = np.asarray(encoder.transform(history), dtype=np.float64)
        candidates = list(state["candidates"])
        if not candidates:
            continue
        hv_before = float(state.get("state_metrics", {}).get("hv", float("nan")))
        for horizon in used_horizons:
            key = str(horizon)
            mean_rewards = [
                float(candidate["mean_reward"][key]) for candidate in candidates
            ]
            kinds = [str(candidate.get("kind", "alternative")) for candidate in candidates]
            baseline_value = resolve_baseline(mean_rewards, kinds, baseline)
            for index, candidate in enumerate(candidates):
                action = candidate["action"]
                features = action_features(action, n_vars)
                columns["X"].append(np.concatenate([state_block, features]))
                columns["y_adv"].append(mean_rewards[index] - baseline_value)
                columns["problem"].append(problem)
                columns["seed"].append(seed)
                columns["generation"].append(generation)
                columns["horizon"].append(int(horizon))
                columns["candidate_index"].append(int(candidate.get("index", index)))
                columns["candidate_kind"].append(kinds[index])
                columns["mean_reward"].append(mean_rewards[index])
                columns["hv_before"].append(hv_before)
                columns["baseline_value"].append(baseline_value)
                columns["mutation_operator"].append(str(action["mutation_operator"]))
                columns["mutation_probability"].append(
                    float(action["mutation_probability"])
                )
                columns["exploration_strength"].append(
                    float(action["exploration_strength"])
                )
                columns["mutation_multiplier"].append(float(features[0]))
                per_rep = [float(value) for value in candidate["reward"][key]]
                columns["n_reps"].append(len(per_rep))
                rewards_rows.append(per_rep)
    columns["rewards"] = rewards_rows
    return columns


def action_effect_snr(
    mean_rewards: Sequence[float], rewards: Sequence[Sequence[float]]
) -> tuple[float | None, float | None, float | None]:
    """Between-candidate variance vs within-candidate replicate variance.

    Same definition as the Phase-2B D4 diagnostic, computed for one
    ``(state, horizon)`` group: the between-action variance is the sample
    variance (``ddof=1``) of the per-candidate mean reward, the noise
    variance the mean over candidates of the sample variance across that
    candidate's replicates.

    Args:
        mean_rewards: Per-candidate mean rewards.
        rewards: Per-candidate replicate rewards, aligned.

    Returns:
        ``(between, within, snr)``; all ``None`` when there are fewer than
        two candidates or the replicate variances are undefined;
        ``snr`` is ``None`` when the noise variance is zero.
    """
    means = np.asarray(list(mean_rewards), dtype=np.float64)
    per_rep = [np.asarray(list(row), dtype=np.float64) for row in rewards]
    if means.size < 2 or len(per_rep) != means.size:
        return None, None, None
    between = float(np.var(means, ddof=1))
    usable = [row for row in per_rep if row.size > 1]
    if not usable:
        return between, None, None
    within = float(np.mean([float(np.var(row, ddof=1)) for row in usable]))
    snr = float(between / within) if within > 0.0 else None
    return between, within, snr


def _pad_rewards(rewards: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Ragged replicate rewards -> ``(n, max_reps)`` padded with NaN + lengths."""
    lengths = np.asarray([len(row) for row in rewards], dtype=np.int64)
    if lengths.size == 0:
        return np.zeros((0, 0), dtype=np.float64), lengths
    width = int(lengths.max())
    matrix = np.full((len(rewards), width), np.nan, dtype=np.float64)
    for row_index, row in enumerate(rewards):
        matrix[row_index, : len(row)] = np.asarray(row, dtype=np.float64)
    return matrix, lengths


def _group_key(columns: dict[str, list[Any]], index: int) -> tuple[str, int, int, int]:
    """``(problem, seed, generation, horizon)`` identity of sample ``index``."""
    return (
        str(columns["problem"][index]),
        int(columns["seed"][index]),
        int(columns["generation"][index]),
        int(columns["horizon"][index]),
    )


def _state_key(columns: dict[str, list[Any]], index: int) -> tuple[str, int, int]:
    """``(problem, seed, generation)`` identity of sample ``index``.

    Distinct from :func:`_group_key`: one population state contributes
    samples at every requested horizon, so counting states must ignore the
    horizon column.
    """
    return (
        str(columns["problem"][index]),
        int(columns["seed"][index]),
        int(columns["generation"][index]),
    )


def summarize_groups(columns: dict[str, list[Any]]) -> dict[str, Any]:
    """Per-horizon statistics (sample count, advantage spread, SNR)."""
    groups: dict[tuple[str, int, int, int], list[int]] = {}
    for index in range(len(columns["y_adv"])):
        groups.setdefault(_group_key(columns, index), []).append(index)
    per_horizon: dict[str, Any] = {}
    for horizon in sorted({key[3] for key in groups}):
        advantages: list[float] = []
        between_values: list[float] = []
        within_values: list[float] = []
        n_states = 0
        n_states_with_signal = 0
        for key, indices in groups.items():
            if key[3] != horizon:
                continue
            n_states += 1
            advantages.extend(float(columns["y_adv"][index]) for index in indices)
            means = [float(columns["mean_reward"][index]) for index in indices]
            rewards = [columns["rewards"][index] for index in indices]
            between, within, _ = action_effect_snr(means, rewards)
            if between is not None and within is not None:
                between_values.append(between)
                within_values.append(within)
                if within > 0.0:
                    n_states_with_signal += 1
        between_mean = float(np.mean(between_values)) if between_values else None
        within_mean = float(np.mean(within_values)) if within_values else None
        snr = (
            float(between_mean / within_mean)
            if between_mean is not None and within_mean not in (None, 0.0)
            else None
        )
        advantage_array = np.asarray(advantages, dtype=np.float64)
        per_horizon[str(horizon)] = {
            "n_samples": int(advantage_array.size),
            "n_states": int(n_states),
            "n_states_with_replicate_signal": int(n_states_with_signal),
            "mean_advantage": (
                float(advantage_array.mean()) if advantage_array.size else None
            ),
            "std_advantage": (
                float(advantage_array.std(ddof=1))
                if advantage_array.size > 1
                else 0.0
                if advantage_array.size
                else None
            ),
            "between_action_var": between_mean,
            "within_action_noise_var": within_mean,
            "action_effect_snr": snr,
        }
    return per_horizon


def _column_arrays(columns: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    """Convert the column lists into the npz array set."""
    n_samples = len(columns["y_adv"])
    rewards, lengths = _pad_rewards(columns["rewards"])
    return {
        "X": np.asarray(columns["X"], dtype=np.float64).reshape(
            n_samples, -1
        ) if n_samples else np.zeros((0, 0), dtype=np.float64),
        "y_adv": np.asarray(columns["y_adv"], dtype=np.float64),
        "problem": np.asarray(columns["problem"], dtype="U16"),
        "seed": np.asarray(columns["seed"], dtype=np.int64),
        "generation": np.asarray(columns["generation"], dtype=np.int64),
        "horizon": np.asarray(columns["horizon"], dtype=np.int64),
        "candidate_index": np.asarray(columns["candidate_index"], dtype=np.int64),
        "candidate_kind": np.asarray(columns["candidate_kind"], dtype="U16"),
        "mean_reward": np.asarray(columns["mean_reward"], dtype=np.float64),
        "hv_before": np.asarray(columns["hv_before"], dtype=np.float64),
        "baseline_value": np.asarray(columns["baseline_value"], dtype=np.float64),
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
    }


def _problem_stats(
    columns: dict[str, list[Any]], per_horizon: dict[str, Any]
) -> dict[str, Any]:
    """Per-problem summary block of the meta JSON."""
    advantages = np.asarray(columns["y_adv"], dtype=np.float64)
    return {
        "n_samples": int(advantages.size),
        "n_states": len(
            {
                _state_key(columns, index)
                for index in range(len(columns["y_adv"]))
            }
        ),
        "n_candidates_per_state": sorted(
            {
                sum(
                    1
                    for index in range(len(columns["y_adv"]))
                    if _group_key(columns, index) == key
                )
                for key in {
                    _group_key(columns, index)
                    for index in range(len(columns["y_adv"]))
                }
            }
        ),
        "horizons": sorted({int(h) for h in columns["horizon"]}),
        "mean_advantage": (
            float(advantages.mean()) if advantages.size else None
        ),
        "std_advantage": (
            float(advantages.std(ddof=1)) if advantages.size > 1 else None
        ),
        "mean_hv_before": (
            float(np.mean(columns["hv_before"])) if advantages.size else None
        ),
        "per_horizon": per_horizon,
    }


def run_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Build the intervention dataset and write the npz/meta artifacts.

    Args:
        args: Parsed arguments from :func:`parse_args`.

    Returns:
        The meta payload as written to ``intervention_meta.json``.

    Raises:
        FileNotFoundError: If the encoder or a snapshot history is missing.
        ValueError: If no counterfactual file yields samples for the
            requested horizons.
    """
    from controller.state_encoder import StateEncoder

    input_dir = Path(args.input_dir)
    snapshots_dir = Path(args.snapshots_dir)
    out_dir = Path(args.out_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    encoder_path = Path(args.encoder)
    if not encoder_path.is_file():
        raise FileNotFoundError(f"encoder not found: {encoder_path}")
    encoder = StateEncoder.load(encoder_path)
    horizons = sorted({int(h) for h in args.horizons})
    if not horizons:
        raise ValueError("--horizons must not be empty")

    files = sorted(input_dir.glob("counterfactual_horizon_*.json"))
    if not files:
        raise FileNotFoundError(
            f"no counterfactual_horizon_*.json in {input_dir}; run "
            f"'experiments/counterfactual_actions.py evaluate-horizon' first"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped: list[dict[str, str]] = []
    used: list[str] = []
    per_problem: dict[str, Any] = {}
    total_samples = 0
    all_columns: dict[str, list[Any]] = {key: [] for key in _EMPTY_COLUMNS}
    all_columns["rewards"] = []
    for path in files:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        problem = str(payload.get("problem", path.stem))
        try:
            columns = extract_problem_samples(
                payload,
                encoder,
                horizons=horizons,
                baseline=str(args.baseline),
                snapshots_dir=snapshots_dir,
            )
        except ValueError as exc:
            skipped.append({"file": path.name, "reason": str(exc)})
            print(f"[skip] {path.name}: {exc}")
            continue
        n_samples = len(columns["y_adv"])
        if n_samples == 0:
            skipped.append({"file": path.name, "reason": "no candidates found"})
            continue
        arrays = _column_arrays(columns)
        out_path = out_dir / f"intervention_dataset_{problem}.npz"
        np.savez_compressed(out_path, **arrays)
        per_horizon = summarize_groups(columns)
        per_problem[problem] = _problem_stats(columns, per_horizon)
        per_problem[problem]["dataset_file"] = out_path.name
        for key in _EMPTY_COLUMNS:
            all_columns[key].extend(columns[key])
        all_columns["rewards"].extend(columns["rewards"])
        total_samples += n_samples
        used.append(path.name)
        print(
            f"[build] {problem}: {n_samples} samples, "
            f"{per_problem[problem]['n_states']} states -> {out_path.name}"
        )

    if total_samples == 0:
        raise ValueError(
            f"no samples built from {input_dir}: every file was skipped "
            f"({[entry['reason'] for entry in skipped]}); long-horizon "
            f"evaluation may still be running"
        )

    pooled_per_horizon = summarize_groups(all_columns)
    meta = {
        "config": {
            "input_dir": str(input_dir),
            "snapshots_dir": str(snapshots_dir),
            "encoder": str(encoder_path),
            "out_dir": str(out_dir),
            "horizons": horizons,
            "baseline": str(args.baseline),
            "n_action_features": N_ACTION_FEATURES,
            "window": int(getattr(encoder, "window", 0)),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "n_samples": int(total_samples),
        "n_states": int(
            len(
                {
                    _state_key(all_columns, index)
                    for index in range(len(all_columns["y_adv"]))
                }
            )
        ),
        "feature_dim": int(
            np.asarray(all_columns["X"], dtype=np.float64).reshape(
                total_samples, -1
            ).shape[1]
        ),
        "files_used": used,
        "files_skipped": skipped,
        "per_problem": per_problem,
        "per_horizon": pooled_per_horizon,
    }
    meta_path = out_dir / "intervention_meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(
        f"[done] {total_samples} samples, {meta['n_states']} states, "
        f"dim={meta['feature_dim']} -> {meta_path}"
    )
    for horizon, entry in pooled_per_horizon.items():
        print(
            f"[snr] h={horizon}: n={entry['n_samples']} "
            f"between={entry['between_action_var']} "
            f"within={entry['within_action_noise_var']} "
            f"snr={entry['action_effect_snr']}"
        )
    return meta


#: Column names of :func:`extract_problem_samples` (``rewards`` is appended).
_EMPTY_COLUMNS: tuple[str, ...] = (
    "X",
    "y_adv",
    "problem",
    "seed",
    "generation",
    "horizon",
    "candidate_index",
    "candidate_kind",
    "mean_reward",
    "hv_before",
    "baseline_value",
    "mutation_operator",
    "mutation_probability",
    "exploration_strength",
    "mutation_multiplier",
    "n_reps",
)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point."""
    return run_dataset(parse_args(argv))


if __name__ == "__main__":
    main()
