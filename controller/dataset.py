from __future__ import annotations

"""Builds supervised ``history -> log mutation probability`` datasets.

Phase 0 trajectories are turned into regression samples for the MLP
controller. For transition ``t >= 1`` of a trajectory, the input is the
encoded window of merged state+reward dicts of generations
``[t - window, t)`` — strictly the information available when the action
of generation ``t`` was chosen — the target is ``log`` of the mutation
probability used at ``t``, and the sample weight rewards above-average
outcomes: ``max(r_t - mean(r_trajectory), 0) + 1e-6`` with
``r_t = delta_hv_t + delta_igd_t``.

Phase 1.5 adds :func:`build_multihead_samples`, which targets the full
action triple ``(mutation_operator, mutation_probability,
exploration_strength)`` with the same history alignment and advantage
weights.

Phase 1.75 adds :func:`load_trajectory_records` (transitions plus run-level
metadata such as ``n_vars``) and the ``mutation_target="multiplier"`` mode
of :func:`build_multihead_samples`, which regresses the log of the
normalized mutation multiplier ``pm * n_vars`` (see
:mod:`controller.action_normalization`) instead of the absolute log
mutation probability.
"""

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from controller.state_encoder import STATE_FEATURES, ProblemAwareEncoder, StateEncoder


def load_trajectories(directory: str | Path) -> list[list[dict[str, Any]]]:
    """Load every recorded trajectory found in a directory.

    Reads each ``*.json`` file except ``index.json``, extracts its
    ``transitions`` list, and sorts it by ``generation``. Files are
    processed in sorted filename order for determinism.

    Args:
        directory: Directory containing trajectory JSON files as written
            by ``EvolutionRecorder.save``.

    Returns:
        List of trajectories, each a generation-sorted list of transition
        dicts.

    Raises:
        NotADirectoryError: If ``directory`` is not an existing directory.
        ValueError: If a trajectory file has no ``transitions`` list.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")
    trajectories: list[list[dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "index.json":
            continue
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        transitions = payload.get("transitions")
        if transitions is None:
            raise ValueError(f"{path} has no 'transitions' list")
        trajectories.append(
            sorted(transitions, key=lambda t: int(t["generation"]))
        )
    return trajectories


def load_trajectory_records(directory: str | Path) -> list[dict[str, Any]]:
    """Load every recorded trajectory with its run-level metadata.

    Phase-1.75 variant of :func:`load_trajectories` that keeps the
    per-run context the normalized mutation target needs (problem name and
    ``n_vars``) alongside the transitions. Reads each ``*.json`` file
    except ``index.json`` in sorted filename order (deterministic) and
    sorts each trajectory's transitions by ``generation``.

    Args:
        directory: Directory containing trajectory JSON files as written
            by ``EvolutionRecorder.save`` (via
            ``experiments.generate_dataset``).

    Returns:
        List of records, each a dict with keys ``"problem"`` (benchmark
        name from the stored config), ``"n_vars"`` (decision-variable
        count from the stored config), ``"seed"``, ``"runtime_sec"``,
        ``"config"`` (the full stored config dict), and ``"transitions"``
        (generation-sorted transition list).

    Raises:
        NotADirectoryError: If ``directory`` is not an existing directory.
        ValueError: If a trajectory file has no ``transitions`` list or
            its config lacks ``"problem"``/``"n_vars"``.
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
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"{path} has no 'config' dict")
        missing = [key for key in ("problem", "n_vars") if key not in config]
        if missing:
            raise ValueError(f"{path} config is missing required keys: {missing}")
        records.append(
            {
                "problem": str(config["problem"]),
                "n_vars": int(config["n_vars"]),
                "seed": int(payload["seed"]),
                "runtime_sec": float(payload["runtime_sec"]),
                "config": config,
                "transitions": sorted(
                    transitions, key=lambda t: int(t["generation"])
                ),
            }
        )
    return records


def merge_state_reward(transition: dict[str, Any]) -> dict[str, float]:
    """Merge one transition's state and reward into a feature dict.

    Args:
        transition: One recorded ``(state, action, reward)`` transition.

    Returns:
        Dict with exactly the six ``STATE_FEATURES`` keys: ``hv``,
        ``igd``, ``diversity`` and ``generation`` from the state, plus
        ``delta_hv`` and ``delta_igd`` from the reward.
    """
    state = transition["state"]
    reward = transition["reward"]
    return {
        key: float(state[key]) if key in state else float(reward[key])
        for key in STATE_FEATURES
    }


def build_supervised_samples(
    trajectories: list[list[dict[str, Any]]],
    encoder: StateEncoder,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn trajectories into weighted regression samples.

    For each trajectory ``j`` and each transition index ``t`` in
    ``1..len-1``:

    * input: ``encoder.transform`` of the merged dicts of transitions
      ``[t - window, t)`` (history strictly before the action at ``t``);
    * target: ``log`` of ``mutation_probability`` of transition ``t``;
    * weight: ``max(r_t - mean_r_j, 0) + 1e-6`` where
      ``r_t = delta_hv_t + delta_igd_t`` and ``mean_r_j`` is the mean of
      ``r`` over all transitions of trajectory ``j``.

    Args:
        trajectories: Generation-sorted trajectories, e.g. from
            :func:`load_trajectories`.
        encoder: Fitted state encoder.
        window: History window in generations; should match
            ``encoder.window``.

    Returns:
        Tuple ``(X, y, w, traj_ids)``: ``X`` of shape
        ``(n, window * 6)``, ``y``/``w`` of shape ``(n,)`` and integer
        ``traj_ids`` of shape ``(n,)`` identifying the source trajectory
        of each sample. Empty arrays (with correct shapes) if no
        trajectory has at least two transitions.

    Raises:
        ValueError: If ``window`` < 1 or a mutation probability is <= 0.
    """
    if int(window) < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    x_rows: list[np.ndarray] = []
    y_vals: list[float] = []
    w_vals: list[float] = []
    id_vals: list[int] = []
    for j, trajectory in enumerate(trajectories):
        if len(trajectory) < 2:
            continue
        merged = [merge_state_reward(t) for t in trajectory]
        rewards = np.asarray(
            [m["delta_hv"] + m["delta_igd"] for m in merged], dtype=float
        )
        mean_reward = float(rewards.mean())
        for t in range(1, len(trajectory)):
            history = merged[max(0, t - int(window)) : t]
            pm = float(trajectory[t]["action"]["mutation_probability"])
            if pm <= 0.0:
                raise ValueError(
                    f"mutation_probability must be > 0 for log target, "
                    f"got {pm} at trajectory {j}, transition {t}"
                )
            x_rows.append(encoder.transform(history))
            y_vals.append(math.log(pm))
            w_vals.append(max(float(rewards[t]) - mean_reward, 0.0) + 1e-6)
            id_vals.append(j)
    if not x_rows:
        return (
            np.zeros((0, int(window) * len(STATE_FEATURES))),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0, dtype=int),
        )
    return (
        np.vstack(x_rows),
        np.asarray(y_vals, dtype=float),
        np.asarray(w_vals, dtype=float),
        np.asarray(id_vals, dtype=int),
    )


#: Operator -> class-index mapping of the Phase-1.5 multi-head targets
#: (``0 = polynomial``, ``1 = gaussian``), matching
#: ``controller.multihead_controller.OPERATOR_CLASSES``.
OPERATOR_TO_INDEX: dict[str, int] = {"polynomial": 0, "gaussian": 1}

#: Imputed exploration strength for trajectories recorded before Phase 1.5
#: whose action dicts lack ``"exploration_strength"``: the NSGA-II
#: ``OperatorConfig`` defaults (``eta_m = 20.0`` for polynomial mutation,
#: ``gaussian_sigma = 0.1`` for Gaussian mutation). These are exactly the
#: values ``NSGAII.current_action()`` would have reported had the key been
#: recorded, so the imputation is neutral with respect to the algorithm's
#: own defaults.
OPERATOR_DEFAULT_EXPLORATION: dict[str, float] = {"polynomial": 20.0, "gaussian": 0.1}

#: Positivity floor applied to target values before taking their logarithm,
#: so degenerate (zero/negative) recorded actions stay finite instead of
#: raising. Unlike :func:`build_supervised_samples`, which rejects
#: non-positive mutation probabilities, the multi-head builder clips them.
_MIN_POSITIVE_TARGET = 1e-12


def build_multihead_samples(
    trajectories: list[list[dict[str, Any]]],
    encoder: StateEncoder | ProblemAwareEncoder,
    window: int,
    *,
    mutation_target: str = "absolute",
    n_vars_list: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn trajectories into weighted multi-head supervised samples.

    Phase-1.5 variant of :func:`build_supervised_samples` targeting the full
    action triple. The input alignment, the sample weights, and the
    trajectory ids are identical to :func:`build_supervised_samples`: for
    each trajectory ``j`` and transition index ``t`` in ``1..len-1`` the
    input is ``encoder.transform`` of the merged dicts of transitions
    ``[t - window, t)`` and the weight is ``max(r_t - mean_r_j, 0) + 1e-6``
    with ``r_t = delta_hv_t + delta_igd_t``.

    Targets per sample:

    * ``y_op``: operator class index — ``0`` for ``"polynomial"``, ``1``
      for ``"gaussian"`` (see :data:`OPERATOR_TO_INDEX`).
    * ``y_logpm``: with ``mutation_target="absolute"`` (default, the
      Phase-1.5 behavior), ``log`` of the mutation probability, clipped
      from below at ``1e-12`` so non-positive records stay finite. With
      ``mutation_target="multiplier"`` (Phase 1.75), ``log`` of the
      *normalized* mutation multiplier ``pm * n_vars_j`` of the source
      trajectory — see :mod:`controller.action_normalization` — which
      removes the trivial problem-scale difference from the target.
    * ``y_logexpl``: ``log`` of the exploration strength, clipped the same
      way. **Imputation:** for legacy Phase-0/1 trajectories whose action
      dicts lack ``"exploration_strength"``, the operator's config default
      is imputed (``eta_m = 20.0`` polynomial, ``sigma = 0.1`` Gaussian;
      see :data:`OPERATOR_DEFAULT_EXPLORATION`) — the value the algorithm
      actually used, so legacy data introduces no label bias.

    Args:
        trajectories: Generation-sorted trajectories, e.g. from
            :func:`load_trajectories`.
        encoder: Fitted encoder; ``X`` rows have ``encoder.dim`` columns
            (``window * 6`` for :class:`StateEncoder`, ``window * 6 + 9``
            for :class:`ProblemAwareEncoder`).
        window: History window in generations; should match
            ``encoder.window``.
        mutation_target: ``"absolute"`` (default; log mutation probability
            target, byte-identical to the Phase-1.5 behavior) or
            ``"multiplier"`` (log of ``pm * n_vars`` per trajectory).
        n_vars_list: Per-trajectory decision-variable counts; required when
            ``mutation_target="multiplier"`` and ignored otherwise. Entry
            ``j`` must be the ``n_vars`` of the problem that produced
            ``trajectories[j]`` (e.g. from :func:`load_trajectory_records`).

    Returns:
        Tuple ``(X, y_op, y_logpm, y_logexpl, w, traj_ids)``: ``X`` of
        shape ``(n, encoder.dim)``, integer ``y_op`` and float
        ``y_logpm``/``y_logexpl``/``w`` of shape ``(n,)``, and integer
        ``traj_ids`` of shape ``(n,)``. Empty arrays (with correct shapes)
        if no trajectory has at least two transitions.

    Raises:
        ValueError: If ``window`` < 1, an action's ``mutation_operator``
            is not one of ``"polynomial"``/``"gaussian"``,
            ``mutation_target`` is unknown, or
            ``mutation_target="multiplier"`` is requested without a
            well-formed ``n_vars_list`` (present, one positive entry per
            trajectory).
        KeyError: If an action dict lacks ``"mutation_operator"`` or
            ``"mutation_probability"`` (both are guaranteed by the
            recorder schema).
    """
    if int(window) < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if mutation_target not in ("absolute", "multiplier"):
        raise ValueError(
            f"mutation_target must be 'absolute' or 'multiplier', got "
            f"{mutation_target!r}"
        )
    n_vars_per_traj: list[int] | None = None
    if mutation_target == "multiplier":
        if n_vars_list is None:
            raise ValueError(
                "n_vars_list is required when mutation_target='multiplier'"
            )
        if len(n_vars_list) != len(trajectories):
            raise ValueError(
                f"n_vars_list must have one entry per trajectory: got "
                f"{len(n_vars_list)} for {len(trajectories)} trajectories"
            )
        n_vars_per_traj = [int(n) for n in n_vars_list]
        for j, n_vars in enumerate(n_vars_per_traj):
            if n_vars < 1:
                raise ValueError(
                    f"n_vars_list[{j}] must be >= 1, got {n_vars}"
                )
    x_rows: list[np.ndarray] = []
    y_op_vals: list[int] = []
    y_logpm_vals: list[float] = []
    y_logexpl_vals: list[float] = []
    w_vals: list[float] = []
    id_vals: list[int] = []
    for j, trajectory in enumerate(trajectories):
        if len(trajectory) < 2:
            continue
        merged = [merge_state_reward(t) for t in trajectory]
        rewards = np.asarray(
            [m["delta_hv"] + m["delta_igd"] for m in merged], dtype=float
        )
        mean_reward = float(rewards.mean())
        for t in range(1, len(trajectory)):
            history = merged[max(0, t - int(window)) : t]
            action = trajectory[t]["action"]
            operator = str(action["mutation_operator"])
            if operator not in OPERATOR_TO_INDEX:
                raise ValueError(
                    f"unsupported mutation_operator {operator!r} at trajectory {j}, "
                    f"transition {t}; expected one of {sorted(OPERATOR_TO_INDEX)}"
                )
            pm = float(action["mutation_probability"])
            # Imputation for legacy 2-key actions (see module constants).
            if "exploration_strength" in action:
                exploration = float(action["exploration_strength"])
            else:
                exploration = OPERATOR_DEFAULT_EXPLORATION[operator]
            x_rows.append(encoder.transform(history))
            y_op_vals.append(OPERATOR_TO_INDEX[operator])
            log_pm_target = math.log(max(pm, _MIN_POSITIVE_TARGET))
            if n_vars_per_traj is not None:
                log_pm_target += math.log(n_vars_per_traj[j])
            y_logpm_vals.append(log_pm_target)
            y_logexpl_vals.append(math.log(max(exploration, _MIN_POSITIVE_TARGET)))
            w_vals.append(max(float(rewards[t]) - mean_reward, 0.0) + 1e-6)
            id_vals.append(j)
    if not x_rows:
        return (
            np.zeros((0, encoder.dim)),
            np.zeros(0, dtype=int),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0, dtype=int),
        )
    return (
        np.vstack(x_rows),
        np.asarray(y_op_vals, dtype=int),
        np.asarray(y_logpm_vals, dtype=float),
        np.asarray(y_logexpl_vals, dtype=float),
        np.asarray(w_vals, dtype=float),
        np.asarray(id_vals, dtype=int),
    )


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    traj_ids: np.ndarray,
    val_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Split samples into train/validation sets by trajectory id.

    Splitting by trajectory (rather than by sample) prevents leakage:
    windows from one run never appear on both sides of the split.

    Args:
        X: Feature matrix of shape ``(n, d)``.
        y: Targets of shape ``(n,)``.
        w: Sample weights of shape ``(n,)``.
        traj_ids: Trajectory id of each sample, shape ``(n,)``.
        val_fraction: Fraction of trajectories assigned to validation;
            must be in ``[0, 1)``. When ``0 < val_fraction < 1`` and at
            least two trajectories exist, at least one trajectory lands in
            each split.
        seed: Random seed for the trajectory permutation.

    Returns:
        Dict with keys ``X_train``, ``y_train``, ``w_train``,
        ``traj_ids_train`` and ``X_val``, ``y_val``, ``w_val``,
        ``traj_ids_val``.

    Raises:
        ValueError: If ``val_fraction`` is outside ``[0, 1)``.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.asarray(w)
    traj_ids = np.asarray(traj_ids)
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(traj_ids)
    shuffled = rng.permutation(unique_ids)
    n_val = int(round(len(unique_ids) * val_fraction))
    if val_fraction > 0.0 and len(unique_ids) > 1:
        n_val = max(1, min(n_val, len(unique_ids) - 1))
    val_ids = shuffled[:n_val]
    val_mask = np.isin(traj_ids, val_ids)
    train_mask = ~val_mask
    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "w_train": w[train_mask],
        "traj_ids_train": traj_ids[train_mask],
        "X_val": X[val_mask],
        "y_val": y[val_mask],
        "w_val": w[val_mask],
        "traj_ids_val": traj_ids[val_mask],
    }
