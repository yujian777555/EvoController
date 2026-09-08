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
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from controller.state_encoder import STATE_FEATURES, StateEncoder


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
