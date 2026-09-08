from __future__ import annotations

"""Encodes windows of evolution history into fixed-length feature vectors.

The controller input at generation ``t`` is the window of merged transition
dicts from generations ``[t - window, t)`` (oldest first). Each merged dict
contributes the six :data:`STATE_FEATURES` values. Histories shorter than
the window are zero-padded at the front, so the most recent state always
occupies the final feature block. Features are z-scored with statistics
fitted over every transition of the training trajectories; the transform
itself only ever reads past transitions, so no future information leaks
into the encoding.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.base import Problem
from controller.problem_features import PROBLEM_FEATURE_NAMES, problem_feature_vector

STATE_FEATURES: list[str] = [
    "hv",
    "igd",
    "diversity",
    "delta_hv",
    "delta_igd",
    "generation",
]


def _transition_features(transition: dict[str, Any]) -> list[float]:
    """Extract the ``STATE_FEATURES`` values from a raw transition dict.

    State keys (``hv``, ``igd``, ``diversity``, ``generation``) are read
    from ``transition["state"]`` and reward keys (``delta_hv``,
    ``delta_igd``) from ``transition["reward"]``.

    Args:
        transition: One recorded ``(state, action, reward)`` transition.

    Returns:
        Feature values in ``STATE_FEATURES`` order.
    """
    state = transition["state"]
    reward = transition["reward"]
    return [
        float(state[key]) if key in state else float(reward[key])
        for key in STATE_FEATURES
    ]


class StateEncoder:
    """Z-score encoder for fixed-window evolution history features.

    Attributes:
        name: Identifier prefix used by controllers built on this encoder.
    """

    def __init__(self, window: int) -> None:
        """Initialize an unfitted encoder.

        Args:
            window: Number of past generations encoded per sample; must
                be >= 1.
        """
        if int(window) < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = int(window)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @property
    def window(self) -> int:
        """History window length in generations."""
        return self._window

    @property
    def dim(self) -> int:
        """Flattened feature dimension: ``window * len(STATE_FEATURES)``."""
        return self._window * len(STATE_FEATURES)

    def fit(self, trajectories: list[list[dict[str, Any]]]) -> "StateEncoder":
        """Compute z-score statistics over all given trajectories.

        The merged ``(state, reward)`` feature vector of every transition
        of every trajectory contributes to the per-feature mean and
        standard deviation. Features with zero variance are assigned a
        standard deviation of 1.0 so the transform stays finite.

        Args:
            trajectories: Recorded trajectories, each a list of transition
                dicts as produced by ``EvolutionRecorder``.

        Returns:
            The fitted encoder (``self``).

        Raises:
            ValueError: If the trajectories contain no transitions.
        """
        rows: list[list[float]] = []
        for trajectory in trajectories:
            for transition in trajectory:
                rows.append(_transition_features(transition))
        if not rows:
            raise ValueError("cannot fit StateEncoder on zero transitions")
        features = np.asarray(rows, dtype=float)
        std = features.std(axis=0)
        self._mean = features.mean(axis=0)
        self._std = np.where(std > 0.0, std, 1.0)
        return self

    def transform(self, history: list[dict[str, Any]]) -> np.ndarray:
        """Encode one history window into a flat feature vector.

        Args:
            history: Merged state+reward dicts (the six ``STATE_FEATURES``
                keys), oldest first. Only the last ``window`` entries are
                used; shorter histories are zero-padded at the front.

        Returns:
            Array of shape ``(window * 6,)``; entry-major layout with the
            oldest entry first and the most recent entry last.

        Raises:
            RuntimeError: If the encoder has not been fitted.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("StateEncoder must be fitted before transform()")
        recent = list(history)[-self._window :]
        n_pad = self._window - len(recent)
        blocks = [np.zeros((n_pad, len(STATE_FEATURES)))]
        if recent:
            values = np.asarray(
                [[float(entry[key]) for key in STATE_FEATURES] for entry in recent],
                dtype=float,
            )
            blocks.append((values - self._mean) / self._std)
        return np.vstack(blocks).reshape(-1)

    def transform_batch(self, histories: list[list[dict[str, Any]]]) -> np.ndarray:
        """Encode multiple history windows.

        Args:
            histories: One history per sample, each as in :meth:`transform`.

        Returns:
            Array of shape ``(len(histories), window * 6)``.
        """
        if not histories:
            return np.zeros((0, self.dim))
        return np.vstack([self.transform(history) for history in histories])

    def save(self, path: str | Path) -> None:
        """Serialize window and z-score statistics to a JSON file.

        Args:
            path: Destination path; parent directories are created.

        Raises:
            RuntimeError: If the encoder has not been fitted.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("cannot save an unfitted StateEncoder")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "window": self._window,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "StateEncoder":
        """Load a fitted encoder saved with :meth:`save`.

        Args:
            path: Path to the JSON file written by :meth:`save`.

        Returns:
            The fitted encoder.
        """
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        encoder = cls(window=int(payload["window"]))
        encoder._mean = np.asarray(payload["mean"], dtype=float)
        encoder._std = np.asarray(payload["std"], dtype=float)
        return encoder


class ProblemAwareEncoder:
    """Z-score encoder combining history features with problem descriptors.

    Extends the :class:`StateEncoder` encoding with a constant-per-problem
    block of the nine :data:`PROBLEM_FEATURE_NAMES` values, so a single
    controller can condition its action on which problem is being
    optimized. Output layout: ``window`` blocks of six z-scored state
    features (front-zero-padded, oldest first, byte-identical semantics to
    :class:`StateEncoder`) followed by this problem's nine z-scored problem
    features.

    Persistence: :meth:`save` writes the window, the problem name, this
    problem's raw (unscaled) feature vector, and all z-score statistics to
    JSON. :meth:`load` restores a fully functional encoder directly from
    the persisted vector, so no ``Problem`` instance is required at load
    time; the problem name is kept for traceability.
    """

    def __init__(self, window: int, problem: Problem) -> None:
        """Initialize an unfitted problem-aware encoder.

        Args:
            window: Number of past generations encoded per sample; must
                be >= 1.
            problem: Problem this encoder encodes histories for. Its
                descriptor vector is computed once here and is
                deterministic (``describe`` default seed).
        """
        if int(window) < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = int(window)
        self._problem: Problem | None = problem
        self._problem_name = str(problem.name)
        self._problem_vector = problem_feature_vector(problem)
        self._state_mean: np.ndarray | None = None
        self._state_std: np.ndarray | None = None
        self._problem_mean: np.ndarray | None = None
        self._problem_std: np.ndarray | None = None

    @property
    def window(self) -> int:
        """History window length in generations."""
        return self._window

    @property
    def problem_name(self) -> str:
        """Name of the problem this encoder encodes histories for."""
        return self._problem_name

    @property
    def dim(self) -> int:
        """Flattened feature dimension: ``window * 6 + 9``."""
        return self._window * len(STATE_FEATURES) + len(PROBLEM_FEATURE_NAMES)

    def fit(
        self,
        trajectories: list[list[dict[str, Any]]],
        problems: list[Problem],
    ) -> "ProblemAwareEncoder":
        """Compute z-score statistics over transitions and problems.

        State statistics use the same merge semantics as
        :class:`StateEncoder`: every transition of every trajectory
        contributes its six merged state+reward features. Problem
        statistics are computed over the descriptor vectors of
        ``problems``, which should contain every problem the controller
        will see. Zero-variance features are assigned a standard deviation
        of 1.0 so the transform stays finite (this also covers fitting
        with a single problem).

        Args:
            trajectories: Recorded trajectories, each a list of transition
                dicts as produced by ``EvolutionRecorder``.
            problems: Non-empty list of problems defining the problem-block
                statistics.

        Returns:
            The fitted encoder (``self``).

        Raises:
            ValueError: If the trajectories contain no transitions or
                ``problems`` is empty.
        """
        # Lazy import: controller.dataset imports this module, so a
        # module-level import would be circular. Semantics are identical
        # to StateEncoder.fit (same six merged features per transition).
        from controller.dataset import merge_state_reward

        rows: list[list[float]] = []
        for trajectory in trajectories:
            for transition in trajectory:
                merged = merge_state_reward(transition)
                rows.append([merged[key] for key in STATE_FEATURES])
        if not rows:
            raise ValueError(
                "cannot fit ProblemAwareEncoder on zero transitions"
            )
        if not problems:
            raise ValueError("cannot fit ProblemAwareEncoder on zero problems")
        features = np.asarray(rows, dtype=float)
        state_std = features.std(axis=0)
        self._state_mean = features.mean(axis=0)
        self._state_std = np.where(state_std > 0.0, state_std, 1.0)
        vectors = np.asarray(
            [problem_feature_vector(problem) for problem in problems],
            dtype=float,
        )
        problem_std = vectors.std(axis=0)
        self._problem_mean = vectors.mean(axis=0)
        self._problem_std = np.where(problem_std > 0.0, problem_std, 1.0)
        return self

    def transform(self, history: list[dict[str, Any]]) -> np.ndarray:
        """Encode one history window plus this problem's descriptor.

        Args:
            history: Merged state+reward dicts (the six ``STATE_FEATURES``
                keys), oldest first. Only the last ``window`` entries are
                used; shorter histories are zero-padded at the front.

        Returns:
            Array of shape ``(window * 6 + 9,)``: the windowed z-scored
            state features (identical layout and values to
            :meth:`StateEncoder.transform`) concatenated with this
            problem's z-scored descriptor. The problem block is constant
            across all histories of the same problem.

        Raises:
            RuntimeError: If the encoder has not been fitted.
        """
        if (
            self._state_mean is None
            or self._state_std is None
            or self._problem_mean is None
            or self._problem_std is None
        ):
            raise RuntimeError(
                "ProblemAwareEncoder must be fitted before transform()"
            )
        recent = list(history)[-self._window :]
        n_pad = self._window - len(recent)
        blocks = [np.zeros((n_pad, len(STATE_FEATURES)))]
        if recent:
            values = np.asarray(
                [[float(entry[key]) for key in STATE_FEATURES] for entry in recent],
                dtype=float,
            )
            blocks.append((values - self._state_mean) / self._state_std)
        state_part = np.vstack(blocks).reshape(-1)
        problem_part = (self._problem_vector - self._problem_mean) / self._problem_std
        return np.concatenate([state_part, problem_part])

    def transform_batch(self, histories: list[list[dict[str, Any]]]) -> np.ndarray:
        """Encode multiple history windows.

        Args:
            histories: One history per sample, each as in :meth:`transform`.

        Returns:
            Array of shape ``(len(histories), window * 6 + 9)``.
        """
        if not histories:
            return np.zeros((0, self.dim))
        return np.vstack([self.transform(history) for history in histories])

    def save(self, path: str | Path) -> None:
        """Serialize window, problem descriptor, and statistics to JSON.

        Args:
            path: Destination path; parent directories are created.

        Raises:
            RuntimeError: If the encoder has not been fitted.
        """
        if (
            self._state_mean is None
            or self._state_std is None
            or self._problem_mean is None
            or self._problem_std is None
        ):
            raise RuntimeError("cannot save an unfitted ProblemAwareEncoder")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "window": self._window,
            "problem_name": self._problem_name,
            "problem_vector": self._problem_vector.tolist(),
            "state_mean": self._state_mean.tolist(),
            "state_std": self._state_std.tolist(),
            "problem_mean": self._problem_mean.tolist(),
            "problem_std": self._problem_std.tolist(),
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ProblemAwareEncoder":
        """Load a fitted encoder saved with :meth:`save`.

        The persisted raw problem feature vector is restored directly, so
        no ``Problem`` instance is needed; ``problem_name`` identifies
        which problem the vector belongs to.

        Args:
            path: Path to the JSON file written by :meth:`save`.

        Returns:
            The fitted encoder.
        """
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        encoder = cls.__new__(cls)
        encoder._window = int(payload["window"])
        encoder._problem = None
        encoder._problem_name = str(payload["problem_name"])
        encoder._problem_vector = np.asarray(payload["problem_vector"], dtype=float)
        encoder._state_mean = np.asarray(payload["state_mean"], dtype=float)
        encoder._state_std = np.asarray(payload["state_std"], dtype=float)
        encoder._problem_mean = np.asarray(payload["problem_mean"], dtype=float)
        encoder._problem_std = np.asarray(payload["problem_std"], dtype=float)
        return encoder
