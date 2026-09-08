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
