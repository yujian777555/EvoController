from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from metrics.indicators import diversity_spread, hypervolume, igd

TRAJECTORY_SCHEMA_VERSION = 1

_REQUIRED_ACTION_KEYS: tuple[str, ...] = ("mutation_operator", "mutation_probability")


def _to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays in nested structures to native Python types.

    Ensures the trajectory JSON is serialization-stable across runs and
    platforms (plain ``int``/``float``/``list``/``dict`` only).

    Args:
        value: Arbitrary nested structure possibly containing numpy types.

    Returns:
        The same structure with numpy types replaced by Python natives.
    """
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


class EvolutionRecorder:
    """Records (state, action, reward) transitions from an evolutionary run.

    One transition is stored per recorded generation. The state holds the
    generation index plus quality indicators of the current nondominated
    front: hypervolume (against a fixed reference point), IGD (against the
    true Pareto front), and Deb's diversity spread. The action is the
    resolved operator configuration used to produce the next generation.
    Rewards follow the improvement convention: generation 0 yields zero
    reward; for t >= 1, ``delta_hv = hv_t - hv_{t-1}`` (higher HV is better)
    and ``delta_igd = igd_{t-1} - igd_t`` (lower IGD is better, so
    improvement is positive).
    """

    def __init__(
        self,
        problem_name: str,
        reference_front: np.ndarray,
        ref_point: np.ndarray,
    ) -> None:
        """Initialize the recorder.

        Args:
            problem_name: Benchmark identifier (e.g. ``"zdt1"``), kept for
                provenance in the recorded metadata.
            reference_front: True Pareto front samples of shape (n, 2),
                used as the IGD reference set.
            ref_point: Hypervolume reference point of shape (2,), must be
                dominated by all points of interest (minimization).
        """
        self._problem_name = str(problem_name)
        self._reference_front = np.asarray(reference_front, dtype=float)
        self._ref_point = np.asarray(ref_point, dtype=float)
        self._transitions: list[dict[str, Any]] = []
        self._prev_hv: float | None = None
        self._prev_igd: float | None = None

    @property
    def problem_name(self) -> str:
        """Benchmark identifier this recorder is attached to."""
        return self._problem_name

    def record(self, generation: int, front: np.ndarray, action: dict) -> None:
        """Record one generation as a (state, action, reward) transition.

        Args:
            generation: Generation index of the given front.
            front: Objective values of the current nondominated front,
                shape (k, 2), minimization.
            action: Resolved operator configuration; must contain the keys
                ``mutation_operator`` (str) and ``mutation_probability``
                (float).

        Raises:
            ValueError: If a required action key is missing.
        """
        missing = [key for key in _REQUIRED_ACTION_KEYS if key not in action]
        if missing:
            raise ValueError(f"action is missing required keys: {missing}")

        front = np.asarray(front, dtype=float)
        hv = float(hypervolume(front, self._ref_point))
        igd_value = float(igd(front, self._reference_front))
        diversity = float(diversity_spread(front))

        if self._prev_hv is None or self._prev_igd is None:
            reward = {"delta_hv": 0.0, "delta_igd": 0.0}
        else:
            reward = {
                "delta_hv": hv - self._prev_hv,
                "delta_igd": self._prev_igd - igd_value,
            }

        transition = {
            "generation": int(generation),
            "state": {
                "generation": int(generation),
                "hv": hv,
                "igd": igd_value,
                "diversity": diversity,
            },
            "action": {
                "mutation_operator": str(action["mutation_operator"]),
                "mutation_probability": float(action["mutation_probability"]),
            },
            "reward": {
                "delta_hv": float(reward["delta_hv"]),
                "delta_igd": float(reward["delta_igd"]),
            },
        }
        self._transitions.append(transition)
        self._prev_hv = hv
        self._prev_igd = igd_value

    def transitions(self) -> list[dict]:
        """Return a deep copy of all recorded transitions.

        Mutating the returned structure never affects the recorder's
        internal state, so callers may freely transform or serialize it.
        """
        return copy.deepcopy(self._transitions)

    def save(
        self,
        path: str | Path,
        config: dict,
        seed: int,
        runtime_sec: float,
    ) -> None:
        """Write the recorded trajectory to a JSON file.

        Schema (UTF-8, indent 2, ``ensure_ascii=False``)::

            {
              "config": dict,
              "seed": int,
              "runtime_sec": float,
              "schema_version": int,
              "transitions": [transition, ...],
              "final": {"hv": float, "igd": float, "diversity": float}
            }

        Args:
            path: Destination file path; parent directories are created.
            config: Experiment configuration for reproducibility.
            seed: Random seed of the run.
            runtime_sec: Wall-clock runtime of the run in seconds.

        Raises:
            ValueError: If no transitions have been recorded yet.
        """
        if not self._transitions:
            raise ValueError("cannot save an empty trajectory; record() first")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        last_state = self._transitions[-1]["state"]
        payload = {
            "config": _to_jsonable(config),
            "seed": int(seed),
            "runtime_sec": float(runtime_sec),
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "transitions": self.transitions(),
            "final": {
                "hv": float(last_state["hv"]),
                "igd": float(last_state["igd"]),
                "diversity": float(last_state["diversity"]),
            },
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
