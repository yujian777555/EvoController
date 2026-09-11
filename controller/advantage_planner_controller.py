from __future__ import annotations

"""Phase 2.75, Task 5: planner on top of an action-advantage predictor.

Phase 2B's :class:`controller.planning_controller.PlanningController` scores
candidate actions by *predicted absolute future hypervolume* and therefore has
to weight horizons (``0.1, 0.2, 0.3, 0.4``) to make candidates comparable —
and its outcome model turned out to be action-insensitive (Phase-2B D1/D2).
:class:`AdvantagePlannerController` keeps the very same interface but consumes
an :class:`controller.advantage_predictor.AdvantagePredictor`, whose output is
already a *relative* quantity (the action's advantage over a per-state
baseline) and therefore directly comparable between candidates. The decision
rule is the plain argmax of the mean predicted advantage over the predictor's
horizons — no horizon weighting.

Candidate sampling is **delegated**, never re-implemented: the controller holds
an internal :class:`PlanningController` whose predictor is a zero stub, calls
its ``predict_action_ex`` and keeps only the ``diagnostics["candidates"]``
list. The sampled actions (``PCG64([candidate_seed, len(history)])`` through
:func:`experiments.generate_dataset.sample_full_action`) are consequently
bit-identical to Phase 2B's, and the feature block is rebuilt with the same
four-feature layout ``[mutation_multiplier, exploration_strength,
onehot_polynomial, onehot_gaussian]`` that
:func:`controller.dataset.build_outcome_samples` and both predictors use
(tested bit-exactly in ``tests/test_advantage_planner.py``).

``macro_actions=True`` replaces the sampled candidate set by the discrete macro
actions of :mod:`controller.macro_actions` (Phase 2.75 Task 3) and takes the
argmax over those; the module is imported lazily inside the method, so a
checkout without it still imports this controller (the branch then raises a
clear error instead of an ImportError at import time).

Example:
    ``planner = AdvantagePlannerController(AdvantagePredictor.load(path))``
    ``action = planner.predict_action(history, encoder, 0.25/30, 8.0/30, n_vars=30)``
"""

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from controller.dataset import OPERATOR_TO_INDEX
from controller.planning_controller import (
    PLANNING_FALLBACK_N_VARS,
    PlanningController,
)

__all__ = ["AdvantagePlannerController", "PLANNING_FALLBACK_N_VARS"]

#: Number of trailing feature columns that carry the action.
N_ACTION_FEATURES = 4


class _ZeroPredictor:
    """Stub predictor used only to capture :class:`PlanningController` candidates.

    Provides exactly the surface ``PlanningController`` consumes
    (``input_dim``, ``horizons``, ``predict``) and always predicts zeros, so
    the internal sampler's own decision is meaningless and discarded — only
    its ``diagnostics["candidates"]`` is used.
    """

    def __init__(self, input_dim: int, horizons: Sequence[int]) -> None:
        self._input_dim = int(input_dim)
        self._horizons = tuple(int(h) for h in horizons)

    @property
    def input_dim(self) -> int:
        """Flattened input width expected by the captured planner."""
        return self._input_dim

    @property
    def horizons(self) -> tuple[int, ...]:
        """Horizons of the captured planner (only the width matters here)."""
        return self._horizons

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return zeros with one column per horizon."""
        rows = np.asarray(X, dtype=np.float64)
        return np.zeros((rows.shape[0], len(self._horizons)), dtype=np.float64)


class AdvantagePlannerController:
    """Action planner that maximises the predicted action advantage.

    Attributes:
        name: Identifier of this controller (reported in experiment configs).
    """

    name: str = "advantage_planner"

    def __init__(
        self,
        predictor: Any,
        n_candidates: int = 16,
        candidate_seed: int = 0,
        pm_mult_range: tuple[float, float] = (0.25, 8.0),
        macro_actions: bool = False,
    ) -> None:
        """Initialize the planner.

        Args:
            predictor: Fitted advantage predictor exposing ``input_dim``,
                ``horizons`` and ``predict`` (i.e.
                :class:`controller.advantage_predictor.AdvantagePredictor`).
            n_candidates: Candidate actions sampled and scored per
                generation (ignored when ``macro_actions`` is True).
            candidate_seed: Seed of the candidate generator, combined with
                ``len(history)`` per call exactly like Phase 2B.
            pm_mult_range: Log-uniform multiplier range around ``1 / n_vars``.
            macro_actions: When True, score the discrete macro action set of
                :mod:`controller.macro_actions` instead of sampled candidates.

        Raises:
            ValueError: If ``n_candidates`` < 1, ``candidate_seed`` < 0,
                ``pm_mult_range`` is invalid, or the predictor does not
                expose ``input_dim``/``horizons``.
        """
        if int(n_candidates) < 1:
            raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
        if int(candidate_seed) < 0:
            raise ValueError(f"candidate_seed must be >= 0, got {candidate_seed}")
        lo, hi = float(pm_mult_range[0]), float(pm_mult_range[1])
        if not lo > 0.0 or hi < lo:
            raise ValueError(
                f"pm_mult_range must satisfy 0 < lo <= hi, got {(lo, hi)}"
            )
        if not hasattr(predictor, "input_dim") or not hasattr(predictor, "horizons"):
            raise ValueError(
                "predictor must expose input_dim and horizons (an "
                "AdvantagePredictor checkpoint)"
            )
        self._predictor = predictor
        self._n_candidates = int(n_candidates)
        self._candidate_seed = int(candidate_seed)
        self._pm_mult_range = (lo, hi)
        self._macro_actions = bool(macro_actions)
        self._predictor_path: str | None = None
        # Candidate capture: the internal planner owns the seeding and the
        # sampling distribution; its zero predictor makes its own decision
        # irrelevant.
        self._sampler = PlanningController(
            _ZeroPredictor(int(predictor.input_dim), tuple(predictor.horizons)),
            n_candidates=int(n_candidates),
            candidate_seed=int(candidate_seed),
            horizon_weights=[1.0] * len(tuple(predictor.horizons)),
            pm_mult_range=(lo, hi),
        )

    @property
    def predictor(self) -> Any:
        """The advantage predictor used for candidate scoring."""
        return self._predictor

    @property
    def n_candidates(self) -> int:
        """Number of sampled candidates scored per generation."""
        return self._n_candidates

    @property
    def candidate_seed(self) -> int:
        """Seed of the candidate generator (combined with history length)."""
        return self._candidate_seed

    @property
    def pm_mult_range(self) -> tuple[float, float]:
        """Log-uniform multiplier range of the candidate sampler."""
        return self._pm_mult_range

    @property
    def macro_actions(self) -> bool:
        """Whether the discrete macro action set is used as the candidate set."""
        return self._macro_actions

    @property
    def predictor_path(self) -> str | None:
        """Recorded location of the predictor checkpoint (metadata only)."""
        return self._predictor_path

    @predictor_path.setter
    def predictor_path(self, value: str | None) -> None:
        self._predictor_path = None if value is None else str(value)

    def _macro_candidates(self, n_vars: int) -> list[dict[str, Any]]:
        """Macro action set materialized for ``n_vars``.

        Raises:
            RuntimeError: If :mod:`controller.macro_actions` is unavailable.
        """
        try:
            from controller.macro_actions import MACRO_ACTIONS, macro_action
        except ImportError as exc:  # pragma: no cover - module ships with repo
            raise RuntimeError(
                "macro_actions=True requires controller/macro_actions.py"
            ) from exc
        candidates: list[dict[str, Any]] = []
        for name, macro in MACRO_ACTIONS.items():
            action = macro_action(name, n_vars)
            candidates.append(
                {
                    "mutation_operator": action["mutation_operator"],
                    "mutation_probability": float(action["mutation_probability"]),
                    "exploration_strength": float(action["exploration_strength"]),
                    "mutation_multiplier": float(macro["multiplier"]),
                }
            )
        return candidates

    def _candidate_actions(
        self,
        history: list[dict[str, Any]],
        encoder: Any,
        pm_min: float,
        pm_max: float,
        n_vars: int,
    ) -> list[dict[str, Any]]:
        """Candidate set: sampled full actions or the macro action set."""
        if self._macro_actions:
            return self._macro_candidates(n_vars)
        _action, diagnostics = self._sampler.predict_action_ex(
            list(history), encoder, pm_min, pm_max, n_vars=n_vars
        )
        return list(diagnostics["candidates"])

    @staticmethod
    def _feature_rows(
        state_block: np.ndarray, candidates: Sequence[dict[str, Any]]
    ) -> np.ndarray:
        """Feature rows for the candidates, in the canonical layout.

        ``[state block, mutation_multiplier, exploration_strength,
        onehot_polynomial, onehot_gaussian]`` — bit-identical to the rows
        :class:`PlanningController` builds and to
        :func:`controller.dataset.build_outcome_samples`.

        Raises:
            ValueError: If a candidate carries an unsupported operator.
        """
        rows: list[np.ndarray] = []
        for candidate in candidates:
            operator = str(candidate["mutation_operator"])
            if operator not in OPERATOR_TO_INDEX:
                raise ValueError(
                    f"unsupported mutation_operator {operator!r}; expected one "
                    f"of {sorted(OPERATOR_TO_INDEX)}"
                )
            one_hot = [0.0, 0.0]
            one_hot[OPERATOR_TO_INDEX[operator]] = 1.0
            rows.append(
                np.concatenate(
                    [
                        np.asarray(state_block, dtype=np.float64),
                        np.asarray(
                            [
                                float(candidate["mutation_multiplier"]),
                                float(candidate["exploration_strength"]),
                                *one_hot,
                            ],
                            dtype=np.float64,
                        ),
                    ]
                )
            )
        return np.vstack(rows)

    def predict_action_ex(
        self,
        history: list[dict[str, Any]],
        encoder: Any,
        pm_min: float,
        pm_max: float,
        n_vars: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Select the candidate with the largest predicted advantage.

        Args:
            history: Merged state+reward dicts, oldest first, covering
                generations strictly before the action to take.
            encoder: Fitted encoder matching the predictor's training encoder.
            pm_min: Accepted for interface compatibility with
                :meth:`MultiHeadController.predict_action`; unused, because
                candidates are drawn from ``pm_mult_range`` around
                ``1 / n_vars`` (macro candidates are fixed points of the
                action space instead).
            pm_max: See ``pm_min``.
            n_vars: Decision-variable count; ``None`` applies the legacy
                fallback :data:`PLANNING_FALLBACK_N_VARS`.

        Returns:
            ``(action, diagnostics)``. ``action`` has exactly the keys
            ``"mutation_operator"``, ``"mutation_probability"`` and
            ``"exploration_strength"``.

            ``diagnostics`` uses the Phase-2B
            :meth:`PlanningController.predict_action_ex` format with the same
            six keys: ``"candidates"`` (per-candidate action dicts including
            ``mutation_multiplier``), ``"predicted_hv"`` (here the *predicted
            advantage vectors*, one row per candidate and one column per
            predictor horizon — the key name is kept for interface parity),
            ``"scores"`` (mean predicted advantage per candidate, the argmax
            objective), ``"selected_index"``, ``"score_margin"`` (top-1 minus
            top-2, ``0.0`` for a single candidate) and ``"score_std"``.

        Raises:
            ValueError: If ``n_vars`` < 1 or the predictor's ``input_dim``
                does not equal ``encoder.transform(history).size + 4``.
        """
        resolved_n_vars = PlanningController._resolve_n_vars(n_vars)
        state_block = np.asarray(
            encoder.transform(list(history)), dtype=np.float64
        )
        expected_dim = int(self._predictor.input_dim)
        if expected_dim != state_block.size + N_ACTION_FEATURES:
            raise ValueError(
                f"predictor input_dim ({expected_dim}) does not match the "
                f"constructed feature width ({state_block.size} + "
                f"{N_ACTION_FEATURES}); the encoder must match the "
                f"predictor's training encoder"
            )
        candidates = self._candidate_actions(
            list(history), encoder, pm_min, pm_max, resolved_n_vars
        )
        rows = self._feature_rows(state_block, candidates)
        predictions = np.asarray(
            self._predictor.predict(rows), dtype=np.float64
        )
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        # Advantages are already relative to the same state baseline, so the
        # candidates are directly comparable: no horizon weighting, just the
        # mean over the predictor's horizons.
        scores = predictions.mean(axis=1)
        best = int(np.argmax(scores))
        selected = candidates[best]
        action = {
            "mutation_operator": str(selected["mutation_operator"]),
            "mutation_probability": float(selected["mutation_probability"]),
            "exploration_strength": float(selected["exploration_strength"]),
        }
        if scores.size > 1:
            top_two = np.sort(scores)[-2:]
            score_margin = float(top_two[-1] - top_two[0])
        else:
            score_margin = 0.0
        diagnostics = {
            "candidates": [
                {
                    "mutation_operator": str(candidate["mutation_operator"]),
                    "mutation_probability": float(candidate["mutation_probability"]),
                    "exploration_strength": float(
                        candidate["exploration_strength"]
                    ),
                    "mutation_multiplier": float(candidate["mutation_multiplier"]),
                }
                for candidate in candidates
            ],
            "predicted_hv": [
                [float(value) for value in predictions[index]]
                for index in range(predictions.shape[0])
            ],
            "scores": [float(value) for value in scores],
            "selected_index": int(best),
            "score_margin": score_margin,
            "score_std": float(np.std(scores)),
        }
        return action, diagnostics

    def predict_action(
        self,
        history: list[dict[str, Any]],
        encoder: Any,
        pm_min: float,
        pm_max: float,
        n_vars: int | None = None,
    ) -> dict[str, Any]:
        """Return only the selected action (thin wrapper over :meth:`predict_action_ex`)."""
        action, _diagnostics = self.predict_action_ex(
            history, encoder, pm_min, pm_max, n_vars=n_vars
        )
        return action

    def save(self, path: str | Path) -> None:
        """Serialize the planner config to JSON (predictor weights excluded)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "predictor_path": self._predictor_path,
            "config": {
                "n_candidates": self._n_candidates,
                "candidate_seed": self._candidate_seed,
                "pm_mult_range": list(self._pm_mult_range),
                "macro_actions": self._macro_actions,
            },
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(
        cls, path: str | Path, predictor: Any
    ) -> "AdvantagePlannerController":
        """Load a planner saved with :meth:`save`, bound to ``predictor``."""
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        config = payload["config"]
        controller = cls(
            predictor,
            n_candidates=int(config["n_candidates"]),
            candidate_seed=int(config["candidate_seed"]),
            pm_mult_range=(
                float(config["pm_mult_range"][0]),
                float(config["pm_mult_range"][1]),
            ),
            macro_actions=bool(config.get("macro_actions", False)),
        )
        controller.predictor_path = payload.get("predictor_path")
        return controller
