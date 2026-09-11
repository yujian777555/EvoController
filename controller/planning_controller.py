from __future__ import annotations

"""Planning controller for Phase 2 Experiment B (search policy optimization).

Experiment A showed that an :class:`controller.outcome_predictor.OutcomePredictor`
can regress future hypervolume from ``(encoded state history, candidate
action)`` with held-out R² > 0.97. :class:`PlanningController` turns that
outcome model into a decision rule: at each generation it samples
``n_candidates`` candidate actions from the Phase-1.5 full action space,
predicts the future-HV vector of every candidate, scores each candidate by
a weighted sum over horizons (``horizon_weights``, favoring long-horizon
outcomes by default), and returns the argmax candidate for execution.

Candidate sampling reuses
:func:`experiments.generate_dataset.sample_full_action` verbatim, so the
draws (uniform operator from {polynomial, gaussian}, log-uniform mutation
multiplier in ``pm_mult_range`` around ``1 / n_vars``, log-uniform
operator-specific exploration strength in ``ETA_M_SAMPLE_RANGE`` /
``SIGMA_SAMPLE_RANGE``) are distribution-identical to the actions the
predictor was trained on. The candidate generator is seeded by
``[candidate_seed, len(history)]``: reproducible for a fixed history, but
different generations (different history lengths) see different candidate
sets, which avoids committing to one fixed candidate grid for the whole
run.

The predictor input layout matches
:func:`controller.dataset.build_outcome_samples` exactly:
``encoder.transform(history)`` concatenated with the four raw action
features ``[mutation_multiplier, exploration_strength,
onehot_polynomial, onehot_gaussian]`` where
``mutation_multiplier = mutation_probability * n_vars``. When
``n_vars`` is not provided, the controller falls back to
:data:`PLANNING_FALLBACK_N_VARS` (the legacy convention of
:data:`controller.dataset.OUTCOME_FALLBACK_N_VARS`): the base probability
is ``1 / 30`` and the multiplier feature is ``pm * 30``.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from controller.outcome_predictor import OutcomePredictor
from experiments.generate_dataset import (
    ETA_M_SAMPLE_RANGE,
    FULL_ACTION_PM_MULT_RANGE,
    SIGMA_SAMPLE_RANGE,
    sample_full_action,
)

__all__ = [
    "PlanningController",
    "DEFAULT_HORIZON_WEIGHTS",
    "PLANNING_FALLBACK_N_VARS",
]

#: Default per-horizon score weights for horizons ``[1, 5, 10, 20]``;
#: long-horizon outcomes dominate so the planner optimizes final HV, not
#: one-step greediness. Sums to 1.0 (the score is a weighted average of
#: predicted future HV).
DEFAULT_HORIZON_WEIGHTS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)

#: Decision-variable count assumed when ``predict_action`` is called
#: without ``n_vars``: raw transitions store no problem metadata, so the
#: legacy convention (``controller.dataset.OUTCOME_FALLBACK_N_VARS``) is
#: reused — base mutation probability ``1 / 30`` and multiplier feature
#: ``pm * 30``.
PLANNING_FALLBACK_N_VARS: int = 30


class PlanningController:
    """Candidate-sampling planner on top of a learned outcome predictor.

    Attributes:
        name: Identifier of this controller (reported in experiment
            configs).
    """

    name: str = "planning_predictor"

    def __init__(
        self,
        predictor: OutcomePredictor,
        n_candidates: int = 16,
        candidate_seed: int = 0,
        horizon_weights: list[float] | None = None,
        pm_mult_range: tuple[float, float] = FULL_ACTION_PM_MULT_RANGE,
    ) -> None:
        """Initialize the planner.

        Args:
            predictor: Fitted outcome predictor; its ``horizons`` fix the
                required length of ``horizon_weights`` and its
                ``input_dim`` (when present) is checked against the
                constructed feature width at deployment.
            n_candidates: Number of candidate actions sampled and scored
                per generation.
            candidate_seed: Seed of the candidate generator; combined with
                ``len(history)`` per call so candidate sets are
                reproducible per history but vary across generations.
            horizon_weights: Nonnegative score weight per predictor
                horizon; ``None`` selects :data:`DEFAULT_HORIZON_WEIGHTS`.
                The weights need not sum to 1 (only their relative size
                matters for the argmax), but the default does.
            pm_mult_range: Log-uniform mutation-multiplier sampling range
                ``(lo, hi)`` with ``0 < lo <= hi``; defaults to the
                Phase-1.5 full-action range ``(0.25, 8.0)``.

        Raises:
            ValueError: If ``n_candidates`` < 1, ``candidate_seed`` < 0,
                ``horizon_weights`` does not have exactly one finite entry
                per predictor horizon, or ``pm_mult_range`` is invalid.
        """
        if int(n_candidates) < 1:
            raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
        if int(candidate_seed) < 0:
            raise ValueError(
                f"candidate_seed must be >= 0, got {candidate_seed}"
            )
        weights = (
            tuple(float(w) for w in DEFAULT_HORIZON_WEIGHTS)
            if horizon_weights is None
            else tuple(float(w) for w in horizon_weights)
        )
        n_horizons = len(predictor.horizons)
        if len(weights) != n_horizons:
            raise ValueError(
                f"horizon_weights must have one entry per predictor horizon "
                f"({n_horizons}), got {len(weights)}"
            )
        if not all(np.isfinite(np.asarray(weights, dtype=np.float64))):
            raise ValueError(f"horizon_weights must be finite, got {weights}")
        lo, hi = float(pm_mult_range[0]), float(pm_mult_range[1])
        if not lo > 0.0 or hi < lo:
            raise ValueError(
                f"pm_mult_range must satisfy 0 < lo <= hi, got {(lo, hi)}"
            )
        self._predictor = predictor
        self._n_candidates = int(n_candidates)
        self._candidate_seed = int(candidate_seed)
        self._horizon_weights = weights
        self._pm_mult_range = (lo, hi)
        self._predictor_path: str | None = None

    @property
    def predictor(self) -> OutcomePredictor:
        """The outcome predictor used for candidate scoring."""
        return self._predictor

    @property
    def n_candidates(self) -> int:
        """Number of candidate actions scored per generation."""
        return self._n_candidates

    @property
    def candidate_seed(self) -> int:
        """Seed of the candidate generator (combined with history length)."""
        return self._candidate_seed

    @property
    def horizon_weights(self) -> tuple[float, ...]:
        """Score weight per predictor horizon."""
        return self._horizon_weights

    @property
    def pm_mult_range(self) -> tuple[float, float]:
        """Log-uniform mutation-multiplier sampling range ``(lo, hi)``."""
        return self._pm_mult_range

    @property
    def predictor_path(self) -> str | None:
        """Recorded location of the predictor checkpoint (metadata only).

        The predictor itself is not serialized by :meth:`save`; this field
        lets experiment code record where the weights live so a run config
        stays self-describing.
        """
        return self._predictor_path

    @predictor_path.setter
    def predictor_path(self, value: str | None) -> None:
        self._predictor_path = None if value is None else str(value)

    @staticmethod
    def _resolve_n_vars(n_vars: int | None) -> int:
        """Resolve the decision-variable count, applying the legacy fallback.

        Args:
            n_vars: Problem dimension, or ``None`` for the legacy fallback
                :data:`PLANNING_FALLBACK_N_VARS`.

        Returns:
            ``int(n_vars)`` or the fallback constant.

        Raises:
            ValueError: If a provided ``n_vars`` is < 1.
        """
        if n_vars is None:
            return PLANNING_FALLBACK_N_VARS
        if int(n_vars) < 1:
            raise ValueError(f"n_vars must be >= 1, got {n_vars}")
        return int(n_vars)

    def predict_action(
        self,
        history: list[dict[str, Any]],
        encoder: Any,
        pm_min: float,
        pm_max: float,
        n_vars: int | None = None,
    ) -> dict[str, Any]:
        """Select the candidate action with the best predicted outcome.

        Thin wrapper over :meth:`predict_action_ex` that discards the
        candidate diagnostics, so the deployed action and the diagnostics
        can never drift apart; that method documents the candidate
        sampling and the weighted-argmax selection rule.

        Args:
            history: Merged state+reward dicts, oldest first, covering
                generations strictly before the action to take.
            encoder: Fitted encoder matching the predictor's training
                encoder (:class:`controller.StateEncoder` or
                :class:`controller.ProblemAwareEncoder`).
            pm_min: Accepted for interface compatibility with
                :meth:`controller.multihead_controller.MultiHeadController.predict_action`;
                unused, because candidate probabilities are drawn from
                ``pm_mult_range`` around ``1 / n_vars`` and therefore
                already lie inside ``[lo / n_vars, hi / n_vars]``.
            pm_max: See ``pm_min``.
            n_vars: Decision-variable count of the problem being solved;
                fixes both the sampling base ``1 / n_vars`` and the
                multiplier feature ``pm * n_vars``. ``None`` applies the
                legacy fallback :data:`PLANNING_FALLBACK_N_VARS`.

        Returns:
            Dict with exactly the keys ``"mutation_operator"``
            (``"polynomial"`` or ``"gaussian"``),
            ``"mutation_probability"``, and ``"exploration_strength"``
            (``eta_m`` for polynomial, ``sigma`` for Gaussian).

        Raises:
            ValueError: If a provided ``n_vars`` is < 1, or the predictor
                exposes an ``input_dim`` that does not equal
                ``encoder.transform(history).size + 4``.
        """
        action, _diagnostics = self.predict_action_ex(
            history, encoder, pm_min, pm_max, n_vars=n_vars
        )
        return action

    def predict_action_ex(
        self,
        history: list[dict[str, Any]],
        encoder: Any,
        pm_min: float,
        pm_max: float,
        n_vars: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Select the best predicted candidate action, plus its diagnostics.

        Samples ``n_candidates`` actions with a generator seeded by
        ``[candidate_seed, len(history)]``, builds one predictor input per
        candidate (``encoder.transform(history)`` concatenated with
        ``[mutation_multiplier, exploration_strength, onehot_polynomial,
        onehot_gaussian]``, exactly the
        :func:`controller.dataset.build_outcome_samples` layout), predicts
        future-HV vectors, scores each candidate as
        ``sum(w_h * pred[i, h])`` over horizons, and selects the argmax
        candidate. Ties break toward the earliest sampled candidate
        (``np.argmax`` semantics), keeping the selection deterministic.

        This is the single implementation of the selection rule:
        :meth:`predict_action` calls it and discards the diagnostics, so
        the returned action and the diagnostics always describe the same
        candidate set.

        Args:
            history: Merged state+reward dicts, oldest first, covering
                generations strictly before the action to take.
            encoder: Fitted encoder matching the predictor's training
                encoder (:class:`controller.StateEncoder` or
                :class:`controller.ProblemAwareEncoder`).
            pm_min: Accepted for interface compatibility with
                :meth:`controller.multihead_controller.MultiHeadController.predict_action`;
                unused, because candidate probabilities are drawn from
                ``pm_mult_range`` around ``1 / n_vars`` and therefore
                already lie inside ``[lo / n_vars, hi / n_vars]``.
            pm_max: See ``pm_min``.
            n_vars: Decision-variable count of the problem being solved;
                fixes both the sampling base ``1 / n_vars`` and the
                multiplier feature ``pm * n_vars``. ``None`` applies the
                legacy fallback :data:`PLANNING_FALLBACK_N_VARS`.

        Returns:
            ``(action, diagnostics)``.

            ``action`` carries exactly the keys ``"mutation_operator"``,
            ``"mutation_probability"`` and ``"exploration_strength"`` and
            is the dict :meth:`predict_action` returns.

            ``diagnostics`` holds one entry per sampled candidate in
            sampling order, so an experiment artifact can reconstruct the
            decision:

            * ``"candidates"``: ``n_candidates`` dicts with keys
              ``"mutation_operator"``, ``"mutation_probability"``,
              ``"exploration_strength"`` and ``"mutation_multiplier"``
              (``mutation_probability * n_vars``, the predictor feature).
            * ``"predicted_hv"``: ``n_candidates`` predicted future-HV
              vectors, one float per predictor horizon (column order of
              ``predictor.horizons``).
            * ``"scores"``: ``n_candidates`` weighted scores
              (``predicted_hv @ horizon_weights``).
            * ``"selected_index"``: index of the executed candidate, equal
              to ``int(np.argmax(scores))``; the executed action is
              ``diagnostics["candidates"]["selected_index"]``.
            * ``"score_margin"``: top-1 minus top-2 score, ``0.0`` when
              only one candidate was sampled.
            * ``"score_std"``: population standard deviation of the
              candidate scores (``0.0`` for a single candidate), i.e. how
              far the predictor separates the candidates at all.

        Raises:
            ValueError: If a provided ``n_vars`` is < 1, or the predictor
                exposes an ``input_dim`` that does not equal
                ``encoder.transform(history).size + 4``.
        """
        history = list(history)
        resolved_n_vars = self._resolve_n_vars(n_vars)
        base_pm = 1.0 / resolved_n_vars
        state_block = np.asarray(encoder.transform(history), dtype=np.float64)
        expected_dim = getattr(self._predictor, "input_dim", None)
        if expected_dim is not None and int(expected_dim) != state_block.size + 4:
            raise ValueError(
                f"predictor input_dim ({expected_dim}) does not match the "
                f"constructed feature width ({state_block.size} + 4); the "
                f"encoder must match the predictor's training encoder"
            )
        rng = np.random.Generator(
            np.random.PCG64([self._candidate_seed, len(history)])
        )
        rows: list[np.ndarray] = []
        candidates: list[tuple[str, float, float]] = []
        multipliers: list[float] = []
        for _ in range(self._n_candidates):
            operator, pm, exploration = sample_full_action(
                rng, base_pm, self._pm_mult_range
            )
            multiplier = float(pm) * resolved_n_vars
            onehot_polynomial = 1.0 if operator == "polynomial" else 0.0
            action_features = np.asarray(
                [multiplier, exploration, onehot_polynomial, 1.0 - onehot_polynomial],
                dtype=np.float64,
            )
            rows.append(np.concatenate([state_block, action_features]))
            candidates.append((operator, float(pm), float(exploration)))
            multipliers.append(multiplier)
        predictions = np.asarray(
            self._predictor.predict(np.vstack(rows)), dtype=np.float64
        )
        scores = predictions @ np.asarray(self._horizon_weights, dtype=np.float64)
        best = int(np.argmax(scores))
        operator, pm, exploration = candidates[best]
        action = {
            "mutation_operator": operator,
            "mutation_probability": float(pm),
            "exploration_strength": float(exploration),
        }
        if scores.size > 1:
            top_two = np.sort(scores)[-2:]
            score_margin = float(top_two[-1] - top_two[0])
        else:
            score_margin = 0.0
        # Diagnostics are derived from the very values the action uses, so
        # ``candidates[selected_index]`` is the executed action.
        candidate_diagnostics = [
            {
                "mutation_operator": candidate_operator,
                "mutation_probability": candidate_pm,
                "exploration_strength": candidate_exploration,
                "mutation_multiplier": candidate_multiplier,
            }
            for (
                candidate_operator,
                candidate_pm,
                candidate_exploration,
            ), candidate_multiplier in zip(candidates, multipliers)
        ]
        diagnostics = {
            "candidates": candidate_diagnostics,
            "predicted_hv": [
                [float(value) for value in predictions[k]]
                for k in range(predictions.shape[0])
            ],
            "scores": [float(value) for value in scores],
            "selected_index": int(best),
            "score_margin": score_margin,
            "score_std": float(np.std(scores)),
        }
        return action, diagnostics

    def save(self, path: str | Path) -> None:
        """Serialize the planner config to JSON.

        The predictor weights are deliberately not embedded; only their
        recorded location (:attr:`predictor_path`) is persisted, and
        :meth:`load` takes the predictor instance explicitly.

        Args:
            path: Destination JSON path; parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "predictor_path": self._predictor_path,
            "config": {
                "n_candidates": self._n_candidates,
                "candidate_seed": self._candidate_seed,
                "horizon_weights": list(self._horizon_weights),
                "pm_mult_range": list(self._pm_mult_range),
            },
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(
        cls, path: str | Path, predictor: OutcomePredictor
    ) -> "PlanningController":
        """Load a planner saved with :meth:`save`.

        Args:
            path: Path to the JSON file written by :meth:`save`.
            predictor: The outcome predictor to plan with (loaded
                separately, e.g. via ``OutcomePredictor.load`` from the
                recorded ``predictor_path``).

        Returns:
            The planner with the persisted config and ``predictor_path``.
        """
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        config = payload["config"]
        controller = cls(
            predictor=predictor,
            n_candidates=int(config["n_candidates"]),
            candidate_seed=int(config["candidate_seed"]),
            horizon_weights=[float(w) for w in config["horizon_weights"]],
            pm_mult_range=(
                float(config["pm_mult_range"][0]),
                float(config["pm_mult_range"][1]),
            ),
        )
        controller.predictor_path = payload.get("predictor_path")
        return controller


#: Re-exported so experiment code can document the candidate exploration
#: ranges without importing the dataset generator a second time.
_CANDIDATE_EXPLORATION_RANGES: dict[str, tuple[float, float]] = {
    "polynomial": ETA_M_SAMPLE_RANGE,
    "gaussian": SIGMA_SAMPLE_RANGE,
}
