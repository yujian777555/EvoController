from __future__ import annotations

"""Tests for the Phase-2B planning controller (candidate-action planner).

Two predictor flavors are exercised:

* a real, lightly-trained :class:`controller.outcome_predictor.OutcomePredictor`
  for the deployment path (determinism, action schema, save/load);
* a deterministic spy predictor that records every feature matrix handed
  to ``predict`` for the mechanism tests: bit-exact feature layout
  against :func:`controller.dataset.build_outcome_samples`, the
  weighted-argmax selection rule, the candidate-seeding scheme, and the
  ``n_vars`` fallback.

All trajectories are synthetic and follow the ``EvolutionRecorder``
schema, mirroring the fixtures of ``test_outcome_predictor.py``.
"""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from controller.dataset import build_outcome_samples, merge_state_reward
from controller.outcome_predictor import OutcomePredictor
from controller.planning_controller import (
    DEFAULT_HORIZON_WEIGHTS,
    PLANNING_FALLBACK_N_VARS,
    PlanningController,
)
from controller.state_encoder import StateEncoder
from experiments.generate_dataset import (
    FULL_ACTION_PM_MULT_RANGE,
    sample_full_action,
)

_WINDOW = 3
_HORIZONS = [1, 2]
_N_GENERATIONS = 12


def _trajectories(n_traj: int = 2) -> list[list[dict[str, Any]]]:
    """Two synthetic recorder-schema trajectories of 12 generations."""
    trajectories: list[list[dict[str, Any]]] = []
    for j in range(n_traj):
        hv = 0.30 + 0.05 * j
        igd = 0.50 - 0.02 * j
        transitions = []
        for t in range(_N_GENERATIONS):
            delta_hv = 0.0 if t == 0 else 0.006 + 0.001 * ((t + j) % 3)
            delta_igd = 0.0 if t == 0 else 0.005 + 0.001 * ((t + j + 1) % 3)
            hv += delta_hv
            igd -= delta_igd
            transitions.append(
                {
                    "generation": t,
                    "state": {
                        "generation": t,
                        "hv": hv,
                        "igd": igd,
                        "diversity": 0.25 + 0.01 * t,
                    },
                    "action": {
                        "mutation_operator": "gaussian" if t % 2 else "polynomial",
                        "mutation_probability": 1.0 / 30.0,
                        "exploration_strength": 20.0 if t % 2 == 0 else 0.1,
                    },
                    "reward": {"delta_hv": delta_hv, "delta_igd": delta_igd},
                }
            )
        trajectories.append(transitions)
    return trajectories


class _SpyPredictor:
    """Recorder-style predictor: logs ``predict`` inputs, echoes a response.

    Provides exactly the surface :class:`PlanningController` consumes
    (``horizons``, optional ``input_dim``, ``predict``) so the mechanism
    tests can observe the deployed feature matrix.
    """

    def __init__(
        self,
        input_dim: int,
        horizons: tuple[int, ...] = tuple(_HORIZONS),
        responder: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self._input_dim = int(input_dim)
        self._horizons = tuple(int(h) for h in horizons)
        self.calls: list[np.ndarray] = []
        self._responder = responder

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def horizons(self) -> tuple[int, ...]:
        return self._horizons

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        self.calls.append(X.copy())
        if self._responder is not None:
            return self._responder(X)
        return np.zeros((X.shape[0], len(self._horizons)), dtype=np.float64)


def _fitted_encoder() -> StateEncoder:
    return StateEncoder(_WINDOW).fit(_trajectories())


def _history_for_t(trajectories: list[list[dict[str, Any]]], t: int) -> list[dict[str, Any]]:
    """Full merged history prefix up to and including time step ``t``.

    This is the deployment convention of ``evaluate_snapshot`` (the whole
    observable history; the encoder tail-slices its window internally),
    so the candidate generator seed ``[seed, len(history)]`` varies per
    generation as documented.
    """
    merged = [merge_state_reward(tr) for tr in trajectories[0]]
    return merged[: t + 1]


def test_default_horizon_weights_match_default_horizons() -> None:
    """The default weights cover the default predictor horizon grid."""
    assert len(DEFAULT_HORIZON_WEIGHTS) == len([1, 5, 10, 20])
    assert sum(DEFAULT_HORIZON_WEIGHTS) == pytest.approx(1.0)
    assert PLANNING_FALLBACK_N_VARS == 30


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_candidates": 0},
        {"n_candidates": -3},
        {"candidate_seed": -1},
        {"horizon_weights": [0.5]},  # wrong length for horizons [1, 2]
        {"horizon_weights": [0.5, float("nan")]},
        {"horizon_weights": [0.5, float("inf")]},
        {"pm_mult_range": (0.0, 8.0)},  # lo must be > 0
        {"pm_mult_range": (4.0, 2.0)},  # hi must be >= lo
    ],
)
def test_constructor_rejects_invalid_config(kwargs: dict[str, Any]) -> None:
    """Invalid constructor settings raise ValueError."""
    predictor = _SpyPredictor(input_dim=_WINDOW * 6 + 4)
    with pytest.raises(ValueError):
        PlanningController(predictor, **kwargs)


def test_constructor_accepts_valid_config_and_exposes_properties() -> None:
    """Valid settings are stored and readable via the properties."""
    predictor = _SpyPredictor(input_dim=_WINDOW * 6 + 4)
    planner = PlanningController(
        predictor,
        n_candidates=5,
        candidate_seed=11,
        horizon_weights=[0.25, 0.75],
        pm_mult_range=(0.5, 4.0),
    )
    assert planner.predictor is predictor
    assert planner.n_candidates == 5
    assert planner.candidate_seed == 11
    assert planner.horizon_weights == (0.25, 0.75)
    assert planner.pm_mult_range == (0.5, 4.0)
    assert planner.predictor_path is None


def test_predict_action_with_real_predictor_is_deterministic() -> None:
    """The deployment path with a fitted OutcomePredictor is reproducible."""
    trajectories = _trajectories()
    encoder = StateEncoder(_WINDOW).fit(trajectories)
    X, y, _, _ = build_outcome_samples(trajectories, encoder, _WINDOW, _HORIZONS)
    assert X.shape[1] == encoder.dim + 4
    predictor = OutcomePredictor(
        input_dim=X.shape[1], horizons=_HORIZONS, hidden_dims=(8, 8), seed=0
    )
    predictor.fit(X, y, epochs=3)
    planner = PlanningController(
        predictor, n_candidates=8, candidate_seed=7, horizon_weights=[0.4, 0.6]
    )
    history = _history_for_t(trajectories, 4)
    action_a = planner.predict_action(history, encoder, 0.0, 1.0)
    action_b = planner.predict_action(history, encoder, 0.0, 1.0)
    assert action_a == action_b
    assert set(action_a) == {
        "mutation_operator",
        "mutation_probability",
        "exploration_strength",
    }
    assert action_a["mutation_operator"] in ("polynomial", "gaussian")
    # Candidates are drawn from the Phase-1.5 full action space.
    lo, hi = FULL_ACTION_PM_MULT_RANGE
    assert lo / PLANNING_FALLBACK_N_VARS <= action_a["mutation_probability"]
    assert action_a["mutation_probability"] <= hi / PLANNING_FALLBACK_N_VARS


def test_predict_action_scores_every_candidate_once() -> None:
    """One predict() call covers exactly n_candidates rows."""
    encoder = _fitted_encoder()
    spy = _SpyPredictor(input_dim=encoder.dim + 4)
    planner = PlanningController(spy, n_candidates=6, horizon_weights=[0.5, 0.5])
    planner.predict_action(_history_for_t(_trajectories(), 4), encoder, 0.0, 1.0)
    assert len(spy.calls) == 1
    rows = spy.calls[0]
    assert rows.shape == (6, encoder.dim + 4)


def test_feature_layout_matches_outcome_samples_bit_exactly() -> None:
    """Rows equal encoder block + [multiplier, exploration, onehot_p, onehot_g].

    The state block must be bit-identical to the corresponding columns of
    :func:`build_outcome_samples` for the same history window, and the
    action block must decode back to the sampled candidate under the
    documented feature order — the planner then feeds the predictor
    exactly the layout it was trained on.
    """
    trajectories = _trajectories()
    encoder = StateEncoder(_WINDOW).fit(trajectories)
    X, _, _, sample_indices = build_outcome_samples(
        trajectories, encoder, _WINDOW, _HORIZONS
    )
    t = 1
    row = int(np.where(sample_indices == t)[0][0])
    history = _history_for_t(trajectories, t)
    state_block = np.asarray(encoder.transform(history), dtype=np.float64)
    # The dataset builder's state block is the planner's state block
    # (the encoder tail-slices its window, so the full prefix encodes
    # exactly the window ``build_outcome_samples`` used).
    assert np.array_equal(X[row, : encoder.dim], state_block)

    spy = _SpyPredictor(input_dim=encoder.dim + 4)
    planner = PlanningController(spy, n_candidates=16, horizon_weights=[0.5, 0.5])
    planner.predict_action(history, encoder, 0.0, 1.0)
    rows = spy.calls[0]
    for candidate in rows:
        assert np.array_equal(candidate[: encoder.dim], state_block)
        multiplier, exploration, onehot_p, onehot_g = candidate[encoder.dim :]
        assert onehot_p + onehot_g == pytest.approx(1.0)
        assert {onehot_p, onehot_g} <= {0.0, 1.0}
        operator = "polynomial" if onehot_p == 1.0 else "gaussian"
        # Multiplier feature is pm * n_vars; with the fallback n_vars = 30
        # every sampled pm lies in the full-action multiplier range.
        assert 0.25 <= multiplier <= 8.0
        if operator == "polynomial":
            assert 2.0 <= exploration <= 50.0
        else:
            assert 0.02 <= exploration <= 0.3


def test_candidates_follow_sample_full_action_with_pinned_seed() -> None:
    """Candidate actions equal sample_full_action draws from PCG64([seed, t]).

    This pins the documented seeding scheme: reproducible per history,
    and the draws are distribution-identical to the training data
    because they come from the very same sampler.
    """
    trajectories = _trajectories()
    encoder = StateEncoder(_WINDOW).fit(trajectories)
    t = 5
    history = _history_for_t(trajectories, t)
    spy = _SpyPredictor(input_dim=encoder.dim + 4)
    seed, n_candidates = 3, 12
    planner = PlanningController(
        spy, n_candidates=n_candidates, candidate_seed=seed, horizon_weights=[0.5, 0.5]
    )
    planner.predict_action(history, encoder, 0.0, 1.0)
    rows = spy.calls[0]

    rng = np.random.Generator(np.random.PCG64([seed, len(history)]))
    for k in range(n_candidates):
        operator, pm, exploration = sample_full_action(
            rng, 1.0 / PLANNING_FALLBACK_N_VARS, FULL_ACTION_PM_MULT_RANGE
        )
        multiplier, expl_feature, onehot_p, onehot_g = rows[k, encoder.dim :]
        assert multiplier == pytest.approx(pm * PLANNING_FALLBACK_N_VARS)
        assert expl_feature == pytest.approx(exploration)
        assert (onehot_p, onehot_g) == (
            (1.0, 0.0) if operator == "polynomial" else (0.0, 1.0)
        )


def test_selected_action_is_weighted_argmax() -> None:
    """The returned action is the argmax of the weighted predicted HV.

    The spy responds with ``pred[i, k] = multiplier_i * (k + 1)`` so the
    planner's score ``sum(w_k * pred[i, k])`` is a known per-candidate
    number; the chosen candidate must be its argmax (earliest on ties).
    """
    trajectories = _trajectories()
    encoder = StateEncoder(_WINDOW).fit(trajectories)
    weights = (0.3, 0.7)

    def responder(rows: np.ndarray) -> np.ndarray:
        multiplier = rows[:, -4]
        factors = np.asarray([k + 1.0 for k in range(len(_HORIZONS))])
        return multiplier[:, None] * factors[None, :]

    spy = _SpyPredictor(input_dim=encoder.dim + 4, responder=responder)
    planner = PlanningController(
        spy, n_candidates=16, candidate_seed=9, horizon_weights=list(weights)
    )
    history = _history_for_t(trajectories, 4)
    action = planner.predict_action(history, encoder, 0.0, 1.0)
    rows = spy.calls[0]
    scores = responder(rows) @ np.asarray(weights, dtype=np.float64)
    best = int(np.argmax(scores))
    multiplier, exploration, onehot_p, onehot_g = rows[best, encoder.dim :]
    assert action["mutation_probability"] == pytest.approx(
        multiplier / PLANNING_FALLBACK_N_VARS
    )
    assert action["exploration_strength"] == pytest.approx(exploration)
    assert action["mutation_operator"] == (
        "polynomial" if onehot_p == 1.0 else "gaussian"
    )


def test_candidates_vary_across_generations() -> None:
    """Different history lengths draw different candidate sets (same seed)."""
    trajectories = _trajectories()
    encoder = StateEncoder(_WINDOW).fit(trajectories)
    spy = _SpyPredictor(input_dim=encoder.dim + 4)
    planner = PlanningController(
        spy, n_candidates=16, candidate_seed=0, horizon_weights=[0.5, 0.5]
    )
    planner.predict_action(_history_for_t(trajectories, 3), encoder, 0.0, 1.0)
    planner.predict_action(_history_for_t(trajectories, 6), encoder, 0.0, 1.0)
    blocks_a = spy.calls[0][:, encoder.dim :]
    blocks_b = spy.calls[1][:, encoder.dim :]
    assert not np.array_equal(blocks_a, blocks_b)
    # Same history length -> identical candidate set again.
    planner.predict_action(_history_for_t(trajectories, 3), encoder, 0.0, 1.0)
    assert np.array_equal(spy.calls[2][:, encoder.dim :], blocks_a)


def test_n_vars_scales_base_pm_and_multiplier() -> None:
    """With n_vars=10 the base pm is 1/10 and the multiplier is pm * 10."""
    trajectories = _trajectories()
    encoder = StateEncoder(_WINDOW).fit(trajectories)
    spy = _SpyPredictor(input_dim=encoder.dim + 4)
    planner = PlanningController(
        spy, n_candidates=16, horizon_weights=[0.5, 0.5]
    )
    history = _history_for_t(trajectories, 4)
    action = planner.predict_action(history, encoder, 0.0, 1.0, n_vars=10)
    rows = spy.calls[0]
    lo, hi = FULL_ACTION_PM_MULT_RANGE
    for k in range(rows.shape[0]):
        multiplier = rows[k, encoder.dim]
        assert lo <= multiplier <= hi
    assert lo / 10.0 <= action["mutation_probability"] <= hi / 10.0


@pytest.mark.parametrize("n_vars", [0, -2])
def test_invalid_n_vars_raises(n_vars: int) -> None:
    """A non-positive n_vars is rejected."""
    encoder = _fitted_encoder()
    spy = _SpyPredictor(input_dim=encoder.dim + 4)
    planner = PlanningController(spy, horizon_weights=[0.5, 0.5])
    with pytest.raises(ValueError):
        planner.predict_action(
            _history_for_t(_trajectories(), 4), encoder, 0.0, 1.0, n_vars=n_vars
        )


def test_input_dim_mismatch_raises() -> None:
    """A predictor trained on another encoder width is rejected."""
    encoder = _fitted_encoder()
    spy = _SpyPredictor(input_dim=encoder.dim + 5)
    planner = PlanningController(spy, horizon_weights=[0.5, 0.5])
    with pytest.raises(ValueError, match="input_dim"):
        planner.predict_action(
            _history_for_t(_trajectories(), 4), encoder, 0.0, 1.0
        )


def test_save_load_round_trip(tmp_path: Path) -> None:
    """save/load restores the config and the predictor_path annotation."""
    trajectories = _trajectories()
    encoder = StateEncoder(_WINDOW).fit(trajectories)
    X, y, _, _ = build_outcome_samples(trajectories, encoder, _WINDOW, _HORIZONS)
    predictor = OutcomePredictor(
        input_dim=X.shape[1], horizons=_HORIZONS, hidden_dims=(8, 8), seed=0
    )
    predictor.fit(X, y, epochs=2)
    predictor_path = tmp_path / "predictor.pt"
    predictor.save(predictor_path)

    planner = PlanningController(
        predictor, n_candidates=8, candidate_seed=4, horizon_weights=[0.25, 0.75]
    )
    planner.predictor_path = str(predictor_path)
    config_path = tmp_path / "planner.json"
    planner.save(config_path)

    restored_predictor = OutcomePredictor.load(predictor_path)
    restored = PlanningController.load(config_path, restored_predictor)
    assert restored.n_candidates == 8
    assert restored.candidate_seed == 4
    assert restored.horizon_weights == (0.25, 0.75)
    assert restored.pm_mult_range == tuple(FULL_ACTION_PM_MULT_RANGE)
    assert restored.predictor_path == str(predictor_path)

    history = _history_for_t(trajectories, 4)
    assert restored.predict_action(history, encoder, 0.0, 1.0) == (
        planner.predict_action(history, encoder, 0.0, 1.0)
    )
