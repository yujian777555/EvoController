from __future__ import annotations

"""Tests for the Phase-2 outcome dataset builder and predictor.

All data is synthetic: three hand-made trajectories of 30 generations in
the ``EvolutionRecorder`` schema, long enough for the default horizon set
``[1, 5, 10, 20]`` (valid time steps are ``t`` with ``t + 20 < 30``, so
``t in [0, 9]``). Optional flags on the trajectory builder drop
``mutation_multiplier`` and/or ``exploration_strength`` from the action
dicts to exercise the documented legacy fallbacks. The learnable-pattern
tests use a feature matrix in which each horizon's target is a linear
function of its own feature, so every output column has an independently
learnable signal.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller.dataset import (
    OPERATOR_DEFAULT_EXPLORATION,
    OUTCOME_FALLBACK_N_VARS,
    build_outcome_samples,
    merge_state_reward,
)
from controller.outcome_predictor import OutcomePredictor
from controller.state_encoder import STATE_FEATURES, StateEncoder

_WINDOW = 3
_N_GENERATIONS = 30
_HORIZONS = [1, 5, 10, 20]
_OPERATORS = ("polynomial", "gaussian")


def _transition(
    generation: int,
    hv: float,
    igd: float,
    diversity: float,
    delta_hv: float,
    delta_igd: float,
    operator: str,
    pm: float,
    exploration: float | None,
    multiplier: float | None,
) -> dict[str, Any]:
    """Build one transition dict in the EvolutionRecorder schema.

    ``exploration``/``multiplier=None`` omits the corresponding action
    key, producing a legacy action dict.
    """
    action: dict[str, Any] = {
        "mutation_operator": str(operator),
        "mutation_probability": float(pm),
    }
    if exploration is not None:
        action["exploration_strength"] = float(exploration)
    if multiplier is not None:
        action["mutation_multiplier"] = float(multiplier)
    return {
        "generation": int(generation),
        "state": {
            "generation": int(generation),
            "hv": float(hv),
            "igd": float(igd),
            "diversity": float(diversity),
        },
        "action": action,
        "reward": {
            "delta_hv": float(delta_hv),
            "delta_igd": float(delta_igd),
        },
    }


def _trajectory(
    traj_index: int,
    n: int = _N_GENERATIONS,
    *,
    with_multiplier: bool = True,
    with_exploration: bool = True,
) -> list[dict[str, Any]]:
    """Synthetic trajectory; the operator alternates per generation.

    Multiplier values are deliberately distinct from ``pm * 30`` so the
    recorded-multiplier path and the legacy fallback path produce
    distinguishable feature columns.
    """
    transitions = []
    for t in range(n):
        operator = _OPERATORS[(t + traj_index) % 2]
        pm = 0.02 + 0.004 * ((3 * t + traj_index) % 7)
        exploration = (
            8.0 + 0.5 * t if operator == "polynomial" else 0.05 + 0.002 * t
        )
        multiplier = 0.5 + 0.3 * ((2 * t + traj_index) % 4)
        reward = 0.008 if t > 0 else 0.0
        transitions.append(
            _transition(
                generation=t,
                hv=0.30 + 0.008 * t + 0.05 * traj_index,
                igd=0.60 - 0.01 * t,
                diversity=0.25 + 0.005 * t,
                delta_hv=reward,
                delta_igd=0.4 * reward,
                operator=operator,
                pm=pm,
                exploration=exploration if with_exploration else None,
                multiplier=multiplier if with_multiplier else None,
            )
        )
    return transitions


@pytest.fixture
def trajectories() -> list[list[dict[str, Any]]]:
    """Three synthetic 30-generation trajectories with full actions."""
    return [_trajectory(j) for j in range(3)]


@pytest.fixture
def encoder(trajectories: list[list[dict[str, Any]]]) -> StateEncoder:
    """Fitted plain encoder with the test window."""
    return StateEncoder(window=_WINDOW).fit(trajectories)


class TestBuildOutcomeSamples:
    def test_shapes_dtypes_and_ids(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        """30 generations with max horizon 20 give t in [0, 9]: 3 x 10
        samples, X of width encoder.dim + 4, everything float64."""
        X, y, traj_ids, sample_indices = build_outcome_samples(
            trajectories, encoder, _WINDOW
        )
        n_expected = 3 * (_N_GENERATIONS - max(_HORIZONS))
        assert X.shape == (n_expected, encoder.dim + 4)
        assert encoder.dim == _WINDOW * len(STATE_FEATURES)
        assert y.shape == (n_expected, len(_HORIZONS))
        assert traj_ids.shape == sample_indices.shape == (n_expected,)
        for array in (X, y, traj_ids, sample_indices):
            assert array.dtype == np.float64
        for j in range(3):
            assert int(np.sum(traj_ids == j)) == _N_GENERATIONS - max(_HORIZONS)
            # Valid t range is exactly [0, 29 - 20].
            assert set(sample_indices[traj_ids == j]) == set(
                range(_N_GENERATIONS - max(_HORIZONS))
            )
        # Samples appear in trajectory-major, time order.
        assert np.array_equal(traj_ids, np.repeat(np.arange(3.0), 10))
        assert np.array_equal(sample_indices, np.tile(np.arange(10.0), 3))

    def test_state_block_matches_encoder_transform(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        """The state block equals encoder.transform of the merged dicts of
        transitions [t - window + 1, t] (inclusive of t)."""
        X, _, traj_ids, sample_indices = build_outcome_samples(
            trajectories, encoder, _WINDOW
        )
        for i in range(X.shape[0]):
            j = int(traj_ids[i])
            t = int(sample_indices[i])
            history = [
                merge_state_reward(trajectories[j][k])
                for k in range(max(0, t - _WINDOW + 1), t + 1)
            ]
            np.testing.assert_array_equal(
                X[i, : encoder.dim], encoder.transform(history)
            )

    def test_y_equals_future_hv_exactly(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        """Each y entry is the verbatim hv of transition t + horizon."""
        X, y, traj_ids, sample_indices = build_outcome_samples(
            trajectories, encoder, _WINDOW, horizons=_HORIZONS
        )
        for i in range(y.shape[0]):
            j = int(traj_ids[i])
            t = int(sample_indices[i])
            for k, h in enumerate(_HORIZONS):
                assert y[i, k] == trajectories[j][t + h]["state"]["hv"]

    def test_operator_onehot(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        """polynomial -> [1, 0], gaussian -> [0, 1] in the last two
        columns, following the recorded operator of generation t."""
        X, _, traj_ids, sample_indices = build_outcome_samples(
            trajectories, encoder, _WINDOW
        )
        poly_col, gauss_col = encoder.dim + 2, encoder.dim + 3
        for i in range(X.shape[0]):
            operator = trajectories[int(traj_ids[i])][int(sample_indices[i])][
                "action"
            ]["mutation_operator"]
            expected = (
                (1.0, 0.0) if operator == "polynomial" else (0.0, 1.0)
            )
            assert (X[i, poly_col], X[i, gauss_col]) == expected

    def test_multiplier_and_exploration_columns(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        """Recorded mutation_multiplier and exploration_strength are used
        verbatim when present."""
        X, _, traj_ids, sample_indices = build_outcome_samples(
            trajectories, encoder, _WINDOW
        )
        mult_col, expl_col = encoder.dim, encoder.dim + 1
        for i in range(X.shape[0]):
            action = trajectories[int(traj_ids[i])][int(sample_indices[i])][
                "action"
            ]
            assert X[i, mult_col] == action["mutation_multiplier"]
            assert X[i, expl_col] == action["exploration_strength"]

    def test_boundary_skip(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        """Default horizons keep only t with t + 20 < 30; horizons=[1]
        keeps t with t + 1 < 30 (29 per trajectory)."""
        _, _, _, sample_indices = build_outcome_samples(
            trajectories, encoder, _WINDOW
        )
        assert sample_indices.max() == _N_GENERATIONS - 1 - max(_HORIZONS)
        X1, y1, traj_ids1, idx1 = build_outcome_samples(
            trajectories, encoder, _WINDOW, horizons=[1]
        )
        assert X1.shape == (3 * 29, encoder.dim + 4)
        assert y1.shape == (3 * 29, 1)
        for j in range(3):
            assert set(idx1[traj_ids1 == j]) == set(range(_N_GENERATIONS - 1))

    def test_multiplier_fallback_uses_pm_times_30(
        self, trajectories: list[list[dict[str, Any]]], encoder: StateEncoder
    ) -> None:
        """Actions lacking mutation_multiplier get pm * 30 (the legacy
        datasets' n_vars; a raw transition carries no n_vars)."""
        legacy = [_trajectory(j, with_multiplier=False) for j in range(3)]
        X, _, traj_ids, sample_indices = build_outcome_samples(
            legacy, encoder, _WINDOW
        )
        assert OUTCOME_FALLBACK_N_VARS == 30
        mult_col = encoder.dim
        for i in range(X.shape[0]):
            action = legacy[int(traj_ids[i])][int(sample_indices[i])]["action"]
            assert X[i, mult_col] == action["mutation_probability"] * 30

    def test_exploration_imputation_for_legacy_actions(
        self, encoder: StateEncoder
    ) -> None:
        """Actions lacking exploration_strength get the operator defaults
        (eta_m=20 polynomial, sigma=0.1 gaussian), matching the Phase-1.5
        multi-head builder convention."""
        legacy = [_trajectory(j, with_exploration=False) for j in range(3)]
        X, _, traj_ids, sample_indices = build_outcome_samples(
            legacy, encoder, _WINDOW
        )
        expl_col = encoder.dim + 1
        for i in range(X.shape[0]):
            operator = legacy[int(traj_ids[i])][int(sample_indices[i])][
                "action"
            ]["mutation_operator"]
            assert X[i, expl_col] == OPERATOR_DEFAULT_EXPLORATION[operator]

    def test_unknown_operator_raises(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        """Unknown operator names are rejected, never silently encoded."""
        trajectories[0][0]["action"]["mutation_operator"] = "simulated_binary"
        with pytest.raises(ValueError, match="mutation_operator"):
            build_outcome_samples(trajectories, encoder, _WINDOW)

    def test_empty_input_returns_empty_arrays(
        self, encoder: StateEncoder
    ) -> None:
        X, y, traj_ids, sample_indices = build_outcome_samples(
            [], encoder, _WINDOW
        )
        assert X.shape == (0, encoder.dim + 4)
        assert y.shape == (0, len(_HORIZONS))
        assert traj_ids.shape == sample_indices.shape == (0,)

    def test_short_trajectory_contributes_no_samples(
        self, encoder: StateEncoder
    ) -> None:
        """A trajectory of length <= max(horizons) yields no samples."""
        short = [_trajectory(0, n=max(_HORIZONS))]
        X, y, traj_ids, sample_indices = build_outcome_samples(
            short, encoder, _WINDOW
        )
        assert X.shape == (0, encoder.dim + 4)
        assert y.shape == (0, len(_HORIZONS))
        assert traj_ids.shape == sample_indices.shape == (0,)


def _learnable_outcome(
    n: int = 200, dim: int = 10, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic dataset with one linearly learnable signal per horizon.

    Horizon k's target depends only on feature k, so every output column
    has an independent, learnable signal.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, dim))
    slopes = [0.5, -0.4, 0.3, -0.2]
    intercepts = [-0.1, 0.2, 0.0, 0.5]
    y = np.column_stack(
        [slopes[k] * X[:, k] + intercepts[k] for k in range(len(slopes))]
    )
    return X, y


class TestOutcomePredictor:
    def test_determinism_same_seed(self) -> None:
        """Same seed + data -> identical predictions within 1e-6."""
        X, y = _learnable_outcome()
        preds = []
        for _ in range(2):
            predictor = OutcomePredictor(
                input_dim=10, horizons=_HORIZONS, hidden_dims=(16,), seed=7
            )
            predictor.fit(X, y, epochs=40, batch_size=32)
            preds.append(predictor.predict(X))
        np.testing.assert_allclose(preds[0], preds[1], atol=1e-6, rtol=0.0)

    def test_loss_decreases_on_learnable_pattern(self) -> None:
        """Training loss drops sharply when each horizon has signal."""
        X, y = _learnable_outcome()
        predictor = OutcomePredictor(
            input_dim=10, horizons=_HORIZONS, hidden_dims=(32, 32), seed=0, lr=5e-3
        )
        history = predictor.fit(X, y, epochs=150, batch_size=32)
        train_loss = history["train_loss"]
        assert len(train_loss) == 150
        assert history["val_loss"] == []
        assert train_loss[-1] < 0.2 * train_loss[0]

    def test_sample_weight_accepted(self) -> None:
        """Weighted training runs and reports one loss per epoch."""
        X, y = _learnable_outcome(n=60)
        rng = np.random.default_rng(0)
        w = rng.uniform(0.1, 2.0, size=60)
        predictor = OutcomePredictor(input_dim=10, hidden_dims=(8,), seed=1)
        history = predictor.fit(X, y, sample_weight=w, epochs=15, batch_size=16)
        assert len(history["train_loss"]) == 15
        assert all(np.isfinite(history["train_loss"]))

    def test_val_loss_reported_when_validation_given(self) -> None:
        X, y = _learnable_outcome()
        X_val, y_val = _learnable_outcome(n=40, seed=7)
        predictor = OutcomePredictor(
            input_dim=10, hidden_dims=(16,), seed=2, lr=5e-3
        )
        history = predictor.fit(
            X, y, epochs=50, batch_size=32, X_val=X_val, y_val=y_val
        )
        assert len(history["val_loss"]) == 50
        assert history["val_loss"][-1] < history["val_loss"][0]

    def test_predict_shape(self) -> None:
        """predict returns (n, len(horizons)) float64; 1-D input is one
        sample."""
        predictor = OutcomePredictor(input_dim=10, horizons=_HORIZONS, seed=0)
        preds = predictor.predict(np.zeros((5, 10)))
        assert preds.shape == (5, 4)
        assert preds.dtype == np.float64
        assert predictor.predict(np.zeros(10)).shape == (1, 4)

    def test_fit_validation(self) -> None:
        predictor = OutcomePredictor(input_dim=10, horizons=_HORIZONS, seed=0)
        with pytest.raises(ValueError):
            predictor.fit(np.zeros((0, 10)), np.zeros((0, 4)))
        with pytest.raises(ValueError):
            predictor.fit(np.zeros((10, 10)), np.zeros((10, 3)))
        with pytest.raises(ValueError):
            predictor.fit(
                np.zeros((10, 10)), np.zeros((10, 4)), sample_weight=np.ones(3)
            )

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        X, y = _learnable_outcome(n=60)
        predictor = OutcomePredictor(
            input_dim=10, horizons=_HORIZONS, hidden_dims=(16, 8), seed=11, lr=2e-3
        )
        predictor.fit(X, y, epochs=5, batch_size=8)
        path = tmp_path / "outcome.pt"
        predictor.save(path)
        loaded = OutcomePredictor.load(path)
        assert loaded.input_dim == predictor.input_dim
        assert loaded.horizons == predictor.horizons
        assert loaded.hidden_dims == predictor.hidden_dims
        np.testing.assert_array_equal(loaded.predict(X), predictor.predict(X))
