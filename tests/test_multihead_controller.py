from __future__ import annotations

"""Tests for the Phase-1.5 multi-head controller and dataset builder.

All data is synthetic: trajectories are hand-made transition dicts in the
``EvolutionRecorder`` schema, covering both the full three-key Phase-1.5
action (``mutation_operator``, ``mutation_probability``,
``exploration_strength``) and the legacy two-key Phase-0/1 action (no
``exploration_strength``) to exercise the documented imputation path. The
learnable-pattern tests use a feature matrix in which the operator label
correlates with one feature, log pm with another, and log exploration
strength with a third, so each head has an independently learnable signal.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks import get_problem
from controller.dataset import (
    OPERATOR_DEFAULT_EXPLORATION,
    build_multihead_samples,
    build_supervised_samples,
    merge_state_reward,
)
from controller.multihead_controller import (
    GAUSSIAN_EXPLORATION_RANGE,
    OPERATOR_CLASSES,
    POLYNOMIAL_EXPLORATION_RANGE,
    MultiHeadController,
)
from controller.state_encoder import STATE_FEATURES, ProblemAwareEncoder, StateEncoder

_WINDOW = 3
_N_TRANSITIONS = 12


def _transition(
    generation: int,
    hv: float,
    igd: float,
    diversity: float,
    delta_hv: float,
    delta_igd: float,
    operator: str,
    pm: float,
    exploration: float | None = None,
) -> dict[str, Any]:
    """Build one transition dict in the EvolutionRecorder schema.

    ``exploration=None`` produces a legacy two-key action dict (Phase-0/1
    data); otherwise the full three-key Phase-1.5 action is written.
    """
    action: dict[str, Any] = {
        "mutation_operator": str(operator),
        "mutation_probability": float(pm),
    }
    if exploration is not None:
        action["exploration_strength"] = float(exploration)
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


def _full_trajectory(traj_index: int, n: int = _N_TRANSITIONS) -> list[dict[str, Any]]:
    """Three-key-action trajectory; the operator alternates per generation."""
    transitions = []
    for t in range(n):
        operator = OPERATOR_CLASSES[(t + traj_index) % 2]
        pm = 0.02 + 0.005 * ((t * 2 + traj_index) % 5)
        exploration = 5.0 + 2.0 * t if operator == "polynomial" else 0.05 + 0.01 * t
        reward = 0.01 * ((t + traj_index) % 3) if t > 0 else 0.0
        transitions.append(
            _transition(
                generation=t,
                hv=0.40 + 0.01 * t + 0.02 * traj_index,
                igd=0.50 - 0.01 * t,
                diversity=0.30 + 0.01 * t,
                delta_hv=reward,
                delta_igd=0.5 * reward,
                operator=operator,
                pm=pm,
                exploration=exploration,
            )
        )
    return transitions


def _legacy_trajectory(traj_index: int, operator: str) -> list[dict[str, Any]]:
    """Two-key-action (legacy) trajectory with a fixed operator."""
    transitions = []
    for t in range(_N_TRANSITIONS):
        reward = 0.01 * (t % 2) if t > 0 else 0.0
        transitions.append(
            _transition(
                generation=t,
                hv=0.35 + 0.01 * t + 0.03 * traj_index,
                igd=0.45 - 0.01 * t,
                diversity=0.25 + 0.02 * t,
                delta_hv=reward,
                delta_igd=reward,
                operator=operator,
                pm=0.03 + 0.002 * t,
                exploration=None,
            )
        )
    return transitions


@pytest.fixture
def full_trajectories() -> list[list[dict[str, Any]]]:
    """Three synthetic three-key trajectories."""
    return [_full_trajectory(j) for j in range(3)]


@pytest.fixture
def encoder(full_trajectories: list[list[dict[str, Any]]]) -> StateEncoder:
    """Fitted plain encoder with the test window."""
    return StateEncoder(window=_WINDOW).fit(full_trajectories)


class TestBuildMultiheadSamples:
    def test_alignment_matches_build_supervised_samples(
        self,
        full_trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        """X, weights, trajectory ids, and log-pm targets are identical to
        the Phase-1 dataset builder given the same trajectories/encoder."""
        X_m, _, y_logpm, _, w_m, ids_m = build_multihead_samples(
            full_trajectories, encoder, _WINDOW
        )
        X_s, y_s, w_s, ids_s = build_supervised_samples(
            full_trajectories, encoder, _WINDOW
        )
        np.testing.assert_array_equal(X_m, X_s)
        np.testing.assert_allclose(y_logpm, y_s, rtol=1e-12)
        np.testing.assert_allclose(w_m, w_s, rtol=1e-12)
        np.testing.assert_array_equal(ids_m, ids_s)

    def test_operator_encoding(
        self,
        full_trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        """y_op is 0 for polynomial and 1 for gaussian, in sample order."""
        _, y_op, _, _, _, _ = build_multihead_samples(
            full_trajectories, encoder, _WINDOW
        )
        expected = np.asarray(
            [
                (t + j) % 2  # _full_trajectory alternates operators this way
                for j in range(3)
                for t in range(1, _N_TRANSITIONS)
            ],
            dtype=int,
        )
        np.testing.assert_array_equal(y_op, expected)
        assert set(np.unique(y_op)) == {0, 1}

    def test_exploration_targets_used_when_present(
        self,
        full_trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        """y_logexpl is the log of the recorded exploration strength."""
        _, _, _, y_logexpl, _, _ = build_multihead_samples(
            full_trajectories, encoder, _WINDOW
        )
        expected = np.asarray(
            [
                math.log(
                    full_trajectories[j][t]["action"]["exploration_strength"]
                )
                for j in range(3)
                for t in range(1, _N_TRANSITIONS)
            ]
        )
        np.testing.assert_allclose(y_logexpl, expected, rtol=1e-12)

    def test_imputation_of_missing_exploration_strength(
        self, encoder: StateEncoder
    ) -> None:
        """Legacy two-key actions get the operator defaults: eta_m=20 for
        polynomial, sigma=0.1 for gaussian."""
        trajectories = [_legacy_trajectory(0, "polynomial"), _legacy_trajectory(1, "gaussian")]
        _, y_op, _, y_logexpl, _, traj_ids = build_multihead_samples(
            trajectories, encoder, _WINDOW
        )
        np.testing.assert_allclose(
            y_logexpl[traj_ids == 0],
            math.log(OPERATOR_DEFAULT_EXPLORATION["polynomial"]),
            rtol=1e-12,
        )
        assert math.log(OPERATOR_DEFAULT_EXPLORATION["polynomial"]) == pytest.approx(
            math.log(20.0)
        )
        np.testing.assert_allclose(
            y_logexpl[traj_ids == 1],
            math.log(OPERATOR_DEFAULT_EXPLORATION["gaussian"]),
            rtol=1e-12,
        )
        assert math.log(OPERATOR_DEFAULT_EXPLORATION["gaussian"]) == pytest.approx(
            math.log(0.1)
        )
        # The operator labels follow the recorded operator even for legacy data.
        assert set(np.unique(y_op[traj_ids == 0])) == {0}
        assert set(np.unique(y_op[traj_ids == 1])) == {1}

    def test_shapes_and_traj_ids(
        self,
        full_trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        X, y_op, y_logpm, y_logexpl, w, traj_ids = build_multihead_samples(
            full_trajectories, encoder, _WINDOW
        )
        n_expected = 3 * (_N_TRANSITIONS - 1)
        assert X.shape == (n_expected, _WINDOW * len(STATE_FEATURES))
        assert y_op.shape == y_logpm.shape == y_logexpl.shape == w.shape == (n_expected,)
        assert y_op.dtype.kind == "i"
        assert traj_ids.shape == (n_expected,)
        for j in range(3):
            assert int(np.sum(traj_ids == j)) == _N_TRANSITIONS - 1
        assert np.all(w >= 1e-6)

    def test_problem_aware_encoder_dim(
        self, full_trajectories: list[list[dict[str, Any]]]
    ) -> None:
        """With a ProblemAwareEncoder the sample width is window*6 + 9."""
        problems = [get_problem("zdt1"), get_problem("zdt2")]
        enc = ProblemAwareEncoder(_WINDOW, problems[0]).fit(full_trajectories, problems)
        X, _, _, _, _, _ = build_multihead_samples(full_trajectories, enc, _WINDOW)
        assert enc.dim == _WINDOW * len(STATE_FEATURES) + 9
        assert X.shape == (3 * (_N_TRANSITIONS - 1), enc.dim)

    def test_empty_input_returns_empty_arrays(self, encoder: StateEncoder) -> None:
        X, y_op, y_logpm, y_logexpl, w, traj_ids = build_multihead_samples(
            [], encoder, _WINDOW
        )
        assert X.shape == (0, encoder.dim)
        assert y_op.shape == (0,) and y_op.dtype.kind == "i"
        assert y_logpm.shape == y_logexpl.shape == w.shape == (0,)
        assert traj_ids.shape == (0,)

    def test_unknown_operator_raises(self, encoder: StateEncoder) -> None:
        """Unknown operator names are rejected, never silently mislabeled."""
        trajectory = _full_trajectory(0)
        trajectory[1]["action"]["mutation_operator"] = "simulated_binary"
        with pytest.raises(ValueError):
            build_multihead_samples([trajectory], encoder, _WINDOW)


def _learnable_multihead(
    n: int = 200, dim: int = 10, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic dataset with one learnable signal per head.

    The operator label correlates with feature 0, the log mutation
    probability with feature 1, and the log exploration strength with
    feature 2, so each head has an independent, linearly learnable target.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, dim))
    y_op = (X[:, 0] > 0.0).astype(np.int64)
    y_logpm = 0.5 * X[:, 1] - 0.1
    y_logexpl = -0.4 * X[:, 2] + 0.2
    return X, y_op, y_logpm, y_logexpl


class TestMultiHeadController:
    def test_determinism_same_seed(self) -> None:
        """Same seed + data -> identical predictions (all three heads)."""
        X, y_op, y_logpm, y_logexpl = _learnable_multihead()
        preds = []
        for _ in range(2):
            controller = MultiHeadController(input_dim=10, hidden_dims=(16,), seed=7)
            controller.fit(X, y_op, y_logpm, y_logexpl, epochs=30, batch_size=16)
            preds.append(controller.predict(X))
        for first, second in zip(preds[0], preds[1]):
            np.testing.assert_allclose(first, second, atol=1e-6, rtol=1e-6)

    def test_loss_decreases_and_operator_learned(self) -> None:
        """Total loss drops and the operator head learns its feature."""
        X, y_op, y_logpm, y_logexpl = _learnable_multihead()
        controller = MultiHeadController(
            input_dim=10, hidden_dims=(32, 32), seed=0, lr=5e-3
        )
        history = controller.fit(X, y_op, y_logpm, y_logexpl, epochs=150, batch_size=32)
        train_loss = history["train_loss"]
        assert len(train_loss) == 150
        assert history["val_loss"] == []
        assert train_loss[-1] < 0.2 * train_loss[0]
        probs, _, _ = controller.predict(X)
        accuracy = float(np.mean(np.argmax(probs, axis=1) == y_op))
        assert accuracy > 0.9

    def test_sample_weight_accepted(self) -> None:
        """Weighted training runs and reports one loss per epoch."""
        X, y_op, y_logpm, y_logexpl = _learnable_multihead(n=60)
        rng = np.random.default_rng(0)
        w = rng.uniform(0.1, 2.0, size=60)
        controller = MultiHeadController(input_dim=10, hidden_dims=(8,), seed=1)
        history = controller.fit(
            X, y_op, y_logpm, y_logexpl, sample_weight=w, epochs=15, batch_size=16
        )
        assert len(history["train_loss"]) == 15
        assert all(math.isfinite(v) for v in history["train_loss"])

    def test_val_loss_reported_when_validation_given(self) -> None:
        X, y_op, y_logpm, y_logexpl = _learnable_multihead()
        X_val, yop_val, ypm_val, yexpl_val = _learnable_multihead(n=40, seed=7)
        controller = MultiHeadController(input_dim=10, hidden_dims=(16,), seed=2, lr=5e-3)
        history = controller.fit(
            X, y_op, y_logpm, y_logexpl,
            epochs=50, batch_size=32,
            X_val=X_val, yop_val=yop_val, ypm_val=ypm_val, yexpl_val=yexpl_val,
        )
        assert len(history["val_loss"]) == 50
        assert history["val_loss"][-1] < history["val_loss"][0]

    def test_fit_validation(self) -> None:
        controller = MultiHeadController(input_dim=10, hidden_dims=(8,), seed=0)
        with pytest.raises(ValueError):
            controller.fit(
                np.zeros((0, 10)), np.zeros(0, dtype=int), np.zeros(0), np.zeros(0)
            )
        # Class indices must lie in {0, 1}.
        with pytest.raises(ValueError):
            controller.fit(
                np.zeros((4, 10)), np.array([0, 1, 2, 0]), np.zeros(4), np.zeros(4)
            )

    def test_predict_shapes(self) -> None:
        controller = MultiHeadController(input_dim=10, hidden_dims=(8,), seed=0)
        probs, log_pm, log_expl = controller.predict(np.zeros((5, 10)))
        assert probs.shape == (5, 2)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), rtol=1e-6)
        assert log_pm.shape == (5,) and log_pm.dtype == np.float64
        assert log_expl.shape == (5,) and log_expl.dtype == np.float64

    def _fit_on_exact_history(
        self,
        encoder: StateEncoder,
        history: list[dict[str, float]],
        operator_idx: int,
        log_pm_target: float,
        log_expl_target: float,
    ) -> MultiHeadController:
        """Train a controller to output constant targets at ``history``.

        Tiles the exact encoded history point so the network outputs the
        target values precisely where ``predict_action`` evaluates it
        (same technique as the Phase-1 controller tests).
        """
        X = np.tile(encoder.transform(history), (24, 1))
        controller = MultiHeadController(
            input_dim=encoder.dim, hidden_dims=(16,), seed=5, lr=5e-2
        )
        controller.fit(
            X,
            np.full(24, operator_idx, dtype=np.int64),
            np.full(24, log_pm_target),
            np.full(24, log_expl_target),
            epochs=300,
            batch_size=24,
        )
        return controller

    @pytest.fixture
    def history(self, full_trajectories: list[list[dict[str, Any]]]) -> list[dict[str, float]]:
        """One merged history window of the test encoder's window length."""
        return [merge_state_reward(t) for t in full_trajectories[0][:_WINDOW]]

    def test_predict_action_in_range(
        self, encoder: StateEncoder, history: list[dict[str, float]]
    ) -> None:
        """In-range predictions pass through; keys and operator are right."""
        controller = self._fit_on_exact_history(
            encoder, history, operator_idx=0,
            log_pm_target=math.log(0.05), log_expl_target=math.log(10.0),
        )
        action = controller.predict_action(history, encoder, pm_min=0.01, pm_max=0.2)
        assert set(action) == {
            "mutation_operator",
            "mutation_probability",
            "exploration_strength",
        }
        assert action["mutation_operator"] == "polynomial"
        assert action["mutation_probability"] == pytest.approx(0.05, abs=0.02)
        assert action["exploration_strength"] == pytest.approx(10.0, rel=0.2)

    def test_predict_action_clips_to_bounds(
        self, encoder: StateEncoder, history: list[dict[str, float]]
    ) -> None:
        """Out-of-range heads are clipped: pm to [pm_min, pm_max], eta_m to [2, 50]."""
        controller = self._fit_on_exact_history(
            encoder, history, operator_idx=0,
            log_pm_target=math.log(0.9), log_expl_target=math.log(100.0),
        )
        action = controller.predict_action(history, encoder, pm_min=0.01, pm_max=0.2)
        assert action["mutation_operator"] == "polynomial"
        assert action["mutation_probability"] == pytest.approx(0.2)
        assert action["exploration_strength"] == pytest.approx(
            POLYNOMIAL_EXPLORATION_RANGE[1]
        )

    def test_predict_action_gaussian_range(
        self, encoder: StateEncoder, history: list[dict[str, float]]
    ) -> None:
        """The gaussian branch clips exploration to [0.02, 0.3] and pm to pm_min."""
        controller = self._fit_on_exact_history(
            encoder, history, operator_idx=1,
            log_pm_target=math.log(0.001), log_expl_target=math.log(0.5),
        )
        action = controller.predict_action(history, encoder, pm_min=0.01, pm_max=0.2)
        assert action["mutation_operator"] == "gaussian"
        assert action["mutation_probability"] == pytest.approx(0.01)
        assert action["exploration_strength"] == pytest.approx(
            GAUSSIAN_EXPLORATION_RANGE[1]
        )
        # The lower gaussian bound applies as well.
        controller_low = self._fit_on_exact_history(
            encoder, history, operator_idx=1,
            log_pm_target=math.log(0.05), log_expl_target=math.log(0.001),
        )
        action_low = controller_low.predict_action(history, encoder, 0.01, 0.2)
        assert action_low["exploration_strength"] == pytest.approx(
            GAUSSIAN_EXPLORATION_RANGE[0]
        )

    def test_predict_action_rejects_invalid_bounds(
        self, encoder: StateEncoder, history: list[dict[str, float]]
    ) -> None:
        controller = MultiHeadController(input_dim=encoder.dim, seed=0)
        with pytest.raises(ValueError):
            controller.predict_action(history, encoder, pm_min=0.0, pm_max=0.5)
        with pytest.raises(ValueError):
            controller.predict_action(history, encoder, pm_min=0.5, pm_max=0.1)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        X, y_op, y_logpm, y_logexpl = _learnable_multihead(n=60)
        controller = MultiHeadController(
            input_dim=10, hidden_dims=(16, 8), seed=11, lr=2e-3, name="mlp2_test"
        )
        controller.fit(X, y_op, y_logpm, y_logexpl, epochs=5, batch_size=8)
        path = tmp_path / "multihead.pt"
        controller.save(path)
        loaded = MultiHeadController.load(path)
        assert loaded.name == "mlp2_test"
        assert loaded.input_dim == controller.input_dim
        assert loaded.hidden_dims == controller.hidden_dims
        for original, restored in zip(controller.predict(X), loaded.predict(X)):
            np.testing.assert_array_equal(original, restored)
