from __future__ import annotations

"""Tests for the Phase 1 controller package.

Synthetic trajectories are hand-made with a known trend: mutation
probability varies deterministically per generation and the reward of a
transition correlates with the mutation probability chosen at that
transition, so the dataset contains a learnable pattern by construction.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller import (
    STATE_FEATURES,
    ConstantController,
    MLPController,
    StateEncoder,
    build_supervised_samples,
    load_trajectories,
    merge_state_reward,
    train_val_split,
)

N_TRAJECTORIES = 3
N_GENERATIONS = 20


def _transition(
    generation: int,
    hv: float,
    igd: float,
    diversity: float,
    delta_hv: float,
    delta_igd: float,
    pm: float,
) -> dict[str, Any]:
    """Build one transition dict in the EvolutionRecorder schema."""
    return {
        "generation": int(generation),
        "state": {
            "generation": int(generation),
            "hv": float(hv),
            "igd": float(igd),
            "diversity": float(diversity),
        },
        "action": {
            "mutation_operator": "polynomial_mutation",
            "mutation_probability": float(pm),
        },
        "reward": {
            "delta_hv": float(delta_hv),
            "delta_igd": float(delta_igd),
        },
    }


def _synthetic_trajectory(
    traj_index: int, n_generations: int = N_GENERATIONS
) -> list[dict[str, Any]]:
    """Build a deterministic trajectory whose rewards correlate with pm.

    ``pm`` cycles through ``[0.05, 0.45]``; transition ``t > 0`` gets a
    reward increasing in the pm chosen at ``t``, so "higher pm -> higher
    reward" is a pattern a controller can in principle learn.
    """
    transitions = []
    for t in range(n_generations):
        pm = 0.05 + 0.4 * (((t * 3 + traj_index * 2) % 7) / 6.0)
        reward = 0.02 + 0.1 * (pm - 0.25) if t > 0 else 0.0
        transitions.append(
            _transition(
                generation=t,
                hv=0.4 + 0.02 * t + 0.05 * traj_index,
                igd=max(0.5 - 0.02 * t, 0.01) + 0.01 * traj_index,
                diversity=0.3 + 0.02 * ((t + traj_index) % 5),
                delta_hv=reward,
                delta_igd=0.5 * reward,
                pm=pm,
            )
        )
    return transitions


def _merged_matrix(trajectories: list[list[dict[str, Any]]]) -> np.ndarray:
    """Stack the merged STATE_FEATURES vectors of all transitions."""
    return np.asarray(
        [
            [merged[key] for key in STATE_FEATURES]
            for trajectory in trajectories
            for transition in trajectory
            for merged in [merge_state_reward(transition)]
        ],
        dtype=float,
    )


@pytest.fixture
def trajectories() -> list[list[dict[str, Any]]]:
    """Three synthetic 20-generation trajectories."""
    return [_synthetic_trajectory(j) for j in range(N_TRAJECTORIES)]


@pytest.fixture
def encoder(trajectories: list[list[dict[str, Any]]]) -> StateEncoder:
    """Fitted encoder with window 4."""
    return StateEncoder(window=4).fit(trajectories)


class TestStateEncoder:
    def test_dim(self) -> None:
        assert StateEncoder(window=4).dim == 4 * len(STATE_FEATURES)
        assert StateEncoder(window=1).dim == len(STATE_FEATURES)

    def test_window_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            StateEncoder(window=0)

    def test_fit_requires_transitions(self) -> None:
        with pytest.raises(ValueError):
            StateEncoder(window=2).fit([[], []])

    def test_transform_requires_fit(
        self, trajectories: list[list[dict[str, Any]]]
    ) -> None:
        merged = merge_state_reward(trajectories[0][0])
        with pytest.raises(RuntimeError):
            StateEncoder(window=2).transform([merged])

    def test_zero_padding_at_front(
        self, trajectories: list[list[dict[str, Any]]]
    ) -> None:
        enc = StateEncoder(window=3).fit(trajectories)
        merged = [merge_state_reward(t) for t in trajectories[0][:2]]
        out = enc.transform(merged)
        assert out.shape == (18,)
        # The first (window - len(history)) feature blocks are exactly zero.
        np.testing.assert_array_equal(out[:6], np.zeros(6))
        # Remaining blocks are the independently computed z-scores, oldest
        # first, most recent last.
        stats = _merged_matrix(trajectories)
        mean = stats.mean(axis=0)
        std = stats.std(axis=0)
        std = np.where(std > 0.0, std, 1.0)
        expected = [
            (np.asarray([m[k] for k in STATE_FEATURES]) - mean) / std
            for m in merged
        ]
        np.testing.assert_allclose(out[6:12], expected[0], rtol=1e-12)
        np.testing.assert_allclose(out[12:18], expected[1], rtol=1e-12)

    def test_uses_only_last_window_entries(
        self, trajectories: list[list[dict[str, Any]]]
    ) -> None:
        enc = StateEncoder(window=3).fit(trajectories)
        merged = [merge_state_reward(t) for t in trajectories[0][:6]]
        np.testing.assert_array_equal(
            enc.transform(merged), enc.transform(merged[-3:])
        )

    def test_zscore_statistics(
        self, trajectories: list[list[dict[str, Any]]]
    ) -> None:
        enc = StateEncoder(window=1).fit(trajectories)
        merged = [
            [merge_state_reward(t)] for trajectory in trajectories for t in trajectory
        ]
        z = enc.transform_batch(merged)
        assert z.shape == (N_TRAJECTORIES * N_GENERATIONS, len(STATE_FEATURES))
        np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-8)
        # All synthetic features have nonzero variance, so std is exactly 1.
        np.testing.assert_allclose(z.std(axis=0), 1.0, rtol=1e-6)

    def test_transform_batch_empty(self, encoder: StateEncoder) -> None:
        out = encoder.transform_batch([])
        assert out.shape == (0, encoder.dim)

    def test_save_load_roundtrip(
        self, trajectories: list[list[dict[str, Any]]], tmp_path: Path
    ) -> None:
        enc = StateEncoder(window=4).fit(trajectories)
        path = tmp_path / "encoder.json"
        enc.save(path)
        loaded = StateEncoder.load(path)
        assert loaded.window == enc.window
        assert loaded.dim == enc.dim
        history = [merge_state_reward(t) for t in trajectories[1][:3]]
        np.testing.assert_array_equal(
            enc.transform(history), loaded.transform(history)
        )

    def test_save_requires_fit(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            StateEncoder(window=2).save(tmp_path / "encoder.json")


class TestLoadTrajectories:
    def test_loads_sorts_and_skips_index(self, tmp_path: Path) -> None:
        traj_a = _synthetic_trajectory(0, n_generations=5)
        traj_b = _synthetic_trajectory(1, n_generations=5)
        for name, traj in [("traj_b.json", traj_b), ("traj_a.json", traj_a)]:
            payload = {
                "config": {"problem": "synthetic"},
                "seed": 0,
                "runtime_sec": 0.1,
                "schema_version": 1,
                # Store transitions out of order to prove sorting happens.
                "transitions": list(reversed(traj)),
                "final": dict(traj[-1]["state"]),
            }
            (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
        # index.json has no 'transitions'; it must be skipped, not parsed.
        (tmp_path / "index.json").write_text(
            json.dumps({"files": ["traj_a.json", "traj_b.json"]}), encoding="utf-8"
        )
        loaded = load_trajectories(tmp_path)
        assert len(loaded) == 2
        for traj in loaded:
            assert len(traj) == 5
            assert [t["generation"] for t in traj] == [0, 1, 2, 3, 4]

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            load_trajectories(tmp_path / "does_not_exist")


class TestMergeStateReward:
    def test_returns_exactly_the_state_features(self) -> None:
        transition = _transition(
            generation=3, hv=0.5, igd=0.2, diversity=0.4,
            delta_hv=0.01, delta_igd=0.005, pm=0.3,
        )
        merged = merge_state_reward(transition)
        assert list(merged.keys()) == STATE_FEATURES
        assert merged["hv"] == 0.5
        assert merged["generation"] == 3.0
        assert merged["delta_igd"] == 0.005


class TestBuildSupervisedSamples:
    def test_sample_count_and_shapes(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        X, y, w, traj_ids = build_supervised_samples(trajectories, encoder, window=4)
        # Each trajectory contributes len - 1 samples (t starts at 1).
        n_expected = N_TRAJECTORIES * (N_GENERATIONS - 1)
        assert X.shape == (n_expected, 4 * len(STATE_FEATURES))
        assert y.shape == (n_expected,)
        assert w.shape == (n_expected,)
        assert traj_ids.shape == (n_expected,)
        for j in range(N_TRAJECTORIES):
            assert int(np.sum(traj_ids == j)) == N_GENERATIONS - 1

    def test_y_is_log_of_actual_pm(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        _, y, _, _ = build_supervised_samples(trajectories, encoder, window=4)
        # Samples are ordered trajectory-major, then by transition index t.
        expected = [
            math.log(trajectories[j][t]["action"]["mutation_probability"])
            for j in range(N_TRAJECTORIES)
            for t in range(1, N_GENERATIONS)
        ]
        np.testing.assert_allclose(y, np.asarray(expected), rtol=1e-12)

    def test_history_alignment(
        self, trajectories: list[list[dict[str, Any]]]
    ) -> None:
        window = 2
        enc = StateEncoder(window=window).fit(trajectories)
        X, _, _, _ = build_supervised_samples(trajectories, enc, window=window)
        # Trajectory 0, t = 2 is sample row 1; its history is transitions [0, 2).
        merged = [merge_state_reward(t) for t in trajectories[0][:2]]
        np.testing.assert_allclose(X[1], enc.transform(merged), rtol=1e-12)
        # t = 1 sees only transition 0 (plus zero-padding inside transform).
        np.testing.assert_allclose(
            X[0], enc.transform(merged[:1]), rtol=1e-12
        )

    def test_weights_exact_on_tiny_trajectory(self) -> None:
        # Rewards r = [0, 0.5, -0.5] -> mean 0 -> w = [0.5 + 1e-6, 1e-6].
        traj = [
            _transition(t, hv=0.5, igd=0.3, diversity=0.4,
                        delta_hv=dh, delta_igd=0.0, pm=0.2)
            for t, dh in enumerate([0.0, 0.5, -0.5])
        ]
        enc = StateEncoder(window=2).fit([traj])
        _, _, w, _ = build_supervised_samples([traj], enc, window=2)
        np.testing.assert_allclose(w, [0.5 + 1e-6, 1e-6], rtol=1e-9)

    def test_weights_positive_and_reward_ordered(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        _, _, w, _ = build_supervised_samples(trajectories, encoder, window=4)
        assert np.all(w >= 1e-6)
        # Below-average rewards get exactly the epsilon floor.
        assert float(np.min(w)) == pytest.approx(1e-6)

    def test_empty_input_returns_empty_arrays(self, encoder: StateEncoder) -> None:
        X, y, w, traj_ids = build_supervised_samples([], encoder, window=4)
        assert X.shape == (0, 4 * len(STATE_FEATURES))
        assert y.shape == (0,)
        assert w.shape == (0,)
        assert traj_ids.shape == (0,)


class TestTrainValSplit:
    def test_no_leakage_and_full_coverage(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        X, y, w, traj_ids = build_supervised_samples(trajectories, encoder, window=4)
        split = train_val_split(X, y, w, traj_ids, val_fraction=1.0 / 3.0, seed=0)
        train_ids = set(split["traj_ids_train"].tolist())
        val_ids = set(split["traj_ids_val"].tolist())
        assert train_ids.isdisjoint(val_ids)
        assert train_ids | val_ids == {0, 1, 2}
        assert len(split["y_train"]) + len(split["y_val"]) == len(y)
        assert len(split["w_val"]) == len(split["y_val"])

    def test_deterministic_given_seed(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        X, y, w, traj_ids = build_supervised_samples(trajectories, encoder, window=4)
        split_a = train_val_split(X, y, w, traj_ids, 0.34, seed=7)
        split_b = train_val_split(X, y, w, traj_ids, 0.34, seed=7)
        np.testing.assert_array_equal(
            split_a["traj_ids_val"], split_b["traj_ids_val"]
        )
        np.testing.assert_array_equal(split_a["X_val"], split_b["X_val"])

    def test_val_fraction_zero_gives_empty_val(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        X, y, w, traj_ids = build_supervised_samples(trajectories, encoder, window=4)
        split = train_val_split(X, y, w, traj_ids, 0.0, seed=0)
        assert split["X_val"].shape[0] == 0
        assert split["X_train"].shape[0] == X.shape[0]

    def test_invalid_fraction_raises(self) -> None:
        with pytest.raises(ValueError):
            train_val_split(
                np.zeros((2, 6)), np.zeros(2), np.ones(2),
                np.array([0, 0]), 1.0, seed=0,
            )


def _learnable_regression(n: int = 120, dim: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Linear target pattern the MLP can learn: y = f(two features)."""
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, size=(n, dim))
    y = 0.5 * X[:, 0] - 0.25 * X[:, 3] + 0.1
    return X, y


class TestMLPController:
    def test_name_reflects_window(self) -> None:
        assert MLPController(input_dim=24).name == "mlp_w4"
        assert MLPController(input_dim=6).name == "mlp_w1"

    def test_determinism_same_seed(self) -> None:
        X, y = _learnable_regression()
        preds = []
        for _ in range(2):
            controller = MLPController(input_dim=12, hidden_dims=(16,), seed=7)
            controller.fit(X, y, epochs=30, batch_size=16)
            preds.append(controller.predict(X))
        np.testing.assert_allclose(preds[0], preds[1], atol=1e-6, rtol=1e-6)

    def test_loss_decreases_on_learnable_pattern(self) -> None:
        X, y = _learnable_regression()
        controller = MLPController(input_dim=12, hidden_dims=(32, 32), seed=0, lr=5e-3)
        history = controller.fit(X, y, epochs=200, batch_size=32)
        train_loss = history["train_loss"]
        assert len(train_loss) == 200
        assert history["val_loss"] == []
        assert train_loss[-1] < 0.05 * train_loss[0]

    def test_val_loss_reported_when_validation_given(self) -> None:
        X, y = _learnable_regression()
        X_val, y_val = _learnable_regression(n=30)
        controller = MLPController(input_dim=12, hidden_dims=(16,), seed=1, lr=5e-3)
        history = controller.fit(
            X, y, epochs=100, batch_size=32, X_val=X_val, y_val=y_val
        )
        assert len(history["val_loss"]) == 100
        assert history["val_loss"][-1] < history["val_loss"][0]

    def test_overfits_small_dataset(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(0.0, 1.0, size=(8, 12))
        y = 0.8 * X[:, 0] - 0.4 * X[:, 5]
        controller = MLPController(input_dim=12, hidden_dims=(16, 16), seed=3, lr=1e-2)
        history = controller.fit(X, y, epochs=500, batch_size=8)
        assert history["train_loss"][-1] < 1e-3
        assert history["train_loss"][-1] < 0.01 * history["train_loss"][0]

    def test_predict_shape(self) -> None:
        controller = MLPController(input_dim=12, hidden_dims=(8,), seed=0)
        out = controller.predict(np.zeros((5, 12)))
        assert out.shape == (5,)
        assert out.dtype == np.float64

    def test_predict_action_respects_pm_bounds(
        self,
        trajectories: list[list[dict[str, Any]]],
    ) -> None:
        window = 2
        enc = StateEncoder(window=window).fit(trajectories)
        history = [merge_state_reward(t) for t in trajectories[0][:window]]
        # Train on the exact encoded history point so the network outputs
        # ~log(0.3) precisely where predict_action will evaluate it.
        X = np.tile(enc.transform(history), (24, 1))
        # Train towards a constant log target near log(0.3).
        controller = MLPController(input_dim=enc.dim, hidden_dims=(16,), seed=5, lr=5e-2)
        controller.fit(X, np.full(24, math.log(0.3)), epochs=300, batch_size=24)
        # Inside the bounds: close to 0.3.
        pm = controller.predict_action(history, enc, pm_min=0.01, pm_max=0.9)
        assert pm == pytest.approx(0.3, abs=0.02)
        # Beyond the bounds: clipped exactly.
        assert controller.predict_action(history, enc, 0.01, 0.2) == pytest.approx(0.2)
        assert controller.predict_action(history, enc, 0.5, 0.9) == pytest.approx(0.5)

    def test_predict_action_rejects_invalid_bounds(
        self, encoder: StateEncoder
    ) -> None:
        controller = MLPController(input_dim=encoder.dim, seed=0)
        history = [{"hv": 0.5, "igd": 0.3, "diversity": 0.4,
                    "delta_hv": 0.0, "delta_igd": 0.0, "generation": 0.0}]
        with pytest.raises(ValueError):
            controller.predict_action(history, encoder, pm_min=0.0, pm_max=0.5)
        with pytest.raises(ValueError):
            controller.predict_action(history, encoder, pm_min=0.5, pm_max=0.1)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        X, y = _learnable_regression(n=40)
        controller = MLPController(input_dim=12, hidden_dims=(16, 8), seed=11, lr=2e-3)
        controller.fit(X, y, epochs=5, batch_size=8)
        path = tmp_path / "mlp.pt"
        controller.save(path)
        loaded = MLPController.load(path)
        assert loaded.name == controller.name
        assert loaded.input_dim == controller.input_dim
        assert loaded.hidden_dims == controller.hidden_dims
        np.testing.assert_array_equal(controller.predict(X), loaded.predict(X))


class TestConstantController:
    def test_weighted_mean(self) -> None:
        controller = ConstantController().fit(
            np.array([0.0, 1.0, 2.0]), sample_weight=np.array([1.0, 1.0, 2.0])
        )
        # Weighted mean: (0*1 + 1*1 + 2*2) / 4 = 1.25.
        np.testing.assert_allclose(
            controller.predict(np.zeros((5, 3))), np.full(5, 1.25)
        )

    def test_unweighted_mean(self) -> None:
        controller = ConstantController().fit(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(
            controller.predict(np.zeros((2, 6))), np.full(2, 2.0)
        )

    def test_same_action_regardless_of_history(
        self, trajectories: list[list[dict[str, Any]]], encoder: StateEncoder
    ) -> None:
        y = np.array([math.log(0.2), math.log(0.4)])
        controller = ConstantController().fit(y, sample_weight=np.array([1.0, 3.0]))
        history_a = [merge_state_reward(t) for t in trajectories[0][:3]]
        history_b = [merge_state_reward(t) for t in trajectories[2][:2]]
        pm_a = controller.predict_action(history_a, encoder, 0.01, 0.9)
        pm_b = controller.predict_action(history_b, encoder, 0.01, 0.9)
        expected_constant = (math.log(0.2) + 3.0 * math.log(0.4)) / 4.0
        assert pm_a == pytest.approx(math.exp(expected_constant))
        assert pm_a == pm_b
        # Bounds clip the constant like any other prediction.
        assert controller.predict_action(history_a, encoder, 0.01, 0.3) == pytest.approx(0.3)
        assert controller.predict_action(history_a, encoder, 0.5, 0.9) == pytest.approx(0.5)

    def test_predict_before_fit_raises(self) -> None:
        controller = ConstantController()
        with pytest.raises(RuntimeError):
            controller.predict(np.zeros((2, 6)))
        with pytest.raises(RuntimeError):
            controller.predict_action([], StateEncoder(1), 0.01, 0.9)

    def test_fit_validation(self) -> None:
        with pytest.raises(ValueError):
            ConstantController().fit(np.array([]))
        with pytest.raises(ValueError):
            ConstantController().fit(np.array([1.0]), sample_weight=np.array([0.0]))
        with pytest.raises(ValueError):
            ConstantController().fit(
                np.array([1.0, 2.0]), sample_weight=np.array([1.0])
            )

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        controller = ConstantController().fit(
            np.array([0.1, 0.3]), sample_weight=np.array([2.0, 1.0])
        )
        path = tmp_path / "constant.json"
        controller.save(path)
        loaded = ConstantController.load(path)
        assert loaded.name == "constant"
        np.testing.assert_array_equal(
            controller.predict(np.zeros((4, 2))), loaded.predict(np.zeros((4, 2)))
        )
