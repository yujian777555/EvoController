from __future__ import annotations

"""Tests for the Phase 1.5 problem-aware features.

Covers ``Problem.describe`` landscape statistics, the canonical
9-dimensional problem feature vector, and the ``ProblemAwareEncoder``
(history window + problem block). Also smoke-tests that the frozen
Phase 1 ``StateEncoder`` behavior is unchanged.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, Problem
from controller.dataset import merge_state_reward
from controller.problem_features import PROBLEM_FEATURE_NAMES, problem_feature_vector
from controller.state_encoder import STATE_FEATURES, ProblemAwareEncoder, StateEncoder

ALL_PROBLEM_CLASSES = [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6]
FIT_PROBLEMS = [ZDT1(), ZDT4(), ZDT6()]

N_TRAJECTORIES = 3
N_GENERATIONS = 12


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
    """Deterministic synthetic trajectory with nonzero-variance features."""
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


@pytest.fixture
def trajectories() -> list[list[dict[str, Any]]]:
    """Three synthetic 12-generation trajectories."""
    return [_synthetic_trajectory(j) for j in range(N_TRAJECTORIES)]


@pytest.fixture
def encoder(trajectories: list[list[dict[str, Any]]]) -> ProblemAwareEncoder:
    """Fitted window-4 encoder for zdt1 with 3-problem statistics."""
    return ProblemAwareEncoder(window=4, problem=ZDT1()).fit(
        trajectories, FIT_PROBLEMS
    )


class TestDescribe:
    @pytest.mark.parametrize("cls", ALL_PROBLEM_CLASSES)
    def test_deterministic_same_seed(self, cls: type[Problem]) -> None:
        assert cls().describe(n_samples=128, seed=3) == cls().describe(
            n_samples=128, seed=3
        )

    def test_seed_changes_samples(self) -> None:
        assert ZDT1().describe(seed=0) != ZDT1().describe(seed=1)

    def test_keys_and_types(self) -> None:
        desc = ZDT1().describe()
        assert set(desc.keys()) == {
            "n_vars",
            "n_objs",
            "bounds_width_mean",
            "f1_mean",
            "f1_std",
            "f2_mean",
            "f2_std",
            "f_corr",
            "ideal_est",
            "nadir_est",
        }
        assert isinstance(desc["n_vars"], int)
        assert isinstance(desc["n_objs"], int)
        for key in (
            "bounds_width_mean",
            "f1_mean",
            "f1_std",
            "f2_mean",
            "f2_std",
            "f_corr",
        ):
            assert isinstance(desc[key], float), key
            assert math.isfinite(desc[key]), key
        for key in ("ideal_est", "nadir_est"):
            assert isinstance(desc[key], list)
            assert len(desc[key]) == 2
            assert all(isinstance(v, float) for v in desc[key])

    def test_zdt1_vs_zdt4_dimensions_and_bounds(self) -> None:
        zdt1 = ZDT1().describe()
        zdt4 = ZDT4().describe()
        assert zdt1["n_vars"] == 30
        assert zdt4["n_vars"] == 10
        # mean(upper - lower): zdt1 is 10 * 1 / 10 = 1 for [0, 1]^30;
        # zdt4 is (1 * 1 + 9 * 10) / 10 = 9.1 (x1 in [0, 1], xi in [-5, 5]).
        assert zdt1["bounds_width_mean"] == pytest.approx(1.0)
        assert zdt4["bounds_width_mean"] == pytest.approx(9.1)

    @pytest.mark.parametrize("cls", ALL_PROBLEM_CLASSES)
    def test_f_corr_in_valid_range(self, cls: type[Problem]) -> None:
        f_corr = cls().describe()["f_corr"]
        assert -1.0 <= f_corr <= 1.0

    @pytest.mark.parametrize("cls", ALL_PROBLEM_CLASSES)
    def test_ideal_not_above_nadir(self, cls: type[Problem]) -> None:
        desc = cls().describe()
        assert desc["ideal_est"][0] <= desc["nadir_est"][0]
        assert desc["ideal_est"][1] <= desc["nadir_est"][1]

    def test_invalid_n_samples_raises(self) -> None:
        with pytest.raises(ValueError):
            ZDT1().describe(n_samples=0)


class TestProblemFeatureVector:
    def test_fixed_names_and_order(self) -> None:
        # Contract: downstream modules rely on this exact order.
        assert PROBLEM_FEATURE_NAMES == [
            "n_vars_log",
            "bounds_width_mean",
            "f1_std",
            "f2_std",
            "f_corr",
            "ideal_est_0",
            "ideal_est_1",
            "nadir_est_0",
            "nadir_est_1",
        ]
        # n_objs is constant (2) across benchmarks -> excluded as
        # zero-variance.
        assert "n_objs" not in PROBLEM_FEATURE_NAMES

    def test_shape_and_dtype(self) -> None:
        vec = problem_feature_vector(ZDT1())
        assert vec.shape == (9,)
        assert vec.dtype == np.float64

    def test_values_match_describe(self) -> None:
        problem = ZDT1()
        desc = problem.describe()
        vec = problem_feature_vector(problem)
        assert vec[0] == pytest.approx(math.log10(30))
        assert vec[1] == pytest.approx(desc["bounds_width_mean"])
        assert vec[2] == pytest.approx(desc["f1_std"])
        assert vec[3] == pytest.approx(desc["f2_std"])
        assert vec[4] == pytest.approx(desc["f_corr"])
        assert vec[5] == pytest.approx(desc["ideal_est"][0])
        assert vec[6] == pytest.approx(desc["ideal_est"][1])
        assert vec[7] == pytest.approx(desc["nadir_est"][0])
        assert vec[8] == pytest.approx(desc["nadir_est"][1])

    def test_differs_across_problems(self) -> None:
        vec1 = problem_feature_vector(ZDT1())
        vec4 = problem_feature_vector(ZDT4())
        assert not np.allclose(vec1, vec4)
        # n_vars_log and bounds_width_mean must differ by construction.
        assert vec1[0] != pytest.approx(vec4[0])
        assert vec1[1] != pytest.approx(vec4[1])

    def test_deterministic(self) -> None:
        np.testing.assert_array_equal(
            problem_feature_vector(ZDT3()), problem_feature_vector(ZDT3())
        )


class TestProblemAwareEncoder:
    def test_dim(self) -> None:
        problem = ZDT1()
        assert ProblemAwareEncoder(window=4, problem=problem).dim == 4 * 6 + 9
        assert ProblemAwareEncoder(window=1, problem=problem).dim == 6 + 9

    def test_window_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ProblemAwareEncoder(window=0, problem=ZDT1())

    def test_fit_requires_transitions(self) -> None:
        with pytest.raises(ValueError):
            ProblemAwareEncoder(window=2, problem=ZDT1()).fit([[], []], FIT_PROBLEMS)

    def test_fit_requires_problems(
        self, trajectories: list[list[dict[str, Any]]]
    ) -> None:
        with pytest.raises(ValueError):
            ProblemAwareEncoder(window=2, problem=ZDT1()).fit(trajectories, [])

    def test_transform_requires_fit(
        self, trajectories: list[list[dict[str, Any]]]
    ) -> None:
        merged = merge_state_reward(trajectories[0][0])
        with pytest.raises(RuntimeError):
            ProblemAwareEncoder(window=2, problem=ZDT1()).transform([merged])

    def test_transform_shape(self, encoder: ProblemAwareEncoder,
                             trajectories: list[list[dict[str, Any]]]) -> None:
        history = [merge_state_reward(t) for t in trajectories[0][:3]]
        out = encoder.transform(history)
        assert out.shape == (encoder.dim,)
        assert out.dtype == np.float64

    def test_zero_padding_and_state_block_matches_state_encoder(
        self, trajectories: list[list[dict[str, Any]]]
    ) -> None:
        window = 3
        pa_enc = ProblemAwareEncoder(window=window, problem=ZDT1()).fit(
            trajectories, FIT_PROBLEMS
        )
        ref_enc = StateEncoder(window=window).fit(trajectories)
        merged = [merge_state_reward(t) for t in trajectories[0][:2]]
        out = pa_enc.transform(merged)
        # The first (window - len(history)) state blocks are exactly zero.
        np.testing.assert_array_equal(out[:6], np.zeros(6))
        # The state block is identical to a StateEncoder fitted on the
        # same trajectories (same merge semantics).
        np.testing.assert_allclose(
            out[: window * 6], ref_enc.transform(merged), rtol=1e-12
        )

    def test_uses_only_last_window_entries(
        self, encoder: ProblemAwareEncoder,
        trajectories: list[list[dict[str, Any]]],
    ) -> None:
        merged = [merge_state_reward(t) for t in trajectories[0][:6]]
        np.testing.assert_array_equal(
            encoder.transform(merged), encoder.transform(merged[-4:])
        )

    def test_problem_block_constant_across_generations(
        self, encoder: ProblemAwareEncoder,
        trajectories: list[list[dict[str, Any]]],
    ) -> None:
        merged = [merge_state_reward(t) for t in trajectories[0]]
        early = encoder.transform(merged[:2])
        late = encoder.transform(merged[-4:])
        np.testing.assert_array_equal(early[-9:], late[-9:])

    def test_problem_block_differs_across_problems(
        self, trajectories: list[list[dict[str, Any]]]
    ) -> None:
        enc1 = ProblemAwareEncoder(window=2, problem=ZDT1()).fit(
            trajectories, FIT_PROBLEMS
        )
        enc4 = ProblemAwareEncoder(window=2, problem=ZDT4()).fit(
            trajectories, FIT_PROBLEMS
        )
        history = [merge_state_reward(t) for t in trajectories[1][:3]]
        out1 = enc1.transform(history)
        out4 = enc4.transform(history)
        # Same trajectories and statistics -> identical state block...
        np.testing.assert_array_equal(out1[:12], out4[:12])
        # ...but a different problem block.
        assert not np.allclose(out1[-9:], out4[-9:])

    def test_state_zscore_sanity(
        self, trajectories: list[list[dict[str, Any]]]
    ) -> None:
        enc = ProblemAwareEncoder(window=1, problem=ZDT1()).fit(
            trajectories, FIT_PROBLEMS
        )
        histories = [
            [merge_state_reward(t)]
            for trajectory in trajectories
            for t in trajectory
        ]
        z = enc.transform_batch(histories)
        assert z.shape == (N_TRAJECTORIES * N_GENERATIONS, 6 + 9)
        state_z = z[:, :6]
        np.testing.assert_allclose(state_z.mean(axis=0), 0.0, atol=1e-8)
        # All synthetic features have nonzero variance -> std exactly 1.
        np.testing.assert_allclose(state_z.std(axis=0), 1.0, rtol=1e-6)

    def test_problem_block_zscore_sanity(
        self, encoder: ProblemAwareEncoder,
        trajectories: list[list[dict[str, Any]]],
    ) -> None:
        vectors = np.asarray(
            [problem_feature_vector(p) for p in FIT_PROBLEMS], dtype=float
        )
        std = vectors.std(axis=0)
        std = np.where(std > 0.0, std, 1.0)
        expected = (problem_feature_vector(ZDT1()) - vectors.mean(axis=0)) / std
        history = [merge_state_reward(t) for t in trajectories[2][:4]]
        np.testing.assert_allclose(
            encoder.transform(history)[-9:], expected, rtol=1e-12
        )

    def test_transform_batch_empty(self, encoder: ProblemAwareEncoder) -> None:
        out = encoder.transform_batch([])
        assert out.shape == (0, encoder.dim)

    def test_save_load_roundtrip(
        self, encoder: ProblemAwareEncoder,
        trajectories: list[list[dict[str, Any]]],
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "pa_encoder.json"
        encoder.save(path)
        loaded = ProblemAwareEncoder.load(path)
        assert loaded.window == encoder.window
        assert loaded.dim == encoder.dim
        assert loaded.problem_name == "zdt1"
        history = [merge_state_reward(t) for t in trajectories[1][:3]]
        np.testing.assert_array_equal(
            encoder.transform(history), loaded.transform(history)
        )

    def test_save_requires_fit(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            ProblemAwareEncoder(window=2, problem=ZDT1()).save(
                tmp_path / "pa_encoder.json"
            )


class TestStateEncoderUnchanged:
    """Smoke tests that the frozen Phase 1 StateEncoder still behaves."""

    def test_fit_transform_roundtrip(
        self, trajectories: list[list[dict[str, Any]]]
    ) -> None:
        enc = StateEncoder(window=2).fit(trajectories)
        history = [merge_state_reward(t) for t in trajectories[0][:3]]
        out = enc.transform(history)
        assert out.shape == (2 * len(STATE_FEATURES),)
        # Uses only the last `window` entries.
        np.testing.assert_array_equal(out, enc.transform(history[-2:]))

    def test_save_load_roundtrip(
        self, trajectories: list[list[dict[str, Any]]], tmp_path: Path
    ) -> None:
        enc = StateEncoder(window=3).fit(trajectories)
        path = tmp_path / "encoder.json"
        enc.save(path)
        loaded = StateEncoder.load(path)
        history = [merge_state_reward(t) for t in trajectories[2][:2]]
        np.testing.assert_array_equal(
            enc.transform(history), loaded.transform(history)
        )
