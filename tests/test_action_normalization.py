from __future__ import annotations

"""Tests for the Phase-1.75 normalized mutation action scale (Task 1).

Covers the strict conversion functions of
:mod:`controller.action_normalization`, the metadata-preserving
:func:`controller.dataset.load_trajectory_records` loader, the
``mutation_target="multiplier"`` path of
:func:`controller.dataset.build_multihead_samples`, and the multiplier mode
of :class:`controller.multihead_controller.MultiHeadController`. All data
is synthetic; controller math tests drive the network to exact constant
head outputs by zeroing the head weights, so the expected values are known
precisely rather than approximately learned.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from controller.action_normalization import (
    log_mutation_multiplier,
    mutation_multiplier,
    mutation_probability,
    pm_from_log_multiplier,
)
from controller.dataset import (
    build_multihead_samples,
    load_trajectory_records,
    merge_state_reward,
)
from controller.multihead_controller import MultiHeadController
from controller.state_encoder import StateEncoder

_WINDOW = 2
_N_TRANSITIONS = 6


class TestConversionFunctions:
    """Round-trip and strict-validation tests for the conversions."""

    def test_mutation_multiplier_round_trip(self) -> None:
        pm = 0.08
        n_vars = 30
        k = mutation_multiplier(pm, n_vars)
        assert k == pytest.approx(2.4)
        assert mutation_probability(k, n_vars) == pytest.approx(pm)

    def test_log_multiplier_round_trip(self) -> None:
        pm = 0.033333333333
        n_vars = 30
        z = log_mutation_multiplier(pm, n_vars)
        assert pm_from_log_multiplier(z, n_vars) == pytest.approx(pm)

    def test_return_types_are_python_floats(self) -> None:
        assert type(mutation_multiplier(0.1, 10)) is float
        assert type(mutation_probability(1.0, 10)) is float
        assert type(log_mutation_multiplier(0.1, 10)) is float
        assert type(pm_from_log_multiplier(0.0, 10)) is float

    def test_identity_at_baseline(self) -> None:
        """pm = 1/n_vars is exactly multiplier 1, i.e. log-multiplier 0."""
        n_vars = 30
        assert mutation_multiplier(1.0 / n_vars, n_vars) == pytest.approx(1.0)
        assert log_mutation_multiplier(1.0 / n_vars, n_vars) == pytest.approx(0.0)

    @pytest.mark.parametrize("n_vars", [0, -1, -30])
    def test_invalid_n_vars_rejected(self, n_vars: int) -> None:
        with pytest.raises(ValueError):
            mutation_multiplier(0.1, n_vars)
        with pytest.raises(ValueError):
            mutation_probability(1.0, n_vars)
        with pytest.raises(ValueError):
            log_mutation_multiplier(0.1, n_vars)
        with pytest.raises(ValueError):
            pm_from_log_multiplier(0.0, n_vars)

    @pytest.mark.parametrize("pm", [0.0, -0.01, -1.0])
    def test_nonpositive_pm_rejected(self, pm: float) -> None:
        with pytest.raises(ValueError):
            mutation_multiplier(pm, 30)
        with pytest.raises(ValueError):
            log_mutation_multiplier(pm, 30)

    @pytest.mark.parametrize("multiplier", [0.0, -0.5])
    def test_nonpositive_multiplier_rejected(self, multiplier: float) -> None:
        with pytest.raises(ValueError):
            mutation_probability(multiplier, 30)


def _transition(
    generation: int,
    operator: str,
    pm: float,
    exploration: float,
    hv: float,
    igd: float,
    delta_hv: float = 0.0,
    delta_igd: float = 0.0,
) -> dict[str, Any]:
    """Build one transition dict in the EvolutionRecorder schema."""
    return {
        "generation": int(generation),
        "state": {
            "generation": int(generation),
            "hv": float(hv),
            "igd": float(igd),
            "diversity": 0.3 + 0.01 * generation,
        },
        "action": {
            "mutation_operator": str(operator),
            "mutation_probability": float(pm),
            "exploration_strength": float(exploration),
        },
        "reward": {"delta_hv": float(delta_hv), "delta_igd": float(delta_igd)},
    }


def _trajectory(seed_offset: int, pm_base: float) -> list[dict[str, Any]]:
    """Deterministic synthetic trajectory with a known pm schedule."""
    transitions = []
    for t in range(_N_TRANSITIONS):
        reward = 0.01 * ((t + seed_offset) % 3) if t > 0 else 0.0
        transitions.append(
            _transition(
                generation=t,
                operator="polynomial" if (t + seed_offset) % 2 == 0 else "gaussian",
                pm=pm_base * (1.0 + 0.1 * t),
                exploration=10.0 + t,
                hv=0.4 + 0.01 * t + 0.02 * seed_offset,
                igd=0.5 - 0.01 * t,
                delta_hv=reward,
                delta_igd=0.5 * reward,
            )
        )
    return transitions


@pytest.fixture
def trajectories() -> list[list[dict[str, Any]]]:
    """Two synthetic trajectories with distinct pm scales."""
    return [_trajectory(0, pm_base=0.05), _trajectory(1, pm_base=0.02)]


@pytest.fixture
def encoder(trajectories: list[list[dict[str, Any]]]) -> StateEncoder:
    return StateEncoder(window=_WINDOW).fit(trajectories)


class TestLoadTrajectoryRecords:
    """Tests for the metadata-preserving trajectory loader."""

    def _write_run(
        self,
        directory: Path,
        name: str,
        problem: str,
        n_vars: int,
        seed: int,
        shuffle_generations: bool = False,
    ) -> list[dict[str, Any]]:
        transitions = _trajectory(seed, pm_base=1.0 / n_vars)
        if shuffle_generations:
            transitions = list(reversed(transitions))
        payload = {
            "config": {"problem": problem, "n_vars": n_vars, "algorithm": "nsga2"},
            "seed": int(seed),
            "runtime_sec": 1.5 + seed,
            "schema_version": 1,
            "transitions": transitions,
            "final": {"hv": 0.6, "igd": 0.4, "diversity": 0.35},
        }
        with (directory / name).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return transitions

    def test_records_carry_metadata_and_sorted_transitions(
        self, tmp_path: Path
    ) -> None:
        written = self._write_run(
            tmp_path, "zdt1_nsga2_seed3.json", "zdt1", 30, 3,
            shuffle_generations=True,
        )
        self._write_run(tmp_path, "zdt4_nsga2_seed7.json", "zdt4", 10, 7)
        (tmp_path / "index.json").write_text("{}", encoding="utf-8")

        records = load_trajectory_records(tmp_path)
        assert len(records) == 2
        # Files are processed in sorted filename order; index.json is skipped.
        assert [r["problem"] for r in records] == ["zdt1", "zdt4"]
        first = records[0]
        assert first["n_vars"] == 30
        assert first["seed"] == 3
        assert first["runtime_sec"] == pytest.approx(4.5)
        assert first["config"]["algorithm"] == "nsga2"
        # Transitions were written reversed; the loader sorts by generation.
        assert [t["generation"] for t in first["transitions"]] == list(
            range(_N_TRANSITIONS)
        )
        assert first["transitions"] == sorted(
            written, key=lambda t: t["generation"]
        )

    def test_missing_transitions_raises(self, tmp_path: Path) -> None:
        with (tmp_path / "broken.json").open("w", encoding="utf-8") as fh:
            json.dump({"config": {"problem": "zdt1", "n_vars": 30}}, fh)
        with pytest.raises(ValueError):
            load_trajectory_records(tmp_path)

    def test_missing_problem_metadata_raises(self, tmp_path: Path) -> None:
        payload = {
            "config": {"algorithm": "nsga2"},
            "seed": 0,
            "runtime_sec": 1.0,
            "transitions": _trajectory(0, 0.05),
        }
        with (tmp_path / "noconfig.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        with pytest.raises(ValueError):
            load_trajectory_records(tmp_path)

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            load_trajectory_records(tmp_path / "does_not_exist")


class TestMultiplierDatasetBuilder:
    """``mutation_target='multiplier'`` path of build_multihead_samples."""

    def test_multiplier_targets_match_manual_computation(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        n_vars_list = [30, 10]
        X, y_op, y_logpm, y_logexpl, w, ids = build_multihead_samples(
            trajectories,
            encoder,
            _WINDOW,
            mutation_target="multiplier",
            n_vars_list=n_vars_list,
        )
        expected = np.asarray(
            [
                math.log(
                    trajectories[j][t]["action"]["mutation_probability"]
                    * n_vars_list[j]
                )
                for j in range(2)
                for t in range(1, _N_TRANSITIONS)
            ]
        )
        np.testing.assert_allclose(y_logpm, expected, rtol=1e-12)
        # Everything except the pm target is identical to absolute mode.
        X_a, y_op_a, y_logpm_a, y_logexpl_a, w_a, ids_a = build_multihead_samples(
            trajectories, encoder, _WINDOW
        )
        np.testing.assert_array_equal(X, X_a)
        np.testing.assert_array_equal(y_op, y_op_a)
        np.testing.assert_allclose(y_logexpl, y_logexpl_a, rtol=1e-12)
        np.testing.assert_allclose(w, w_a, rtol=1e-12)
        np.testing.assert_array_equal(ids, ids_a)
        # Absolute mode stays log(pm); multiplier mode differs by log(n_vars).
        np.testing.assert_allclose(
            y_logpm - y_logpm_a,
            np.asarray([math.log(n_vars_list[j]) for j in ids], dtype=float),
            rtol=1e-12,
        )

    def test_default_is_absolute(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        """Default mutation_target reproduces the Phase-1.5 log-pm target."""
        _, _, y_logpm, _, _, _ = build_multihead_samples(
            trajectories, encoder, _WINDOW
        )
        expected = np.asarray(
            [
                math.log(trajectories[j][t]["action"]["mutation_probability"])
                for j in range(2)
                for t in range(1, _N_TRANSITIONS)
            ]
        )
        np.testing.assert_allclose(y_logpm, expected, rtol=1e-12)

    def test_multiplier_requires_n_vars_list(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        with pytest.raises(ValueError):
            build_multihead_samples(
                trajectories, encoder, _WINDOW, mutation_target="multiplier"
            )

    def test_n_vars_list_length_must_match(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        with pytest.raises(ValueError):
            build_multihead_samples(
                trajectories,
                encoder,
                _WINDOW,
                mutation_target="multiplier",
                n_vars_list=[30],
            )

    def test_n_vars_must_be_positive(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        with pytest.raises(ValueError):
            build_multihead_samples(
                trajectories,
                encoder,
                _WINDOW,
                mutation_target="multiplier",
                n_vars_list=[30, 0],
            )

    def test_unknown_mutation_target_rejected(
        self,
        trajectories: list[list[dict[str, Any]]],
        encoder: StateEncoder,
    ) -> None:
        with pytest.raises(ValueError):
            build_multihead_samples(
                trajectories, encoder, _WINDOW, mutation_target="relative"
            )


def _constant_head_controller(
    input_dim: int,
    log_pm_output: float,
    operator_idx: int = 0,
    log_expl_output: float = math.log(10.0),
    mutation_target: str = "absolute",
) -> MultiHeadController:
    """Controller whose heads output exact constants for any input.

    All head weights are zeroed and the biases set to the desired outputs,
    so the (float32) network reproduces the requested values deterministically
    without any training.
    """
    controller = MultiHeadController(
        input_dim=input_dim, hidden_dims=(8,), seed=0, mutation_target=mutation_target
    )
    with torch.no_grad():
        for head in (
            controller._net.operator_head,
            controller._net.pm_head,
            controller._net.exploration_head,
        ):
            head.weight.zero_()
            head.bias.zero_()
        controller._net.pm_head.bias.fill_(log_pm_output)
        controller._net.operator_head.bias[operator_idx] = 1.0
        controller._net.exploration_head.bias.fill_(log_expl_output)
    return controller


@pytest.fixture
def history(trajectories: list[list[dict[str, Any]]]) -> list[dict[str, float]]:
    """One merged history window of the test encoder's window length."""
    return [merge_state_reward(t) for t in trajectories[0][:_WINDOW]]


class TestMultiplierController:
    """Multiplier-mode ``predict_action`` of MultiHeadController."""

    def test_predict_action_math(
        self, encoder: StateEncoder, history: list[dict[str, float]]
    ) -> None:
        """Known log-multiplier output -> pm = multiplier / n_vars."""
        n_vars = 30
        log_mult = math.log(2.5)
        controller = _constant_head_controller(
            encoder.dim, log_mult, mutation_target="multiplier"
        )
        action = controller.predict_action(
            history, encoder, pm_min=1e-9, pm_max=1.0, n_vars=n_vars
        )
        assert set(action) == {
            "mutation_operator",
            "mutation_probability",
            "exploration_strength",
        }
        assert action["mutation_operator"] == "polynomial"
        assert action["mutation_probability"] == pytest.approx(2.5 / n_vars, rel=1e-5)
        assert action["exploration_strength"] == pytest.approx(10.0, rel=1e-5)

    def test_predict_action_clips_multiplier_range(
        self, encoder: StateEncoder, history: list[dict[str, float]]
    ) -> None:
        """Log-multiplier predictions are clipped to [log 0.25, log 8.0]."""
        n_vars = 30
        high = _constant_head_controller(
            encoder.dim, math.log(100.0), mutation_target="multiplier"
        )
        action = high.predict_action(history, encoder, 1e-9, 1.0, n_vars=n_vars)
        assert action["mutation_probability"] == pytest.approx(8.0 / n_vars, rel=1e-5)
        low = _constant_head_controller(
            encoder.dim, math.log(0.01), mutation_target="multiplier"
        )
        action = low.predict_action(history, encoder, 1e-9, 1.0, n_vars=n_vars)
        assert action["mutation_probability"] == pytest.approx(0.25 / n_vars, rel=1e-5)

    def test_n_vars_required_in_multiplier_mode(
        self, encoder: StateEncoder, history: list[dict[str, float]]
    ) -> None:
        controller = _constant_head_controller(
            encoder.dim, 0.0, mutation_target="multiplier"
        )
        with pytest.raises(ValueError):
            controller.predict_action(history, encoder, 1e-9, 1.0)
        with pytest.raises(ValueError):
            controller.predict_action(history, encoder, 1e-9, 1.0, n_vars=0)

    def test_invalid_mutation_target_rejected(self) -> None:
        with pytest.raises(ValueError):
            MultiHeadController(input_dim=4, mutation_target="relative")

    def test_absolute_mode_unchanged(
        self, encoder: StateEncoder, history: list[dict[str, float]]
    ) -> None:
        """Default mode still maps the head output through [pm_min, pm_max]."""
        log_pm = math.log(0.05)
        controller = _constant_head_controller(encoder.dim, log_pm)
        assert controller.mutation_target == "absolute"
        action = controller.predict_action(history, encoder, pm_min=0.01, pm_max=0.2)
        assert action["mutation_probability"] == pytest.approx(0.05, rel=1e-5)
        # Absolute mode does not require n_vars; clipping still uses pm bounds.
        clipped = _constant_head_controller(encoder.dim, math.log(0.9))
        action = clipped.predict_action(history, encoder, pm_min=0.01, pm_max=0.2)
        assert action["mutation_probability"] == pytest.approx(0.2, rel=1e-5)

    def test_save_load_preserves_mutation_target(
        self, tmp_path: Path, encoder: StateEncoder, history: list[dict[str, float]]
    ) -> None:
        controller = _constant_head_controller(
            encoder.dim, math.log(3.0), mutation_target="multiplier"
        )
        path = tmp_path / "controller.pt"
        controller.save(path)
        loaded = MultiHeadController.load(path)
        assert loaded.mutation_target == "multiplier"
        action = loaded.predict_action(history, encoder, 1e-9, 1.0, n_vars=30)
        assert action["mutation_probability"] == pytest.approx(0.1, rel=1e-5)

    def test_load_legacy_checkpoint_defaults_to_absolute(self, tmp_path: Path) -> None:
        """Checkpoints saved before Phase 1.75 (no mutation_target key) load
        as absolute-mode controllers."""
        controller = MultiHeadController(input_dim=4, hidden_dims=(8,), seed=0)
        path = tmp_path / "legacy.pt"
        controller.save(path)
        payload = torch.load(path, map_location="cpu")
        del payload["config"]["mutation_target"]
        torch.save(payload, path)
        loaded = MultiHeadController.load(path)
        assert loaded.mutation_target == "absolute"
