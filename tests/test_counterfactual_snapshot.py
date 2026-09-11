from __future__ import annotations

"""Tests for NSGA-II snapshot/restore and the counterfactual evaluator.

Covers the Phase-1.75 Task-5 contract:

* ``NSGAII.snapshot_state()`` / ``restore_state()`` replay identity:
  snapshot at generation 5, step, restore, same step must reproduce the
  next generation bit-identically (population, generation counter, action
  record), and one snapshot must support repeated restores.
* Restoring and stepping with a *different* action must give a different
  result (the branch actually depends on the action).
* A tiny end-to-end counterfactual run (zdt1, 2 states, 4 alternatives,
  2 reps, pop 20, 8 generations) is deterministic across repeated
  invocations and yields percentile ranks in [0, 1].
* Phase 2B: the same tiny run with ``--controller-type planning`` (a
  PlanningController config JSON wrapping a synthetic-trained
  OutcomePredictor) is deterministic, yields ranks in [0, 1], and rejects
  a missing ``--predictor``.

Runtime is dominated by the torch import; the NSGA-II runs are tiny
(pop 20, <= 8 generations) and stay well under 60 s in total.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from algorithms.nsga2 import NSGAII, OperatorConfig
from benchmarks import get_problem
from controller.dataset import build_outcome_samples
from controller.multihead_controller import MultiHeadController
from controller.outcome_predictor import OutcomePredictor
from controller.planning_controller import PlanningController
from controller.state_encoder import StateEncoder
from experiments import counterfactual_actions as cfa

_POP_SIZE = 20
_SEED = 7
_SNAPSHOT_GEN = 5
_ACTION = {
    "mutation_prob": 0.1,
    "mutation_operator": "gaussian",
    "exploration_strength": 0.2,
}


def _make_algo(seed: int = _SEED, pop_size: int = _POP_SIZE) -> NSGAII:
    """Create a small deterministic NSGA-II instance on ZDT1."""
    return NSGAII(
        problem=get_problem("zdt1"),
        pop_size=pop_size,
        operators=OperatorConfig(),
        seed=seed,
    )


def _step_with_action(algo: NSGAII) -> None:
    """Advance one generation with the shared explicit action."""
    algo.step(
        mutation_prob=_ACTION["mutation_prob"],
        mutation_operator=_ACTION["mutation_operator"],
        exploration_strength=_ACTION["exploration_strength"],
    )


def test_snapshot_restore_replay_identity() -> None:
    """Snapshot at gen 5, step, restore, same step -> bit-identical state."""
    algo = _make_algo()
    algo.initialize()
    for _ in range(_SNAPSHOT_GEN):
        algo.step()
    assert algo.generation == _SNAPSHOT_GEN
    snapshot = algo.snapshot_state()

    _step_with_action(algo)
    expected_f = algo.population_f
    expected_x = algo.population_x
    expected_gen = algo.generation
    expected_action = algo.current_action()

    algo.restore_state(snapshot)
    assert algo.generation == _SNAPSHOT_GEN
    _step_with_action(algo)
    np.testing.assert_array_equal(algo.population_f, expected_f)
    np.testing.assert_array_equal(algo.population_x, expected_x)
    assert algo.generation == expected_gen
    assert algo.current_action() == expected_action

    # The snapshot is never mutated: a second restore replays identically.
    algo.restore_state(snapshot)
    _step_with_action(algo)
    np.testing.assert_array_equal(algo.population_f, expected_f)
    np.testing.assert_array_equal(algo.population_x, expected_x)


def test_snapshot_restore_default_step_replay_identity() -> None:
    """Replay identity also holds for the default (no-override) step."""
    algo = _make_algo(seed=11)
    algo.initialize()
    for _ in range(_SNAPSHOT_GEN):
        algo.step()
    snapshot = algo.snapshot_state()
    algo.step()
    expected_f = algo.population_f
    algo.restore_state(snapshot)
    algo.step()
    np.testing.assert_array_equal(algo.population_f, expected_f)


def test_restore_then_different_action_differs() -> None:
    """Restore followed by a different action must give a different result."""
    algo = _make_algo()
    algo.initialize()
    for _ in range(_SNAPSHOT_GEN):
        algo.step()
    snapshot = algo.snapshot_state()
    algo.step()
    default_f = algo.population_f
    default_x = algo.population_x

    algo.restore_state(snapshot)
    algo.step(mutation_prob=1.0, mutation_operator="gaussian", exploration_strength=0.3)
    assert not np.array_equal(algo.population_f, default_f)
    assert not np.array_equal(algo.population_x, default_x)


def test_restore_preserves_last_action_record() -> None:
    """current_action() after restore reports the action at snapshot time."""
    algo = _make_algo()
    algo.initialize()
    algo.step(mutation_prob=0.5, mutation_operator="gaussian", exploration_strength=0.25)
    snapshot = algo.snapshot_state()
    action_at_snapshot = algo.current_action()
    algo.step()
    algo.step()
    assert algo.current_action() != action_at_snapshot  # defaults restored by step()
    algo.restore_state(snapshot)
    assert algo.current_action() == action_at_snapshot
    assert algo.generation == 1


def test_restore_rejects_incompatible_pop_size() -> None:
    """A snapshot cannot be restored into an instance with another pop_size."""
    algo = _make_algo(pop_size=20)
    algo.initialize()
    algo.step()
    snapshot = algo.snapshot_state()
    other = _make_algo(pop_size=40)
    with pytest.raises(ValueError, match="pop_size"):
        other.restore_state(snapshot)


def test_restore_rejects_foreign_operator_config() -> None:
    """A snapshot cannot be restored into an instance with another config."""
    algo = _make_algo()
    algo.initialize()
    snapshot = algo.snapshot_state()
    other = NSGAII(
        problem=get_problem("zdt1"),
        pop_size=_POP_SIZE,
        operators=OperatorConfig(eta_m=5.0),
        seed=_SEED,
    )
    with pytest.raises(ValueError, match="config"):
        other.restore_state(snapshot)


def test_snapshot_before_initialize_raises() -> None:
    """There is no replayable population state before initialize()."""
    algo = _make_algo()
    with pytest.raises(RuntimeError, match="initialize"):
        algo.snapshot_state()


def test_snapshot_generations_grid() -> None:
    """Default grid: 40 unique generations evenly spread over 2..98."""
    gens = cfa.snapshot_generations(generations=100, states_per_run=40)
    assert len(gens) == 40
    assert gens[0] == 2 and gens[-1] == 98
    assert len(set(gens)) == len(gens)
    # Tiny grid used by the evaluator test below.
    assert cfa.snapshot_generations(generations=8, states_per_run=2) == [2, 6]
    with pytest.raises(ValueError):
        cfa.snapshot_generations(generations=3, states_per_run=2)


def test_percentile_rank_of_first() -> None:
    """Midrank percentile: best -> 1.0, worst -> 0.0, all-tied -> 0.5."""
    assert cfa.percentile_rank_of_first([3.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert cfa.percentile_rank_of_first([1.0, 3.0, 2.0]) == pytest.approx(0.0)
    assert cfa.percentile_rank_of_first([1.0, 1.0, 1.0]) == pytest.approx(0.5)
    assert cfa.percentile_rank_of_first([2.0, 1.0, 3.0]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        cfa.percentile_rank_of_first([1.0])


# --- Tiny end-to-end counterfactual run ------------------------------------

_TINY_GENS = 8
_TINY_STATES = 2
_TINY_ALTERNATIVES = 4
_TINY_REPS = 2
_TINY_WINDOW = 3


def _tiny_trajectories() -> list[list[dict[str, Any]]]:
    """Two synthetic trajectories (recorder schema) to fit the tiny encoder."""
    trajectories: list[list[dict[str, Any]]] = []
    for j in range(2):
        hv = 0.30 + 0.05 * j
        igd_value = 0.50 - 0.02 * j
        transitions = []
        for t in range(_TINY_GENS + 1):
            delta_hv = 0.0 if t == 0 else 0.006 + 0.001 * ((t + j) % 3)
            delta_igd = 0.0 if t == 0 else 0.005 + 0.001 * ((t + j + 1) % 3)
            hv += delta_hv
            igd_value -= delta_igd
            transitions.append(
                {
                    "generation": t,
                    "state": {
                        "generation": t,
                        "hv": hv,
                        "igd": igd_value,
                        "diversity": 0.25 + 0.01 * t,
                    },
                    "action": {
                        "mutation_operator": "polynomial",
                        "mutation_probability": 1.0 / 30.0,
                        "exploration_strength": 20.0,
                    },
                    "reward": {"delta_hv": delta_hv, "delta_igd": delta_igd},
                }
            )
        trajectories.append(transitions)
    return trajectories


@pytest.fixture(scope="module")
def tiny_counterfactual(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Harvest 2 snapshots and run the tiny evaluator twice (determinism)."""
    root = tmp_path_factory.mktemp("counterfactual")
    encoder = StateEncoder(_TINY_WINDOW).fit(_tiny_trajectories())
    encoder_path = root / "encoder.json"
    encoder.save(encoder_path)
    controller = MultiHeadController(input_dim=encoder.dim, seed=0, name="cf_test")
    controller_path = root / "controller.pt"
    controller.save(controller_path)

    snapshots_dir = root / "snapshots"
    snapshot_paths = cfa.harvest_snapshots(
        "zdt1",
        seed=1000,
        generations=_TINY_GENS,
        pop_size=_POP_SIZE,
        states_per_run=_TINY_STATES,
        n_reference_points=200,
        ref_point=np.asarray([1.1, 1.1]),
        out_dir=snapshots_dir,
    )

    def _evaluate(out_dir: Path) -> dict[str, Any]:
        args = cfa.parse_args(
            [
                "evaluate",
                "--problem", "zdt1",
                "--snapshots-dir", str(snapshots_dir),
                "--out-dir", str(out_dir),
                "--controller", str(controller_path),
                "--encoder", str(encoder_path),
                "--n-alternatives", str(_TINY_ALTERNATIVES),
                "--n-reps", str(_TINY_REPS),
            ]
        )
        return cfa.run_evaluate(args)

    payload_a = _evaluate(root / "eval_a")
    payload_b = _evaluate(root / "eval_b")
    return {
        "root": root,
        "snapshot_paths": snapshot_paths,
        "payload_a": payload_a,
        "payload_b": payload_b,
    }


def test_harvest_writes_snapshot_per_state(tiny_counterfactual: dict[str, Any]) -> None:
    """Harvest wrote one pickle per snapshot generation (gens 2 and 6)."""
    paths = tiny_counterfactual["snapshot_paths"]
    assert len(paths) == _TINY_STATES
    names = sorted(p.name for p in paths)
    assert names == ["zdt1__seed1000__gen2.pkl", "zdt1__seed1000__gen6.pkl"]
    for path in paths:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        assert set(payload) >= {"problem", "seed", "generation", "state", "history"}
        assert payload["problem"] == "zdt1"
        # History covers generations 0..g (what the controller may observe).
        assert len(payload["history"]) == payload["generation"] + 1


def test_evaluator_is_deterministic(tiny_counterfactual: dict[str, Any]) -> None:
    """Two identical invocations produce identical payloads and JSON bytes."""
    payload_a = tiny_counterfactual["payload_a"]
    payload_b = tiny_counterfactual["payload_b"]
    assert payload_a == payload_b
    root = tiny_counterfactual["root"]
    bytes_a = (root / "eval_a" / "counterfactual_zdt1.json").read_bytes()
    bytes_b = (root / "eval_b" / "counterfactual_zdt1.json").read_bytes()
    assert bytes_a == bytes_b


def test_evaluator_records_schema_and_ranks(tiny_counterfactual: dict[str, Any]) -> None:
    """Per-state records are complete; percentile ranks lie in [0, 1]."""
    payload = tiny_counterfactual["payload_a"]
    assert payload["problem"] == "zdt1"
    summary = payload["summary"]
    assert summary["n_states"] == _TINY_STATES
    assert summary["seeds"] == [1000]
    assert 0.0 <= summary["mean_percentile_rank"] <= 1.0
    for state in payload["states"]:
        assert 0.0 <= state["controller_percentile_rank"] <= 1.0
        assert len(state["controller_percentile_rank_per_rep"]) == _TINY_REPS
        for rank in state["controller_percentile_rank_per_rep"]:
            assert 0.0 <= rank <= 1.0
        candidates = state["candidates"]
        assert len(candidates) == 1 + _TINY_ALTERNATIVES
        assert candidates[0]["kind"] == "controller"
        assert all(c["kind"] == "alternative" for c in candidates[1:])
        for candidate in candidates:
            assert len(candidate["rewards"]) == _TINY_REPS
            assert set(candidate["action"]) == {
                "mutation_operator",
                "mutation_probability",
                "exploration_strength",
            }
        # The controller action respects the deployment pm bounds.
        pm = state["controller_action"]["mutation_probability"]
        assert 0.25 / 30.0 - 1e-12 <= pm <= 8.0 / 30.0 + 1e-12


def test_aggregate_merges_and_bootstraps(tiny_counterfactual: dict[str, Any]) -> None:
    """Aggregate merges the per-problem file, is deterministic, has CI+thirds."""
    root = tiny_counterfactual["root"]
    args = cfa.parse_args(
        ["aggregate", "--problems", "zdt1", "--results-dir", str(root / "eval_a")]
    )
    payload_a = cfa.run_aggregate(args)
    payload_b = cfa.run_aggregate(args)
    assert payload_a == payload_b
    overall = payload_a["overall"]
    assert overall["n_states"] == _TINY_STATES
    lo, hi = overall["bootstrap_ci_95"]
    assert lo <= overall["mean_percentile_rank"] <= hi
    assert payload_a["per_problem"]["zdt1"]["n_states"] == _TINY_STATES
    thirds = payload_a["generation_thirds"]
    assert set(thirds) == {"early", "mid", "late"}
    # Gens 2 and 6 of 8 -> relative 0.25 (early) and 0.75 (late); mid empty.
    assert thirds["early"]["n_states"] == 1
    assert thirds["mid"]["n_states"] == 0
    assert thirds["mid"]["mean_percentile_rank"] is None
    assert thirds["late"]["n_states"] == 1
    assert (root / "eval_a" / "counterfactual.json").exists()


# --- Phase 2B: planning-controller counterfactual run ------------------------

#: Horizons of the tiny outcome predictor; small so the 9-transition
#: synthetic trajectories yield outcome samples (``t + max(h) < 9``).
_TINY_PREDICTOR_HORIZONS = [1, 2]
_TINY_N_CANDIDATES = 4


@pytest.fixture(scope="module")
def tiny_planning_counterfactual(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """Tiny evaluator run with ``--controller-type planning``, twice.

    Mirrors :func:`tiny_counterfactual` but the index-0 controller is a
    :class:`PlanningController` config JSON bound to a synthetically
    trained :class:`OutcomePredictor` checkpoint.
    """
    root = tmp_path_factory.mktemp("counterfactual_planning")
    trajectories = _tiny_trajectories()
    encoder = StateEncoder(_TINY_WINDOW).fit(trajectories)
    encoder_path = root / "encoder.json"
    encoder.save(encoder_path)

    X, y, _, _ = build_outcome_samples(
        trajectories, encoder, _TINY_WINDOW, _TINY_PREDICTOR_HORIZONS
    )
    assert X.shape[0] > 0
    predictor = OutcomePredictor(
        input_dim=X.shape[1],
        horizons=_TINY_PREDICTOR_HORIZONS,
        hidden_dims=(8, 8),
        seed=0,
    )
    predictor.fit(X, y, epochs=3)
    predictor_path = root / "predictor.pt"
    predictor.save(predictor_path)
    planner = PlanningController(
        predictor,
        n_candidates=_TINY_N_CANDIDATES,
        candidate_seed=0,
        horizon_weights=[0.5, 0.5],
    )
    planner.predictor_path = str(predictor_path)
    planner_path = root / "planner.json"
    planner.save(planner_path)

    snapshots_dir = root / "snapshots"
    cfa.harvest_snapshots(
        "zdt1",
        seed=1000,
        generations=_TINY_GENS,
        pop_size=_POP_SIZE,
        states_per_run=_TINY_STATES,
        n_reference_points=200,
        ref_point=np.asarray([1.1, 1.1]),
        out_dir=snapshots_dir,
    )

    def _evaluate(out_dir: Path) -> dict[str, Any]:
        args = cfa.parse_args(
            [
                "evaluate",
                "--problem", "zdt1",
                "--snapshots-dir", str(snapshots_dir),
                "--out-dir", str(out_dir),
                "--controller", str(planner_path),
                "--controller-type", "planning",
                "--predictor", str(predictor_path),
                "--encoder", str(encoder_path),
                "--n-alternatives", str(_TINY_ALTERNATIVES),
                "--n-reps", str(_TINY_REPS),
            ]
        )
        return cfa.run_evaluate(args)

    payload_a = _evaluate(root / "eval_a")
    payload_b = _evaluate(root / "eval_b")
    return {
        "root": root,
        "planner_path": planner_path,
        "predictor_path": predictor_path,
        "payload_a": payload_a,
        "payload_b": payload_b,
    }


def test_planner_config_json_matches_loader_contract(
    tiny_planning_counterfactual: dict[str, Any],
) -> None:
    """The saved planner JSON carries the keys ``_load_planning_controller`` reads."""
    with tiny_planning_counterfactual["planner_path"].open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["name"] == "planning_predictor"
    assert payload["predictor_path"] == str(tiny_planning_counterfactual["predictor_path"])
    config = payload["config"]
    assert config["n_candidates"] == _TINY_N_CANDIDATES
    assert config["candidate_seed"] == 0
    assert config["horizon_weights"] == [0.5, 0.5]
    assert len(config["pm_mult_range"]) == 2


def test_planning_evaluator_is_deterministic(
    tiny_planning_counterfactual: dict[str, Any],
) -> None:
    """Two identical planning invocations produce identical payloads and bytes."""
    payload_a = tiny_planning_counterfactual["payload_a"]
    payload_b = tiny_planning_counterfactual["payload_b"]
    assert payload_a == payload_b
    root = tiny_planning_counterfactual["root"]
    bytes_a = (root / "eval_a" / "counterfactual_zdt1.json").read_bytes()
    bytes_b = (root / "eval_b" / "counterfactual_zdt1.json").read_bytes()
    assert bytes_a == bytes_b


def test_planning_evaluator_records_schema_and_ranks(
    tiny_planning_counterfactual: dict[str, Any],
) -> None:
    """Planning run: schema complete, ranks in [0, 1], config self-describing."""
    payload = tiny_planning_counterfactual["payload_a"]
    config = payload["config"]
    assert config["controller_type"] == "planning"
    assert config["predictor"].endswith("predictor.pt")
    summary = payload["summary"]
    assert summary["n_states"] == _TINY_STATES
    assert 0.0 <= summary["mean_percentile_rank"] <= 1.0
    for state in payload["states"]:
        assert 0.0 <= state["controller_percentile_rank"] <= 1.0
        candidates = state["candidates"]
        assert len(candidates) == 1 + _TINY_ALTERNATIVES
        assert candidates[0]["kind"] == "controller"
        assert all(c["kind"] == "alternative" for c in candidates[1:])
        # The planner action respects the full-action pm bounds on ZDT1
        # (n_vars = 30).
        pm = state["controller_action"]["mutation_probability"]
        assert 0.25 / 30.0 - 1e-12 <= pm <= 8.0 / 30.0 + 1e-12


def test_planning_controller_type_requires_predictor(
    tiny_planning_counterfactual: dict[str, Any],
) -> None:
    """``--controller-type planning`` without ``--predictor`` is rejected."""
    fixture = tiny_planning_counterfactual
    args = cfa.parse_args(
        [
            "evaluate",
            "--problem", "zdt1",
            "--snapshots-dir", str(fixture["root"] / "snapshots"),
            "--out-dir", str(fixture["root"] / "eval_missing"),
            "--controller", str(fixture["planner_path"]),
            "--controller-type", "planning",
            "--encoder", str(fixture["root"] / "encoder.json"),
            "--n-alternatives", str(_TINY_ALTERNATIVES),
            "--n-reps", str(_TINY_REPS),
        ]
    )
    with pytest.raises(ValueError, match="predictor"):
        cfa.run_evaluate(args)
