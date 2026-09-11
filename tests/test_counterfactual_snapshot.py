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
* Phase 2B Task 4: the ``evaluate-horizon`` / ``aggregate-horizon`` pair is
  deterministic, records per-horizon percentile ranks, oracle regret and
  predicted-vs-realized correlations (null without an outcome model or for
  horizons the predictor does not cover), spells rewards as
  ``hv(after h generations) - hv(before the branch)``, and accumulates
  hypervolume over a multi-generation branch on an HV-positive state.

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


# --- Phase 2B Task 4: long-horizon counterfactual evaluator ------------------

#: Horizons of the tiny ``evaluate-horizon`` runs. ``2`` is one of the tiny
#: planning predictor's horizons (``[1, 2]``) while ``4`` is not, so the
#: planning run exercises both the available and the unavailable case.
_TINY_HORIZONS = ["2", "4"]
_TINY_HORIZON_ALTERNATIVES = 3
_TINY_HORIZON_REPS = 2
#: Harvest length of the HV-positive fixture: with pop 20 the generation-2
#: state still has HV = 0 under the (1.1, 1.1) reference point, while the
#: generation-58 state of a 60-generation run has HV > 0.
_TINY_LONG_GENS = 60


def _horizon_argv(
    *,
    out_dir: Path,
    snapshots_dir: Path,
    root: Path,
    controller: Path,
    controller_type: str = "multihead",
    predictor: Path | None = None,
) -> list[str]:
    """Command line of a tiny ``evaluate-horizon`` run on the shared zdt1 state."""
    argv = [
        "evaluate-horizon",
        "--problem", "zdt1",
        "--snapshots-dir", str(snapshots_dir),
        "--out-dir", str(out_dir),
        "--controller", str(controller),
        "--controller-type", controller_type,
        "--encoder", str(root / "encoder.json"),
        "--horizons", *_TINY_HORIZONS,
        "--n-alternatives", str(_TINY_HORIZON_ALTERNATIVES),
        "--n-reps", str(_TINY_HORIZON_REPS),
    ]
    if predictor is not None:
        argv += ["--predictor", str(predictor)]
    return argv


@pytest.fixture(scope="module")
def tiny_horizon_counterfactual(tiny_counterfactual: dict[str, Any]) -> dict[str, Any]:
    """Run the tiny multihead ``evaluate-horizon`` twice (determinism)."""
    root = tiny_counterfactual["root"]
    snapshots_dir = root / "snapshots"

    def _run(out_dir: Path) -> dict[str, Any]:
        return cfa.run_evaluate_horizon(
            cfa.parse_args(
                _horizon_argv(
                    out_dir=out_dir,
                    snapshots_dir=snapshots_dir,
                    root=root,
                    controller=root / "controller.pt",
                )
            )
        )

    return {
        "root": root,
        "payload_a": _run(root / "horizon_a"),
        "payload_b": _run(root / "horizon_b"),
    }


def test_horizon_evaluator_is_deterministic(
    tiny_horizon_counterfactual: dict[str, Any],
) -> None:
    """Two identical invocations produce identical payloads and JSON bytes."""
    payload_a = tiny_horizon_counterfactual["payload_a"]
    payload_b = tiny_horizon_counterfactual["payload_b"]
    assert payload_a == payload_b
    root = tiny_horizon_counterfactual["root"]
    bytes_a = (root / "horizon_a" / "counterfactual_horizon_zdt1.json").read_bytes()
    bytes_b = (root / "horizon_b" / "counterfactual_horizon_zdt1.json").read_bytes()
    assert bytes_a == bytes_b


def test_horizon_evaluator_records_schema_and_ranks(
    tiny_horizon_counterfactual: dict[str, Any],
) -> None:
    """Per-horizon records are complete, self-consistent and in range."""
    root = tiny_horizon_counterfactual["root"]
    out_path = root / "horizon_a" / "counterfactual_horizon_zdt1.json"
    assert out_path.is_file()
    payload = tiny_horizon_counterfactual["payload_a"]

    assert payload["problem"] == "zdt1"
    config = payload["config"]
    assert config["horizons"] == [2, 4]
    assert config["branch_generations"] == 4
    assert config["controller_type"] == "multihead"
    assert config["predictor"] is None
    assert config["n_alternatives"] == _TINY_HORIZON_ALTERNATIVES
    assert config["n_reps"] == _TINY_HORIZON_REPS

    summary = payload["summary"]
    assert summary["n_states"] == _TINY_STATES
    assert summary["seeds"] == [1000]
    assert set(summary["per_horizon"]) == set(_TINY_HORIZONS)
    for key, entry in summary["per_horizon"].items():
        assert entry["n_states"] == _TINY_STATES
        assert 0.0 <= entry["mean_percentile_rank"] <= 1.0
        assert entry["mean_oracle_regret"] >= 0.0
        # No outcome model -> correlations are reported as null, never faked.
        assert entry["n_states_with_correlation"] == 0
        assert entry["mean_spearman"] is None
        assert entry["mean_kendall"] is None

    for state in payload["states"]:
        assert set(state["per_horizon"]) == set(_TINY_HORIZONS)
        assert state["prediction_available"] is False
        candidates = state["candidates"]
        # Phase 2.75D protocol: controller (index 0) + NSGA-II default action
        # (index 1, Target B baseline) + sampled alternatives.
        assert len(candidates) == 2 + _TINY_HORIZON_ALTERNATIVES
        assert candidates[0]["kind"] == "controller"
        assert candidates[1]["kind"] == "default"
        assert all(c["kind"] == "alternative" for c in candidates[2:])
        for candidate in candidates:
            assert candidate["predicted_reward"] is None
            assert set(candidate["future_hv"]) == set(_TINY_HORIZONS)
            for key in _TINY_HORIZONS:
                assert len(candidate["future_hv"][key]) == _TINY_HORIZON_REPS
                assert len(candidate["reward"][key]) == _TINY_HORIZON_REPS
                for rep in range(_TINY_HORIZON_REPS):
                    # reward_h = hv(after h generations) - hv(before branch)
                    assert candidate["reward"][key][rep] == pytest.approx(
                        candidate["future_hv"][key][rep]
                        - state["state_metrics"]["hv"]
                    )
        # NB: hypervolume is not pointwise monotone in the horizon here --
        # NSGA-II truncates an oversized first front by crowding distance,
        # which can drop a nondominated point. The HV-positive fixture below
        # checks the accumulation behaviour on a non-degenerate state.
        for key in _TINY_HORIZONS:
            entry = state["per_horizon"][key]
            means = [float(np.mean(c["reward"][key])) for c in candidates]
            ranks_per_rep = [
                cfa.percentile_rank_of_first(
                    [c["reward"][key][rep] for c in candidates]
                )
                for rep in range(_TINY_HORIZON_REPS)
            ]
            assert 0.0 <= entry["controller_percentile_rank"] <= 1.0
            assert entry["controller_percentile_rank"] == pytest.approx(
                float(np.mean(ranks_per_rep))
            )
            assert entry["controller_percentile_rank_per_rep"] == pytest.approx(
                ranks_per_rep
            )
            assert entry["controller_mean_reward"] == pytest.approx(means[0])
            assert entry["best_candidate_index"] == int(np.argmax(means))
            assert entry["best_mean_reward"] == pytest.approx(max(means))
            assert entry["oracle_regret"] == pytest.approx(max(means) - means[0])
            assert entry["oracle_regret"] >= 0.0
            assert entry["candidate_reward_spread"] == pytest.approx(
                max(means) - min(means)
            )
            assert entry["planner_is_oracle_argmax"] == (
                entry["best_candidate_index"] == 0
            )


@pytest.fixture(scope="module")
def tiny_long_horizon(tiny_counterfactual: dict[str, Any]) -> dict[str, Any]:
    """``evaluate-horizon`` on a late, HV-positive zdt1 snapshot.

    The 8-generation fixture states all have HV = 0 under the (1.1, 1.1)
    reference point, so they cannot demonstrate that a branch accumulates
    hypervolume over generations. This fixture harvests a 60-generation run
    (reusing the fixture's encoder and controller) whose generation-58 state
    has HV > 0, so the multi-generation rewards are non-degenerate.
    """
    root = tiny_counterfactual["root"]
    snapshots_dir = root / "snapshots_long"
    cfa.harvest_snapshots(
        "zdt1",
        seed=1000,
        generations=_TINY_LONG_GENS,
        pop_size=_POP_SIZE,
        states_per_run=_TINY_STATES,
        n_reference_points=200,
        ref_point=np.asarray([1.1, 1.1]),
        out_dir=snapshots_dir,
    )
    payload = cfa.run_evaluate_horizon(
        cfa.parse_args(
            _horizon_argv(
                out_dir=root / "horizon_long",
                snapshots_dir=snapshots_dir,
                root=root,
                controller=root / "controller.pt",
            )
        )
    )
    return {"root": root, "payload": payload}


def test_horizon_evaluator_branches_accumulate_hypervolume(
    tiny_long_horizon: dict[str, Any],
) -> None:
    """On an HV-positive state longer branches really run more generations."""
    payload = tiny_long_horizon["payload"]
    assert payload["config"]["generations"] == _TINY_LONG_GENS
    late_state = max(payload["states"], key=lambda s: s["state_metrics"]["hv"])
    hv_before = late_state["state_metrics"]["hv"]
    assert hv_before > 0.0
    h2 = late_state["per_horizon"]["2"]
    h4 = late_state["per_horizon"]["4"]
    assert h2["n_candidates"] == 2 + _TINY_HORIZON_ALTERNATIVES
    assert 0.0 <= h2["controller_percentile_rank"] <= 1.0
    assert 0.0 <= h4["controller_percentile_rank"] <= 1.0
    assert h2["oracle_regret"] >= 0.0 and h4["oracle_regret"] >= 0.0
    assert h2["candidate_reward_spread"] >= 0.0 and h4["candidate_reward_spread"] >= 0.0

    h2_values = [v for c in late_state["candidates"] for v in c["future_hv"]["2"]]
    h4_values = [v for c in late_state["candidates"] for v in c["future_hv"]["4"]]
    # The branch gains hypervolume over generations, and the 4-generation
    # horizon is strictly better than the 2-generation one on this state --
    # the difference the one-step Phase-1.75 evaluator cannot observe.
    assert max(h2_values) > hv_before
    assert max(h4_values) > max(h2_values)
    # Rewards stay measured against the branch point and are never clipped:
    # crowding-distance truncation can make a longer horizon marginally worse
    # than a shorter one (min(h4 - h2) < 0 for this fixture), and that is
    # recorded rather than clamped.
    for candidate in late_state["candidates"]:
        for key in _TINY_HORIZONS:
            for rep in range(_TINY_HORIZON_REPS):
                assert candidate["reward"][key][rep] == pytest.approx(
                    candidate["future_hv"][key][rep] - hv_before
                )
    assert min(h4_values) >= hv_before - 1e-2

    # Recompute the ranks and regrets from the recorded candidate rewards.
    # This state has non-zero, candidate-dependent rewards, so a swapped
    # replicate/candidate axis (a bug the degenerate fixture cannot expose)
    # would make these differ.
    candidates = late_state["candidates"]
    # The check below only has teeth if the candidate rewards actually differ.
    assert max(h4_values) - min(h4_values) > 1e-6
    for key in _TINY_HORIZONS:
        entry = late_state["per_horizon"][key]
        ranks_per_rep = [
            cfa.percentile_rank_of_first([c["reward"][key][rep] for c in candidates])
            for rep in range(_TINY_HORIZON_REPS)
        ]
        means = [float(np.mean(c["reward"][key])) for c in candidates]
        assert entry["controller_percentile_rank_per_rep"] == pytest.approx(
            ranks_per_rep
        )
        assert entry["controller_percentile_rank"] == pytest.approx(
            float(np.mean(ranks_per_rep))
        )
        assert entry["controller_mean_reward"] == pytest.approx(means[0])
        assert entry["best_candidate_index"] == int(np.argmax(means))
        assert entry["best_mean_reward"] == pytest.approx(max(means))
        assert entry["oracle_regret"] == pytest.approx(max(means) - means[0])
        assert entry["candidate_reward_spread"] == pytest.approx(
            max(means) - min(means)
        )


@pytest.fixture(scope="module")
def tiny_horizon_planning(tiny_planning_counterfactual: dict[str, Any]) -> dict[str, Any]:
    """Tiny ``evaluate-horizon`` run with ``--controller-type planning``."""
    root = tiny_planning_counterfactual["root"]
    payload = cfa.run_evaluate_horizon(
        cfa.parse_args(
            _horizon_argv(
                out_dir=root / "horizon_planning",
                snapshots_dir=root / "snapshots",
                root=root,
                controller=tiny_planning_counterfactual["planner_path"],
                controller_type="planning",
                predictor=tiny_planning_counterfactual["predictor_path"],
            )
        )
    )
    return {"root": root, "payload": payload}


def test_planning_horizon_reports_predicted_vs_realized_ranking(
    tiny_horizon_planning: dict[str, Any],
) -> None:
    """Predictions are attached where the predictor covers the horizon."""
    payload = tiny_horizon_planning["payload"]
    assert payload["config"]["controller_type"] == "planning"
    assert payload["config"]["predictor"].endswith("predictor.pt")
    assert payload["summary"]["per_horizon"]["2"]["n_states_with_correlation"] <= (
        _TINY_STATES
    )
    for state in payload["states"]:
        assert state["prediction_available"] is True
        # Horizon 2 is one of the predictor's horizons ([1, 2]); 4 is not.
        assert state["per_horizon"]["2"]["prediction_available"] is True
        assert state["per_horizon"]["4"]["prediction_available"] is False
        assert state["per_horizon"]["4"]["spearman_predicted_vs_realized"] is None
        assert state["per_horizon"]["4"]["kendall_predicted_vs_realized"] is None
        assert state["per_horizon"]["2"]["n_candidates"] == (
            2 + _TINY_HORIZON_ALTERNATIVES
        )
        for key in _TINY_HORIZONS:
            entry = state["per_horizon"][key]
            spearman = entry["spearman_predicted_vs_realized"]
            kendall = entry["kendall_predicted_vs_realized"]
            assert spearman is None or -1.0 <= spearman <= 1.0
            assert kendall is None or -1.0 <= kendall <= 1.0
            assert (spearman is None) == (kendall is None)
        for candidate in state["candidates"]:
            predicted = candidate["predicted_reward"]
            assert set(predicted) == {"2"}
            expected_kind = {0: "controller", 1: "default"}.get(
                candidate["index"], "alternative"
            )
            assert candidate["kind"] == expected_kind


def test_aggregate_horizon_merges_bootstraps_and_thirds(
    tiny_horizon_planning: dict[str, Any],
) -> None:
    """Aggregate-horizon pools per-horizon stats, CI, thirds; deterministic."""
    root = tiny_horizon_planning["root"]
    results_dir = root / "horizon_planning"
    args = cfa.parse_args(
        ["aggregate-horizon", "--problems", "zdt1", "--results-dir", str(results_dir)]
    )
    payload_a = cfa.run_aggregate_horizon(args)
    payload_b = cfa.run_aggregate_horizon(args)
    assert payload_a == payload_b
    out_path = results_dir / "counterfactual_horizon.json"
    assert out_path.is_file()

    assert payload_a["config"]["horizons"] == [2, 4]
    assert set(payload_a["horizons"]) == set(_TINY_HORIZONS)
    for entry in payload_a["horizons"].values():
        assert entry["n_states"] == _TINY_STATES
        lo, hi = entry["bootstrap_ci_95"]
        assert lo <= entry["mean_percentile_rank"] <= hi
        assert entry["mean_oracle_regret"] >= 0.0
        thirds = entry["generation_thirds"]
        assert set(thirds) == {"early", "mid", "late"}
        # Gens 2 and 6 of 8 -> 0.25 (early) and 0.75 (late); mid is empty.
        assert thirds["early"]["n_states"] == 1
        assert thirds["mid"]["n_states"] == 0
        assert thirds["mid"]["mean_percentile_rank"] is None
        assert thirds["late"]["n_states"] == 1
    assert payload_a["per_problem"]["zdt1"]["2"]["n_states"] == _TINY_STATES
