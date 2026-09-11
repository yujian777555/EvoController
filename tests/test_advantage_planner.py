from __future__ import annotations

"""Tests for the Phase-2.75 advantage planner, trainer and evaluation harness.

Everything is synthetic: a stub advantage predictor drives the controller
tests, a hand-written intervention ``.npz`` drives the trainer, and a tiny
in-memory model drives the micro end-to-end evaluation. No trained artefact,
Phase-1.75 controller or real dataset is required.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller import advantage_planner_controller as apc
from controller.advantage_planner_controller import AdvantagePlannerController
from controller.advantage_predictor import AdvantagePredictor, ranking_metrics
from controller.dataset import merge_state_reward
from controller.macro_actions import MACRO_ACTIONS
from controller.planning_controller import PlanningController
from controller.state_encoder import StateEncoder
from experiments import run_phase2_75 as run275
from experiments import train_advantage_predictor as tap

_WINDOW = 3
_N_GENERATIONS = 12
_N_VARS = 30


# --- shared synthetic data ---------------------------------------------------


def _trajectories(n_traj: int = 2) -> list[list[dict[str, Any]]]:
    """Two recorder-schema trajectories used to fit the tiny encoder."""
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
                        "mutation_operator": "polynomial",
                        "mutation_probability": 1.0 / _N_VARS,
                        "exploration_strength": 20.0,
                    },
                    "reward": {"delta_hv": delta_hv, "delta_igd": delta_igd},
                }
            )
        trajectories.append(transitions)
    return trajectories


def _fitted_encoder() -> StateEncoder:
    return StateEncoder(_WINDOW).fit(_trajectories())


def _history(t: int = 4) -> list[dict[str, Any]]:
    return [merge_state_reward(tr) for tr in _trajectories()[0][: t + 1]]


class _StubPredictor:
    """Advantage predictor stub with a caller-fixed response matrix."""

    def __init__(
        self, input_dim: int, horizons: tuple[int, ...], values: np.ndarray | None = None
    ) -> None:
        self._input_dim = int(input_dim)
        self._horizons = tuple(int(h) for h in horizons)
        self._values = values
        self.calls: list[np.ndarray] = []

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def horizons(self) -> tuple[int, ...]:
        return self._horizons

    def predict(self, X: np.ndarray) -> np.ndarray:
        rows = np.asarray(X, dtype=np.float64)
        self.calls.append(rows.copy())
        if self._values is not None:
            return np.asarray(self._values, dtype=np.float64)
        return np.zeros((rows.shape[0], len(self._horizons)), dtype=np.float64)


# --- AdvantagePlannerController ---------------------------------------------


def test_action_contract_is_identical_to_phase2b() -> None:
    """Three keys, deterministic, and the Phase-2B action schema."""
    encoder = _fitted_encoder()
    predictor = _StubPredictor(encoder.dim + 4, (5, 10, 20))
    planner = AdvantagePlannerController(predictor, n_candidates=8, candidate_seed=3)
    history = _history()
    first = planner.predict_action(history, encoder, 0.0, 1.0, n_vars=_N_VARS)
    second = planner.predict_action(history, encoder, 0.0, 1.0, n_vars=_N_VARS)
    assert first == second
    assert set(first) == {
        "mutation_operator",
        "mutation_probability",
        "exploration_strength",
    }
    assert first["mutation_operator"] in ("polynomial", "gaussian")
    lo, hi = planner.pm_mult_range
    assert lo / _N_VARS <= first["mutation_probability"] <= hi / _N_VARS
    # the legacy n_vars fallback keeps the probability in [0, 1]
    fallback = planner.predict_action(history, encoder, 0.0, 1.0)
    assert set(fallback) == set(first)


def test_candidates_and_feature_rows_match_planning_controller() -> None:
    """Sampling and the feature block are bit-identical to Phase 2B's planner.

    The advantage planner delegates candidate sampling to an internal
    ``PlanningController``; this test pins that the sampled candidates and the
    exact feature rows handed to the predictor are the same as Phase 2B's for
    the same history, seed and candidate count.
    """
    encoder = _fitted_encoder()
    history = _history()
    seed, n_candidates = 7, 12

    advantage_spy = _StubPredictor(encoder.dim + 4, (5, 10, 20))
    advantage_planner = AdvantagePlannerController(
        advantage_spy, n_candidates=n_candidates, candidate_seed=seed
    )
    _action, diagnostics = advantage_planner.predict_action_ex(
        history, encoder, 0.0, 1.0, n_vars=_N_VARS
    )

    phase2b_spy = _StubPredictor(encoder.dim + 4, (5, 10, 20))
    phase2b_planner = PlanningController(
        phase2b_spy,
        n_candidates=n_candidates,
        candidate_seed=seed,
        horizon_weights=[1.0, 1.0, 1.0],
    )
    _phase2b_action, phase2b_diagnostics = phase2b_planner.predict_action_ex(
        history, encoder, 0.0, 1.0, n_vars=_N_VARS
    )

    assert diagnostics["candidates"] == phase2b_diagnostics["candidates"]
    assert len(diagnostics["candidates"]) == n_candidates
    np.testing.assert_array_equal(advantage_spy.calls[0], phase2b_spy.calls[0])
    # ... and the canonical layout is state block + 4 action features
    state_block = np.asarray(encoder.transform(history), dtype=np.float64)
    rows = advantage_spy.calls[0]
    assert rows.shape == (n_candidates, state_block.size + 4)
    np.testing.assert_array_equal(rows[:, : state_block.size], np.tile(state_block, (n_candidates, 1)))
    for row, candidate in zip(rows, diagnostics["candidates"]):
        assert row[-4] == pytest.approx(candidate["mutation_multiplier"])
        assert row[-3] == pytest.approx(candidate["exploration_strength"])
        assert row[-2] + row[-1] == pytest.approx(1.0)
        assert (row[-2] == 1.0) == (candidate["mutation_operator"] == "polynomial")


def test_selects_the_candidate_with_the_largest_mean_advantage() -> None:
    """The argmax objective is the mean predicted advantage over horizons."""
    encoder = _fitted_encoder()
    n_candidates = 6
    # candidate 3 has the largest mean advantage
    values = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [-0.1, 0.0, 0.1],
            [0.5, 0.0, 0.0],
            [0.4, 0.5, 0.6],
            [0.2, 0.2, 0.2],
            [0.0, 0.0, 0.0],
        ]
    )
    predictor = _StubPredictor(encoder.dim + 4, (5, 10, 20), values=values)
    planner = AdvantagePlannerController(
        predictor, n_candidates=n_candidates, candidate_seed=0
    )
    action, diagnostics = planner.predict_action_ex(
        _history(), encoder, 0.0, 1.0, n_vars=_N_VARS
    )
    expected_index = int(np.argmax(values.mean(axis=1)))
    assert diagnostics["selected_index"] == expected_index
    assert diagnostics["scores"] == pytest.approx(values.mean(axis=1).tolist())
    assert diagnostics["predicted_hv"] == values.tolist()
    selected = diagnostics["candidates"][expected_index]
    assert action == {
        "mutation_operator": selected["mutation_operator"],
        "mutation_probability": selected["mutation_probability"],
        "exploration_strength": selected["exploration_strength"],
    }
    ordered = np.sort(values.mean(axis=1))
    assert diagnostics["score_margin"] == pytest.approx(ordered[-1] - ordered[-2])
    assert diagnostics["score_std"] == pytest.approx(
        float(np.std(values.mean(axis=1)))
    )
    assert set(diagnostics) == {
        "candidates",
        "predicted_hv",
        "scores",
        "selected_index",
        "score_margin",
        "score_std",
    }


def test_macro_action_branch_scores_the_discrete_set() -> None:
    """macro_actions=True plans over controller.macro_actions' table."""
    encoder = _fitted_encoder()
    n_macros = len(MACRO_ACTIONS)
    values = np.zeros((n_macros, 3))
    values[-1] = 1.0  # the last macro of the table wins
    predictor = _StubPredictor(encoder.dim + 4, (5, 10, 20), values=values)
    planner = AdvantagePlannerController(predictor, macro_actions=True)
    assert planner.macro_actions is True
    action, diagnostics = planner.predict_action_ex(
        _history(), encoder, 0.0, 1.0, n_vars=_N_VARS
    )
    assert len(diagnostics["candidates"]) == n_macros
    assert diagnostics["selected_index"] == n_macros - 1
    macro = list(MACRO_ACTIONS.values())[-1]
    assert action["mutation_operator"] == macro["mutation_operator"]
    assert action["mutation_probability"] == pytest.approx(
        float(macro["multiplier"]) / _N_VARS
    )
    assert action["exploration_strength"] == pytest.approx(
        float(macro["exploration_strength"])
    )
    # the macro set lives inside the deployment pm bounds
    lo, hi = planner.pm_mult_range
    for candidate in diagnostics["candidates"]:
        assert lo / _N_VARS <= candidate["mutation_probability"] <= hi / _N_VARS


def test_save_load_round_trip(tmp_path: Path) -> None:
    """save/load restores the config and the predictor_path annotation."""
    encoder = _fitted_encoder()
    predictor = _StubPredictor(encoder.dim + 4, (5, 10, 20))
    planner = AdvantagePlannerController(
        predictor, n_candidates=5, candidate_seed=11, pm_mult_range=(0.5, 4.0)
    )
    planner.predictor_path = "model_contrastive.pt"
    path = tmp_path / "planner.json"
    planner.save(path)
    restored = AdvantagePlannerController.load(path, predictor)
    assert restored.n_candidates == 5
    assert restored.candidate_seed == 11
    assert restored.pm_mult_range == (0.5, 4.0)
    assert restored.macro_actions is False
    assert restored.predictor_path == "model_contrastive.pt"
    history = _history()
    assert restored.predict_action(history, encoder, 0.0, 1.0, n_vars=_N_VARS) == (
        planner.predict_action(history, encoder, 0.0, 1.0, n_vars=_N_VARS)
    )


def test_configuration_and_dimension_errors() -> None:
    """Invalid configurations and mismatched widths fail loudly."""
    encoder = _fitted_encoder()
    predictor = _StubPredictor(encoder.dim + 4, (5, 10, 20))
    with pytest.raises(ValueError, match="n_candidates"):
        AdvantagePlannerController(predictor, n_candidates=0)
    with pytest.raises(ValueError, match="candidate_seed"):
        AdvantagePlannerController(predictor, candidate_seed=-1)
    with pytest.raises(ValueError, match="pm_mult_range"):
        AdvantagePlannerController(predictor, pm_mult_range=(0.0, 1.0))
    with pytest.raises(ValueError, match="input_dim"):
        AdvantagePlannerController(object())
    planner = AdvantagePlannerController(predictor)
    with pytest.raises(ValueError, match="n_vars"):
        planner.predict_action(_history(), encoder, 0.0, 1.0, n_vars=0)
    narrow = AdvantagePlannerController(_StubPredictor(encoder.dim + 9, (5,)), n_candidates=2)
    with pytest.raises(ValueError, match="input_dim"):
        narrow.predict_action(_history(), encoder, 0.0, 1.0, n_vars=_N_VARS)


# --- training script ---------------------------------------------------------


def _write_dataset(
    root: Path,
    *,
    n_states: int = 8,
    n_candidates: int = 4,
    horizons: tuple[int, ...] = (5, 10, 20),
    seed: int = 0,
) -> Path:
    """Write a synthetic intervention npz + meta + encoder and return the dir."""
    rng = np.random.Generator(np.random.PCG64(seed))
    encoder = _fitted_encoder()
    encoder_path = root / "encoder.json"
    encoder.save(encoder_path)
    input_dim = encoder.dim + 4
    rows_x: list[np.ndarray] = []
    rows_y: list[float] = []
    rows_problem: list[str] = []
    rows_seed: list[int] = []
    rows_generation: list[int] = []
    rows_horizon: list[int] = []
    rows_candidate: list[int] = []
    for state in range(n_states):
        problem, run_seed, generation = "zdt1", 1000 + state, 50 + state
        # State blocks are small noise (they must not determine the action
        # effect); the advantage is a clean function of the candidate's
        # multiplier feature, i.e. of the action block of X -- learnable, as a
        # real intervention dataset would be.
        state_block = rng.normal(scale=0.1, size=encoder.dim)
        for candidate in range(n_candidates):
            multiplier = 0.5 + candidate
            x_row = np.concatenate(
                [state_block, [multiplier, 10.0 + candidate, 1.0, 0.0]]
            )
            for horizon in horizons:
                advantage = (candidate - (n_candidates - 1) / 2.0) * (horizon / 10.0)
                rows_x.append(x_row)
                rows_y.append(float(advantage))
                rows_problem.append(problem)
                rows_seed.append(run_seed)
                rows_generation.append(generation)
                rows_horizon.append(int(horizon))
                rows_candidate.append(candidate)
    npz_path = root / "intervention_dataset_zdt1.npz"
    np.savez_compressed(
        npz_path,
        X=np.vstack(rows_x),
        y_adv=np.asarray(rows_y, dtype=np.float64),
        problem=np.asarray(rows_problem, dtype="U16"),
        seed=np.asarray(rows_seed, dtype=np.int64),
        generation=np.asarray(rows_generation, dtype=np.int64),
        horizon=np.asarray(rows_horizon, dtype=np.int64),
        candidate_index=np.asarray(rows_candidate, dtype=np.int64),
    )
    with (root / "intervention_meta.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": {"encoder": str(encoder_path)}}, fh)
    return root


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic intervention dataset directory."""
    return _write_dataset(tmp_path_factory.mktemp("dataset"))


def test_state_level_split_has_no_leakage(dataset_dir: Path) -> None:
    """No state appears on both sides of the split."""
    arrays, files = tap.load_merged_arrays(dataset_dir)
    assert files == ["intervention_dataset_zdt1.npz"]
    blocks, skipped = tap.build_state_blocks(arrays, [5, 10, 20])
    assert skipped == []
    assert len(blocks) == 8
    for block in blocks:
        assert block["X"].shape == (4, arrays["X"].shape[1])
        assert block["y"].shape == (4, 3)
    train, val = tap.split_blocks(blocks, 0.25, seed=0)
    train_keys = {block["key"] for block in train}
    val_keys = {block["key"] for block in val}
    assert train_keys and val_keys
    assert train_keys.isdisjoint(val_keys)
    assert train_keys | val_keys == {block["key"] for block in blocks}
    assert len(val_keys) == 2
    # every row of a state is on exactly one side
    x_train, _ = tap.stack_blocks(train)
    x_val, _ = tap.stack_blocks(val)
    assert x_train.shape[0] == len(train) * 4
    assert x_val.shape[0] == len(val) * 4
    train_rows = {tuple(row) for row in x_train}
    val_rows = {tuple(row) for row in x_val}
    assert train_rows.isdisjoint(val_rows)
    with pytest.raises(ValueError, match="val_fraction"):
        tap.split_blocks(blocks, 1.0, seed=0)


def test_contrastive_pairs_are_inside_the_state(dataset_dir: Path) -> None:
    """Each positive is paired with the next worse candidate of its own state."""
    arrays, _ = tap.load_merged_arrays(dataset_dir)
    blocks, _ = tap.build_state_blocks(arrays, [5, 10, 20])
    x_pos, y_pos, x_neg, y_neg = tap.contrastive_pairs(blocks)
    assert x_pos.shape[0] == len(blocks) * 3  # 4 candidates -> 3 adjacent pairs
    assert x_pos.shape[1] == x_neg.shape[1] == blocks[0]["X"].shape[1]
    assert y_pos.shape == y_neg.shape == (x_pos.shape[0], 3)
    # positives must have a strictly larger mean advantage than their partner
    assert np.all(y_pos.mean(axis=1) > y_neg.mean(axis=1))


def test_training_writes_both_paths(dataset_dir: Path, tmp_path: Path) -> None:
    """contrastive=on trains both models; contrastive=off only the MSE model."""
    out_dir = tmp_path / "model_on"
    meta = tap.main(
        [
            "--dataset-dir", str(dataset_dir),
            "--out-dir", str(out_dir),
            "--epochs", "30",
            "--batch-size", "16",
            "--horizons", "5", "10", "20",
            "--train-seed", "0",
            "--val-fraction", "0.25",
            "--hidden-dims", "16", "16",
        ]
    )
    for name in ("model_mse.pt", "model_contrastive.pt", "encoder.json", "training_meta.json"):
        assert (out_dir / name).is_file(), name
    assert set(meta) >= {"config", "split", "losses", "validation", "artifacts"}
    assert meta["config"]["contrastive"] == "on"
    assert meta["split"]["n_states_train"] == 6
    assert meta["split"]["n_states_val"] == 2
    assert set(meta["split"]["val_state_keys"]).isdisjoint(meta["split"]["train_state_keys"])
    assert len(meta["losses"]["mse"]["train_loss"]) == 30
    assert meta["losses"]["contrastive"] is not None
    assert meta["validation"]["contrastive"]["n_states"] == 2
    for horizon in ("5", "10", "20"):
        entry = meta["validation"]["contrastive"]["per_horizon"][horizon]
        assert set(entry) >= {
            "n_groups",
            "spearman_mean",
            "kendall_mean",
            "oracle_hit_rate",
            "regret_mean",
            "oracle_gap_mean",
        }
    # the synthetic action effect is learnable: both arms rank the held-out
    # states positively, the contrastive one at least as well as the MSE one
    mse_overall = meta["validation"]["mse"]["overall"]
    contrastive_overall = meta["validation"]["contrastive"]["overall"]
    assert mse_overall["spearman_mean"] > 0.5
    assert contrastive_overall["spearman_mean"] > 0.8
    assert contrastive_overall["spearman_mean"] >= mse_overall["spearman_mean"]
    assert contrastive_overall["oracle_hit_rate"] >= mse_overall["oracle_hit_rate"]
    assert meta["losses"]["mse"]["train_loss"][-1] < meta["losses"]["mse"]["train_loss"][0]
    # loaded models reproduce the saved predictor
    loaded = AdvantagePredictor.load(out_dir / "model_contrastive.pt")
    assert loaded.horizons == (5, 10, 20)
    assert loaded.input_dim == meta["config"]["input_dim"]

    out_dir_off = tmp_path / "model_off"
    meta_off = tap.main(
        [
            "--dataset-dir", str(dataset_dir),
            "--out-dir", str(out_dir_off),
            "--epochs", "2",
            "--batch-size", "16",
            "--horizons", "5", "10", "20",
            "--contrastive", "off",
            "--hidden-dims", "8",
        ]
    )
    assert (out_dir_off / "model_mse.pt").is_file()
    assert not (out_dir_off / "model_contrastive.pt").exists()
    assert meta_off["validation"]["contrastive"] is None
    assert meta_off["artifacts"]["model_contrastive"] is None


def test_load_merged_arrays_rejects_missing_or_bad_input(tmp_path: Path) -> None:
    """Missing directory, empty directory and malformed npz are explicit errors."""
    with pytest.raises(FileNotFoundError, match="dataset directory"):
        tap.load_merged_arrays(tmp_path / "absent")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no intervention_dataset"):
        tap.load_merged_arrays(empty)
    broken = tmp_path / "broken"
    broken.mkdir()
    np.savez_compressed(broken / "intervention_dataset_zdt1.npz", X=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="missing arrays"):
        tap.load_merged_arrays(broken)


# --- evaluation harness ------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model_dir(
    tmp_path_factory: pytest.TempPathFactory, dataset_dir: Path
) -> Path:
    """A trained advantage model + encoder, as train_advantage_predictor writes it."""
    out_dir = tmp_path_factory.mktemp("tiny_model")
    tap.main(
        [
            "--dataset-dir", str(dataset_dir),
            "--out-dir", str(out_dir),
            "--epochs", "3",
            "--batch-size", "16",
            "--horizons", "5", "10", "20",
            "--hidden-dims", "8",
        ]
    )
    return out_dir


def test_shard_grid_is_deterministic_and_complete() -> None:
    """Shards partition the flattened grid without gaps or overlaps."""
    arms = ["advantage_planner", "fixed_nsga2"]
    grid = run275.shard_grid(arms, ["zdt1", "zdt2"], [1, 2], 0, 1)
    assert len(grid) == 8
    shards = [
        run275.shard_grid(arms, ["zdt1", "zdt2"], [1, 2], index, 3) for index in range(3)
    ]
    assert sorted(sum(shards, [])) == sorted(grid)
    for shard in shards:
        assert len(shard) == len(set(shard))
    with pytest.raises(ValueError, match="num_shards"):
        run275.shard_grid(arms, ["zdt1"], [1], 0, 0)
    with pytest.raises(ValueError, match="shard must lie"):
        run275.shard_grid(arms, ["zdt1"], [1], 3, 2)


def test_micro_end_to_end_eval_and_aggregate(
    tiny_model_dir: Path, tmp_path: Path
) -> None:
    """1 problem x 1 seed x 5 generations, then aggregate with Holm/CI/failure."""
    out_dir = tmp_path / "eval"
    payload = run275.main(
        [
            "--stage", "all",
            "--model-dir", str(tiny_model_dir),
            "--out-dir", str(out_dir),
            "--problems", "zdt1",
            "--seeds", "1000",
            "--arms", "advantage_planner", "fixed_nsga2",
            "--pop-size", "20",
            "--generations", "5",
            "--n-candidates", "4",
            "--phase1-75-dir", str(tmp_path / "absent_phase175"),
        ]
    )
    results_path = out_dir / "results.json"
    comparison_path = out_dir / "comparison.json"
    assert results_path.is_file() and comparison_path.is_file()
    assert len(payload["runs"]) == 2
    runs = sorted((out_dir / "runs").glob("*.json"))
    assert [path.name for path in runs] == [
        "advantage_planner__zdt1__seed1000.json",
        "fixed_nsga2__zdt1__seed1000.json",
    ]
    with runs[0].open("r", encoding="utf-8") as fh:
        run_payload = json.load(fh)
    assert len(run_payload["transitions"]) == 6  # generation 0 + 5 steps
    assert run_payload["config"]["arm"] == "advantage_planner"
    assert "metrics" in run_payload
    actions = [transition["action"] for transition in run_payload["transitions"]]
    assert all(set(action) == {
        "mutation_operator",
        "mutation_probability",
        "exploration_strength",
    } for action in actions)

    with comparison_path.open("r", encoding="utf-8") as fh:
        comparison = json.load(fh)
    config = comparison["config"]
    assert config["control_arm"] == "advantage_planner"
    assert "fixed_nsga2" in config["comparison_arms"]
    entry = comparison["problems"]["zdt1"]["comparisons"]["fixed_nsga2"]["final_hv"]
    assert entry["n_paired"] == 1
    assert entry["bootstrap_ci_95"] is not None
    assert entry["control_failure_rate"] in (0.0, 1.0)
    assert entry["arm_failure_rate"] in (0.0, 1.0)
    # a single pair cannot define a Wilcoxon test, so the p-value stays None
    assert entry["wilcoxon_p"] is None
    assert entry["holm_p"] is None


def test_load_context_requires_the_model_artifacts(tmp_path: Path) -> None:
    """A missing advantage model is reported before any run starts."""
    args = run275.parse_args(
        [
            "--stage", "eval",
            "--model-dir", str(tmp_path / "absent"),
            "--arms", "advantage_planner",
            "--out-dir", str(tmp_path / "eval"),
        ]
    )
    with pytest.raises(FileNotFoundError, match="advantage predictor"):
        run275.load_context(args, ["zdt1"])
    # fixed_nsga2 needs no artefact at all
    fixed_args = run275.parse_args(["--arms", "fixed_nsga2"])
    assert run275.load_context(fixed_args, ["zdt1"]) == {
        run275.ARM_FIXED: None
    }


def test_parse_args_defaults() -> None:
    """CLI defaults match the documented Phase-2.75 grid."""
    args = run275.parse_args([])
    assert args.stage == "all"
    assert args.model_dir == run275.DEFAULT_MODEL_DIR
    assert args.out_dir == run275.DEFAULT_OUT_DIR
    assert tuple(args.problems) == run275.DEFAULT_PROBLEMS
    assert tuple(args.seeds) == run275.DEFAULT_SEEDS
    assert tuple(args.arms) == run275.ARMS
    assert args.pop_size == 100 and args.generations == 100
    assert args.n_candidates == 16
    assert args.shard == 0 and args.num_shards == 1
    assert args.predictor_file == run275.DEFAULT_PREDICTOR_FILE
    train_args = tap.parse_args([])
    assert train_args.dataset_dir == tap.DEFAULT_DATASET_DIR
    assert train_args.out_dir == tap.DEFAULT_OUT_DIR
    assert train_args.epochs == 300
    assert train_args.contrastive == "on"
    assert tuple(train_args.horizons) == tap.DEFAULT_HORIZONS


def test_ranking_metrics_is_reused_for_validation() -> None:
    """The trainer reports the documented decision-quality metrics."""
    realized = np.asarray([[0.0, 0.5, 1.0], [1.0, 0.0, -1.0]])
    metrics = ranking_metrics(realized.copy(), realized)
    assert metrics["spearman_mean"] == pytest.approx(1.0)
    assert metrics["oracle_hit_rate"] == pytest.approx(1.0)
    assert metrics["regret_mean"] == pytest.approx(0.0)


def test_zero_predictor_stub_is_documented() -> None:
    """The internal sampler stub predicts zeros with the right width."""
    stub = apc._ZeroPredictor(7, (5, 10))
    assert stub.input_dim == 7
    assert stub.horizons == (5, 10)
    assert stub.predict(np.zeros((3, 7))).shape == (3, 2)
