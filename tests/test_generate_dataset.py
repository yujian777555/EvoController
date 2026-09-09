"""End-to-end tests for the Phase-0/1 dataset generation script.

Runs ``experiments.generate_dataset.main`` with small settings (one problem,
one seed, 20 generations, pop 40) into a temporary directory and validates
the produced trajectory JSON and the invocation index, for the fixed policy
(Phase-0 baseline), the random action policy (Phase-1), and the full
operator + exploration action space (Phase-1.5).

Settings note: with pop 20 / 5 generations the ZDT1 population never reaches
f2 < 1.1, so hypervolume against ref point (1.1, 1.1) stays exactly 0.0 and
the "NSGA-II improves HV" sanity check cannot hold. Pop 40 / 20 generations
is the smallest setting found that reliably yields a positive final HV
(~1.2 s wall clock).

Phase 1.75 additions under test: the normalized ``mutation_multiplier``
action key (fixed / random / full modes), the inclusive ``--seed-range``
CLI (including its mutual exclusion with ``--seeds``), and the per-problem
``n_runs`` / ``n_success`` / ``n_failed`` / ``zero_hv_count`` counts in
``index.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from algorithms.nsga2 import OperatorConfig
from experiments import generate_dataset

_PROBLEM = "zdt1"
_SEED = 0
_GENERATIONS = 20
_POP_SIZE = 40


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a tiny dataset once per module and return the output dir."""
    out_dir = tmp_path_factory.mktemp("trajectory")
    generate_dataset.main(
        [
            "--problems", _PROBLEM,
            "--seeds", str(_SEED),
            "--generations", str(_GENERATIONS),
            "--pop-size", str(_POP_SIZE),
            "--out-dir", str(out_dir),
        ]
    )
    return out_dir


@pytest.fixture(scope="module")
def trajectory(dataset: Path) -> dict:
    """Load the single trajectory JSON produced by the tiny run."""
    path = dataset / f"{_PROBLEM}_nsga2_seed{_SEED}.json"
    assert path.exists(), f"trajectory file missing: {path}"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_trajectory_required_keys(trajectory: dict) -> None:
    """The trajectory JSON exposes the Phase-0 top-level schema."""
    for key in ("config", "seed", "runtime_sec", "schema_version", "transitions", "final"):
        assert key in trajectory
    assert trajectory["seed"] == _SEED
    assert trajectory["runtime_sec"] > 0.0


def test_config_fully_determines_run(trajectory: dict) -> None:
    """Config contains problem, algorithm, operators, reference settings, timestamp."""
    config = trajectory["config"]
    assert config["problem"] == _PROBLEM
    assert config["algorithm"] == "nsga2"
    assert config["pop_size"] == _POP_SIZE
    assert config["generations"] == _GENERATIONS
    assert config["ref_point"] == [1.1, 1.1]
    assert config["n_reference_points"] == 200
    assert "timestamp_utc" in config
    defaults = OperatorConfig()
    ops = config["operators"]
    assert ops["crossover_operator"] == defaults.crossover_operator
    assert ops["crossover_prob"] == pytest.approx(defaults.crossover_prob)
    assert ops["mutation_operator"] == defaults.mutation_operator
    assert ops["mutation_probability"] == pytest.approx(1.0 / config["n_vars"])
    assert ops["eta_c"] == pytest.approx(defaults.eta_c)
    assert ops["eta_m"] == pytest.approx(defaults.eta_m)
    assert ops["gaussian_sigma"] == pytest.approx(defaults.gaussian_sigma)
    assert config["policy"] == "fixed"
    assert config["action_space"] == "pm"
    assert config["pm_mult_range"] == pytest.approx([0.5, 5.0])
    assert config["base_mutation_prob"] == pytest.approx(1.0 / config["n_vars"])
    assert config["exploration_ranges"]["polynomial_eta_m"] == pytest.approx([2.0, 50.0])
    assert config["exploration_ranges"]["gaussian_sigma"] == pytest.approx([0.02, 0.3])


def test_transitions_length_and_actions(trajectory: dict) -> None:
    """One transition per generation including generation 0; fixed policy."""
    transitions = trajectory["transitions"]
    assert len(transitions) == _GENERATIONS + 1
    assert [t["generation"] for t in transitions] == list(range(_GENERATIONS + 1))
    expected_p_m = 1.0 / trajectory["config"]["n_vars"]
    for t in transitions:
        # The fixed-policy action is the constant resolved config default.
        # (The algorithm-level action schema is the 3-key Phase-1.5 shape;
        # the recorder persists mutation_operator/mutation_probability and,
        # once its schema stores extra action keys, exploration_strength.)
        assert t["action"]["mutation_operator"] == "polynomial"
        assert t["action"]["mutation_probability"] == pytest.approx(expected_p_m)
        if "exploration_strength" in t["action"]:
            assert t["action"]["exploration_strength"] == pytest.approx(
                OperatorConfig().eta_m
            )


def test_generation_zero_reward_is_zero(trajectory: dict) -> None:
    """By convention the first recorded generation carries zero reward."""
    reward = trajectory["transitions"][0]["reward"]
    assert reward["delta_hv"] == 0.0
    assert reward["delta_igd"] == 0.0


def test_nsga2_improves_hypervolume(trajectory: dict) -> None:
    """Sanity check: final hypervolume exceeds the initial one."""
    transitions = trajectory["transitions"]
    hv0 = transitions[0]["state"]["hv"]
    hv_final = transitions[-1]["state"]["hv"]
    assert hv_final > hv0
    assert trajectory["final"]["hv"] == pytest.approx(hv_final)


def test_index_json(dataset: Path) -> None:
    """index.json summarizes the invocation with one entry per run."""
    index_path = dataset / "index.json"
    assert index_path.exists()
    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)
    assert index["n_runs"] == 1
    assert "created_utc" in index
    (run,) = index["runs"]
    for key in ("file", "problem", "seed", "final_hv", "final_igd", "runtime_sec"):
        assert key in run
    assert run["file"] == f"{_PROBLEM}_nsga2_seed{_SEED}.json"
    assert run["problem"] == _PROBLEM
    assert run["seed"] == _SEED
    assert (dataset / run["file"]).exists()


_RANDOM_SEED = 3


def _run_random(out_dir: Path, seed: int) -> None:
    """Run the random policy with small settings into ``out_dir``."""
    generate_dataset.main(
        [
            "--problems", _PROBLEM,
            "--seeds", str(seed),
            "--generations", str(_GENERATIONS),
            "--pop-size", str(_POP_SIZE),
            "--policy", "random",
            "--out-dir", str(out_dir),
        ]
    )


def _load_trajectory(out_dir: Path, seed: int) -> dict:
    """Load the trajectory JSON of a single-run output directory."""
    path = out_dir / f"{_PROBLEM}_nsga2_seed{seed}.json"
    assert path.exists(), f"trajectory file missing: {path}"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def random_trajectory(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Generate one tiny random-policy trajectory and return it."""
    out_dir = tmp_path_factory.mktemp("trajectory_random")
    _run_random(out_dir, _RANDOM_SEED)
    return _load_trajectory(out_dir, _RANDOM_SEED)


def test_random_policy_varies_mutation_probability(random_trajectory: dict) -> None:
    """Random policy: mutation_probability varies across generations."""
    transitions = random_trajectory["transitions"]
    assert len(transitions) == _GENERATIONS + 1
    pms = np.array([t["action"]["mutation_probability"] for t in transitions])
    assert float(np.std(pms)) > 0.0
    # Generation 0 is recorded before any step: resolved default action.
    assert pms[0] == pytest.approx(1.0 / random_trajectory["config"]["n_vars"])
    assert np.all(pms > 0.0) and np.all(pms <= 1.0)


def test_random_policy_config_keys(random_trajectory: dict) -> None:
    """Random-policy config records policy, action space, range, base pm."""
    config = random_trajectory["config"]
    assert config["policy"] == "random"
    assert config["action_space"] == "pm"
    assert config["pm_mult_range"] == pytest.approx([0.5, 5.0])
    base_pm = 1.0 / config["n_vars"]
    assert config["base_mutation_prob"] == pytest.approx(base_pm)
    # The operator default stays resolved to 1/n_vars (the sampling base).
    assert config["operators"]["mutation_probability"] == pytest.approx(base_pm)
    # Sampled probabilities lie within the log-uniform multiplier band.
    pms = np.array(
        [t["action"]["mutation_probability"] for t in random_trajectory["transitions"][1:]]
    )
    assert np.all(pms >= base_pm * 0.5)
    assert np.all(pms <= base_pm * 5.0)


def test_random_policy_reproducible(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Same seed + same policy -> identical action sequences and states."""
    dir_a = tmp_path_factory.mktemp("repro_a")
    dir_b = tmp_path_factory.mktemp("repro_b")
    _run_random(dir_a, _RANDOM_SEED)
    _run_random(dir_b, _RANDOM_SEED)
    traj_a = _load_trajectory(dir_a, _RANDOM_SEED)
    traj_b = _load_trajectory(dir_b, _RANDOM_SEED)
    pms_a = [t["action"]["mutation_probability"] for t in traj_a["transitions"]]
    pms_b = [t["action"]["mutation_probability"] for t in traj_b["transitions"]]
    assert pms_a == pms_b
    states_a = [t["state"] for t in traj_a["transitions"]]
    states_b = [t["state"] for t in traj_b["transitions"]]
    assert states_a == states_b


def test_random_policy_different_seed_differs(random_trajectory: dict, tmp_path: Path) -> None:
    """A different seed produces a different action sequence."""
    _run_random(tmp_path, _RANDOM_SEED + 1)
    other = _load_trajectory(tmp_path, _RANDOM_SEED + 1)
    pms_a = [t["action"]["mutation_probability"] for t in random_trajectory["transitions"]]
    pms_b = [t["action"]["mutation_probability"] for t in other["transitions"]]
    assert pms_a != pms_b


# --- Phase 1.5: full action space (operator + exploration strength) ---

_FULL_SEED = 20
_FULL_GENERATIONS = 5


def _run_full(out_dir: Path, seed: int) -> None:
    """Run the full action space with tiny settings into ``out_dir``."""
    generate_dataset.main(
        [
            "--problems", _PROBLEM,
            "--seeds", str(seed),
            "--generations", str(_FULL_GENERATIONS),
            "--pop-size", str(_POP_SIZE),
            "--action-space", "full",
            "--out-dir", str(out_dir),
        ]
    )


@pytest.fixture(scope="module")
def full_trajectory(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Generate one tiny full-action-space trajectory and return it."""
    out_dir = tmp_path_factory.mktemp("trajectory_full")
    _run_full(out_dir, _FULL_SEED)
    return _load_trajectory(out_dir, _FULL_SEED)


def test_full_action_space_operators_and_pm(full_trajectory: dict) -> None:
    """Full space: operators come from {polynomial, gaussian} and pm varies."""
    transitions = full_trajectory["transitions"]
    assert len(transitions) == _FULL_GENERATIONS + 1
    operators = {t["action"]["mutation_operator"] for t in transitions}
    assert operators <= {"polynomial", "gaussian"}
    pms = np.array([t["action"]["mutation_probability"] for t in transitions])
    assert float(np.std(pms)) > 0.0
    # Generation 0 is recorded before any step: resolved default action.
    base_pm = 1.0 / full_trajectory["config"]["n_vars"]
    assert pms[0] == pytest.approx(base_pm)
    # Sampled probabilities lie within the widened multiplier band.
    sampled = pms[1:]
    assert np.all(sampled >= base_pm * 0.25)
    assert np.all(sampled <= base_pm * 8.0)


def test_full_action_space_config_keys(full_trajectory: dict) -> None:
    """Full mode records the action space and the widened pm range."""
    config = full_trajectory["config"]
    assert config["action_space"] == "full"
    assert config["pm_mult_range"] == pytest.approx([0.25, 8.0])
    assert config["operators"]["gaussian_sigma"] == pytest.approx(
        OperatorConfig().gaussian_sigma
    )
    assert config["exploration_ranges"]["polynomial_eta_m"] == pytest.approx([2.0, 50.0])
    assert config["exploration_ranges"]["gaussian_sigma"] == pytest.approx([0.02, 0.3])


def test_full_action_space_reproducible(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Same seed + full action space -> identical action sequences and states."""
    dir_a = tmp_path_factory.mktemp("full_repro_a")
    dir_b = tmp_path_factory.mktemp("full_repro_b")
    _run_full(dir_a, _FULL_SEED)
    _run_full(dir_b, _FULL_SEED)
    traj_a = _load_trajectory(dir_a, _FULL_SEED)
    traj_b = _load_trajectory(dir_b, _FULL_SEED)
    actions_a = [t["action"] for t in traj_a["transitions"]]
    actions_b = [t["action"] for t in traj_b["transitions"]]
    assert actions_a == actions_b
    states_a = [t["state"] for t in traj_a["transitions"]]
    states_b = [t["state"] for t in traj_b["transitions"]]
    assert states_a == states_b


# --- Phase 1.75: mutation multiplier, --seed-range, per-problem index counts ---


def test_fixed_policy_records_mutation_multiplier(trajectory: dict) -> None:
    """Fixed policy: every action carries multiplier == pm * n_vars == 1.0."""
    n_vars = trajectory["config"]["n_vars"]
    for t in trajectory["transitions"]:
        action = t["action"]
        assert "mutation_multiplier" in action
        assert action["mutation_multiplier"] == pytest.approx(
            action["mutation_probability"] * n_vars
        )
        # The fixed policy always uses the resolved default 1 / n_vars.
        assert action["mutation_multiplier"] == pytest.approx(1.0)


def test_random_policy_records_mutation_multiplier(random_trajectory: dict) -> None:
    """Random policy: multiplier == pm * n_vars and lies in [0.5, 5.0]."""
    n_vars = random_trajectory["config"]["n_vars"]
    transitions = random_trajectory["transitions"]
    # Generation 0 is recorded before any step: default action, multiplier 1.0.
    assert transitions[0]["action"]["mutation_multiplier"] == pytest.approx(1.0)
    multipliers = np.array([t["action"]["mutation_multiplier"] for t in transitions])
    pms = np.array([t["action"]["mutation_probability"] for t in transitions])
    assert multipliers == pytest.approx(pms * n_vars)
    assert float(np.std(multipliers)) > 0.0
    sampled = multipliers[1:]
    assert np.all(sampled >= 0.5)
    assert np.all(sampled <= 5.0)


def test_full_action_space_records_mutation_multiplier(full_trajectory: dict) -> None:
    """Full space: multiplier == pm * n_vars and lies in [0.25, 8.0]."""
    n_vars = full_trajectory["config"]["n_vars"]
    transitions = full_trajectory["transitions"]
    assert transitions[0]["action"]["mutation_multiplier"] == pytest.approx(1.0)
    multipliers = np.array([t["action"]["mutation_multiplier"] for t in transitions])
    pms = np.array([t["action"]["mutation_probability"] for t in transitions])
    assert multipliers == pytest.approx(pms * n_vars)
    sampled = multipliers[1:]
    assert np.all(sampled >= 0.25)
    assert np.all(sampled <= 8.0)


def test_seed_range_expands_to_inclusive_seeds() -> None:
    """--seed-range START STOP expands to the inclusive seed list START..STOP."""
    args = generate_dataset.parse_args(["--seed-range", "200", "204"])
    assert args.seeds == [200, 201, 202, 203, 204]
    args = generate_dataset.parse_args(["--seed-range", "7", "7"])
    assert args.seeds == [7]


def test_seeds_default_unchanged() -> None:
    """Without --seeds/--seed-range the default seed grid is preserved."""
    args = generate_dataset.parse_args([])
    assert args.seeds == list(generate_dataset.DEFAULT_SEEDS)
    assert args.seed_range is None


def test_seed_range_mutually_exclusive_with_seeds() -> None:
    """Passing both --seeds and --seed-range is a CLI error."""
    with pytest.raises(SystemExit):
        generate_dataset.parse_args(["--seeds", "1", "2", "--seed-range", "0", "3"])


def test_seed_range_rejects_reversed_range() -> None:
    """--seed-range requires START <= STOP."""
    with pytest.raises(SystemExit):
        generate_dataset.parse_args(["--seed-range", "5", "2"])


def test_seed_range_end_to_end(tmp_path: Path) -> None:
    """--seed-range drives a full generation run (2 seeds -> 2 files)."""
    summaries = generate_dataset.main(
        [
            "--problems", _PROBLEM,
            "--seed-range", "0", "1",
            "--generations", str(_GENERATIONS),
            "--pop-size", str(_POP_SIZE),
            "--out-dir", str(tmp_path),
        ]
    )
    assert [s["seed"] for s in summaries] == [0, 1]
    for summary in summaries:
        assert (tmp_path / summary["file"]).exists()
    with (tmp_path / "index.json").open(encoding="utf-8") as fh:
        index = json.load(fh)
    assert index["n_runs"] == 2
    stats = index["per_problem"][_PROBLEM]
    assert stats["n_runs"] == 2
    assert stats["n_success"] == 2
    assert stats["n_failed"] == 0
    assert stats["zero_hv_count"] == 0


def test_index_per_problem_counts(dataset: Path) -> None:
    """index.json reports per-problem success/failure (zero-HV) counts."""
    with (dataset / "index.json").open(encoding="utf-8") as fh:
        index = json.load(fh)
    assert index["per_problem"] == {
        _PROBLEM: {"n_runs": 1, "n_success": 1, "n_failed": 0, "zero_hv_count": 0}
    }


def test_index_per_problem_counts_zero_hv(tmp_path: Path) -> None:
    """A run whose final HV is exactly 0 is counted as failed/zero-HV.

    Pop 20 / 5 generations on ZDT1 never reaches f2 < 1.1, so the final
    hypervolume against ref point (1.1, 1.1) stays exactly 0.0 (see the
    module docstring).
    """
    summaries = generate_dataset.main(
        [
            "--problems", _PROBLEM,
            "--seeds", str(_SEED),
            "--generations", "5",
            "--pop-size", "20",
            "--out-dir", str(tmp_path),
        ]
    )
    assert summaries[0]["final_hv"] == 0.0
    with (tmp_path / "index.json").open(encoding="utf-8") as fh:
        index = json.load(fh)
    stats = index["per_problem"][_PROBLEM]
    assert stats == {"n_runs": 1, "n_success": 0, "n_failed": 1, "zero_hv_count": 1}
