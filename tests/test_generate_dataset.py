"""End-to-end tests for the Phase-0/1 dataset generation script.

Runs ``experiments.generate_dataset.main`` with small settings (one problem,
one seed, 20 generations, pop 40) into a temporary directory and validates
the produced trajectory JSON and the invocation index, for both the fixed
policy (Phase-0 baseline) and the random action policy (Phase-1).

Settings note: with pop 20 / 5 generations the ZDT1 population never reaches
f2 < 1.1, so hypervolume against ref point (1.1, 1.1) stays exactly 0.0 and
the "NSGA-II improves HV" sanity check cannot hold. Pop 40 / 20 generations
is the smallest setting found that reliably yields a positive final HV
(~1.2 s wall clock).
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
    assert config["policy"] == "fixed"
    assert config["pm_mult_range"] == pytest.approx([0.5, 5.0])
    assert config["base_mutation_prob"] == pytest.approx(1.0 / config["n_vars"])


def test_transitions_length_and_actions(trajectory: dict) -> None:
    """One transition per generation including generation 0; fixed policy."""
    transitions = trajectory["transitions"]
    assert len(transitions) == _GENERATIONS + 1
    assert [t["generation"] for t in transitions] == list(range(_GENERATIONS + 1))
    expected_p_m = 1.0 / trajectory["config"]["n_vars"]
    for t in transitions:
        assert t["action"]["mutation_operator"] == "polynomial"
        assert t["action"]["mutation_probability"] == pytest.approx(expected_p_m)


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
    """Random-policy config records policy, sampling range, and base pm."""
    config = random_trajectory["config"]
    assert config["policy"] == "random"
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
