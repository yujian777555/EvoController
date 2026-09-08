"""End-to-end test for the Phase-0 dataset generation script.

Runs ``experiments.generate_dataset.main`` with small settings (one problem,
one seed, 20 generations, pop 40) into a temporary directory and validates
the produced trajectory JSON and the invocation index.

Settings note: with pop 20 / 5 generations the ZDT1 population never reaches
f2 < 1.1, so hypervolume against ref point (1.1, 1.1) stays exactly 0.0 and
the "NSGA-II improves HV" sanity check cannot hold. Pop 40 / 20 generations
is the smallest setting found that reliably yields a positive final HV
(~1.2 s wall clock).
"""

from __future__ import annotations

import json
from pathlib import Path

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
