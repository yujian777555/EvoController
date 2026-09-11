from __future__ import annotations

"""Tests for the Phase-2B planning-controller evaluation harness.

Covers the tiny end-to-end pipeline (eval -> aggregate) with a synthetic
trained outcome predictor and a fabricated Phase-1.75 results.json: the
per-run JSON schema, the merged ``results.json`` schema, the paired
``comparison.json`` with all Phase-1.75 arms present, and the guards of
the Wilcoxon helper.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller.outcome_predictor import OutcomePredictor
from controller.state_encoder import StateEncoder
from experiments.run_phase1_75 import ARMS as PHASE1_75_ARMS
from experiments.run_phase2b import (
    ARM_PLANNING,
    _wilcoxon_greater,
    parse_args,
    run_experiment,
)


def _make_trajectory(seed: int, *, n_gens: int = 6) -> list[dict[str, Any]]:
    """Fabricate one deterministic synthetic transition list."""
    rng = np.random.default_rng(seed)
    transitions: list[dict[str, Any]] = []
    hv = 0.5
    igd = 0.3
    for generation in range(n_gens + 1):
        if generation > 0:
            hv += float(rng.random()) * 0.01
            igd = max(igd - float(rng.random()) * 0.005, 1e-6)
        transitions.append(
            {
                "generation": generation,
                "state": {
                    "generation": generation,
                    "hv": hv,
                    "igd": igd,
                    "diversity": float(rng.random()),
                },
                "action": {
                    "mutation_operator": "polynomial",
                    "mutation_probability": 1.0 / 30.0,
                    "exploration_strength": 20.0,
                },
                "reward": {"delta_hv": 0.0, "delta_igd": 0.0},
            }
        )
    return transitions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _make_predictor_dir(predictor_dir: Path, *, window: int = 3) -> None:
    """Train and save a tiny outcome predictor plus its state encoder."""
    trajectories = [_make_trajectory(seed) for seed in range(4)]
    encoder = StateEncoder(window).fit(trajectories)
    rng = np.random.default_rng(0)
    input_dim = encoder.dim + 4
    X = rng.normal(size=(64, input_dim))
    y = 0.5 + 0.1 * X[:, :4] + rng.normal(scale=0.01, size=(64, 4))
    predictor = OutcomePredictor(
        input_dim=input_dim, horizons=[1, 5, 10, 20], hidden_dims=(16,), seed=0
    )
    predictor.fit(X, y, epochs=3, batch_size=16)
    predictor_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(predictor_dir / "predictor.pt")
    encoder.save(predictor_dir / "encoder.json")
    _write_json(
        predictor_dir / "training_meta.json",
        {"input_dim": int(input_dim), "horizons": [1, 5, 10, 20], "window": int(window)},
    )


def _make_phase1_75_results(path: Path, *, problem: str = "zdt1", seed: int = 1000) -> None:
    """Fabricate a minimal Phase-1.75 results.json with all nine arms."""
    rng = np.random.default_rng(1)
    runs = {
        f"{arm}|{problem}|{seed}": {
            "final_hv": float(0.7 + rng.random() * 0.1),
            "final_igd": float(0.02 + rng.random() * 0.01),
            "auc_hv": float(0.5 + rng.random() * 0.1),
            "runtime_sec": 1.0,
            "failed": False,
            "trajectory_file": f"runs/{arm}__{problem}__seed{seed}.json",
        }
        for arm in PHASE1_75_ARMS
    }
    _write_json(
        path,
        {
            "config": {"phase": 1.75},
            "failure_thresholds": {"thresholds": {problem: 0.0}},
            "arms": list(PHASE1_75_ARMS),
            "problems": [problem],
            "eval_seeds": [seed],
            "runs": runs,
        },
    )


@pytest.fixture(scope="module")
def mini_experiment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the tiny end-to-end Phase-2B pipeline once for all assertions."""
    tmp = tmp_path_factory.mktemp("phase2b")
    predictor_dir = tmp / "predictor"
    _make_predictor_dir(predictor_dir)
    phase1_75_path = tmp / "phase1_75_results.json"
    _make_phase1_75_results(phase1_75_path)
    out_dir = tmp / "out"
    args = parse_args(
        [
            "--stage", "all",
            "--predictor-dir", str(predictor_dir),
            "--phase1-75-results", str(phase1_75_path),
            "--out-dir", str(out_dir),
            "--problems", "zdt1",
            "--seeds", "1000",
            "--generations", "5",
            "--pop-size", "20",
            "--n-candidates", "4",
        ]
    )
    results = run_experiment(args)
    return {"results": results, "out_dir": out_dir}


def test_run_json_schema(mini_experiment: dict[str, Any]) -> None:
    out_dir = mini_experiment["out_dir"]
    run_path = out_dir / "runs" / f"{ARM_PLANNING}__zdt1__seed1000.json"
    assert run_path.is_file()
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    config = payload["config"]
    assert config["arm"] == ARM_PLANNING
    assert config["problem"] == "zdt1"
    assert config["seed"] == 1000
    assert config["controller"]["n_candidates"] == 4
    assert config["pm_absolute_range"] == pytest.approx([0.25 / 30, 8.0 / 30])
    assert set(payload["metrics"]) == {
        "final_hv",
        "final_igd",
        "auc_hv",
        "runtime_sec",
        "failed",
    }
    # Generation 0 plus five stepped generations, each with a used action.
    assert len(payload["transitions"]) == 6
    for transition in payload["transitions"]:
        assert set(transition["action"]) >= {
            "mutation_operator",
            "mutation_probability",
            "exploration_strength",
        }
    assert payload["metrics"]["failed"] is False


def test_results_json_schema(mini_experiment: dict[str, Any]) -> None:
    results = mini_experiment["results"]
    assert set(results) == {"config", "runs", "failure_thresholds"}
    key = f"{ARM_PLANNING}|zdt1|1000"
    assert key in results["runs"]
    run = results["runs"][key]
    assert set(run) == {
        "final_hv",
        "final_igd",
        "auc_hv",
        "runtime_sec",
        "failed",
        "trajectory_file",
    }
    assert run["trajectory_file"] == f"runs/{ARM_PLANNING}__zdt1__seed1000.json"
    assert 0.0 <= run["final_hv"] <= 1.21
    assert 0.0 <= run["auc_hv"] <= 1.21
    config = results["config"]
    assert config["arm"] == ARM_PLANNING
    assert config["eval_seeds"] == [1000]
    assert results["failure_thresholds"]["thresholds"]["zdt1"] == 0.0
    results_path = mini_experiment["out_dir"] / "results.json"
    assert results_path.is_file()


def test_comparison_json_all_arms(mini_experiment: dict[str, Any]) -> None:
    comparison_path = mini_experiment["out_dir"] / "comparison.json"
    assert comparison_path.is_file()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    report = comparison["problems"]["zdt1"]
    planning = report[ARM_PLANNING]
    assert planning["n_runs"] == 1
    assert planning["n_failed"] == 0
    for metric in ("final_hv", "auc_hv"):
        assert set(planning[metric]) == {"mean", "std"}
    comparisons = report["comparisons"]
    for arm in PHASE1_75_ARMS:
        assert arm in comparisons, arm
        entry = comparisons[arm]
        assert entry["n_paired"] == 1
        for metric in ("final_hv", "auc_hv"):
            assert set(entry[metric]) == {"mean", "std"}
            assert set(entry["wilcoxon"][metric]) == {"statistic", "p_value"}
        # One pair is too few for the signed-rank test: guarded to None.
        assert entry["wilcoxon"]["final_hv"]["p_value"] is None


def test_missing_predictor_dir_raises(tmp_path: Path) -> None:
    phase1_75_path = tmp_path / "phase1_75.json"
    _make_phase1_75_results(phase1_75_path)
    args = parse_args(
        [
            "--stage", "eval",
            "--predictor-dir", str(tmp_path / "nope"),
            "--phase1-75-results", str(phase1_75_path),
            "--out-dir", str(tmp_path / "out"),
            "--problems", "zdt1",
            "--seeds", "1000",
            "--generations", "2",
            "--pop-size", "20",
        ]
    )
    with pytest.raises(FileNotFoundError, match="predictor artifact"):
        run_experiment(args)


def test_wilcoxon_greater_guards() -> None:
    # Fewer than two pairs and all-zero differences are undefined.
    assert _wilcoxon_greater([1.0], [0.5])["p_value"] is None
    assert _wilcoxon_greater([1.0, 1.0], [1.0, 1.0])["p_value"] is None
    # A consistent positive shift must give a small one-sided p-value.
    rng = np.random.default_rng(2)
    baseline = 0.8 + rng.normal(scale=0.01, size=20)
    planning = baseline + 0.05
    result = _wilcoxon_greater(planning, baseline)
    assert result["p_value"] is not None
    assert result["p_value"] < 0.001
    reverse = _wilcoxon_greater(baseline, planning)
    assert reverse["p_value"] is not None
    assert reverse["p_value"] > 0.999
