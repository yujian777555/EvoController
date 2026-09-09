from __future__ import annotations

"""Tests for the Phase-1.75 fair closed-loop evaluation harness.

Covers the tiny end-to-end pipeline (thresholds -> train -> eval ->
aggregate) on a synthetic mini corpus, the exact ``results.json`` schema,
paired-seed pairing across arms, failure-threshold ordering, controller
persistence, the state-scramble proof artifact, and the state-independence
of the generation-only arm.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.run_phase1_75 import (
    ARMS,
    DEFAULT_PM_MULT_RANGE,
    LEARNED_ARMS,
    GenerationOnlyPolicy,
    parse_args,
    run_experiment,
)


def _make_trajectory(
    problem: str,
    seed: int,
    *,
    n_gens: int = 6,
    action_space: str = "full",
    policy: str = "random",
) -> dict[str, Any]:
    """Fabricate one deterministic synthetic trajectory payload."""
    rng = np.random.default_rng(seed)
    transitions: list[dict[str, Any]] = []
    hv = 0.5
    igd = 0.3
    for generation in range(n_gens + 1):
        if generation > 0:
            hv += float(rng.random()) * 0.01
            igd = max(igd - float(rng.random()) * 0.005, 1e-6)
        operator = "polynomial" if rng.random() < 0.5 else "gaussian"
        exploration = (
            float(rng.uniform(2.0, 50.0))
            if operator == "polynomial"
            else float(rng.uniform(0.02, 0.3))
        )
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
                    "mutation_operator": operator,
                    "mutation_probability": float(rng.uniform(0.01, 0.2)),
                    "exploration_strength": exploration,
                },
                "reward": {"delta_hv": 0.0, "delta_igd": 0.0},
            }
        )
    return {
        "config": {
            "problem": problem,
            "n_vars": 30,
            "algorithm": "nsga2",
            "pop_size": 20,
            "generations": n_gens,
            "policy": policy,
            "action_space": action_space,
        },
        "seed": int(seed),
        "runtime_sec": 0.1,
        "schema_version": 1,
        "transitions": transitions,
        "final": {
            "hv": hv,
            "igd": igd,
            "diversity": transitions[-1]["state"]["diversity"],
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


@pytest.fixture(scope="module")
def mini_experiment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the tiny end-to-end pipeline once for all assertions."""
    tmp = tmp_path_factory.mktemp("phase1_75")
    train_dir = tmp / "train"
    # Four full-action trajectories (training corpus) + one pm-only record
    # that the open-loop fit must ignore.
    for seed in range(4):
        _write_json(
            train_dir / f"zdt1_nsga2_seed{seed}.json",
            _make_trajectory("zdt1", seed),
        )
    _write_json(
        train_dir / "zdt1_nsga2_seed90.json",
        _make_trajectory("zdt1", 90, action_space="pm"),
    )
    fixed_dir = tmp / "fixed"
    _write_json(
        fixed_dir / "zdt1_nsga2_seed0.json",
        _make_trajectory("zdt1", 5000, policy="fixed"),
    )
    static_tuning = tmp / "static_full_tuning.json"
    _write_json(
        static_tuning,
        {
            "global": {
                "operator": "polynomial",
                "multiplier": 1.0,
                "exploration_strength": 20.0,
            },
            "per_problem": {
                "zdt1": {
                    "operator": "gaussian",
                    "multiplier": 1.5,
                    "exploration_strength": 0.1,
                }
            },
        },
    )
    out_dir = tmp / "out"
    args = parse_args(
        [
            "--stage", "all",
            "--train-dirs", str(train_dir),
            "--fixed-dir", str(fixed_dir),
            "--static-tuning", str(static_tuning),
            "--out-dir", str(out_dir),
            "--problems", "zdt1",
            "--seeds", "1000",
            "--generations", "5",
            "--pop-size", "20",
            "--epochs", "5",
            "--window", "3",
        ]
    )
    results = run_experiment(args)
    return {"results": results, "out_dir": out_dir, "fixed_dir": fixed_dir}


def test_all_nine_arms_ran(mini_experiment: dict[str, Any]) -> None:
    results = mini_experiment["results"]
    assert list(results["arms"]) == list(ARMS)
    assert len(ARMS) == 9
    for arm in ARMS:
        assert f"{arm}|zdt1|1000" in results["runs"], arm


def test_results_schema_exact(mini_experiment: dict[str, Any]) -> None:
    results = mini_experiment["results"]
    assert set(results) == {
        "config",
        "failure_thresholds",
        "arms",
        "problems",
        "eval_seeds",
        "runs",
        "training",
    }
    config = results["config"]
    for key in (
        "arms",
        "problems",
        "eval_seeds",
        "train_dirs",
        "epochs",
        "window",
        "pm_mult_range",
        "exploration_ranges",
        "wall_time_sec",
    ):
        assert key in config, key
    assert config["pm_mult_range"] == list(DEFAULT_PM_MULT_RANGE)
    run = results["runs"]["fixed_nsga2|zdt1|1000"]
    assert set(run) == {
        "final_hv",
        "final_igd",
        "auc_hv",
        "runtime_sec",
        "failed",
        "trajectory_file",
    }
    assert run["trajectory_file"].startswith("runs/")
    for arm in LEARNED_ARMS:
        entry = results["training"][arm]
        assert len(entry["train_loss"]) == 5
        assert entry["n_train_samples"] > 0
        assert entry["final_val_loss"] is not None


def test_paired_seeds_identical_across_arms(mini_experiment: dict[str, Any]) -> None:
    results = mini_experiment["results"]
    seeds_per_arm = {
        arm: {
            int(key.split("|")[2])
            for key in results["runs"]
            if key.startswith(f"{arm}|")
        }
        for arm in ARMS
    }
    for arm, seeds in seeds_per_arm.items():
        assert seeds == {1000}, arm
    assert results["eval_seeds"] == [1000]


def test_thresholds_written_before_eval(mini_experiment: dict[str, Any]) -> None:
    out_dir = mini_experiment["out_dir"]
    thresholds_path = out_dir / "failure_thresholds.json"
    assert thresholds_path.is_file()
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    assert "zdt1" in thresholds["thresholds"]
    # The eval run configs carry the threshold that was loaded at eval time,
    # proving the file existed before eval used it.
    run_path = out_dir / "runs" / "fixed_nsga2__zdt1__seed1000.json"
    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    assert run_payload["config"]["failure_threshold"] == pytest.approx(
        thresholds["thresholds"]["zdt1"]
    )
    results_path = out_dir / "results.json"
    assert thresholds_path.stat().st_mtime_ns <= results_path.stat().st_mtime_ns


def test_controllers_persisted(mini_experiment: dict[str, Any]) -> None:
    controllers_dir = mini_experiment["out_dir"] / "controllers"
    for arm in LEARNED_ARMS:
        assert (controllers_dir / f"{arm}.pt").is_file(), arm
    assert (controllers_dir / "encoders" / "zdt1.json").is_file()
    assert (controllers_dir / "static_full_global.json").is_file()
    assert (controllers_dir / "static_full_per_problem" / "zdt1.json").is_file()
    assert (controllers_dir / "open_loop_global.json").is_file()
    assert (controllers_dir / "open_loop_per_problem.json").is_file()
    assert (controllers_dir / "train_info.json").is_file()


def test_state_scrambled_shuffle_proof(mini_experiment: dict[str, Any]) -> None:
    training = mini_experiment["results"]["training"]
    scrambled = training["state_scrambled_mlp"]
    assert scrambled["train_history_shuffled"] is True
    assert scrambled["x_train_checksum_unshuffled"] is not None
    assert scrambled["x_train_checksum_shuffled"] is not None
    assert (
        scrambled["x_train_checksum_unshuffled"]
        != scrambled["x_train_checksum_shuffled"]
    )
    # Non-scrambled arms must not carry a shuffle proof.
    normalized = training["mlp2_closed_loop_normalized"]
    assert normalized["train_history_shuffled"] is False
    assert normalized["x_train_checksum_shuffled"] is None


def test_generation_only_ignores_population_state(
    mini_experiment: dict[str, Any],
) -> None:
    from controller.multihead_controller import MultiHeadController

    out_dir = mini_experiment["out_dir"]
    controller = MultiHeadController.load(
        out_dir / "controllers" / "generation_only_mlp.pt"
    )
    info = json.loads(
        (out_dir / "controllers" / "train_info.json").read_text(encoding="utf-8")
    )
    policy = GenerationOnlyPolicy(
        controller,
        info["generation_only"]["problem_vectors_zscored"],
        mutation_target=info["generation_only"]["mutation_target"],
    )
    history_a = [
        {key: 0.0 for key in ("hv", "igd", "diversity", "delta_hv", "delta_igd", "generation")}
    ]
    history_b = [
        {key: 100.0 for key in ("hv", "igd", "diversity", "delta_hv", "delta_igd", "generation")}
        for _ in range(5)
    ]
    action_a = policy.predict_action(
        3, 5, "zdt1", n_vars=30, pm_mult_range=(0.25, 8.0), history=history_a
    )
    action_b = policy.predict_action(
        3, 5, "zdt1", n_vars=30, pm_mult_range=(0.25, 8.0), history=history_b
    )
    assert action_a == action_b
    assert set(action_a) == {
        "mutation_operator",
        "mutation_probability",
        "exploration_strength",
    }


def test_missing_train_dir_raises(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--stage", "train",
            "--train-dirs", str(tmp_path / "nope"),
            "--out-dir", str(tmp_path / "out"),
            "--static-tuning", str(tmp_path / "tuning.json"),
        ]
    )
    with pytest.raises(FileNotFoundError, match="training directory"):
        run_experiment(args)


def test_eval_protocol_recorded_in_run_config(mini_experiment: dict[str, Any]) -> None:
    out_dir = mini_experiment["out_dir"]
    for arm in ARMS:
        payload = json.loads(
            (out_dir / "runs" / f"{arm}__zdt1__seed1000.json").read_text(
                encoding="utf-8"
            )
        )
        config = payload["config"]
        assert config["arm"] == arm
        assert config["problem"] == "zdt1"
        assert config["seed"] == 1000
        assert "protocol" in config
        assert set(payload["metrics"]) == {
            "final_hv",
            "final_igd",
            "auc_hv",
            "runtime_sec",
            "failed",
        }
