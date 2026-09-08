"""End-to-end test for the Phase-1 experiment runner (experiments/run_phase1.py).

Writes three tiny synthetic random-policy training trajectories (recorder
schema, varying mutation probabilities and rewards), then runs
``run_phase1.run_experiment`` with minimal settings (zdt1, one eval seed,
6 generations, pop 20, 10 epochs) into a temporary directory and validates:

* ``results.json`` exists with all four arms (``fixed``, ``mlp_w10``,
  ``mlp_w1``, ``constant``), per-arm metric aggregates, per-seed raw values,
  training losses, and Wilcoxon reports;
* one trajectory JSON per arm is saved in the Phase-0 recorder schema;
* deployment semantics: the fixed arm keeps the default ``1/n_vars``
  mutation probability, the constant arm emits a (near-)constant probability,
  and the MLP arms emit probabilities inside the deployment range
  ``[0.5/n_vars, 5.0/n_vars]`` — each chosen from history strictly before
  the target generation.

Settings note: with pop 20 / 6 generations on ZDT1 the hypervolume against
ref point (1.1, 1.1) may stay 0.0 (cf. tests/test_generate_dataset.py); the
assertions here concern wiring and contract compliance, not optimization
quality. Runtime is dominated by the torch import and stays well under 90 s.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from experiments import run_phase1

_PROBLEM = "zdt1"
_N_VARS = 30  # ZDT1 decision variables; fixes the expected pm bounds.
_EVAL_SEED = 100
_GENERATIONS = 6
_POP_SIZE = 20
_EPOCHS = 10
_ARMS = ("fixed", "mlp_w10", "mlp_w1", "constant")
_PM_MIN = 0.5 / _N_VARS  # --pm-mult-range 0.5 5.0 on 1/n_vars
_PM_MAX = 5.0 / _N_VARS
_N_TRAIN_TRAJECTORIES = 3
_TRANSITIONS_PER_TRAJECTORY = 6  # generations 0..5 -> 5 supervised samples each

_METRIC_KEYS = (
    "final_hv_mean",
    "final_hv_std",
    "final_igd_mean",
    "final_igd_std",
    "auc_hv_mean",
    "auc_hv_std",
    "mean_runtime_sec",
)


def _make_transition(
    generation: int,
    pm: float,
    hv: float,
    igd_value: float,
    diversity: float,
    delta_hv: float,
    delta_igd: float,
) -> dict[str, Any]:
    """Build one transition dict in the Phase-0 recorder schema."""
    return {
        "generation": generation,
        "state": {
            "generation": generation,
            "hv": hv,
            "igd": igd_value,
            "diversity": diversity,
        },
        "action": {
            "mutation_operator": "polynomial",
            "mutation_probability": pm,
        },
        "reward": {"delta_hv": delta_hv, "delta_igd": delta_igd},
    }


def _write_training_trajectories(train_dir: Path) -> None:
    """Write three synthetic random-policy trajectory JSONs.

    Mutation probabilities vary within and across trajectories (all inside
    the deployment range), HV improves while IGD decreases, and rewards vary
    per transition so that the reward-weighted training targets and sample
    weights are non-degenerate. Generation 0 carries zero reward by the
    recorder convention.
    """
    for j in range(_N_TRAIN_TRAJECTORIES):
        hv = 0.30 + 0.02 * j
        igd_value = 0.50 - 0.01 * j
        transitions = []
        for t in range(_TRANSITIONS_PER_TRAJECTORY):
            pm = 0.020 + 0.008 * j + 0.006 * (t % 3)
            if t == 0:
                delta_hv = delta_igd = 0.0
            else:
                delta_hv = 0.010 + 0.002 * j + 0.003 * (t % 2)
                delta_igd = 0.008 + 0.001 * j + 0.002 * ((t + 1) % 2)
                hv += delta_hv
                igd_value -= delta_igd
            diversity = 0.25 + 0.01 * t + 0.02 * j
            transitions.append(
                _make_transition(t, pm, hv, igd_value, diversity, delta_hv, delta_igd)
            )
        payload = {
            "config": {
                "problem": _PROBLEM,
                "n_vars": _N_VARS,
                "algorithm": "nsga2",
                "policy": "random_pm",
                "pop_size": 100,
                "generations": _TRANSITIONS_PER_TRAJECTORY - 1,
            },
            "seed": j,
            "runtime_sec": 1.0 + 0.1 * j,
            "schema_version": 1,
            "transitions": transitions,
            "final": {
                "hv": hv,
                "igd": igd_value,
                "diversity": transitions[-1]["state"]["diversity"],
            },
        }
        out_path = train_dir / f"random_policy_traj{j}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the tiny Phase-1 experiment once per module; return paths/payload."""
    root = tmp_path_factory.mktemp("phase1")
    train_dir = root / "train"
    train_dir.mkdir()
    _write_training_trajectories(train_dir)
    out_dir = root / "out"
    args = run_phase1.parse_args(
        [
            "--train-dir", str(train_dir),
            "--out-dir", str(out_dir),
            "--problems", _PROBLEM,
            "--eval-seeds", str(_EVAL_SEED),
            "--generations", str(_GENERATIONS),
            "--pop-size", str(_POP_SIZE),
            "--epochs", str(_EPOCHS),
        ]
    )
    payload = run_phase1.run_experiment(args)
    return {"out_dir": out_dir, "train_dir": train_dir, "payload": payload}


def _load_eval_trajectory(out_dir: Path, arm: str) -> dict[str, Any]:
    """Load the saved eval trajectory JSON of one arm."""
    path = out_dir / "trajectories" / f"{arm}_{_PROBLEM}_seed{_EVAL_SEED}.json"
    assert path.exists(), f"trajectory file missing: {path}"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _controller_pms(trajectory: dict[str, Any]) -> list[float]:
    """Mutation probabilities of transitions t >= 1 (the controller actions).

    Transition 0 is recorded before any controller decision and always holds
    the algorithm default, so it is excluded from controller-action checks.
    """
    return [
        float(t["action"]["mutation_probability"])
        for t in trajectory["transitions"][1:]
    ]


def test_results_json_written(experiment: dict[str, Any]) -> None:
    """results.json exists, matches the returned payload, has the core schema."""
    results_path = experiment["out_dir"] / "results.json"
    assert results_path.exists()
    with results_path.open(encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk == experiment["payload"]
    payload = experiment["payload"]
    for key in ("config", "arms", "training", "results", "wilcoxon_vs_fixed", "wall_time_sec"):
        assert key in payload
    assert payload["arms"] == list(_ARMS)
    assert payload["wall_time_sec"] > 0.0


def test_all_arms_have_metric_aggregates(experiment: dict[str, Any]) -> None:
    """Every arm x problem entry carries the aggregate metric keys."""
    results = experiment["payload"]["results"]
    for arm in _ARMS:
        assert arm in results, f"arm missing from results: {arm}"
        entry = results[arm][_PROBLEM]
        for key in _METRIC_KEYS:
            assert key in entry, f"{arm}/{_PROBLEM} missing metric {key}"
            assert math.isfinite(entry[key]), f"{arm}/{_PROBLEM} {key} not finite"
        assert entry["n_seeds"] == 1


def test_per_seed_raw_values(experiment: dict[str, Any]) -> None:
    """Per-seed raw values are stored for later analysis (AGENTS.md Rule 2)."""
    results = experiment["payload"]["results"]
    for arm in _ARMS:
        seeds = results[arm][_PROBLEM]["seeds"]
        assert str(_EVAL_SEED) in seeds
        raw = seeds[str(_EVAL_SEED)]
        for key in ("final_hv", "final_igd", "auc_hv", "runtime_sec", "trajectory_file"):
            assert key in raw, f"{arm} seed record missing {key}"
        assert raw["trajectory_file"] == f"{arm}_{_PROBLEM}_seed{_EVAL_SEED}.json"
        # Aggregate of a single seed must equal the raw value.
        assert results[arm][_PROBLEM]["final_hv_mean"] == pytest.approx(raw["final_hv"])


def test_training_report(experiment: dict[str, Any]) -> None:
    """Training losses and sample counts are reported per controller arm."""
    training = experiment["payload"]["training"]
    n_samples = _N_TRAIN_TRAJECTORIES * (_TRANSITIONS_PER_TRAJECTORY - 1)
    for arm, window in (("mlp_w10", 10), ("mlp_w1", 1)):
        entry = training[arm]
        assert entry["window"] == window
        assert entry["input_dim"] == window * 6
        assert len(entry["train_loss"]) == _EPOCHS
        assert len(entry["val_loss"]) == _EPOCHS
        assert entry["final_train_loss"] == pytest.approx(entry["train_loss"][-1])
        assert entry["final_val_loss"] == pytest.approx(entry["val_loss"][-1])
        assert entry["n_train_samples"] + entry["n_val_samples"] == n_samples
        assert entry["n_train_samples"] > 0 and entry["n_val_samples"] > 0
    constant = training["constant"]
    assert constant["kind"] == "constant"
    # The learned constant is the reward-weighted mean of log(pm) over the
    # synthetic training pms, which all lie in [0.02, 0.046].
    assert 0.01 < math.exp(constant["log_pm_constant"]) < 0.2


def test_config_records_experiment_setup(experiment: dict[str, Any]) -> None:
    """Full config (training params, seeds, windows) is persisted."""
    config = experiment["payload"]["config"]
    assert config["problems"] == [_PROBLEM]
    assert config["eval_seeds"] == [_EVAL_SEED]
    assert config["pop_size"] == _POP_SIZE
    assert config["generations"] == _GENERATIONS
    assert config["pm_mult_range"] == [0.5, 5.0]
    training = config["training"]
    assert training["epochs"] == _EPOCHS
    assert training["windows"] == {"mlp_w10": 10, "mlp_w1": 1}
    assert training["n_train_trajectories"] == _N_TRAIN_TRAJECTORIES
    assert "train_seed" in training and "lr" in training and "batch_size" in training


def test_wilcoxon_report(experiment: dict[str, Any]) -> None:
    """Each controller arm gets a Wilcoxon p-value vs the fixed baseline."""
    report = experiment["payload"]["wilcoxon_vs_fixed"]
    assert "fixed" not in report
    for arm in ("mlp_w10", "mlp_w1", "constant"):
        entry = report[arm][_PROBLEM]
        assert entry["zero_method"] == "zsplit"
        assert entry["alternative"] == "greater"
        assert entry["metric"] == "final_hv"
        assert entry["vs"] == "fixed"
        assert entry["n_seeds"] == 1
        # With a single seed the p-value is computable but uninformative;
        # accept None (degenerate) or a valid probability.
        if entry["p_value"] is not None:
            assert 0.0 <= entry["p_value"] <= 1.0


def test_trajectory_files_per_arm(experiment: dict[str, Any]) -> None:
    """One trajectory JSON per arm, in the recorder schema, gens 0..G."""
    for arm in _ARMS:
        trajectory = _load_eval_trajectory(experiment["out_dir"], arm)
        for key in ("config", "seed", "runtime_sec", "schema_version", "transitions", "final"):
            assert key in trajectory
        assert trajectory["seed"] == _EVAL_SEED
        transitions = trajectory["transitions"]
        assert len(transitions) == _GENERATIONS + 1
        assert [t["generation"] for t in transitions] == list(range(_GENERATIONS + 1))
        for t in transitions:
            # Phase 1.5: NSGA-II current_action() reports exploration_strength too
            assert {"mutation_operator", "mutation_probability"} <= set(t["action"])
            assert t["action"]["mutation_operator"] == "polynomial"
        config = trajectory["config"]
        assert config["arm"] == arm
        assert config["problem"] == _PROBLEM
        if arm == "fixed":
            assert config["controller"] is None
        else:
            assert config["pm_min"] == pytest.approx(_PM_MIN)
            assert config["pm_max"] == pytest.approx(_PM_MAX)
    assert _load_eval_trajectory(experiment["out_dir"], "mlp_w10")["config"]["window"] == 10
    assert _load_eval_trajectory(experiment["out_dir"], "mlp_w1")["config"]["window"] == 1


def test_fixed_arm_uses_default_pm(experiment: dict[str, Any]) -> None:
    """The fixed arm keeps the default 1/n_vars mutation probability."""
    trajectory = _load_eval_trajectory(experiment["out_dir"], "fixed")
    pms = [t["action"]["mutation_probability"] for t in trajectory["transitions"]]
    assert pms == [pytest.approx(1.0 / _N_VARS)] * len(pms)


def test_constant_arm_pm_near_constant(experiment: dict[str, Any]) -> None:
    """The constant (no-history) ablation emits one clipped constant pm."""
    trajectory = _load_eval_trajectory(experiment["out_dir"], "constant")
    pms = _controller_pms(trajectory)
    assert len(pms) == _GENERATIONS
    assert max(pms) - min(pms) <= 1e-12
    assert all(_PM_MIN - 1e-12 <= pm <= _PM_MAX + 1e-12 for pm in pms)


def test_mlp_arms_pm_within_bounds(experiment: dict[str, Any]) -> None:
    """MLP arm actions are clipped to [pm_min, pm_max] every generation."""
    for arm in ("mlp_w10", "mlp_w1"):
        trajectory = _load_eval_trajectory(experiment["out_dir"], arm)
        pms = _controller_pms(trajectory)
        assert len(pms) == _GENERATIONS
        assert all(
            _PM_MIN - 1e-12 <= pm <= _PM_MAX + 1e-12 for pm in pms
        ), f"{arm} emitted pm outside [{_PM_MIN}, {_PM_MAX}]: {pms}"
