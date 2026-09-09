from __future__ import annotations

"""End-to-end test for the Phase-1.5 experiment runner (experiments/run_phase1_5.py).

Writes two tiny synthetic training trajectories — one with the full
three-key Phase-1.5 action (``mutation_operator`` alternating between
polynomial and gaussian, varying ``mutation_probability`` and
``exploration_strength``) and one with the legacy two-key Phase-0/1 action
(imputation path) — plus a minimal fake Phase-1 ``results.json`` supplying
the reused arms. Then ``run_phase1_5.run_experiment`` runs with minimal
settings (zdt1, one eval seed, 5 generations, pop 20, 10 epochs, window 3)
into a temporary directory and validates:

* ``results.json`` exists with all five arms (``mlp2``, ``mlp2_nopf`` plus
  the reused ``fixed``, ``constant``, ``mlp_w10``), per-arm metric
  aggregates, training losses, and both Wilcoxon reports;
* the reused arms carry exactly the numbers from the fake Phase-1 file;
* the action-dynamics summary exists for the new arms;
* one trajectory JSON per new arm is saved, whose controller actions hold
  all three action keys and respect the deployment bounds
  (pm in [0.25/n_vars, 8/n_vars], exploration strength in the
  operator-specific range).

Settings note: with pop 20 / 5 generations on ZDT1 the hypervolume against
ref point (1.1, 1.1) may stay 0.0 (cf. tests/test_run_phase1.py); the
assertions here concern wiring and contract compliance, not optimization
quality. Runtime is dominated by the torch import and stays well under 90 s.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from experiments import run_phase1_5

_PROBLEM = "zdt1"
_N_VARS = 30  # ZDT1 decision variables; fixes the expected pm bounds.
_EVAL_SEED = 100
_GENERATIONS = 5
_POP_SIZE = 20
_EPOCHS = 10
_WINDOW = 3
_NEW_ARMS = ("mlp2", "mlp2_nopf")
_REUSED_ARMS = ("fixed", "constant", "mlp_w10")
_ALL_ARMS = _NEW_ARMS + _REUSED_ARMS
_PM_MIN = 0.25 / _N_VARS  # (1/n_vars) * [0.25, 8.0]
_PM_MAX = 8.0 / _N_VARS
_N_TRANSITIONS = 12  # generations 0..11 -> 11 supervised samples per trajectory

_FAKE_PHASE1_HV = {"fixed": 0.61, "constant": 0.60, "mlp_w10": 0.62}

_METRIC_KEYS = (
    "final_hv_mean",
    "final_hv_std",
    "final_igd_mean",
    "final_igd_std",
    "auc_hv_mean",
    "auc_hv_std",
    "mean_runtime_sec",
)

_ACTION_DYNAMICS_KEYS = (
    "mean_within_run_pm_std",
    "mean_operator_switch_rate",
    "mean_pm",
)


def _make_transition(
    generation: int,
    hv: float,
    igd_value: float,
    diversity: float,
    delta_hv: float,
    delta_igd: float,
    operator: str,
    pm: float,
    exploration: float | None,
) -> dict[str, Any]:
    """Build one transition dict in the recorder schema (2- or 3-key action)."""
    action: dict[str, Any] = {
        "mutation_operator": operator,
        "mutation_probability": pm,
    }
    if exploration is not None:
        action["exploration_strength"] = exploration
    return {
        "generation": generation,
        "state": {
            "generation": generation,
            "hv": hv,
            "igd": igd_value,
            "diversity": diversity,
        },
        "action": action,
        "reward": {"delta_hv": delta_hv, "delta_igd": delta_igd},
    }


def _write_trajectory(path: Path, full_actions: bool, offset: float) -> None:
    """Write one synthetic training trajectory JSON.

    HV improves and IGD decreases with varying rewards so the
    advantage-weighted targets are non-degenerate; generation 0 carries
    zero reward by the recorder convention. ``full_actions`` selects the
    three-key Phase-1.5 action (alternating operators) over the legacy
    two-key action (polynomial only, exploration strength imputed at
    dataset build time).
    """
    hv = 0.30 + offset
    igd_value = 0.50 - offset
    transitions = []
    for t in range(_N_TRANSITIONS):
        if full_actions:
            operator = "polynomial" if t % 2 == 0 else "gaussian"
            exploration = 5.0 + t if operator == "polynomial" else 0.05 + 0.005 * t
        else:
            operator = "polynomial"
            exploration = None
        pm = 0.020 + 0.004 * (t % 4)
        if t == 0:
            delta_hv = delta_igd = 0.0
        else:
            delta_hv = 0.008 + 0.002 * (t % 3)
            delta_igd = 0.006 + 0.001 * ((t + 1) % 3)
            hv += delta_hv
            igd_value -= delta_igd
        transitions.append(
            _make_transition(
                t, hv, igd_value, 0.25 + 0.01 * t,
                delta_hv, delta_igd, operator, pm, exploration,
            )
        )
    payload = {
        "config": {
            "problem": _PROBLEM,
            "n_vars": _N_VARS,
            "algorithm": "nsga2",
            "policy": "random",
            "action_space": "full" if full_actions else "pm",
            "pop_size": 100,
            "generations": _N_TRANSITIONS - 1,
        },
        "seed": 0,
        "runtime_sec": 1.0,
        "schema_version": 1,
        "transitions": transitions,
        "final": {"hv": hv, "igd": igd_value, "diversity": transitions[-1]["state"]["diversity"]},
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_fake_phase1_results(path: Path) -> None:
    """Write a minimal Phase-1 results.json fixture for the reused arms.

    Only the ``results[arm][problem]["seeds"][str(seed)]`` records are read
    by the Phase-1.5 runner; every reused arm gets a distinct final_hv so
    the reuse wiring is observable in the assertions.
    """

    def _entry(hv: float) -> dict[str, Any]:
        return {
            "n_seeds": 1,
            "seeds": {
                str(_EVAL_SEED): {
                    "final_hv": hv,
                    "final_igd": 0.05,
                    "auc_hv": 0.40,
                    "runtime_sec": 12.5,
                    "trajectory_file": f"phase1_arm_{_PROBLEM}_seed{_EVAL_SEED}.json",
                }
            },
        }

    payload = {
        "config": {"phase": 1},
        "arms": ["fixed", "mlp_w10", "mlp_w1", "constant"],
        "results": {
            arm: {_PROBLEM: _entry(hv)} for arm, hv in _FAKE_PHASE1_HV.items()
        },
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the tiny Phase-1.5 experiment once per module; return paths/payload."""
    root = tmp_path_factory.mktemp("phase1_5")
    train_dir_full = root / "train_full"
    train_dir_legacy = root / "train_legacy"
    train_dir_full.mkdir()
    train_dir_legacy.mkdir()
    _write_trajectory(train_dir_full / "full_policy_traj0.json", full_actions=True, offset=0.0)
    _write_trajectory(train_dir_legacy / "legacy_policy_traj1.json", full_actions=False, offset=0.02)
    phase1_results = root / "phase1_results.json"
    _write_fake_phase1_results(phase1_results)
    out_dir = root / "out"
    args = run_phase1_5.parse_args(
        [
            "--train-dirs", str(train_dir_full), str(train_dir_legacy),
            "--out-dir", str(out_dir),
            "--phase1-results", str(phase1_results),
            "--problems", _PROBLEM,
            "--eval-seeds", str(_EVAL_SEED),
            "--generations", str(_GENERATIONS),
            "--pop-size", str(_POP_SIZE),
            "--epochs", str(_EPOCHS),
            "--window", str(_WINDOW),
        ]
    )
    payload = run_phase1_5.run_experiment(args)
    return {"out_dir": out_dir, "payload": payload}


def _load_eval_trajectory(out_dir: Path, arm: str) -> dict[str, Any]:
    """Load the saved eval trajectory JSON of one new arm."""
    path = out_dir / "trajectories" / f"{arm}_{_PROBLEM}_seed{_EVAL_SEED}.json"
    assert path.exists(), f"trajectory file missing: {path}"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_results_json_written(experiment: dict[str, Any]) -> None:
    """results.json exists, matches the returned payload, has the core schema."""
    results_path = experiment["out_dir"] / "results.json"
    assert results_path.exists()
    with results_path.open(encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk == experiment["payload"]
    payload = experiment["payload"]
    for key in (
        "config", "arms", "training", "results",
        "wilcoxon_vs_fixed", "wilcoxon_vs_constant", "action_dynamics",
        "wall_time_sec",
    ):
        assert key in payload, f"results.json missing top-level key {key}"
    assert payload["arms"] == list(_ALL_ARMS)
    assert payload["wall_time_sec"] > 0.0


def test_all_five_arms_have_metric_aggregates(experiment: dict[str, Any]) -> None:
    """Every arm x problem entry (2 new + 3 reused) carries the metric keys."""
    results = experiment["payload"]["results"]
    for arm in _ALL_ARMS:
        assert arm in results, f"arm missing from results: {arm}"
        entry = results[arm][_PROBLEM]
        for key in _METRIC_KEYS:
            assert key in entry, f"{arm}/{_PROBLEM} missing metric {key}"
            assert math.isfinite(entry[key]), f"{arm}/{_PROBLEM} {key} not finite"
        assert entry["n_seeds"] == 1


def test_reused_arms_carry_phase1_values(experiment: dict[str, Any]) -> None:
    """The reused arms report exactly the fake Phase-1 per-seed values."""
    results = experiment["payload"]["results"]
    for arm, hv in _FAKE_PHASE1_HV.items():
        entry = results[arm][_PROBLEM]
        assert entry["final_hv_mean"] == pytest.approx(hv)
        raw = entry["seeds"][str(_EVAL_SEED)]
        assert raw["final_hv"] == pytest.approx(hv)
        assert raw["final_igd"] == pytest.approx(0.05)
        assert raw["auc_hv"] == pytest.approx(0.40)


def test_training_report(experiment: dict[str, Any]) -> None:
    """Training losses and sample counts are reported per new arm."""
    training = experiment["payload"]["training"]
    n_samples = 2 * (_N_TRANSITIONS - 1)
    for arm, input_dim in (("mlp2", _WINDOW * 6 + 9), ("mlp2_nopf", _WINDOW * 6)):
        entry = training[arm]
        assert entry["kind"] == "multihead_mlp"
        assert entry["window"] == _WINDOW
        assert entry["input_dim"] == input_dim
        assert len(entry["train_loss"]) == _EPOCHS
        assert entry["n_train_samples"] + entry["n_val_samples"] == n_samples
        assert entry["n_train_samples"] > 0 and entry["n_val_samples"] > 0
    assert training["mlp2"]["problem_aware"] is True
    assert training["mlp2_nopf"]["problem_aware"] is False


def test_wilcoxon_reports_against_both_baselines(experiment: dict[str, Any]) -> None:
    """Every arm is tested vs fixed AND vs constant; baselines are excluded."""
    payload = experiment["payload"]
    vs_fixed = payload["wilcoxon_vs_fixed"]
    assert "fixed" not in vs_fixed
    for arm in ("mlp2", "mlp2_nopf", "constant", "mlp_w10"):
        entry = vs_fixed[arm][_PROBLEM]
        assert entry["vs"] == "fixed"
        assert entry["zero_method"] == "zsplit"
        assert entry["alternative"] == "greater"
        assert entry["metric"] == "final_hv"
        if entry["p_value"] is not None:
            assert 0.0 <= entry["p_value"] <= 1.0
    vs_constant = payload["wilcoxon_vs_constant"]
    assert "constant" not in vs_constant
    for arm in ("mlp2", "mlp2_nopf", "fixed", "mlp_w10"):
        assert vs_constant[arm][_PROBLEM]["vs"] == "constant"


def test_action_dynamics_summary(experiment: dict[str, Any]) -> None:
    """Action-dynamics keys exist per new arm x problem with sane values."""
    dynamics = experiment["payload"]["action_dynamics"]
    for arm in _NEW_ARMS:
        entry = dynamics[arm][_PROBLEM]
        for key in _ACTION_DYNAMICS_KEYS:
            assert key in entry, f"{arm}/{_PROBLEM} missing dynamics key {key}"
        assert entry["mean_within_run_pm_std"] >= 0.0
        assert 0.0 <= entry["mean_operator_switch_rate"] <= 1.0
        assert _PM_MIN - 1e-12 <= entry["mean_pm"] <= _PM_MAX + 1e-12


def test_new_arm_trajectories_have_full_bounded_actions(experiment: dict[str, Any]) -> None:
    """New-arm trajectories hold 3-key actions within the deployment bounds."""
    for arm in _NEW_ARMS:
        trajectory = _load_eval_trajectory(experiment["out_dir"], arm)
        assert trajectory["config"]["arm"] == arm
        assert trajectory["config"]["window"] == _WINDOW
        transitions = trajectory["transitions"]
        assert len(transitions) == _GENERATIONS + 1
        # Transition 0 is the algorithm default; t >= 1 are controller actions.
        for t in transitions[1:]:
            action = t["action"]
            assert {
                "mutation_operator",
                "mutation_probability",
                "exploration_strength",
            } <= set(action)
            assert action["mutation_operator"] in ("polynomial", "gaussian")
            pm = float(action["mutation_probability"])
            assert _PM_MIN - 1e-12 <= pm <= _PM_MAX + 1e-12, (
                f"{arm} emitted pm {pm} outside [{_PM_MIN}, {_PM_MAX}]"
            )
            exploration = float(action["exploration_strength"])
            if action["mutation_operator"] == "polynomial":
                assert 2.0 - 1e-9 <= exploration <= 50.0 + 1e-9
            else:
                assert 0.02 - 1e-9 <= exploration <= 0.3 + 1e-9


def test_config_records_experiment_setup(experiment: dict[str, Any]) -> None:
    """Full config (train dirs, phase-1 source, action space) is persisted."""
    config = experiment["payload"]["config"]
    assert config["phase"] == 1.5
    assert config["problems"] == [_PROBLEM]
    assert config["eval_seeds"] == [_EVAL_SEED]
    assert config["pop_size"] == _POP_SIZE
    assert config["generations"] == _GENERATIONS
    assert config["window"] == _WINDOW
    assert config["action_space"] == "full"
    assert config["pm_mult_range"] == [0.25, 8.0]
    assert len(config["train_dirs"]) == 2
    training = config["training"]
    assert training["epochs"] == _EPOCHS
    assert training["n_train_trajectories"] == 2
    assert training["n_samples"]["mlp2"] == 2 * (_N_TRANSITIONS - 1)
