from __future__ import annotations

"""Tests for the Phase-2.75D protocol fix and the advantage-target comparison.

Covers three things:

* ``evaluate-horizon --include-default-action`` really inserts the NSGA-II
  default action (``kind="default"``) and ``--no-include-default-action``
  reproduces the historical candidate sampling bit-for-bit;
* the one-step ``evaluate`` subcommand is untouched — including a golden
  SHA-256 of its output, captured before this change on this machine;
* the v2 builder's Target C (``future_improvement``) is hand-computable and
  the target comparison script runs end to end on synthetic per-target
  datasets with a shared state-level split.
"""

import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller.dataset import merge_state_reward
from controller.multihead_controller import MultiHeadController
from controller.state_encoder import StateEncoder
from experiments import build_intervention_dataset_v2 as bids
from experiments import compare_advantage_targets as cat
from experiments import counterfactual_actions as cfa
from experiments.generate_dataset import (
    FULL_ACTION_PM_MULT_RANGE,
    sample_full_action,
)

_WINDOW = 3
_GENS = 8
_STATES = 2
_POP = 20
_N_ALTERNATIVES = 3
_N_REPS = 2
_HORIZONS = ("2", "4")
#: SHA-256 of ``counterfactual_zdt1.json`` produced by the one-step
#: ``evaluate`` subcommand on the golden fixture below, captured before the
#: Phase-2.75D candidate change.
GOLDEN_ONE_STEP_SHA256 = (
    "aa6fbda3c0fe112219f9251a983baab1ceaba9ee362b2564efc19b7357b318fb"
)


# --- shared fixture ----------------------------------------------------------


def _trajectories() -> list[list[dict[str, Any]]]:
    """Two 9-transition trajectories (the golden fixture's corpus)."""
    trajectories: list[list[dict[str, Any]]] = []
    for j in range(2):
        hv = 0.30 + 0.05 * j
        igd_value = 0.50 - 0.02 * j
        transitions = []
        for t in range(_GENS + 1):
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


def _write_snapshots(root: Path) -> tuple[StateEncoder, Path, Path]:
    """Harvest two zdt1 snapshots plus the encoder and controller artefacts."""
    encoder = StateEncoder(_WINDOW).fit(_trajectories())
    encoder_path = root / "encoder.json"
    encoder.save(encoder_path)
    controller = MultiHeadController(input_dim=encoder.dim, seed=0, name="cf_verify")
    controller_path = root / "controller.pt"
    controller.save(controller_path)
    cfa.harvest_snapshots(
        "zdt1",
        seed=1000,
        generations=_GENS,
        pop_size=_POP,
        states_per_run=_STATES,
        n_reference_points=200,
        ref_point=np.asarray([1.1, 1.1]),
        out_dir=root / "snapshots",
    )
    return encoder, encoder_path, controller_path


@pytest.fixture(scope="module")
def counterfactual_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Encoder + controller + two snapshots for the horizon-path tests."""
    root = tmp_path_factory.mktemp("targets_cf")
    encoder, encoder_path, controller_path = _write_snapshots(root)
    return {
        "root": root,
        "encoder": encoder,
        "encoder_path": encoder_path,
        "controller_path": controller_path,
        "snapshots_dir": root / "snapshots",
    }


def _horizon_args(fixture: dict[str, Any], out_dir: Path, *extra: str) -> list[str]:
    return [
        "evaluate-horizon",
        "--problem", "zdt1",
        "--snapshots-dir", str(fixture["snapshots_dir"]),
        "--out-dir", str(out_dir),
        "--controller", str(fixture["controller_path"]),
        "--encoder", str(fixture["encoder_path"]),
        "--horizons", *_HORIZONS,
        "--n-alternatives", str(_N_ALTERNATIVES),
        "--n-reps", str(_N_REPS),
        *extra,
    ]


# --- Task 1: protocol fix ----------------------------------------------------


def test_include_default_action_adds_one_candidate(
    counterfactual_fixture: dict[str, Any], tmp_path: Path
) -> None:
    """The default action is candidate 1, tagged 'default', +1 candidate."""
    fixture = counterfactual_fixture
    with_default = cfa.run_evaluate_horizon(
        cfa.parse_args(_horizon_args(fixture, tmp_path / "with"))
    )
    without = cfa.run_evaluate_horizon(
        cfa.parse_args(
            _horizon_args(
                fixture, tmp_path / "without", "--no-include-default-action"
            )
        )
    )
    assert with_default["config"]["include_default_action"] is True
    assert without["config"]["include_default_action"] is False

    for state_with, state_without in zip(
        with_default["states"], without["states"]
    ):
        candidates_with = state_with["candidates"]
        candidates_without = state_without["candidates"]
        assert len(candidates_with) == 1 + 1 + _N_ALTERNATIVES
        assert len(candidates_without) == 1 + _N_ALTERNATIVES
        assert [c["kind"] for c in candidates_with] == [
            "controller",
            "default",
            "alternative",
            "alternative",
            "alternative",
        ]
        assert [c["kind"] for c in candidates_without] == [
            "controller",
            "alternative",
            "alternative",
            "alternative",
        ]
        # the protocol candidate is exactly the NSGA-II default action
        assert candidates_with[1]["action"] == {
            "mutation_operator": "polynomial",
            "mutation_probability": 1.0 / 30.0,
            "exploration_strength": 20.0,
        }
        # the sampled alternatives are unaffected by the insertion
        assert [c["action"] for c in candidates_with[2:]] == [
            c["action"] for c in candidates_without[1:]
        ]
        # it is a fully branched candidate, not a placeholder
        for horizon in _HORIZONS:
            assert len(candidates_with[1]["reward"][horizon]) == _N_REPS
            assert candidates_with[1]["mean_reward"][horizon] is not None
        assert state_with["per_horizon"]["2"]["n_candidates"] == 5
        assert state_without["per_horizon"]["2"]["n_candidates"] == 4


def test_legacy_candidate_sampling_is_reproduced(
    counterfactual_fixture: dict[str, Any], tmp_path: Path
) -> None:
    """--no-include-default-action keeps sample_full_action draws intact."""
    fixture = counterfactual_fixture
    payload = cfa.run_evaluate_horizon(
        cfa.parse_args(
            _horizon_args(
                fixture, tmp_path / "legacy", "--no-include-default-action"
            )
        )
    )
    for state in payload["states"]:
        hash_seed = cfa._snapshot_hash_seed(
            "zdt1", int(state["seed"]), int(state["generation"])
        )
        alternatives = state["candidates"][1:]
        assert len(alternatives) == _N_ALTERNATIVES
        for k, candidate in enumerate(alternatives, start=1):
            rng = np.random.Generator(np.random.PCG64([hash_seed, k]))
            operator, pm, exploration = sample_full_action(
                rng, 1.0 / 30.0, FULL_ACTION_PM_MULT_RANGE
            )
            assert candidate["action"]["mutation_operator"] == operator
            assert candidate["action"]["mutation_probability"] == pytest.approx(pm)
            assert candidate["action"]["exploration_strength"] == pytest.approx(
                exploration
            )


def test_one_step_evaluate_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-step path has no default candidate and is byte-identical.

    The golden hash was captured before the Phase-2.75D candidate change with
    the fixture living in ``.tmp_cfverify`` and referenced by *relative* paths
    (the run config records those path strings, so the directory name is part
    of the bytes). The test therefore rebuilds the fixture under the same
    relative name from the repository root.
    """
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    root = Path(".tmp_cfverify")
    if root.exists():
        shutil.rmtree(root)
    try:
        _write_snapshots(root)
        out_dir = root / "baseline"
        cfa.run_evaluate(
            cfa.parse_args(
                [
                    "evaluate",
                    "--problem", "zdt1",
                    "--snapshots-dir", str(root / "snapshots"),
                    "--out-dir", str(out_dir),
                    "--controller", str(root / "controller.pt"),
                    "--encoder", str(root / "encoder.json"),
                    "--n-alternatives", "4",
                    "--n-reps", "2",
                ]
            )
        )
        out_path = out_dir / "counterfactual_zdt1.json"
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        for state in payload["states"]:
            kinds = [candidate["kind"] for candidate in state["candidates"]]
            assert kinds == ["controller"] + ["alternative"] * 4
            assert "default" not in kinds
            assert len(state["candidates"]) == 5
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        assert digest == GOLDEN_ONE_STEP_SHA256, (
            "the one-step evaluate output changed; the Phase-2.75D candidate "
            "fix must not touch it"
        )
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_default_candidate_is_not_added_to_the_one_step_parser() -> None:
    """The new switch exists on evaluate-horizon only."""
    horizon_args = cfa.parse_args(
        ["evaluate-horizon", "--problem", "zdt1", "--controller", "c", "--encoder", "e"]
    )
    assert horizon_args.include_default_action is True
    assert (
        cfa.parse_args(
            [
                "evaluate-horizon", "--problem", "zdt1", "--controller", "c",
                "--encoder", "e", "--no-include-default-action",
            ]
        ).include_default_action
        is False
    )
    one_step = cfa.parse_args(
        ["evaluate", "--problem", "zdt1", "--controller", "c", "--encoder", "e"]
    )
    assert not hasattr(one_step, "include_default_action")
    assert cfa.DEFAULT_ACTION_OPERATOR == "polynomial"
    assert cfa.DEFAULT_ACTION_MULTIPLIER == 1.0
    assert cfa.DEFAULT_ACTION_EXPLORATION == 20.0


# --- Task 2: Target C --------------------------------------------------------


def test_future_improvement_target_is_hand_computable() -> None:
    """Target C = (mean_reward + hv_before) - hv_before = the absolute gain."""
    means = {5: [0.10, 0.30, 0.20], 10: [0.20, 0.60, 0.40]}
    targets, baselines = bids.advantage_targets(
        means,
        baseline="future_improvement",
        horizons=[5, 10],
        default_index=0,
        hv_before=0.25,
    )
    assert targets[5] == pytest.approx([0.10, 0.30, 0.20])
    assert targets[10] == pytest.approx([0.20, 0.60, 0.40])
    assert baselines == {5: 0.0, 10: 0.0}
    with pytest.raises(ValueError, match="hv_before"):
        bids.advantage_targets(
            means,
            baseline="future_improvement",
            horizons=[5],
            default_index=0,
        )
    # it is not final_hv: that one is centred at the proxy horizon and copied
    final_targets, final_baselines = bids.advantage_targets(
        means, baseline="final_hv", horizons=[5, 10], default_index=0
    )
    assert final_targets[5] == pytest.approx([-0.20, 0.20, 0.0])
    assert final_targets[5] == final_targets[10]
    assert final_baselines[5] == pytest.approx(0.40)
    assert "future_improvement" in bids.ADVANTAGE_BASELINES
    assert bids.ADVANTAGE_BASELINES[0] == "state_mean"


def test_default_kind_candidate_is_the_baseline() -> None:
    """A kind='default' candidate wins the 'exact' rule over the controller."""
    candidates = [
        {"kind": "controller", "action": {"mutation_operator": "gaussian",
                                          "mutation_probability": 0.05,
                                          "exploration_strength": 0.1}},
        {"kind": "alternative", "action": {"mutation_operator": "polynomial",
                                           "mutation_probability": 0.02,
                                           "exploration_strength": 5.0}},
        {"kind": "default", "action": {"mutation_operator": "polynomial",
                                       "mutation_probability": 1.0 / 30.0,
                                       "exploration_strength": 20.0}},
    ]
    assert bids.select_default_candidate(candidates, 30) == (2, "exact")
    controller_only = [candidates[0], candidates[1]]
    assert bids.select_default_candidate(controller_only, 30) == (0, "controller")


def test_v2_builder_uses_the_default_candidate_for_target_b(tmp_path: Path) -> None:
    """End to end: the default_action target is measured against that row."""
    root = tmp_path / "v2"
    input_dir = root / "counterfactual"
    input_dir.mkdir(parents=True)
    snapshots_dir = root / "snapshots"
    snapshots_dir.mkdir()
    encoder = StateEncoder(10).fit(_trajectories())
    encoder_path = root / "encoder.json"
    encoder.save(encoder_path)
    history = [merge_state_reward(tr) for tr in _trajectories()[0]]
    with (snapshots_dir / "zdt1__seed1000__gen50.pkl").open("wb") as fh:
        pickle.dump({"history": history}, fh)

    def _candidate(index: int, kind: str, multiplier: float, mean: float) -> dict:
        return {
            "index": index,
            "kind": kind,
            "action": {
                "mutation_operator": "polynomial",
                "mutation_probability": multiplier / 30.0,
                "exploration_strength": 20.0,
            },
            "reward": {"5": [mean, mean, mean]},
            "mean_reward": {"5": mean},
        }

    payload = {
        "problem": "zdt1",
        "config": {"horizons": [5], "generations": 100, "ref_point": [1.1, 1.1],
                   "n_reference_points": 200},
        "states": [
            {
                "problem": "zdt1",
                "seed": 1000,
                "generation": 50,
                "state_metrics": {"hv": 0.25},
                "candidates": [
                    _candidate(0, "controller", 4.0, 0.30),
                    _candidate(1, "default", 1.0, 0.10),
                    _candidate(2, "alternative", 0.5, 0.20),
                ],
            }
        ],
    }
    with (input_dir / "counterfactual_horizon_zdt1.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(payload, fh)

    out_dir = root / "out"
    meta = bids.run_build(
        bids.parse_args(
            [
                "--input-dir", str(input_dir),
                "--snapshots-dir", str(snapshots_dir),
                "--encoder", str(encoder_path),
                "--out-dir", str(out_dir),
                "--horizons", "5",
                "--advantage-baseline", "state_mean", "default_action",
                "future_improvement",
            ]
        )
    )
    assert meta["default_action"]["rule_counts"] == {"zdt1:exact": 1}
    with np.load(out_dir / "default_action" / "intervention_dataset_zdt1.npz") as arrays:
        # 0.30 - 0.10, 0.10 - 0.10, 0.20 - 0.10 (baseline = the default row)
        assert arrays["y_adv"].tolist() == pytest.approx([0.20, 0.0, 0.10])
        assert arrays["candidate_kind"].tolist() == [
            "controller",
            "default",
            "alternative",
        ]
    with np.load(out_dir / "future_improvement" / "intervention_dataset_zdt1.npz") as arrays:
        # Target C keeps the absolute gains (no centering)
        assert arrays["y_adv"].tolist() == pytest.approx([0.30, 0.10, 0.20])
        assert arrays["baseline_value"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    with np.load(out_dir / "state_mean" / "intervention_dataset_zdt1.npz") as arrays:
        assert arrays["y_adv"].tolist() == pytest.approx([0.10, -0.10, 0.0])


# --- Task 3: target comparison ----------------------------------------------


@pytest.fixture(scope="module")
def target_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic three-target dataset in the v2 layout."""
    root = tmp_path_factory.mktemp("targets_dataset")
    input_dir = root / "counterfactual"
    input_dir.mkdir(parents=True)
    snapshots_dir = root / "snapshots"
    snapshots_dir.mkdir(parents=True)
    encoder = StateEncoder(10).fit(_trajectories())
    encoder_path = root / "encoder.json"
    encoder.save(encoder_path)
    history = [merge_state_reward(tr) for tr in _trajectories()[0]]
    rng = np.random.Generator(np.random.PCG64(0))
    states = []
    for state_index in range(12):
        seed = 1000 + state_index
        generation = 50 + state_index
        with (snapshots_dir / f"zdt1__seed{seed}__gen{generation}.pkl").open("wb") as fh:
            pickle.dump({"history": history}, fh)
        candidates = []
        for candidate_index in range(4):
            kind = (
                "controller"
                if candidate_index == 0
                else "default"
                if candidate_index == 1
                else "alternative"
            )
            multiplier = 0.5 + candidate_index
            # the action effect grows with the candidate index and the horizon
            base = (candidate_index - 1.5) * 0.05 + 0.2
            candidates.append(
                {
                    "index": candidate_index,
                    "kind": kind,
                    "action": {
                        "mutation_operator": "polynomial",
                        "mutation_probability": multiplier / 30.0,
                        "exploration_strength": 20.0 + candidate_index,
                    },
                    "reward": {
                        "5": [base - 0.001, base, base + 0.001],
                        "10": [base * 2 - 0.001, base * 2, base * 2 + 0.001],
                        "20": [base * 4 - 0.001, base * 4, base * 4 + 0.001],
                    },
                    "mean_reward": {"5": base, "10": base * 2, "20": base * 4},
                }
            )
        states.append(
            {
                "problem": "zdt1",
                "seed": seed,
                "generation": generation,
                "state_metrics": {"hv": 0.2 + 0.01 * state_index},
                "candidates": candidates,
            }
        )
    with (input_dir / "counterfactual_horizon_zdt1.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(
            {
                "problem": "zdt1",
                "config": {"horizons": [5, 10, 20], "generations": 100,
                           "ref_point": [1.1, 1.1], "n_reference_points": 200},
                "states": states,
            },
            fh,
        )
    dataset_dir = root / "dataset"
    bids.run_build(
        bids.parse_args(
            [
                "--input-dir", str(input_dir),
                "--snapshots-dir", str(snapshots_dir),
                "--encoder", str(encoder_path),
                "--out-dir", str(dataset_dir),
                "--horizons", "5", "10", "20",
                "--advantage-baseline", "all",
            ]
        )
    )
    return dataset_dir


def test_compare_targets_end_to_end(
    target_dataset: Path, tmp_path: Path
) -> None:
    """All three targets run on one shared split and report every metric."""
    out_path = tmp_path / "comparison.json"
    payload = cat.main(
        [
            "--dataset-dir", str(target_dataset),
            "--out", str(out_path),
            "--model-dir", str(tmp_path / "models"),
            "--epochs", "20",
            "--hidden-dims", "16",
            "--batch-size", "16",
            "--val-fraction", "0.25",
            "--horizons", "5", "10", "20",
        ]
    )
    assert out_path.is_file()
    with out_path.open("r", encoding="utf-8") as fh:
        written = json.load(fh)
    assert set(written) >= {
        "config", "data", "split", "targets", "ranking",
        "best_target_by_spearman",
    }
    assert written["config"]["targets"] == list(cat.DEFAULT_TARGETS)
    assert written["config"]["contrastive"] == "on"
    # the v2 layout: 60 state + 9 problem + 3 runtime + 4 action columns
    # Phase-2.75D: the v2 builder now defaults to the compact 64-dim layout
    # (state 60 + action 4), so the comparison consumes that representation.
    assert written["data"]["input_dim"] == 64
    assert written["data"]["n_samples"] > 0
    assert written["split"]["shared_across_targets"] is True
    train_keys = set(written["split"]["train_state_keys"])
    val_keys = set(written["split"]["val_state_keys"])
    assert train_keys and val_keys and train_keys.isdisjoint(val_keys)
    assert set(written["targets"]) == set(cat.DEFAULT_TARGETS)
    for target in cat.DEFAULT_TARGETS:
        entry = written["targets"][target]
        assert Path(entry["model"]).is_file()
        assert len(entry["train_loss"]) == 20
        assert set(entry["per_horizon"]) == {"5", "10", "20"}
        for metrics in entry["per_horizon"].values():
            assert set(metrics) >= {
                "n_groups",
                "spearman_mean",
                "kendall_mean",
                "oracle_hit_rate",
                "regret_mean",
                "oracle_gap_mean",
            }
        assert set(entry["overall"]) >= set(cat._SCALAR_METRICS)
    # 3 targets x (3 horizons + 1 overall row)
    assert len(written["ranking"]) == 3 * (3 + 1)
    assert {row["target"] for row in written["ranking"]} == set(cat.DEFAULT_TARGETS)
    assert {row["horizon"] for row in written["ranking"]} == {5, 10, 20, None}
    assert set(written["best_target_by_spearman"]) == {"5", "10", "20", "overall"}
    # the shared split means every target saw exactly the same states
    assert payload["split"]["train_state_keys"] == written["split"]["train_state_keys"]
    assert written["data"]["n_states"] == 12


def test_compare_targets_rejects_mismatched_features(
    target_dataset: Path, tmp_path: Path
) -> None:
    """Datasets that disagree on X are refused instead of silently mixed."""
    broken = tmp_path / "dataset"
    shutil.copytree(target_dataset, broken)
    path = broken / "future_improvement" / "intervention_dataset_zdt1.npz"
    with np.load(path) as arrays:
        stored = {name: arrays[name] for name in arrays.files}
    stored["X"] = stored["X"] + 1.0
    np.savez_compressed(path, **stored)
    with pytest.raises(ValueError, match="different feature matrix"):
        cat.main(
            [
                "--dataset-dir", str(broken),
                "--out", str(tmp_path / "comparison.json"),
                "--model-dir", str(tmp_path / "models"),
                "--epochs", "2",
                "--hidden-dims", "8",
                "--horizons", "5",
            ]
        )


def test_compare_targets_cli_defaults(target_dataset: Path) -> None:
    """CLI defaults match the documented Phase-2.75D grid."""
    args = cat.parse_args([])
    assert args.dataset_dir == cat.DEFAULT_DATASET_DIR
    assert args.out == cat.DEFAULT_OUT
    assert tuple(args.targets) == cat.DEFAULT_TARGETS
    assert args.epochs == 300
    assert args.contrastive == "on"
    assert tuple(args.horizons) == cat.DEFAULT_HORIZONS
    assert args.keep_controller is False
    assert cat.parse_args(["--keep-controller"]).keep_controller is True
    with pytest.raises(FileNotFoundError, match="dataset directory"):
        cat.main(["--dataset-dir", str(target_dataset / "absent")])
