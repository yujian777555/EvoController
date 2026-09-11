from __future__ import annotations

"""Tests for the Phase-2.75C problem-aware intervention dataset (v2).

Everything is synthetic: a hand-written ``counterfactual_horizon_*.json`` plus
matching snapshot pickle and trajectory-fitted encoder. The tests pin

* the 76-column layout and the machine-readable ``feature_layout`` ranges,
* the hand-computed advantage of all three definitions
  (``state_mean`` / ``default_action`` / ``final_hv``),
* the default-action stand-in rules (exact -> controller -> nearest),
* that v1 stays bit-identical on the primary definition (regression guard).
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks import get_problem
from controller.dataset import merge_state_reward
from controller.problem_features import PROBLEM_FEATURE_NAMES, problem_feature_vector
from controller.state_encoder import StateEncoder
from experiments import build_intervention_dataset as v1
from experiments import build_intervention_dataset_v2 as v2

_WINDOW = 10  # v2 declares a 60-column state block
_N_VARS = 30
_HORIZONS = (5, 10, 20)
_N_REPS = 3
_REP_DELTA = 0.001


# --- fixture -----------------------------------------------------------------


def _trajectories(n_traj: int = 2, generations: int = 12) -> list[list[dict[str, Any]]]:
    """Two recorder-schema trajectories used to fit the tiny encoder."""
    trajectories: list[list[dict[str, Any]]] = []
    for j in range(n_traj):
        hv = 0.30 + 0.05 * j
        igd = 0.50 - 0.02 * j
        transitions = []
        for t in range(generations):
            delta_hv = 0.0 if t == 0 else 0.006 + 0.001 * ((t + j) % 3)
            delta_igd = 0.0 if t == 0 else 0.005 + 0.001 * ((t + j + 1) % 3)
            hv += delta_hv
            igd -= delta_igd
            transitions.append(
                {
                    "generation": t,
                    "state": {"generation": t, "hv": hv, "igd": igd, "diversity": 0.25},
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


def _history() -> list[dict[str, Any]]:
    return [merge_state_reward(tr) for tr in _trajectories()[0][:6]]


def _action(
    index: int, multiplier: float | None = None, operator: str = "polynomial"
) -> dict[str, Any]:
    """A candidate action; deliberately never the NSGA-II default tuple."""
    return {
        "mutation_operator": operator,
        "mutation_probability": (
            multiplier if multiplier is not None else 1.5 + index
        )
        / _N_VARS,
        "exploration_strength": 25.0 + 5.0 * index,
    }


def _payload(
    *,
    problem: str = "zdt1",
    seed: int = 1000,
    generation: int = 50,
    horizons: tuple[int, ...] = _HORIZONS,
    mean_rewards: dict[int, list[float]] | None = None,
    actions: list[dict[str, Any]] | None = None,
    kinds: list[str] | None = None,
    generations: int = 100,
    hv: float = 0.25,
) -> dict[str, Any]:
    """A minimal ``counterfactual_horizon_{problem}.json`` payload."""
    if actions is None:
        actions = [_action(0), _action(1), _action(2)]
    if kinds is None:
        kinds = ["controller", "alternative", "alternative"]
    if mean_rewards is None:
        # proportional to the horizon; at h=5 this is [0.10, 0.30, 0.20]
        mean_rewards = {
            h: [0.02 * h, 0.06 * h, 0.04 * h] for h in horizons
        }
    candidates = []
    for index, (action, kind) in enumerate(zip(actions, kinds)):
        candidates.append(
            {
                "index": index,
                "kind": kind,
                "action": action,
                "future_hv": {str(h): [0.0] * _N_REPS for h in horizons},
                "reward": {
                    str(h): [
                        mean_rewards[h][index] + _REP_DELTA * (rep - 1)
                        for rep in range(_N_REPS)
                    ]
                    for h in horizons
                },
                "mean_reward": {str(h): mean_rewards[h][index] for h in horizons},
                "predicted_reward": None,
            }
        )
    return {
        "problem": problem,
        "config": {
            "horizons": list(horizons),
            "n_alternatives": len(candidates) - 1,
            "n_reps": _N_REPS,
            "generations": int(generations),
            "ref_point": [1.1, 1.1],
            "n_reference_points": 200,
        },
        "states": [
            {
                "problem": problem,
                "seed": seed,
                "generation": generation,
                "state_metrics": {"hv": hv, "igd": 0.4},
                "controller_action": candidates[0]["action"],
                "candidates": candidates,
                "per_horizon": {},
            }
        ],
        "summary": {},
    }


def _write_fixture(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Write payload + snapshot + encoder; return their paths and objects."""
    root.mkdir(parents=True, exist_ok=True)
    input_dir = root / "counterfactual"
    input_dir.mkdir(exist_ok=True)
    path = input_dir / f"counterfactual_horizon_{payload['problem']}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    snapshots_dir = root / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    state = payload["states"][0]
    snapshot = snapshots_dir / (
        f"{payload['problem']}__seed{state['seed']}__gen{state['generation']}.pkl"
    )
    with snapshot.open("wb") as fh:
        pickle.dump({"history": _history()}, fh)
    encoder = StateEncoder(_WINDOW).fit(_trajectories())
    encoder_path = root / "encoder.json"
    encoder.save(encoder_path)
    return {
        "root": root,
        "input_dir": input_dir,
        "snapshots_dir": snapshots_dir,
        "encoder": encoder,
        "encoder_path": encoder_path,
        "payload": payload,
    }


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Encoder + input/snapshot fixture with one intervention file."""
    return _write_fixture(tmp_path_factory.mktemp("phase275c"), _payload())


# --- Task 1: 76-column layout ------------------------------------------------


def test_feature_layout_is_contiguous_and_declared() -> None:
    """The declared layout covers exactly the 76 columns, in order."""
    layout = v2.feature_layout()
    assert [block["name"] for block in layout] == [
        "state",
        "problem",
        "runtime",
        "action",
    ]
    assert layout[0]["start"] == 0
    for previous, current in zip(layout, layout[1:]):
        assert current["start"] == previous["stop"]
    assert layout[-1]["stop"] == v2.FEATURE_DIM == 76
    assert layout[1]["columns"] == list(PROBLEM_FEATURE_NAMES)
    assert layout[2]["columns"] == list(v2.RUNTIME_FEATURE_NAMES)
    assert layout[0]["stop"] == v2.STATE_BLOCK == 60
    assert v2.PROBLEM_BLOCK == 9 and v2.RUNTIME_BLOCK == 3 and v2.ACTION_BLOCK == 4


def test_feature_blocks_match_their_sources(fixture: dict[str, Any]) -> None:
    """Every block equals the quantity it claims to carry."""
    extracted = v2.extract_state_rows(
        fixture["payload"],
        fixture["encoder"],
        horizons=list(_HORIZONS),
        snapshots_dir=fixture["snapshots_dir"],
        problem_mean=np.zeros(v2.PROBLEM_BLOCK),
        problem_std=np.ones(v2.PROBLEM_BLOCK),
    )
    columns = extracted["columns"]
    problem = get_problem("zdt1")
    n_candidates = len(fixture["payload"]["states"][0]["candidates"])
    horizon = _HORIZONS[0]
    state_block = np.asarray(fixture["encoder"].transform(_history()), dtype=np.float64)
    expected_problem = np.asarray(problem_feature_vector(problem), dtype=np.float64)
    hv_reference = extracted["hv_reference"]
    assert hv_reference > 0.0

    for candidate_index in range(n_candidates):
        row = np.asarray(columns["X"][candidate_index], dtype=np.float64)
        assert row.shape == (v2.FEATURE_DIM,)
        np.testing.assert_array_equal(row[:60], state_block)
        np.testing.assert_allclose(row[60:69], expected_problem, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            row[69:72],
            [0.25, 50.0 / 100.0, 0.25 / hv_reference],
            rtol=0,
            atol=1e-12,
        )
        action = fixture["payload"]["states"][0]["candidates"][candidate_index]["action"]
        np.testing.assert_allclose(
            row[72:], v1.action_features(action, _N_VARS), rtol=0, atol=1e-12
        )
        assert columns["horizon"][candidate_index] == horizon
        assert columns["baseline_value"][candidate_index] == pytest.approx(0.20)


def test_problem_block_is_zscored_over_the_built_problems(
    fixture: dict[str, Any], tmp_path: Path
) -> None:
    """The problem block follows the recorded z-score statistics."""
    args = v2.parse_args(
        [
            "--input-dir", str(fixture["input_dir"]),
            "--snapshots-dir", str(fixture["snapshots_dir"]),
            "--encoder", str(fixture["encoder_path"]),
            "--out-dir", str(tmp_path / "out"),
            "--horizons", *[str(h) for h in _HORIZONS],
        ]
    )
    meta = v2.run_build(args)
    mean = np.asarray(meta["problem_feature_mean"], dtype=np.float64)
    std = np.asarray(meta["problem_feature_std"], dtype=np.float64)
    # a single problem is its own mean, so the block is all zeros
    assert mean.shape == (v2.PROBLEM_BLOCK,)
    raw = np.asarray(problem_feature_vector(get_problem("zdt1")), dtype=np.float64)
    np.testing.assert_allclose((raw - mean) / std, np.zeros(v2.PROBLEM_BLOCK), atol=1e-12)
    np.testing.assert_allclose(std, np.ones(v2.PROBLEM_BLOCK))
    assert meta["runtime_feature_names"] == list(v2.RUNTIME_FEATURE_NAMES)


def test_runtime_context_validation() -> None:
    """Runtime features are the documented three quantities."""
    block = v2.runtime_context_block(
        hv_before=0.5, generation=25, max_generation=100, hv_reference=2.0
    )
    np.testing.assert_allclose(block, [0.5, 0.25, 0.25])
    with pytest.raises(ValueError, match="max_generation"):
        v2.runtime_context_block(
            hv_before=0.5, generation=1, max_generation=0, hv_reference=1.0
        )
    with pytest.raises(ValueError, match="hv_reference"):
        v2.runtime_context_block(
            hv_before=0.5, generation=1, max_generation=10, hv_reference=0.0
        )


# --- Task 2: advantage definitions ------------------------------------------


def test_advantage_targets_hand_computed() -> None:
    """All three definitions against hand-computed numbers."""
    means = {5: [0.10, 0.30, 0.20], 10: [0.20, 0.60, 0.40], 20: [0.30, 0.90, 0.60]}
    state_mean, baselines = v2.advantage_targets(
        means, baseline="state_mean", horizons=[5, 10, 20], default_index=0
    )
    assert baselines[5] == pytest.approx(0.20)
    assert baselines[20] == pytest.approx(0.60)
    assert state_mean[5] == pytest.approx([-0.10, 0.10, 0.0])
    assert state_mean[20] == pytest.approx([-0.30, 0.30, 0.0])

    default_action, default_baselines = v2.advantage_targets(
        means, baseline="default_action", horizons=[5, 10, 20], default_index=0
    )
    assert default_baselines[5] == pytest.approx(0.10)
    assert default_baselines[20] == pytest.approx(0.30)
    assert default_action[5] == pytest.approx([0.0, 0.20, 0.10])
    assert default_action[20] == pytest.approx([0.0, 0.60, 0.30])

    final, final_baselines = v2.advantage_targets(
        means, baseline="final_hv", horizons=[5, 10, 20], default_index=1
    )
    # the proxy is the largest horizon; every horizon gets its numbers
    assert all(value == pytest.approx(0.60) for value in final_baselines.values())
    for horizon in (5, 10, 20):
        assert final[horizon] == pytest.approx([-0.30, 0.30, 0.0])
    with pytest.raises(ValueError, match="unknown advantage baseline"):
        v2.advantage_targets(means, baseline="nope", horizons=[5], default_index=0)
    with pytest.raises(ValueError, match="missing from the state record"):
        v2.advantage_targets(means, baseline="state_mean", horizons=[7], default_index=0)
    assert v2.proxy_horizon([5, 20, 10]) == 20
    with pytest.raises(ValueError, match="must not be empty"):
        v2.proxy_horizon([])


def test_default_action_selection_rules() -> None:
    """exact -> controller -> nearest, with the distance metric documented."""
    default_action = {
        "mutation_operator": "polynomial",
        "mutation_probability": 1.0 / _N_VARS,
        "exploration_strength": 20.0,
    }
    exact = [
        {"kind": "alternative", "action": _action(1)},
        {"kind": "controller", "action": default_action},
    ]
    assert v2.select_default_candidate(exact, _N_VARS) == (1, "exact")

    controller_only = [
        {"kind": "alternative", "action": _action(1)},
        {"kind": "controller", "action": _action(2)},
    ]
    assert v2.select_default_candidate(controller_only, _N_VARS) == (1, "controller")

    nearest_only = [
        {"kind": "alternative", "action": _action(0, multiplier=4.0)},
        {"kind": "alternative", "action": _action(1, multiplier=1.2)},
        {"kind": "alternative", "action": _action(2, multiplier=0.3)},
    ]
    index, rule = v2.select_default_candidate(nearest_only, _N_VARS)
    assert rule == "nearest"
    assert index == 1  # multiplier 1.2 is closest to the default 1.0
    # a Gaussian candidate is penalized relative to a polynomial one
    mixed = [
        {"kind": "alternative", "action": _action(0, multiplier=2.0, operator="gaussian")},
        {"kind": "alternative", "action": _action(1, multiplier=3.0)},
    ]
    assert v2.select_default_candidate(mixed, _N_VARS)[0] == 1
    distance_same, same = v2.default_action_distance(default_action, _N_VARS)
    assert distance_same == pytest.approx(0.0)
    assert same is True
    distance_other, same_other = v2.default_action_distance(
        {"mutation_operator": "gaussian", "mutation_probability": 1.0 / _N_VARS,
         "exploration_strength": 20.0},
        _N_VARS,
    )
    assert same_other is False
    assert distance_other == pytest.approx(v2.OPERATOR_PENALTY)
    with pytest.raises(ValueError, match="empty list"):
        v2.select_default_candidate([], _N_VARS)


def test_run_build_advantages_match_the_payload(
    fixture: dict[str, Any], tmp_path: Path
) -> None:
    """End-to-end: each npz carries hand-computed targets for its definition."""
    out_dir = tmp_path / "out_all"
    args = v2.parse_args(
        [
            "--input-dir", str(fixture["input_dir"]),
            "--snapshots-dir", str(fixture["snapshots_dir"]),
            "--encoder", str(fixture["encoder_path"]),
            "--out-dir", str(out_dir),
            "--horizons", *[str(h) for h in _HORIZONS],
            "--feature-set", "context",
        ]
    )
    meta = v2.run_build(args)
    assert meta["feature_dim"] == 76
    assert meta["proxy_horizon"] == 20
    assert "proxy" in meta["limitation"]
    assert meta["default_action"]["selection_order"] == [
        "exact",
        "controller",
        "nearest",
    ]
    # the fixture's candidate 0 is the controller, so that rule fired
    assert meta["default_action"]["rule_counts"] == {"zdt1:controller": 1}
    assert set(meta["artifacts"]) == set(v2.ADVANTAGE_BASELINES)
    # v1 meta fields are all present
    assert set(meta) >= {
        "config", "n_samples", "n_states", "feature_dim", "files_used",
        "files_skipped", "per_problem", "per_horizon",
    }

    expected_state_mean = [0.10 - 0.20, 0.30 - 0.20, 0.20 - 0.20]
    expected_default = [0.10 - 0.10, 0.30 - 0.10, 0.20 - 0.10]
    # the proxy horizon is 20, where the state mean is 0.80
    expected_final = [0.40 - 0.80, 1.20 - 0.80, 0.80 - 0.80]
    for baseline, expected_first_horizon in (
        ("state_mean", expected_state_mean),
        ("default_action", expected_default),
        ("final_hv", expected_final),
    ):
        path = out_dir / baseline / "intervention_dataset_zdt1.npz"
        assert path.is_file()
        assert (out_dir / baseline / "intervention_meta.json").is_file()
        with np.load(path) as arrays:
            assert arrays["X"].shape == (9, 76)  # 3 candidates x 3 horizons
            assert set(arrays["horizon"].tolist()) == {5, 10, 20}
            np.testing.assert_allclose(
                arrays["y_adv"][:3], expected_first_horizon, atol=1e-12
            )
            if baseline == "final_hv":
                # every horizon carries the proxy-horizon target
                np.testing.assert_allclose(
                    arrays["y_adv"], np.tile(expected_final, 3), atol=1e-12
                )
            np.testing.assert_allclose(
                arrays["state_block"], arrays["X"][:, :60], atol=0
            )
            np.testing.assert_allclose(
                arrays["action_features"], arrays["X"][:, 72:], atol=0
            )
    summary = meta["advantage_baselines"]
    assert summary["state_mean"]["n_samples"] == 9
    # the mean advantage is zero for the state_mean definition, by construction
    assert summary["state_mean"]["per_horizon"]["5"]["mean"] == pytest.approx(0.0)
    assert summary["default_action"]["per_horizon"]["5"]["mean"] == pytest.approx(0.1)
    assert summary["final_hv"]["per_horizon"]["20"]["mean"] == pytest.approx(0.0)


def test_single_baseline_selection_writes_one_file(
    fixture: dict[str, Any], tmp_path: Path
) -> None:
    """``--advantage-baseline`` restricts which definitions are written."""
    out_dir = tmp_path / "out_single"
    args = v2.parse_args(
        [
            "--input-dir", str(fixture["input_dir"]),
            "--snapshots-dir", str(fixture["snapshots_dir"]),
            "--encoder", str(fixture["encoder_path"]),
            "--out-dir", str(out_dir),
            "--horizons", "5", "10",
            "--advantage-baseline", "default_action",
        ]
    )
    meta = v2.run_build(args)
    assert meta["config"]["advantage_baselines"] == ["default_action"]
    assert meta["proxy_horizon"] is None and meta["limitation"] is None
    written = sorted(
        path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*.npz")
    )
    assert written == ["default_action/intervention_dataset_zdt1.npz"]
    assert (out_dir / "intervention_meta_v2.json").is_file()


def test_missing_snapshot_and_empty_inputs_error(
    fixture: dict[str, Any], tmp_path: Path
) -> None:
    """Missing histories and unusable inputs raise explicit errors."""
    payload = _payload(seed=4242, generation=77)
    with pytest.raises(FileNotFoundError, match="zdt1__seed4242__gen77.pkl"):
        v2.extract_state_rows(
            payload,
            fixture["encoder"],
            horizons=[5],
            snapshots_dir=fixture["snapshots_dir"],
            problem_mean=np.zeros(9),
            problem_std=np.ones(9),
        )
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no counterfactual_horizon"):
        v2.run_build(
            v2.parse_args(
                [
                    "--input-dir", str(empty),
                    "--encoder", str(fixture["encoder_path"]),
                    "--out-dir", str(tmp_path / "out_empty"),
                ]
            )
        )
    # a file whose horizons do not intersect the request is skipped and, with
    # nothing left, the build reports the reason
    other = tmp_path / "other"
    _write_fixture(other, _payload(horizons=(7, 9)))
    with pytest.raises(ValueError, match="no samples built"):
        v2.run_build(
            v2.parse_args(
                [
                    "--input-dir", str(other / "counterfactual"),
                    "--snapshots-dir", str(other / "snapshots"),
                    "--encoder", str(other / "encoder.json"),
                    "--out-dir", str(tmp_path / "out_other"),
                    "--horizons", "5", "10", "20",
                ]
            )
        )
    with pytest.raises(FileNotFoundError, match="encoder not found"):
        v2.run_build(
            v2.parse_args(
                [
                    "--input-dir", str(fixture["input_dir"]),
                    "--encoder", str(tmp_path / "absent.json"),
                    "--out-dir", str(tmp_path / "out_absent"),
                ]
            )
        )


# --- v1 regression guard -----------------------------------------------------


def test_v1_outputs_are_unchanged_by_v2(fixture: dict[str, Any], tmp_path: Path) -> None:
    """v1 and v2 agree bit-exactly on the shared columns and definitions.

    v1 is the published Phase-2.75 dataset recipe; v2 must extend it without
    changing it: same state block, same action block, same ``state_mean``
    target, 64 vs 76 columns.
    """
    v1_out = tmp_path / "v1"
    v2_out = tmp_path / "v2"
    horizons = [str(h) for h in _HORIZONS]
    v1.run_dataset(
        v1.parse_args(
            [
                "--input-dir", str(fixture["input_dir"]),
                "--snapshots-dir", str(fixture["snapshots_dir"]),
                "--encoder", str(fixture["encoder_path"]),
                "--out-dir", str(v1_out),
                "--horizons", *horizons,
                "--baseline", "state_mean",
            ]
        )
    )
    v2.run_build(
        v2.parse_args(
            [
                "--input-dir", str(fixture["input_dir"]),
                "--snapshots-dir", str(fixture["snapshots_dir"]),
                "--encoder", str(fixture["encoder_path"]),
                "--out-dir", str(v2_out),
                "--horizons", *horizons,
                "--advantage-baseline", "state_mean",
                "--feature-set", "context",
            ]
        )
    )
    with np.load(
        v1_out / "intervention_dataset_zdt1.npz"
    ) as old, np.load(
        v2_out / "state_mean" / "intervention_dataset_zdt1.npz"
    ) as new:
        assert old["X"].shape[1] == 64
        assert new["X"].shape[1] == 76
        assert old["X"].shape[0] == new["X"].shape[0]
        np.testing.assert_array_equal(new["X"][:, :60], old["X"][:, :60])
        np.testing.assert_array_equal(new["X"][:, 72:], old["X"][:, 60:64])
        np.testing.assert_array_equal(new["y_adv"], old["y_adv"])
        np.testing.assert_array_equal(new["mean_reward"], old["mean_reward"])
        np.testing.assert_array_equal(new["horizon"], old["horizon"])
        np.testing.assert_array_equal(new["candidate_index"], old["candidate_index"])
        np.testing.assert_array_equal(new["baseline_value"], old["baseline_value"])
        # v1's array set is still fully contained in v2's
        assert set(old.files) <= set(new.files)


def test_compact_feature_set_is_the_default_and_matches_v1_layout(
    fixture: dict[str, Any], tmp_path: Path
) -> None:
    """Phase-2.75D: compact is the default and reproduces the v1 64-dim layout."""
    v2_out = tmp_path / "v2_compact"
    horizons = [str(h) for h in _HORIZONS]
    v2.run_build(
        v2.parse_args(
            [
                "--input-dir", str(fixture["input_dir"]),
                "--snapshots-dir", str(fixture["snapshots_dir"]),
                "--encoder", str(fixture["encoder_path"]),
                "--out-dir", str(v2_out),
                "--horizons", *horizons,
                "--advantage-baseline", "state_mean",
                # no --feature-set: the default must be compact
            ]
        )
    )
    meta = json.loads(
        (v2_out / "state_mean" / "intervention_meta.json").read_text(encoding="utf-8")
    )
    assert meta["feature_dim"] == 64
    assert [block["name"] for block in meta["feature_layout"]] == ["state", "action"]
    with np.load(v2_out / "state_mean" / "intervention_dataset_zdt1.npz") as compact:
        assert compact["X"].shape[1] == 64
        # The compact layout is exactly [state 60][action 4].
        np.testing.assert_array_equal(
            compact["X"][:, :60], compact["state_block"]
        )
        np.testing.assert_array_equal(
            compact["X"][:, 60:], compact["action_features"]
        )
