from __future__ import annotations

"""Tests for the Phase-2.75 macro action set and its SNR analysis.

The SNR tests use two hand-made intervention worlds with known structure:

* **aligned** — candidates are repeated macro representatives whose outcomes
  depend only on the macro (macro regions are homogeneous), so the macro
  discretisation concentrates the signal: ``variance_explained_by_macro`` is
  ~1 and ``macro_snr`` exceeds ``continuous_snr``;
* **misaligned** — macro members disagree as much as different macros do, so
  the macro means carry no information: ``variance_explained_by_macro``
  collapses and both macro SNRs drop far below ``continuous_snr``.

No real intervention file or trained model is needed.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller import macro_actions as mac
from controller.dataset import build_outcome_samples, merge_state_reward
from controller.state_encoder import StateEncoder

_HORIZON = 5
_WINDOW = 3
_N_VARS = 30
_REP_DELTA = 0.001


# --- synthetic intervention files -------------------------------------------


def _candidate(
    index: int,
    operator: str,
    multiplier: float,
    exploration: float,
    mean_reward: float,
    *,
    horizon: int = _HORIZON,
) -> dict[str, Any]:
    """One intervention candidate with symmetric replicate rewards."""
    return {
        "index": index,
        "kind": "controller" if index == 0 else "alternative",
        "action": {
            "mutation_operator": operator,
            "mutation_probability": multiplier / _N_VARS,
            "exploration_strength": exploration,
        },
        "future_hv": {str(horizon): [0.0, 0.0, 0.0]},
        "reward": {
            str(horizon): [
                mean_reward - _REP_DELTA,
                mean_reward,
                mean_reward + _REP_DELTA,
            ]
        },
        "mean_reward": {str(horizon): mean_reward},
        "predicted_reward": None,
    }


def _write_intervention(
    path: Path,
    candidates: list[dict[str, Any]],
    *,
    horizon: int = _HORIZON,
    problem: str = "zdt1",
    seed: int = 1000,
    generation: int = 50,
) -> Path:
    """Write a minimal ``counterfactual_horizon_{problem}.json``."""
    payload = {
        "problem": problem,
        "config": {
            "horizons": [horizon],
            "n_alternatives": max(len(candidates) - 1, 0),
            "n_reps": 3,
            "generations": 100,
        },
        "states": [
            {
                "problem": problem,
                "seed": seed,
                "generation": generation,
                "state_metrics": {"hv": 0.5, "igd": 0.4},
                "candidates": candidates,
                "per_horizon": {},
            }
        ],
        "summary": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def _aligned_candidates() -> list[dict[str, Any]]:
    """Four macro representatives, repeated twice, with macro-driven outcomes."""
    plan = [
        ("explore_high", 1.0),
        ("neutral", 0.5),
        ("exploit_high", -0.5),
        ("operator_switch_gaussian", -1.0),
    ]
    candidates: list[dict[str, Any]] = []
    for repeat in range(2):
        for name, reward in plan:
            macro = mac.MACRO_ACTIONS[name]
            candidates.append(
                _candidate(
                    len(candidates),
                    str(macro["mutation_operator"]),
                    float(macro["multiplier"]),
                    float(macro["exploration_strength"]),
                    reward,
                )
            )
    return candidates


def _misaligned_candidates() -> list[dict[str, Any]]:
    """Same macro representatives, but outcomes alternate inside each macro."""
    plan = [
        ("explore_high", 1.0),
        ("neutral", 1.0),
        ("exploit_high", 1.0),
        ("operator_switch_gaussian", 1.0),
    ]
    candidates: list[dict[str, Any]] = []
    for repeat in range(2):
        for name, _ in plan:
            macro = mac.MACRO_ACTIONS[name]
            # the second member of every macro flips the sign: macro identity
            # says nothing about the outcome
            reward = 1.0 if repeat == 0 else -1.0
            candidates.append(
                _candidate(
                    len(candidates),
                    str(macro["mutation_operator"]),
                    float(macro["multiplier"]),
                    float(macro["exploration_strength"]),
                    reward,
                )
            )
    return candidates


# --- macro table -------------------------------------------------------------


def test_macro_table_covers_the_intent_spectrum() -> None:
    """Enough macros, both operators, and the full explore..exploit axis."""
    assert 5 <= len(mac.MACRO_ACTIONS) <= 8
    for name, macro in mac.MACRO_ACTIONS.items():
        assert isinstance(name, str) and name
        assert set(macro) == {
            "mutation_operator",
            "multiplier",
            "exploration_strength",
        }
    operators = {macro["mutation_operator"] for macro in mac.MACRO_ACTIONS.values()}
    assert operators == {"polynomial", "gaussian"}

    multipliers = {
        name: float(macro["multiplier"]) for name, macro in mac.MACRO_ACTIONS.items()
    }
    # a high-exploration end, a neutral default, a high-exploitation end
    explore = min(multipliers, key=lambda name: multipliers[name])
    exploit = max(multipliers, key=lambda name: multipliers[name])
    assert multipliers[exploit] >= 2.0
    assert multipliers[explore] <= 0.5
    neutral = mac.MACRO_ACTIONS["neutral"]
    assert neutral["mutation_operator"] == "polynomial"
    assert float(neutral["multiplier"]) == pytest.approx(1.0)
    assert float(neutral["exploration_strength"]) == pytest.approx(20.0)
    # exploration must be monotone along the axis: exploring uses a flatter
    # polynomial distribution (smaller eta_m) than exploiting
    assert float(mac.MACRO_ACTIONS[exploit]["exploration_strength"]) < float(
        mac.MACRO_ACTIONS[explore]["exploration_strength"]
    )


def test_macro_values_are_inside_the_sampled_action_space() -> None:
    """Every macro is a legal point of the Phase-1.5 full action space."""
    lo, hi = mac.MULTIPLIER_RANGE
    for name, macro in mac.MACRO_ACTIONS.items():
        multiplier = float(macro["multiplier"])
        exploration = float(macro["exploration_strength"])
        assert lo <= multiplier <= hi, name
        if macro["mutation_operator"] == "polynomial":
            assert mac.ETA_M_RANGE[0] <= exploration <= mac.ETA_M_RANGE[1], name
        else:
            assert mac.SIGMA_RANGE[0] <= exploration <= mac.SIGMA_RANGE[1], name


def test_module_constants_match_the_canonical_sources() -> None:
    """Local range/fallback copies cannot silently drift from their sources."""
    from controller.dataset import OUTCOME_FALLBACK_N_VARS, OPERATOR_TO_INDEX
    from experiments.generate_dataset import (
        ETA_M_SAMPLE_RANGE,
        FULL_ACTION_PM_MULT_RANGE,
        SIGMA_SAMPLE_RANGE,
    )

    assert mac.MULTIPLIER_RANGE == tuple(FULL_ACTION_PM_MULT_RANGE)
    assert mac.ETA_M_RANGE == tuple(ETA_M_SAMPLE_RANGE)
    assert mac.SIGMA_RANGE == tuple(SIGMA_SAMPLE_RANGE)
    assert mac.DEFAULT_N_VARS == OUTCOME_FALLBACK_N_VARS
    assert mac.N_ACTION_FEATURES == 4
    assert set(OPERATOR_TO_INDEX) == {"polynomial", "gaussian"}
    assert mac.macro_names() == tuple(mac.MACRO_ACTIONS)


# --- materialization ---------------------------------------------------------


def test_macro_action_returns_the_step_contract() -> None:
    """Exactly the three keys ``NSGAII.step`` takes, with pm = multiplier/n_vars."""
    for name, macro in mac.MACRO_ACTIONS.items():
        for n_vars in (10, 30):
            action = mac.macro_action(name, n_vars)
            assert set(action) == {
                "mutation_operator",
                "mutation_probability",
                "exploration_strength",
            }
            assert action["mutation_operator"] == macro["mutation_operator"]
            assert action["mutation_probability"] == pytest.approx(
                float(macro["multiplier"]) / n_vars
            )
            assert action["exploration_strength"] == pytest.approx(
                float(macro["exploration_strength"])
            )
    # the legacy fallback keeps the absolute probability inside [0, 1]
    fallback = mac.macro_action("explore_high")
    assert fallback["mutation_probability"] == pytest.approx(
        6.0 / mac.DEFAULT_N_VARS
    )
    with pytest.raises(ValueError, match="unknown macro"):
        mac.macro_action("nope")
    with pytest.raises(ValueError, match="n_vars"):
        mac.macro_action("neutral", 0)


def test_macro_actions_are_injectable_into_nsga2() -> None:
    """The materialized dict is accepted by the algorithm and reported back."""
    from algorithms.nsga2 import NSGAII, OperatorConfig
    from benchmarks import get_problem

    problem = get_problem("zdt1")
    algorithm = NSGAII(problem, pop_size=20, operators=OperatorConfig(), seed=0)
    algorithm.initialize()
    action = mac.macro_action("explore_high", problem.n_vars)
    algorithm.step(
        mutation_prob=action["mutation_probability"],
        mutation_operator=action["mutation_operator"],
        exploration_strength=action["exploration_strength"],
    )
    assert algorithm.current_action() == action
    assert algorithm.generation == 1


# --- feature layout ----------------------------------------------------------


def test_action_features_match_build_outcome_samples_bit_exactly() -> None:
    """The macro feature block is the dataset builder's action block."""
    trajectories = []
    for j in range(2):
        hv, igd = 0.3 + 0.05 * j, 0.5 - 0.02 * j
        transitions = []
        for t in range(12):
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
                        "mutation_probability": 2.0 / _N_VARS,
                        "exploration_strength": 15.0,
                    },
                    "reward": {"delta_hv": delta_hv, "delta_igd": delta_igd},
                }
            )
        trajectories.append(transitions)
    encoder = StateEncoder(_WINDOW).fit(trajectories)
    X, _, _, sample_indices = build_outcome_samples(
        trajectories, encoder, _WINDOW, [1, 2]
    )
    row = int(np.where(sample_indices == 1)[0][0])
    action = trajectories[0][1]["action"]
    features = mac.action_features(action, _N_VARS)
    np.testing.assert_allclose(X[row, encoder.dim :], features, rtol=0.0, atol=1e-12)
    assert features[0] == pytest.approx(2.0)
    assert features[1] == pytest.approx(15.0)
    np.testing.assert_allclose(features[2:], [1.0, 0.0])

    # a macro entry carries "multiplier" directly
    macro = mac.MACRO_ACTIONS["exploit_high"]
    macro_features = mac.action_features(macro, _N_VARS)
    np.testing.assert_allclose(
        macro_features,
        [float(macro["multiplier"]), float(macro["exploration_strength"]), 1.0, 0.0],
    )
    gaussian = mac.action_features(mac.MACRO_ACTIONS["gaussian_explore"], _N_VARS)
    np.testing.assert_allclose(gaussian[2:], [0.0, 1.0])
    with pytest.raises(ValueError, match="unsupported mutation_operator"):
        mac.action_features({"mutation_operator": "nope", "multiplier": 1.0,
                             "exploration_strength": 1.0}, _N_VARS)
    with pytest.raises(ValueError, match="multiplier"):
        mac.action_features({"mutation_operator": "polynomial",
                             "exploration_strength": 1.0}, _N_VARS)


# --- assignment --------------------------------------------------------------


def test_assign_macro_is_nearest_neighbour_inside_the_operator_family() -> None:
    """Exact macro points map to themselves; families never mix."""
    for name, macro in mac.MACRO_ACTIONS.items():
        assert (
            mac.assign_macro(
                str(macro["mutation_operator"]),
                float(macro["multiplier"]),
                float(macro["exploration_strength"]),
            )
            == name
        )
    # a Gaussian candidate never lands on a polynomial macro and vice versa
    assert mac.assign_macro("gaussian", 0.5, 0.05) in {
        "operator_switch_gaussian",
        "gaussian_explore",
    }
    assert mac.assign_macro("polynomial", 0.3, 45.0) in {
        "exploit_high",
        "exploit_mild",
    }
    # cross-family distance is undefined, not large
    assert mac.macro_distance("gaussian", 1.0, 0.1, "neutral") is None
    assert mac.macro_distance("polynomial", 1.0, 20.0, "neutral") == pytest.approx(0.0)
    # distance grows with the multiplier gap
    near = mac.macro_distance("polynomial", 1.1, 20.0, "neutral")
    far = mac.macro_distance("polynomial", 4.0, 20.0, "neutral")
    assert near is not None and far is not None and far > near
    with pytest.raises(ValueError, match="positive"):
        mac.macro_distance("polynomial", 0.0, 20.0, "neutral")
    with pytest.raises(ValueError, match="unknown macro"):
        mac.macro_distance("polynomial", 1.0, 20.0, "nope")


def test_decompose_group_identities_and_hand_values() -> None:
    """total = between + within, with hand-checked numbers."""
    components = mac.decompose_group(
        [0.0, 0.0, 1.0, 1.0],
        ["a", "a", "b", "b"],
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
    )
    assert components["n_candidates"] == 4
    assert components["n_macros"] == 2
    assert components["total_var"] == pytest.approx(0.25)
    assert components["between_macro_var"] == pytest.approx(0.25)
    assert components["within_macro_var"] == pytest.approx(0.0)
    assert components["between_macro_var"] + components["within_macro_var"] == (
        pytest.approx(components["total_var"])
    )
    assert components["replicate_noise_var"] == pytest.approx(0.0)
    noisy = mac.decompose_group(
        [0.0, 1.0],
        ["a", "b"],
        [[0.0, 0.1], [1.0, 1.1]],
    )
    assert noisy["replicate_noise_var"] == pytest.approx(0.005)
    assert mac.decompose_group([], [], [])["n_candidates"] == 0
    with pytest.raises(ValueError, match="aligned"):
        mac.decompose_group([0.0, 1.0], ["a"], [[0.0], [1.0]])


# --- SNR analysis ------------------------------------------------------------


def test_macro_snr_table_on_macro_aligned_world(tmp_path: Path) -> None:
    """Homogeneous macro regions concentrate the signal: macro SNR wins."""
    path = _write_intervention(tmp_path / "counterfactual_horizon_zdt1.json",
                               _aligned_candidates())
    payload = mac.macro_snr_table([path])
    entry = payload["per_horizon"][str(_HORIZON)]
    assert entry["n_states"] == 1
    assert entry["n_candidates"] == 8
    assert entry["variance_explained_by_macro"] > 0.95
    assert entry["continuous_snr"] is not None and entry["macro_snr"] is not None
    assert entry["macro_snr"] > entry["continuous_snr"]
    # every candidate is exactly a macro representative
    assert set(payload["macro_assignment"].values()) == {
        "explore_high",
        "neutral",
        "exploit_high",
        "operator_switch_gaussian",
    }
    assert sum(payload["macro_counts"].values()) == 8
    assert payload["continuous_snr"] == pytest.approx(entry["continuous_snr"])
    assert payload["macro_snr"] == pytest.approx(entry["macro_snr"])


def test_macro_snr_table_on_misaligned_world(tmp_path: Path) -> None:
    """When macro members disagree, the discretisation destroys the signal."""
    path = _write_intervention(tmp_path / "counterfactual_horizon_zdt1.json",
                               _misaligned_candidates())
    payload = mac.macro_snr_table([path])
    entry = payload["per_horizon"][str(_HORIZON)]
    assert entry["variance_explained_by_macro"] < 0.3
    assert entry["continuous_snr"] > 0.0
    assert entry["macro_snr"] < entry["continuous_snr"] / 2.0
    assert entry["macro_snr_within_as_noise"] < entry["continuous_snr"] / 2.0


def test_macro_snr_table_schema_and_errors(tmp_path: Path) -> None:
    """Artifact schema, horizon filter and explicit failure modes."""
    path = _write_intervention(tmp_path / "counterfactual_horizon_zdt1.json",
                               _aligned_candidates())
    payload = mac.macro_snr_table([path], horizons=[_HORIZON])
    assert set(payload) >= {
        "continuous_snr",
        "macro_snr",
        "macro_assignment",
        "per_horizon",
        "macro_actions",
        "macro_counts",
        "definitions",
    }
    assert payload["files"] == ["counterfactual_horizon_zdt1.json"]
    assert len(payload["macro_assignment"]) == 8
    assert payload["macro_actions"]["neutral"]["multiplier"] == pytest.approx(1.0)
    assert set(payload["definitions"]) >= {
        "continuous_snr",
        "macro_snr",
        "macro_snr_within_as_noise",
        "variance_explained_by_macro",
        "assignment_rule",
    }
    with pytest.raises(FileNotFoundError, match="no intervention files"):
        mac.macro_snr_table([])
    with pytest.raises(FileNotFoundError, match="missing intervention files"):
        mac.macro_snr_table([tmp_path / "absent.json"])
    with pytest.raises(ValueError, match="no samples"):
        mac.macro_snr_table([path], horizons=[99])


def test_cli_writes_the_artifact(tmp_path: Path) -> None:
    """``python -m controller.macro_actions`` entry point."""
    input_dir = tmp_path / "counterfactual"
    _write_intervention(input_dir / "counterfactual_horizon_zdt1.json",
                        _aligned_candidates())
    out_path = tmp_path / "macro_action_snr.json"
    payload = mac.main(
        ["--input-dir", str(input_dir), "--out", str(out_path), "--horizons", "5"]
    )
    assert out_path.is_file()
    with out_path.open("r", encoding="utf-8") as fh:
        written = json.load(fh)
    assert written["n_candidates"] == 8
    assert written["continuous_snr"] == pytest.approx(payload["continuous_snr"])
    args = mac.parse_args([])
    assert args.input_dir == "results/phase2b/counterfactual"
    assert args.out == mac.DEFAULT_OUT
    assert args.horizons is None
