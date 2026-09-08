from __future__ import annotations

import json
import math

import numpy as np
import pytest

from trajectory.recorder import TRAJECTORY_SCHEMA_VERSION, EvolutionRecorder

REF_POINT = np.array([1.1, 1.1])


def _zdt1_reference(n_points: int = 21) -> np.ndarray:
    """Samples of the ZDT1 Pareto front f2 = 1 - sqrt(f1) for IGD reference."""
    f1 = np.linspace(0.0, 1.0, n_points)
    return np.column_stack([f1, 1.0 - np.sqrt(f1)])


def _improving_fronts() -> list[np.ndarray]:
    """Three synthetic fronts that monotonically approach the ZDT1 front."""
    return [
        np.array([[0.0, 1.5], [0.4, 1.0], [1.0, 0.5]]),
        np.array([[0.0, 1.3], [0.4, 0.7], [1.0, 0.3]]),
        np.array([[0.0, 1.0], [0.25, 0.5], [0.5, 0.29], [1.0, 0.0]]),
    ]


def _action() -> dict:
    return {"mutation_operator": "polynomial", "mutation_probability": 1.0 / 30.0}


@pytest.fixture()
def recorder() -> EvolutionRecorder:
    return EvolutionRecorder(
        problem_name="zdt1",
        reference_front=_zdt1_reference(),
        ref_point=REF_POINT,
    )


def test_record_three_generations_and_rewards(recorder: EvolutionRecorder) -> None:
    for gen, front in enumerate(_improving_fronts()):
        recorder.record(gen, front, _action())

    transitions = recorder.transitions()
    assert len(transitions) == 3

    assert transitions[0]["reward"] == {"delta_hv": 0.0, "delta_igd": 0.0}
    for t in (1, 2):
        assert transitions[t]["reward"]["delta_hv"] > 0.0
        assert transitions[t]["reward"]["delta_igd"] > 0.0

    # Rewards must equal the metric deltas between consecutive states.
    for t in (1, 2):
        prev, cur = transitions[t - 1]["state"], transitions[t]["state"]
        assert math.isclose(
            transitions[t]["reward"]["delta_hv"], cur["hv"] - prev["hv"]
        )
        assert math.isclose(
            transitions[t]["reward"]["delta_igd"], prev["igd"] - cur["igd"]
        )


def test_state_keys_exact(recorder: EvolutionRecorder) -> None:
    recorder.record(0, _improving_fronts()[0], _action())
    state = recorder.transitions()[0]["state"]
    assert set(state.keys()) == {"generation", "hv", "igd", "diversity"}
    assert state["generation"] == 0
    assert isinstance(state["hv"], float)
    assert isinstance(state["igd"], float)
    assert isinstance(state["diversity"], float)


def test_action_stored_with_resolved_values(recorder: EvolutionRecorder) -> None:
    recorder.record(0, _improving_fronts()[0], _action())
    action = recorder.transitions()[0]["action"]
    assert action == {
        "mutation_operator": "polynomial",
        "mutation_probability": pytest.approx(1.0 / 30.0),
    }


def test_missing_action_key_raises(recorder: EvolutionRecorder) -> None:
    with pytest.raises(ValueError):
        recorder.record(0, _improving_fronts()[0], {"mutation_operator": "polynomial"})
    with pytest.raises(ValueError):
        recorder.record(0, _improving_fronts()[0], {"mutation_probability": 0.1})


def test_transitions_returns_deep_copy(recorder: EvolutionRecorder) -> None:
    for gen, front in enumerate(_improving_fronts()):
        recorder.record(gen, front, _action())

    snapshot = recorder.transitions()
    snapshot[0]["state"]["hv"] = -999.0
    snapshot[0]["action"]["mutation_operator"] = "corrupted"
    snapshot.clear()

    fresh = recorder.transitions()
    assert len(fresh) == 3
    assert fresh[0]["state"]["hv"] > 0.0
    assert fresh[0]["action"]["mutation_operator"] == "polynomial"


def test_save_schema_and_roundtrip(
    recorder: EvolutionRecorder, tmp_path
) -> None:
    fronts = _improving_fronts()
    for gen, front in enumerate(fronts):
        recorder.record(gen, front, _action())

    config = {
        "problem": "zdt1",
        "algorithm": "nsga2",
        "pop_size": 100,
        "n_generations": 3,
    }
    out = tmp_path / "nested" / "dir" / "trajectory.json"
    recorder.save(out, config=config, seed=42, runtime_sec=1.5)

    assert out.exists()
    with out.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    assert {
        "config",
        "seed",
        "runtime_sec",
        "schema_version",
        "transitions",
        "final",
    } <= set(payload.keys())
    assert payload["config"] == config
    assert payload["seed"] == 42
    assert payload["runtime_sec"] == pytest.approx(1.5)
    assert payload["schema_version"] == TRAJECTORY_SCHEMA_VERSION
    assert len(payload["transitions"]) == 3

    last_state = payload["transitions"][-1]["state"]
    for key in ("hv", "igd", "diversity"):
        assert payload["final"][key] == pytest.approx(last_state[key])

    # Every number must round-trip as a native Python int/float.
    def _check_native(value: object) -> None:
        if isinstance(value, dict):
            for v in value.values():
                _check_native(v)
        elif isinstance(value, list):
            for v in value:
                _check_native(v)
        else:
            assert not isinstance(value, (np.integer, np.floating))

    _check_native(payload)


def test_save_without_transitions_raises(
    recorder: EvolutionRecorder, tmp_path
) -> None:
    with pytest.raises(ValueError):
        recorder.save(tmp_path / "empty.json", config={}, seed=0, runtime_sec=0.0)
