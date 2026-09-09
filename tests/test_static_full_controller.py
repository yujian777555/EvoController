"""Tests for the Phase-1.75 matched full-action static baseline.

Covers :class:`controller.static_full_controller.StaticFullController`
(action math, validation, save/load) and ``experiments.tune_static_full``
(seeded random search over the full action space on training seeds only):
determinism of the shard artifact, shard/aggregate merge equivalence with a
monolithic run, and the hard guarantee that tuning seeds may never overlap
the held-out test seeds.

Tuner tests use a tiny budget (4 candidates, zdt1 only, seed 300, 5
generations, pop 20) so the module runs in a few seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from controller.static_full_controller import StaticFullController
from experiments import tune_static_full

# --- StaticFullController ---------------------------------------------------


def test_predict_action_scales_pm_with_n_vars() -> None:
    """pm equals multiplier / n_vars; operator and exploration pass through."""
    controller = StaticFullController("polynomial", 2.5, 20.0)
    action = controller.predict_action(n_vars=30)
    assert action == {
        "mutation_operator": "polynomial",
        "mutation_probability": pytest.approx(2.5 / 30.0),
        "exploration_strength": pytest.approx(20.0),
    }
    action_10 = controller.predict_action(n_vars=10)
    assert action_10["mutation_probability"] == pytest.approx(2.5 / 10.0)


def test_predict_action_gaussian() -> None:
    """Gaussian operator and sigma exploration strength pass through."""
    controller = StaticFullController("gaussian", 1.0, 0.1)
    action = controller.predict_action(n_vars=30)
    assert action["mutation_operator"] == "gaussian"
    assert action["mutation_probability"] == pytest.approx(1.0 / 30.0)
    assert action["exploration_strength"] == pytest.approx(0.1)


def test_predict_action_ignores_history() -> None:
    """A static controller must return the same action for any history."""
    controller = StaticFullController("polynomial", 0.5, 5.0)
    history = [{"generation": 3, "hv": 0.5, "igd": 0.2, "delta_hv": 0.01}]
    assert controller.predict_action(None, n_vars=30) == controller.predict_action(
        history, n_vars=30
    )


def test_name_attribute() -> None:
    """The controller exposes a non-empty string name for experiment records."""
    controller = StaticFullController("polynomial", 1.0, 20.0)
    assert isinstance(controller.name, str)
    assert controller.name


def test_constructor_validation() -> None:
    """Invalid operator/multiplier/exploration are rejected with ValueError."""
    with pytest.raises(ValueError):
        StaticFullController("uniform", 1.0, 20.0)
    with pytest.raises(ValueError):
        StaticFullController("polynomial", 0.0, 20.0)
    with pytest.raises(ValueError):
        StaticFullController("polynomial", -1.0, 20.0)
    with pytest.raises(ValueError):
        StaticFullController("polynomial", 1.0, 0.0)
    with pytest.raises(ValueError):
        StaticFullController("gaussian", 1.0, -0.1)


def test_predict_action_validates_n_vars() -> None:
    """n_vars < 1 is rejected with ValueError."""
    controller = StaticFullController("polynomial", 1.0, 20.0)
    with pytest.raises(ValueError):
        controller.predict_action(n_vars=0)
    with pytest.raises(ValueError):
        controller.predict_action(n_vars=-3)


def test_save_load_round_trip(tmp_path: Path) -> None:
    """JSON save/load preserves parameters, name, and emitted actions."""
    controller = StaticFullController("gaussian", 3.25, 0.25)
    path = tmp_path / "controller.json"
    controller.save(path)
    assert path.exists()
    loaded = StaticFullController.load(path)
    assert loaded.name == controller.name
    assert loaded.predict_action(n_vars=30) == controller.predict_action(n_vars=30)


def test_load_rejects_invalid_payload(tmp_path: Path) -> None:
    """A tampered JSON payload fails validation on load."""
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"operator": "polynomial", "multiplier": -1.0, "exploration_strength": 20.0}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        StaticFullController.load(path)


# --- tune_static_full tuner --------------------------------------------------

_TINY = [
    "--problems", "zdt1",
    "--seeds", "300",
    "--candidates", "4",
    "--generations", "5",
    "--pop-size", "20",
    "--seed", "7",
]


def _run_search(out_dir: Path, *extra: str) -> None:
    """Run one tuning search (shard mode) with the tiny budget."""
    tune_static_full.main([*_TINY, "--out-dir", str(out_dir), *extra])


def _run_aggregate(out_dir: Path, num_shards: int) -> dict:
    """Aggregate the shard files under ``out_dir`` and return the payload."""
    payload = tune_static_full.main(
        ["--aggregate", "--num-shards", str(num_shards), "--out-dir", str(out_dir)]
    )
    assert payload is not None
    return payload


def test_search_deterministic_identical_artifact(tmp_path: Path) -> None:
    """Two identical searches produce byte-identical shard artifacts."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    _run_search(dir_a)
    _run_search(dir_b)
    shard_a = (dir_a / "tuning_shard_0.json").read_bytes()
    shard_b = (dir_b / "tuning_shard_0.json").read_bytes()
    assert shard_a == shard_b


def test_search_artifact_schema(tmp_path: Path) -> None:
    """The shard artifact records candidates, per-problem metrics, and config."""
    _run_search(tmp_path)
    with (tmp_path / "tuning_shard_0.json").open(encoding="utf-8") as fh:
        shard = json.load(fh)
    config = shard["config"]
    assert config["seeds"] == [300]
    assert config["held_out_seeds"] == list(range(1000, 1020))
    assert config["candidates"] == 4
    assert config["problems"] == ["zdt1"]
    candidates = shard["candidates"]
    assert [c["index"] for c in candidates] == [0, 1, 2, 3]
    for candidate in candidates:
        assert candidate["operator"] in ("polynomial", "gaussian")
        assert 0.25 <= candidate["multiplier"] <= 8.0
        if candidate["operator"] == "polynomial":
            assert 2.0 <= candidate["exploration_strength"] <= 50.0
        else:
            assert 0.02 <= candidate["exploration_strength"] <= 0.3
        per_problem = candidate["per_problem"]["zdt1"]
        assert per_problem["mean_auc_hv"] >= 0.0
        assert per_problem["final_hv"] >= 0.0
        assert set(per_problem["per_seed"]) == {"300"}


def test_shard_merge_matches_monolithic(tmp_path: Path) -> None:
    """Two half-shards aggregate to the same winners as one full search."""
    mono = tmp_path / "mono"
    _run_search(mono)
    mono_payload = _run_aggregate(mono, num_shards=1)

    split = tmp_path / "split"
    _run_search(split, "--num-shards", "2", "--shard-index", "0")
    _run_search(split, "--num-shards", "2", "--shard-index", "1")
    split_payload = _run_aggregate(split, num_shards=2)

    assert split_payload["global"] == mono_payload["global"]
    assert split_payload["per_problem"] == mono_payload["per_problem"]
    assert split_payload["config"]["seeds"] == mono_payload["config"]["seeds"]


def test_aggregate_schema(tmp_path: Path) -> None:
    """The merged artifact follows the static_full_tuning.json schema."""
    _run_search(tmp_path)
    payload = _run_aggregate(tmp_path, num_shards=1)
    assert (tmp_path / "static_full_tuning.json").exists()
    for key in ("operator", "multiplier", "exploration_strength", "mean_auc_hv", "final_hv"):
        assert key in payload["global"]
        assert key in payload["per_problem"]["zdt1"]
    config = payload["config"]
    for key in (
        "seeds", "candidates", "problems", "generations",
        "pop_size", "search_seed", "held_out_seeds",
    ):
        assert key in config


def test_held_out_overlap_raises(tmp_path: Path) -> None:
    """Tuning seeds overlapping the held-out set are rejected with ValueError."""
    with pytest.raises(ValueError):
        tune_static_full.main(
            [
                *_TINY,
                "--seeds", "300", "1000",
                "--out-dir", str(tmp_path),
            ]
        )


def test_held_out_overlap_raises_custom_range(tmp_path: Path) -> None:
    """A custom --held-out-seeds range is enforced as well."""
    with pytest.raises(ValueError):
        tune_static_full.main(
            [
                *_TINY,
                "--held-out-seeds", "300",
                "--out-dir", str(tmp_path),
            ]
        )


def test_aggregate_missing_shard_raises(tmp_path: Path) -> None:
    """Aggregation refuses to merge when an expected shard file is missing."""
    _run_search(tmp_path, "--num-shards", "2", "--shard-index", "0")
    with pytest.raises(FileNotFoundError):
        _run_aggregate(tmp_path, num_shards=2)
