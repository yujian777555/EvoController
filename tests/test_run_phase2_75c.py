from __future__ import annotations

"""Tests for the Phase-2.75C runner (long-horizon collection + transfer).

Everything is synthetic and fast: the ``evaluate-horizon`` invocation is
replaced by a fake that writes a minimal artifact, and the transfer stages run
on a hand-written intervention dataset with four problems. The tests pin

* the manifest/shard/resume mechanics of the collection stage (idempotent
  re-runs, ``--force``, failure recording, cost estimates),
* that the cross-problem split never trains on a test-problem state,
* the ``generalization.json`` schema including per ``(problem, horizon)``
  decision metrics.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller.dataset import merge_state_reward
from controller.state_encoder import StateEncoder
from experiments import run_phase2_75c as run275c

_HORIZONS = (5, 10, 20)


# --- shared synthetic fixtures ----------------------------------------------


def _encoder(directory: Path) -> Path:
    """Write a fitted (tiny) encoder and return its path."""
    trajectories = []
    for j in range(2):
        hv, igd = 0.3 + 0.05 * j, 0.5 - 0.02 * j
        transitions = []
        for t in range(6):
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
                        "mutation_probability": 1.0 / 30.0,
                        "exploration_strength": 20.0,
                    },
                    "reward": {"delta_hv": delta_hv, "delta_igd": delta_igd},
                }
            )
        trajectories.append(transitions)
    encoder = StateEncoder(1).fit(trajectories)
    path = directory / "encoder.json"
    directory.mkdir(parents=True, exist_ok=True)
    encoder.save(path)
    return path


def _write_intervention_dataset(
    directory: Path,
    problems: tuple[str, ...],
    *,
    n_states: int = 8,
    n_candidates: int = 4,
    horizons: tuple[int, ...] = _HORIZONS,
    x_dim: int = 12,
    seed: int = 0,
    with_controller: bool = True,
) -> Path:
    """One npz per problem plus a v1-shaped meta with the encoder path."""
    directory.mkdir(parents=True, exist_ok=True)
    encoder_path = _encoder(directory)
    rng = np.random.Generator(np.random.PCG64(seed))
    for problem in problems:
        rows_x, rows_y, rows_problem = [], [], []
        rows_seed, rows_generation, rows_horizon, rows_candidate = [], [], [], []
        rows_kind = []
        for state in range(n_states):
            state_block = rng.normal(scale=0.1, size=x_dim - 4)
            for candidate in range(n_candidates):
                multiplier = 0.5 + candidate
                row = np.concatenate(
                    [state_block, [multiplier, 10.0 + candidate, 1.0, 0.0]]
                )
                for horizon in horizons:
                    rows_x.append(row)
                    rows_y.append(
                        (candidate - (n_candidates - 1) / 2.0) * (horizon / 10.0)
                    )
                    rows_problem.append(problem)
                    rows_seed.append(1000 + state)
                    rows_generation.append(50 + state)
                    rows_horizon.append(int(horizon))
                    rows_candidate.append(candidate)
                    rows_kind.append(
                        "controller"
                        if with_controller and candidate == 0
                        else "alternative"
                    )
        np.savez_compressed(
            directory / f"intervention_dataset_{problem}.npz",
            X=np.vstack(rows_x),
            y_adv=np.asarray(rows_y, dtype=np.float64),
            problem=np.asarray(rows_problem, dtype="U16"),
            seed=np.asarray(rows_seed, dtype=np.int64),
            generation=np.asarray(rows_generation, dtype=np.int64),
            horizon=np.asarray(rows_horizon, dtype=np.int64),
            candidate_index=np.asarray(rows_candidate, dtype=np.int64),
            candidate_kind=np.asarray(rows_kind, dtype="U16"),
        )
    with (directory / "intervention_meta.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": {"encoder": str(encoder_path)}}, fh)
    return directory


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Four-problem synthetic intervention dataset (zdt4 is the test problem)."""
    return _write_intervention_dataset(
        tmp_path_factory.mktemp("transfer_dataset"),
        ("zdt1", "zdt2", "zdt3", "zdt4"),
        n_states=20,
    )


# --- Task 3: estimates, sharding, manifest ----------------------------------


def test_estimate_collection_is_explicit_about_the_grid() -> None:
    """Cost estimate: (1 + alternatives) * reps * max(horizons) per state."""
    estimate = run275c.estimate_collection(
        ["zdt4", "zdt6"],
        horizons=[20, 50, 100],
        n_alternatives=10,
        n_reps=3,
        states_per_problem={"zdt4": 30, "zdt6": 30},
        seconds_per_generation=0.4,
    )
    assert estimate["branch_generations"] == 100
    assert estimate["per_problem"]["zdt4"]["generations"] == 30 * 11 * 3 * 100
    assert estimate["per_problem"]["zdt4"]["estimate_sec"] == pytest.approx(
        30 * 11 * 3 * 100 * 0.4
    )
    assert estimate["total_generations"] == 2 * 99000
    assert estimate["total_estimate_sec"] == pytest.approx(2 * 99000 * 0.4)
    # a missing problem contributes zero rather than blowing up
    partial = run275c.estimate_collection(
        ["zdt4", "zdt9"],
        horizons=[20],
        n_alternatives=1,
        n_reps=1,
        states_per_problem={"zdt4": 5},
        seconds_per_generation=1.0,
    )
    assert partial["per_problem"]["zdt9"]["n_states"] == 0
    with pytest.raises(ValueError, match="horizons"):
        run275c.estimate_collection(
            ["zdt4"], horizons=[], n_alternatives=1, n_reps=1,
            states_per_problem={"zdt4": 1}, seconds_per_generation=1.0,
        )
    with pytest.raises(ValueError, match="n_alternatives"):
        run275c.estimate_collection(
            ["zdt4"], horizons=[5], n_alternatives=0, n_reps=1,
            states_per_problem={"zdt4": 1}, seconds_per_generation=1.0,
        )


def test_format_duration() -> None:
    """Durations render as H:MM:SS above an hour."""
    assert run275c.format_duration(0) == "0m00s"
    assert run275c.format_duration(59) == "0m59s"
    assert run275c.format_duration(61) == "1m01s"
    assert run275c.format_duration(3661) == "1h01m01s"


def test_shard_problems_partitions_and_validates() -> None:
    """Shards cover the problem list exactly once."""
    shards = [run275c.shard_problems(["a", "b", "c", "d"], index, 3) for index in range(3)]
    assert sorted(sum(shards, [])) == ["a", "b", "c", "d"]
    assert run275c.shard_problems(["a", "b"], 0, 1) == ["a", "b"]
    with pytest.raises(ValueError, match="num_shards"):
        run275c.shard_problems(["a"], 0, 0)
    with pytest.raises(ValueError, match="shard must lie"):
        run275c.shard_problems(["a"], 2, 2)


class _FakeEvaluation:
    """Stand-in for ``evaluate-horizon``: writes a minimal artifact."""

    def __init__(self, *, fail_on: tuple[str, ...] = ()) -> None:
        self.calls: list[str] = []
        self.fail_on = set(fail_on)

    def __call__(self, namespace: Any) -> dict[str, Any]:
        problem = str(namespace.problem)
        self.calls.append(problem)
        if problem in self.fail_on:
            raise RuntimeError("boom")
        out_dir = Path(namespace.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "problem": problem,
            "config": {
                "horizons": [int(h) for h in namespace.horizons],
                "n_alternatives": int(namespace.n_alternatives),
                "n_reps": int(namespace.n_reps),
                "generations": 100,
            },
            "states": [
                {
                    "problem": problem,
                    "seed": 1000 + index,
                    "generation": 50 + index,
                    "candidates": [
                        {"index": candidate}
                        for candidate in range(int(namespace.n_alternatives) + 1)
                    ],
                }
                for index in range(3)
            ],
            "summary": {},
        }
        path = out_dir / f"counterfactual_horizon_{problem}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return payload


def _collect_args(tmp_path: Path, problems: list[str]) -> list[str]:
    return [
        "--stage", "collect",
        "--problems", *problems,
        "--counterfactual-dir", str(tmp_path / "counterfactual"),
        "--snapshots-dir", str(tmp_path / "snapshots"),
        "--collect-horizons", "20", "50", "100",
        "--max-states", "30",
        "--n-alternatives", "10",
        "--n-reps", "3",
    ]


def test_collect_writes_manifest_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second run skips completed problems; ``--force`` redoes them."""
    fake = _FakeEvaluation()
    monkeypatch.setattr(run275c, "run_horizon_evaluation", fake)
    args = run275c.parse_args(_collect_args(tmp_path, ["zdt4", "zdt6"]))
    manifest = run275c.run_collect_stage(args)
    assert fake.calls == ["zdt4", "zdt6"]
    assert manifest["problems"]["zdt4"]["status"] == "completed"
    assert manifest["problems"]["zdt6"]["n_states"] == 3
    assert manifest["problems"]["zdt6"]["n_candidates"] == 11
    assert manifest["config"]["horizons"] == [20, 50, 100]
    assert manifest["summary"]["shard_completed"] == ["zdt4", "zdt6"]
    assert manifest["estimate"]["branch_generations"] == 100
    assert run275c.manifest_path(args.counterfactual_dir).is_file()

    # idempotent: nothing is recomputed
    manifest_again = run275c.run_collect_stage(
        run275c.parse_args(_collect_args(tmp_path, ["zdt4", "zdt6"]))
    )
    assert fake.calls == ["zdt4", "zdt6"]
    assert manifest_again["summary"]["shard_completed"] == ["zdt4", "zdt6"]

    # --force redoes both
    forced = run275c.run_collect_stage(
        run275c.parse_args([*_collect_args(tmp_path, ["zdt4", "zdt6"]), "--force"])
    )
    assert fake.calls == ["zdt4", "zdt6", "zdt4", "zdt6"]
    assert forced["problems"]["zdt4"]["status"] == "completed"

    # a shard only touches its own problems
    sharded_run = run275c.run_collect_stage(
        run275c.parse_args(
            [*_collect_args(tmp_path, ["zdt4", "zdt6"]), "--shard", "1", "--num-shards", "2"]
        )
    )
    assert sharded_run["config"]["problems"] == ["zdt6"]
    assert fake.calls[-1] == "zdt6"


def test_collect_records_failures_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing problem is recorded while the others still run."""
    fake = _FakeEvaluation(fail_on=("zdt4",))
    monkeypatch.setattr(run275c, "run_horizon_evaluation", fake)
    manifest = run275c.run_collect_stage(
        run275c.parse_args(_collect_args(tmp_path, ["zdt4", "zdt6"]))
    )
    assert fake.calls == ["zdt4", "zdt6"]
    assert manifest["problems"]["zdt4"]["status"] == "failed"
    assert "boom" in manifest["problems"]["zdt4"]["error"]
    assert manifest["problems"]["zdt6"]["status"] == "completed"
    assert manifest["summary"]["shard_failed"] == ["zdt4"]
    rerun = run275c.run_collect_stage(
        run275c.parse_args(_collect_args(tmp_path, ["zdt4", "zdt6"]))
    )
    # only the failed problem is retried; the completed one is skipped
    assert fake.calls == ["zdt4", "zdt6", "zdt4"]
    assert rerun["problems"]["zdt4"]["status"] == "failed"
    assert rerun["problems"]["zdt6"]["status"] == "completed"


def test_available_states_uses_the_corpus_when_present(tmp_path: Path) -> None:
    """The estimate uses the real snapshot count, capped by --max-states."""
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    for index in range(7):
        (snapshots / f"zdt4__seed1000__gen{index}.pkl").touch()
    assert run275c.available_states(snapshots, "zdt4", 30) == 7
    assert run275c.available_states(snapshots, "zdt4", 3) == 3
    assert run275c.available_states(snapshots, "zdt6", 30) == 30  # no corpus -> cap


# --- Task 4: transfer --------------------------------------------------------


def test_alternatives_filter_and_leak_free_split(
    tmp_path: Path, dataset_dir: Path
) -> None:
    """The controller row is droppable and no test state is ever trained on."""
    kept, files = run275c.load_problem_arrays(
        dataset_dir, ["zdt1", "zdt4"], drop_controller=True
    )
    assert len(files) == 2
    assert set(np.unique(kept["candidate_kind"]).tolist()) == {"alternative"}
    all_rows, _ = run275c.load_problem_arrays(
        dataset_dir, ["zdt1", "zdt4"], drop_controller=False
    )
    assert set(np.unique(all_rows["candidate_kind"]).tolist()) == {
        "controller",
        "alternative",
    }
    assert all_rows["X"].shape[0] > kept["X"].shape[0]
    # only the requested problems are kept
    assert set(np.unique(kept["problem"]).tolist()) == {"zdt1", "zdt4"}
    with pytest.raises(FileNotFoundError, match="no intervention_dataset"):
        run275c.load_problem_arrays(dataset_dir, ["zdt9"], drop_controller=True)

    args = run275c.parse_args(
        [
            "--stage", "train",
            "--dataset-dir", str(dataset_dir),
            "--model-dir", str(tmp_path / "models"),
            "--train-problems", "zdt1", "zdt2", "zdt3",
            "--test-problems", "zdt4",
            "--horizons", "5", "10", "20",
            "--epochs", "5",
            "--hidden-dims", "8",
            "--val-fraction", "0.25",
        ]
    )
    record = run275c.run_train_stage(args)
    split = record["split"]
    train_keys = set(split["train_state_keys"])
    evaluation_keys = set(split["evaluation_state_keys"])
    assert train_keys and evaluation_keys
    assert train_keys.isdisjoint(evaluation_keys)
    assert all(key.startswith("zdt4|") for key in evaluation_keys)
    # the zero-shot arm never sees a zdt4 state; the in-distribution arm does
    # (but still not the held-out zdt4 states, which is what the split checks)
    assert record["arms"]["cross_problem"]["train_problems"] == ["zdt1", "zdt2", "zdt3"]
    assert "zdt4" in record["arms"]["all_problems"]["train_problems"]
    assert split["evaluation_problem_counts"] == {"zdt4": len(evaluation_keys)}


def test_generalization_schema(tmp_path: Path, dataset_dir: Path) -> None:
    """``--stage all`` writes the documented per (problem, horizon) report."""
    out_dir = tmp_path / "out"
    payload = run275c.main(
        [
            "--stage", "all",
            "--dataset-dir", str(dataset_dir),
            "--model-dir", str(out_dir / "models"),
            "--out", str(out_dir / "generalization.json"),
            "--out-dir", str(out_dir),
            "--train-problems", "zdt1", "zdt2", "zdt3",
            "--test-problems", "zdt4",
            "--horizons", "5", "10", "20",
            "--epochs", "30",
            "--batch-size", "16",
            "--hidden-dims", "16", "16",
            "--val-fraction", "0.25",
        ]
    )
    assert (out_dir / "generalization.json").is_file()
    with (out_dir / "generalization.json").open("r", encoding="utf-8") as fh:
        written = json.load(fh)
    assert set(written) >= {
        "config", "training_problem_sets", "arms", "per_problem", "split", "losses",
    }
    assert written["training_problem_sets"] == {
        "cross_problem": ["zdt1", "zdt2", "zdt3"],
        "all_problems": ["zdt1", "zdt2", "zdt3", "zdt4"],
    }
    assert set(written["arms"]) == {"cross_problem", "all_problems"}
    for name in ("cross_problem", "all_problems"):
        arm = written["arms"][name]
        assert Path(arm["model"]).is_file()
        assert set(arm["pooled"]) >= {
            "spearman_mean", "kendall_mean", "oracle_hit_rate", "regret_mean",
        }
        entry = written["per_problem"]["zdt4"][name]
        assert set(entry) == {"n_states", "per_horizon", "overall"}
        for horizon in ("5", "10", "20"):
            metrics = entry["per_horizon"][horizon]
            assert set(metrics) >= {
                "n_groups",
                "spearman_mean",
                "kendall_mean",
                "oracle_hit_rate",
                "regret_mean",
                "oracle_gap_mean",
            }
    # the synthetic action effect is learnable from the action block
    assert written["per_problem"]["zdt4"]["all_problems"]["overall"][
        "spearman_mean"
    ] > 0.8
    # the saved models must be loadable again
    from controller.advantage_predictor import AdvantagePredictor

    predictor = AdvantagePredictor.load(written["arms"]["all_problems"]["model"])
    assert predictor.input_dim == written["config"]["input_dim"]
    # ... and the encoder is copied next to them
    assert (out_dir / "models" / "encoder.json").is_file()
    assert payload["config"]["evaluation_candidates"] == "alternatives"


def test_stage_generalize_reruns_from_the_dataset(
    tmp_path: Path, dataset_dir: Path
) -> None:
    """``--stage generalize`` trains implicitly and writes the same report."""
    out_dir = tmp_path / "out_again"
    payload = run275c.main(
        [
            "--stage", "generalize",
            "--dataset-dir", str(dataset_dir),
            "--model-dir", str(out_dir / "models"),
            "--out", str(out_dir / "generalization.json"),
            "--train-problems", "zdt1", "zdt2", "zdt3",
            "--test-problems", "zdt4",
            "--horizons", "5", "10",
            "--epochs", "3",
            "--hidden-dims", "8",
        ]
    )
    assert (out_dir / "generalization.json").is_file()
    assert set(payload["per_problem"]["zdt4"]) == {"cross_problem", "all_problems"}
    assert set(payload["config"]["horizons"]) == {5, 10}


def test_resolve_dataset_dir_and_cli_defaults(tmp_path: Path) -> None:
    """Dataset resolution follows the feature set; bad paths are explicit."""
    v1_args = run275c.parse_args(["--feature-set", "v1"])
    assert run275c.resolve_dataset_dir(v1_args) == Path(run275c.DEFAULT_V1_DATASET_DIR)
    v2_args = run275c.parse_args(
        ["--feature-set", "v2", "--out-dir", str(tmp_path), "--advantage-baseline", "final_hv"]
    )
    with pytest.raises(ValueError, match="dataset directory not found"):
        run275c.resolve_dataset_dir(v2_args)
    (tmp_path / "final_hv").mkdir()
    assert run275c.resolve_dataset_dir(v2_args) == tmp_path / "final_hv"

    args = run275c.parse_args([])
    assert args.stage == "all"
    assert tuple(args.problems) == run275c.DEFAULT_COLLECT_PROBLEMS
    assert tuple(args.collect_horizons) == run275c.DEFAULT_COLLECT_HORIZONS
    assert args.max_states == 30 and args.n_alternatives == 10 and args.n_reps == 3
    assert tuple(args.train_problems) == run275c.DEFAULT_TRAIN_PROBLEMS
    assert tuple(args.test_problems) == run275c.DEFAULT_TEST_PROBLEMS
    assert args.feature_set == "v1"
    assert args.counterfactual_dir == run275c.DEFAULT_COUNTERFACTUAL_DIR
    assert args.out == run275c.DEFAULT_GENERALIZATION_OUT
    assert args.evaluation_candidates == "alternatives"
    assert args.shard == 0 and args.num_shards == 1 and args.force is False


def test_collect_namespace_matches_the_producer_cli(tmp_path: Path) -> None:
    """The generated ``evaluate-horizon`` namespace carries the long grid."""
    args = run275c.parse_args(_collect_args(tmp_path, ["zdt4"]))
    namespace = run275c._evaluate_namespace(args, "zdt4")
    assert namespace.command == "evaluate-horizon"
    assert namespace.problem == "zdt4"
    assert list(namespace.horizons) == [20, 50, 100]
    assert namespace.n_alternatives == 10
    assert namespace.n_reps == 3
    assert namespace.max_states == 30
    assert namespace.controller_type == "planning"
    assert str(namespace.out_dir) == str(tmp_path / "counterfactual")
