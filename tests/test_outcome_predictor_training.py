from __future__ import annotations

"""End-to-end test of the Phase-2 Experiment-A training/evaluation harness.

Runs ``experiments/train_outcome_predictor.py`` and
``experiments/evaluate_outcome_predictor.py`` on three tiny synthetic
trajectory files (30 generations each) and checks the artifact set and the
``evaluation.json`` schema (per-horizon model/persistence/linear metrics,
Wilcoxon p-values, success-criteria booleans).

The test is skipped while the parallel contract modules
(``controller.outcome_predictor.OutcomePredictor`` and
``controller.dataset.build_outcome_samples``) are not yet available.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip(
    "controller.outcome_predictor",
    reason="controller.outcome_predictor (parallel contract) not available yet",
)
_dataset = pytest.importorskip("controller.dataset")
if not hasattr(_dataset, "build_outcome_samples"):
    pytest.skip(
        "controller.dataset.build_outcome_samples (parallel contract) "
        "not available yet",
        allow_module_level=True,
    )

from experiments.evaluate_outcome_predictor import main as evaluate_main
from experiments.train_outcome_predictor import main as train_main

#: Horizons used by the tiny end-to-end run (the contract defaults).
HORIZONS = [1, 5, 10, 20]
#: Generations per synthetic trajectory (31 transitions, indices 0..30).
N_GENS = 30


def _make_trajectory(seed: int, n_gens: int = N_GENS) -> dict[str, Any]:
    """Fabricate one synthetic trajectory payload in the recorder schema.

    The hypervolume follows a smooth saturating curve (plus small seeded
    noise) so the forecasting task is learnable; actions carry the full
    Phase-1.75 action keys (``mutation_operator``, ``mutation_probability``,
    ``exploration_strength``, ``mutation_multiplier``).
    """
    rng = np.random.default_rng(seed)
    transitions: list[dict[str, Any]] = []
    hv = 0.0
    igd = 1.0
    for generation in range(n_gens + 1):
        prev_hv, prev_igd = hv, igd
        target = 0.85 * (1.0 - np.exp(-generation / 12.0))
        hv = float(target + rng.normal(0.0, 0.002)) if generation else 0.0
        igd = float(max(1.0 - 0.9 * (1.0 - np.exp(-generation / 10.0)), 1e-6))
        operator = "polynomial" if generation % 2 == 0 else "gaussian"
        exploration = (
            float(rng.uniform(2.0, 50.0))
            if operator == "polynomial"
            else float(rng.uniform(0.02, 0.3))
        )
        multiplier = float(rng.uniform(0.25, 8.0))
        transitions.append(
            {
                "generation": generation,
                "state": {
                    "generation": generation,
                    "hv": hv,
                    "igd": igd,
                    "diversity": float(rng.uniform(0.2, 0.8)),
                },
                "action": {
                    "mutation_operator": operator,
                    "mutation_probability": multiplier / 30.0,
                    "exploration_strength": exploration,
                    "mutation_multiplier": multiplier,
                },
                "reward": {
                    "delta_hv": hv - prev_hv,
                    "delta_igd": igd - prev_igd,
                },
            }
        )
    return {
        "config": {
            "problem": "zdt1",
            "n_vars": 30,
            "algorithm": "nsga2",
            "pop_size": 20,
            "generations": n_gens,
            "policy": "random",
            "action_space": "full",
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


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the tiny train+evaluate pipeline once for all assertions."""
    tmp = tmp_path_factory.mktemp("phase2_outcome")
    train_dir = tmp / "train"
    train_dir.mkdir()
    for seed in range(3):
        payload = _make_trajectory(seed)
        with (train_dir / f"zdt1_nsga2_seed{seed}.json").open(
            "w", encoding="utf-8"
        ) as fh:
            json.dump(payload, fh)
    out_dir = tmp / "out"
    meta = train_main(
        [
            "--train-dirs",
            str(train_dir),
            "--out-dir",
            str(out_dir),
            "--window",
            "4",
            "--horizons",
            *[str(h) for h in HORIZONS],
            "--epochs",
            "10",
            "--batch-size",
            "32",
        ]
    )
    evaluation = evaluate_main(["--model-dir", str(out_dir)])
    return {
        "meta": meta,
        "evaluation": evaluation,
        "out_dir": out_dir,
        "train_dir": train_dir,
    }


def test_artifacts_written(trained_model: dict[str, Any]) -> None:
    """Training and evaluation persist the full artifact set."""
    out_dir = trained_model["out_dir"]
    for name in (
        "predictor.pt",
        "encoder.json",
        "training_meta.json",
        "evaluation.json",
    ):
        assert (out_dir / name).is_file(), name


def test_training_meta_schema(trained_model: dict[str, Any]) -> None:
    """``training_meta.json`` carries config, counts, losses, and timing."""
    meta = trained_model["meta"]
    meta_path = trained_model["out_dir"] / "training_meta.json"
    on_disk = json.loads(meta_path.read_text(encoding="utf-8"))
    assert on_disk["n_samples"] == meta["n_samples"]
    for key in (
        "config",
        "n_samples",
        "n_train",
        "n_val",
        "train_loss",
        "val_loss",
        "horizons",
        "window",
        "wall_time_sec",
        "timestamp_utc",
    ):
        assert key in meta, key
    assert meta["horizons"] == HORIZONS
    assert meta["window"] == 4
    assert meta["n_samples"] > 0
    assert meta["n_train"] + meta["n_val"] == meta["n_samples"]
    assert len(meta["train_loss"]) == 10
    assert meta["config"]["train_dirs"] == [str(trained_model["train_dir"])]
    assert meta["wall_time_sec"] > 0.0


def test_evaluation_schema(trained_model: dict[str, Any]) -> None:
    """``evaluation.json`` has per-horizon metrics for model and baselines."""
    evaluation = trained_model["evaluation"]
    assert set(evaluation["per_horizon"]) == {str(h) for h in HORIZONS}
    for h in HORIZONS:
        entry = evaluation["per_horizon"][str(h)]
        for method in ("model", "persistence", "linear"):
            assert set(entry[method]) == {"mse", "mae", "r2"}, (h, method)
            assert entry[method]["mse"] >= 0.0
            assert entry[method]["mae"] >= 0.0
        assert "wilcoxon_p_vs_persistence" in entry
        assert "wilcoxon_p_vs_linear" in entry
        for key in ("wilcoxon_p_vs_persistence", "wilcoxon_p_vs_linear"):
            p_value = entry[key]
            assert p_value is None or 0.0 <= p_value <= 1.0
    assert set(evaluation["success_criteria"]) == {
        "model_beats_persistence_all_horizons",
        "r2_above_0.5_h1_h5",
        "long_horizon_h20_beats_baselines",
    }
    for value in evaluation["success_criteria"].values():
        assert isinstance(value, bool)
    assert evaluation["n_samples"] > 0
    assert evaluation["config"]["horizons"] == HORIZONS
    assert evaluation["config"]["window"] == 4


def test_persistence_baseline_is_nontrivial(
    trained_model: dict[str, Any],
) -> None:
    """Persistence has positive MSE (HV rises along the synthetic curve)."""
    per_horizon = trained_model["evaluation"]["per_horizon"]
    assert per_horizon["20"]["persistence"]["mse"] > 0.0
    # The persistence baseline predicts current HV, which underestimates a
    # rising curve: its long-horizon error exceeds its short-horizon error.
    assert (
        per_horizon["20"]["persistence"]["mse"]
        > per_horizon["1"]["persistence"]["mse"]
    )
