from __future__ import annotations

"""Oracle learnability benchmark (Phase 2.8 Task 1).

Decides whether the action effect is *learnable at all* under the frozen
compact representation, independently of the neural training recipe that
Phase 2.75D showed to be action-blind.

Protocol
--------
Same data, same state-level split as the final target comparison
(``results/phase2_75d/dataset_final``, 1200 states, 960 train / 240 val), but
the learners are strong non-neural baselines: ridge regression, random forest
and histogram gradient boosting. If these also fail to rank actions within a
state, the limitation is the *representation / action design*; if they
succeed while the MLP does not, the limitation is the *optimisation
objective*.

Reported per model and horizon: within-state Spearman/Kendall, top-k hit rate,
regret, oracle hit, and the **action-sensitivity ratio**

    within-state std(prediction) / within-state std(realised)

which is the diagnostic that exposed the Phase-2.75D failure (ratio 0.001 for
the MLP).

Usage::

    python experiments/oracle_learnability.py \
        --dataset-dir results/phase2_75d/dataset_final \
        --split-json results/phase2_75d/target_comparison_final.json \
        --out results/phase2_75d/oracle_learnability.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.advantage_predictor import ranking_metrics

DEFAULT_DATASET_DIR = "results/phase2_75d/dataset_final"
DEFAULT_SPLIT_JSON = "results/phase2_75d/target_comparison_final.json"
DEFAULT_OUT = "results/phase2_75d/oracle_learnability.json"


def _load_target(dataset_dir: Path, target: str) -> dict[str, np.ndarray | list[str]]:
    """Concatenate one baseline's npz files into flat arrays.

    Returns:
        Dict with ``X`` ``(n, d)``, ``y`` ``(n,)``, ``horizon`` ``(n,)``,
        ``kind`` ``(n,)`` and ``state`` (list of ``problem|seed|generation``).
    """
    xs, ys, hs, ks, states = [], [], [], [], []
    for path in sorted((dataset_dir / target).glob("intervention_dataset_*.npz")):
        data = np.load(path, allow_pickle=True)
        xs.append(data["X"])
        ys.append(data["y_adv"])
        hs.append(data["horizon"])
        ks.append(data["candidate_kind"])
        states.extend(
            f"{data['problem'][i]}|{int(data['seed'][i])}|{int(data['generation'][i])}"
            for i in range(len(data["X"]))
        )
    return {
        "X": np.concatenate(xs),
        "y": np.concatenate(ys),
        "horizon": np.concatenate(hs),
        "kind": np.concatenate(ks),
        "state": states,
    }


def _evaluate(
    pred: np.ndarray,
    realized: np.ndarray,
    states: list[str],
    horizons: np.ndarray,
    kinds: np.ndarray,
    horizon_values: Sequence[int],
) -> dict[str, Any]:
    """Per-horizon ranking metrics plus the action-sensitivity ratio."""
    per_horizon: dict[str, Any] = {}
    for hi, horizon in enumerate(horizon_values):
        rows = np.flatnonzero(horizons == horizon)
        groups: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            if kinds[row] == "controller":
                continue
            groups[states[row]].append(int(row))
        usable = [g for g in groups.values() if len(g) >= 2]
        if not usable:
            per_horizon[str(horizon)] = {"n_groups": 0}
            continue
        pred_matrix = np.asarray([pred[g, hi] for g in usable])
        real_matrix = np.asarray([realized[g] for g in usable])
        metrics = ranking_metrics(pred_matrix, real_matrix)
        pred_within = float(np.mean([row.std() for row in pred_matrix]))
        real_within = float(np.mean([row.std() for row in real_matrix]))
        metrics["action_sensitivity_ratio"] = (
            float(pred_within / real_within) if real_within > 0 else None
        )
        metrics["mean_within_state_pred_std"] = pred_within
        metrics["mean_within_state_real_std"] = real_within
        per_horizon[str(horizon)] = metrics
    return per_horizon


def _fit_models(seed: int) -> dict[str, Any]:
    """The non-neural learner zoo (imported lazily so the CLI stays optional)."""
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LinearRegression

    return {
        # ``Ridge`` is unusable here: sklearn 1.0.2 calls scipy with the
        # removed ``sym_pos`` argument. Plain OLS is a fine linear oracle.
        "linear_ols": lambda: LinearRegression(n_jobs=-1),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=seed
        ),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the oracle benchmark and write the JSON artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-json", default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--targets", nargs="+", default=["state_mean"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir)
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))["split"]
    val_keys = set(split["val_state_keys"])
    horizon_values = [int(h) for h in split.get("horizons", [5, 10, 20])]

    report: dict[str, Any] = {
        "config": vars(args),
        "split": {
            "n_states_train": split.get("n_states_train"),
            "n_states_val": split.get("n_states_val"),
        },
        "models": {},
    }
    for target in args.targets:
        data = _load_target(dataset_dir, target)
        states: list[str] = data["state"]  # type: ignore[assignment]
        is_val = np.asarray([s in val_keys for s in states])
        X, y = data["X"], data["y"]  # type: ignore[assignment]
        # Three target columns share one X, so the model is trained per
        # horizon row-subset: rows carry their own horizon label.
        horizon = data["horizon"]  # type: ignore[assignment]
        kinds = data["kind"]  # type: ignore[assignment]
        entry: dict[str, Any] = {"n_samples": int(X.shape[0]), "per_model": {}}
        for name, factory in _fit_models(args.seed).items():
            started = time.perf_counter()
            pred = np.zeros((X.shape[0], len(horizon_values)))
            for hi, horizon_value in enumerate(horizon_values):
                train_rows = np.flatnonzero((horizon == horizon_value) & ~is_val)
                val_rows = np.flatnonzero((horizon == horizon_value) & is_val)
                if train_rows.size == 0 or val_rows.size == 0:
                    continue
                model = factory()
                model.fit(X[train_rows], y[train_rows])
                pred[val_rows, hi] = model.predict(X[val_rows])
            val_mask = is_val
            metrics = _evaluate(
                pred[val_mask],
                y[val_mask],
                [s for s, keep in zip(states, is_val) if keep],
                horizon[val_mask],
                kinds[val_mask],
                horizon_values,
            )
            entry["per_model"][name] = {
                "per_horizon": metrics,
                "wall_time_sec": time.perf_counter() - started,
            }
            overall = metrics.get(str(horizon_values[-1]), {})
            print(
                f"[{target}/{name}] h={horizon_values[-1]} "
                f"rho={overall.get('spearman_mean')} "
                f"sens={overall.get('action_sensitivity_ratio')}"
            )
        report["models"][target] = entry

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {out_path}")
    return report


if __name__ == "__main__":
    main()
