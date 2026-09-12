from __future__ import annotations

"""Action identifiability audit (Phase 2.8 Task 2, Tasks 1-3).

Answers whether the action-credit failure is a *learning-method* problem or an
*identifiability* problem: does the measured outcome actually carry a
detectable, action-attributable signal, and can it be recovered from the
action channel alone (or from pairwise preferences) without a neural model?

Task 1 - action effect variance
    Per horizon, decomposes the total outcome variance into between-action and
    replicate (rollout) noise, reports the resulting SNR and its per-state
    distribution.

Task 2 - action-only learnability
    Trains non-neural regressors on the **action features only** (the last four
    columns of ``X``) and asks whether they rank actions within a state. If the
    action channel alone carries information, a model that never sees the state
    should still beat chance.

Task 3 - pairwise action ranking
    Turns the task into preference learning: for pairs ``(a_i, a_j)`` of the
    same state, predict which outcome is larger from the feature difference.
    Reports accuracy / AUC on held-out states plus the Kendall tau of the
    within-state ordering induced by pairwise win counts.

Usage::

    python experiments/action_identifiability_audit.py \
        --dataset-dir results/phase2_75d/dataset_final \
        --split-json results/phase2_75d/target_comparison_final.json \
        --out results/phase2_75d/action_identifiability.json
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
DEFAULT_OUT = "results/phase2_8/action_identifiability.json"
ACTION_FEATURES = 4


def _load(dataset_dir: Path, target: str) -> dict[str, Any]:
    """Concatenate one baseline's npz files into flat arrays plus state keys."""
    xs, ys, hs, ks, rs, states, problems = [], [], [], [], [], [], []
    for path in sorted((dataset_dir / target).glob("intervention_dataset_*.npz")):
        data = np.load(path, allow_pickle=True)
        xs.append(data["X"])
        ys.append(data["y_adv"])
        hs.append(data["horizon"])
        ks.append(data["candidate_kind"])
        rs.append(data["rewards"])
        states.extend(
            f"{data['problem'][i]}|{int(data['seed'][i])}|{int(data['generation'][i])}"
            for i in range(len(data["X"]))
        )
        problems.extend(str(p) for p in data["problem"])
    return {
        "X": np.concatenate(xs),
        "y": np.concatenate(ys),
        "horizon": np.concatenate(hs),
        "kind": np.concatenate(ks),
        "rewards": np.concatenate(rs),
        "state": states,
        "problem": problems,
    }


def _groups(
    states: Sequence[str], horizons: np.ndarray, kinds: np.ndarray, horizon: int
) -> dict[str, list[int]]:
    """Row indices per state at one horizon, excluding the controller candidate."""
    out: dict[str, list[int]] = defaultdict(list)
    for row in np.flatnonzero(horizons == horizon):
        if kinds[row] == "controller":
            continue
        out[states[row]].append(int(row))
    return {k: v for k, v in out.items() if len(v) >= 2}


def task1_variance(data: dict[str, Any], horizons: Sequence[int]) -> dict[str, Any]:
    """Between-action variance, rollout noise and SNR per horizon."""
    per_horizon: dict[str, Any] = {}
    for horizon in horizons:
        groups = _groups(data["state"], data["horizon"], data["kind"], int(horizon))
        between, noise, snrs = [], [], []
        for rows in groups.values():
            means = []
            variances = []
            for row in rows:
                reps = np.asarray(data["rewards"][row], dtype=float)
                reps = reps[np.isfinite(reps)]
                if reps.size == 0:
                    continue
                means.append(float(reps.mean()))
                variances.append(float(reps.var(ddof=1)) if reps.size > 1 else 0.0)
            if len(means) < 2:
                continue
            b = float(np.var(means, ddof=1))
            w = float(np.mean(variances))
            between.append(b)
            noise.append(w)
            snrs.append(b / w if w > 0 else None)
        defined = [s for s in snrs if s is not None]
        per_horizon[str(horizon)] = {
            "n_states": len(between),
            "action_variance": float(np.mean(between)) if between else None,
            "rollout_noise_variance": float(np.mean(noise)) if noise else None,
            "snr_pooled": (
                float(np.mean(between) / np.mean(noise))
                if noise and np.mean(noise) > 0
                else None
            ),
            "snr_per_state_median": float(np.median(defined)) if defined else None,
            "snr_per_state_mean": float(np.mean(defined)) if defined else None,
            "snr_per_state_q25": float(np.percentile(defined, 25)) if defined else None,
            "snr_per_state_q75": float(np.percentile(defined, 75)) if defined else None,
            "fraction_states_snr_above_1": (
                float(np.mean([s > 1.0 for s in defined])) if defined else None
            ),
            "fraction_states_snr_above_3": (
                float(np.mean([s > 3.0 for s in defined])) if defined else None
            ),
        }
    return per_horizon


def _ranking_from_scores(
    scores: np.ndarray, data: dict[str, Any], horizon: int, states: Sequence[str]
) -> dict[str, Any]:
    """Within-state ranking metrics of arbitrary per-row scores."""
    groups = _groups(states, data["horizon"], data["kind"], int(horizon))
    usable = [rows for rows in groups.values() if len(rows) >= 2]
    if not usable:
        return {"n_groups": 0}
    pred = np.asarray([scores[rows] for rows in usable])
    real = np.asarray([data["y"][rows] for rows in usable])
    metrics = ranking_metrics(pred, real)
    real_std = float(np.mean([row.std() for row in real]))
    metrics["action_sensitivity_ratio"] = (
        float(np.mean([row.std() for row in pred]) / real_std) if real_std > 0 else None
    )
    return metrics


def task2_action_only(
    data: dict[str, Any],
    is_val: np.ndarray,
    horizons: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    """Non-neural regressors on the action features only."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    X_action = data["X"][:, -ACTION_FEATURES:]
    X_state = data["X"][:, :-ACTION_FEATURES]
    models = {
        "linear_ols": lambda: LinearRegression(n_jobs=-1),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=seed
        ),
    }
    report: dict[str, Any] = {}
    for subset_name, features in (
        ("action_only", X_action),
        ("state_only", X_state),
        ("concat", data["X"]),
    ):
        entry: dict[str, Any] = {}
        for name, factory in models.items():
            scores = np.zeros(data["X"].shape[0])
            for horizon in horizons:
                train = np.flatnonzero((data["horizon"] == horizon) & ~is_val)
                test = np.flatnonzero((data["horizon"] == horizon) & is_val)
                if train.size == 0 or test.size == 0:
                    continue
                model = factory()
                model.fit(features[train], data["y"][train])
                scores[test] = model.predict(features[test])
            val_rows = np.flatnonzero(is_val)
            val_states = [data["state"][i] for i in val_rows]
            sub = {
                "X": data["X"][val_rows],
                "y": data["y"][val_rows],
                "horizon": data["horizon"][val_rows],
                "kind": data["kind"][val_rows],
                "rewards": data["rewards"][val_rows],
            }
            horizon_metrics = {
                str(h): _ranking_from_scores(
                    scores[val_rows], sub, int(h), val_states
                )
                for h in horizons
            }
            entry[name] = {"per_horizon": horizon_metrics}
        report[subset_name] = entry
    return report


def task3_pairwise(
    data: dict[str, Any],
    is_val: np.ndarray,
    horizons: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    """Preference learning on same-state action pairs."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    report: dict[str, Any] = {}
    for horizon in horizons:
        groups = _groups(data["state"], data["horizon"], data["kind"], int(horizon))
        train_diffs, train_labels, val_diffs, val_labels = [], [], [], []
        val_group_rows: dict[str, list[int]] = {}
        for key, rows in groups.items():
            pairs = []
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    pair = (rows[i], rows[j])
                    pairs.append(pair)
            if not pairs:
                continue
            if key in {s for s, keep in zip(data["state"], is_val) if keep}:
                val_group_rows[key] = rows
                for a, b in pairs:
                    diff = data["X"][a] - data["X"][b]
                    label = 1.0 if data["y"][a] > data["y"][b] else 0.0
                    if data["y"][a] == data["y"][b]:
                        continue
                    val_diffs.append(diff)
                    val_labels.append(label)
            else:
                for a, b in pairs:
                    if data["y"][a] == data["y"][b]:
                        continue
                    train_diffs.append(data["X"][a] - data["X"][b])
                    train_labels.append(1.0 if data["y"][a] > data["y"][b] else 0.0)
        if not train_diffs or not val_diffs:
            report[str(horizon)] = {"n_train_pairs": len(train_diffs), "n_val_pairs": len(val_diffs)}
            continue
        Xtr = np.asarray(train_diffs)
        ytr = np.asarray(train_labels)
        Xva = np.asarray(val_diffs)
        yva = np.asarray(val_labels)
        entry: dict[str, Any] = {
            "n_train_pairs": int(Xtr.shape[0]),
            "n_val_pairs": int(Xva.shape[0]),
            "models": {},
        }
        for name, model in (
            ("logistic", LogisticRegression(max_iter=1000)),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1
                ),
            ),
        ):
            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xva)[:, 1]
            accuracy = float(np.mean((proba > 0.5) == (yva > 0.5)))
            auc = float(roc_auc_score(yva, proba)) if len(set(yva)) > 1 else None
            # Win-count ordering inside each val state -> Kendall vs realised.
            # Batch every state's pair differences into one predict call:
            # a per-pair ``predict_proba`` made this loop take minutes.
            tau_inputs: list[tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]] = []
            for rows in val_group_rows.values():
                pairs = [
                    (i, j)
                    for i in range(len(rows))
                    for j in range(i + 1, len(rows))
                    if data["y"][rows[i]] != data["y"][rows[j]]
                ]
                if not pairs:
                    continue
                diffs = np.asarray(
                    [data["X"][rows[i]] - data["X"][rows[j]] for i, j in pairs]
                )
                realised = np.asarray([data["y"][r] for r in rows])
                tau_inputs.append((diffs, realised, pairs))
            taus = []
            if tau_inputs:
                flat = np.vstack([entry[0] for entry in tau_inputs])
                proba_all = model.predict_proba(flat)[:, 1]
                offset = 0
                for diffs, realised, pairs in tau_inputs:
                    probs = proba_all[offset : offset + diffs.shape[0]]
                    offset += diffs.shape[0]
                    wins = np.zeros(realised.size)
                    for (i, j), p in zip(pairs, probs):
                        if p > 0.5:
                            wins[i] += 1.0
                        elif p < 0.5:
                            wins[j] += 1.0
                    if wins.std() > 0 and realised.std() > 0:
                        from scipy import stats as _stats

                        tau = _stats.kendalltau(wins, realised).statistic
                        if np.isfinite(tau):
                            taus.append(float(tau))
            entry["models"][name] = {
                "pairwise_accuracy": accuracy,
                "pairwise_auc": auc,
                "within_state_kendall_from_wins": (
                    float(np.mean(taus)) if taus else None
                ),
                "n_states_ranked": len(taus),
            }
        report[str(horizon)] = entry
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the audit and write the JSON artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-json", default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--target", default="state_mean")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir)
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))["split"]
    val_keys = set(split["val_state_keys"])

    data = _load(dataset_dir, args.target)
    is_val = np.asarray([s in val_keys for s in data["state"]])

    started = time.perf_counter()
    # Per-problem breakdown: the Phase-2.8 gate is evaluated across problems
    # (a pooled SNR can hide a problem whose actions are indistinguishable).
    per_problem: dict[str, Any] = {}
    problems = np.asarray(data["problem"])
    for name in sorted(set(problems.tolist())):
        mask = problems == name
        subset = {key: value for key, value in data.items() if key != "state"}
        subset["state"] = [s for s, keep in zip(data["state"], mask) if keep]
        for key in ("X", "y", "horizon", "kind", "rewards"):
            subset[key] = data[key][mask]
        per_problem[name] = {
            "task1_action_effect_variance": task1_variance(subset, args.horizons)
        }
    report: dict[str, Any] = {
        "config": vars(args),
        "task1_action_effect_variance": task1_variance(data, args.horizons),
        "task1_by_problem": per_problem,
        "task2_action_only_learnability": task2_action_only(
            data, is_val, args.horizons, args.seed
        ),
        "task3_pairwise_ranking": task3_pairwise(
            data, is_val, args.horizons, args.seed
        ),
        "wall_time_sec": time.perf_counter() - started,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {out_path}")

    print("\n=== Task 1: action effect variance / SNR ===")
    for horizon, stats in sorted(report["task1_action_effect_variance"].items(), key=lambda kv: int(kv[0])):
        print(
            f"  h={horizon:>2}: action_var={stats['action_variance']:.6f} "
            f"noise={stats['rollout_noise_variance']:.6f} SNR={stats['snr_pooled']:.2f} "
            f"(median per-state {stats['snr_per_state_median']:.2f}, "
            f"{stats['fraction_states_snr_above_1']:.2f} of states >1)"
        )
    print("\n=== Task 2: action-only vs state-only vs concat (val Spearman) ===")
    for subset, models in report["task2_action_only_learnability"].items():
        for name, entry in models.items():
            row = " ".join(
                f"h{h}={entry['per_horizon'][h].get('spearman_mean')}"
                for h in sorted(entry["per_horizon"], key=int)
            )
            print(f"  {subset:12} {name:24} {row}")
    print("\n=== Task 3: pairwise ranking (val) ===")
    for horizon, entry in sorted(report["task3_pairwise_ranking"].items(), key=lambda kv: int(kv[0])):
        if "models" not in entry:
            print(f"  h={horizon}: {entry}")
            continue
        for name, m in entry["models"].items():
            print(
                f"  h={horizon:>2} {name:14} acc={m['pairwise_accuracy']:.4f} "
                f"auc={m['pairwise_auc']} kendall={m['within_state_kendall_from_wins']} "
                f"n_states={m['n_states_ranked']}"
            )
    return report


if __name__ == "__main__":
    main()
