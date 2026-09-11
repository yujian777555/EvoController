from __future__ import annotations

"""Per-problem diagnosis of advantage-predictor ranking quality.

Phase 2.75C Task 3 asks why the Phase-2.75 planner fails on the difficult
landscapes (ZDT4/ZDT6) even though the pooled intervention SNR is high.
This script answers the *ranking* half of that question: it evaluates the
trained :class:`~controller.advantage_predictor.AdvantagePredictor` on the
held-out (validation) states only, broken down by problem and horizon, and
reports the decision-quality metrics from
:func:`~controller.advantage_predictor.ranking_metrics`.

It also reports, per problem, the intervention action-effect SNR so the
ranking quality can be read next to the signal strength available.

Usage::

    python experiments/analyze_advantage_generalization.py \
        --model-dir results/phase2_75/model \
        --dataset-dir results/phase2_75 \
        --out results/phase2_75/generalization_diagnosis.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Allow ``python experiments/analyze_advantage_generalization.py`` from the
# repo root: the script directory (not the repo root) is on sys.path then.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.advantage_predictor import (
    AdvantagePredictor,
    ranking_metrics,
)


def _state_key(problem: str, seed: int, generation: int) -> str:
    """Canonical ``"problem|seed|generation"`` state identifier."""
    return f"{problem}|{int(seed)}|{int(generation)}"


def _load_states(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    """Collect every intervention row keyed by state.

    Returns:
        Mapping ``state_key -> {"problem": str, "rows": np.ndarray}`` where
        ``rows`` holds the row indices of that state inside the concatenated
        dataset (problem files are concatenated in sorted order).
    """
    states: dict[str, dict[str, Any]] = {}
    offset = 0
    for path in sorted(dataset_dir.glob("intervention_dataset_*.npz")):
        data = np.load(path, allow_pickle=True)
        n = len(data["X"])
        starts: dict[str, list[int]] = defaultdict(list)
        for i in range(n):
            key = _state_key(
                str(data["problem"][i]), int(data["seed"][i]), int(data["generation"][i])
            )
            starts[key].append(offset + i)
        for key, rows in starts.items():
            states[key] = {
                "problem": key.split("|")[0],
                "rows": np.asarray(rows, dtype=int),
            }
        offset += n
    return states


def _rank_table(
    predictor: AdvantagePredictor,
    states: dict[str, dict[str, Any]],
    dataset_dir: Path,
    problems: Sequence[str],
    horizons: Sequence[int],
) -> dict[str, Any]:
    """Per problem x horizon ranking metrics on the given states."""
    # Concatenate datasets once so row indices are globally valid.
    xs, ys, hzs, kinds = [], [], [], []
    for path in sorted(dataset_dir.glob("intervention_dataset_*.npz")):
        data = np.load(path, allow_pickle=True)
        xs.append(data["X"])
        ys.append(data["y_adv"])
        hzs.append(data["horizon"])
        kinds.append(data["candidate_kind"])
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    hz = np.concatenate(hzs)
    kind = np.concatenate(kinds)
    pred = predictor.predict(X)

    report: dict[str, Any] = {}
    for problem in problems:
        per_horizon: dict[str, Any] = {}
        for hi, h in enumerate(horizons):
            groups_pred, groups_real = [], []
            for key, info in states.items():
                if info["problem"] != problem:
                    continue
                rows = info["rows"][hz[info["rows"]] == h]
                if rows.size < 2:
                    continue
                # Alternative candidates only: the controller action is the
                # index-0 arm under test, not part of the candidate pool whose
                # ranking the planner relies on.
                alt = rows[kind[rows] != "controller"]
                if alt.size >= 2:
                    rows = alt
                groups_pred.append(pred[rows, hi])
                groups_real.append(y[rows])
            if not groups_pred:
                per_horizon[str(h)] = {"n_groups": 0}
                continue
            metrics = ranking_metrics(np.asarray(groups_pred), np.asarray(groups_real))
            per_horizon[str(h)] = metrics
        report[problem] = per_horizon
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the diagnosis and write the JSON artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="results/phase2_75/model")
    parser.add_argument("--dataset-dir", default="results/phase2_75")
    parser.add_argument("--out", default="results/phase2_75/generalization_diagnosis.json")
    parser.add_argument("--problems", nargs="+", default=["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20])
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / "training_meta.json").read_text(encoding="utf-8"))
    val_keys = [str(k) for k in meta["split"]["val_state_keys"]]
    train_keys = set(meta["split"].get("train_state_keys", []))

    dataset_dir = Path(args.dataset_dir)
    all_states = _load_states(dataset_dir)
    val_states = {k: v for k, v in all_states.items() if k in set(val_keys)}

    payload: dict[str, Any] = {"config": vars(args), "n_val_states": len(val_states)}
    for name, model_file in (
        ("contrastive", "model_contrastive.pt"),
        ("mse", "model_mse.pt"),
    ):
        path = model_dir / model_file
        if not path.is_file():
            continue
        predictor = AdvantagePredictor.load(path)
        payload[f"val_{name}"] = _rank_table(
            predictor, val_states, dataset_dir, args.problems, args.horizons
        )

    snr_source = dataset_dir / "intervention_meta.json"
    if snr_source.is_file():
        snr_meta = json.loads(snr_source.read_text(encoding="utf-8"))
        payload["snr_by_problem"] = {
            problem: {
                h: entry.get("action_effect_snr")
                for h, entry in info.get("per_horizon", {}).items()
            }
            for problem, info in snr_meta.get("per_problem", {}).items()
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {out_path}")

    for model_name in ("val_contrastive", "val_mse"):
        table = payload.get(model_name)
        if not table:
            continue
        print(f"\n=== {model_name} (held-out states, alternatives only) ===")
        print(f"{'problem':8}{'SNR(h20)':>10}" + "".join(f"{'rho h=' + str(h):>12}" for h in args.horizons))
        for problem in args.problems:
            row = f"{problem:8}"
            snr = payload.get("snr_by_problem", {}).get(problem, {}).get("20")
            row += f"{snr:>10.2f}" if snr is not None else f"{'n/a':>10}"
            for h in args.horizons:
                entry = table.get(problem, {}).get(str(h), {})
                rho = entry.get("spearman_mean")
                row += f"{rho:>12.3f}" if rho is not None else f"{'n/a':>12}"
            print(row)
    return payload


if __name__ == "__main__":
    main()
