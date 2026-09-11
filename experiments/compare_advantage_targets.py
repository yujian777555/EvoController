from __future__ import annotations

"""Phase 2.75D Task 3: compare the three advantage targets head to head.

``docs/PHASE2_75D_PLAN.md`` asks for a systematic comparison of three learning
targets on *one and the same* intervention dataset:

``state_mean`` (Target A)
    ``R(action) - mean(R(all actions))`` — superiority over the contemporary
    candidate field.
``default_action`` (Target B)
    ``R(action) - R(default NSGA-II action)`` — improvement over the standard
    evolutionary policy. Requires the ``kind="default"`` candidate that
    ``evaluate-horizon --include-default-action`` inserts (Phase 2.75D protocol
    fix); on older corpora this target degrades to the controller baseline.
``future_improvement`` (Target C)
    ``future_metric - current_metric`` — the branch's absolute hypervolume
    gain over the snapshot (``= mean_reward``), with no centering at all.

All three datasets share ``X`` and the state keys and differ **only** in
``y_adv`` (enforced: the script refuses to compare datasets whose features or
state keys disagree). Every target is therefore trained with the same
architecture, seed, epochs and, crucially, the *same state-level train/val
split* — the split is computed once and applied to all targets, so the
comparison isolates the target definition.

Decision quality is reported per ``(target, horizon)``: Spearman, Kendall,
oracle hit rate, regret and oracle gap, plus an ``overall`` average, written to
``--out`` (default ``results/phase2_75d/target_comparison.json``) together with
the flat ``ranking`` table used for the paper figure.

Dataset layout (as written by ``experiments/build_intervention_dataset_v2.py``)::

    {dataset_dir}/{target}/intervention_dataset_{problem}.npz
    {dataset_dir}/{target}/intervention_meta.json

Example:
    ``python experiments/compare_advantage_targets.py \\
        --dataset-dir results/phase2_75d/dataset --epochs 300``
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    # Allow ``python experiments/compare_advantage_targets.py`` from the repo
    # root: the script directory (not the repo root) is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.advantage_predictor import AdvantagePredictor, ranking_metrics
from experiments import train_advantage_predictor as tap
from experiments.run_phase2_75c import load_problem_arrays

#: Directory holding one sub-directory per advantage target.
DEFAULT_DATASET_DIR = "results/phase2_75d/dataset"
#: Output of the comparison.
DEFAULT_OUT = "results/phase2_75d/target_comparison.json"
#: Trained models of the three targets.
DEFAULT_MODEL_DIR = "results/phase2_75d/models"
#: Targets compared by default (A, B, C of the plan).
DEFAULT_TARGETS: tuple[str, ...] = (
    "state_mean",
    "default_action",
    "future_improvement",
)
#: Advantage horizons.
DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)
#: Scalar metrics averaged into the ``overall`` block.
_SCALAR_METRICS: tuple[str, ...] = (
    "spearman_mean",
    "kendall_mean",
    "oracle_hit_rate",
    "regret_mean",
    "oracle_gap_mean",
)
#: Keys that identify one intervention row across targets.
_ROW_KEYS: tuple[str, ...] = (
    "problem",
    "seed",
    "generation",
    "horizon",
    "candidate_index",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments of the target comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2.75D Task 3: train the same AdvantagePredictor on the "
            "state_mean / default_action / future_improvement targets of one "
            "shared intervention dataset and compare decision quality."
        )
    )
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR,
                        help="Root with one sub-directory per target "
                        "(default: %(default)s).")
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS),
                        help="Target sub-directories to compare (default: %(default)s).")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT,
                        help="Comparison JSON (default: %(default)s).")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Where the per-target models are saved "
                        "(default: %(default)s).")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs per target (default: %(default)s).")
    parser.add_argument("--contrastive", choices=["on", "off"], default="on",
                        help="Paired hinge training (default: %(default)s).")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS),
                        help="Advantage horizons (default: %(default)s).")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of states held out, shared by all targets "
                        "(default: %(default)s).")
    parser.add_argument("--train-seed", type=int, default=0,
                        help="Seed of the split and of the networks (default: %(default)s).")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 128],
                        help="Hidden layer widths (default: %(default)s).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size (default: %(default)s).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: %(default)s).")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="Hinge margin (default: %(default)s).")
    parser.add_argument("--problems", nargs="+", default=None,
                        help="Problem whitelist; default: every problem found "
                        "in the first target directory.")
    parser.add_argument("--keep-controller", action="store_true",
                        help="Keep the controller candidate in the ranking pool "
                        "(default: dropped, matching "
                        "experiments/analyze_advantage_generalization.py).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print the per-epoch training loss.")
    return parser.parse_args(argv)


def row_keys(arrays: dict[str, np.ndarray]) -> list[tuple[str, int, int, int, int]]:
    """Identity tuple of every row, in file order."""
    return [
        (
            str(arrays["problem"][index]),
            int(arrays["seed"][index]),
            int(arrays["generation"][index]),
            int(arrays["horizon"][index]),
            int(arrays["candidate_index"][index]),
        )
        for index in range(len(arrays["problem"]))
    ]


def discover_problems(dataset_dir: str | Path) -> list[str]:
    """Sorted problem names found in a target's npz files.

    Raises:
        FileNotFoundError: If the directory holds no dataset file.
    """
    labels: set[str] = set()
    files = sorted(Path(dataset_dir).glob("intervention_dataset_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no intervention_dataset_*.npz in {dataset_dir}"
        )
    for path in files:
        with np.load(path) as arrays:
            labels.update(str(value) for value in np.unique(arrays["problem"]))
    return sorted(labels)


def load_targets(
    dataset_dir: str | Path,
    targets: Sequence[str],
    problems: Sequence[str] | None,
    *,
    drop_controller: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[str]], list[str]]:
    """Load every target's dataset and verify they describe the same rows.

    Args:
        dataset_dir: Root with one sub-directory per target.
        targets: Target names.
        problems: Optional problem whitelist; ``None`` uses the problems of the
            first target.
        drop_controller: Drop ``kind == "controller"`` rows.

    Returns:
        ``(datasets, files, problems)`` with ``datasets[target]`` the merged
        arrays.

    Raises:
        FileNotFoundError: If a target directory has no dataset.
        ValueError: If the targets disagree on X or on the row identity, i.e.
            they are not the same intervention dataset with different targets.
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {root}")
    first_target = str(targets[0])
    resolved_problems = (
        [str(problem) for problem in problems]
        if problems
        else discover_problems(root / first_target)
    )
    probe, probe_files = load_problem_arrays(
        root / first_target, resolved_problems, drop_controller=drop_controller
    )
    reference_keys = row_keys(probe)
    datasets: dict[str, dict[str, np.ndarray]] = {first_target: probe}
    files: dict[str, list[str]] = {first_target: probe_files}
    for target in targets[1:]:
        arrays, target_files = load_problem_arrays(
            root / str(target), resolved_problems, drop_controller=drop_controller
        )
        if row_keys(arrays) != reference_keys:
            raise ValueError(
                f"target {target!r} does not describe the same rows as "
                f"{first_target!r}: the datasets must share one intervention "
                f"grid and differ only in y_adv"
            )
        if not np.array_equal(
            np.asarray(arrays["X"], dtype=np.float64),
            np.asarray(probe["X"], dtype=np.float64),
        ):
            raise ValueError(
                f"target {target!r} has a different feature matrix X than "
                f"{first_target!r}; only y_adv may differ between targets"
            )
        datasets[str(target)] = arrays
        files[str(target)] = target_files
    return datasets, files, list(resolved_problems)


def shared_split(
    blocks_by_target: dict[str, list[dict[str, Any]]],
    *,
    val_fraction: float,
    train_seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Split every target on the **same** state-level train/val partition.

    The partition is derived once (from the first target's state keys) and
    applied to all targets, so the comparison cannot be confounded by a
    different split.

    Args:
        blocks_by_target: State blocks per target.
        val_fraction: Fraction of states held out.
        train_seed: Seed of the state permutation.

    Returns:
        ``(train_blocks, val_blocks, split_record)``.

    Raises:
        ValueError: If a target's state keys differ from the reference keys.
    """
    reference = next(iter(blocks_by_target.values()))
    reference_keys = sorted(str(block["key"]) for block in reference)
    for target, blocks in blocks_by_target.items():
        keys = sorted(str(block["key"]) for block in blocks)
        if keys != reference_keys:
            raise ValueError(
                f"target {target!r} covers different states than the reference "
                f"target; the comparison needs one shared state set"
            )
    _train, val = tap.split_blocks(reference, float(val_fraction), int(train_seed))
    val_keys = {str(block["key"]) for block in val}
    train_keys = {str(block["key"]) for block in reference} - val_keys
    train_blocks: dict[str, list[dict[str, Any]]] = {}
    val_blocks: dict[str, list[dict[str, Any]]] = {}
    for target, blocks in blocks_by_target.items():
        train_blocks[target] = [
            block for block in blocks if str(block["key"]) in train_keys
        ]
        val_blocks[target] = [block for block in blocks if str(block["key"]) in val_keys]
    split_record = {
        "unit": "state = (problem, seed, generation); shared by every target",
        "n_states_total": len(reference),
        "n_states_train": len(train_keys),
        "n_states_val": len(val_keys),
        "train_state_keys": sorted(train_keys),
        "val_state_keys": sorted(val_keys),
        "shared_across_targets": True,
    }
    return train_blocks, val_blocks, split_record


def train_target(
    target: str,
    train_blocks: Sequence[dict[str, Any]],
    val_blocks: Sequence[dict[str, Any]],
    horizons: Sequence[int],
    args: argparse.Namespace,
    input_dim: int,
) -> dict[str, Any]:
    """Train one target's predictor on the shared split.

    Returns:
        Record with the model path, row counts and loss curves.
    """
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    predictor = AdvantagePredictor(
        input_dim=int(input_dim),
        horizons=[int(h) for h in horizons],
        hidden_dims=tuple(int(w) for w in args.hidden_dims),
        seed=int(args.train_seed),
        lr=float(args.lr),
    )
    x_val, y_val = tap.stack_blocks(val_blocks)
    if str(args.contrastive) == "on":
        x_train, y_train, x_neg, y_neg = tap.contrastive_pairs(train_blocks)
        history = predictor.fit(
            x_train,
            y_train,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            X_val=x_val,
            y_val=y_val,
            X_neg=x_neg,
            y_neg=y_neg,
            margin=float(args.margin),
            verbose=bool(args.verbose),
        )
        n_rows = int(x_train.shape[0])
        n_pairs = int(x_neg.shape[0])
    else:
        x_train, y_train = tap.stack_blocks(train_blocks)
        history = predictor.fit(
            x_train,
            y_train,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            X_val=x_val,
            y_val=y_val,
            verbose=bool(args.verbose),
        )
        n_rows = int(x_train.shape[0])
        n_pairs = 0
    model_path = model_dir / f"model_{target}.pt"
    predictor.save(model_path)
    print(
        f"[train] {target}: {len(train_blocks)} train states, {n_rows} rows, "
        f"{n_pairs} pairs -> {model_path}"
    )
    return {
        "model": str(model_path),
        "n_states_train": len(train_blocks),
        "n_states_val": len(val_blocks),
        "n_rows": n_rows,
        "n_contrastive_pairs": n_pairs,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
    }


def evaluate_target(
    predictor: AdvantagePredictor,
    val_blocks: Sequence[dict[str, Any]],
    horizons: Sequence[int],
) -> dict[str, Any]:
    """Decision-quality metrics of one target's predictor on the val states.

    The realized side is the *target's own* held-out values: the question is
    whether the model learned the ordering the target defines, which is what
    the target comparison is about.

    Returns:
        ``{"n_states", "per_horizon": {str(h): metrics}, "overall": {...}}``.
    """
    per_horizon: dict[str, Any] = {}
    for column, horizon in enumerate(horizons):
        predictions = np.asarray(
            [predictor.predict(block["X"])[:, column] for block in val_blocks]
        )
        realized = np.asarray([block["y"][:, column] for block in val_blocks])
        per_horizon[str(int(horizon))] = ranking_metrics(predictions, realized)
    overall: dict[str, Any] = {"n_states": len(val_blocks)}
    for metric in _SCALAR_METRICS:
        values = [
            entry[metric]
            for entry in per_horizon.values()
            if entry.get(metric) is not None
        ]
        overall[metric] = float(np.mean(values)) if values else None
    return {"n_states": len(val_blocks), "per_horizon": per_horizon, "overall": overall}


def ranking_table(
    targets: dict[str, Any], horizons: Sequence[int]
) -> list[dict[str, Any]]:
    """Flat ``(target, horizon)`` table of the decision metrics."""
    table: list[dict[str, Any]] = []
    for target in sorted(targets):
        entry = targets[target]
        for horizon in horizons:
            metrics = entry["per_horizon"].get(str(int(horizon)), {})
            row: dict[str, Any] = {"target": target, "horizon": int(horizon)}
            for metric in (*_SCALAR_METRICS, "n_groups"):
                row[metric] = metrics.get(metric)
            table.append(row)
        overall = dict(entry["overall"])
        overall.pop("n_states", None)
        table.append(
            {
                "target": target,
                "horizon": None,
                **{metric: overall.get(metric) for metric in _SCALAR_METRICS},
                "n_groups": entry["overall"].get("n_states"),
            }
        )
    return table


def best_target_per_horizon(
    targets: dict[str, Any], horizons: Sequence[int]
) -> dict[str, str | None]:
    """Target with the highest Spearman per horizon (and overall)."""
    best: dict[str, str | None] = {}
    for horizon in list(horizons) + [None]:
        key = "overall" if horizon is None else str(int(horizon))
        candidates: list[tuple[float, str]] = []
        for target in sorted(targets):
            entry = (
                targets[target]["overall"]
                if horizon is None
                else targets[target]["per_horizon"].get(key, {})
            )
            value = entry.get("spearman_mean")
            if value is not None:
                candidates.append((float(value), target))
        best["overall" if horizon is None else key] = (
            max(candidates)[1] if candidates else None
        )
    return best


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Train every target on the shared split and write the comparison.

    Returns:
        The payload written to ``--out``.

    Raises:
        FileNotFoundError: If a target dataset is missing.
        ValueError: If the targets do not share one intervention grid.
    """
    started = time.perf_counter()
    targets = [str(target) for target in args.targets]
    if not targets:
        raise ValueError("--targets must not be empty")
    dataset_dir = Path(args.dataset_dir)
    datasets, files, problems = load_targets(
        dataset_dir,
        targets,
        args.problems,
        drop_controller=not bool(args.keep_controller),
    )
    horizons = [int(h) for h in args.horizons]
    blocks_by_target: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, int] = {}
    for target, arrays in datasets.items():
        blocks, target_skipped = tap.build_state_blocks(arrays, horizons)
        if not blocks:
            raise ValueError(
                f"target {target!r} has no usable state blocks for horizons "
                f"{horizons}"
            )
        blocks_by_target[target] = blocks
        skipped[target] = len(target_skipped)
    train_blocks, val_blocks, split_record = shared_split(
        blocks_by_target,
        val_fraction=float(args.val_fraction),
        train_seed=int(args.train_seed),
    )
    input_dim = int(np.asarray(next(iter(datasets.values()))["X"]).shape[1])
    n_samples = int(np.asarray(next(iter(datasets.values()))["y_adv"]).size)

    identical: list[list[str]] = []
    for index, target in enumerate(targets):
        for other in targets[index + 1 :]:
            if np.array_equal(datasets[target]["y_adv"], datasets[other]["y_adv"]):
                identical.append([target, other])

    report: dict[str, Any] = {}
    for target in targets:
        record = train_target(
            target,
            train_blocks[target],
            val_blocks[target],
            horizons,
            args,
            input_dim,
        )
        predictor = AdvantagePredictor.load(record["model"])
        evaluation = evaluate_target(predictor, val_blocks[target], horizons)
        report[target] = {**record, **evaluation}
    table = ranking_table(report, horizons)
    payload: dict[str, Any] = {
        "config": {
            "dataset_dir": str(dataset_dir),
            "targets": targets,
            "horizons": horizons,
            "epochs": int(args.epochs),
            "contrastive": str(args.contrastive),
            "margin": float(args.margin),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "hidden_dims": [int(w) for w in args.hidden_dims],
            "train_seed": int(args.train_seed),
            "val_fraction": float(args.val_fraction),
            "controller_candidate_kept": bool(args.keep_controller),
            "model_dir": str(args.model_dir),
            "protocol": (
                "one shared intervention dataset (identical X and state keys, "
                "only y_adv differs); identical architecture/epochs/seed and "
                "one shared state-level split for every target"
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "data": {
            "targets": targets,
            "problems": problems,
            "input_dim": input_dim,
            "n_samples": n_samples,
            "n_states": len(next(iter(blocks_by_target.values()))),
            "n_states_dropped": skipped,
            "files": files,
            "integrity": (
                "row identity and X verified identical across targets; "
                "identical y_adv pairs: " + (str(identical) if identical else "none")
            ),
        },
        "split": split_record,
        "targets": report,
        "ranking": table,
        "best_target_by_spearman": best_target_per_horizon(report, horizons),
        "wall_time_sec": float(time.perf_counter() - started),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    _print_table(table, payload["best_target_by_spearman"])
    print(f"[done] wrote {out_path}")
    return payload


def _print_table(table: Sequence[dict[str, Any]], best: dict[str, str | None]) -> None:
    """Compact stdout table of the comparison."""
    print(f"{'target':22}{'h':>5}{'rho':>9}{'tau':>9}{'hit':>8}{'regret':>10}{'gap':>9}")

    def _fmt(value: Any) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    for row in table:
        horizon = "all" if row["horizon"] is None else str(row["horizon"])
        print(
            f"{row['target']:22}{horizon:>5}{_fmt(row['spearman_mean']):>9}"
            f"{_fmt(row['kendall_mean']):>9}{_fmt(row['oracle_hit_rate']):>8}"
            f"{_fmt(row['regret_mean']):>10}{_fmt(row['oracle_gap_mean']):>9}"
        )
    print(f"[best by spearman] {json.dumps(best)}")


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point."""
    return run_comparison(parse_args(argv))


if __name__ == "__main__":
    main()
