from __future__ import annotations

"""Phase 2.75, Task 4: train the action-advantage predictor.

Reads every ``intervention_dataset_{problem}.npz`` written by
``experiments/build_intervention_dataset.py``, merges them, splits
**by evolution state** (all candidates *and* all horizons of one
``(problem, seed, generation)`` stay on the same side — otherwise the same
population state leaks into both splits), and trains two models with an
identical budget:

* ``model_mse.pt`` — plain sample-weighted MSE;
* ``model_contrastive.pt`` — the same, plus a paired hinge term against
  action-shuffled negatives (``--contrastive on``, the default).

Negatives are built inside each state: candidates are ranked by their mean
advantage over horizons, and every candidate except the worst is paired with
the *next worse* candidate of the same state, so the hinge asks the model to
order the state's own candidate set correctly — the failure mode Phase-2B's
D1/D2 diagnostics measured. Both arms are trained on exactly the same rows
(the subset with an available negative, ~all rows when a state has several
candidates) so the comparison isolates the loss, not the data.

Decision quality (not regression error) is what Phase 2.75 needs, so the
validation report uses :func:`controller.advantage_predictor.ranking_metrics`
per horizon: Spearman/Kendall, oracle hit rate, regret and oracle gap on the
held-out states.

Outputs (``--out-dir``): ``model_mse.pt``, ``model_contrastive.pt`` (unless
``--contrastive off``), ``encoder.json`` (a copy of the encoder the dataset
was built with, so the model directory is self-contained) and
``training_meta.json`` (config, loss curves, split statistics, per-horizon
validation metrics of both arms).

Example:
    ``python experiments/train_advantage_predictor.py --epochs 300``
    ``python experiments/train_advantage_predictor.py --contrastive off``
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
    # Allow ``python experiments/train_advantage_predictor.py`` from the repo
    # root: the script directory (not the repo root) is on sys.path then.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.advantage_predictor import AdvantagePredictor, ranking_metrics

#: Directory with the intervention npz/meta artifacts.
DEFAULT_DATASET_DIR = "results/phase2_75"
#: Model output directory (Phase-2.75 isolated).
DEFAULT_OUT_DIR = "results/phase2_75/model"
#: Horizon grid of the advantage predictor.
DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)
#: Scalar metrics averaged into the ``overall`` block of the report.
_SCALAR_METRICS: tuple[str, ...] = (
    "spearman_mean",
    "kendall_mean",
    "oracle_hit_rate",
    "regret_mean",
    "oracle_gap_mean",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments of the advantage-predictor training."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2.75 Task 4: train MSE and contrastive action-advantage "
            "predictors on same-snapshot intervention data (state-level "
            "train/val split, decision-quality validation)."
        )
    )
    parser.add_argument(
        "--dataset-dir", type=str, default=DEFAULT_DATASET_DIR,
        help="Directory with intervention_dataset_*.npz and "
        "intervention_meta.json (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=DEFAULT_OUT_DIR,
        help="Model output directory (default: %(default)s).",
    )
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs (default: %(default)s).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: %(default)s).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size (default: %(default)s).")
    parser.add_argument("--train-seed", type=int, default=0,
                        help="Seed of the split and of the networks (default: %(default)s).")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of *states* held out (default: %(default)s).")
    parser.add_argument(
        "--contrastive", choices=["on", "off"], default="on",
        help="'on' (default) trains the MSE control and the contrastive "
        "model; 'off' trains only the MSE model.",
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS),
        help="Advantage horizons to train on (default: %(default)s); must be "
        "present in the dataset.",
    )
    parser.add_argument(
        "--hidden-dims", nargs="+", type=int, default=[128, 128],
        help="Hidden layer widths of the MLP (default: %(default)s).",
    )
    parser.add_argument("--margin", type=float, default=0.0,
                        help="Hinge margin of the contrastive term (default: %(default)s).")
    parser.add_argument(
        "--encoder", type=str, default=None,
        help="Encoder JSON to embed in the model directory; default: the "
        "encoder recorded in intervention_meta.json.",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Print the per-epoch training loss.")
    return parser.parse_args(argv)


def load_merged_arrays(
    dataset_dir: str | Path,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load and concatenate every ``intervention_dataset_*.npz``.

    Args:
        dataset_dir: Directory with the npz artifacts.

    Returns:
        ``(arrays, files)`` with the concatenated columns ``X``, ``y_adv``,
        ``problem``, ``seed``, ``generation``, ``horizon`` and
        ``candidate_index``, plus the file names that contributed.

    Raises:
        FileNotFoundError: If the directory holds no npz file.
        ValueError: If a file lacks a required array.
    """
    directory = Path(dataset_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {directory}")
    files = sorted(directory.glob("intervention_dataset_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no intervention_dataset_*.npz in {directory}; run "
            f"experiments/build_intervention_dataset.py first"
        )
    required = ("X", "y_adv", "problem", "seed", "generation", "horizon",
                "candidate_index")
    columns: dict[str, list[np.ndarray]] = {name: [] for name in required}
    used: list[str] = []
    for path in files:
        with np.load(path) as arrays:
            missing = [name for name in required if name not in arrays]
            if missing:
                raise ValueError(f"{path} is missing arrays {missing}")
            for name in required:
                columns[name].append(np.asarray(arrays[name]))
        used.append(path.name)
    merged = {
        name: np.concatenate(values) if values else np.zeros(0)
        for name, values in columns.items()
    }
    return merged, used


def build_state_blocks(
    arrays: dict[str, np.ndarray], horizons: Sequence[int]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collapse the horizon axis into per-state candidate blocks.

    One row of the npz is a ``(state, horizon, candidate)`` observation, but
    the feature matrix carries no horizon information, so all horizons of a
    ``(state, candidate)`` pair share the same ``X``. Each block therefore
    holds one row per candidate and a target vector over ``horizons``.

    Args:
        arrays: Merged arrays from :func:`load_merged_arrays`.
        horizons: Requested horizons, ascending.

    Returns:
        ``(blocks, skipped)`` where each block is a dict with ``key``
        (``"problem|seed|gen"``), ``problem``, ``X`` (``(n_candidates, d)``),
        ``y`` (``(n_candidates, len(horizons))``) and ``mean_advantage``;
        ``skipped`` names the states dropped because their rows do not cover
        every (candidate, horizon) cell.
    """
    horizon_list = [int(h) for h in horizons]
    wanted = {int(h) for h in horizon_list}
    blocks: list[dict[str, Any]] = []
    skipped: list[str] = []
    keys = np.asarray(
        [
            f"{arrays['problem'][i]}|{int(arrays['seed'][i])}|"
            f"{int(arrays['generation'][i])}"
            for i in range(len(arrays["y_adv"]))
        ],
        dtype=object,
    )
    for key in sorted(set(keys.tolist())):
        rows = np.flatnonzero(keys == key)
        horizons_here = {int(arrays["horizon"][i]) for i in rows}
        if not wanted.issubset(horizons_here):
            skipped.append(str(key))
            continue
        candidates = sorted({int(arrays["candidate_index"][i]) for i in rows})
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        consistent = True
        for candidate in candidates:
            targets: list[float] = []
            reference: np.ndarray | None = None
            for horizon in horizon_list:
                match = [
                    i
                    for i in rows
                    if int(arrays["horizon"][i]) == horizon
                    and int(arrays["candidate_index"][i]) == candidate
                ]
                if len(match) != 1:
                    consistent = False
                    break
                index = match[0]
                if reference is None:
                    reference = np.asarray(arrays["X"][index], dtype=np.float64)
                elif not np.array_equal(reference, np.asarray(arrays["X"][index])):
                    raise ValueError(
                        f"{key}: candidate {candidate} has horizon-dependent "
                        f"features; the dataset is malformed"
                    )
                targets.append(float(arrays["y_adv"][index]))
            if not consistent or reference is None:
                break
            x_rows.append(reference)
            y_rows.append(np.asarray(targets, dtype=np.float64))
        if not consistent or len(x_rows) < 2:
            skipped.append(str(key))
            continue
        y_matrix = np.vstack(y_rows)
        blocks.append(
            {
                "key": str(key),
                "problem": str(key).split("|")[0],
                "X": np.vstack(x_rows),
                "y": y_matrix,
                "mean_advantage": y_matrix.mean(axis=1),
            }
        )
    return blocks, skipped


def split_blocks(
    blocks: Sequence[dict[str, Any]], val_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split blocks (states) into train/validation deterministically.

    The unit of the split is the **state**, never the row: every candidate and
    horizon of a state lands on the same side.

    Args:
        blocks: State blocks from :func:`build_state_blocks`.
        val_fraction: Fraction of states held out.
        seed: Seed of the state permutation.

    Returns:
        ``(train_blocks, val_blocks)``.

    Raises:
        ValueError: If ``val_fraction`` is outside ``[0, 1)`` or the split
            would leave one side empty.
    """
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError(f"val_fraction must lie in [0, 1), got {val_fraction}")
    ordered = sorted(blocks, key=lambda block: block["key"])
    order = np.random.Generator(np.random.PCG64(int(seed))).permutation(len(ordered))
    n_val = int(round(len(ordered) * float(val_fraction)))
    if float(val_fraction) > 0.0 and len(ordered) > 1:
        n_val = max(1, min(n_val, len(ordered) - 1))
    val_keys = {ordered[int(i)]["key"] for i in order[:n_val]}
    train = [block for block in ordered if block["key"] not in val_keys]
    val = [block for block in ordered if block["key"] in val_keys]
    if not train or not val:
        raise ValueError(
            f"split produced train={len(train)} val={len(val)}; provide more "
            f"states or a different --val-fraction"
        )
    return train, val


def contrastive_pairs(
    blocks: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Paired positives/negatives for the hinge term, state by state.

    Inside each state the candidates are ranked by mean advantage; candidate
    at rank ``r`` is paired with the candidate at rank ``r - 1`` (its next
    worse sibling), which is the hardest true ordering constraint of that
    state and teaches the model the state's own candidate order.

    Args:
        blocks: Blocks used for training.

    Returns:
        ``(X_pos, y_pos, X_neg, y_neg)``; the worst candidate of each state
        contributes no pair and is therefore absent from ``X_pos``.
    """
    x_pos: list[np.ndarray] = []
    y_pos: list[np.ndarray] = []
    x_neg: list[np.ndarray] = []
    y_neg: list[np.ndarray] = []
    for block in blocks:
        order = np.argsort(block["mean_advantage"], kind="stable")
        for rank in range(1, len(order)):
            better = int(order[rank])
            worse = int(order[rank - 1])
            x_pos.append(block["X"][better])
            y_pos.append(block["y"][better])
            x_neg.append(block["X"][worse])
            y_neg.append(block["y"][worse])
    if not x_pos:
        return (
            np.zeros((0, 0)),
            np.zeros((0, 0)),
            np.zeros((0, 0)),
            np.zeros((0, 0)),
        )
    return (
        np.vstack(x_pos),
        np.vstack(y_pos),
        np.vstack(x_neg),
        np.vstack(y_neg),
    )


def stack_blocks(
    blocks: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the ``X``/``y`` of several blocks."""
    if not blocks:
        return np.zeros((0, 0)), np.zeros((0, 0))
    return (
        np.vstack([block["X"] for block in blocks]),
        np.vstack([block["y"] for block in blocks]),
    )


def evaluate_blocks(
    predictor: AdvantagePredictor,
    blocks: Sequence[dict[str, Any]],
    horizons: Sequence[int],
) -> dict[str, Any]:
    """Decision-quality metrics per horizon on a set of state blocks.

    Args:
        predictor: Fitted advantage predictor.
        blocks: State blocks (the validation split).
        horizons: Horizon grid, aligned with the target columns.

    Returns:
        ``{"n_states", "per_horizon": {str(h): metrics}, "overall": {...}}``
        where ``overall`` averages the scalar metrics over horizons.
    """
    per_horizon: dict[str, dict[str, Any]] = {}
    for column, horizon in enumerate(horizons):
        predictions = []
        realized = []
        for block in blocks:
            predictions.append(predictor.predict(block["X"])[:, column])
            realized.append(block["y"][:, column])
        per_horizon[str(int(horizon))] = ranking_metrics(
            np.asarray(predictions, dtype=np.float64),
            np.asarray(realized, dtype=np.float64),
        )
    overall: dict[str, Any] = {"n_states": len(blocks)}
    for metric in _SCALAR_METRICS:
        values = [
            entry[metric]
            for entry in per_horizon.values()
            if entry.get(metric) is not None
        ]
        overall[metric] = float(np.mean(values)) if values else None
    return {"n_states": len(blocks), "per_horizon": per_horizon, "overall": overall}


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Train the MSE and (optionally) contrastive predictors, write artifacts.

    Args:
        args: Parsed arguments from :func:`parse_args`.

    Returns:
        The metadata payload written to ``training_meta.json``.

    Raises:
        FileNotFoundError: If the dataset, its meta, or the encoder is missing.
        ValueError: If no state block survives the (candidate, horizon) grid
            requirement.
    """
    from controller.state_encoder import StateEncoder

    started = time.perf_counter()
    dataset_dir = Path(args.dataset_dir)
    arrays, files = load_merged_arrays(dataset_dir)
    horizons = sorted({int(h) for h in args.horizons})
    blocks, skipped = build_state_blocks(arrays, horizons)
    if not blocks:
        raise ValueError(
            f"no usable state blocks for horizons {horizons} in {dataset_dir} "
            f"({len(skipped)} states lacked a full candidate x horizon grid)"
        )
    train_blocks, val_blocks = split_blocks(blocks, args.val_fraction, args.train_seed)
    print(
        f"[split] {len(train_blocks)} train / {len(val_blocks)} val states "
        f"({len(skipped)} dropped), horizons={horizons}"
    )

    x_train, y_train = stack_blocks(train_blocks)
    x_pos, y_pos, x_neg, y_neg = contrastive_pairs(train_blocks)
    x_val, y_val = stack_blocks(val_blocks)
    if x_pos.shape[0] == 0:
        raise ValueError("contrastive pairing produced no rows; need >= 2 candidates per state")

    input_dim = int(x_train.shape[1])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _new_predictor() -> AdvantagePredictor:
        return AdvantagePredictor(
            input_dim=input_dim,
            horizons=horizons,
            hidden_dims=tuple(int(w) for w in args.hidden_dims),
            seed=int(args.train_seed),
            lr=float(args.lr),
        )

    mse_model = _new_predictor()
    print(f"[train] mse: {x_pos.shape[0]} rows, input_dim={input_dim}")
    mse_history = mse_model.fit(
        x_pos,
        y_pos,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        X_val=x_val,
        y_val=y_val,
        verbose=bool(args.verbose),
    )
    mse_model.save(out_dir / "model_mse.pt")
    mse_metrics = evaluate_blocks(mse_model, val_blocks, horizons)
    print(f"[val] mse overall: {json.dumps(mse_metrics['overall'])}")

    contrastive_model: AdvantagePredictor | None = None
    contrastive_history: dict[str, list[float]] | None = None
    contrastive_metrics: dict[str, Any] | None = None
    if str(args.contrastive) == "on":
        contrastive_model = _new_predictor()
        print(
            f"[train] contrastive: {x_pos.shape[0]} positive / "
            f"{x_neg.shape[0]} paired negatives, margin={args.margin}"
        )
        contrastive_history = contrastive_model.fit(
            x_pos,
            y_pos,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            X_val=x_val,
            y_val=y_val,
            X_neg=x_neg,
            y_neg=y_neg,
            margin=float(args.margin),
            verbose=bool(args.verbose),
        )
        contrastive_model.save(out_dir / "model_contrastive.pt")
        contrastive_metrics = evaluate_blocks(contrastive_model, val_blocks, horizons)
        print(f"[val] contrastive overall: {json.dumps(contrastive_metrics['overall'])}")

    encoder_source = args.encoder
    if encoder_source is None:
        meta_path = dataset_dir / "intervention_meta.json"
        if meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as fh:
                encoder_source = json.load(fh).get("config", {}).get("encoder")
    if encoder_source is None:
        raise FileNotFoundError(
            "no encoder given and intervention_meta.json records none; pass "
            "--encoder"
        )
    encoder = StateEncoder.load(encoder_source)
    encoder.save(out_dir / "encoder.json")

    meta: dict[str, Any] = {
        "config": {
            "dataset_dir": str(dataset_dir),
            "dataset_files": files,
            "out_dir": str(out_dir),
            "encoder_source": str(encoder_source),
            "horizons": horizons,
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "train_seed": int(args.train_seed),
            "val_fraction": float(args.val_fraction),
            "contrastive": str(args.contrastive),
            "margin": float(args.margin),
            "hidden_dims": [int(w) for w in args.hidden_dims],
            "input_dim": input_dim,
        },
        "split": {
            "unit": "state = (problem, seed, generation); all candidates and horizons stay together",
            "n_states_total": len(blocks),
            "n_states_train": len(train_blocks),
            "n_states_val": len(val_blocks),
            "n_states_dropped": len(skipped),
            "dropped_states": skipped,
            "n_rows_train_positive": int(x_pos.shape[0]),
            "n_rows_train_full": int(x_train.shape[0]),
            "n_rows_val": int(x_val.shape[0]),
            "val_state_keys": sorted(block["key"] for block in val_blocks),
            "train_state_keys": sorted(block["key"] for block in train_blocks),
        },
        "losses": {
            "mse": mse_history,
            "contrastive": contrastive_history,
        },
        "validation": {
            "mse": mse_metrics,
            "contrastive": contrastive_metrics,
        },
        "artifacts": {
            "model_mse": "model_mse.pt",
            "model_contrastive": (
                "model_contrastive.pt" if contrastive_model is not None else None
            ),
            "encoder": "encoder.json",
        },
        "wall_time_sec": float(time.perf_counter() - started),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_dir / "training_meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"[done] wrote {meta_path}")
    return meta


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point."""
    return run_training(parse_args(argv))


if __name__ == "__main__":
    main()
