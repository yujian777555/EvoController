from __future__ import annotations

"""Phase 2 Redesign, Experiment A: train the Evolution Outcome Predictor.

Implements the training half of ``docs/PHASE2_REDESIGN_PLAN.md`` Experiment A.
Instead of imitating recorded actions (Phase 1/1.5/1.75 behavior cloning),
the outcome predictor learns the long-horizon consequence of an action::

    (encoded state history, action features) -> absolute future HV at t + h

for every horizon ``h`` in ``--horizons``. The trained artifacts written to
``--out-dir`` are:

* ``predictor.pt`` — the fitted
  :class:`controller.outcome_predictor.OutcomePredictor`;
* ``encoder.json`` — the fitted :class:`controller.state_encoder.StateEncoder`
  (z-score statistics of the training distribution);
* ``training_meta.json`` — full run metadata: CLI config, sample counts,
  train/validation loss curves, horizons, window, wall time, UTC timestamp.

The train/validation split is by trajectory id (never by sample), so windows
from one optimization run never appear on both sides of the split.

Example:
    ``python experiments/train_outcome_predictor.py --epochs 300``
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
    # Allow ``python experiments/train_outcome_predictor.py`` from the repo
    # root: the script directory (not the repo root) is on sys.path then.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.dataset import (
    build_outcome_samples,
    load_trajectories,
    train_val_split,
)
from controller.outcome_predictor import OutcomePredictor
from controller.state_encoder import StateEncoder

#: Default Phase-1.75 500-trajectory full-action training corpus.
DEFAULT_TRAIN_DIRS: tuple[str, ...] = ("results/trajectory_phase1_75",)
#: Default artifact directory for the trained outcome predictor.
DEFAULT_OUT_DIR = "results/phase2_outcome"
#: Default prediction horizons (generations ahead).
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 20)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the outcome-predictor training run.

    Args:
        argv: Argument list to parse; ``None`` reads ``sys.argv``.

    Returns:
        Parsed namespace with data, model, training, and output settings.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Phase-2 Experiment A: train an Evolution Outcome Predictor that "
            "maps (state history, action) to absolute future hypervolume at "
            "multiple horizons."
        )
    )
    parser.add_argument(
        "--train-dirs",
        nargs="+",
        default=list(DEFAULT_TRAIN_DIRS),
        help=(
            "Directories with training trajectory JSONs (union is used). "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Artifact output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="History window of the state encoder in generations (default: %(default)s).",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
        help="Prediction horizons in generations ahead (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Training epochs of the outcome predictor (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of trajectories held out for validation, split by "
            "trajectory id (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=0,
        help="Seed for model init/training and the train/val split (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print predictor training progress.",
    )
    return parser.parse_args(argv)


def _load_train_trajectories(
    train_dirs: Sequence[str | Path],
) -> list[list[dict[str, Any]]]:
    """Load the concatenation of all training trajectories.

    Args:
        train_dirs: Directories with trajectory JSONs; missing directories
            are skipped with a warning.

    Returns:
        Loaded generation-sorted trajectories from every directory.

    Raises:
        FileNotFoundError: If none of the directories exists.
        ValueError: If no trajectories were found.
    """
    trajectories: list[list[dict[str, Any]]] = []
    any_found = False
    for directory in train_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            print(f"[warn] training directory missing, skipped: {directory}")
            continue
        any_found = True
        loaded = load_trajectories(directory)
        print(f"[load] {directory}: {len(loaded)} trajectories")
        trajectories.extend(loaded)
    if not any_found:
        raise FileNotFoundError(
            f"no training directory found among {list(map(str, train_dirs))}"
        )
    if not trajectories:
        raise ValueError(
            f"no training trajectories found in {list(map(str, train_dirs))}"
        )
    return trajectories


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Train the outcome predictor and persist all artifacts.

    Pipeline: load trajectories from every ``--train-dirs`` entry -> fit the
    :class:`StateEncoder` on the union -> build ``(history, action) ->
    future HV`` samples via ``build_outcome_samples`` -> split by trajectory
    id -> train :class:`OutcomePredictor` -> write ``predictor.pt``,
    ``encoder.json`` and ``training_meta.json`` to ``--out-dir``.

    Args:
        args: Parsed arguments from :func:`parse_args`.

    Returns:
        The ``training_meta.json`` payload.

    Raises:
        ValueError: If sample construction yields zero samples.
    """
    start = time.perf_counter()
    horizons = sorted({int(h) for h in args.horizons})
    trajectories = _load_train_trajectories(args.train_dirs)
    print(
        f"[load] {len(trajectories)} trajectories total; "
        f"fitting StateEncoder(window={args.window})"
    )
    encoder = StateEncoder(args.window).fit(trajectories)

    X, y, traj_ids, _sample_indices = build_outcome_samples(
        trajectories, encoder, args.window, horizons=horizons
    )
    if X.shape[0] == 0:
        raise ValueError(
            "build_outcome_samples produced zero samples; check that the "
            "training trajectories are long enough for the largest horizon "
            f"({max(horizons)})"
        )
    print(
        f"[samples] n={X.shape[0]}, input_dim={X.shape[1]}, "
        f"horizons={horizons}, trajectories={len(np.unique(traj_ids))}"
    )

    # Outcome samples are unweighted; train_val_split requires a weight
    # vector, so a uniform one is supplied.
    w = np.ones(X.shape[0], dtype=float)
    split = train_val_split(
        X, y, w, traj_ids, args.val_fraction, args.train_seed
    )
    print(
        f"[split] n_train={split['X_train'].shape[0]}, "
        f"n_val={split['X_val'].shape[0]} "
        f"(by trajectory, val_fraction={args.val_fraction})"
    )

    predictor = OutcomePredictor(
        input_dim=int(X.shape[1]),
        horizons=horizons,
        seed=args.train_seed,
        lr=args.lr,
    )
    history = predictor.fit(
        split["X_train"],
        split["y_train"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=split["X_val"],
        y_val=split["y_val"],
        verbose=args.verbose,
    )
    train_loss = [float(v) for v in history["train_loss"]]
    val_loss = [float(v) for v in history["val_loss"]]
    if train_loss:
        print(
            f"[train] final train_loss={train_loss[-1]:.6f}"
            + (
                f", final val_loss={val_loss[-1]:.6f}"
                if val_loss
                else ""
            )
        )

    wall_time_sec = time.perf_counter() - start
    meta: dict[str, Any] = {
        "config": {
            "train_dirs": [str(d) for d in args.train_dirs],
            "out_dir": str(args.out_dir),
            "window": int(args.window),
            "horizons": horizons,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "val_fraction": float(args.val_fraction),
            "train_seed": int(args.train_seed),
        },
        "n_samples": int(X.shape[0]),
        "n_train": int(split["X_train"].shape[0]),
        "n_val": int(split["X_val"].shape[0]),
        "n_trajectories": int(len(trajectories)),
        "input_dim": int(X.shape[1]),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "horizons": horizons,
        "window": int(args.window),
        "wall_time_sec": float(wall_time_sec),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(out_dir / "predictor.pt")
    encoder.save(out_dir / "encoder.json")
    meta_path = out_dir / "training_meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(
        f"[done] wrote predictor.pt, encoder.json, training_meta.json -> "
        f"{out_dir} ({wall_time_sec:.2f}s wall)"
    )
    return meta


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point: parse arguments and train the outcome predictor.

    Args:
        argv: Optional argument list; ``None`` reads ``sys.argv``.

    Returns:
        The ``training_meta.json`` payload.
    """
    return run_training(parse_args(argv))


if __name__ == "__main__":
    main()
