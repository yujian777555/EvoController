from __future__ import annotations

"""Phase 2 Redesign, Experiment A: evaluate the Evolution Outcome Predictor.

Loads the artifacts written by ``experiments/train_outcome_predictor.py``
(``predictor.pt``, ``encoder.json``, ``training_meta.json``) and scores the
predictor's absolute future-hypervolume forecasts against two forecasting
baselines at every trained horizon ``h``:

* **persistence** — ``hv[t + h] = hv[t]`` (the current HV of the sample,
  read directly from the source transition ``state.hv``);
* **linear trend** — ``hv[t + h] = hv[t] + h * mean(delta_hv)`` where the
  mean runs over the sample's history window ``[t - window, t)``.

Per horizon the script reports MSE, MAE and R^2 for the model and both
baselines, plus one-sided paired Wilcoxon signed-rank tests
(``zero_method="zsplit"``, ``alternative="less"``) on the per-sample squared
errors — model vs persistence and model vs linear — where ``less`` means the
model's errors are smaller.

Success criteria (Experiment A gate):

* ``model_beats_persistence_all_horizons`` — model MSE below persistence MSE
  at every horizon;
* ``r2_above_0.5_h1_h5`` — model R^2 above 0.5 at horizons 1 and 5 (the
  short-horizon regime where behavior cloning failed);
* ``long_horizon_h20_beats_baselines`` — model MSE below both baselines at
  horizon 20 (or the largest trained horizon if 20 was not trained).

Example:
    ``python experiments/evaluate_outcome_predictor.py``
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import wilcoxon

if __package__ in (None, ""):
    # Allow ``python experiments/evaluate_outcome_predictor.py`` from the
    # repo root: the script directory (not the repo root) is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.dataset import build_outcome_samples, load_trajectories
from controller.outcome_predictor import OutcomePredictor
from controller.state_encoder import StateEncoder

#: Default artifact directory written by the training script.
DEFAULT_MODEL_DIR = "results/phase2_outcome"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the outcome-predictor evaluation.

    Args:
        argv: Argument list to parse; ``None`` reads ``sys.argv``.

    Returns:
        Parsed namespace with model, evaluation-data, and output settings.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Phase-2 Experiment A: evaluate a trained Evolution Outcome "
            "Predictor against persistence and linear-trend forecasting "
            "baselines with paired Wilcoxon tests."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=(
            "Directory with predictor.pt, encoder.json and "
            "training_meta.json (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--eval-dirs",
        nargs="+",
        default=None,
        help=(
            "Directories with evaluation trajectory JSONs. Default: the "
            "train-dirs recorded in the model's training_meta.json."
        ),
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help=(
            "Evaluation JSON destination (default: "
            "<model-dir>/evaluation.json)."
        ),
    )
    return parser.parse_args(argv)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute MSE, MAE and R^2 of a forecast.

    Args:
        y_true: Ground-truth future hypervolume, shape ``(n,)``.
        y_pred: Predicted future hypervolume, shape ``(n,)``.

    Returns:
        Dict with float ``mse``/``mae`` and ``r2`` (``None`` when the
        ground truth has zero variance and R^2 is undefined).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2: float | None = None
    if ss_tot > 0.0:
        r2 = float(1.0 - ss_res / ss_tot)
    return {
        "mse": float(np.mean(errors**2)),
        "mae": float(np.mean(np.abs(errors))),
        "r2": r2,
    }


def _wilcoxon_less(
    model_se: np.ndarray, baseline_se: np.ndarray
) -> float | None:
    """One-sided paired Wilcoxon test: are the model's errors smaller?

    Tests the per-sample squared errors with ``zero_method="zsplit"`` and
    ``alternative="less"`` (the median of ``model_se - baseline_se`` is
    below zero, i.e. the model beats the baseline).

    Args:
        model_se: Per-sample squared errors of the model, shape ``(n,)``.
        baseline_se: Per-sample squared errors of the baseline, ``(n,)``.

    Returns:
        The p-value, or ``None`` when the test is degenerate (no samples
        or all paired differences exactly zero).
    """
    model_se = np.asarray(model_se, dtype=float)
    baseline_se = np.asarray(baseline_se, dtype=float)
    if model_se.size == 0:
        return None
    try:
        result = wilcoxon(
            model_se,
            baseline_se,
            zero_method="zsplit",
            alternative="less",
        )
    except ValueError:
        # Degenerate input (e.g. all differences exactly zero).
        return None
    return float(result.pvalue)


def _baseline_predictions(
    trajectories: list[list[dict[str, Any]]],
    traj_ids: np.ndarray,
    sample_indices: np.ndarray,
    horizon: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Persistence and linear-trend forecasts of ``hv[t + horizon]``.

    For sample ``i`` at transition index ``t`` of trajectory ``j``
    (``traj_ids[i] == j``, ``sample_indices[i] == t``):

    * persistence: ``hv[t]`` (``transitions[t]["state"]["hv"]``);
    * linear trend: ``hv[t] + horizon * mean(delta_hv)`` over the sample's
      history window ``transitions[max(0, t - window):t]`` (the same window
      the encoder saw; empty for ``t == 0``, where the trend is 0).

    Args:
        trajectories: Generation-sorted evaluation trajectories.
        traj_ids: Source trajectory index per sample, shape ``(n,)``.
        sample_indices: Source transition index ``t`` per sample, ``(n,)``.
        horizon: Forecast horizon ``h`` in generations.
        window: History window length used to build the samples.

    Returns:
        Tuple ``(persistence_pred, linear_pred)``, each shape ``(n,)``.
    """
    persistence = np.zeros(len(traj_ids), dtype=float)
    linear = np.zeros(len(traj_ids), dtype=float)
    for i, (traj_id, t_raw) in enumerate(zip(traj_ids, sample_indices)):
        trajectory = trajectories[int(traj_id)]
        t = int(t_raw)
        hv_t = float(trajectory[t]["state"]["hv"])
        history = trajectory[max(0, t - int(window)) : t]
        if history:
            mean_delta = float(
                np.mean(
                    [float(entry["reward"]["delta_hv"]) for entry in history]
                )
            )
        else:
            mean_delta = 0.0
        persistence[i] = hv_t
        linear[i] = hv_t + float(horizon) * mean_delta
    return persistence, linear


def _success_criteria(
    per_horizon: dict[str, dict[str, Any]], horizons: Sequence[int]
) -> dict[str, bool]:
    """Evaluate the Experiment-A gate from the per-horizon metrics.

    Args:
        per_horizon: Mapping ``str(h) -> metrics`` as built in
            :func:`run_evaluation`.
        horizons: Trained horizons (ascending).

    Returns:
        Dict with the three boolean success criteria.
    """
    beats_persistence = all(
        per_horizon[str(h)]["model"]["mse"]
        < per_horizon[str(h)]["persistence"]["mse"]
        for h in horizons
    )
    short = [
        per_horizon[str(h)]["model"]["r2"]
        for h in (1, 5)
        if str(h) in per_horizon
    ]
    r2_short = bool(short) and all(
        r2 is not None and r2 > 0.5 for r2 in short
    )
    long_h = 20 if 20 in horizons else max(horizons)
    long_metrics = per_horizon[str(long_h)]
    long_beats = (
        long_metrics["model"]["mse"] < long_metrics["persistence"]["mse"]
        and long_metrics["model"]["mse"] < long_metrics["linear"]["mse"]
    )
    return {
        "model_beats_persistence_all_horizons": bool(beats_persistence),
        "r2_above_0.5_h1_h5": bool(r2_short),
        "long_horizon_h20_beats_baselines": bool(long_beats),
    }


def _print_summary(
    per_horizon: dict[str, dict[str, Any]], horizons: Sequence[int]
) -> None:
    """Print a compact per-horizon metrics table to stdout."""

    def _fmt(value: Any, digits: int = 6) -> str:
        return "n/a" if value is None else f"{value:.{digits}g}"

    header = (
        f"{'h':>4} | {'model_mse':>10} | {'pers_mse':>10} | {'lin_mse':>10} "
        f"| {'model_r2':>8} | {'p_vs_pers':>9} | {'p_vs_lin':>9}"
    )
    print(header)
    print("-" * len(header))
    for h in horizons:
        entry = per_horizon[str(h)]
        print(
            f"{h:>4} | {_fmt(entry['model']['mse']):>10} "
            f"| {_fmt(entry['persistence']['mse']):>10} "
            f"| {_fmt(entry['linear']['mse']):>10} "
            f"| {_fmt(entry['model']['r2']):>8} "
            f"| {_fmt(entry['wilcoxon_p_vs_persistence'], 3):>9} "
            f"| {_fmt(entry['wilcoxon_p_vs_linear'], 3):>9}"
        )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the trained outcome predictor and write ``evaluation.json``.

    Args:
        args: Parsed arguments from :func:`parse_args`.

    Returns:
        The evaluation payload as written to ``--out-file``.

    Raises:
        FileNotFoundError: If model artifacts or evaluation directories are
            missing.
        ValueError: If no evaluation trajectories or samples were found.
    """
    model_dir = Path(args.model_dir)
    meta_path = model_dir / "training_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"training metadata not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    window = int(meta["window"])
    horizons = [int(h) for h in meta["horizons"]]

    predictor = OutcomePredictor.load(model_dir / "predictor.pt")
    encoder = StateEncoder.load(model_dir / "encoder.json")

    eval_dirs = (
        [str(d) for d in args.eval_dirs]
        if args.eval_dirs
        else [str(d) for d in meta["config"]["train_dirs"]]
    )
    trajectories: list[list[dict[str, Any]]] = []
    for directory in eval_dirs:
        directory_path = Path(directory)
        if not directory_path.is_dir():
            raise FileNotFoundError(
                f"evaluation directory not found: {directory_path}"
            )
        trajectories.extend(load_trajectories(directory_path))
    if not trajectories:
        raise ValueError(f"no evaluation trajectories found in {eval_dirs}")
    print(f"[load] {len(trajectories)} evaluation trajectories from {eval_dirs}")

    X, y, traj_ids, sample_indices = build_outcome_samples(
        trajectories, encoder, window, horizons=horizons
    )
    if X.shape[0] == 0:
        raise ValueError(
            "build_outcome_samples produced zero samples on the evaluation "
            "trajectories"
        )
    predictions = np.asarray(predictor.predict(X), dtype=float)
    print(f"[eval] n={X.shape[0]} samples, horizons={horizons}")

    per_horizon: dict[str, dict[str, Any]] = {}
    for k, h in enumerate(horizons):
        y_true = y[:, k]
        y_model = predictions[:, k]
        persistence, linear = _baseline_predictions(
            trajectories, traj_ids, sample_indices, h, window
        )
        model_se = (y_model - y_true) ** 2
        per_horizon[str(h)] = {
            "model": _regression_metrics(y_true, y_model),
            "persistence": _regression_metrics(y_true, persistence),
            "linear": _regression_metrics(y_true, linear),
            "wilcoxon_p_vs_persistence": _wilcoxon_less(
                model_se, (persistence - y_true) ** 2
            ),
            "wilcoxon_p_vs_linear": _wilcoxon_less(
                model_se, (linear - y_true) ** 2
            ),
        }

    payload: dict[str, Any] = {
        "per_horizon": per_horizon,
        "success_criteria": _success_criteria(per_horizon, horizons),
        "config": {
            "model_dir": str(model_dir),
            "eval_dirs": eval_dirs,
            "window": window,
            "horizons": horizons,
            "n_eval_trajectories": len(trajectories),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "n_samples": int(X.shape[0]),
    }

    _print_summary(per_horizon, horizons)
    print(f"[gate] {json.dumps(payload['success_criteria'])}")

    out_file = (
        Path(args.out_file) if args.out_file else model_dir / "evaluation.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[done] wrote evaluation -> {out_file}")
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point: parse arguments and evaluate the predictor.

    Args:
        argv: Optional argument list; ``None`` reads ``sys.argv``.

    Returns:
        The evaluation payload as written to the output JSON.
    """
    return run_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
