from __future__ import annotations

"""Phase 2B: does the OutcomePredictor actually use the action features?

Experiment A trained :class:`controller.outcome_predictor.OutcomePredictor`
to regress absolute future hypervolume from ``(encoded state history,
candidate action)`` and reported held-out R^2 of 0.979-0.999. A high
*global* R^2 is not evidence that the action matters: future HV is largely
determined by the state the run is already in, so a model that ignored the
four action features entirely could still score highly. This script runs
five action-conditional diagnostics on a trained predictor and writes
``results/phase2b/action_conditioning.json``:

* **D1 action ablation** — re-predict with the four action features
  replaced by (a) zeros, (b) their per-column means, (c) a cross-sample
  permutation, and compare R^2/MSE against the untouched baseline. A
  near-constant R^2 means the model reads state only.
* **D2 within-state action ranking** — rank several *different* actions
  evaluated at the *same* state and correlate the predictor's ordering
  with the realized ordering (Spearman/Kendall). This is the diagnostic
  that separates "predicts the future" from "predicts the action's
  effect".
* **D3 top-1 regret** — realized outcome of the action the planner would
  pick minus the realized outcome of the oracle best action at that state.
* **D4 action-effect SNR** — between-action variance of the realized
  outcome divided by the within-action replicate noise variance (three
  branch seeds per candidate).
* **D5 predicted-vs-actual regression** — Pearson r, slope and intercept
  of realized outcome on predicted outcome, pooled over all states.

Two data sources back D2/D3:

* *approximate* (recorded trajectories only, cheap): samples are grouped
  by ``(problem, generation)`` and ranked by the planner's weighted score
  against the realized weighted outcome. Limitation: the state differs
  inside such a group, so a positive correlation can be produced by state
  quality rather than by action ranking; the group analysis is therefore
  reported as the shallow checksum, not as the causal test.
* *strict* (harvested snapshots, expensive): one snapshot state, K
  candidate actions sampled exactly like the counterfactual evaluator
  samples its alternatives, and a real NSGA-II branch of ``h`` generations
  per candidate and replicate. Only this setup varies the action while
  holding the state fixed, so it is the primary evidence; D3/D4/D5 come
  from it.

The realized quantity is the hypervolume *gain* over the branch point
``hv(t + h) - hv(t)``, and the predictor's quantity is its weighted score
over the trained horizons shifted by the same branch point
(``sum(w_h * pred_h) - hv(t)``); within one state the shift is constant,
so this does not change any within-state ranking.

Cost of the strict part (measured on the Phase-2B host, ~0.39 s per
NSGA-II generation at pop 100): ``states x candidates x horizon x reps``
generations, i.e. 20 x 8 x 5 x 3 = 2400 generations ~ 16 min. The
approximate part is a handful of forward passes (seconds).

Example:
    ``python experiments/diagnose_action_conditioning.py``
    ``python experiments/diagnose_action_conditioning.py --no-strict``
"""

import argparse
import json
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats

if __package__ in (None, ""):
    # Allow ``python experiments/diagnose_action_conditioning.py`` from the
    # repo root: the script directory (not the repo root) is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Default artifact directory written by the Experiment-A training script.
DEFAULT_MODEL_DIR = "results/phase2_outcome"
#: Default (held-out) trajectory directory used for D1/D2-approximate.
DEFAULT_EVAL_DIR = "results/trajectory_full"
#: Default Phase-1.75 snapshot directory used by the strict D2/D3/D4/D5.
DEFAULT_SNAPSHOTS_DIR = "results/phase1_75/snapshots"
#: Default output artifact (Phase-2B isolated directory).
DEFAULT_OUT_FILE = "results/phase2b/action_conditioning.json"
#: Number of trailing columns of ``X`` that carry the action features:
#: ``[mutation_multiplier, exploration_strength, onehot_polynomial,
#: onehot_gaussian]``.
N_ACTION_FEATURES = 4
#: Planner horizon weights used to turn predicted HV vectors into the
#: scalar the deployment rule ranks by (``PlanningController`` defaults).
DEFAULT_HORIZON_WEIGHTS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)
#: Problem grid of the strict version.
DEFAULT_PROBLEMS: tuple[str, ...] = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")
#: Strict-version defaults (20 states = 4 per problem).
DEFAULT_STRICT_STATES_PER_PROBLEM = 4
DEFAULT_STRICT_CANDIDATES = 8
DEFAULT_STRICT_HORIZON = 5
DEFAULT_STRICT_REPS = 3
#: Deployment pm multiplier bounds (Phase-1.5 full-action contract).
DEFAULT_PM_MULT_RANGE: tuple[float, float] = (0.25, 8.0)
#: Default hypervolume reference point of the harvest/evaluation protocol.
DEFAULT_REF_POINT: tuple[float, float] = (1.1, 1.1)
#: Number of points sampled from the true front (harvest protocol value).
DEFAULT_N_REFERENCE_POINTS = 200
#: Fixed seed of every stochastic step of this diagnostic (permutation,
#: bootstrap-free resampling), so a rerun with the same inputs reproduces
#: the same numbers.
DEFAULT_SEED = 0
#: Minimum group size for a within-group rank correlation.
MIN_GROUP_SIZE = 3
#: Significance level of the per-state rank correlations.
ALPHA = 0.05
#: Declared conclusion thresholds (fixed before looking at the results).
#: (a) ablation: R^2 drop of at least this much means the action features
#: carry information the model uses; (b) strict ranking: mean Spearman at
#: least this high on at least this fraction of states.
ABLATION_R2_DROP_THRESHOLD = 0.05
STRICT_SPEARMAN_THRESHOLD = 0.3
STRICT_SIGNIFICANT_FRACTION_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# pure metric helpers (synthetic-data testable)
# ---------------------------------------------------------------------------


def _as_1d(values: Any) -> np.ndarray:
    """Flatten ``values`` into a float64 1-D array."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float | None]:
    """MSE and R^2 of a prediction, mirroring ``evaluate_outcome_predictor``.

    Args:
        y_true: Ground truth, flattened.
        y_pred: Prediction, flattened (same size).

    Returns:
        ``{"r2": float | None, "mse": float}``; ``r2`` is ``None`` when the
        ground truth has zero variance (undefined).

    Raises:
        ValueError: If the two inputs have different sizes.
    """
    truth = _as_1d(y_true)
    pred = _as_1d(y_pred)
    if truth.size != pred.size:
        raise ValueError(
            f"y_true has {truth.size} entries but y_pred has {pred.size}"
        )
    errors = pred - truth
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2: float | None = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None
    return {"r2": r2, "mse": float(np.mean(errors**2))}


def ablate_action_block(
    X: np.ndarray, *, seed: int = DEFAULT_SEED
) -> dict[str, np.ndarray]:
    """Copies of ``X`` with the action block ablated in three ways.

    Args:
        X: Feature matrix of shape ``(n, d)`` whose last
            :data:`N_ACTION_FEATURES` columns are the action features.
        seed: Seed of the cross-sample permutation.

    Returns:
        ``{"baseline", "zero", "mean", "shuffled"}`` matrices of the same
        shape: the input unchanged, the input with the action block zeroed,
        with the action block replaced by its per-column mean, and with the
        rows of the action block permuted across samples (the joint action
        distribution is preserved, only its pairing with the state is
        destroyed).

    Raises:
        ValueError: If ``X`` has fewer than :data:`N_ACTION_FEATURES` columns.
    """
    features = np.asarray(X, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] < N_ACTION_FEATURES:
        raise ValueError(
            f"X must be 2-D with at least {N_ACTION_FEATURES} columns, "
            f"got shape {features.shape}"
        )
    block = features[:, -N_ACTION_FEATURES:]
    zeroed = features.copy()
    zeroed[:, -N_ACTION_FEATURES:] = 0.0
    meaned = features.copy()
    meaned[:, -N_ACTION_FEATURES:] = block.mean(axis=0, keepdims=True)
    shuffled = features.copy()
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    permutation = rng.permutation(features.shape[0])
    shuffled[:, -N_ACTION_FEATURES:] = block[permutation]
    return {
        "baseline": features.copy(),
        "zero": zeroed,
        "mean": meaned,
        "shuffled": shuffled,
    }


def d1_action_ablation(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, float | None]]:
    """D1: R^2/MSE of the model with the action features ablated.

    Args:
        predict_fn: Callable mapping a feature matrix to predictions with
            the same shape as ``y``.
        X: Feature matrix (state block + action block).
        y: Ground-truth targets.
        seed: Permutation seed of the shuffled variant.

    Returns:
        ``{"zero": {...}, "mean": {...}, "shuffled": {...}, "baseline":
        {...}}``, each ``{"r2", "mse"}`` against the same ``y``.
    """
    variants = ablate_action_block(X, seed=seed)
    return {
        name: regression_metrics(y, predict_fn(variants[name]))
        for name in ("zero", "mean", "shuffled", "baseline")
    }


def weighted_score(matrix: Any, weights: Sequence[float]) -> np.ndarray:
    """Weighted sum of the columns of ``matrix`` (one score per row).

    Raises:
        ValueError: If the row width does not match ``len(weights)``.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {values.shape}")
    weight_array = np.asarray(list(weights), dtype=np.float64)
    if values.shape[1] != weight_array.size:
        raise ValueError(
            f"matrix has {values.shape[1]} columns but {weight_array.size} "
            f"weights were given"
        )
    return values @ weight_array


def _scipy_stat(result: Any) -> float | None:
    """Finite ``statistic``/``correlation`` of a scipy result, else ``None``."""
    value = getattr(result, "statistic", None)
    if value is None:
        value = getattr(result, "correlation", None)
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def ranking_correlations(
    predicted: Any, realized: Any
) -> tuple[float | None, float | None, float | None]:
    """Spearman/Kendall correlation of two per-candidate score vectors.

    Args:
        predicted: Predicted scores, one per candidate.
        realized: Realized scores, one per candidate.

    Returns:
        ``(spearman, kendall, spearman_p_value)``; all ``None`` when the
        comparison is undefined (fewer than :data:`MIN_GROUP_SIZE`
        candidates, size mismatch, or a constant input).
    """
    x = _as_1d(predicted)
    y = _as_1d(realized)
    if x.size < MIN_GROUP_SIZE or x.size != y.size:
        return None, None, None
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None, None, None
    spearman = stats.spearmanr(x, y)
    kendall = stats.kendalltau(x, y)
    p_value = getattr(spearman, "pvalue", None)
    return (
        _scipy_stat(spearman),
        _scipy_stat(kendall),
        None if p_value is None else float(p_value),
    )


def top1_regret(predicted: Any, realized: Any) -> float:
    """D3 core: realized(best) - realized(argmax prediction).

    Ties in ``predicted`` resolve to the earliest candidate, matching
    ``np.argmax`` (the planner's own tie rule).

    Raises:
        ValueError: If the two inputs are empty or of different sizes.
    """
    x = _as_1d(predicted)
    y = _as_1d(realized)
    if x.size == 0 or x.size != y.size:
        raise ValueError(
            f"predicted/realized must be non-empty and equally sized, got "
            f"{x.size} and {y.size}"
        )
    chosen = int(np.argmax(x))
    best = int(np.argmax(y))
    return float(y[best] - y[chosen])


def action_effect_snr(
    realized_matrices: Sequence[np.ndarray],
) -> tuple[float | None, float | None, float | None]:
    """D4 core: between-action variance vs within-action replicate variance.

    For each state (one ``(candidates, reps)`` matrix) the between-action
    variance is the sample variance (``ddof=1``) of the per-candidate mean
    outcome, and the within-action noise variance is the mean over
    candidates of the sample variance across replicates. Both are averaged
    over states; the SNR is their ratio.

    Args:
        realized_matrices: One ``(n_candidates, n_reps)`` matrix per state;
            each needs at least two candidates and two replicates.

    Returns:
        ``(between_action_var, within_action_noise_var, snr)``; all
        ``None`` when no state has enough candidates/replicates.
    """
    between: list[float] = []
    within: list[float] = []
    for matrix in realized_matrices:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
            continue
        between.append(float(np.var(values.mean(axis=1), ddof=1)))
        within.append(float(np.mean(np.var(values, axis=1, ddof=1))))
    if not between:
        return None, None, None
    between_mean = float(np.mean(between))
    within_mean = float(np.mean(within))
    snr = float(between_mean / within_mean) if within_mean > 0.0 else None
    return between_mean, within_mean, snr


def linear_fit(
    predicted: Any, realized: Any
) -> tuple[float | None, float | None, float | None]:
    """D5 core: Pearson r, slope and intercept of realized ~ predicted.

    Returns:
        ``(pearson_r, slope, intercept)``; all ``None`` when either input is
        constant or has fewer than two entries.
    """
    x = _as_1d(predicted)
    y = _as_1d(realized)
    if x.size < 2 or x.size != y.size:
        return None, None, None
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None, None, None
    pearson = _scipy_stat(stats.pearsonr(x, y))
    slope, intercept = np.polyfit(x, y, 1)
    return pearson, float(slope), float(intercept)


def _summarize_rank_series(
    correlations: Sequence[tuple[float | None, float | None, float | None]]
) -> dict[str, Any]:
    """Aggregate per-state ``(spearman, kendall, p)`` triples."""
    spearman = [rho for rho, _, _ in correlations if rho is not None]
    kendall = [tau for _, tau, _ in correlations if tau is not None]
    p_values = [p for _, _, p in correlations if p is not None]
    significant = [p for p in p_values if p < ALPHA]
    return {
        "n_states": len(spearman),
        "spearman_mean": float(np.mean(spearman)) if spearman else None,
        "spearman_std": (
            float(np.std(spearman, ddof=1)) if len(spearman) > 1 else 0.0
            if spearman
            else None
        ),
        "kendall_mean": float(np.mean(kendall)) if kendall else None,
        "significant_fraction": (
            float(len(significant) / len(p_values)) if p_values else None
        ),
    }


def within_group_ranking(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    groups: dict[str, list[int]],
    *,
    weights: Sequence[float] = DEFAULT_HORIZON_WEIGHTS,
) -> dict[str, Any]:
    """D2 approximate: rank samples inside each ``(problem, generation)`` group.

    Both sides use the planner's scalar objective: the prediction is the
    weighted sum of the predicted HV vector, the realized side the weighted
    sum of the realized future-HV vector. Limitation (reported in the
    artifact): the state differs across the samples of a group, so a
    positive correlation does not by itself prove action sensitivity --
    the action-shuffled control computed by the caller separates the two.

    Args:
        predict_fn: Callable mapping a feature matrix to predictions.
        X: Feature matrix of all samples.
        y: Realized future-HV matrix ``(n, n_horizons)``.
        groups: ``{group label: sample indices}``; groups with fewer than
            :data:`MIN_GROUP_SIZE` members are skipped.
        weights: Horizon weights of the scalar objective.

    Returns:
        ``{"n_groups", "spearman_mean", "spearman_std", "kendall_mean",
        "significant_fraction"}``.
    """
    predictions = weighted_score(predict_fn(X), weights)
    realized = weighted_score(y, weights)
    correlations: list[tuple[float | None, float | None, float | None]] = []
    for indices in groups.values():
        if len(indices) < MIN_GROUP_SIZE:
            continue
        correlations.append(
            ranking_correlations(
                predictions[np.asarray(indices, dtype=int)],
                realized[np.asarray(indices, dtype=int)],
            )
        )
    summary = _summarize_rank_series(correlations)
    return {
        "n_groups": summary["n_states"],
        "spearman_mean": summary["spearman_mean"],
        "spearman_std": summary["spearman_std"],
        "kendall_mean": summary["kendall_mean"],
        "significant_fraction": summary["significant_fraction"],
    }


def strict_statistics(
    states: Sequence[dict[str, Any]],
    *,
    horizon_column: int,
    weights: Sequence[float] = DEFAULT_HORIZON_WEIGHTS,
) -> dict[str, dict[str, Any]]:
    """D2-strict/D3/D4/D5 from per-state strict records.

    Args:
        states: Records with ``scores`` (planner-weighted predicted HV per
            candidate), ``predicted`` (``(n_candidates, n_horizons)``),
            ``realized`` (``(n_candidates, n_reps)`` HV gains) and
            ``hv_before``.
        horizon_column: Column of ``predicted`` aligned with the branch
            horizon used to realize ``realized``.
        weights: Horizon weights of the planner's scalar score.

    Returns:
        ``{"d2_strict": {...}, "d3": {...}, "d4": {...}, "d5": {...}}``.
    """
    weighted_correlations: list[tuple[float | None, float | None, float | None]] = []
    column_correlations: list[tuple[float | None, float | None, float | None]] = []
    regrets: list[float] = []
    agreements: list[float] = []
    predicted_pool: list[float] = []
    realized_pool: list[float] = []
    realized_matrices: list[np.ndarray] = []
    n_degenerate = 0
    for state in states:
        realized = np.asarray(state["realized"], dtype=np.float64)
        mean_realized = realized.mean(axis=1)
        # A state where every candidate produced exactly the same realized
        # outcome (typically an early generation whose branch cannot gain
        # hypervolume yet) carries no action signal: including it would only
        # dilute the ranking and regret averages with structural zeros.
        if float(mean_realized.max() - mean_realized.min()) <= 1e-12:
            n_degenerate += 1
            continue
        scores = _as_1d(state["scores"])
        weighted_correlations.append(ranking_correlations(scores, mean_realized))
        column_correlations.append(
            ranking_correlations(
                np.asarray(state["predicted"], dtype=np.float64)[:, horizon_column],
                mean_realized,
            )
        )
        regrets.append(top1_regret(scores, mean_realized))
        agreements.append(
            1.0 if int(np.argmax(scores)) == int(np.argmax(mean_realized)) else 0.0
        )
        predicted_pool.extend((scores - float(state["hv_before"])).tolist())
        realized_pool.extend(mean_realized.tolist())
        realized_matrices.append(realized)

    weighted = _summarize_rank_series(weighted_correlations)
    column = _summarize_rank_series(column_correlations)
    for summary in (weighted, column):
        summary.pop("n_states", None)
    regret_array = np.asarray(regrets, dtype=np.float64)
    between, within, snr = action_effect_snr(realized_matrices)
    pearson, slope, intercept = linear_fit(predicted_pool, realized_pool)
    return {
        "d2_strict": {
            "n_states": len(states),
            "n_states_scored": len(regrets),
            "n_states_excluded_degenerate": int(n_degenerate),
            "spearman_mean": weighted["spearman_mean"],
            "spearman_std": weighted["spearman_std"],
            "kendall_mean": weighted["kendall_mean"],
            "significant_fraction": weighted["significant_fraction"],
            "spearman_mean_horizon_column": column["spearman_mean"],
            "kendall_mean_horizon_column": column["kendall_mean"],
            "top1_selection_agreement": (
                float(np.mean(agreements)) if agreements else None
            ),
        },
        "d3": {
            "mean": float(regret_array.mean()) if regret_array.size else None,
            "median": float(np.median(regret_array)) if regret_array.size else None,
            "std": (
                float(regret_array.std(ddof=1)) if regret_array.size > 1 else 0.0
                if regret_array.size
                else None
            ),
            "n_states": int(regret_array.size),
        },
        "d4": {
            "between_action_var": between,
            "within_action_noise_var": within,
            "snr": snr,
        },
        "d5": {"pearson_r": pearson, "slope": slope, "intercept": intercept},
    }


def conclude(
    d1: dict[str, dict[str, float | None]],
    d2_strict: dict[str, Any] | None,
) -> dict[str, Any]:
    """Declared decision rule over D1 and the strict D2.

    The rule is fixed before running the experiment:

    * ``max_ablation_r2_drop`` is ``baseline R^2 - max(ablated R^2)``; a drop
      of at least :data:`ABLATION_R2_DROP_THRESHOLD` means the model reads
      the action features.
    * the strict within-state ranking counts as action sensitivity when its
      mean Spearman is at least :data:`STRICT_SPEARMAN_THRESHOLD` and the
      significant fraction is at least
      :data:`STRICT_SIGNIFICANT_FRACTION_THRESHOLD`.

    Args:
        d1: Output of :func:`d1_action_ablation`.
        d2_strict: Strict D2 block, or ``None`` when it was not run.

    Returns:
        ``{"action_information_used": bool, "evidence": str}``.
    """
    baseline_r2 = d1["baseline"]["r2"]
    ablated_r2 = [
        d1[name]["r2"]
        for name in ("zero", "mean", "shuffled")
        if d1[name]["r2"] is not None
    ]
    max_drop = (
        float(baseline_r2 - max(ablated_r2))
        if baseline_r2 is not None and ablated_r2
        else None
    )
    ablation_positive = max_drop is not None and max_drop >= ABLATION_R2_DROP_THRESHOLD
    spearman = None if d2_strict is None else d2_strict.get("spearman_mean")
    significant = (
        None if d2_strict is None else d2_strict.get("significant_fraction")
    )
    ranking_positive = bool(
        spearman is not None
        and significant is not None
        and spearman >= STRICT_SPEARMAN_THRESHOLD
        and significant >= STRICT_SIGNIFICANT_FRACTION_THRESHOLD
    )
    evidence = (
        f"D1 baseline R2={baseline_r2 if baseline_r2 is None else round(baseline_r2, 4)}, "
        f"max ablation drop={None if max_drop is None else round(max_drop, 4)} "
        f"(threshold {ABLATION_R2_DROP_THRESHOLD}); "
        f"D2 strict within-state Spearman mean="
        f"{None if spearman is None else round(spearman, 4)}, significant fraction="
        f"{None if significant is None else round(significant, 4)} "
        f"(thresholds {STRICT_SPEARMAN_THRESHOLD}/{STRICT_SIGNIFICANT_FRACTION_THRESHOLD}); "
        f"high global R2 is therefore "
        f"{'backed by' if (ablation_positive or ranking_positive) else 'not backed by'} "
        f"action-conditional evidence."
    )
    return {
        "action_information_used": bool(ablation_positive or ranking_positive),
        "evidence": evidence,
    }


def build_payload(
    *,
    d1: dict[str, Any],
    d2: dict[str, Any],
    d3: dict[str, Any],
    d4: dict[str, Any],
    d5: dict[str, Any],
    conclusion: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the artifact with exactly the documented top-level keys."""
    return {
        "d1_action_ablation": d1,
        "d2_within_state_ranking": d2,
        "d3_top1_regret": d3,
        "d4_action_effect_snr": d4,
        "d5_pred_vs_actual": d5,
        "conclusion": conclusion,
        "config": config,
    }


# ---------------------------------------------------------------------------
# data plumbing
# ---------------------------------------------------------------------------


def load_eval_samples(
    eval_dir: str | Path,
    encoder: Any,
    window: int,
    horizons: Sequence[int],
    *,
    problems: Sequence[str] | None = None,
    normalize_multiplier: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], int]:
    """Build ``(X, y)`` samples plus their ``(problem, generation)`` groups.

    Loads the recorded trajectories with their run configs, optionally
    restores the normalized ``mutation_multiplier`` feature
    (``mutation_probability * n_vars``) for datasets that store it only
    implicitly, and calls
    :func:`controller.dataset.build_outcome_samples`.

    Args:
        eval_dir: Directory with trajectory JSONs.
        encoder: Fitted :class:`controller.state_encoder.StateEncoder`.
        window: History window (must match the encoder/predictor).
        horizons: Predicted generation offsets.
        problems: Optional problem whitelist.
        normalize_multiplier: When True, fill in a missing
            ``mutation_multiplier`` action key with ``pm * n_vars`` of the
            source run. Without it,
            :func:`controller.dataset.build_outcome_samples` falls back to
            ``OUTCOME_FALLBACK_N_VARS = 30`` for every problem, which
            mis-scales the feature on ZDT4 (``n_vars = 10``) relative to
            the training corpus.

    Returns:
        ``(X, y, groups, n_trajectories)`` where ``groups`` maps
        ``"{problem}|{generation}"`` to the sample indices of that group.

    Raises:
        NotADirectoryError: If ``eval_dir`` is not a directory.
        ValueError: If no trajectory or no sample was produced.
    """
    from controller.dataset import build_outcome_samples, load_trajectory_records

    eval_path = Path(eval_dir)
    if not eval_path.is_dir():
        raise NotADirectoryError(f"not a directory: {eval_path}")
    records = load_trajectory_records(eval_path)
    if problems:
        wanted = {str(name) for name in problems}
        records = [record for record in records if record["problem"] in wanted]
    if not records:
        raise ValueError(f"no trajectories for problems {problems} in {eval_path}")

    trajectories: list[list[dict[str, Any]]] = []
    labels: list[str] = []
    for record in records:
        n_vars = int(record["n_vars"])
        transitions = record["transitions"]
        if normalize_multiplier:
            for transition in transitions:
                action = transition["action"]
                if "mutation_multiplier" not in action:
                    action["mutation_multiplier"] = (
                        float(action["mutation_probability"]) * n_vars
                    )
        trajectories.append(transitions)
        labels.append(str(record["problem"]))

    X, y, traj_ids, sample_indices = build_outcome_samples(
        trajectories, encoder, int(window), horizons=list(horizons)
    )
    if X.shape[0] == 0:
        raise ValueError(f"no outcome samples built from {eval_path}")
    problem_labels = [labels[int(traj_id)] for traj_id in traj_ids]
    groups = group_samples_by_generation(problem_labels, sample_indices)
    return X, y, groups, len(trajectories)


def group_samples_by_generation(
    problem_labels: Sequence[str], sample_indices: np.ndarray
) -> dict[str, list[int]]:
    """Group sample indices by ``(problem, generation)`` label."""
    groups: dict[str, list[int]] = {}
    for index, (problem, t_raw) in enumerate(zip(problem_labels, sample_indices)):
        key = f"{problem}|{int(t_raw)}"
        groups.setdefault(key, []).append(int(index))
    return groups


def select_strict_snapshots(
    snapshots_dir: str | Path,
    problems: Sequence[str],
    states_per_problem: int,
) -> list[Path]:
    """Evenly spread ``states_per_problem`` snapshots over each problem.

    Raises:
        FileNotFoundError: If no snapshot matches any requested problem.
    """
    directory = Path(snapshots_dir)
    selected: list[Path] = []
    for problem in problems:
        files = sorted(
            directory.glob(f"{problem}__seed*__gen*.pkl"),
            key=lambda path: (
                int(path.stem.split("__")[1].removeprefix("seed")),
                int(path.stem.split("__")[2].removeprefix("gen")),
                path.name,
            ),
        )
        if not files:
            continue
        if len(files) > int(states_per_problem):
            positions = (
                np.linspace(0, len(files) - 1, int(states_per_problem))
                .round()
                .astype(int)
            )
            files = [files[i] for i in dict.fromkeys(int(i) for i in positions)]
        selected.extend(files)
    if not selected:
        raise FileNotFoundError(
            f"no snapshots for {list(problems)} in {directory}"
        )
    return selected


def _snapshot_hash_seed(problem: str, seed: int, generation: int) -> int:
    """Snapshot identity hash, identical to the counterfactual evaluator."""
    return zlib.crc32(f"{problem}|{seed}|{generation}".encode("utf-8"))


def _branch_seed(snapshot_hash_seed: int, candidate_index: int) -> int:
    """Per-candidate branch seed, identical to the counterfactual evaluator."""
    return zlib.crc32(f"{snapshot_hash_seed}|{candidate_index}".encode("utf-8"))


def candidate_feature_rows(
    state_block: np.ndarray,
    n_vars: int,
    hash_seed: int,
    n_candidates: int,
    pm_mult_range: tuple[float, float],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Feature rows for ``n_candidates`` sampled actions at one state.

    Candidate ``k`` (``1..n_candidates``) is drawn with
    ``Generator(PCG64([hash_seed, k]))`` through
    :func:`experiments.generate_dataset.sample_full_action`, the sampling
    scheme of the counterfactual evaluator's alternatives; the feature
    layout is the :func:`controller.dataset.build_outcome_samples` layout
    (state block + ``[multiplier, exploration, onehot_polynomial,
    onehot_gaussian]``).

    Returns:
        ``(rows, actions)`` with ``rows`` of shape
        ``(n_candidates, state_block.size + 4)``.
    """
    from controller.dataset import OPERATOR_TO_INDEX
    from experiments.generate_dataset import sample_full_action

    rows: list[np.ndarray] = []
    actions: list[dict[str, Any]] = []
    for k in range(1, int(n_candidates) + 1):
        rng = np.random.Generator(np.random.PCG64([int(hash_seed), k]))
        operator, pm, exploration = sample_full_action(
            rng, 1.0 / int(n_vars), pm_mult_range
        )
        one_hot = [0.0, 0.0]
        one_hot[OPERATOR_TO_INDEX[str(operator)]] = 1.0
        multiplier = float(pm) * int(n_vars)
        rows.append(
            np.concatenate(
                [
                    np.asarray(state_block, dtype=np.float64),
                    np.asarray([multiplier, float(exploration), *one_hot]),
                ]
            )
        )
        actions.append(
            {
                "mutation_operator": str(operator),
                "mutation_probability": float(pm),
                "exploration_strength": float(exploration),
                "mutation_multiplier": multiplier,
            }
        )
    return np.vstack(rows), actions


def run_strict_states(
    snapshot_paths: Sequence[Path],
    encoder: Any,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    *,
    n_candidates: int,
    horizon: int,
    n_reps: int,
    pm_mult_range: tuple[float, float] = DEFAULT_PM_MULT_RANGE,
    ref_point: Sequence[float] = DEFAULT_REF_POINT,
    n_reference_points: int = DEFAULT_N_REFERENCE_POINTS,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Branch every candidate action at every snapshot state.

    For each snapshot: restore the state, evaluate ``n_candidates`` sampled
    actions with the predictor, then — for every candidate and every
    replicate — restore the snapshot, reseed the algorithm RNG with
    ``Generator(PCG64([branch_seed, rep]))`` and run ``horizon`` NSGA-II
    generations with that single action, recording the hypervolume gain
    over the branch point.

    Returns:
        One record per state with ``problem``, ``seed``, ``generation``,
        ``hv_before``, ``predicted`` (``(n_candidates, n_horizons)``),
        ``scores`` (planner-weighted predicted HV), ``actions`` and
        ``realized`` (``(n_candidates, n_reps)`` HV gains).
    """
    import pickle

    from algorithms.nsga2 import NSGAII, OperatorConfig
    from benchmarks import get_problem
    from metrics.indicators import hypervolume

    records: list[dict[str, Any]] = []
    for position, path in enumerate(snapshot_paths, start=1):
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        problem_name = str(payload["problem"])
        problem = get_problem(problem_name)
        n_vars = int(problem.n_vars)
        seed = int(payload["seed"])
        generation = int(payload["generation"])
        snapshot = payload["state"]
        algorithm = NSGAII(
            problem,
            pop_size=int(snapshot["config"]["pop_size"]),
            operators=OperatorConfig(),
            seed=0,
        )
        algorithm.restore_state(snapshot)
        hv_before = float(
            hypervolume(algorithm.nondominated_front(), np.asarray(ref_point, float))
        )
        state_block = np.asarray(
            encoder.transform(list(payload["history"])), dtype=np.float64
        )
        hash_seed = _snapshot_hash_seed(problem_name, seed, generation)
        rows, actions = candidate_feature_rows(
            state_block, n_vars, hash_seed, n_candidates, pm_mult_range
        )
        predicted = np.asarray(predict_fn(rows), dtype=np.float64)
        scores = weighted_score(predicted, DEFAULT_HORIZON_WEIGHTS)
        realized = np.zeros((len(actions), int(n_reps)), dtype=np.float64)
        for k, action in enumerate(actions):
            branch = _branch_seed(hash_seed, k)
            for rep in range(int(n_reps)):
                algorithm.restore_state(snapshot)
                algorithm.rng = np.random.Generator(np.random.PCG64([branch, rep]))
                for _ in range(int(horizon)):
                    algorithm.step(
                        mutation_prob=action["mutation_probability"],
                        mutation_operator=action["mutation_operator"],
                        exploration_strength=action["exploration_strength"],
                    )
                realized[k, rep] = float(
                    hypervolume(
                        algorithm.nondominated_front(),
                        np.asarray(ref_point, float),
                    )
                ) - hv_before
        records.append(
            {
                "problem": problem_name,
                "seed": seed,
                "generation": generation,
                "hv_before": hv_before,
                "predicted": predicted,
                "scores": scores,
                "actions": actions,
                "realized": realized,
            }
        )
        if verbose:
            mean_realized = realized.mean(axis=1)
            print(
                f"[strict] {position}/{len(snapshot_paths)} {problem_name} "
                f"seed={seed} gen={generation} hv_before={hv_before:.5f} "
                f"best_gain={float(mean_realized.max()):.5f} "
                f"planner_gain={float(mean_realized[int(np.argmax(scores))]):.5f}"
            )
    return records


def _strict_block_reason(exc: Exception) -> dict[str, Any]:
    """Fallback block recorded when the strict version cannot run."""
    return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments of the action-conditioning diagnostic."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2B: test whether the OutcomePredictor uses the action "
            "features (D1 ablation, D2 within-state ranking, D3 top-1 "
            "regret, D4 action-effect SNR, D5 predicted-vs-actual)."
        )
    )
    parser.add_argument(
        "--model-dir", type=str, default=DEFAULT_MODEL_DIR,
        help="Directory with predictor.pt/encoder.json/training_meta.json "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--eval-dir", type=str, default=DEFAULT_EVAL_DIR,
        help="Held-out trajectory directory for D1/D2-approximate "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--snapshots-dir", type=str, default=DEFAULT_SNAPSHOTS_DIR,
        help="Snapshot directory for the strict D2/D3/D4/D5 (default: %(default)s).",
    )
    parser.add_argument(
        "--out-file", type=str, default=DEFAULT_OUT_FILE,
        help="Output JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--problems", nargs="+", default=list(DEFAULT_PROBLEMS),
        help="Problems to include (default: %(default)s).",
    )
    parser.add_argument(
        "--no-strict", action="store_true",
        help="Skip the snapshot branching version (fast, diagnostics only).",
    )
    parser.add_argument(
        "--strict-states-per-problem", type=int,
        default=DEFAULT_STRICT_STATES_PER_PROBLEM,
        help="Snapshots per problem in the strict version (default: %(default)s).",
    )
    parser.add_argument(
        "--strict-candidates", type=int, default=DEFAULT_STRICT_CANDIDATES,
        help="Candidate actions per state in the strict version (default: %(default)s).",
    )
    parser.add_argument(
        "--strict-horizon", type=int, default=DEFAULT_STRICT_HORIZON,
        help="NSGA-II generations per branch in the strict version "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--strict-reps", type=int, default=DEFAULT_STRICT_REPS,
        help="Replicate branch seeds per candidate, used for the D4 noise "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--no-multiplier-normalization", action="store_true",
        help="Keep build_outcome_samples' n_vars=30 multiplier fallback "
        "instead of restoring pm * n_vars from each run config.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Seed of the action-feature permutation (default: %(default)s).",
    )
    return parser.parse_args(argv)


def run_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    """Run D1-D5 and write the artifact.

    Returns:
        The payload exactly as written to ``--out-file``.

    Raises:
        FileNotFoundError: If the model artifacts or data directories are
            missing.
        ValueError: If no evaluation samples can be built.
    """
    from controller.outcome_predictor import OutcomePredictor
    from controller.state_encoder import StateEncoder

    model_dir = Path(args.model_dir)
    meta_path = model_dir / "training_meta.json"
    for path in (meta_path, model_dir / "predictor.pt", model_dir / "encoder.json"):
        if not path.is_file():
            raise FileNotFoundError(f"missing model artifact: {path}")
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    window = int(meta["window"])
    horizons = [int(h) for h in meta["horizons"]]

    predictor = OutcomePredictor.load(model_dir / "predictor.pt")
    encoder = StateEncoder.load(model_dir / "encoder.json")
    predict_fn = predictor.predict

    X, y, groups, n_trajectories = load_eval_samples(
        args.eval_dir,
        encoder,
        window,
        horizons,
        problems=args.problems,
        normalize_multiplier=not args.no_multiplier_normalization,
    )
    print(f"[load] {X.shape[0]} samples, X={X.shape}, horizons={horizons}")
    d1 = d1_action_ablation(predict_fn, X, y, seed=int(args.seed))
    print(f"[D1] {json.dumps({k: v['r2'] for k, v in d1.items()})}")

    d2_approx = within_group_ranking(predict_fn, X, y, groups)
    control_matrix = ablate_action_block(X, seed=int(args.seed))["shuffled"]
    d2_control = within_group_ranking(predict_fn, control_matrix, y, groups)
    print(
        f"[D2-approx] n_groups={d2_approx['n_groups']} "
        f"spearman={d2_approx['spearman_mean']} (action-shuffled control "
        f"{d2_control['spearman_mean']})"
    )

    strict_block: dict[str, Any] | None = None
    d3 = {"mean": None, "median": None, "std": None, "n_states": 0}
    d4 = {"between_action_var": None, "within_action_noise_var": None, "snr": None}
    d5 = {"pearson_r": None, "slope": None, "intercept": None}
    if args.no_strict:
        strict_block = _strict_block_reason(RuntimeError("--no-strict requested"))
    else:
        try:
            paths = select_strict_snapshots(
                args.snapshots_dir,
                args.problems,
                int(args.strict_states_per_problem),
            )
            print(
                f"[strict] {len(paths)} states x {args.strict_candidates} "
                f"candidates x {args.strict_horizon} generations x "
                f"{args.strict_reps} reps"
            )
            records = run_strict_states(
                paths,
                encoder,
                predict_fn,
                n_candidates=int(args.strict_candidates),
                horizon=int(args.strict_horizon),
                n_reps=int(args.strict_reps),
            )
            horizon_column = (
                horizons.index(int(args.strict_horizon))
                if int(args.strict_horizon) in horizons
                else len(horizons) - 1
            )
            stats = strict_statistics(records, horizon_column=horizon_column)
            strict_block = {
                "available": True,
                "horizon": int(args.strict_horizon),
                "n_candidates": int(args.strict_candidates),
                "n_reps": int(args.strict_reps),
                "n_snapshots": len(records),
                **stats["d2_strict"],
            }
            d3 = stats["d3"]
            d4 = stats["d4"]
            d5 = stats["d5"]
        except Exception as exc:  # pragma: no cover - environment dependent
            strict_block = _strict_block_reason(exc)
            print(f"[strict] unavailable: {strict_block['reason']}")

    if strict_block is not None and strict_block.get("available"):
        print(
            f"[strict] spearman={strict_block['spearman_mean']} "
            f"kendall={strict_block['kendall_mean']} "
            f"agreement={strict_block['top1_selection_agreement']}"
        )
    d2 = dict(d2_approx)
    d2["strict_version"] = strict_block
    conclusion = conclude(
        d1, strict_block if strict_block and strict_block.get("available") else None
    )
    config = {
        "model_dir": str(model_dir),
        "eval_dir": str(args.eval_dir),
        "snapshots_dir": str(args.snapshots_dir),
        "problems": [str(p) for p in args.problems],
        "window": window,
        "horizons": horizons,
        "horizon_weights": list(DEFAULT_HORIZON_WEIGHTS),
        "n_samples": int(X.shape[0]),
        "n_trajectories": int(n_trajectories),
        "n_action_features": N_ACTION_FEATURES,
        "ablation_variants": ["zero", "mean", "shuffled"],
        "multiplier_normalization": not args.no_multiplier_normalization,
        "strict_enabled": not args.no_strict,
        "strict_states_per_problem": int(args.strict_states_per_problem),
        "strict_candidates": int(args.strict_candidates),
        "strict_horizon": int(args.strict_horizon),
        "strict_reps": int(args.strict_reps),
        "pm_mult_range": list(DEFAULT_PM_MULT_RANGE),
        "ref_point": list(DEFAULT_REF_POINT),
        "n_reference_points": int(DEFAULT_N_REFERENCE_POINTS),
        "seed": int(args.seed),
        "alpha": ALPHA,
        "conclusion_thresholds": {
            "ablation_r2_drop": ABLATION_R2_DROP_THRESHOLD,
            "strict_spearman": STRICT_SPEARMAN_THRESHOLD,
            "strict_significant_fraction": STRICT_SIGNIFICANT_FRACTION_THRESHOLD,
        },
        "d2_approximate_note": (
            "top-level D2 ranks samples inside (problem, generation) groups; "
            "the state differs inside a group, so it is a checksum, not the "
            "causal test -- strict_version holds the fixed-state evidence"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload = build_payload(
        d1=d1,
        d2=d2,
        d3=d3,
        d4=d4,
        d5=d5,
        conclusion=conclusion,
        config=config,
    )
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[done] conclusion={json.dumps(conclusion)}")
    print(f"[done] wrote {out_file}")
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point."""
    args = parse_args(argv)
    return run_diagnosis(args)


if __name__ == "__main__":
    main()
