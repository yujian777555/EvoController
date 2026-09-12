from __future__ import annotations

"""Phase 2.9 Task 1: Action Outcome Signature representation.

Phase 2.8 showed that the long-horizon action effect is measurable (pooled SNR
4.82 at h=20) but *not identifiable* from the 4-dimensional action parameters:
three strong non-neural baselines reach only pairwise AUC 0.498-0.525 and
win-count Kendall 0.14-0.17, and the action-only ranking quality equals the
concatenated one. The Phase-2.9 hypothesis is that the **representation** is the
bottleneck, so this script augments every ``(state, action)`` sample with an
*Action Outcome Signature* — how that action historically performed in **other**
states — and re-runs the Phase-2.8 audit protocol unchanged.

Signature definition (per row, per horizon)
-------------------------------------------
Signature of an action:

* operator (polynomial / gaussian, read from the action block's one-hot);
* the mutation multiplier binned into quartiles **within the operator**;
* the exploration strength binned into quartiles **within the operator** (eta_m
  and sigma live on different scales, so a shared binning would be meaningless);

which yields ``2 * 4 * 4 = 32`` buckets. Bin edges are quantiles of the
*training* actions only (``--bin-edges-from train``), so no validation-side
information enters the feature definition.

Why quartiles (measured on the actual corpus, 960 training states x 11
non-controller candidates = 10,560 rows per horizon): the bucket occupancy of
larger grids collapses, because the action sampler reuses a small set of
operator/parameter combinations.

======  =======  =========  ======  =======  ======
bins/op buckets  occupied   min     median   max
======  =======  =========  ======  =======  ======
3 x 3   18       18         315     526      1271
4 x 4   32       32         50      303      1097
5 x 5   50       50         38      193      993
6 x 6   72       72         8       133      966
8 x 8   128      120        0       77.5     971
======  =======  =========  ======  =======  ======

4 x 4 is the coarsest grid that already uses the whole action space well and
the finest one whose every bucket still holds >= 50 real rows, so a
leave-one-state-out mean and standard deviation stay meaningful and the
zero-fill fallback never triggers on this corpus (3 x 3 would leave 42% of the
rows sharing a bucket with a different parameter region). Binning *within*
operator is required because eta_m and sigma are different scales: the fitted
training edges at these quartiles are 5.38 / 13.93 / 20.0 for polynomial
exploration versus 0.039 / 0.078 / 0.151 for gaussian, so a shared binning
would place every gaussian action in the lowest bucket.

For every row the six features aggregate the outcomes of rows in the **same
bucket, same horizon** that belong to **other states** (leave-one-state-out):

===========================  ==================================================
``loo_mean_adv``             mean advantage of those rows (0.0 when empty)
``loo_std_adv``              standard deviation (0.0 when fewer than two rows)
``loo_frac_positive``        fraction with advantage > 0 (0.0 when empty)
``loo_count``                ``log1p`` of the number of aggregated rows
``loo_mean_rank``            mean within-state percentile rank (0.5 when empty)
``operator_loo_mean_adv``    same mean restricted to the row's operator
===========================  ==================================================

Empty buckets are filled with zeros (and 0.5 for the rank) rather than a global
mean, so a missing signature never smuggles state information into the feature.
``--signature-source`` decides whose outcomes may be aggregated: ``train``
(default) keeps the protocol inductive — validation rows are described only by
training-state outcomes — while ``all`` measures the transductive upper bound
(other validation states included, i.e. optimistic).

Evaluation protocol (identical to ``experiments/action_identifiability_audit.py``)
---------------------------------------------------------------------------------
The same state-level 960/240 split from
``results/phase2_75d/target_comparison_final.json`` is reused, the controller
candidate is excluded exactly like the audit's ``_groups``, and the audit's own
functions are called for the metrics that must stay comparable:

* :func:`experiments.action_identifiability_audit.task3_pairwise` — pairwise
  AUC and win-count Kendall per variant (the same pair construction, the same
  logistic / random-forest classifiers);
* :func:`experiments.action_identifiability_audit._ranking_from_scores` —
  within-state Spearman/Kendall/top-1/regret from per-row scores;
* :func:`experiments.action_identifiability_audit._load` and ``_groups`` — data
  loading and state grouping.

The regressors use the audit's parameterisation verbatim (OLS, RandomForest
``n_estimators=200, min_samples_leaf=2``, HistGradientBoosting
``max_iter=300, learning_rate=0.05``), trained per horizon on the training
states and scored on the held-out states.

Variants: ``action_only`` (4), ``action_plus_signature`` (4+6),
``concat`` (64), ``concat_plus_signature`` (70), ``signature_only`` (6).
Horizon-level top-k hit rate (k=3) and a state-level bootstrap CI of the
pairwise AUC are added on top of the audit's metrics.

Decision gate (``docs/PHASE2_9_ACTION_REPRESENTATION_REDESIGN_PLAN.md``)
-----------------------------------------------------------------------
The signature variant is judged at horizon 20 (the highest-SNR horizon of
Phase 2.8) on its best model by AUC:

* AUC > 0.55 **and** win-count Kendall > 0.25 **and** the 95% CI lower bound
  above 0.50 -> ``representation_is_bottleneck`` (continue with Tasks 2-3);
* AUC <= 0.53 -> ``representation_not_sufficient`` (freeze the analysis around
  the fundamental difficulty of action credit assignment);
* otherwise -> ``indeterminate``.

Output: ``results/phase2_9/action_representation.json`` (``--out``), including
the signature definition, bin edges, bucket statistics, every variant's
per-horizon metrics, the pairwise tables and the gate verdict.

Example:
    ``python experiments/action_representation_redesign.py``
    ``python experiments/action_representation_redesign.py --signature-source all``
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

if __package__ in (None, ""):
    # Allow ``python experiments/action_representation_redesign.py`` from the
    # repo root: the script directory (not the repo root) is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.action_identifiability_audit import (
    ACTION_FEATURES,
    DEFAULT_DATASET_DIR,
    DEFAULT_SPLIT_JSON,
    _groups,
    _load,
    _ranking_from_scores,
    task3_pairwise,
)

#: Artifact of this experiment (Phase-2.9 isolated).
DEFAULT_OUT = "results/phase2_9/action_representation.json"
#: Advantage horizons audited.
DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)
#: Horizon of the decision gate (highest pooled SNR in Phase 2.8).
GATE_HORIZON = 20
#: Signature feature names, in order.
CORE_SIGNATURE_FEATURES: tuple[str, ...] = (
    "loo_mean_adv",
    "loo_std_adv",
    "loo_frac_positive",
    "loo_count",
)
OPTIONAL_SIGNATURE_FEATURES: tuple[str, ...] = (
    "loo_mean_rank",
    "operator_loo_mean_adv",
)
SIGNATURE_FEATURES: tuple[str, ...] = (
    CORE_SIGNATURE_FEATURES + OPTIONAL_SIGNATURE_FEATURES
)
#: Fill values for buckets without any aggregated row.
SIGNATURE_FILL: dict[str, float] = {
    "loo_mean_adv": 0.0,
    "loo_std_adv": 0.0,
    "loo_frac_positive": 0.0,
    "loo_count": 0.0,
    "loo_mean_rank": 0.5,
    "operator_loo_mean_adv": 0.0,
}
#: Bin granularity of the continuous action parameters (quantile bins per
#: operator); 2 operators * 4 * 4 = 32 buckets.
MULTIPLIER_BINS = 4
EXPLORATION_BINS = 4
#: Variants compared by the audit protocol.
VARIANTS: tuple[str, ...] = (
    "action_only",
    "action_plus_signature",
    "concat",
    "concat_plus_signature",
    "signature_only",
)
#: Gate thresholds of the Phase-2.9 plan.
GATE_AUC = 0.55
GATE_KENDALL = 0.25
GATE_FAILURE_AUC = 0.53


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments of the signature experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2.9 Task 1: augment the action representation with a "
            "leave-one-state-out Action Outcome Signature and re-run the "
            "Phase-2.8 action-identifiability audit."
        )
    )
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR,
                        help="Directory with one sub-directory per target "
                        "(default: %(default)s).")
    parser.add_argument("--target", type=str, default="state_mean",
                        help="Advantage target sub-directory (default: %(default)s).")
    parser.add_argument("--split-json", type=str, default=DEFAULT_SPLIT_JSON,
                        help="Phase-2.75D report holding the shared 960/240 "
                        "state split (default: %(default)s).")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT,
                        help="Artifact JSON (default: %(default)s).")
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=list(DEFAULT_HORIZONS),
                        help="Advantage horizons (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed of the regressors (default: %(default)s).")
    parser.add_argument(
        "--signature-source", choices=["train", "all"], default="train",
        help="Whose outcomes may be aggregated into a signature: 'train' "
        "(default) keeps the protocol inductive, 'all' is the transductive "
        "upper bound that also uses other validation states.",
    )
    parser.add_argument(
        "--bin-edges-from", choices=["train", "all"], default="train",
        help="States used to fit the quantile bin edges of the signature "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--signature-features", choices=["core", "full"], default="full",
        help="'core' uses the four recommended features, 'full' (default) adds "
        "loo_mean_rank and operator_loo_mean_adv.",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["linear_ols", "random_forest", "hist_gradient_boosting"],
        default=["linear_ols", "random_forest", "hist_gradient_boosting"],
        help="Non-neural regressors of the audit protocol (default: all three).",
    )
    parser.add_argument("--top-k", type=int, default=3,
                        help="k of the top-k hit rate (default: %(default)s).")
    parser.add_argument("--n-bootstrap", type=int, default=200,
                        help="State-level bootstrap resamples of the pairwise AUC "
                        "(0 disables, default: %(default)s).")
    parser.add_argument("--max-train-states", type=int, default=0,
                        help="Optional cap on training states (smoke runs; "
                        "0 = all, default: %(default)s).")
    parser.add_argument("--max-val-states", type=int, default=0,
                        help="Optional cap on validation states (0 = all).")
    parser.add_argument("--skip-pairwise", action="store_true",
                        help="Skip the pairwise audit (ranking metrics only).")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# signature construction
# ---------------------------------------------------------------------------


def action_parameters(data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Operator / multiplier / exploration of every row.

    Reads the dataset's own action columns when present (the Phase-2.75D npz
    files carry ``mutation_operator``, ``mutation_multiplier`` and
    ``exploration_strength``) and otherwise falls back to the four action
    features at the end of ``X``.

    Returns:
        ``{"operator": str array, "multiplier": float array,
        "exploration": float array}``.
    """
    if {"mutation_operator", "mutation_multiplier", "exploration_strength"} <= set(
        data
    ):
        return {
            "operator": np.asarray(data["mutation_operator"]).astype(str),
            "multiplier": np.asarray(data["mutation_multiplier"], dtype=np.float64),
            "exploration": np.asarray(
                data["exploration_strength"], dtype=np.float64
            ),
        }
    action = np.asarray(data["X"], dtype=np.float64)[:, -ACTION_FEATURES:]
    polynomial = action[:, 2] >= action[:, 3]
    return {
        "operator": np.where(polynomial, "polynomial", "gaussian"),
        "multiplier": action[:, 0],
        "exploration": action[:, 1],
    }


def quantile_bin_edges(
    values: np.ndarray, operator: np.ndarray, n_bins: int
) -> dict[str, list[float]]:
    """Quantile bin edges of one action parameter, per operator.

    Args:
        values: Parameter values of the rows used to fit the edges.
        operator: Operator label per row.
        n_bins: Number of bins.

    Returns:
        ``{operator: [edges]}`` with ``n_bins - 1`` interior quantiles per
        operator (an empty list when the operator has no usable rows).

    Raises:
        ValueError: If ``n_bins`` < 2.
    """
    if int(n_bins) < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    edges: dict[str, list[float]] = {}
    for name in sorted(set(str(value) for value in operator)):
        selected = np.asarray(
            values[operator == name], dtype=np.float64
        )
        selected = selected[np.isfinite(selected)]
        if selected.size == 0:
            edges[name] = []
            continue
        quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
        edges[name] = [float(value) for value in np.quantile(selected, quantiles)]
    return edges


def assign_bin(
    values: np.ndarray, operator: np.ndarray, edges: dict[str, list[float]]
) -> np.ndarray:
    """Bin index of every row, using its operator's edges (clipped).

    Raises:
        ValueError: If an operator has no edges (empty source split).
    """
    result = np.zeros(len(values), dtype=np.int64)
    for name in sorted(set(str(value) for value in operator)):
        mask = operator == name
        if name not in edges or not edges[name]:
            raise ValueError(
                f"no bin edges for operator {name!r}; the bin-edge source split "
                f"contains no such action"
            )
        result[mask] = np.digitize(
            np.asarray(values[mask], dtype=np.float64), edges[name]
        )
    return result


def signature_bucket(
    data: dict[str, Any], *, multiplier_edges: dict[str, list[float]],
    exploration_edges: dict[str, list[float]],
) -> np.ndarray:
    """Integer bucket id of every row (operator x multiplier bin x exploration bin).

    Returns:
        Array of shape ``(n_rows,)`` with values in
        ``[0, 2 * MULTIPLIER_BINS * EXPLORATION_BINS)``.
    """
    parameters = action_parameters(data)
    operator = parameters["operator"]
    for name, table, expected in (
        ("multiplier", multiplier_edges, MULTIPLIER_BINS),
        ("exploration", exploration_edges, EXPLORATION_BINS),
    ):
        for key, values in table.items():
            if len(values) != int(expected) - 1:
                raise ValueError(
                    f"{name} edges for operator {key!r} have {len(values)} "
                    f"interior quantiles, expected {int(expected) - 1}"
                )
    multiplier_bin = assign_bin(
        parameters["multiplier"], operator, multiplier_edges
    )
    exploration_bin = assign_bin(
        parameters["exploration"], operator, exploration_edges
    )
    operator_index = np.where(operator == "polynomial", 0, 1)
    return (
        (operator_index * MULTIPLIER_BINS + multiplier_bin) * EXPLORATION_BINS
        + exploration_bin
    )


def within_state_ranks(
    data: dict[str, Any], horizons: Sequence[int]
) -> np.ndarray:
    """Within-state percentile rank (``[0, 1]``) of every row's advantage.

    Rows outside any horizon group keep rank ``0.5`` (neutral).

    Returns:
        Array of shape ``(n_rows,)``.
    """
    ranks = np.full(len(data["y"]), 0.5, dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    for horizon in horizons:
        groups = _groups(data["state"], data["horizon"], data["kind"], int(horizon))
        for rows in groups.values():
            values = y[np.asarray(rows, dtype=int)]
            if values.size < 2:
                continue
            order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
            ranks[np.asarray(rows, dtype=int)] = order / (values.size - 1)
    return ranks


def build_signature(
    data: dict[str, Any],
    *,
    horizons: Sequence[int],
    source_mask: np.ndarray,
    loo_mask: np.ndarray | None = None,
    multiplier_edges: dict[str, list[float]] | None = None,
    exploration_edges: dict[str, list[float]] | None = None,
    feature_names: Sequence[str] = SIGNATURE_FEATURES,
) -> dict[str, Any]:
    """Leave-one-state-out action outcome signatures for every row.

    For each row the aggregation runs over the rows of the **same horizon and
    signature bucket** that are in ``source_mask`` and in a **different state**
    than the row (leave-one-state-out). When ``source_mask`` is the training
    split the construction is inductive; when it is all rows it also uses other
    validation states' outcomes (transductive upper bound).

    Args:
        data: Audit-style dataset (``X``/``y``/``horizon``/``kind``/``state``).
        horizons: Horizons to build signatures for.
        source_mask: Rows whose outcomes may be aggregated.
        loo_mask: Rows whose own state must be excluded; ``None`` uses
            ``source_mask``.
        multiplier_edges: Pre-computed bin edges; ``None`` fits them on
            ``source_mask``.
        exploration_edges: See ``multiplier_edges``.
        feature_names: Signature columns to emit.

    Returns:
        ``{"features": (n_rows, len(feature_names)) array, "feature_names": [...],
        "multiplier_edges": {...}, "exploration_edges": {...},
        "source_mask": source_mask, "loo_mask": loo_mask,
        "bucket_counts": {horizon: n_source_rows},
        "bucket_sizes": {horizon: rows-per-occupied-bucket summary},
        "fill_fraction": float}``.
    """
    operator = action_parameters(data)["operator"]
    if multiplier_edges is None or exploration_edges is None:
        fitted_multiplier = quantile_bin_edges(
            action_parameters(data)["multiplier"][source_mask],
            operator[source_mask],
            MULTIPLIER_BINS,
        )
        fitted_exploration = quantile_bin_edges(
            action_parameters(data)["exploration"][source_mask],
            operator[source_mask],
            EXPLORATION_BINS,
        )
        multiplier_edges = multiplier_edges or fitted_multiplier
        exploration_edges = exploration_edges or fitted_exploration
    buckets = signature_bucket(
        data,
        multiplier_edges=multiplier_edges,
        exploration_edges=exploration_edges,
    )
    ranks = within_state_ranks(data, horizons)
    y = np.asarray(data["y"], dtype=np.float64)
    horizon_array = np.asarray(data["horizon"], dtype=np.int64)
    state_array = np.asarray(data["state"]).astype(str)
    if loo_mask is None:
        loo_mask = source_mask
    n_rows = len(y)
    features = np.zeros((n_rows, len(feature_names)), dtype=np.float64)
    for position, name in enumerate(feature_names):
        features[:, position] = SIGNATURE_FILL[name]
    bucket_counts: dict[str, int] = {}
    bucket_sizes: dict[str, dict[str, Any]] = {}
    filled_rows = 0
    total_rows = 0
    for horizon in horizons:
        horizon_mask = horizon_array == int(horizon)
        source = horizon_mask & source_mask
        # Aggregation tables: per bucket totals over the source rows (rebuilt
        # per horizon because the advantage is horizon-specific).
        per_bucket: dict[int, dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "sum": 0.0, "sumsq": 0.0, "positive": 0.0,
                     "rank_sum": 0.0}
        )
        per_state_bucket: dict[tuple[str, int], dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "sum": 0.0, "sumsq": 0.0, "positive": 0.0,
                     "rank_sum": 0.0}
        )
        per_operator: dict[str, dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "sum": 0.0}
        )
        per_state_operator: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "sum": 0.0}
        )
        source_rows = np.flatnonzero(source)
        bucket_counts[str(int(horizon))] = int(source_rows.size)
        for row in source_rows:
            bucket = int(buckets[row])
            state = state_array[row]
            value = float(y[row])
            entry = per_bucket[bucket]
            entry["count"] += 1.0
            entry["sum"] += value
            entry["sumsq"] += value * value
            entry["positive"] += 1.0 if value > 0.0 else 0.0
            entry["rank_sum"] += float(ranks[row])
            key = (state, bucket)
            state_entry = per_state_bucket[key]
            state_entry["count"] += 1.0
            state_entry["sum"] += value
            state_entry["sumsq"] += value * value
            state_entry["positive"] += 1.0 if value > 0.0 else 0.0
            state_entry["rank_sum"] += float(ranks[row])
            operator_name = str(operator[row])
            operator_entry = per_operator[operator_name]
            operator_entry["count"] += 1.0
            operator_entry["sum"] += value
            state_operator = per_state_operator[(state, operator_name)]
            state_operator["count"] += 1.0
            state_operator["sum"] += value
        sizes = np.asarray(
            [entry["count"] for entry in per_bucket.values()], dtype=np.float64
        )
        bucket_sizes[str(int(horizon))] = {
            "n_buckets_occupied": int(sizes.size),
            "n_buckets_total": int(2 * MULTIPLIER_BINS * EXPLORATION_BINS),
            "rows_per_bucket_min": int(sizes.min()) if sizes.size else 0,
            "rows_per_bucket_median": float(np.median(sizes)) if sizes.size else 0.0,
            "rows_per_bucket_max": int(sizes.max()) if sizes.size else 0,
        }
        for row in np.flatnonzero(horizon_mask & loo_mask):
            total_rows += 1
            bucket = int(buckets[row])
            state = state_array[row]
            entry = per_bucket.get(bucket)
            if entry is None:
                filled_rows += 1
                continue
            own = per_state_bucket.get((state, bucket))
            count = entry["count"] - (own["count"] if own else 0.0)
            total = entry["sum"] - (own["sum"] if own else 0.0)
            sumsq = entry["sumsq"] - (own["sumsq"] if own else 0.0)
            positive = entry["positive"] - (own["positive"] if own else 0.0)
            rank_sum = entry["rank_sum"] - (own["rank_sum"] if own else 0.0)
            operator_name = str(operator[row])
            operator_entry = per_operator[operator_name]
            own_operator = per_state_operator.get((state, operator_name))
            operator_count = operator_entry["count"] - (
                own_operator["count"] if own_operator else 0.0
            )
            operator_sum = operator_entry["sum"] - (
                own_operator["sum"] if own_operator else 0.0
            )
            values: dict[str, float] = {
                "loo_count": float(np.log1p(max(count, 0.0))),
                "loo_mean_adv": SIGNATURE_FILL["loo_mean_adv"],
                "loo_std_adv": SIGNATURE_FILL["loo_std_adv"],
                "loo_frac_positive": SIGNATURE_FILL["loo_frac_positive"],
                "loo_mean_rank": SIGNATURE_FILL["loo_mean_rank"],
                "operator_loo_mean_adv": (
                    float(operator_sum / operator_count)
                    if operator_count > 0
                    else SIGNATURE_FILL["operator_loo_mean_adv"]
                ),
            }
            if count > 0:
                mean = total / count
                values["loo_mean_adv"] = float(mean)
                values["loo_frac_positive"] = float(positive / count)
                values["loo_mean_rank"] = float(rank_sum / count)
                if count > 1:
                    variance = max(sumsq / count - mean * mean, 0.0)
                    values["loo_std_adv"] = float(np.sqrt(variance))
            else:
                filled_rows += 1
            for position, name in enumerate(feature_names):
                features[row, position] = values[name]
    fill_fraction = float(filled_rows / max(total_rows, 1))
    return {
        "features": features,
        "feature_names": [str(name) for name in feature_names],
        "multiplier_edges": multiplier_edges,
        "exploration_edges": exploration_edges,
        "bucket_counts": bucket_counts,
        "bucket_sizes": bucket_sizes,
        "fill_fraction": fill_fraction,
    }


def variant_matrices(
    data: dict[str, Any], signature: np.ndarray
) -> dict[str, np.ndarray]:
    """Feature matrix of every audit variant.

    Returns:
        ``{variant: matrix}`` with ``action_only`` (4 columns),
        ``action_plus_signature`` (4 + |signature|), ``concat`` (all of ``X``),
        ``concat_plus_signature`` and ``signature_only``.
    """
    context = np.asarray(data["X"], dtype=np.float64)
    action = context[:, -ACTION_FEATURES:]
    return {
        "action_only": action,
        "action_plus_signature": np.hstack([action, signature]),
        "concat": context,
        "concat_plus_signature": np.hstack([context, signature]),
        "signature_only": signature,
    }


# ---------------------------------------------------------------------------
# evaluation (audit protocol)
# ---------------------------------------------------------------------------


def _model_factories(seed: int, names: Sequence[str]) -> dict[str, Callable[[], Any]]:
    """Regressor factories with the audit's parameters verbatim.

    The parameterisation mirrors ``task2_action_only`` in
    ``experiments/action_identifiability_audit.py`` (OLS, RandomForest
    ``n_estimators=200, min_samples_leaf=2``, HistGradientBoosting
    ``max_iter=300, learning_rate=0.05``) so numbers stay comparable.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    available: dict[str, Callable[[], Any]] = {
        "linear_ols": lambda: LinearRegression(n_jobs=-1),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=seed
        ),
    }
    return {name: available[name] for name in names}


def _topk_hit_rate(
    scores: np.ndarray,
    data: dict[str, Any],
    horizon: int,
    states: Sequence[str],
    k: int,
) -> float | None:
    """Fraction of states whose true best candidate is in the top-k predictions.

    ``states`` aligns with ``scores``/``data`` row by row (the audit passes the
    held-out state keys of its own filtered sub-dataset).
    """
    groups = _groups(states, data["horizon"], data["kind"], int(horizon))
    hits = 0
    total = 0
    for rows in groups.values():
        if len(rows) < 2:
            continue
        values = np.asarray(data["y"])[np.asarray(rows, dtype=int)]
        best = int(np.argmax(values))
        predicted = np.asarray(scores)[np.asarray(rows, dtype=int)]
        order = np.argsort(-predicted, kind="stable")[: int(k)]
        hits += 1 if best in set(int(index) for index in order) else 0
        total += 1
    return float(hits / total) if total else None


def evaluate_variants(
    data: dict[str, Any],
    is_val: np.ndarray,
    matrices: dict[str, np.ndarray],
    horizons: Sequence[int],
    *,
    model_names: Sequence[str],
    seed: int,
    top_k: int,
) -> dict[str, Any]:
    """Train the audit baselines per variant and score the held-out states.

    Args:
        data: Audit dataset (used for the split, targets and grouping).
        is_val: Boolean row mask of the held-out states.
        matrices: Variant feature matrices from :func:`variant_matrices`.
        horizons: Horizons to evaluate.
        model_names: Regressors to train per variant.
        seed: Random seed.
        top_k: k of the top-k hit rate.

    Returns:
        ``{variant: {"dim", "models": {name: {"per_horizon": {...}}}}`` with the
        audit's ranking metrics plus ``top{k}_hit_rate``.
    """
    report: dict[str, Any] = {}
    for variant, matrix in matrices.items():
        factories = _model_factories(seed, model_names)
        entry: dict[str, Any] = {"dim": int(matrix.shape[1]), "models": {}}
        for name, factory in factories.items():
            scores = np.zeros(matrix.shape[0], dtype=np.float64)
            for horizon in horizons:
                train = np.flatnonzero((data["horizon"] == horizon) & ~is_val)
                test = np.flatnonzero((data["horizon"] == horizon) & is_val)
                if train.size == 0 or test.size == 0:
                    continue
                model = factory()
                model.fit(matrix[train], np.asarray(data["y"])[train])
                scores[test] = model.predict(matrix[test])
            val_rows = np.flatnonzero(is_val)
            subset = {
                "X": matrix[val_rows],
                "y": np.asarray(data["y"])[val_rows],
                "horizon": np.asarray(data["horizon"])[val_rows],
                "kind": np.asarray(data["kind"])[val_rows],
                "rewards": np.asarray(data["rewards"])[val_rows],
            }
            val_states = [str(data["state"][row]) for row in val_rows]
            per_horizon: dict[str, Any] = {}
            for horizon in horizons:
                metrics = _ranking_from_scores(
                    scores[val_rows], subset, int(horizon), val_states
                )
                metrics[f"top{int(top_k)}_hit_rate"] = _topk_hit_rate(
                    scores[val_rows], subset, int(horizon), val_states, int(top_k)
                )
                per_horizon[str(int(horizon))] = metrics
            entry["models"][name] = {"per_horizon": per_horizon}
        report[variant] = entry
        print(
            f"[ranking] {variant:24} dim={entry['dim']:3d} "
            + " ".join(
                f"h{h}:rho={entry['models'][model_names[0]]['per_horizon'][str(h)].get('spearman_mean')}"
                for h in horizons
            )
        )
    return report


def pairwise_auc_ci(
    matrix: np.ndarray,
    data: dict[str, Any],
    is_val: np.ndarray,
    horizon: int,
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Pairwise AUC of one variant plus a state-level bootstrap 95% CI.

    A logistic-regression preference model is fitted on the training-state
    pairs (identical construction to
    :func:`experiments.action_identifiability_audit.task3_pairwise`); the
    confidence interval resamples **validation states** with replacement and
    recomputes the AUC from the fixed model's scores, i.e. it quantifies how
    much the estimate depends on which states were held out.

    Returns:
        ``{"auc", "accuracy", "n_pairs", "n_states", "ci95": [lo, hi],
        "n_bootstrap": int}``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    groups = _groups(data["state"], data["horizon"], data["kind"], int(horizon))
    train_diffs, train_labels = [], []
    val_diffs, val_labels, val_state_of_pair = [], [], []
    for key, rows in groups.items():
        pairs = [
            (rows[i], rows[j])
            for i in range(len(rows))
            for j in range(i + 1, len(rows))
            if data["y"][rows[i]] != data["y"][rows[j]]
        ]
        if not pairs:
            continue
        is_val_state = bool(is_val[rows[0]])
        for a, b in pairs:
            diff = matrix[a] - matrix[b]
            label = 1.0 if data["y"][a] > data["y"][b] else 0.0
            if is_val_state:
                val_diffs.append(diff)
                val_labels.append(label)
                val_state_of_pair.append(key)
            else:
                train_diffs.append(diff)
                train_labels.append(label)
    if not train_diffs or not val_diffs:
        return {"auc": None, "accuracy": None, "n_pairs": 0, "n_states": 0,
                "ci95": None, "n_bootstrap": 0}
    model = LogisticRegression(max_iter=1000)
    model.fit(np.asarray(train_diffs), np.asarray(train_labels))
    proba = model.predict_proba(np.asarray(val_diffs))[:, 1]
    labels = np.asarray(val_labels)
    auc = float(roc_auc_score(labels, proba)) if len(set(labels.tolist())) > 1 else None
    accuracy = float(np.mean((proba > 0.5) == (labels > 0.5)))
    states = np.asarray(val_state_of_pair)
    unique_states = sorted(set(states.tolist()))
    ci: list[float] | None = None
    n_used = 0
    if n_bootstrap > 0 and len(unique_states) > 1 and auc is not None:
        rng = np.random.Generator(np.random.PCG64(seed))
        by_state = {key: np.flatnonzero(states == key) for key in unique_states}
        aucs: list[float] = []
        for _ in range(int(n_bootstrap)):
            drawn = rng.choice(len(unique_states), size=len(unique_states), replace=True)
            indices = np.concatenate([by_state[unique_states[i]] for i in drawn])
            resampled_labels = labels[indices]
            if len(set(resampled_labels.tolist())) < 2:
                continue
            aucs.append(float(roc_auc_score(resampled_labels, proba[indices])))
        if len(aucs) > 1:
            ci = [
                float(np.percentile(aucs, 2.5)),
                float(np.percentile(aucs, 97.5)),
            ]
            n_used = len(aucs)
    return {
        "auc": auc,
        "accuracy": accuracy,
        "n_pairs": int(len(val_diffs)),
        "n_states": len(unique_states),
        "ci95": ci,
        "n_bootstrap": n_used,
    }


# ---------------------------------------------------------------------------
# decision gate
# ---------------------------------------------------------------------------


def evaluate_gate(
    pairwise: dict[str, Any],
    *,
    horizon: int = GATE_HORIZON,
    baseline_variant: str = "action_only",
    candidate_variant: str = "action_plus_signature",
    auc_threshold: float = GATE_AUC,
    kendall_threshold: float = GATE_KENDALL,
    failure_auc: float = GATE_FAILURE_AUC,
) -> dict[str, Any]:
    """Judge the signature representation against the Phase-2.9 gate.

    The verdict uses the candidate variant's **best model by AUC** at
    ``horizon`` and requires AUC above ``auc_threshold``, win-count Kendall
    above ``kendall_threshold`` and (when a CI is available) a lower bound
    above 0.50. An AUC at or below ``failure_auc`` is the plan's failure case.

    Args:
        pairwise: ``{variant: {horizon: entry}}`` as returned by the audit's
            ``task3_pairwise`` (plus optional ``{"auc_ci": {...}}`` entries).
        horizon: Horizon of the verdict.
        baseline_variant: Variant the signature is compared against.
        candidate_variant: Variant carrying the signature features.
        auc_threshold: Success threshold of the plan.
        kendall_threshold: Success threshold of the plan.
        failure_auc: Upper bound of the plan's failure band.

    Returns:
        ``{"verdict", "evidence", "numbers", "baseline", "candidate"}``.
    """
    def _best(variant: str) -> dict[str, Any]:
        entry = (pairwise.get(variant) or {}).get(str(int(horizon))) or {}
        models = entry.get("models", {})
        best_name, best_auc, best_kendall = None, None, None
        for name, metrics in models.items():
            auc = metrics.get("pairwise_auc")
            if auc is None:
                continue
            if best_auc is None or auc > best_auc:
                best_name = name
                best_auc = float(auc)
                best_kendall = metrics.get("within_state_kendall_from_wins")
        return {
            "model": best_name,
            "auc": best_auc,
            "kendall": best_kendall,
            "n_train_pairs": entry.get("n_train_pairs"),
            "n_val_pairs": entry.get("n_val_pairs"),
        }

    baseline = _best(baseline_variant)
    candidate = _best(candidate_variant)
    ci = (
        (pairwise.get(candidate_variant) or {})
        .get("auc_ci", {})
        .get(str(int(horizon)), {})
    )
    ci95 = ci.get("ci95")
    numbers = {
        "horizon": int(horizon),
        "baseline": baseline,
        "candidate": candidate,
        "auc_threshold": float(auc_threshold),
        "kendall_threshold": float(kendall_threshold),
        "failure_auc": float(failure_auc),
        "candidate_auc_ci95": ci95,
        "auc_delta": (
            None
            if baseline["auc"] is None or candidate["auc"] is None
            else float(candidate["auc"] - baseline["auc"])
        ),
        "kendall_delta": (
            None
            if baseline["kendall"] is None or candidate["kendall"] is None
            else float(candidate["kendall"] - baseline["kendall"])
        ),
    }
    if candidate["auc"] is None:
        return {
            "verdict": "indeterminate",
            "evidence": "the signature variant produced no pairwise AUC",
            "numbers": numbers,
            "baseline": baseline,
            "candidate": candidate,
        }
    auc = float(candidate["auc"])
    kendall = float(candidate["kendall"] or 0.0)
    significant = ci95 is None or float(ci95[0]) > 0.5
    if auc > auc_threshold and kendall > kendall_threshold and significant:
        verdict = "representation_is_bottleneck"
        evidence = (
            f"pairwise AUC {auc:.3f} > {auc_threshold} and win-count Kendall "
            f"{kendall:.3f} > {kendall_threshold} at h={horizon}"
            + (f" (95% CI lower bound {ci95[0]:.3f} > 0.5)" if ci95 else "")
            + f"; baseline {baseline_variant} AUC {baseline['auc']}"
        )
    elif auc <= failure_auc:
        verdict = "representation_not_sufficient"
        evidence = (
            f"pairwise AUC {auc:.3f} <= {failure_auc} at h={horizon}: the "
            f"signature representation does not restore identifiability "
            f"(baseline {baseline_variant} AUC {baseline['auc']})"
        )
    else:
        verdict = "indeterminate"
        evidence = (
            f"pairwise AUC {auc:.3f} at h={horizon} sits between the failure "
            f"band ({failure_auc}) and the success threshold ({auc_threshold}), "
            f"or the Kendall {kendall:.3f} misses {kendall_threshold}"
        )
    return {
        "verdict": verdict,
        "evidence": evidence,
        "numbers": numbers,
        "baseline": baseline,
        "candidate": candidate,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Build signatures, evaluate every variant and write the artifact.

    Returns:
        The payload written to ``--out``.
    """
    started = time.perf_counter()
    dataset_dir = Path(args.dataset_dir)
    split_payload = json.loads(
        Path(args.split_json).read_text(encoding="utf-8")
    )
    split = split_payload["split"]
    val_keys = set(split["val_state_keys"])
    train_keys = set(split["train_state_keys"])
    data = _load(dataset_dir, str(args.target))
    state_array = np.asarray(data["state"]).astype(str)
    # The audit excludes the controller candidate from every within-state
    # analysis; the signature and the evaluation use the same row universe.
    candidate_mask = np.asarray(data["kind"]).astype(str) != "controller"
    is_val = np.asarray([state in val_keys for state in state_array])
    is_train = np.asarray([state in train_keys for state in state_array])
    if args.max_train_states:
        keep = sorted({state for state, flag in zip(state_array, is_train) if flag})
        keep = set(keep[: int(args.max_train_states)])
        is_train = is_train & np.asarray([state in keep for state in state_array])
    if args.max_val_states:
        keep = sorted({state for state, flag in zip(state_array, is_val) if flag})
        keep = set(keep[: int(args.max_val_states)])
        is_val = is_val & np.asarray([state in keep for state in state_array])
    usable = candidate_mask & (is_train | is_val)
    filtered = {
        key: value
        for key, value in data.items()
        if key not in {"state", "problem"}
    }
    for key in ("X", "y", "horizon", "kind", "rewards"):
        filtered[key] = np.asarray(data[key])[usable]
    for key in ("mutation_operator", "mutation_multiplier", "exploration_strength"):
        if key in data:
            filtered[key] = np.asarray(data[key])[usable]
    filtered["state"] = [state for state, keep in zip(state_array, usable) if keep]
    filtered["problem"] = [
        problem for problem, keep in zip(data["problem"], usable) if keep
    ]
    is_val = is_val[usable]
    is_train = is_train[usable]
    horizons = [int(h) for h in args.horizons]

    source_mask = is_train if str(args.signature_source) == "train" else (
        is_train | is_val
    )
    edge_mask = is_train if str(args.bin_edges_from) == "train" else (
        is_train | is_val
    )
    feature_names = (
        CORE_SIGNATURE_FEATURES
        if str(args.signature_features) == "core"
        else SIGNATURE_FEATURES
    )
    operator = action_parameters(filtered)["operator"]
    parameters = action_parameters(filtered)
    multiplier_edges = quantile_bin_edges(
        parameters["multiplier"][edge_mask], operator[edge_mask], MULTIPLIER_BINS
    )
    exploration_edges = quantile_bin_edges(
        parameters["exploration"][edge_mask], operator[edge_mask], EXPLORATION_BINS
    )
    signature = build_signature(
        filtered,
        horizons=horizons,
        source_mask=source_mask,
        loo_mask=np.ones(len(filtered["y"]), dtype=bool),
        multiplier_edges=multiplier_edges,
        exploration_edges=exploration_edges,
        feature_names=feature_names,
    )
    print(
        f"[signature] {len(feature_names)} features, "
        f"{len(multiplier_edges)} operator(s), source={args.signature_source}, "
        f"edges from {args.bin_edges_from}, fill={signature['fill_fraction']:.3f}"
    )
    matrices = variant_matrices(filtered, signature["features"])
    ranking = evaluate_variants(
        filtered,
        is_val,
        matrices,
        horizons,
        model_names=[str(name) for name in args.models],
        seed=int(args.seed),
        top_k=int(args.top_k),
    )
    pairwise: dict[str, Any] = {}
    auc_ci: dict[str, Any] = {}
    if not args.skip_pairwise:
        for variant in VARIANTS:
            augmented = dict(filtered)
            augmented["X"] = matrices[variant]
            pairwise[variant] = task3_pairwise(
                augmented, is_val, horizons, int(args.seed)
            )
            print(
                f"[pairwise] {variant:24} "
                + " ".join(
                    f"h{h}:auc="
                    f"{(pairwise[variant].get(str(h)) or {}).get('models', {}).get('logistic', {}).get('pairwise_auc', 'n/a')}"
                    for h in horizons
                )
            )
        for variant in ("action_only", "action_plus_signature", "signature_only"):
            auc_ci[variant] = {}
            for horizon in horizons:
                auc_ci[variant][str(horizon)] = pairwise_auc_ci(
                    matrices[variant],
                    filtered,
                    is_val,
                    int(horizon),
                    seed=int(args.seed),
                    n_bootstrap=int(args.n_bootstrap),
                )
        for variant, per_horizon in auc_ci.items():
            pairwise.setdefault(variant, {})
            pairwise[variant]["auc_ci"] = per_horizon
    gate = evaluate_gate(
        pairwise, horizon=GATE_HORIZON,
        baseline_variant="action_only",
        candidate_variant="action_plus_signature",
    )
    payload: dict[str, Any] = {
        "config": {
            "dataset_dir": str(dataset_dir),
            "target": str(args.target),
            "split_json": str(args.split_json),
            "horizons": horizons,
            "seed": int(args.seed),
            "signature_source": str(args.signature_source),
            "bin_edges_from": str(args.bin_edges_from),
            "signature_features": str(args.signature_features),
            "models": [str(name) for name in args.models],
            "top_k": int(args.top_k),
            "n_bootstrap": int(args.n_bootstrap),
            "variants": list(VARIANTS),
            "protocol": (
                "Phase-2.8 audit protocol: same state-level split, controller "
                "candidate excluded, audit regressors (OLS / RandomForest / "
                "HistGradientBoosting) per horizon, pairwise AUC and win-count "
                "Kendall from experiments.action_identifiability_audit"
            ),
            # No wall-clock timestamp: the artifact must be byte-stable across
            # re-runs with the same inputs (only wall_time_sec varies).
        },
        "data": {
            "n_rows": int(len(filtered["y"])),
            "n_train_rows": int(is_train.sum()),
            "n_val_rows": int(is_val.sum()),
            "n_states": int(len(set(filtered["state"]))),
            "n_train_states": int(len({s for s, f in zip(filtered["state"], is_train) if f})),
            "n_val_states": int(len({s for s, f in zip(filtered["state"], is_val) if f})),
            "split_unit": split.get("unit"),
            "controller_rows_excluded": int((~candidate_mask).sum()),
            "max_train_states": int(args.max_train_states),
            "max_val_states": int(args.max_val_states),
            "horizons": horizons,
        },
        "signature": {
            "feature_names": signature["feature_names"],
            "core_features": list(CORE_SIGNATURE_FEATURES),
            "optional_features": list(OPTIONAL_SIGNATURE_FEATURES),
            "n_buckets": int(2 * MULTIPLIER_BINS * EXPLORATION_BINS),
            "multiplier_bins": int(MULTIPLIER_BINS),
            "exploration_bins": int(EXPLORATION_BINS),
            "multiplier_edges": signature["multiplier_edges"],
            "exploration_edges": signature["exploration_edges"],
            "fill_values": dict(SIGNATURE_FILL),
            "fill_fraction": signature["fill_fraction"],
            "bucket_counts_by_horizon": signature["bucket_counts"],
            "bucket_sizes_by_horizon": signature["bucket_sizes"],
            "definition": (
                "per (state, action, horizon): statistics of the advantage of "
                "rows in the same signature bucket (operator x multiplier "
                "quartile x exploration quartile, bins fitted per operator) "
                "that belong to OTHER states of the aggregation source; empty "
                "buckets are filled with zeros (rank 0.5), never with a global "
                "mean, so no state information leaks into the feature"
            ),
        },
        "variants": ranking,
        "pairwise": pairwise,
        "gate": gate,
        "wall_time_sec": float(time.perf_counter() - started),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\n[gate] {gate['verdict']}: {gate['evidence']}")
    print(f"[done] wrote {out_path} ({payload['wall_time_sec']:.1f}s)")
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point."""
    return run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
