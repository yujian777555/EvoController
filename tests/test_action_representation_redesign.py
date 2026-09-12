from __future__ import annotations

"""Tests for the Phase-2.9 Task-1 Action Outcome Signature redesign.

The signature must satisfy three contract properties, and each is checked
against hand-computed numbers rather than against a second implementation:

1. **leave-one-state-out** -?the aggregation of a row never contains a row of
   the row's own state, so tampering with one state's outcomes changes every
   *other* state's signature and leaves that state's own signature untouched;
2. **deterministic per-operator quantile binning** -?bins are fitted per
   operator (eta_m and sigma are not comparable scales) and are a pure function
   of the edge source;
3. **zero-fill for empty buckets** -?a bucket with no aggregated row yields the
   documented fill values (0.0, and 0.5 for the rank), never a global mean.

The driver is exercised end to end on a synthetic on-disk dataset that follows
the Phase-2.75D ``intervention_dataset_*.npz`` schema plus a split JSON, with a
single ``linear_ols`` model, and is asserted to be byte-deterministic apart from
``wall_time_sec``.

No real dataset is read and no real evaluation is run: the fixture is
8 states x 6 candidates x 2 horizons.
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pytest

from experiments import action_representation_redesign as arr

_HORIZONS = (5, 20)
#: Action parameters of the tiny hand-computed fixture, one entry per candidate:
#: (operator, mutation_multiplier, exploration_strength). Two values per operator
#: is enough for the quartile edges to separate every candidate into its own
#: bucket.
_ACTIONS: tuple[tuple[str, float, float], ...] = (
    ("polynomial", 1.0, 4.0),
    ("polynomial", 3.0, 40.0),
    ("gaussian", 1.0, 0.05),
    ("gaussian", 3.0, 0.25),
)


# --- fixture helpers ---------------------------------------------------------


@contextmanager
def _serial_joblib() -> Iterator[None]:
    """Run sklearn's parallel fits on one thread.

    ``RandomForestClassifier(n_jobs=-1)`` inside the audit's ``task3_pairwise``
    builds a ``multiprocessing.pool.ThreadPool``, which on Windows needs a named
    pipe -?unavailable in this repository's test sandbox. The override changes
    scheduling only: every tree still uses its own seed, so the fitted forest and
    every metric are identical (asserted by the determinism check below).
    """
    import joblib

    with joblib.parallel_backend("sequential"):
        yield


def _run(argv: Sequence[str] | argparse.Namespace) -> dict[str, Any]:
    """Run the driver from CLI arguments (or an already parsed Namespace)."""
    args = argv if isinstance(argv, argparse.Namespace) else arr.parse_args(list(argv))
    with _serial_joblib():
        return arr.run_experiment(args)


def _action_features(operator: str, multiplier: float, exploration: float) -> list[float]:
    """The repo-wide action block: [mult, expl, onehot_polynomial, onehot_gaussian]."""
    return [
        float(multiplier),
        float(exploration),
        1.0 if operator == "polynomial" else 0.0,
        1.0 if operator == "gaussian" else 0.0,
    ]


def _tiny_data(
    *,
    y_by_state: dict[str, Sequence[float]] | None = None,
    horizons: Sequence[int] = (20,),
    horizon_scale: float = 0.0,
) -> dict[str, Any]:
    """Two states x four shared candidate actions x ``horizons``.

    Actions are identical across states, so every signature bucket holds exactly
    one row per state and the leave-one-state-out aggregation reduces to a single
    source row -?which makes the expected numbers hand-computable.

    Args:
        y_by_state: Advantage per state, one value per candidate (defaults are
            distinct within each state so ranks are order-independent).
        horizons: Horizons to emit.
        horizon_scale: Multiplies the advantage by ``1 + horizon_scale * h``,
            which makes the signature horizon-specific.
    """
    default_y = {
        "zdt1|1|0": (1.0, -2.0, 0.5, -0.5),
        "zdt1|2|0": (2.0, -1.0, 0.25, -0.75),
    }
    y_by_state = dict(y_by_state or default_y)
    states = sorted(y_by_state)
    xs, ys, hs, kinds, rewards, state_keys = [], [], [], [], [], []
    operators, multipliers, explorations = [], [], []
    for state in states:
        for horizon in horizons:
            for index, action in enumerate(_ACTIONS):
                operator, multiplier, exploration = action
                value = float(y_by_state[state][index]) * (
                    1.0 + float(horizon_scale) * float(horizon)
                )
                xs.append(
                    [float(states.index(state)), 1.0]
                    + _action_features(operator, multiplier, exploration)
                )
                ys.append(value)
                hs.append(int(horizon))
                kinds.append("alternative")
                rewards.append([value, value])
                state_keys.append(state)
                operators.append(operator)
                multipliers.append(multiplier)
                explorations.append(exploration)
    return {
        "X": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "horizon": np.asarray(hs, dtype=np.int64),
        "kind": np.asarray(kinds, dtype=object),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "state": state_keys,
        "mutation_operator": np.asarray(operators, dtype=object),
        "mutation_multiplier": np.asarray(multipliers, dtype=np.float64),
        "exploration_strength": np.asarray(explorations, dtype=np.float64),
    }


def _signature(
    data: dict[str, Any],
    *,
    source_mask: np.ndarray,
    loo_mask: np.ndarray | None = None,
    horizons: Sequence[int] = (20,),
    feature_names: Sequence[str] = arr.SIGNATURE_FEATURES,
) -> dict[str, Any]:
    """Build a signature over ``data`` (full feature set by default)."""
    return arr.build_signature(
        data,
        horizons=horizons,
        source_mask=source_mask,
        loo_mask=loo_mask,
        feature_names=feature_names,
    )


def _write_fixture(
    root: Path,
    *,
    n_states: int = 8,
    horizons: Sequence[int] = _HORIZONS,
) -> tuple[Path, Path]:
    """Write a synthetic Phase-2.75D dataset plus its shared split JSON.

    Returns:
        ``(dataset_dir, split_json)``.
    """
    dataset_dir = root / "dataset"
    target_dir = dataset_dir / "state_mean"
    target_dir.mkdir(parents=True, exist_ok=True)
    actions = (
        ("polynomial", 1.0, 4.0),
        ("polynomial", 2.0, 20.0),
        ("polynomial", 3.0, 40.0),
        ("gaussian", 1.0, 0.05),
        ("gaussian", 2.0, 0.15),
        ("gaussian", 3.0, 0.25),
    )
    xs, ys, hs, kinds, rewards = [], [], [], [], []
    problems, seeds, generations = [], [], []
    operators, multipliers, explorations = [], [], []
    # Candidates are emitted in a non-monotone order: real states receive their
    # candidate actions from a seeded sampler, and a monotone order would make
    # every same-state pair carry the same preference label.
    order = (2, 0, 3, 1, 5, 4)
    for state in range(n_states):
        bias = 0.1 * state
        for candidate in order:
            operator, multiplier, exploration = actions[candidate]
            base = 1.0 + 0.5 * candidate - 0.2 * int(operator == "gaussian")
            for horizon in horizons:
                xs.append(
                    [
                        float(state),
                        float(state * state),
                        float(bias),
                        float(horizon),
                        1.0,
                    ]
                    + _action_features(operator, multiplier, exploration)
                )
                ys.append(base * (1.0 + 0.01 * horizon) + bias)
                hs.append(int(horizon))
                kinds.append(
                    "controller"
                    if candidate == 0
                    else ("default" if candidate == 1 else "alternative")
                )
                rewards.append([base, base + 0.01 * (state + 1)])
                problems.append("zdt1")
                seeds.append(1000 + state)
                generations.append(2)
                operators.append(operator)
                multipliers.append(multiplier)
                explorations.append(exploration)
    np.savez(
        target_dir / "intervention_dataset_zdt1.npz",
        X=np.asarray(xs, dtype=np.float64),
        y_adv=np.asarray(ys, dtype=np.float64),
        horizon=np.asarray(hs, dtype=np.int64),
        candidate_kind=np.asarray(kinds, dtype=object),
        rewards=np.asarray(rewards, dtype=np.float64),
        problem=np.asarray(problems, dtype=object),
        seed=np.asarray(seeds, dtype=np.int64),
        generation=np.asarray(generations, dtype=np.int64),
        mutation_operator=np.asarray(operators, dtype=object),
        mutation_multiplier=np.asarray(multipliers, dtype=np.float64),
        exploration_strength=np.asarray(explorations, dtype=np.float64),
    )
    keys = [f"zdt1|{1000 + state}|2" for state in range(n_states)]
    split = {
        "split": {
            "unit": "state = (problem, seed, generation); shared by every target",
            "n_states_total": n_states,
            "n_states_train": n_states // 2,
            "n_states_val": n_states - n_states // 2,
            "train_state_keys": keys[: n_states // 2],
            "val_state_keys": keys[n_states // 2 :],
            "shared_across_targets": True,
        }
    }
    split_json = root / "split.json"
    split_json.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return dataset_dir, split_json


# --- CLI ---------------------------------------------------------------------


def test_cli_defaults() -> None:
    """The defaults document the intended protocol (inductive, all models)."""
    args = arr.parse_args([])
    assert args.dataset_dir == arr.DEFAULT_DATASET_DIR
    assert args.dataset_dir == "results/phase2_75d/dataset_final"
    assert args.split_json == arr.DEFAULT_SPLIT_JSON
    assert args.target == "state_mean"
    assert args.out == "results/phase2_9/action_representation.json"
    assert list(args.horizons) == [5, 10, 20]
    assert args.signature_source == "train"
    assert args.bin_edges_from == "train"
    assert args.signature_features == "full"
    assert args.models == ["linear_ols", "random_forest", "hist_gradient_boosting"]
    assert args.top_k == 3
    assert args.n_bootstrap == 200
    assert args.seed == 0
    assert args.max_train_states == 0 and args.max_val_states == 0
    assert args.skip_pairwise is False
    assert arr.GATE_HORIZON == 20
    assert arr.GATE_AUC == 0.55 and arr.GATE_KENDALL == 0.25
    assert arr.GATE_FAILURE_AUC == 0.53
    assert arr.MULTIPLIER_BINS == 4 and arr.EXPLORATION_BINS == 4
    assert list(arr.CORE_SIGNATURE_FEATURES) == [
        "loo_mean_adv",
        "loo_std_adv",
        "loo_frac_positive",
        "loo_count",
    ]
    assert set(arr.SIGNATURE_FILL) == set(arr.SIGNATURE_FEATURES)
    assert arr.SIGNATURE_FILL["loo_mean_rank"] == 0.5
    assert all(
        value == 0.0 for name, value in arr.SIGNATURE_FILL.items() if name != "loo_mean_rank"
    )


# --- action parameters -------------------------------------------------------


def test_action_parameters_uses_explicit_columns_then_x() -> None:
    """The explicit npz columns are preferred; the X block is the fallback."""
    data = _tiny_data()
    explicit = arr.action_parameters(data)
    assert list(explicit["operator"]) == (
        ["polynomial", "polynomial", "gaussian", "gaussian"] * 2
    )
    np.testing.assert_allclose(explicit["multiplier"], [1.0, 3.0, 1.0, 3.0] * 2)
    np.testing.assert_allclose(explicit["exploration"], [4.0, 40.0, 0.05, 0.25] * 2)

    fallback = arr.action_parameters(
        {key: value for key, value in data.items() if not key.startswith(("mutation_", "exploration"))}
    )
    np.testing.assert_array_equal(fallback["operator"], explicit["operator"])
    np.testing.assert_allclose(fallback["multiplier"], explicit["multiplier"])
    np.testing.assert_allclose(fallback["exploration"], explicit["exploration"])


# --- binning -----------------------------------------------------------------


def test_quantile_bin_edges_and_assign_bin_hand_computed() -> None:
    """Edges are the interior quantiles per operator and bins are equal-count."""
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    operator = np.asarray(["polynomial"] * 4 + ["gaussian"] * 4, dtype=object)
    edges = arr.quantile_bin_edges(values, operator, 4)
    assert set(edges) == {"polynomial", "gaussian"}
    np.testing.assert_allclose(edges["polynomial"], [1.75, 2.5, 3.25])
    np.testing.assert_allclose(edges["gaussian"], [17.5, 25.0, 32.5])
    bins = arr.assign_bin(values, operator, edges)
    np.testing.assert_array_equal(bins, [0, 1, 2, 3, 0, 1, 2, 3])
    # per-operator, the two groups are binned with their own edges
    separate = arr.quantile_bin_edges(
        np.asarray([1.0, 10.0]), np.asarray(["polynomial", "gaussian"], dtype=object), 4
    )
    np.testing.assert_allclose(separate["polynomial"], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="n_bins must be >= 2"):
        arr.quantile_bin_edges(values, operator, 1)
    with pytest.raises(ValueError, match="no bin edges for operator"):
        arr.assign_bin(values, operator, {"polynomial": [1.75, 2.5, 3.25]})
    # non-finite values are ignored when fitting but never crash the binning
    with_nan = arr.quantile_bin_edges(
        np.asarray([1.0, 2.0, 3.0, 4.0, np.nan]),
        np.asarray(["polynomial"] * 5, dtype=object),
        4,
    )
    assert len(with_nan["polynomial"]) == 3
    # an operator absent from the edge source yields no edges at all
    assert arr.quantile_bin_edges(
        np.asarray([1.0]), np.asarray(["polynomial"], dtype=object), 4
    ) == {"polynomial": [1.0, 1.0, 1.0]}


def test_signature_bucket_is_deterministic_and_in_range() -> None:
    """Bucket ids are bounded by 2 * 4 * 4 and identical across calls."""
    data = _tiny_data()
    parameters = arr.action_parameters(data)
    edges_multiplier = arr.quantile_bin_edges(
        parameters["multiplier"], parameters["operator"], arr.MULTIPLIER_BINS
    )
    edges_exploration = arr.quantile_bin_edges(
        parameters["exploration"], parameters["operator"], arr.EXPLORATION_BINS
    )
    first = arr.signature_bucket(
        data, multiplier_edges=edges_multiplier, exploration_edges=edges_exploration
    )
    second = arr.signature_bucket(
        data, multiplier_edges=edges_multiplier, exploration_edges=edges_exploration
    )
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.int64
    assert first.min() >= 0
    assert first.max() < 2 * arr.MULTIPLIER_BINS * arr.EXPLORATION_BINS
    # the two polynomial candidates differ, and so do the two gaussian ones
    assert len(set(first.tolist())) == 4
    # polynomial and gaussian land in disjoint halves of the bucket space
    assert first[0] < 2 * arr.MULTIPLIER_BINS * arr.EXPLORATION_BINS / 2
    assert first[2] >= 2 * arr.MULTIPLIER_BINS * arr.EXPLORATION_BINS / 2
    with pytest.raises(ValueError, match="interior quantiles"):
        arr.signature_bucket(
            data,
            multiplier_edges={"polynomial": [1.0], "gaussian": [1.0]},
            exploration_edges=edges_exploration,
        )


# --- signature contract ------------------------------------------------------


def test_signature_values_hand_computed() -> None:
    """Every feature is checked against hand-computed LOO aggregates."""
    data = _tiny_data()
    mask = np.ones(len(data["y"]), dtype=bool)
    built = _signature(data, source_mask=mask)
    features = built["features"]
    names = list(built["feature_names"])
    column = {name: features[:, index] for index, name in enumerate(names)}
    # each bucket holds exactly one row per state, so the LOO count is 1
    np.testing.assert_allclose(column["loo_count"], np.log1p(1.0))
    np.testing.assert_allclose(column["loo_std_adv"], 0.0)
    # state A rows are described by state B's outcomes (and vice versa)
    y_a = np.asarray([1.0, -2.0, 0.5, -0.5])
    y_b = np.asarray([2.0, -1.0, 0.25, -0.75])
    np.testing.assert_allclose(column["loo_mean_adv"], np.concatenate([y_b, y_a]))
    np.testing.assert_allclose(
        column["loo_frac_positive"], np.concatenate([y_b > 0, y_a > 0]).astype(float)
    )
    # ranks of the aggregated state's own outcomes. State A holds y =
    # [1.0, -2.0, 0.5, -0.5] and state B holds y = [2.0, -1.0, 0.25, -0.75];
    # both have the same within-state percentile pattern [1, 0, 2/3, 1/3].
    ranks = np.asarray([1.0, 0.0, 2.0 / 3.0, 1.0 / 3.0] * 2)
    np.testing.assert_allclose(column["loo_mean_rank"], ranks)
    # operator-restricted mean: the first two candidates are polynomial
    polynomial_b = float(np.mean(y_b[:2]))
    polynomial_a = float(np.mean(y_a[:2]))
    gaussian_b = float(np.mean(y_b[2:]))
    gaussian_a = float(np.mean(y_a[2:]))
    np.testing.assert_allclose(
        column["operator_loo_mean_adv"],
        np.concatenate(
            [[polynomial_b] * 2, [gaussian_b] * 2, [polynomial_a] * 2, [gaussian_a] * 2]
        ),
    )
    assert built["fill_fraction"] == 0.0
    # four buckets, each occupied by exactly one row per state
    assert built["bucket_sizes"]["20"] == {
        "n_buckets_occupied": 4,
        "n_buckets_total": 32,
        "rows_per_bucket_min": 2,
        "rows_per_bucket_median": 2.0,
        "rows_per_bucket_max": 2,
    }
    assert built["bucket_counts"]["20"] == 8


def test_signature_is_leave_one_state_out() -> None:
    """Tampering with one state's outcomes never moves that state's signature."""
    data = _tiny_data(horizons=(20,))
    mask = np.ones(len(data["y"]), dtype=bool)
    before = _signature(data, source_mask=mask)["features"]

    state_b = np.asarray([key == "zdt1|2|0" for key in data["state"]])
    tampered = dict(data)
    shifted = np.asarray(data["y"], dtype=np.float64).copy()
    shifted[state_b] += 100.0
    tampered["y"] = shifted
    after = _signature(tampered, source_mask=mask)["features"]

    # rows of the tampered state are unchanged...
    np.testing.assert_allclose(after[state_b], before[state_b])
    # ...while every other state's signature moves
    assert not np.allclose(after[~state_b], before[~state_b])
    # the untouched state's loo_mean_adv is exactly the tampered state's outcomes
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(after[~state_b][0], before[~state_b][0])
    assert after[~state_b][0, 0] == pytest.approx(102.0)


def test_signature_training_source_excludes_validation_states() -> None:
    """With an inductive source mask, val-state outcomes never enter a feature."""
    data = _tiny_data(horizons=(20,))
    state_b = np.asarray([key == "zdt1|2|0" for key in data["state"]])
    source = ~state_b  # only the first state may be aggregated
    mask = np.ones(len(data["y"]), dtype=bool)
    features = _signature(data, source_mask=source, loo_mask=mask)["features"]
    # the held-out state is described by the source state's outcomes...
    np.testing.assert_allclose(
        features[state_b][:, 0], np.asarray([1.0, -2.0, 0.5, -0.5])
    )
    # ...while the source state's own rows have nothing left to aggregate
    empty = np.asarray([arr.SIGNATURE_FILL[name] for name in arr.SIGNATURE_FEATURES])
    np.testing.assert_allclose(features[~state_b], np.tile(empty, (4, 1)))
    # tampering with the never-aggregated state leaves every feature identical
    tampered = dict(data)
    shifted = np.asarray(data["y"], dtype=np.float64).copy()
    shifted[state_b] += 50.0
    tampered["y"] = shifted
    np.testing.assert_allclose(
        _signature(tampered, source_mask=source, loo_mask=mask)["features"], features
    )


def test_signature_empty_bucket_uses_fill_not_global_mean() -> None:
    """A bucket without aggregated rows is filled by constants, not an average."""
    data = _tiny_data(horizons=(20,))
    mask = np.ones(len(data["y"]), dtype=bool)
    state_a = np.asarray([key == "zdt1|1|0" for key in data["state"]])
    built = _signature(data, source_mask=state_a)
    features = built["features"]
    names = list(built["feature_names"])
    column = {name: features[:, index] for index, name in enumerate(names)}
    # the second state has no source rows at all -> fully filled
    np.testing.assert_allclose(column["loo_mean_adv"][~state_a], 0.0)
    np.testing.assert_allclose(column["loo_std_adv"][~state_a], 0.0)
    np.testing.assert_allclose(column["loo_frac_positive"][~state_a], 0.0)
    np.testing.assert_allclose(column["loo_count"][~state_a], 0.0)
    np.testing.assert_allclose(column["loo_mean_rank"][~state_a], 0.5)
    np.testing.assert_allclose(column["operator_loo_mean_adv"][~state_a], 0.0)
    # the global mean of the source state is explicitly *not* used
    assert float(np.mean(np.asarray(data["y"])[state_a])) != 0.0
    # source-state rows are empty too (their own state is excluded)
    np.testing.assert_allclose(column["loo_mean_rank"][state_a], 0.5)
    assert built["fill_fraction"] == 1.0


def test_signature_is_horizon_specific_and_order_invariant() -> None:
    """Same bucket/horizon semantics, independent of row order."""
    data = _tiny_data(horizons=(5, 20), horizon_scale=0.01)
    mask = np.ones(len(data["y"]), dtype=bool)
    built = _signature(data, source_mask=mask, horizons=(5, 20))
    features = built["features"]
    horizon = np.asarray(data["horizon"])
    # the two horizons carry different advantages -> different signature values
    assert not np.allclose(features[horizon == 5][0], features[horizon == 20][0])
    # a permutation of the rows permutes the features identically
    order = np.asarray([3, 0, 7, 1, 6, 2, 5, 4, 11, 8, 15, 9, 14, 10, 13, 12])
    shuffled = {
        key: (
            [value[index] for index in order]
            if key == "state"
            else np.asarray(value)[order]
            if isinstance(value, np.ndarray)
            else value
        )
        for key, value in data.items()
    }
    rebuilt = _signature(shuffled, source_mask=mask, horizons=(5, 20))["features"]
    np.testing.assert_allclose(rebuilt, features[order], atol=1e-12)


def test_within_state_ranks_hand_computed() -> None:
    """Ranks are within-state percentiles in [0, 1]; singletons stay neutral."""
    data = _tiny_data(horizons=(20,))
    ranks = arr.within_state_ranks(data, (20,))
    np.testing.assert_allclose(ranks, [1.0, 0.0, 2 / 3, 1 / 3] * 2)
    singleton = _tiny_data(horizons=(7,))
    np.testing.assert_allclose(arr.within_state_ranks(singleton, (3,)), np.full(8, 0.5))

# --- variants and metrics ----------------------------------------------------


def test_variant_matrices_shapes() -> None:
    """Each variant stacks exactly the documented columns."""
    data = _tiny_data(horizons=(5, 20))
    mask = np.ones(len(data["y"]), dtype=bool)
    signature = _signature(data, source_mask=mask, horizons=(5, 20))["features"]
    matrices = arr.variant_matrices(data, signature)
    assert matrices["action_only"].shape == (16, 4)
    assert matrices["action_plus_signature"].shape == (16, 4 + 6)
    assert matrices["concat"].shape == (16, data["X"].shape[1])
    assert matrices["concat_plus_signature"].shape == (16, data["X"].shape[1] + 6)
    assert matrices["signature_only"].shape == (16, 6)
    np.testing.assert_allclose(
        matrices["action_only"], data["X"][:, -arr.ACTION_FEATURES:]
    )
    np.testing.assert_allclose(
        matrices["concat_plus_signature"][:, -6:], signature
    )
    # the core subset drops only the two optional columns
    core = _signature(
        data,
        source_mask=mask,
        horizons=(5, 20),
        feature_names=arr.CORE_SIGNATURE_FEATURES,
    )["features"]
    np.testing.assert_allclose(core, signature[:, :4])
    assert list(arr.VARIANTS) == [
        "action_only",
        "action_plus_signature",
        "concat",
        "concat_plus_signature",
        "signature_only",
    ]


def test_topk_hit_rate_hand_computed() -> None:
    """Top-k hit rate follows the argmax of the realised advantage."""
    data = _tiny_data(horizons=(20,))
    # State A: scores [0.5, 0.4, 0.1, 0.2] put the true best (candidate 0) first.
    # State B: scores [0.2, 0.9, 0.5, 0.1] put the true best (candidate 0, the
    # largest advantage) third, so it is missed at k=1 and hit at k=3.
    scores = np.asarray([0.5, 0.4, 0.1, 0.2, 0.2, 0.9, 0.5, 0.1])
    assert arr._topk_hit_rate(scores, data, 20, data["state"], 1) == pytest.approx(0.5)
    assert arr._topk_hit_rate(scores, data, 20, data["state"], 2) == pytest.approx(0.5)
    assert arr._topk_hit_rate(scores, data, 20, data["state"], 3) == pytest.approx(1.0)
    assert arr._topk_hit_rate(scores, data, 99, data["state"], 1) is None


def test_pairwise_auc_ci_is_none_without_pairs() -> None:
    """A horizon without validation pairs reports an empty CI record."""
    data = _tiny_data(horizons=(20,))
    matrix = np.asarray(data["X"])
    is_val = np.asarray([key == "zdt1|2|0" for key in data["state"]])
    record = arr.pairwise_auc_ci(
        matrix, data, is_val, 5, seed=0, n_bootstrap=10
    )
    assert record == {
        "auc": None,
        "accuracy": None,
        "n_pairs": 0,
        "n_states": 0,
        "ci95": None,
        "n_bootstrap": 0,
    }


def test_pairwise_auc_ci_on_a_separating_signal() -> None:
    """A perfectly ordered signal yields AUC 1.0 and a degenerate CI."""
    rng = np.random.Generator(np.random.PCG64(0))
    states, xs, ys, hs, kinds = [], [], [], [], []
    for state in range(6):
        # the candidate order is deliberately not monotone, otherwise every
        # same-state pair would carry the same preference label
        for candidate in (2, 0, 3, 1):
            states.append(f"zdt1|{state}|0")
            xs.append([float(candidate), 1.0, float(state)])
            # advantage is a strictly monotone function of candidate index
            ys.append(float(candidate) + 0.001 * rng.standard_normal())
            hs.append(20)
            kinds.append("alternative")
    data = {
        "X": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "horizon": np.asarray(hs, dtype=np.int64),
        "kind": np.asarray(kinds, dtype=object),
        "rewards": np.zeros((24, 2)),
        "state": states,
    }
    is_val = np.asarray([int(state.split("|")[1]) >= 3 for state in states])
    record = arr.pairwise_auc_ci(
        data["X"], data, is_val, 20, seed=0, n_bootstrap=50
    )
    assert record["auc"] == pytest.approx(1.0)
    assert record["accuracy"] == pytest.approx(1.0)
    assert record["n_states"] == 3
    assert record["ci95"] == [1.0, 1.0]
    assert record["n_bootstrap"] == 50


# --- gate --------------------------------------------------------------------


def _pairwise_entry(
    auc: float | None,
    kendall: float | None,
    *,
    ci95: list[float] | None = None,
    model: str = "logistic",
) -> dict[str, Any]:
    """One variant's pairwise record for the crafted gate tests."""
    models = {}
    if auc is not None:
        models[model] = {
            "pairwise_accuracy": 0.5,
            "pairwise_auc": auc,
            "within_state_kendall_from_wins": kendall,
            "n_states_ranked": 10,
        }
    entry: dict[str, Any] = {
        "20": {"n_train_pairs": 100, "n_val_pairs": 40, "models": models},
    }
    if ci95 is not None:
        entry["auc_ci"] = {"20": {"auc": auc, "ci95": ci95, "n_bootstrap": 200}}
    return entry


def test_gate_verdicts() -> None:
    """The three verdicts and the CI/Kendall guard rails."""
    baseline = _pairwise_entry(0.50, 0.15)
    success = arr.evaluate_gate(
        {
            "action_only": baseline,
            "action_plus_signature": _pairwise_entry(0.60, 0.30, ci95=[0.52, 0.68]),
        }
    )
    assert success["verdict"] == "representation_is_bottleneck"
    assert success["numbers"]["auc_delta"] == pytest.approx(0.10)
    assert success["numbers"]["kendall_delta"] == pytest.approx(0.15)
    assert success["candidate"]["model"] == "logistic"
    assert "0.600 > 0.55" in success["evidence"]

    failure = arr.evaluate_gate(
        {
            "action_only": baseline,
            "action_plus_signature": _pairwise_entry(0.52, 0.30, ci95=[0.48, 0.56]),
        }
    )
    assert failure["verdict"] == "representation_not_sufficient"

    indeterminate = arr.evaluate_gate(
        {
            "action_only": baseline,
            "action_plus_signature": _pairwise_entry(0.54, 0.30),
        }
    )
    assert indeterminate["verdict"] == "indeterminate"

    # a high AUC with a weak rank correlation is not a success
    weak_kendall = arr.evaluate_gate(
        {
            "action_only": baseline,
            "action_plus_signature": _pairwise_entry(0.60, 0.20),
        }
    )
    assert weak_kendall["verdict"] == "indeterminate"

    # a CI whose lower bound touches 0.50 is not a success either
    straddling = arr.evaluate_gate(
        {
            "action_only": baseline,
            "action_plus_signature": _pairwise_entry(0.60, 0.40, ci95=[0.50, 0.70]),
        }
    )
    assert straddling["verdict"] == "indeterminate"
    assert "0.500" not in straddling["evidence"]

    # without a CI the AUC/Kendall thresholds alone decide
    no_ci = arr.evaluate_gate(
        {
            "action_only": baseline,
            "action_plus_signature": _pairwise_entry(0.58, 0.40),
        }
    )
    assert no_ci["verdict"] == "representation_is_bottleneck"
    assert "CI" not in no_ci["evidence"]

    # the best model by AUC is the one judged, and its Kendall is the one used
    multi = _pairwise_entry(0.50, 0.90, model="random_forest")
    multi["20"]["models"]["logistic"] = {
        "pairwise_accuracy": 0.5,
        "pairwise_auc": 0.62,
        "within_state_kendall_from_wins": 0.10,
        "n_states_ranked": 10,
    }
    picked = arr.evaluate_gate(
        {"action_only": baseline, "action_plus_signature": multi}
    )
    assert picked["candidate"]["model"] == "logistic"
    assert picked["verdict"] == "indeterminate"  # best AUC but Kendall 0.10

    # a missing candidate record is indeterminate, not a crash
    missing = arr.evaluate_gate({})
    assert missing["verdict"] == "indeterminate"
    assert missing["numbers"]["candidate"]["auc"] is None
    assert missing["numbers"]["auc_delta"] is None
    # an off-horizon evaluation is equally indeterminate
    assert arr.evaluate_gate(
        {"action_plus_signature": _pairwise_entry(0.9, 0.9)}, horizon=10
    )["verdict"] == "indeterminate"


# --- end to end --------------------------------------------------------------


def test_run_experiment_writes_the_documented_artifact(tmp_path: Path) -> None:
    """The driver writes the full schema and is byte-stable across re-runs."""
    dataset_dir, split_json = _write_fixture(tmp_path)
    out_one = tmp_path / "out" / "one.json"
    out_two = tmp_path / "out" / "two.json"
    argv = [
        "--dataset-dir",
        str(dataset_dir),
        "--split-json",
        str(split_json),
        "--out",
        str(out_one),
        "--horizons",
        "5",
        "20",
        "--models",
        "linear_ols",
        "--n-bootstrap",
        "10",
        "--top-k",
        "2",
        "--seed",
        "7",
    ]
    payload = _run(argv)
    assert out_one.is_file()

    assert set(payload) == {
        "config",
        "data",
        "signature",
        "variants",
        "pairwise",
        "gate",
        "wall_time_sec",
    }
    config = payload["config"]
    assert config["horizons"] == [5, 20]
    assert config["signature_source"] == "train"
    assert config["bin_edges_from"] == "train"
    assert config["models"] == ["linear_ols"]
    assert config["seed"] == 7
    assert config["variants"] == list(arr.VARIANTS)
    assert "Phase-2.8 audit protocol" in config["protocol"]

    data_record = payload["data"]
    # 8 states x 6 candidates x 2 horizons = 96 rows, of which 16 are controller
    assert data_record["n_rows"] == 96 - 16
    assert data_record["controller_rows_excluded"] == 16
    assert data_record["n_states"] == 8
    assert data_record["n_train_states"] == 4
    assert data_record["n_val_states"] == 4
    assert data_record["n_train_rows"] == 40
    assert data_record["n_val_rows"] == 40
    assert data_record["split_unit"].startswith("state = (problem, seed, generation)")

    signature = payload["signature"]
    assert signature["n_buckets"] == 32
    assert signature["multiplier_bins"] == 4 and signature["exploration_bins"] == 4
    assert set(signature["multiplier_edges"]) == {"polynomial", "gaussian"}
    assert all(
        len(edges) == 3 for edges in signature["multiplier_edges"].values()
    )
    assert signature["fill_fraction"] == pytest.approx(0.0)
    assert signature["bucket_sizes_by_horizon"]["20"]["n_buckets_occupied"] > 0
    # 4 training states x 5 non-controller candidates
    assert signature["bucket_counts_by_horizon"]["20"] == 20
    assert signature["bucket_counts_by_horizon"]["5"] == 20
    assert signature["definition"].startswith("per (state, action, horizon)")

    assert set(payload["variants"]) == set(arr.VARIANTS)
    assert payload["variants"]["action_only"]["dim"] == 4
    assert payload["variants"]["concat"]["dim"] == payload["variants"]["action_only"]["dim"] + 5
    assert payload["variants"]["action_plus_signature"]["dim"] == 10
    assert payload["variants"]["signature_only"]["dim"] == 6
    for variant in arr.VARIANTS:
        metrics = payload["variants"][variant]["models"]["linear_ols"]["per_horizon"]
        assert set(metrics) == {"5", "20"}
        assert "top2_hit_rate" in metrics["20"]
        assert "spearman_mean" in metrics["20"]
        assert "regret_mean" in metrics["20"]

    assert set(payload["pairwise"]) == set(arr.VARIANTS)
    for variant in arr.VARIANTS:
        assert payload["pairwise"][variant]["20"]["n_val_pairs"] > 0
        assert set(payload["pairwise"][variant]["20"]["models"]) == {
            "logistic",
            "random_forest",
        }
    for variant in ("action_only", "action_plus_signature", "signature_only"):
        ci = payload["pairwise"][variant]["auc_ci"]["20"]
        assert ci["n_pairs"] > 0
        assert ci["ci95"] is not None
        assert ci["ci95"][0] <= ci["auc"] <= ci["ci95"][1]
    assert "auc_ci" not in payload["pairwise"]["concat"]
    assert payload["gate"]["numbers"]["horizon"] == 20
    assert payload["gate"]["numbers"]["candidate"]["auc"] is not None
    assert payload["gate"]["verdict"] in {
        "representation_is_bottleneck",
        "representation_not_sufficient",
        "indeterminate",
    }

    # the same run written to a second path must be identical apart from timing
    repeat = list(argv)
    repeat[5] = str(out_two)
    _run(repeat)
    first = json.loads(out_one.read_text(encoding="utf-8"))
    second = json.loads(out_two.read_text(encoding="utf-8"))
    first.pop("wall_time_sec")
    second.pop("wall_time_sec")
    assert first == second


def test_run_experiment_honours_state_caps_and_skip_pairwise(tmp_path: Path) -> None:
    """Caps shrink the split consistently and ``--skip-pairwise`` degrades safely."""
    dataset_dir, split_json = _write_fixture(tmp_path)
    out = tmp_path / "capped.json"
    payload = _run(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--split-json",
            str(split_json),
            "--out",
            str(out),
            "--horizons",
            "20",
            "--models",
            "linear_ols",
            "--max-train-states",
            "2",
            "--max-val-states",
            "1",
            "--skip-pairwise",
        ]
    )
    assert payload["data"]["n_train_states"] == 2
    assert payload["data"]["n_val_states"] == 1
    # 2 capped training states and 1 capped validation state, 5 candidates each
    # and 2 horizons in the dataset (--horizons only selects what is evaluated)
    assert payload["data"]["n_train_rows"] == 20
    assert payload["data"]["n_val_rows"] == 10
    assert payload["data"]["n_rows"] == 30
    assert payload["pairwise"] == {}
    assert payload["gate"]["verdict"] == "indeterminate"
    assert "no pairwise AUC" in payload["gate"]["evidence"]
    # the signature is still built for both splits
    assert payload["signature"]["bucket_counts_by_horizon"]["20"] == 10
    assert payload["signature"]["fill_fraction"] == 0.0
    assert set(payload["variants"]["action_only"]["models"]) == {"linear_ols"}


def test_run_experiment_signature_source_all_is_a_transductive_upper_bound(
    tmp_path: Path,
) -> None:
    """``--signature-source all`` aggregates validation states too."""
    dataset_dir, split_json = _write_fixture(tmp_path)
    common = [
        "--dataset-dir",
        str(dataset_dir),
        "--split-json",
        str(split_json),
        "--horizons",
        "20",
        "--models",
        "linear_ols",
        "--n-bootstrap",
        "0",
    ]
    train_only = arr.run_experiment(
        arr.parse_args(common + ["--out", str(tmp_path / "t.json"), "--skip-pairwise"])
    )
    all_rows = arr.run_experiment(
        arr.parse_args(
            common
            + [
                "--out",
                str(tmp_path / "a.json"),
                "--skip-pairwise",
                "--signature-source",
                "all",
                "--bin-edges-from",
                "all",
            ]
        )
    )
    assert train_only["signature"]["bucket_counts_by_horizon"]["20"] == 20
    assert all_rows["signature"]["bucket_counts_by_horizon"]["20"] == 40
    assert train_only["config"]["signature_source"] == "train"
    assert all_rows["config"]["signature_source"] == "all"
    # the inductive run aggregates one row per training state into a bucket, the
    # transductive one aggregates one row per state of both splits
    assert (
        train_only["signature"]["bucket_sizes_by_horizon"]["20"][
            "rows_per_bucket_median"
        ]
        == 4.0
    )
    assert (
        all_rows["signature"]["bucket_sizes_by_horizon"]["20"][
            "rows_per_bucket_median"
        ]
        == 8.0
    )
    assert train_only["signature"]["fill_fraction"] == 0.0
    assert all_rows["signature"]["fill_fraction"] == 0.0


def test_run_experiment_core_feature_subset(tmp_path: Path) -> None:
    """``--signature-features core`` emits four columns and skips the CI stage."""
    dataset_dir, split_json = _write_fixture(tmp_path)
    payload = _run(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--split-json",
            str(split_json),
            "--out",
            str(tmp_path / "core.json"),
            "--horizons",
            "20",
            "--models",
            "linear_ols",
            "--signature-features",
            "core",
            "--n-bootstrap",
            "0",
        ]
    )
    assert payload["signature"]["feature_names"] == list(arr.CORE_SIGNATURE_FEATURES)
    assert payload["variants"]["action_plus_signature"]["dim"] == 4 + 4
    assert payload["variants"]["signature_only"]["dim"] == 4
    # n_bootstrap=0 keeps the AUC but reports no interval
    ci = payload["pairwise"]["action_plus_signature"]["auc_ci"]["20"]
    assert ci["auc"] is not None
    assert ci["ci95"] is None
    assert ci["n_bootstrap"] == 0
    assert payload["gate"]["verdict"] in {
        "representation_is_bottleneck",
        "representation_not_sufficient",
        "indeterminate",
    }
