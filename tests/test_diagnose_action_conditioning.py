from __future__ import annotations

"""Tests for the Phase-2B action-conditioning diagnostic.

Everything here is synthetic: the diagnostics are pure functions over
``(X, y)`` matrices and per-state records, so the tests can construct an
"action-ignoring" and an "action-dependent" world with known answers and
assert that D1-D5 plus the declared conclusion rule point the right way.
No trained model, no trajectory file and no NSGA-II run is touched, so the
whole module stays fast (no torch import).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments import diagnose_action_conditioning as dac

_N_STATE_FEATURES = 6
_ACTION_FEATURES = dac.N_ACTION_FEATURES


# --- synthetic worlds --------------------------------------------------------


def _sample_matrix(n: int = 60, seed: int = 0) -> np.ndarray:
    """A random ``(n, 6 + 4)`` sample matrix with a stable state block."""
    rng = np.random.Generator(np.random.PCG64(seed))
    state = rng.normal(size=(n, _N_STATE_FEATURES))
    action = np.column_stack(
        [
            rng.uniform(0.25, 8.0, size=n),  # mutation multiplier
            rng.uniform(2.0, 50.0, size=n),  # exploration
            rng.integers(0, 2, size=n).astype(float),  # onehot polynomial
        ]
    )
    action = np.column_stack([action, 1.0 - action[:, -1]])
    return np.column_stack([state, action])


def _state_only_model(X: np.ndarray) -> np.ndarray:
    """A predictor that ignores the action block entirely."""
    state = X[:, :-_ACTION_FEATURES]
    return (2.0 * state[:, 0] - state[:, 1] + 0.5).reshape(-1, 1)


def _action_dependent_model(X: np.ndarray) -> np.ndarray:
    """A predictor whose output is dominated by the action multiplier."""
    multiplier = X[:, -_ACTION_FEATURES]
    return (3.0 * multiplier).reshape(-1, 1)


# --- D1 ----------------------------------------------------------------------


def test_ablate_action_block_builds_documented_variants() -> None:
    """zero/mean/shuffled act on the last four columns only."""
    X = _sample_matrix()
    variants = dac.ablate_action_block(X, seed=3)
    assert set(variants) == {"baseline", "zero", "mean", "shuffled"}
    for matrix in variants.values():
        assert matrix.shape == X.shape
    # baseline is a copy of the input; state block is never touched.
    np.testing.assert_array_equal(variants["baseline"], X)
    np.testing.assert_array_equal(variants["zero"][:, :-4], X[:, :-4])
    np.testing.assert_array_equal(variants["shuffled"][:, :-4], X[:, :-4])
    # zero variant really is zero.
    np.testing.assert_array_equal(variants["zero"][:, -4:], np.zeros_like(X[:, -4:]))
    # mean variant is the per-column mean.
    np.testing.assert_allclose(
        variants["mean"][:, -4:],
        np.tile(X[:, -4:].mean(axis=0), (X.shape[0], 1)),
    )
    # shuffling preserves the multiset of action rows, destroys the pairing.
    original = {tuple(row) for row in np.round(X[:, -4:], 12)}
    shuffled = {tuple(row) for row in np.round(variants["shuffled"][:, -4:], 12)}
    assert original == shuffled
    assert not np.array_equal(variants["shuffled"][:, -4:], X[:, -4:])
    # same seed -> same permutation.
    again = dac.ablate_action_block(X, seed=3)
    np.testing.assert_array_equal(again["shuffled"], variants["shuffled"])


def test_ablate_action_block_rejects_narrow_input() -> None:
    """Too few columns for an action block is an error."""
    with pytest.raises(ValueError, match="columns"):
        dac.ablate_action_block(np.zeros((4, 3)))


def test_d1_r2_unchanged_for_action_ignoring_model() -> None:
    """If the model reads state only, ablating the action changes nothing."""
    X = _sample_matrix()
    y = _state_only_model(X)
    d1 = dac.d1_action_ablation(_state_only_model, X, y, seed=1)
    assert set(d1) == {"zero", "mean", "shuffled", "baseline"}
    assert d1["baseline"]["r2"] == pytest.approx(1.0)
    for name in ("zero", "mean", "shuffled"):
        assert d1[name]["r2"] == pytest.approx(1.0)
        assert d1[name]["mse"] == pytest.approx(d1["baseline"]["mse"])
    conclusion = dac.conclude(d1, None)
    assert conclusion["action_information_used"] is False
    assert "not backed by" in conclusion["evidence"]


def test_d1_r2_collapses_for_action_dependent_model() -> None:
    """If the model reads the action, ablating it destroys R^2."""
    X = _sample_matrix()
    y = _action_dependent_model(X)
    d1 = dac.d1_action_ablation(_action_dependent_model, X, y, seed=1)
    assert d1["baseline"]["r2"] == pytest.approx(1.0)
    for name in ("zero", "mean", "shuffled"):
        assert d1[name]["r2"] < 0.5
    assert d1["baseline"]["r2"] - max(
        d1[name]["r2"] for name in ("zero", "mean", "shuffled")
    ) > dac.ABLATION_R2_DROP_THRESHOLD
    assert dac.conclude(d1, None)["action_information_used"] is True


def test_regression_metrics_undefined_r2_for_constant_truth() -> None:
    """R^2 is undefined (None) when the ground truth has no variance."""
    metrics = dac.regression_metrics([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert metrics["r2"] is None
    assert metrics["mse"] == pytest.approx((0.0 + 1.0 + 4.0) / 3.0)
    with pytest.raises(ValueError, match="entries"):
        dac.regression_metrics([1.0, 2.0], [1.0])


# --- D2/D3/D4/D5 cores -------------------------------------------------------


def test_ranking_correlations_direction_and_undefined_cases() -> None:
    """Spearman/Kendall point the right way and refuse degenerate input."""
    same = dac.ranking_correlations([1, 2, 3, 4], [1, 2, 3, 4])
    assert same[0] == pytest.approx(1.0)
    assert same[1] == pytest.approx(1.0)
    assert same[2] is not None and same[2] < 0.05
    reversed_ = dac.ranking_correlations([1, 2, 3, 4], [4, 3, 2, 1])
    assert reversed_[0] == pytest.approx(-1.0)
    assert reversed_[1] == pytest.approx(-1.0)
    # too few candidates / constant inputs -> undefined
    assert dac.ranking_correlations([1, 2], [1, 2]) == (None, None, None)
    assert dac.ranking_correlations([1, 1, 1, 1], [1, 2, 3, 4]) == (None, None, None)
    assert dac.ranking_correlations([1, 2, 3, 4], [2, 2, 2, 2]) == (None, None, None)


def test_top1_regret_zero_when_best_is_chosen() -> None:
    """Regret is 0 for the oracle action and positive for a bad pick."""
    realized = [0.1, 0.5, 0.3]
    assert dac.top1_regret([0.1, 0.9, 0.5], realized) == pytest.approx(0.0)
    assert dac.top1_regret([0.9, 0.1, 0.5], realized) == pytest.approx(0.4)
    # ties resolve to the earliest candidate (np.argmax semantics)
    assert dac.top1_regret([0.2, 0.2, 0.2], realized) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        dac.top1_regret([], [])


def test_action_effect_snr_separates_signal_from_noise() -> None:
    """Between-action variance dominates when the action really matters."""
    strong = np.asarray([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
    between, within, snr = dac.action_effect_snr([strong])
    assert between == pytest.approx(float(np.var([0.0, 1.0, 2.0, 3.0], ddof=1)))
    assert within == pytest.approx(0.005)
    assert snr == pytest.approx(between / within)
    assert snr > 100.0
    # identical candidates: no between-action variance at all
    flat = np.tile(np.asarray([[1.0, 1.2, 0.8]]), (3, 1))
    between_flat, within_flat, snr_flat = dac.action_effect_snr([flat])
    assert between_flat == pytest.approx(0.0)
    assert within_flat > 0.0
    assert snr_flat == pytest.approx(0.0)
    # unusable shapes are skipped, and an all-skipped input yields Nones
    assert dac.action_effect_snr([np.zeros((1, 3)), np.zeros((3, 1))]) == (
        None,
        None,
        None,
    )


def test_linear_fit_recovers_a_perfect_line() -> None:
    """Realized = 2 * predicted + 1 is recovered exactly."""
    predicted = [0.0, 1.0, 2.0, 3.0]
    realized = [1.0, 3.0, 5.0, 7.0]
    pearson, slope, intercept = dac.linear_fit(predicted, realized)
    assert pearson == pytest.approx(1.0)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)
    assert dac.linear_fit([1.0, 1.0], [1.0, 2.0]) == (None, None, None)


def test_within_group_ranking_follows_the_realized_order() -> None:
    """Group-level Spearman is positive when predictions track outcomes."""
    X = _sample_matrix(n=12, seed=5)
    # realized future HV: 4 horizons whose weighted sum matches the
    # predictor below within every group.
    y = np.column_stack(
        [
            np.arange(12, dtype=float) * 0.1,
            np.arange(12, dtype=float) * 0.2,
            np.arange(12, dtype=float) * 0.3,
            np.arange(12, dtype=float) * 0.4,
        ]
    )

    def aligned(X_in: np.ndarray) -> np.ndarray:
        # same ordering as y: proportional to the sample's row position.
        order = np.arange(X_in.shape[0], dtype=float)
        return np.column_stack([order, order, order, order])

    groups = {"g0": [0, 1, 2, 3], "g1": [4, 5, 6, 7], "g2": [8, 9, 10, 11]}
    forward = dac.within_group_ranking(aligned, X, y, groups)
    assert forward["n_groups"] == 3
    assert forward["spearman_mean"] == pytest.approx(1.0)
    assert forward["kendall_mean"] == pytest.approx(1.0)
    assert forward["significant_fraction"] == pytest.approx(1.0)

    def reversed_model(X_in: np.ndarray) -> np.ndarray:
        order = -np.arange(X_in.shape[0], dtype=float)
        return np.column_stack([order, order, order, order])

    backward = dac.within_group_ranking(reversed_model, X, y, groups)
    assert backward["spearman_mean"] == pytest.approx(-1.0)


def test_within_group_ranking_skips_small_groups() -> None:
    """Groups below the minimum size contribute nothing."""
    X = _sample_matrix(n=4, seed=6)
    y = np.ones((4, 4))
    groups = {"tiny": [0, 1], "ok": [0, 1, 2, 3]}
    summary = dac.within_group_ranking(lambda Z: np.ones((Z.shape[0], 4)), X, y, groups)
    assert summary["n_groups"] == 0
    assert summary["spearman_mean"] is None


# --- strict statistics and artifact schema -----------------------------------


def _strict_state(
    *,
    hv_before: float = 0.0,
    scores: list[float] | None = None,
    means: list[float] | None = None,
    replicate_offset: float = 0.1,
) -> dict[str, Any]:
    """One synthetic strict record: 4 candidates x 2 replicates."""
    scores = [0.0, 1.0, 2.0, 3.0] if scores is None else scores
    means = [0.0, 1.0, 2.0, 3.0] if means is None else means
    realized = np.asarray(
        [[mean, mean + replicate_offset] for mean in means], dtype=float
    )
    predicted = np.asarray(
        [[score, score, score, score] for score in scores], dtype=float
    )
    return {
        "scores": np.asarray(scores, dtype=float),
        "predicted": predicted,
        "realized": realized,
        "hv_before": float(hv_before),
    }


def test_strict_statistics_reports_every_diagnostic() -> None:
    """A perfectly ranked state yields rho=1, zero regret and known SNR."""
    records = [_strict_state()]
    stats = dac.strict_statistics(records, horizon_column=1)
    assert set(stats) == {"d2_strict", "d3", "d4", "d5"}
    strict = stats["d2_strict"]
    assert strict["n_states"] == 1
    assert strict["n_states_scored"] == 1
    assert strict["n_states_excluded_degenerate"] == 0
    assert strict["spearman_mean"] == pytest.approx(1.0)
    assert strict["kendall_mean"] == pytest.approx(1.0)
    assert strict["spearman_mean_horizon_column"] == pytest.approx(1.0)
    assert strict["significant_fraction"] == pytest.approx(1.0)
    assert strict["top1_selection_agreement"] == pytest.approx(1.0)
    assert set(stats["d3"]) == {"mean", "median", "std", "n_states"}
    assert stats["d3"]["mean"] == pytest.approx(0.0)
    assert stats["d3"]["n_states"] == 1
    assert set(stats["d4"]) == {
        "between_action_var",
        "within_action_noise_var",
        "snr",
    }
    assert stats["d4"]["snr"] == pytest.approx(
        stats["d4"]["between_action_var"] / stats["d4"]["within_action_noise_var"]
    )
    assert stats["d4"]["snr"] > 100.0
    assert set(stats["d5"]) == {"pearson_r", "slope", "intercept"}
    assert stats["d5"]["pearson_r"] == pytest.approx(1.0)
    assert stats["d5"]["slope"] == pytest.approx(1.0)


def test_strict_statistics_penalises_reversed_ranking() -> None:
    """A predictor ranking the candidates backwards shows negative rho."""
    records = [_strict_state(scores=[3.0, 2.0, 1.0, 0.0])]
    stats = dac.strict_statistics(records, horizon_column=1)
    assert stats["d2_strict"]["spearman_mean"] == pytest.approx(-1.0)
    assert stats["d2_strict"]["top1_selection_agreement"] == pytest.approx(0.0)
    assert stats["d3"]["mean"] == pytest.approx(3.0)
    assert stats["d5"]["pearson_r"] == pytest.approx(-1.0)
    assert dac.conclude(
        {"baseline": {"r2": 1.0, "mse": 0.0},
         "zero": {"r2": 1.0, "mse": 0.0},
         "mean": {"r2": 1.0, "mse": 0.0},
         "shuffled": {"r2": 1.0, "mse": 0.0}},
        stats["d2_strict"],
    )["action_information_used"] is False


def test_strict_statistics_excludes_degenerate_states() -> None:
    """States where every candidate ties carry no signal and are excluded."""
    records = [
        _strict_state(),
        _strict_state(means=[1.0, 1.0, 1.0, 1.0], replicate_offset=0.0),
    ]
    stats = dac.strict_statistics(records, horizon_column=1)
    strict = stats["d2_strict"]
    assert strict["n_states"] == 2
    assert strict["n_states_scored"] == 1
    assert strict["n_states_excluded_degenerate"] == 1
    assert stats["d3"]["n_states"] == 1
    # the excluded state must not dilute the regret to zero
    assert stats["d3"]["mean"] == pytest.approx(0.0)


def test_conclude_uses_the_strict_ranking_when_present() -> None:
    """Ablation flat but strong within-state ranking -> action is used."""
    flat_d1 = {
        name: {"r2": 1.0, "mse": 0.0}
        for name in ("baseline", "zero", "mean", "shuffled")
    }
    strong = {"spearman_mean": 0.6, "significant_fraction": 0.9}
    weak = {"spearman_mean": 0.05, "significant_fraction": 0.2}
    assert dac.conclude(flat_d1, strong)["action_information_used"] is True
    assert dac.conclude(flat_d1, weak)["action_information_used"] is False
    assert dac.conclude(flat_d1, None)["action_information_used"] is False
    assert "Spearman" in dac.conclude(flat_d1, strong)["evidence"]


def test_build_payload_has_the_documented_schema() -> None:
    """The artifact carries exactly the contracted keys at every level."""
    payload = dac.build_payload(
        d1={
            name: {"r2": 1.0, "mse": 0.0}
            for name in ("zero", "mean", "shuffled", "baseline")
        },
        d2={
            "n_groups": 3,
            "spearman_mean": 0.1,
            "spearman_std": 0.2,
            "kendall_mean": 0.05,
            "significant_fraction": 0.0,
            "strict_version": None,
        },
        d3={"mean": 0.0, "median": 0.0, "std": 0.0, "n_states": 1},
        d4={"between_action_var": 1.0, "within_action_noise_var": 0.1, "snr": 10.0},
        d5={"pearson_r": 0.5, "slope": 1.0, "intercept": 0.0},
        conclusion={"action_information_used": False, "evidence": "e"},
        config={"model_dir": "m"},
    )
    assert set(payload) == {
        "d1_action_ablation",
        "d2_within_state_ranking",
        "d3_top1_regret",
        "d4_action_effect_snr",
        "d5_pred_vs_actual",
        "conclusion",
        "config",
    }
    assert set(payload["d1_action_ablation"]) == {
        "zero",
        "mean",
        "shuffled",
        "baseline",
    }
    assert all(
        set(entry) == {"r2", "mse"}
        for entry in payload["d1_action_ablation"].values()
    )
    assert set(payload["d2_within_state_ranking"]) == {
        "n_groups",
        "spearman_mean",
        "spearman_std",
        "kendall_mean",
        "significant_fraction",
        "strict_version",
    }
    assert set(payload["d3_top1_regret"]) == {"mean", "median", "std", "n_states"}
    assert set(payload["d4_action_effect_snr"]) == {
        "between_action_var",
        "within_action_noise_var",
        "snr",
    }
    assert set(payload["d5_pred_vs_actual"]) == {"pearson_r", "slope", "intercept"}
    assert set(payload["conclusion"]) == {"action_information_used", "evidence"}


def test_strict_block_reason_is_json_friendly() -> None:
    """The not-run fallback records a reason instead of raising."""
    reason = dac._strict_block_reason(RuntimeError("--no-strict requested"))
    assert reason["available"] is False
    assert "no-strict" in reason["reason"]


# --- candidate construction and CLI -----------------------------------------


def test_candidate_feature_rows_match_the_documented_layout() -> None:
    """Rows are state block + [multiplier, exploration, onehot_p, onehot_g]."""
    from experiments.generate_dataset import (
        FULL_ACTION_PM_MULT_RANGE,
        sample_full_action,
    )

    state_block = np.arange(6, dtype=float)
    rows, actions = dac.candidate_feature_rows(
        state_block, 30, 12345, 4, FULL_ACTION_PM_MULT_RANGE
    )
    assert rows.shape == (4, state_block.size + _ACTION_FEATURES)
    assert len(actions) == 4
    rng = np.random.Generator(np.random.PCG64([12345, 1]))
    operator, pm, exploration = sample_full_action(
        rng, 1.0 / 30.0, FULL_ACTION_PM_MULT_RANGE
    )
    assert actions[0]["mutation_operator"] == operator
    assert actions[0]["mutation_probability"] == pytest.approx(pm)
    assert actions[0]["exploration_strength"] == pytest.approx(exploration)
    assert actions[0]["mutation_multiplier"] == pytest.approx(pm * 30.0)
    for k, action in enumerate(actions):
        np.testing.assert_array_equal(rows[k, : state_block.size], state_block)
        multiplier, exploration_feature, onehot_p, onehot_g = rows[k, state_block.size :]
        assert multiplier == pytest.approx(action["mutation_multiplier"])
        assert exploration_feature == pytest.approx(action["exploration_strength"])
        assert onehot_p + onehot_g == pytest.approx(1.0)
        assert (onehot_p == 1.0) == (action["mutation_operator"] == "polynomial")
    # deterministic for the same hash seed
    again, _ = dac.candidate_feature_rows(
        state_block, 30, 12345, 4, FULL_ACTION_PM_MULT_RANGE
    )
    np.testing.assert_array_equal(again, rows)


def test_select_strict_snapshots_spreads_over_the_grid(tmp_path: Path) -> None:
    """Snapshots are evenly subsampled per problem and validated."""
    for generation in (2, 6, 10, 14, 18):
        (tmp_path / f"zdt1__seed1000__gen{generation}.pkl").touch()
    (tmp_path / "zdt2__seed1000__gen6.pkl").touch()
    (tmp_path / "harvest_config.json").write_text("{}", encoding="utf-8")

    selected = dac.select_strict_snapshots(tmp_path, ["zdt1"], 2)
    assert [path.name for path in selected] == [
        "zdt1__seed1000__gen2.pkl",
        "zdt1__seed1000__gen18.pkl",
    ]
    both = dac.select_strict_snapshots(tmp_path, ["zdt1", "zdt2"], 3)
    assert {path.name.split("__")[0] for path in both} == {"zdt1", "zdt2"}
    assert len(both) == 4  # 3 zdt1 snapshots + 1 available zdt2 snapshot
    with pytest.raises(FileNotFoundError, match="no snapshots"):
        dac.select_strict_snapshots(tmp_path, ["zdt6"], 2)


def test_parse_args_defaults() -> None:
    """The CLI defaults match the documented strict grid."""
    args = dac.parse_args([])
    assert args.model_dir == dac.DEFAULT_MODEL_DIR
    assert args.eval_dir == dac.DEFAULT_EVAL_DIR
    assert args.out_file == dac.DEFAULT_OUT_FILE
    assert args.strict_states_per_problem == 4
    assert args.strict_candidates == 8
    assert args.strict_horizon == 5
    assert args.strict_reps == 3
    assert args.no_strict is False
    assert args.no_multiplier_normalization is False
    assert tuple(args.problems) == dac.DEFAULT_PROBLEMS
    assert dac.parse_args(["--no-strict"]).no_strict is True
