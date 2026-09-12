from __future__ import annotations

"""Tests for the Phase-2.75 advantage pipeline.

Everything is synthetic: the intervention dataset is built from hand-written
``counterfactual_horizon_*.json`` payloads plus matching snapshot pickles, and
the advantage predictor is trained on a deliberately constructed problem where

* the supervised target is constant within a state (so a plain MSE fit has no
  incentive to read the action channel at all — the Phase-2B failure mode
  measured by its D1 diagnostic), while
* paired action-shuffled negatives encode the true within-state ordering, so
  the hinge term is the only signal that can teach it.

Assertions are therefore about mechanism (does the contrastive term buy
within-state ranking?) rather than about absolute loss values.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks import get_problem
from controller.advantage_predictor import (
    AdvantagePredictor,
    hinge_ranking_loss,
    ranking_metrics,
)
from controller.dataset import build_outcome_samples, merge_state_reward
from controller.state_encoder import StateEncoder
from experiments import build_intervention_dataset as bid

_WINDOW = 3
_N_GENERATIONS = 12
_N_VARS = 30


# --- shared synthetic data ---------------------------------------------------


def _trajectories(n_traj: int = 2) -> list[list[dict[str, Any]]]:
    """Two recorder-schema trajectories used to fit the tiny encoder."""
    trajectories: list[list[dict[str, Any]]] = []
    for j in range(n_traj):
        hv = 0.30 + 0.05 * j
        igd = 0.50 - 0.02 * j
        transitions = []
        for t in range(_N_GENERATIONS):
            delta_hv = 0.0 if t == 0 else 0.006 + 0.001 * ((t + j) % 3)
            delta_igd = 0.0 if t == 0 else 0.005 + 0.001 * ((t + j + 1) % 3)
            hv += delta_hv
            igd -= delta_igd
            transitions.append(
                {
                    "generation": t,
                    "state": {
                        "generation": t,
                        "hv": hv,
                        "igd": igd,
                        "diversity": 0.25 + 0.01 * t,
                    },
                    "action": {
                        "mutation_operator": "polynomial",
                        "mutation_probability": 1.0 / _N_VARS,
                        "exploration_strength": 20.0,
                    },
                    "reward": {"delta_hv": delta_hv, "delta_igd": delta_igd},
                }
            )
        trajectories.append(transitions)
    return trajectories


def _fitted_encoder() -> StateEncoder:
    return StateEncoder(_WINDOW).fit(_trajectories())


def _history() -> list[dict[str, Any]]:
    """Merged history of the synthetic first trajectory (generations 0..4)."""
    return [merge_state_reward(tr) for tr in _trajectories()[0][:5]]


def _action(index: int) -> dict[str, Any]:
    """A deterministic action with a distinct multiplier per index."""
    return {
        "mutation_operator": "polynomial" if index % 2 == 0 else "gaussian",
        "mutation_probability": (0.5 + index) / _N_VARS,
        "exploration_strength": 10.0 + index,
    }


def _counterfactual_payload(
    *,
    problem: str = "zdt1",
    seed: int = 1000,
    generation: int = 4,
    horizons: tuple[int, ...] = (2, 4),
    n_reps: int = 2,
    mean_rewards: list[float] | None = None,
    kinds: list[str] | None = None,
) -> dict[str, Any]:
    """A minimal ``counterfactual_horizon_{problem}.json`` payload."""
    mean_rewards = [0.25, 0.30, 0.20, 0.40] if mean_rewards is None else mean_rewards
    kinds = (
        ["controller", "alternative", "alternative", "alternative"]
        if kinds is None
        else kinds
    )
    candidates = []
    for index, (mean_reward, kind) in enumerate(zip(mean_rewards, kinds)):
        # Replicate rewards are symmetric around the stored mean, so the
        # fixture is internally consistent and has non-zero noise variance.
        per_rep_offsets = [
            0.01 * (rep - (n_reps - 1) / 2.0) for rep in range(n_reps)
        ]
        candidates.append(
            {
                "index": index,
                "kind": kind,
                "action": _action(index),
                "future_hv": {str(h): [0.0] * n_reps for h in horizons},
                "reward": {
                    str(h): [mean_reward + offset for offset in per_rep_offsets]
                    for h in horizons
                },
                "mean_reward": {str(h): mean_reward for h in horizons},
                "predicted_reward": None,
            }
        )
    return {
        "problem": problem,
        "config": {
            "horizons": list(horizons),
            "n_alternatives": len(candidates) - 1,
            "n_reps": n_reps,
            "generations": 100,
            "n_reference_points": 200,
        },
        "states": [
            {
                "problem": problem,
                "seed": seed,
                "generation": generation,
                "state_metrics": {"hv": 0.123, "igd": 0.5},
                "controller_action": candidates[0]["action"],
                "candidates": candidates,
                "per_horizon": {},
            }
        ],
        "summary": {},
    }


def _write_snapshot(
    snapshots_dir: Path,
    problem: str = "zdt1",
    seed: int = 1000,
    generation: int = 4,
    history: list[dict[str, Any]] | None = None,
) -> Path:
    """Write the snapshot pickle the dataset builder joins on."""
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    path = snapshots_dir / f"{problem}__seed{seed}__gen{generation}.pkl"
    with path.open("wb") as fh:
        pickle.dump({"history": _history() if history is None else history}, fh)
    return path


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Encoder + input/snapshot directories with one usable and one skippable file."""
    root = tmp_path_factory.mktemp("phase275")
    encoder = _fitted_encoder()
    encoder_path = root / "encoder.json"
    encoder.save(encoder_path)

    input_dir = root / "counterfactual"
    input_dir.mkdir()
    payload = _counterfactual_payload()
    with (input_dir / "counterfactual_horizon_zdt1.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(payload, fh)
    skipped = _counterfactual_payload(problem="zdt2", horizons=(7, 9))
    with (input_dir / "counterfactual_horizon_zdt2.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(skipped, fh)

    snapshots_dir = root / "snapshots"
    _write_snapshot(snapshots_dir)
    return {
        "root": root,
        "encoder": encoder,
        "encoder_path": encoder_path,
        "input_dir": input_dir,
        "snapshots_dir": snapshots_dir,
        "payload": payload,
    }


# --- action features / baselines ---------------------------------------------


def test_action_features_match_build_outcome_samples_bit_exactly() -> None:
    """The 4 action features are the dataset builder's action block."""
    trajectories = _trajectories()
    encoder = StateEncoder(_WINDOW).fit(trajectories)
    horizons = [1, 2]
    X, _, _, sample_indices = build_outcome_samples(
        trajectories, encoder, _WINDOW, horizons
    )
    t = 1
    row = int(np.where(sample_indices == t)[0][0])
    action = trajectories[0][t]["action"]
    features = bid.action_features(action, _N_VARS)
    assert features.shape == (bid.N_ACTION_FEATURES,)
    # build_outcome_samples uses the stored multiplier; for this synthetic
    # corpus it equals pm * n_vars, the builder's definition.
    np.testing.assert_allclose(
        X[row, encoder.dim :],
        features,
        rtol=0.0,
        atol=1e-12,
    )
    # ... and the multiplier really is pm * n_vars.
    assert features[0] == pytest.approx(
        action["mutation_probability"] * _N_VARS
    )
    # one-hot block
    assert features[2] + features[3] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="mutation_operator"):
        bid.action_features({"mutation_operator": "nope"}, _N_VARS)


def test_resolve_baseline_hand_computed() -> None:
    """All four baseline options, hand-checked, plus their failure modes."""
    means = [0.25, 0.30, 0.20, 0.40]
    kinds = ["controller", "alternative", "alternative", "alternative"]
    assert bid.resolve_baseline(means, kinds, "state_mean") == pytest.approx(
        np.mean(means)
    )
    assert bid.resolve_baseline(means, kinds, "oracle_best") == pytest.approx(0.40)
    assert bid.resolve_baseline(means, kinds, "worst") == pytest.approx(0.20)
    assert bid.resolve_baseline(means, kinds, "controller") == pytest.approx(0.25)
    with pytest.raises(ValueError, match="controller"):
        bid.resolve_baseline(means, ["alternative"] * 4, "controller")
    with pytest.raises(ValueError, match="zero candidates"):
        bid.resolve_baseline([], [], "state_mean")
    with pytest.raises(ValueError, match="unknown baseline"):
        bid.resolve_baseline(means, kinds, "nope")


def test_extract_problem_samples_layout_and_advantages(workspace: dict[str, Any]) -> None:
    """X layout, meta columns and y_adv (state_mean baseline) are correct."""
    payload = workspace["payload"]
    encoder = workspace["encoder"]
    columns = bid.extract_problem_samples(
        payload,
        encoder,
        horizons=[2, 4],
        baseline="state_mean",
        snapshots_dir=workspace["snapshots_dir"],
    )
    n_candidates = len(payload["states"][0]["candidates"])
    assert len(columns["y_adv"]) == n_candidates * 2  # two horizons
    state_block = np.asarray(encoder.transform(_history()), dtype=np.float64)
    for index, row in enumerate(columns["X"]):
        assert row.shape == (state_block.size + bid.N_ACTION_FEATURES,)
        np.testing.assert_array_equal(row[: state_block.size], state_block)
    # first horizon block: advantages around the state mean of that horizon
    means = [0.25, 0.30, 0.20, 0.40]
    for index, mean_reward in enumerate(means):
        assert columns["mean_reward"][index] == pytest.approx(mean_reward)
        assert columns["y_adv"][index] == pytest.approx(
            mean_reward - float(np.mean(means))
        )
        assert columns["baseline_value"][index] == pytest.approx(float(np.mean(means)))
        assert columns["candidate_kind"][index] == (
            "controller" if index == 0 else "alternative"
        )
        assert columns["horizon"][index] == 2
        assert columns["mutation_multiplier"][index] == pytest.approx(
            _action(index)["mutation_probability"] * _N_VARS
        )
        assert columns["n_reps"][index] == 2
        assert len(columns["rewards"][index]) == 2
    assert columns["hv_before"][0] == pytest.approx(0.123)
    # second horizon block repeats the candidates at horizon 4
    assert columns["horizon"][n_candidates] == 4


def test_extract_problem_samples_baseline_variants(workspace: dict[str, Any]) -> None:
    """oracle_best / worst / controller shift the target as expected."""
    payload = workspace["payload"]
    encoder = workspace["encoder"]
    means = [0.25, 0.30, 0.20, 0.40]
    expected = {
        "oracle_best": 0.40,
        "worst": 0.20,
        "controller": 0.25,
    }
    for option, baseline in expected.items():
        columns = bid.extract_problem_samples(
            payload,
            encoder,
            horizons=[2],
            baseline=option,
            snapshots_dir=workspace["snapshots_dir"],
        )
        for index, mean_reward in enumerate(means):
            assert columns["y_adv"][index] == pytest.approx(mean_reward - baseline)
        # the oracle-best baseline makes every advantage non-positive
        if option == "oracle_best":
            assert max(columns["y_adv"]) == pytest.approx(0.0)


def test_extract_problem_samples_missing_snapshot_raises(
    workspace: dict[str, Any],
) -> None:
    """A state whose snapshot pickle is absent is an explicit error."""
    payload = _counterfactual_payload(seed=4242, generation=77)
    with pytest.raises(FileNotFoundError, match="zdt1__seed4242__gen77.pkl"):
        bid.extract_problem_samples(
            payload,
            workspace["encoder"],
            horizons=[2],
            baseline="state_mean",
            snapshots_dir=workspace["snapshots_dir"],
        )


def test_extract_problem_samples_rejects_absent_horizons(
    workspace: dict[str, Any],
) -> None:
    """Requesting horizons the file does not carry is an explicit error."""
    with pytest.raises(ValueError, match="contain none of"):
        bid.extract_problem_samples(
            workspace["payload"],
            workspace["encoder"],
            horizons=[5, 10, 20],
            baseline="state_mean",
            snapshots_dir=workspace["snapshots_dir"],
        )


def test_action_effect_snr_matches_hand_computation() -> None:
    """SNR = between-candidate variance / mean replicate variance."""
    means = [0.0, 1.0, 2.0, 3.0]
    rewards = [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]
    between, within, snr = bid.action_effect_snr(means, rewards)
    assert between == pytest.approx(float(np.var(means, ddof=1)))
    assert within == pytest.approx(0.005)
    assert snr == pytest.approx(between / within)
    # no replicate information -> no noise estimate, hence no SNR
    assert bid.action_effect_snr(means, [[0.0], [1.0], [2.0], [3.0]]) == (
        between,
        None,
        None,
    )
    # a single candidate cannot define a between-candidate variance
    assert bid.action_effect_snr([1.0], [[1.0, 1.0]]) == (None, None, None)


def test_run_dataset_writes_npz_and_meta(workspace: dict[str, Any]) -> None:
    """End-to-end: arrays, meta counters, per-horizon SNR and skipped files."""
    out_dir = workspace["root"] / "out"
    args = bid.parse_args(
        [
            "--input-dir", str(workspace["input_dir"]),
            "--snapshots-dir", str(workspace["snapshots_dir"]),
            "--encoder", str(workspace["encoder_path"]),
            "--out-dir", str(out_dir),
            "--horizons", "2", "4",
        ]
    )
    meta = bid.run_dataset(args)

    npz_path = out_dir / "intervention_dataset_zdt1.npz"
    assert npz_path.is_file()
    with np.load(npz_path) as arrays:
        assert set(arrays) >= {
            "X",
            "y_adv",
            "problem",
            "seed",
            "generation",
            "horizon",
            "candidate_index",
            "candidate_kind",
            "mean_reward",
            "rewards",
            "n_reps",
        }
        assert arrays["X"].shape[0] == arrays["y_adv"].size
        assert arrays["X"].shape[1] == workspace["encoder"].dim + 4
        assert set(arrays["horizon"].tolist()) == {2, 4}
        assert arrays["problem"].tolist() == ["zdt1"] * arrays["X"].shape[0]
        assert np.allclose(
            arrays["y_adv"].reshape(2, -1).mean(axis=1), 0.0, atol=1e-12
        )

    assert meta["n_samples"] == 8
    assert meta["n_states"] == 1
    assert meta["feature_dim"] == workspace["encoder"].dim + 4
    assert meta["files_used"] == ["counterfactual_horizon_zdt1.json"]
    assert [entry["file"] for entry in meta["files_skipped"]] == [
        "counterfactual_horizon_zdt2.json"
    ]
    assert "contain none of" in meta["files_skipped"][0]["reason"]
    assert set(meta["per_horizon"]) == {"2", "4"}
    for entry in meta["per_horizon"].values():
        assert entry["n_samples"] == 4
        assert entry["n_states"] == 1
        assert entry["between_action_var"] is not None
        assert entry["within_action_noise_var"] is not None
        assert entry["action_effect_snr"] == pytest.approx(
            entry["between_action_var"] / entry["within_action_noise_var"]
        )
    assert meta["config"]["baseline"] == "state_mean"
    assert meta["per_problem"]["zdt1"]["n_candidates_per_state"] == [4]
    assert (out_dir / "intervention_meta.json").is_file()


def test_run_dataset_without_matching_horizons_raises(
    workspace: dict[str, Any], tmp_path: Path
) -> None:
    """No usable file (evaluation still running) is an explicit error."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with (input_dir / "counterfactual_horizon_zdt2.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(_counterfactual_payload(problem="zdt2", horizons=(7, 9)), fh)
    args = bid.parse_args(
        [
            "--input-dir", str(input_dir),
            "--snapshots-dir", str(workspace["snapshots_dir"]),
            "--encoder", str(workspace["encoder_path"]),
            "--out-dir", str(tmp_path / "out"),
            "--horizons", "5", "10", "20",
        ]
    )
    with pytest.raises(ValueError, match="may still be running"):
        bid.run_dataset(args)


# --- AdvantagePredictor ------------------------------------------------------


def test_hinge_ranking_loss_values() -> None:
    """Satisfied pairs contribute 0; violated pairs contribute the shortfall."""
    import torch

    pos = torch.tensor([[1.0, 0.5]])
    satisfied = torch.tensor([[0.0, 0.0]])
    assert float(hinge_ranking_loss(pos, satisfied, margin=0.5).sum()) == 0.0
    violated = torch.tensor([[0.8, 0.5]])
    loss = hinge_ranking_loss(pos, violated, margin=0.5)
    np.testing.assert_allclose(loss.detach().numpy(), [[0.3, 0.5]], atol=1e-12)
    assert float(hinge_ranking_loss(pos, pos, margin=0.0).sum()) == 0.0
    with pytest.raises(ValueError, match="shape"):
        hinge_ranking_loss(pos, torch.zeros(2, 2))


def _synthetic_action_world(
    n_states: int = 40, n_candidates: int = 6, seed: int = 0
) -> dict[str, Any]:
    """States whose supervised target is constant, but whose action order is real.

    The state block is 12-dimensional noise and the action block carries a
    multiplier; the *advantage target* is 0 for every training row (an MSE
    fit is perfectly satisfied by ignoring the action), while the
    ground-truth within-state ordering is increasing in the multiplier.

    Training rows are aligned pairs: every ordered ``(higher, lower)``
    candidate pair of a state contributes one positive and one negative, so
    the hinge constrains the full within-state order. ``X_all`` holds the
    unique candidate rows used for evaluation.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    state_dim = 12
    multipliers = np.linspace(0.5, 6.0, n_candidates)
    X_all, X_pos, X_neg, realized = [], [], [], []
    for _ in range(n_states):
        state = rng.normal(size=state_dim)
        # Shuffle which row holds which multiplier, so the ordering cannot be
        # read off the row position -- it has to come from the action block.
        state_multipliers = multipliers[rng.permutation(n_candidates)]
        rows = [
            np.concatenate([state, np.asarray([value, 10.0 + value, 1.0, 0.0])])
            for value in state_multipliers
        ]
        X_all.extend(rows)
        realized.append(state_multipliers)
        by_multiplier = [
            rows[index] for index in np.argsort(state_multipliers, kind="stable")
        ]
        for higher in range(n_candidates):
            for lower in range(higher):
                X_pos.append(by_multiplier[higher])
                X_neg.append(by_multiplier[lower])
    X_pos = np.asarray(X_pos, dtype=np.float64)
    X_neg = np.asarray(X_neg, dtype=np.float64)
    n_pairs = X_pos.shape[0]
    return {
        "X": X_pos,
        "X_neg": X_neg,
        "X_all": np.asarray(X_all, dtype=np.float64),
        "y": np.zeros((n_pairs, 1), dtype=np.float64),
        "y_neg": np.zeros((n_pairs, 1), dtype=np.float64),
        "realized": np.asarray(realized, dtype=np.float64),
        "state_dim": state_dim,
    }


def _per_state_predictions(
    predictor: AdvantagePredictor, X: np.ndarray, n_candidates: int
) -> np.ndarray:
    """Predictions reshaped to ``(n_states, n_candidates)``."""
    predictions = predictor.predict(X)
    return predictions[:, 0].reshape(-1, n_candidates)


def test_predictor_is_deterministic() -> None:
    """Same seed and data -> identical parameters and predictions."""
    world = _synthetic_action_world(n_states=6, n_candidates=3)
    first = AdvantagePredictor(input_dim=world["X"].shape[1], horizons=[5],
                               hidden_dims=(8, 8), seed=0)
    second = AdvantagePredictor(input_dim=world["X"].shape[1], horizons=[5],
                                hidden_dims=(8, 8), seed=0)
    first.fit(world["X"], world["y"], epochs=3)
    second.fit(world["X"], world["y"], epochs=3)
    np.testing.assert_array_equal(first.predict(world["X"]), second.predict(world["X"]))


def test_fit_reduces_training_loss() -> None:
    """The reported train_loss decreases over epochs."""
    world = _synthetic_action_world(n_states=5, n_candidates=3)
    predictor = AdvantagePredictor(
        input_dim=world["X"].shape[1], horizons=[5], hidden_dims=(8, 8), seed=0
    )
    history = predictor.fit(world["X"], world["y"], epochs=20, batch_size=16)
    assert set(history) == {"train_loss", "val_loss"}
    assert len(history["train_loss"]) == 20
    assert history["val_loss"] == []
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_contrastive_training_learns_within_state_ranking() -> None:
    """MSE alone cannot learn the action order; the hinge term does.

    The supervised target is constant, so a plain fit has no gradient
    pushing it to read the action block (Phase 2B's D1 failure mode). With
    paired negatives the hinge forces positives above their lower-multiplier
    counterparts, which is exactly the ground-truth ordering, and the
    within-state Spearman must jump.
    """
    world = _synthetic_action_world()
    n_candidates = world["realized"].shape[1]
    input_dim = world["X"].shape[1]

    plain = AdvantagePredictor(input_dim=input_dim, horizons=[5],
                               hidden_dims=(32, 32), seed=0)
    plain.fit(world["X"], world["y"], epochs=150, batch_size=64)
    plain_metrics = ranking_metrics(
        _per_state_predictions(plain, world["X_all"], n_candidates),
        world["realized"],
    )

    contrastive = AdvantagePredictor(input_dim=input_dim, horizons=[5],
                                     hidden_dims=(32, 32), seed=0)
    history = contrastive.fit(
        world["X"],
        world["y"],
        X_neg=world["X_neg"],
        y_neg=world["y_neg"],
        margin=0.1,
        epochs=150,
        batch_size=64,
    )
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert contrastive.margin == pytest.approx(0.1)
    contrastive_metrics = ranking_metrics(
        _per_state_predictions(contrastive, world["X_all"], n_candidates),
        world["realized"],
    )

    assert contrastive_metrics["spearman_mean"] > 0.8
    assert contrastive_metrics["kendall_mean"] > 0.6
    assert contrastive_metrics["oracle_hit_rate"] == pytest.approx(1.0)
    assert contrastive_metrics["regret_mean"] == pytest.approx(0.0, abs=1e-12)
    plain_spearman = plain_metrics["spearman_mean"]
    assert plain_spearman is None or plain_spearman < 0.5
    assert contrastive_metrics["spearman_mean"] > (plain_spearman or 0.0)


def test_fit_validates_paired_negatives() -> None:
    """Negatives must be paired and shape-checked."""
    world = _synthetic_action_world(n_states=4, n_candidates=3)
    predictor = AdvantagePredictor(
        input_dim=world["X"].shape[1], horizons=[5], hidden_dims=(8, 8), seed=0
    )
    with pytest.raises(ValueError, match="together"):
        predictor.fit(world["X"], world["y"], X_neg=world["X_neg"], epochs=1)
    with pytest.raises(ValueError, match="X_neg shape"):
        predictor.fit(
            world["X"],
            world["y"],
            X_neg=world["X_neg"][:-1],
            y_neg=world["y_neg"][:-1],
            epochs=1,
        )
    with pytest.raises(ValueError, match="zero samples"):
        predictor.fit(np.zeros((0, world["X"].shape[1])), np.zeros((0, 1)), epochs=1)


def test_save_load_round_trip(tmp_path: Path) -> None:
    """save/load restores the weights, the config and the predictions."""
    world = _synthetic_action_world(n_states=6, n_candidates=3)
    predictor = AdvantagePredictor(
        input_dim=world["X"].shape[1], horizons=[5, 10], hidden_dims=(8, 8), seed=0
    )
    predictor.fit(
        world["X"],
        np.zeros((world["X"].shape[0], 2)),
        X_neg=world["X_neg"][: world["X"].shape[0]],
        y_neg=np.zeros((world["X"].shape[0], 2)),
        margin=0.2,
        epochs=5,
    )
    path = tmp_path / "advantage.pt"
    predictor.save(path)
    restored = AdvantagePredictor.load(path)
    assert restored.input_dim == predictor.input_dim
    assert restored.horizons == (5, 10)
    assert restored.hidden_dims == (8, 8)
    assert restored.margin == pytest.approx(0.2)
    np.testing.assert_array_equal(restored.predict(world["X"]), predictor.predict(world["X"]))


def test_predictor_attributes_and_validation() -> None:
    """Constructor validation and the documented properties."""
    predictor = AdvantagePredictor(input_dim=5, horizons=[5, 10, 20],
                                   hidden_dims=(4,), seed=3)
    assert predictor.input_dim == 5
    assert predictor.horizons == (5, 10, 20)
    assert predictor.hidden_dims == (4,)
    assert predictor.margin == 0.0
    assert predictor.name == "advantage_h5_10_20"
    with pytest.raises(ValueError, match="input_dim"):
        AdvantagePredictor(input_dim=0)
    with pytest.raises(ValueError, match="horizons"):
        AdvantagePredictor(input_dim=4, horizons=[])
    with pytest.raises(ValueError, match="horizon"):
        AdvantagePredictor(input_dim=4, horizons=[0])


# --- ranking metrics ---------------------------------------------------------


def test_ranking_metrics_perfect_and_reversed() -> None:
    """Perfect prediction: rho=1, hit rate 1, zero regret."""
    realized = np.asarray([[0.0, 0.5, 1.0, 2.0]])
    perfect = ranking_metrics(realized.copy(), realized)
    assert perfect["n_groups"] == 1
    assert perfect["spearman_mean"] == pytest.approx(1.0)
    assert perfect["kendall_mean"] == pytest.approx(1.0)
    assert perfect["oracle_hit_rate"] == pytest.approx(1.0)
    assert perfect["regret_mean"] == pytest.approx(0.0)
    assert perfect["oracle_gap_mean"] == pytest.approx(2.0 - np.mean([0, 0.5, 1, 2]))

    reversed_metrics = ranking_metrics(realized[:, ::-1].copy(), realized)
    assert reversed_metrics["spearman_mean"] == pytest.approx(-1.0)
    assert reversed_metrics["kendall_mean"] == pytest.approx(-1.0)
    assert reversed_metrics["oracle_hit_rate"] == pytest.approx(0.0)
    assert reversed_metrics["regret_mean"] == pytest.approx(2.0)


def test_ranking_metrics_ties_and_undefined_groups() -> None:
    """Ties resolve to the earliest candidate; degenerate groups are skipped."""
    realized = np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    tied = np.asarray([[0.5, 0.5, 0.5], [0.0, 1.0, 2.0]])
    metrics = ranking_metrics(tied, realized)
    assert metrics["n_groups"] == 2
    # first group: argmax(pred) == 0 -> realized 0.0, best 2.0 -> regret 2.0
    # second group: perfect -> regret 0.0
    assert metrics["regret_mean"] == pytest.approx(1.0)
    assert metrics["oracle_hit_rate"] == pytest.approx(0.5)
    assert metrics["spearman_mean"] == pytest.approx(1.0)  # only group 2 defined
    # a single group of two candidates cannot define a rank correlation
    single = ranking_metrics(np.asarray([[0.0, 1.0]]), np.asarray([[0.0, 1.0]]))
    assert single["spearman_mean"] is None
    assert single["oracle_hit_rate"] == pytest.approx(1.0)
    assert single["regret_mean"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="does not match"):
        ranking_metrics(np.zeros((2, 3)), np.zeros((3, 2)))


def test_ranking_metrics_top_k_hit_rate() -> None:
    """Phase-2.75D: top-k hit rate generalizes oracle-hit beyond k=1."""
    from controller.advantage_predictor import ranking_metrics

    realized = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])

    perfect = ranking_metrics(realized.copy(), realized)["top_k_hit_rate"]
    assert perfect == {"1": 1.0, "3": 1.0, "5": 1.0}

    # Pick the realized-worst action: outside top-1 and top-3, inside top-5.
    worst = ranking_metrics(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]), realized)
    assert worst["top_k_hit_rate"] == {"1": 0.0, "3": 0.0, "5": 1.0}
    assert worst["oracle_hit_rate"] == 0.0

    # Pick the middle action: inside top-3, outside top-1.
    middle = ranking_metrics(np.array([[1.0, 2.0, 9.0, 4.0, 5.0]]), realized)
    assert middle["top_k_hit_rate"] == {"1": 0.0, "3": 1.0, "5": 1.0}

    # With fewer candidates than k, the hit is capped at the group size.
    small = ranking_metrics(np.array([[1.0, 2.0]]), np.array([[2.0, 1.0]]))
    assert small["top_k_hit_rate"] == {"1": 0.0, "3": 1.0, "5": 1.0}
