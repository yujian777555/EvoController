from __future__ import annotations

"""Tests for the intervention signal analysis (Phase 2.75D Task 2)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.analyze_intervention_signal import (
    _candidate_means,
    _percentile_rank_of_first,
    analyze_file,
    main,
)


def _candidate(
    kind: str, index: int, horizon: str, rewards: list[float]
) -> dict[str, Any]:
    """One candidate record in the ``evaluate-horizon`` schema."""
    return {
        "index": index,
        "kind": kind,
        "reward": {horizon: list(rewards)},
        "mean_reward": {horizon: float(np.mean(rewards))},
        "future_hv": {horizon: list(rewards)},
    }


def _state(
    seed: int, generation: int, horizon: str, rewards_by_kind: dict[str, list[list[float]]]
) -> dict[str, Any]:
    """One state record holding the given candidates."""
    candidates = []
    for index, (kind, reward_lists) in enumerate(rewards_by_kind.items()):
        candidates.append(
            {
                "index": index,
                "kind": kind,
                "reward": {horizon: list(reward_lists)},
                "mean_reward": {horizon: float(np.mean(reward_lists))},
            }
        )
    return {
        "problem": "zdt1",
        "seed": seed,
        "generation": generation,
        "candidates": candidates,
        "per_horizon": {horizon: {"n_candidates": len(candidates)}},
    }


def test_candidate_means_hand_computed() -> None:
    """Means and replicate variances are the plain statistics of ``reward``."""
    candidates = [
        _candidate("controller", 0, "5", [1.0, 3.0]),      # mean 2, var 2
        _candidate("default", 1, "5", [4.0, 4.0]),         # mean 4, var 0
        _candidate("alternative", 2, "5", [0.0, 2.0, 4.0]),  # mean 2, var 4
    ]
    kinds, means, within = _candidate_means(candidates, "5")
    assert kinds == ["controller", "default", "alternative"]
    np.testing.assert_allclose(means, [2.0, 4.0, 2.0])
    np.testing.assert_allclose(within, [2.0, 0.0, 4.0])


def test_candidate_means_skips_missing_horizon() -> None:
    """Candidates without the requested horizon are dropped, not zero-filled."""
    candidates = [_candidate("controller", 0, "5", [1.0, 1.0])]
    kinds, means, _ = _candidate_means(candidates, "10")
    assert kinds == [] and means.size == 0


def test_percentile_rank_semantics() -> None:
    """Worst action -> 0, best -> 1, middle -> 0.5, ties -> midrank."""
    assert _percentile_rank_of_first(np.array([0.0, 1.0, 2.0])) == pytest.approx(0.0)
    assert _percentile_rank_of_first(np.array([2.0, 1.0, 0.0])) == pytest.approx(1.0)
    assert _percentile_rank_of_first(np.array([1.0, 0.0, 2.0])) == pytest.approx(0.5)
    # Two candidates tied at the top -> midrank of the tied block.
    assert _percentile_rank_of_first(np.array([1.0, 1.0, 0.0])) == pytest.approx(0.75)
    assert _percentile_rank_of_first(np.array([5.0])) == pytest.approx(0.5)


def test_analyze_file_reports_signal_and_ranking() -> None:
    """End-to-end statistics on two hand-built states."""
    payload = {
        "problem": "zdt1",
        "states": [
            # State A: controller is the best action -> rank 1.0, regret 0.
            _state(1, 2, "5", {
                "controller": [3.0, 3.0],
                "alternative": [1.0, 1.0],
            }),
            # State B: controller is worst -> rank 0.0, regret 2.0.
            _state(1, 6, "5", {
                "controller": [1.0, 1.0],
                "alternative": [3.0, 3.0],
            }),
        ],
    }
    entry = analyze_file(payload, None)
    assert entry["problem"] == "zdt1"
    stats = entry["per_horizon"]["5"]
    assert stats["n_states"] == 2
    # Between-action variance is 2.0 for both states: means are {3, 1}, so the
    # ddof=1 variance is ((3-2)^2 + (1-2)^2) / 1 = 2.
    assert stats["action_variance"] == pytest.approx(2.0)
    # Replicates are identical -> zero noise, SNR undefined (division by zero).
    assert stats["replicate_noise_variance"] == pytest.approx(0.0)
    assert stats["horizon_snr"] is None
    ranking = stats["oracle_ranking"]
    assert ranking["mean_percentile_rank"] == pytest.approx(0.5)  # 1.0 and 0.0
    assert ranking["top_10pct_share"] == pytest.approx(0.5)
    assert ranking["bottom_10pct_share"] == pytest.approx(0.5)
    assert ranking["is_oracle_argmax_share"] == pytest.approx(0.5)
    assert stats["regret"]["mean"] == pytest.approx(1.0)  # (0 + 2) / 2
    assert stats["oracle_gap"]["mean"] == pytest.approx(1.0)


def test_analyze_file_filters_problems() -> None:
    """A problem outside the filter yields an empty result."""
    payload = {"problem": "zdt9", "states": []}
    assert analyze_file(payload, ["zdt1"]) == {}
    assert analyze_file(payload, None)["problem"] == "zdt9"


def test_main_writes_pooled_view(tmp_path: Path) -> None:
    """The CLI writes per-problem and pooled statistics."""
    input_dir = tmp_path / "cf"
    input_dir.mkdir()
    for problem in ("zdt1", "zdt2"):
        payload = {
            "problem": problem,
            "states": [
                _state(1, 2, "5", {"controller": [2.0, 2.0], "alternative": [1.0, 1.0]}),
                _state(1, 6, "5", {"controller": [1.0, 2.0], "alternative": [3.0, 3.0]}),
            ],
        }
        (input_dir / f"counterfactual_horizon_{problem}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    out = tmp_path / "signal.json"
    result = main(["--input-dir", str(input_dir), "--out", str(out)])
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["per_problem"]) == {"zdt1", "zdt2"}
    assert "5" in payload["pooled"]
    pooled = payload["pooled"]["5"]
    assert pooled["n_problems"] == 2
    # State 2 has replicate noise -> the pooled SNR is defined here.
    assert pooled["horizon_snr"] is not None
    assert 0.0 <= pooled["mean_percentile_rank"] <= 1.0
    assert pooled["regret_mean"] >= 0.0
    assert result["pooled"]["5"]["n_problems"] == 2


def test_main_raises_without_input(tmp_path: Path) -> None:
    """An empty input directory is a hard error."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        main(["--input-dir", str(empty), "--out", str(tmp_path / "x.json")])
