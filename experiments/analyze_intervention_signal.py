from __future__ import annotations

"""Intervention **signal** analysis (Phase 2.75D Task 2).

Separates *signal* questions from *model* questions: before asking whether a
predictor can learn action advantages, this script measures whether the
intervention data even contains a usable action signal, and how the
planner's realized action ranks.

Input is the per-problem artifact written by
``experiments/counterfactual_actions.py evaluate-horizon``
(``counterfactual_horizon_{problem}.json``), which stores, per snapshot state
and per candidate action, the realized reward at each horizon for several
replicate RNG seeds.

Reported per (problem, horizon):

* **action variance** — between-action variance of the realized rewards
  (variance of the per-candidate replicate means), i.e. how much the choice
  of action moves the outcome at all.
* **horizon SNR** — ``between_action_var / within_action_noise_var`` where the
  within term is the mean replicate variance (the same statistic as the
  Phase-2B D4 diagnostic), reported per horizon so the horizon dependence is
  visible.
* **oracle ranking distribution** — where the controller (planner) action
  lands among the candidates: the histogram of its percentile rank
  (top-10% / top-25% / middle / bottom-25% / bottom-10%), the share of states
  where it is the oracle argmax, and the mean/median rank.
* **regret / oracle gap** — ``regret = best − planner`` realized reward and
  ``oracle_gap = best − mean(realized)`` (the state's headroom over an average
  candidate), both in the reward's own units.

Usage::

    python experiments/analyze_intervention_signal.py \
        --input-dir results/phase2_75d/counterfactual \
        --out results/phase2_75d/intervention_signal.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_INPUT_DIR = "results/phase2_75d/counterfactual"
DEFAULT_OUT = "results/phase2_75d/intervention_signal.json"


def _candidate_means(
    candidates: Sequence[dict[str, Any]], horizon: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-candidate mean reward and mean replicate variance at ``horizon``.

    Args:
        candidates: Candidate records of one state (each with ``reward``,
            ``mean_reward`` and ``kind``).
        horizon: Horizon key as stored in the artifact (e.g. ``"20"``).

    Returns:
        ``(kinds, means, within_vars)`` — parallel lists of the candidate kind
        (``"controller"`` / ``"default"`` / ``"alternative"``), the mean reward
        across replicates, and the replicate variance (0.0 when a candidate
        has a single replicate).
    """
    kinds: list[str] = []
    means: list[float] = []
    within: list[float] = []
    for candidate in candidates:
        rewards = candidate.get("reward", {}).get(horizon)
        if rewards is None or len(rewards) == 0:
            continue
        values = np.asarray(rewards, dtype=np.float64)
        kinds.append(str(candidate.get("kind", "alternative")))
        means.append(float(values.mean()))
        within.append(float(values.var(ddof=1)) if values.size > 1 else 0.0)
    return kinds, np.asarray(means), np.asarray(within)


def _percentile_rank_of_first(values: np.ndarray) -> float:
    """Midrank percentile of ``values[0]`` among ``values`` (0 = worst)."""
    if values.size < 2:
        return 0.5
    less = float(np.sum(values < values[0]))
    ties = float(np.sum(values == values[0]))
    return float((less + 0.5 * (ties - 1)) / (values.size - 1))


def analyze_file(payload: dict[str, Any], problems_filter: Sequence[str] | None) -> dict[str, Any]:
    """Extract the signal statistics of one ``counterfactual_horizon`` artifact."""
    problem = str(payload.get("problem"))
    if problems_filter and problem not in set(problems_filter):
        return {}
    horizons = sorted(
        {str(h) for state in payload.get("states", []) for h in state.get("per_horizon", {})},
        key=int,
    )
    per_horizon: dict[str, Any] = {}
    for horizon in horizons:
        between_terms: list[float] = []
        within_terms: list[float] = []
        ranks: list[float] = []
        regrets: list[float] = []
        gaps: list[float] = []
        oracle_hits: list[float] = []
        for state in payload.get("states", []):
            kinds, means, within = _candidate_means(state.get("candidates", []), horizon)
            if means.size < 2:
                continue
            between_terms.append(float(means.var(ddof=1)))
            within_terms.append(float(within.mean()))
            # Planner action is the index-0 ``controller`` candidate.
            controller_index = next(
                (i for i, kind in enumerate(kinds) if kind == "controller"), None
            )
            if controller_index is None:
                continue
            ranks.append(_percentile_rank_of_first(
                np.roll(means, -controller_index)  # move controller to front
            ))
            best = float(means.max())
            planner = float(means[controller_index])
            regrets.append(best - planner)
            gaps.append(best - float(means.mean()))
            oracle_hits.append(1.0 if planner >= best - 1e-12 else 0.0)

        between = float(np.mean(between_terms)) if between_terms else None
        within = float(np.mean(within_terms)) if within_terms else None
        snr = (
            float(between / within)
            if between is not None and within is not None and within > 0.0
            else None
        )
        rank_array = np.asarray(ranks, dtype=np.float64) if ranks else np.asarray([])
        per_horizon[horizon] = {
            "n_states": int(len(between_terms)),
            "action_variance": between,
            "replicate_noise_variance": within,
            "horizon_snr": snr,
            "oracle_ranking": {
                "mean_percentile_rank": float(rank_array.mean()) if rank_array.size else None,
                "median_percentile_rank": float(np.median(rank_array)) if rank_array.size else None,
                "std_percentile_rank": float(rank_array.std(ddof=1)) if rank_array.size > 1 else None,
                "top_10pct_share": float(np.mean(rank_array >= 0.9)) if rank_array.size else None,
                "top_25pct_share": float(np.mean(rank_array >= 0.75)) if rank_array.size else None,
                "bottom_25pct_share": float(np.mean(rank_array <= 0.25)) if rank_array.size else None,
                "bottom_10pct_share": float(np.mean(rank_array <= 0.1)) if rank_array.size else None,
                "is_oracle_argmax_share": float(np.mean(oracle_hits)) if oracle_hits else None,
            },
            "regret": {
                "mean": float(np.mean(regrets)) if regrets else None,
                "median": float(np.median(regrets)) if regrets else None,
                "max": float(np.max(regrets)) if regrets else None,
            },
            "oracle_gap": {
                "mean": float(np.mean(gaps)) if gaps else None,
                "median": float(np.median(gaps)) if gaps else None,
            },
        }
    return {"problem": problem, "per_horizon": per_horizon}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the signal analysis and write the JSON artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--problems", nargs="+", default=None,
                        help="Restrict to these problems (default: all files found).")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("counterfactual_horizon_*.json"))
    if not files:
        raise FileNotFoundError(f"no counterfactual_horizon_*.json in {input_dir}")

    per_problem: dict[str, Any] = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = analyze_file(payload, args.problems)
        if entry:
            per_problem[entry["problem"]] = entry["per_horizon"]

    # Pooled view: average the per-problem statistics where defined.
    pooled: dict[str, Any] = {}
    horizon_keys = sorted(
        {h for entry in per_problem.values() for h in entry}, key=int
    )
    for horizon in horizon_keys:
        rows = [entry[horizon] for entry in per_problem.values() if horizon in entry]
        if not rows:
            continue
        pooled[horizon] = {
            "n_problems": len(rows),
            "action_variance": float(np.mean([r["action_variance"] for r in rows])),
            "horizon_snr": float(np.mean([
                r["horizon_snr"] for r in rows if r["horizon_snr"] is not None
            ])) if any(r["horizon_snr"] is not None for r in rows) else None,
            "mean_percentile_rank": float(np.mean([
                r["oracle_ranking"]["mean_percentile_rank"] for r in rows
                if r["oracle_ranking"]["mean_percentile_rank"] is not None
            ])),
            "oracle_argmax_share": float(np.mean([
                r["oracle_ranking"]["is_oracle_argmax_share"] for r in rows
                if r["oracle_ranking"]["is_oracle_argmax_share"] is not None
            ])),
            "regret_mean": float(np.mean([r["regret"]["mean"] for r in rows])),
            "oracle_gap_mean": float(np.mean([r["oracle_gap"]["mean"] for r in rows])),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"config": vars(args), "per_problem": per_problem, "pooled": pooled},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[done] wrote {out_path}")

    print(f"\n{'problem':8}{'h':>4}{'action_var':>12}{'SNR':>8}{'rank':>8}"
          f"{'oracle_hit':>11}{'regret':>10}{'oracle_gap':>11}")
    for problem, entry in sorted(per_problem.items()):
        for horizon, stats in sorted(entry.items(), key=lambda kv: int(kv[0])):
            snr = stats["horizon_snr"]
            print(
                f"{problem:8}{horizon:>4}{stats['action_variance']:>12.6f}"
                f"{snr if snr is not None else float('nan'):>8.2f}"
                f"{stats['oracle_ranking']['mean_percentile_rank']:>8.3f}"
                f"{stats['oracle_ranking']['is_oracle_argmax_share']:>11.3f}"
                f"{stats['regret']['mean']:>10.4f}"
                f"{stats['oracle_gap']['mean']:>11.4f}"
            )
    print("\n[pooled]")
    for horizon, stats in sorted(pooled.items(), key=lambda kv: int(kv[0])):
        print(f"  h={horizon}: SNR={stats['horizon_snr']:.2f} "
              f"rank={stats['mean_percentile_rank']:.3f} "
              f"oracle_hit={stats['oracle_argmax_share']:.3f} "
              f"regret={stats['regret_mean']:.4f} gap={stats['oracle_gap_mean']:.4f}")
    return {"per_problem": per_problem, "pooled": pooled}


if __name__ == "__main__":
    main()
