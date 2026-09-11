from __future__ import annotations

"""Phase 2.75, Task 3: discrete macro actions and their action-effect SNR.

Phase 2B measured the causal signal of the *continuous* action space
(``(operator, mutation multiplier, exploration strength)``) on same-snapshot
interventions and found an action-effect SNR of only 1.35 — between-candidate
outcome variance barely above replicate noise — which makes any ranking
learner degenerate. Task 3 asks whether a **higher-level, discrete action
representation** carries a better signal-to-noise ratio.

This module defines a macro action table spanning the intervention spectrum
(:data:`MACRO_ACTIONS`), converts a macro into the exact 3-key dict
``NSGAII.step`` accepts (:func:`macro_action`), rebuilds the 4-dimensional
feature block used by every Phase-2 predictor (:func:`action_features`), and
measures the SNR of the macro discretisation on intervention data
(:func:`macro_snr_table`).

Macro set (multiplier is ``mutation_probability * n_vars``; eta_m / sigma are
the exploration strengths of the polynomial / Gaussian operator):

===========================  ==========  ==========  ===========  =====================
name                         operator    multiplier  exploration  intent
===========================  ==========  ==========  ===========  =====================
``explore_high``             polynomial  6.0         eta_m 4.0    large, flat jumps
``explore_mild``             polynomial  3.0         eta_m 12.0   moderate exploration
``neutral``                  polynomial  1.0         eta_m 20.0   NSGA-II default
``exploit_mild``             polynomial  0.5         eta_m 32.0   local refinement
``exploit_high``             polynomial  0.25        eta_m 50.0   fine local refinement
``operator_switch_gaussian`` gaussian    1.0         sigma 0.12   jump out of a basin
``gaussian_explore``         gaussian    2.0         sigma 0.25   strong Gaussian kick
===========================  ==========  ==========  ===========  =====================

Candidate actions are mapped to macros by **nearest neighbour inside the same
operator family**: the distance is the Euclidean norm of the log-multiplier
distance (normalized by the full-action multiplier range) and the exploration
distance (normalized by that operator's sampling range). Cross-operator
comparisons are never made, because eta_m and sigma are not comparable scales;
a candidate therefore always lands on a macro of its own operator.

SNR definitions (all computed per ``(state, horizon)`` group and then
aggregated; replicate rewards come from the intervention file's
``reward{h: [per-rep]}``):

* ``continuous_snr`` — ``var(per-candidate mean reward, ddof=1)`` divided by
  ``mean(per-candidate replicate variance, ddof=1)``: the Phase-2B D4
  statistic over the raw candidate set.
* ``macro_snr`` — the same ratio with the macro-level means (mean reward of
  every candidate assigned to a macro) in the numerator, i.e. the literal
  "between-macro / within-macro replicate noise" of the task description.
  Note the mathematical bound: because a macro mean averages its members'
  outcomes, the between-macro variance can never exceed the between-candidate
  variance (law of total variance), so this number is only informative next
  to the pooled decomposition below.
* pooled decomposition (``per_horizon``) — per state the candidate means are
  centred on that state's mean (removing run-to-run scale differences),
  pooled over states, and split into ``between_macro`` + ``within_macro``
  population variances:
  ``variance_explained_by_macro = between / (between + within)`` says how
  homogeneous the macro regions are, and
  ``macro_snr_within_as_noise = between / (within + replicate noise)`` is the
  decisive quantity: it treats the residual spread *inside* a macro as noise
  (a planner that can only choose a macro cannot control it) and is directly
  comparable to ``continuous_snr``, which treats that spread as signal (a
  planner that picks concrete actions can use it).

CLI:
    ``python -m controller.macro_actions --input-dir results/phase2b/counterfactual \
        --out results/phase2_75/macro_action_snr.json``
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from controller.dataset import OPERATOR_TO_INDEX

__all__ = [
    "MACRO_ACTIONS",
    "MULTIPLIER_RANGE",
    "ETA_M_RANGE",
    "SIGMA_RANGE",
    "DEFAULT_N_VARS",
    "macro_names",
    "macro_action",
    "action_features",
    "assign_macro",
    "decompose_group",
    "macro_snr_table",
    "parse_args",
    "main",
]

#: Deployment multiplier bounds of the Phase-1.5 full action space (mirrors
#: ``experiments.generate_dataset.FULL_ACTION_PM_MULT_RANGE``; asserted equal
#: by the test suite).
MULTIPLIER_RANGE: tuple[float, float] = (0.25, 8.0)
#: Polynomial exploration (eta_m) sampling range (mirrors
#: ``experiments.generate_dataset.ETA_M_SAMPLE_RANGE``).
ETA_M_RANGE: tuple[float, float] = (2.0, 50.0)
#: Gaussian exploration (sigma) sampling range (mirrors
#: ``experiments.generate_dataset.SIGMA_SAMPLE_RANGE``).
SIGMA_RANGE: tuple[float, float] = (0.02, 0.3)
#: Decision-variable count assumed when a macro is materialized without an
#: explicit ``n_vars`` (mirrors ``controller.dataset.OUTCOME_FALLBACK_N_VARS``).
DEFAULT_N_VARS: int = 30
#: Number of trailing feature columns that carry the action.
N_ACTION_FEATURES = 4
#: Default output of the CLI.
DEFAULT_OUT = "results/phase2_75/macro_action_snr.json"

#: Discrete macro action table. ``multiplier`` is ``pm * n_vars``;
#: ``exploration_strength`` is eta_m for polynomial and sigma for Gaussian.
MACRO_ACTIONS: dict[str, dict[str, Any]] = {
    "explore_high": {
        "mutation_operator": "polynomial",
        "multiplier": 6.0,
        "exploration_strength": 4.0,
    },
    "explore_mild": {
        "mutation_operator": "polynomial",
        "multiplier": 3.0,
        "exploration_strength": 12.0,
    },
    "neutral": {
        "mutation_operator": "polynomial",
        "multiplier": 1.0,
        "exploration_strength": 20.0,
    },
    "exploit_mild": {
        "mutation_operator": "polynomial",
        "multiplier": 0.5,
        "exploration_strength": 32.0,
    },
    "exploit_high": {
        "mutation_operator": "polynomial",
        "multiplier": 0.25,
        "exploration_strength": 50.0,
    },
    "operator_switch_gaussian": {
        "mutation_operator": "gaussian",
        "multiplier": 1.0,
        "exploration_strength": 0.12,
    },
    "gaussian_explore": {
        "mutation_operator": "gaussian",
        "multiplier": 2.0,
        "exploration_strength": 0.25,
    },
}


def macro_names() -> tuple[str, ...]:
    """Names of the macro actions, in table order."""
    return tuple(MACRO_ACTIONS)


def _exploration_range(operator: str) -> tuple[float, float]:
    """Sampling range of the exploration strength of one operator."""
    if operator == "polynomial":
        return ETA_M_RANGE
    if operator == "gaussian":
        return SIGMA_RANGE
    raise ValueError(
        f"unsupported mutation_operator {operator!r}; expected one of "
        f"{sorted(OPERATOR_TO_INDEX)}"
    )


def macro_action(name: str, n_vars: int | None = None) -> dict[str, Any]:
    """Materialize a macro action into the dict ``NSGAII.step`` consumes.

    Args:
        name: Key of :data:`MACRO_ACTIONS`.
        n_vars: Decision-variable count; the macro's ``multiplier`` becomes
            ``mutation_probability = multiplier / n_vars``. ``None`` applies
            :data:`DEFAULT_N_VARS` (the legacy fallback of the Phase-0/1
            corpora).

    Returns:
        Dict with exactly the keys ``"mutation_operator"``,
        ``"mutation_probability"`` and ``"exploration_strength"`` — the action
        contract shared by ``NSGAII.step``, the trajectory recorder and the
        Phase-2 planners.

    Raises:
        ValueError: If ``name`` is unknown or ``n_vars`` is < 1.
    """
    if name not in MACRO_ACTIONS:
        raise ValueError(
            f"unknown macro action {name!r}; expected one of {sorted(MACRO_ACTIONS)}"
        )
    if n_vars is None:
        resolved_n_vars = DEFAULT_N_VARS
    else:
        if int(n_vars) < 1:
            raise ValueError(f"n_vars must be >= 1, got {n_vars}")
        resolved_n_vars = int(n_vars)
    macro = MACRO_ACTIONS[name]
    return {
        "mutation_operator": str(macro["mutation_operator"]),
        "mutation_probability": float(macro["multiplier"]) / resolved_n_vars,
        "exploration_strength": float(macro["exploration_strength"]),
    }


def action_features(action: dict[str, Any], n_vars: int) -> np.ndarray:
    """Four action features in the :func:`build_outcome_samples` layout.

    ``[mutation_multiplier, exploration_strength, onehot_polynomial,
    onehot_gaussian]`` with ``mutation_multiplier = mutation_probability *
    n_vars`` (identical to the block
    :class:`controller.planning_controller.PlanningController` feeds the
    outcome/advantage predictors). An action that already carries
    ``"multiplier"`` (a macro entry) uses it directly.

    Args:
        action: Action dict with ``mutation_operator`` plus either
            ``mutation_probability`` or ``multiplier``, and
            ``exploration_strength``.
        n_vars: Decision-variable count of the problem.

    Returns:
        Array of shape ``(4,)``.

    Raises:
        ValueError: If the operator is unsupported or neither a multiplier
            nor a mutation probability is present.
    """
    operator = str(action["mutation_operator"])
    if operator not in OPERATOR_TO_INDEX:
        raise ValueError(
            f"unsupported mutation_operator {operator!r}; expected one of "
            f"{sorted(OPERATOR_TO_INDEX)}"
        )
    if "multiplier" in action:
        multiplier = float(action["multiplier"])
    elif "mutation_probability" in action:
        multiplier = float(action["mutation_probability"]) * int(n_vars)
    else:
        raise ValueError(
            "action must carry either 'multiplier' or 'mutation_probability'"
        )
    one_hot = [0.0, 0.0]
    one_hot[OPERATOR_TO_INDEX[operator]] = 1.0
    return np.asarray(
        [multiplier, float(action["exploration_strength"]), *one_hot],
        dtype=np.float64,
    )


def macro_distance(
    operator: str, multiplier: float, exploration: float, name: str
) -> float | None:
    """Normalized distance from a candidate action to one macro.

    Only actions and macros of the same mutation operator are comparable
    (eta_m and sigma live on different scales); for a different operator the
    distance is ``None``.

    Args:
        operator: Candidate's mutation operator.
        multiplier: Candidate's normalized mutation multiplier.
        exploration: Candidate's exploration strength.
        name: Macro name.

    Returns:
        ``sqrt(d_multiplier^2 + d_exploration^2)`` with both components
        normalized to ``[0, ~1]`` by the corresponding sampling range, or
        ``None`` when the operators differ.

    Raises:
        ValueError: If ``multiplier`` is not positive or ``name`` is unknown.
    """
    if name not in MACRO_ACTIONS:
        raise ValueError(f"unknown macro action {name!r}")
    if float(multiplier) <= 0.0:
        raise ValueError(f"multiplier must be positive, got {multiplier}")
    macro = MACRO_ACTIONS[name]
    if str(macro["mutation_operator"]) != str(operator):
        return None
    if str(operator) not in OPERATOR_TO_INDEX:
        raise ValueError(f"unsupported mutation_operator {operator!r}")
    multiplier_lo, multiplier_hi = MULTIPLIER_RANGE
    if multiplier_hi <= multiplier_lo:
        return None
    delta_multiplier = abs(
        np.log(float(multiplier) / float(macro["multiplier"]))
    ) / np.log(multiplier_hi / multiplier_lo)
    exploration_lo, exploration_hi = _exploration_range(str(operator))
    delta_exploration = abs(
        float(exploration) - float(macro["exploration_strength"])
    ) / (exploration_hi - exploration_lo)
    return float(np.hypot(delta_multiplier, delta_exploration))


def assign_macro(operator: str, multiplier: float, exploration: float) -> str:
    """Nearest macro of the candidate's own operator family.

    Args:
        operator: Candidate's mutation operator.
        multiplier: Candidate's normalized mutation multiplier.
        exploration: Candidate's exploration strength.

    Returns:
        The macro name with the smallest :func:`macro_distance`; ties break
        toward the first entry of :data:`MACRO_ACTIONS` (deterministic).

    Raises:
        ValueError: If no macro shares the candidate's operator.
    """
    best_name: str | None = None
    best_distance: float | None = None
    for name in MACRO_ACTIONS:
        distance = macro_distance(operator, multiplier, exploration, name)
        if distance is None:
            continue
        if best_distance is None or distance < best_distance:
            best_name, best_distance = name, distance
    if best_name is None:
        raise ValueError(
            f"no macro action uses mutation_operator {operator!r}; "
            f"add one to MACRO_ACTIONS"
        )
    return best_name


def decompose_group(
    candidate_means: Sequence[float],
    labels: Sequence[str],
    replicate_rewards: Sequence[Sequence[float]],
) -> dict[str, float | int | None]:
    """Variance decomposition of one ``(state, horizon)`` intervention group.

    Args:
        candidate_means: ``mean_reward`` of every candidate.
        labels: Macro assignment per candidate, aligned.
        replicate_rewards: Per-candidate replicate rewards, aligned.

    Returns:
        ``candidate_var`` (ddof=1 over candidates), ``macro_var`` (ddof=1
        over the macro means), ``total_var``/``between_macro_var``/
        ``within_macro_var`` (population, ddof=0, so the law-of-total-variance
        identity holds exactly), ``replicate_noise_var`` (mean over
        candidates of their ddof=1 replicate variance), plus
        ``n_candidates``/``n_macros``. Entries are ``None`` when the group
        cannot define them (fewer than two candidates or macros, or no
        candidate with two replicates).

    Raises:
        ValueError: If the three inputs have inconsistent lengths.
    """
    means = np.asarray(list(candidate_means), dtype=np.float64)
    label_list = [str(label) for label in labels]
    rewards = [np.asarray(list(row), dtype=np.float64) for row in replicate_rewards]
    if means.size != len(label_list) or means.size != len(rewards):
        raise ValueError(
            f"candidate_means ({means.size}), labels ({len(label_list)}) and "
            f"replicate_rewards ({len(rewards)}) must be aligned"
        )
    result: dict[str, float | int | None] = {
        "n_candidates": int(means.size),
        "n_macros": None,
        "candidate_var": None,
        "macro_var": None,
        "total_var": None,
        "between_macro_var": None,
        "within_macro_var": None,
        "replicate_noise_var": None,
    }
    if means.size == 0:
        return result
    unique_labels = sorted(set(label_list))
    result["n_macros"] = len(unique_labels)
    macro_means = np.asarray(
        [means[[i for i, label in enumerate(label_list) if label == name]].mean()
         for name in unique_labels],
        dtype=np.float64,
    )
    grand_mean = float(means.mean())
    total_population = float(np.mean((means - grand_mean) ** 2))
    # SSB / N with SSB = sum_m n_m * (mean_m - grand_mean)^2: the population
    # between-group variance (sum over groups, not mean over groups).
    between_population = float(
        np.sum(
            [
                (np.sum([label == name for label in label_list]) / means.size)
                * (macro_mean - grand_mean) ** 2
                for name, macro_mean in zip(unique_labels, macro_means)
            ]
        )
    )
    within_population = 0.0
    for name, macro_mean in zip(unique_labels, macro_means):
        members = np.asarray(
            [means[i] for i, label in enumerate(label_list) if label == name]
        )
        within_population += float(np.sum((members - macro_mean) ** 2)) / means.size
    result["total_var"] = total_population
    result["between_macro_var"] = between_population
    result["within_macro_var"] = within_population
    if means.size > 1:
        result["candidate_var"] = float(np.var(means, ddof=1))
    if macro_means.size > 1:
        result["macro_var"] = float(np.var(macro_means, ddof=1))
    replicate_variances = [
        float(np.var(row, ddof=1)) for row in rewards if row.size > 1
    ]
    if replicate_variances:
        result["replicate_noise_var"] = float(np.mean(replicate_variances))
    return result


def _intervention_groups(
    paths: Sequence[Path], horizons: Sequence[int] | None
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, str], list[str]]:
    """Read intervention files into per-horizon groups of assigned candidates.

    Returns:
        ``(groups, assignment, files)`` where ``groups[horizon]`` is a list of
        per-state dicts with ``candidate_means``, ``labels``,
        ``replicate_rewards`` and a state key, ``assignment`` maps a
        ``"{file}|seed{seed}|gen{generation}|cand{index}"`` key to its macro
        (this is the ``macro_assignment`` block of the artifact), and
        ``files`` lists the files that contributed samples.

    Raises:
        FileNotFoundError: If no input file exists.
        ValueError: If a file has no usable horizon or a candidate action is
            malformed.
    """
    groups: dict[int, list[dict[str, Any]]] = {}
    assignment: dict[str, str] = {}
    files: list[str] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        problem = str(payload.get("problem", Path(path).stem))
        file_horizons = [int(h) for h in payload.get("config", {}).get("horizons", [])]
        wanted = (
            [h for h in file_horizons if h in {int(x) for x in horizons}]
            if horizons is not None
            else list(file_horizons)
        )
        if not wanted:
            continue
        files.append(Path(path).name)
        n_vars = float(_n_vars_of(problem))
        for state in payload["states"]:
            seed = int(state["seed"])
            generation = int(state["generation"])
            candidates = list(state["candidates"])
            for horizon in wanted:
                key = str(horizon)
                means: list[float] = []
                labels: list[str] = []
                rewards: list[list[float]] = []
                for index, candidate in enumerate(candidates):
                    action = candidate["action"]
                    operator = str(action["mutation_operator"])
                    multiplier = float(action["mutation_probability"]) * n_vars
                    exploration = float(action["exploration_strength"])
                    label = assign_macro(operator, multiplier, exploration)
                    assignment[
                        f"{problem}|seed{seed}|gen{generation}|cand{candidate.get('index', index)}"
                    ] = label
                    means.append(float(candidate["mean_reward"][key]))
                    labels.append(label)
                    rewards.append([float(v) for v in candidate["reward"][key]])
                groups.setdefault(horizon, []).append(
                    {
                        "state": f"{problem}|seed{seed}|gen{generation}",
                        "candidate_means": means,
                        "labels": labels,
                        "replicate_rewards": rewards,
                    }
                )
    return groups, assignment, files


def _n_vars_of(problem: str) -> int:
    """Decision-variable count of a benchmark name (lazy import)."""
    from benchmarks import get_problem

    return int(get_problem(problem).n_vars)


def _horizon_statistics(groups: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-state SNRs plus the pooled state-centred decomposition.

    Args:
        groups: Per-state records of one horizon (see
            :func:`_intervention_groups`).

    Returns:
        Dict with the per-state averaged ``continuous_snr`` / ``macro_snr``,
        the pooled ``variance_explained_by_macro`` /
        ``macro_snr_within_as_noise``, the pooled variance components and the
        macro assignment counts.
    """
    continuous: list[float] = []
    macro: list[float] = []
    pooled_means: list[float] = []
    pooled_labels: list[str] = []
    pooled_replicate_variances: list[float] = []
    counts: dict[str, int] = {name: 0 for name in MACRO_ACTIONS}
    for group in groups:
        components = decompose_group(
            group["candidate_means"], group["labels"], group["replicate_rewards"]
        )
        noise = components["replicate_noise_var"]
        candidate_var = components["candidate_var"]
        macro_var = components["macro_var"]
        if noise is not None and noise > 0.0:
            if candidate_var is not None:
                continuous.append(float(candidate_var) / float(noise))
            if macro_var is not None:
                macro.append(float(macro_var) / float(noise))
        centre = float(np.mean(group["candidate_means"]))
        pooled_means.extend(float(value) - centre for value in group["candidate_means"])
        pooled_labels.extend(str(label) for label in group["labels"])
        for row in group["replicate_rewards"]:
            array = np.asarray(row, dtype=np.float64)
            if array.size > 1:
                pooled_replicate_variances.append(float(np.var(array, ddof=1)))
        for label in group["labels"]:
            counts[str(label)] = counts.get(str(label), 0) + 1
    pooled = decompose_group(
        pooled_means,
        pooled_labels,
        [[] for _ in pooled_means],
    )
    between = pooled["between_macro_var"]
    within = pooled["within_macro_var"]
    replicate_noise = (
        float(np.mean(pooled_replicate_variances))
        if pooled_replicate_variances
        else None
    )
    explained = (
        float(between) / (float(between) + float(within))
        if between is not None and within is not None and (between + within) > 0.0
        else None
    )
    within_as_noise = (
        float(between) / (float(within) + replicate_noise)
        if between is not None
        and within is not None
        and replicate_noise is not None
        and (float(within) + replicate_noise) > 0.0
        else None
    )
    return {
        "continuous_snr": float(np.mean(continuous)) if continuous else None,
        "macro_snr": float(np.mean(macro)) if macro else None,
        "macro_snr_within_as_noise": within_as_noise,
        "variance_explained_by_macro": explained,
        "between_macro_var_pooled": between,
        "within_macro_var_pooled": within,
        "replicate_noise_var_pooled": replicate_noise,
        "candidate_var_pooled": pooled["total_var"],
        "n_states": len(groups),
        "n_candidates": int(sum(len(group["candidate_means"]) for group in groups)),
        "n_states_with_defined_snr": len(continuous),
        "macro_counts": counts,
    }


def macro_snr_table(
    intervention_paths: Sequence[Path], horizons: Sequence[int] | None = None
) -> dict[str, Any]:
    """Compare the macro and continuous action-space SNRs.

    Args:
        intervention_paths: ``counterfactual_horizon_*.json`` files produced
            by ``experiments/counterfactual_actions.py evaluate-horizon``.
        horizons: Optional horizon filter; ``None`` uses every horizon the
            files carry.

    Returns:
        Artifact with ``continuous_snr``, ``macro_snr`` (D4-style per-state
        averages), the pooled decision-relevant
        ``macro_snr_within_as_noise`` / ``variance_explained_by_macro``, the
        ``macro_assignment`` of every candidate, the macro table, per-horizon
        statistics and the formula definitions. Values that the data cannot
        define are ``None``.

    Raises:
        FileNotFoundError: If no input file exists.
        ValueError: If the files contain no usable samples.
    """
    paths = [Path(path) for path in intervention_paths]
    if not paths:
        raise FileNotFoundError("no intervention files given")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing intervention files: {missing}")
    groups, assignment, files = _intervention_groups(paths, horizons)
    per_horizon = {
        str(horizon): _horizon_statistics(groups[horizon])
        for horizon in sorted(groups)
    }
    if not per_horizon:
        raise ValueError(
            "no samples found in the given intervention files (check the "
            "--horizons filter and that the long-horizon evaluation finished)"
        )

    def _average(key: str) -> float | None:
        values = [
            entry[key]
            for entry in per_horizon.values()
            if entry.get(key) is not None
        ]
        return float(np.mean(values)) if values else None

    total_candidates = sum(
        int(entry["n_candidates"]) for entry in per_horizon.values()
    )
    return {
        "continuous_snr": _average("continuous_snr"),
        "macro_snr": _average("macro_snr"),
        "macro_assignment": assignment,
        "macro_snr_within_as_noise": _average("macro_snr_within_as_noise"),
        "variance_explained_by_macro": _average("variance_explained_by_macro"),
        "n_states": int(sum(int(entry["n_states"]) for entry in per_horizon.values())),
        "n_candidates": int(total_candidates),
        "files": files,
        "per_horizon": per_horizon,
        "macro_actions": {
            name: {
                "mutation_operator": str(macro["mutation_operator"]),
                "multiplier": float(macro["multiplier"]),
                "exploration_strength": float(macro["exploration_strength"]),
            }
            for name, macro in MACRO_ACTIONS.items()
        },
        "macro_counts": {
            name: sum(
                int(entry["macro_counts"].get(name, 0))
                for entry in per_horizon.values()
            )
            for name in MACRO_ACTIONS
        },
        "definitions": {
            "continuous_snr": (
                "per state: var(candidate mean reward, ddof=1) / "
                "mean(per-candidate replicate variance, ddof=1); averaged "
                "over states and horizons (Phase-2B D4 statistic)"
            ),
            "macro_snr": (
                "same ratio with the per-macro mean rewards in the numerator; "
                "bounded above by continuous_snr up to the noise term because "
                "macro means average their members (law of total variance)"
            ),
            "macro_snr_within_as_noise": (
                "pooled state-centred between-macro variance / (within-macro "
                "variance + replicate noise); the decision-relevant macro SNR, "
                "comparable to continuous_snr"
            ),
            "variance_explained_by_macro": (
                "pooled between-macro variance / (between + within); how "
                "homogeneous the macro regions are"
            ),
            "assignment_rule": (
                "nearest macro inside the same mutation-operator family; "
                "distance = hypot(normalized log-multiplier distance, "
                "normalized exploration distance)"
            ),
            "n_states": (
                "sum over horizons of the per-horizon state counts (a state "
                "observed at three horizons counts three times); "
                "per_horizon[h].n_states is the count for one horizon"
            ),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments of the macro-action SNR analysis."""
    parser = argparse.ArgumentParser(
        prog="python -m controller.macro_actions",
        description=(
            "Phase 2.75 Task 3: SNR of the discrete macro action space vs the "
            "continuous action space on same-snapshot intervention data."
        ),
    )
    parser.add_argument(
        "--input-dir", type=str, default="results/phase2b/counterfactual",
        help="Directory with counterfactual_horizon_*.json files "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--out", type=str, default=DEFAULT_OUT,
        help="Output JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=None,
        help="Horizon filter; default: every horizon present in the files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point: measure macro vs continuous SNR and write the artifact."""
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob("counterfactual_horizon_*.json"))
    if not paths:
        raise FileNotFoundError(
            f"no counterfactual_horizon_*.json in {input_dir}"
        )
    payload = macro_snr_table(paths, horizons=args.horizons)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(
        f"[macro] continuous_snr={payload['continuous_snr']} "
        f"macro_snr={payload['macro_snr']} "
        f"macro_snr_within_as_noise={payload['macro_snr_within_as_noise']} "
        f"variance_explained_by_macro={payload['variance_explained_by_macro']}"
    )
    for horizon, entry in payload["per_horizon"].items():
        print(
            f"[macro] h={horizon}: n_states={entry['n_states']} "
            f"n_candidates={entry['n_candidates']} "
            f"continuous={entry['continuous_snr']} macro={entry['macro_snr']} "
            f"explained={entry['variance_explained_by_macro']}"
        )
    print(f"[done] wrote {out_path}")
    return payload


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
