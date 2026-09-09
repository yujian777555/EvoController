"""Phase 1.75: tune matched full-action static baselines on training seeds only.

This script performs a seeded random search over the full Phase-1.5 action
space — mutation operator in ``{polynomial, gaussian}``, mutation-probability
multiplier log-uniform in ``[0.25, 8.0]`` (on the natural NSGA-II scale
``1 / n_vars``), and exploration strength log-uniform in the operator-specific
range (``eta_m in [2, 50]`` polynomial, ``sigma in [0.02, 0.3]`` Gaussian) —
and selects the static tuple that maximizes mean anytime AUC-HV (trapezoidal
integral of the per-generation hypervolume, normalized by the number of
generations), breaking ties by final hypervolume.

Scientific-control guarantees:

* Candidates are evaluated by running NSGA-II on the tuning seeds given by
  ``--seeds`` ONLY. ``--held-out-seeds`` (default ``1000..1019``) declares
  the test set; any overlap with ``--seeds`` raises ``ValueError`` before a
  single run starts, so no baseline hyperparameter can be tuned on test
  seeds (Phase-1.75 global constraint).
* The search is fully deterministic: candidate ``i`` is drawn from
  ``np.random.Generator(np.random.PCG64([search_seed, i]))``, independent of
  the sharding layout, so a sharded search is bit-identical to a monolithic
  one. Shard artifacts contain no timestamps or runtimes and are byte
  reproducible.

Sharding: ``--num-shards K --shard-index I`` evaluates the candidates with
``index % K == I`` and writes ``{out_dir}/tuning_shard_{I}.json``.
``--aggregate --num-shards K`` merges shard files ``0..K-1`` (all must exist
and cover every candidate exactly once) into
``{out_dir}/static_full_tuning.json`` with schema::

    {
      "global":      {"operator", "multiplier", "exploration_strength",
                      "mean_auc_hv", "final_hv", "candidate_index"},
      "per_problem": {problem: {...same...}},
      "config":      {"seeds", "candidates", "problems", "generations",
                      "pop_size", "search_seed", "held_out_seeds", ...}
    }

``global`` is the candidate with the best mean AUC-HV averaged over all
problems (the ``static_full_global`` arm); ``per_problem`` holds the best
candidate per problem (the oracle-style ``static_full_per_problem`` arm).

Example:
    ``python experiments/tune_static_full.py --num-shards 4 --shard-index 0``
    ``python experiments/tune_static_full.py --aggregate --num-shards 4``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    # Allow ``python experiments/tune_static_full.py`` from the repo root: the
    # script directory (not the repo root) is on sys.path in that mode.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms import NSGAII, OperatorConfig
from benchmarks import get_problem
from controller.multihead_controller import (
    GAUSSIAN_EXPLORATION_RANGE,
    POLYNOMIAL_EXPLORATION_RANGE,
)
from controller.static_full_controller import SUPPORTED_OPERATORS, StaticFullController
from metrics import hypervolume

#: Default tuning problem grid (all Phase-0 ZDT benchmarks).
DEFAULT_PROBLEMS: tuple[str, ...] = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")
#: Default tuning seeds; disjoint from the Phase-1.75 held-out test seeds.
DEFAULT_SEEDS: tuple[int, ...] = (300, 301, 302)
#: Held-out test seeds of Phase 1.75; tuning on them is forbidden.
DEFAULT_HELD_OUT_SEEDS: tuple[int, ...] = tuple(range(1000, 1020))
#: Default number of random-search candidates.
DEFAULT_CANDIDATES = 128
#: Default output directory for tuning artifacts.
DEFAULT_OUT_DIR = "results/phase1_75"
#: Mutation-probability multiplier search range (on ``1 / n_vars``).
PM_MULT_RANGE: tuple[float, float] = (0.25, 8.0)
#: File-name template of the per-shard artifact inside the output directory.
SHARD_FILE_TEMPLATE = "tuning_shard_{index}.json"
#: File name of the merged tuning artifact inside the output directory.
AGGREGATE_FILE_NAME = "static_full_tuning.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the static full-action tuner.

    Args:
        argv: Argument list to parse; ``None`` reads ``sys.argv``.

    Returns:
        Parsed namespace with search, sharding, and output settings.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Phase-1.75: seeded random search for the best static full-action "
            "(operator, pm multiplier, exploration strength) tuple, evaluated "
            "on training seeds only (never the held-out test seeds)."
        )
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help=(
            "Merge tuning_shard_{0..K-1}.json under --out-dir into "
            f"{AGGREGATE_FILE_NAME} instead of running a search shard."
        ),
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=list(DEFAULT_PROBLEMS),
        help="Benchmark problems to tune on (default: %(default)s).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=(
            "Training/tuning seeds used for candidate evaluation "
            "(default: %(default)s). Must not overlap --held-out-seeds."
        ),
    )
    parser.add_argument(
        "--held-out-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_HELD_OUT_SEEDS),
        help=(
            "Held-out test seeds that tuning must never touch "
            "(default: 1000..1019)."
        ),
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CANDIDATES,
        help="Number of random-search candidates (default: %(default)s).",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=100,
        help="NSGA-II population size for candidate evaluation (default: %(default)s).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="NSGA-II generations per candidate evaluation (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Search RNG seed; candidate i is drawn from "
            "PCG64([seed, i]) (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=(
            "Total number of search shards; in search mode the candidates are "
            "split by index modulo, in --aggregate mode this many shard files "
            "are merged (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help=(
            "Index of this shard in search mode; evaluates candidates with "
            "index %% num_shards == shard_index (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for tuning artifacts (default: %(default)s).",
    )
    parser.add_argument(
        "--ref-point",
        nargs=2,
        type=float,
        default=[1.1, 1.1],
        metavar=("REF_F1", "REF_F2"),
        help="Hypervolume reference point (default: %(default)s).",
    )
    return parser.parse_args(argv)


def check_held_out_disjoint(seeds: Sequence[int], held_out_seeds: Sequence[int]) -> None:
    """Enforce that no tuning seed is a held-out test seed.

    Args:
        seeds: Seeds used for tuning (candidate evaluation).
        held_out_seeds: Declared held-out test seeds.

    Raises:
        ValueError: If the two sets overlap; the message lists the offending
            seeds.
    """
    overlap = sorted(set(int(s) for s in seeds) & set(int(s) for s in held_out_seeds))
    if overlap:
        raise ValueError(
            f"tuning seeds overlap the held-out test seeds: {overlap}; "
            "no baseline hyperparameter may be tuned on held-out seeds"
        )


def sample_candidate(index: int, search_seed: int) -> dict[str, Any]:
    """Draw one search candidate deterministically.

    Candidate ``index`` is drawn from its own stream
    ``np.random.Generator(np.random.PCG64([search_seed, index]))``, so the
    candidate set does not depend on the sharding layout: any shard subset
    reproduces exactly the candidates a monolithic run would evaluate.

    Args:
        index: Candidate index in ``[0, candidates)``.
        search_seed: Search RNG seed (the ``--seed`` CLI value).

    Returns:
        Dict with ``index``, ``operator`` (``"polynomial"`` or
        ``"gaussian"``), ``multiplier`` (log-uniform in
        :data:`PM_MULT_RANGE`), and ``exploration_strength`` (log-uniform in
        the operator-specific range of
        :data:`POLYNOMIAL_EXPLORATION_RANGE` /
        :data:`GAUSSIAN_EXPLORATION_RANGE`).
    """
    rng = np.random.Generator(np.random.PCG64([int(search_seed), int(index)]))
    operator = SUPPORTED_OPERATORS[int(rng.integers(0, len(SUPPORTED_OPERATORS)))]
    multiplier = float(
        math.exp(rng.uniform(math.log(PM_MULT_RANGE[0]), math.log(PM_MULT_RANGE[1])))
    )
    lo, hi = (
        POLYNOMIAL_EXPLORATION_RANGE if operator == "polynomial" else GAUSSIAN_EXPLORATION_RANGE
    )
    exploration = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return {
        "index": int(index),
        "operator": operator,
        "multiplier": multiplier,
        "exploration_strength": exploration,
    }


def shard_candidate_indices(
    candidates: int, shard_index: int, num_shards: int
) -> list[int]:
    """Return the candidate indices evaluated by one search shard.

    Sharding is by index modulo: shard ``i`` of ``k`` evaluates candidates
    ``i, i + k, i + 2k, ...``, a deterministic subset of the full grid.

    Args:
        candidates: Total number of candidates; must be >= 1.
        shard_index: Index of this shard; must satisfy
            ``0 <= shard_index < num_shards``.
        num_shards: Total number of shards; must be >= 1.

    Returns:
        Sorted list of candidate indices of this shard (possibly empty when
        there are more shards than candidates).

    Raises:
        ValueError: On out-of-range shard configuration.
    """
    if int(candidates) < 1:
        raise ValueError(f"candidates must be >= 1, got {candidates}")
    if int(num_shards) < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= int(shard_index) < int(num_shards):
        raise ValueError(
            f"shard_index must satisfy 0 <= shard_index < num_shards, "
            f"got shard_index={shard_index}, num_shards={num_shards}"
        )
    return [i for i in range(int(candidates)) if i % int(num_shards) == int(shard_index)]


def _auc_hv_series(hv_series: Sequence[float], generations: int) -> float:
    """Anytime AUC-HV of a per-generation HV series, generation 0 included.

    Trapezoidal integral over the generation axis normalized by
    ``generations`` — identical to :func:`experiments.run_phase1._auc_hv`
    for contiguous per-generation records, but computed directly from the
    HV series (no recorder transitions needed).

    Args:
        hv_series: Hypervolume after each generation, starting at
            generation 0.
        generations: Number of NSGA-II generations executed (the divisor).

    Returns:
        Normalized AUC-HV; the final HV for degenerate one-point series.
    """
    hv = np.asarray(hv_series, dtype=float)
    if hv.size < 2 or generations <= 0:
        return float(hv[-1]) if hv.size else 0.0
    integral = float(np.sum(0.5 * (hv[:-1] + hv[1:])))
    return integral / float(generations)


def _evaluate_candidate_on_seed(
    candidate: dict[str, Any],
    problem_name: str,
    seed: int,
    *,
    pop_size: int,
    generations: int,
    ref_point: np.ndarray,
) -> dict[str, float]:
    """Run one NSGA-II evaluation of one static candidate on one problem/seed.

    The candidate's action triple is produced by a
    :class:`StaticFullController` and injected via
    ``NSGAII.step(mutation_prob=..., mutation_operator=...,
    exploration_strength=...)`` at every generation, exactly matching the
    deployment semantics of the closed-loop controller arms. The
    hypervolume of the nondominated front is recorded after
    ``initialize()`` (generation 0) and after every generation.

    Args:
        candidate: Candidate dict as returned by :func:`sample_candidate`.
        problem_name: Benchmark identifier accepted by ``get_problem``.
        seed: NSGA-II random seed (a tuning seed, never a held-out seed).
        pop_size: Population size.
        generations: Number of generations to execute.
        ref_point: Hypervolume reference point, shape ``(2,)``.

    Returns:
        Dict with ``auc_hv`` (normalized anytime AUC-HV) and ``final_hv``.
    """
    problem = get_problem(problem_name)
    controller = StaticFullController(
        candidate["operator"], candidate["multiplier"], candidate["exploration_strength"]
    )
    action = controller.predict_action(n_vars=problem.n_vars)
    algorithm = NSGAII(problem, pop_size=pop_size, operators=OperatorConfig(), seed=seed)
    algorithm.initialize()
    hv_series = [hypervolume(algorithm.nondominated_front(), ref_point)]
    for _ in range(int(generations)):
        algorithm.step(
            mutation_prob=action["mutation_probability"],
            mutation_operator=action["mutation_operator"],
            exploration_strength=action["exploration_strength"],
        )
        hv_series.append(hypervolume(algorithm.nondominated_front(), ref_point))
    return {
        "auc_hv": _auc_hv_series(hv_series, int(generations)),
        "final_hv": float(hv_series[-1]),
    }


def evaluate_candidate(
    candidate: dict[str, Any],
    problems: Sequence[str],
    seeds: Sequence[int],
    *,
    pop_size: int,
    generations: int,
    ref_point: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one candidate on the full (problem, tuning seed) grid.

    Args:
        candidate: Candidate dict as returned by :func:`sample_candidate`.
        problems: Normalized problem names.
        seeds: Tuning seeds (held-out disjointness already enforced).
        pop_size: Population size.
        generations: Number of generations per run.
        ref_point: Hypervolume reference point, shape ``(2,)``.

    Returns:
        The candidate dict augmented with ``per_problem`` (per problem:
        ``mean_auc_hv`` and ``final_hv`` averaged over seeds plus raw
        ``per_seed`` values) and the cross-problem aggregates
        ``global_auc_hv`` / ``global_final_hv`` used for global ranking.
    """
    per_problem: dict[str, Any] = {}
    for problem_name in problems:
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            metrics = _evaluate_candidate_on_seed(
                candidate,
                problem_name,
                int(seed),
                pop_size=pop_size,
                generations=generations,
                ref_point=ref_point,
            )
            per_seed[str(int(seed))] = metrics
        auc_values = [m["auc_hv"] for m in per_seed.values()]
        final_values = [m["final_hv"] for m in per_seed.values()]
        per_problem[problem_name] = {
            "mean_auc_hv": float(np.mean(auc_values)),
            "final_hv": float(np.mean(final_values)),
            "per_seed": per_seed,
        }
    result = dict(candidate)
    result["per_problem"] = per_problem
    result["global_auc_hv"] = float(
        np.mean([per_problem[p]["mean_auc_hv"] for p in problems])
    )
    result["global_final_hv"] = float(
        np.mean([per_problem[p]["final_hv"] for p in problems])
    )
    return result


def _search_config(args: argparse.Namespace, candidate_indices: list[int]) -> dict[str, Any]:
    """Assemble the deterministic configuration block of a shard artifact.

    Contains everything that determines the search outcome (seeds, held-out
    declaration, candidate count, problems, budget, search seed, reference
    point, sharding layout). No timestamps or runtimes are stored, so two
    identical invocations produce byte-identical shard artifacts.
    """
    return {
        "seeds": [int(s) for s in args.seeds],
        "held_out_seeds": [int(s) for s in args.held_out_seeds],
        "problems": [get_problem(name).name for name in args.problems],
        "candidates": int(args.candidates),
        "pop_size": int(args.pop_size),
        "generations": int(args.generations),
        "search_seed": int(args.seed),
        "ref_point": [float(args.ref_point[0]), float(args.ref_point[1])],
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "candidate_indices": [int(i) for i in candidate_indices],
    }


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate this shard's candidates and write the shard artifact.

    Enforces the held-out seed guarantee before any evaluation, draws the
    shard's deterministic candidate subset, evaluates each candidate on the
    (problem, tuning seed) grid, and writes
    ``{out_dir}/tuning_shard_{shard_index}.json``.

    Args:
        args: Parsed arguments as produced by :func:`parse_args`.

    Returns:
        The shard payload exactly as written to disk.
    """
    check_held_out_disjoint(args.seeds, args.held_out_seeds)
    indices = shard_candidate_indices(args.candidates, args.shard_index, args.num_shards)
    problems = [get_problem(name).name for name in args.problems]
    seeds = [int(s) for s in args.seeds]
    ref_point = np.asarray(args.ref_point, dtype=float)

    evaluated: list[dict[str, Any]] = []
    for index in indices:
        candidate = sample_candidate(index, args.seed)
        result = evaluate_candidate(
            candidate,
            problems,
            seeds,
            pop_size=int(args.pop_size),
            generations=int(args.generations),
            ref_point=ref_point,
        )
        evaluated.append(result)
        print(
            f"[candidate {index}] {result['operator']} mult={result['multiplier']:.4f} "
            f"expl={result['exploration_strength']:.4f} "
            f"global_auc_hv={result['global_auc_hv']:.6f}"
        )

    payload: dict[str, Any] = {
        "kind": "static_full_tuning_shard",
        "config": _search_config(args, indices),
        "candidates": evaluated,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / SHARD_FILE_TEMPLATE.format(index=int(args.shard_index))
    with shard_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[done] wrote {len(evaluated)} candidates -> {shard_path}")
    return payload


#: Shard config keys that must agree across all merged shard files.
_CONSISTENT_CONFIG_KEYS: tuple[str, ...] = (
    "seeds",
    "held_out_seeds",
    "problems",
    "candidates",
    "pop_size",
    "generations",
    "search_seed",
    "ref_point",
    "num_shards",
)


def _load_shards(out_dir: Path, num_shards: int) -> list[dict[str, Any]]:
    """Load and validate the ``num_shards`` shard artifacts under ``out_dir``.

    Every expected shard file must exist, all shards must agree on the
    search-defining config keys (:data:`_CONSISTENT_CONFIG_KEYS`), each
    shard's tuning seeds must be disjoint from its declared held-out seeds,
    and the union of evaluated candidate indices must cover
    ``range(candidates)`` exactly once (no missing or duplicated
    candidates — a merge over a partial search is never produced silently).

    Args:
        out_dir: Directory containing the shard artifacts.
        num_shards: Number of expected shard files.

    Returns:
        The shard payloads ordered by shard index.

    Raises:
        FileNotFoundError: If an expected shard file is missing.
        ValueError: If shard configs disagree, held-out disjointness is
            violated, or candidate coverage is incomplete/duplicated.
    """
    shards: list[dict[str, Any]] = []
    for index in range(int(num_shards)):
        path = out_dir / SHARD_FILE_TEMPLATE.format(index=index)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing tuning shard: {path}; run the search with "
                f"--num-shards {num_shards} --shard-index {index} first"
            )
        with path.open("r", encoding="utf-8") as fh:
            shards.append(json.load(fh))

    reference = shards[0]["config"]
    for index, shard in enumerate(shards):
        config = shard["config"]
        for key in _CONSISTENT_CONFIG_KEYS:
            if config.get(key) != reference.get(key):
                raise ValueError(
                    f"shard {index} config {key!r}={config.get(key)!r} disagrees "
                    f"with shard 0 ({reference.get(key)!r}); shards of different "
                    "searches must not be merged"
                )
        check_held_out_disjoint(config["seeds"], config["held_out_seeds"])

    covered: list[int] = []
    for shard in shards:
        covered.extend(int(c["index"]) for c in shard["candidates"])
    expected = list(range(int(reference["candidates"])))
    if sorted(covered) != expected:
        raise ValueError(
            f"candidate coverage mismatch: expected indices 0..{len(expected) - 1} "
            f"exactly once, got {sorted(covered)}"
        )
    return shards


def _winner_entry(candidate: dict[str, Any], problem: str | None) -> dict[str, Any]:
    """Format one winning candidate as an aggregate-artifact entry.

    Args:
        candidate: Evaluated candidate dict (see :func:`evaluate_candidate`).
        problem: ``None`` for the cross-problem global entry (uses
            ``global_auc_hv`` / ``global_final_hv``), otherwise the problem
            name (uses the per-problem means).
    """
    if problem is None:
        mean_auc_hv = float(candidate["global_auc_hv"])
        final_hv = float(candidate["global_final_hv"])
    else:
        mean_auc_hv = float(candidate["per_problem"][problem]["mean_auc_hv"])
        final_hv = float(candidate["per_problem"][problem]["final_hv"])
    return {
        "operator": candidate["operator"],
        "multiplier": float(candidate["multiplier"]),
        "exploration_strength": float(candidate["exploration_strength"]),
        "mean_auc_hv": mean_auc_hv,
        "final_hv": final_hv,
        "candidate_index": int(candidate["index"]),
    }


def run_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    """Merge all shard artifacts into ``static_full_tuning.json``.

    The global winner maximizes ``(global_auc_hv, global_final_hv)`` (lexico-
    graphic: mean anytime AUC-HV first, final HV as tie-breaker); each
    per-problem winner maximizes ``(mean_auc_hv, final_hv)`` of that problem.
    The merged artifact is deterministic given the shard files.

    Args:
        args: Parsed arguments; only ``--num-shards`` and ``--out-dir`` are
            used (the merged configuration is taken from the shard files,
            which the search already validated).

    Returns:
        The aggregate payload exactly as written to disk.
    """
    out_dir = Path(args.out_dir)
    shards = _load_shards(out_dir, args.num_shards)
    config = shards[0]["config"]
    problems = [str(p) for p in config["problems"]]
    candidates = [c for shard in shards for c in shard["candidates"]]

    global_best = max(
        candidates, key=lambda c: (c["global_auc_hv"], c["global_final_hv"])
    )
    per_problem: dict[str, Any] = {}
    for problem in problems:
        best = max(
            candidates,
            key=lambda c: (
                c["per_problem"][problem]["mean_auc_hv"],
                c["per_problem"][problem]["final_hv"],
            ),
        )
        per_problem[problem] = _winner_entry(best, problem)

    payload: dict[str, Any] = {
        "kind": "static_full_tuning",
        "global": _winner_entry(global_best, None),
        "per_problem": per_problem,
        "config": {
            "seeds": [int(s) for s in config["seeds"]],
            "candidates": int(config["candidates"]),
            "problems": problems,
            "generations": int(config["generations"]),
            "pop_size": int(config["pop_size"]),
            "search_seed": int(config["search_seed"]),
            "held_out_seeds": [int(s) for s in config["held_out_seeds"]],
            "ref_point": [float(v) for v in config["ref_point"]],
            "num_shards": int(config["num_shards"]),
        },
    }
    out_path = out_dir / AGGREGATE_FILE_NAME
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(
        f"[done] best global: {payload['global']['operator']} "
        f"mult={payload['global']['multiplier']:.4f} "
        f"expl={payload['global']['exploration_strength']:.4f} "
        f"mean_auc_hv={payload['global']['mean_auc_hv']:.6f} -> {out_path}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point: run one search shard or aggregate the shard files.

    Args:
        argv: Optional argument list; ``None`` reads ``sys.argv``.

    Returns:
        The shard payload (search mode) or aggregate payload
        (``--aggregate`` mode) exactly as written to disk.
    """
    args = parse_args(argv)
    if args.aggregate:
        return run_aggregate(args)
    return run_search(args)


if __name__ == "__main__":
    main()
