"""Phase 0/1: generate (state, action, reward) trajectory datasets with NSGA-II.

For each (problem, seed) pair this script runs NSGA-II with a chosen action
policy, records every generation with
:class:`trajectory.recorder.EvolutionRecorder`, and writes one JSON
trajectory file per run plus an ``index.json`` summarizing the invocation.

Policies:

* ``fixed`` (default): the Phase-0 baseline — mutation probability stays at
  the resolved ``OperatorConfig`` default (``1 / n_vars``) for all
  generations.
* ``random``: Phase-1 action randomization — each generation the mutation
  probability is sampled as ``(1 / n_vars) * exp(U(log lo, log hi))`` with
  ``(lo, hi) = --pm-mult-range`` and injected via
  ``NSGAII.step(mutation_prob=pm_t)``. Sampling uses a dedicated generator
  seeded by ``(seed, crc32(problem_name))`` so runs are reproducible.

The stored ``config`` dict fully determines the run (problem, algorithm,
population size, generation count, resolved operator settings, policy
settings, reference point, reference-front size, and a UTC timestamp),
satisfying the EvoController experiment-recording rule (AGENTS.md Rule 2).

Example:
    ``python experiments/generate_dataset.py`` runs the full default grid:
    5 ZDT problems x seeds {0..4} x 100 generations x pop 100.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    # Allow ``python experiments/generate_dataset.py`` from the repo root:
    # the script directory (not the repo root) is on sys.path in that mode.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.nsga2 import NSGAII, OperatorConfig
from benchmarks import get_problem
from trajectory.recorder import EvolutionRecorder

#: Default problem grid (all Phase-0 ZDT benchmarks).
DEFAULT_PROBLEMS: tuple[str, ...] = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")
#: Default seed grid; AGENTS.md requires at least 5 random seeds per experiment.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for dataset generation.

    Args:
        argv: Argument list to parse; ``None`` reads ``sys.argv``.

    Returns:
        Parsed namespace with problems, seeds, generations, pop_size,
        out_dir, n_reference_points, ref_point, policy, and pm_mult_range.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate NSGA-II trajectories (state, action, reward) "
            "on ZDT benchmarks for the EvoController dataset."
        )
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=list(DEFAULT_PROBLEMS),
        help="Benchmark problems (default: %(default)s).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds (default: %(default)s).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of NSGA-II generations per run (default: %(default)s).",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=100,
        help="Population size (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/trajectory",
        help="Output directory for trajectory JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--n-reference-points",
        type=int,
        default=200,
        help="Points sampled from the true Pareto front for IGD (default: %(default)s).",
    )
    parser.add_argument(
        "--ref-point",
        nargs=2,
        type=float,
        default=[1.1, 1.1],
        metavar=("REF_F1", "REF_F2"),
        help="Hypervolume reference point (default: %(default)s).",
    )
    parser.add_argument(
        "--policy",
        choices=("fixed", "random"),
        default="fixed",
        help=(
            "Action policy: 'fixed' keeps the default mutation probability "
            "(Phase-0 baseline); 'random' samples a per-generation mutation "
            "probability around 1/n_vars (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--pm-mult-range",
        nargs=2,
        type=float,
        default=[0.5, 5.0],
        metavar=("LO", "HI"),
        help=(
            "Log-uniform multiplier range for the random policy: "
            "pm_t = (1/n_vars) * exp(U(log LO, log HI)) (default: %(default)s)."
        ),
    )
    return parser.parse_args(argv)


def build_config(
    problem_name: str,
    n_vars: int,
    pop_size: int,
    generations: int,
    operators: OperatorConfig,
    mutation_probability: float,
    ref_point: np.ndarray,
    n_reference_points: int,
    policy: str,
    pm_mult_range: tuple[float, float],
    base_mutation_prob: float,
) -> dict[str, Any]:
    """Build the configuration dict stored inside each trajectory JSON.

    The config fully determines the run: every operator setting is resolved
    (including the effective mutation probability), the action policy and
    its sampling range are recorded, and a UTC timestamp is attached for
    provenance.

    Args:
        problem_name: Benchmark identifier (e.g. ``"zdt1"``).
        n_vars: Number of decision variables of the problem.
        pop_size: NSGA-II population size.
        generations: Number of generations executed.
        operators: Variation operator configuration used by the run.
        mutation_probability: Effective per-variable mutation probability
            (resolved from ``operators.mutation_prob``).
        ref_point: Hypervolume reference point, shape ``(2,)``.
        n_reference_points: Size of the IGD reference front sample.
        policy: Action policy identifier (``"fixed"`` or ``"random"``).
        pm_mult_range: Log-uniform multiplier range ``(lo, hi)`` used by the
            random policy; recorded for both policies.
        base_mutation_prob: Base per-variable mutation probability
            ``1 / n_vars`` around which the random policy samples.

    Returns:
        JSON-serializable configuration dict.
    """
    return {
        "problem": problem_name,
        "n_vars": int(n_vars),
        "algorithm": "nsga2",
        "pop_size": int(pop_size),
        "generations": int(generations),
        "operators": {
            "crossover_operator": operators.crossover_operator,
            "crossover_prob": float(operators.crossover_prob),
            "mutation_operator": operators.mutation_operator,
            "mutation_probability": float(mutation_probability),
            "eta_c": float(operators.eta_c),
            "eta_m": float(operators.eta_m),
        },
        "policy": str(policy),
        "pm_mult_range": [float(pm_mult_range[0]), float(pm_mult_range[1])],
        "base_mutation_prob": float(base_mutation_prob),
        "ref_point": [float(ref_point[0]), float(ref_point[1])],
        "n_reference_points": int(n_reference_points),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def make_policy_rng(problem_name: str, seed: int) -> np.random.Generator:
    """Create the dedicated action-sampling generator of the random policy.

    Seeded by ``(seed, crc32(problem_name))`` so that draws are
    reproducible per (problem, seed) pair and independent of the NSGA-II
    generator (action sampling never perturbs the algorithm's randomness).

    Args:
        problem_name: Benchmark identifier; hashed with ``zlib.crc32``
            (stable across platforms and Python runs).
        seed: Random seed of the run.

    Returns:
        A fresh ``numpy.random.Generator`` (PCG64).
    """
    name_hash = zlib.crc32(problem_name.encode("utf-8"))
    return np.random.Generator(np.random.PCG64([seed & 0xFFFFFFFFFFFFFFFF, name_hash]))


def sample_random_mutation_prob(
    rng: np.random.Generator, base_mutation_prob: float, pm_mult_range: tuple[float, float]
) -> float:
    """Sample one generation's mutation probability for the random policy.

    ``pm = base_mutation_prob * exp(U(log lo, log hi))``: log-uniform in the
    multiplier so multiplicative up- and down-moves are equally likely.

    Args:
        rng: The dedicated policy generator (see :func:`make_policy_rng`).
        base_mutation_prob: Base per-variable probability ``1 / n_vars``.
        pm_mult_range: Multiplier range ``(lo, hi)`` with ``0 < lo <= hi``.

    Returns:
        The sampled per-variable mutation probability.
    """
    lo, hi = float(pm_mult_range[0]), float(pm_mult_range[1])
    return base_mutation_prob * float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def run_single(
    problem_name: str,
    seed: int,
    generations: int,
    pop_size: int,
    n_reference_points: int,
    ref_point: np.ndarray,
    out_dir: Path,
    policy: str = "fixed",
    pm_mult_range: tuple[float, float] = (0.5, 5.0),
) -> dict[str, Any]:
    """Run one NSGA-II trajectory and save it to JSON.

    Initializes the population, records generation 0, then performs
    ``generations`` steps, recording after each step. Under the ``"random"``
    policy each step's mutation probability is sampled from a dedicated
    generator and injected via ``step(mutation_prob=pm_t)``; the recorded
    action is the value actually used (``algorithm.current_action()`` after
    the step). Wall-clock runtime covers the full run (initialization +
    evolution loop).

    Args:
        problem_name: Benchmark identifier accepted by ``get_problem``.
        seed: Random seed for the run.
        generations: Number of NSGA-II generations to execute.
        pop_size: Population size.
        n_reference_points: Points sampled from the true Pareto front.
        ref_point: Hypervolume reference point, shape ``(2,)``.
        out_dir: Output directory; the file is written to
            ``{out_dir}/{problem_name}_nsga2_seed{seed}.json``.
        policy: ``"fixed"`` (constant default mutation probability) or
            ``"random"`` (log-uniform per-generation sampling).
        pm_mult_range: Multiplier range ``(lo, hi)`` for the random policy;
            must satisfy ``0 < lo <= hi``.

    Returns:
        Summary dict with keys ``file``, ``problem``, ``seed``,
        ``final_hv``, ``final_igd``, ``runtime_sec``.

    Raises:
        ValueError: If ``policy`` is unknown or ``pm_mult_range`` is invalid.
    """
    if policy not in ("fixed", "random"):
        raise ValueError(f"unsupported policy {policy!r}; expected 'fixed' or 'random'")
    lo, hi = float(pm_mult_range[0]), float(pm_mult_range[1])
    if not lo > 0.0 or hi < lo:
        raise ValueError(f"pm_mult_range must satisfy 0 < lo <= hi, got {(lo, hi)}")

    problem = get_problem(problem_name)
    operators = OperatorConfig()
    algorithm = NSGAII(problem, pop_size=pop_size, operators=operators, seed=seed)
    recorder = EvolutionRecorder(
        problem_name=problem.name,
        reference_front=problem.reference_front(n_points=n_reference_points),
        ref_point=ref_point,
    )
    base_pm = 1.0 / problem.n_vars
    policy_rng = make_policy_rng(problem.name, seed) if policy == "random" else None

    start = time.perf_counter()
    algorithm.initialize()
    recorder.record(algorithm.generation, algorithm.nondominated_front(), algorithm.current_action())
    for _ in range(generations):
        if policy_rng is not None:
            pm_t = sample_random_mutation_prob(policy_rng, base_pm, (lo, hi))
            algorithm.step(mutation_prob=pm_t)
        else:
            algorithm.step()
        recorder.record(algorithm.generation, algorithm.nondominated_front(), algorithm.current_action())
    runtime_sec = time.perf_counter() - start

    config = build_config(
        problem_name=problem.name,
        n_vars=problem.n_vars,
        pop_size=pop_size,
        generations=generations,
        operators=operators,
        mutation_probability=base_pm,
        ref_point=ref_point,
        n_reference_points=n_reference_points,
        policy=policy,
        pm_mult_range=(lo, hi),
        base_mutation_prob=base_pm,
    )

    out_path = out_dir / f"{problem.name}_nsga2_seed{seed}.json"
    recorder.save(out_path, config=config, seed=seed, runtime_sec=runtime_sec)

    transitions = recorder.transitions()
    summary = {
        "file": out_path.name,
        "problem": problem.name,
        "seed": int(seed),
        "final_hv": float(transitions[-1]["state"]["hv"]),
        "final_igd": float(transitions[-1]["state"]["igd"]),
        "runtime_sec": float(runtime_sec),
    }
    print(
        f"[done] {problem.name} seed={seed} gens={generations} pop={pop_size} "
        f"hv={summary['final_hv']:.6f} igd={summary['final_igd']:.6f} "
        f"({runtime_sec:.2f}s) -> {out_path.name}"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Run the full (problem, seed) grid and write the trajectory dataset.

    Args:
        argv: Optional argument list; ``None`` reads ``sys.argv``.

    Returns:
        List of per-run summary dicts (also written to ``index.json``).
    """
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_point = np.asarray(args.ref_point, dtype=float)

    print(
        f"Phase-0/1 dataset generation: {len(args.problems)} problems x "
        f"{len(args.seeds)} seeds x {args.generations} generations x "
        f"pop {args.pop_size}; policy={args.policy} out_dir={out_dir}"
    )

    summaries: list[dict[str, Any]] = []
    for problem_name in args.problems:
        for seed in args.seeds:
            summaries.append(
                run_single(
                    problem_name=problem_name,
                    seed=seed,
                    generations=args.generations,
                    pop_size=args.pop_size,
                    n_reference_points=args.n_reference_points,
                    ref_point=ref_point,
                    out_dir=out_dir,
                    policy=args.policy,
                    pm_mult_range=(float(args.pm_mult_range[0]), float(args.pm_mult_range[1])),
                )
            )

    index_path = out_dir / "index.json"
    index_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(summaries),
        "runs": summaries,
    }
    with index_path.open("w", encoding="utf-8") as fh:
        json.dump(index_payload, fh, indent=2, ensure_ascii=False)
    print(f"[done] wrote index for {len(summaries)} runs -> {index_path}")
    return summaries


if __name__ == "__main__":
    main()
