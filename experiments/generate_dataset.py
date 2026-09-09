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

Action spaces (``--action-space``):

* ``pm`` (default): only the mutation probability is controlled (Phase-1
  behavior).
* ``full``: Phase-1.5 expanded action space — each generation additionally
  samples the mutation operator uniformly from {polynomial, gaussian} and
  an exploration strength (eta_m for polynomial from
  ``exp(U(log 2, log 50))``, sigma for Gaussian from
  ``exp(U(log 0.02, log 0.3))``), injected via
  ``NSGAII.step(mutation_prob=..., mutation_operator=...,
  exploration_strength=...)``. Selecting ``full`` implies per-generation
  randomized actions and widens the default ``--pm-mult-range`` to
  ``0.25 8.0``.

The stored ``config`` dict fully determines the run (problem, ``n_vars``,
algorithm, population size, generation count, resolved operator settings,
policy settings, reference point, reference-front size, and a UTC
timestamp), satisfying the EvoController experiment-recording rule
(AGENTS.md Rule 2).

Phase 1.75 additions:

* Every recorded action dict carries ``mutation_multiplier``, the
  scale-normalized mutation action ``mutation_probability * n_vars``
  (the natural NSGA-II scale ``1 / n_vars`` maps to multiplier 1.0),
  computed from the action actually used in each generation.
* ``--seed-range START STOP`` expands to the inclusive seed list
  ``START..STOP``; it is mutually exclusive with ``--seeds``.
* ``index.json`` additionally reports per-problem run counts:
  ``n_runs``, ``n_success`` (final HV > 0), ``n_failed``, and
  ``zero_hv_count`` (final HV == 0). A per-problem failure threshold is
  deliberately not decided at generation time, so only the exact zero-HV
  indicator is recorded here.

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
#: Default log-uniform pm multiplier range of the pm action space.
DEFAULT_PM_MULT_RANGE: tuple[float, float] = (0.5, 5.0)
#: Widened default pm multiplier range of the full action space (Phase 1.5).
FULL_ACTION_PM_MULT_RANGE: tuple[float, float] = (0.25, 8.0)
#: Log-uniform sampling range of the polynomial eta_m exploration strength.
ETA_M_SAMPLE_RANGE: tuple[float, float] = (2.0, 50.0)
#: Log-uniform sampling range of the Gaussian sigma exploration strength.
SIGMA_SAMPLE_RANGE: tuple[float, float] = (0.02, 0.3)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for dataset generation.

    ``--seeds`` and ``--seed-range START STOP`` are mutually exclusive;
    ``--seed-range`` expands to the inclusive list ``START..STOP`` and is
    stored into ``args.seeds`` so downstream code sees one resolved seed
    list. When neither is given, ``args.seeds`` falls back to
    ``DEFAULT_SEEDS``.

    Args:
        argv: Argument list to parse; ``None`` reads ``sys.argv``.

    Returns:
        Parsed namespace with problems, seeds (resolved), seed_range,
        generations, pop_size, out_dir, n_reference_points, ref_point,
        policy, action_space, and pm_mult_range.

    Raises:
        SystemExit: If both ``--seeds`` and ``--seed-range`` are given, or
            ``--seed-range`` has ``START > STOP`` (argparse error, exit
            code 2).
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
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds (default: %s)." % (list(DEFAULT_SEEDS),),
    )
    seed_group.add_argument(
        "--seed-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "STOP"),
        help=(
            "Inclusive seed range; expands to seeds START..STOP "
            "(mutually exclusive with --seeds)."
        ),
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
        "--action-space",
        choices=("pm", "full"),
        default="pm",
        help=(
            "Controller action space: 'pm' varies only the mutation "
            "probability (Phase-1 behavior); 'full' additionally samples "
            "the mutation operator and the exploration strength each "
            "generation (Phase-1.5). 'full' implies randomized actions "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--pm-mult-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help=(
            "Log-uniform multiplier range for the randomized policies: "
            "pm_t = (1/n_vars) * exp(U(log LO, log HI)). Defaults to "
            "0.5 5.0 for --action-space pm and to 0.25 8.0 for "
            "--action-space full."
        ),
    )
    args = parser.parse_args(argv)
    if args.seed_range is not None:
        start, stop = int(args.seed_range[0]), int(args.seed_range[1])
        if stop < start:
            parser.error(
                f"--seed-range requires START <= STOP, got {start} {stop}"
            )
        args.seeds = list(range(start, stop + 1))
    elif args.seeds is None:
        args.seeds = list(DEFAULT_SEEDS)
    return args


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
    action_space: str,
) -> dict[str, Any]:
    """Build the configuration dict stored inside each trajectory JSON.

    The config fully determines the run: every operator setting is resolved
    (including the effective mutation probability and the Gaussian sigma
    default), the action policy, action space, and sampling ranges are
    recorded, and a UTC timestamp is attached for provenance.

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
            randomized policies; recorded for all policies.
        base_mutation_prob: Base per-variable mutation probability
            ``1 / n_vars`` around which randomized policies sample.
        action_space: Controller action space (``"pm"`` or ``"full"``).

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
            "gaussian_sigma": float(operators.gaussian_sigma),
        },
        "policy": str(policy),
        "action_space": str(action_space),
        "pm_mult_range": [float(pm_mult_range[0]), float(pm_mult_range[1])],
        "base_mutation_prob": float(base_mutation_prob),
        "exploration_ranges": {
            "polynomial_eta_m": [float(ETA_M_SAMPLE_RANGE[0]), float(ETA_M_SAMPLE_RANGE[1])],
            "gaussian_sigma": [float(SIGMA_SAMPLE_RANGE[0]), float(SIGMA_SAMPLE_RANGE[1])],
        },
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


def sample_full_action(
    rng: np.random.Generator, base_mutation_prob: float, pm_mult_range: tuple[float, float]
) -> tuple[str, float, float]:
    """Sample one generation's action of the full (Phase-1.5) action space.

    The mutation operator is drawn uniformly from {polynomial, gaussian},
    the mutation probability is log-uniform in the multiplier around
    ``base_mutation_prob`` (see :func:`sample_random_mutation_prob`), and
    the exploration strength is log-uniform in
    ``ETA_M_SAMPLE_RANGE = (2, 50)`` for polynomial (eta_m) or
    ``SIGMA_SAMPLE_RANGE = (0.02, 0.3)`` for Gaussian (sigma). Draws happen
    in the fixed order operator -> pm -> exploration strength so sequences
    are reproducible given the generator seed.

    Args:
        rng: The dedicated policy generator (see :func:`make_policy_rng`).
        base_mutation_prob: Base per-variable probability ``1 / n_vars``.
        pm_mult_range: Multiplier range ``(lo, hi)`` with ``0 < lo <= hi``.

    Returns:
        ``(mutation_operator, mutation_prob, exploration_strength)``.
    """
    mutation_operator = "polynomial" if int(rng.integers(0, 2)) == 0 else "gaussian"
    pm = sample_random_mutation_prob(rng, base_mutation_prob, pm_mult_range)
    lo, hi = ETA_M_SAMPLE_RANGE if mutation_operator == "polynomial" else SIGMA_SAMPLE_RANGE
    exploration_strength = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return mutation_operator, pm, exploration_strength


def action_with_multiplier(action: dict[str, Any], n_vars: int) -> dict[str, Any]:
    """Attach the scale-normalized mutation multiplier to an action dict.

    Phase 1.75 records, alongside the raw per-variable mutation
    probability, the normalized multiplier ``pm * n_vars`` so the natural
    NSGA-II scale ``1 / n_vars`` maps to multiplier 1.0 regardless of
    problem dimension (see ``docs/PHASE1_75_PLAN.md`` Task 1). The
    multiplier is computed from the action actually used, i.e. from
    ``NSGAII.current_action()["mutation_probability"]``.

    Args:
        action: Action dict as returned by ``NSGAII.current_action()``;
            must contain ``"mutation_probability"``. Not mutated.
        n_vars: Number of decision variables of the problem.

    Returns:
        A copy of ``action`` with the additional float key
        ``"mutation_multiplier"`` (persisted by the trajectory recorder as
        an extra action key).
    """
    enriched = dict(action)
    enriched["mutation_multiplier"] = float(action["mutation_probability"]) * int(n_vars)
    return enriched


def run_single(
    problem_name: str,
    seed: int,
    generations: int,
    pop_size: int,
    n_reference_points: int,
    ref_point: np.ndarray,
    out_dir: Path,
    policy: str = "fixed",
    pm_mult_range: tuple[float, float] | None = None,
    action_space: str = "pm",
) -> dict[str, Any]:
    """Run one NSGA-II trajectory and save it to JSON.

    Initializes the population, records generation 0, then performs
    ``generations`` steps, recording after each step. Under the ``"random"``
    policy each step's mutation probability is sampled from a dedicated
    generator and injected via ``step(mutation_prob=pm_t)``; under the
    ``"full"`` action space the operator and exploration strength are
    sampled as well (see :func:`sample_full_action`) and injected via
    ``step(mutation_prob=..., mutation_operator=...,
    exploration_strength=...)``. The recorded action is the value actually
    used (``algorithm.current_action()`` after the step), enriched with the
    normalized ``mutation_multiplier`` (see :func:`action_with_multiplier`).
    Wall-clock runtime covers the full run (initialization + evolution
    loop).

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
            ``"random"`` (log-uniform per-generation sampling). The
            ``"full"`` action space implies randomized actions regardless
            of this setting.
        pm_mult_range: Multiplier range ``(lo, hi)`` for the randomized
            policies; must satisfy ``0 < lo <= hi``. ``None`` resolves to
            ``DEFAULT_PM_MULT_RANGE`` for ``action_space == "pm"`` and to
            ``FULL_ACTION_PM_MULT_RANGE`` for ``action_space == "full"``.
        action_space: ``"pm"`` (only the mutation probability is
            controlled, Phase-1 behavior) or ``"full"`` (operator +
            mutation probability + exploration strength, Phase-1.5).

    Returns:
        Summary dict with keys ``file``, ``problem``, ``seed``,
        ``final_hv``, ``final_igd``, ``runtime_sec``.

    Raises:
        ValueError: If ``policy`` or ``action_space`` is unknown or
            ``pm_mult_range`` is invalid.
    """
    if policy not in ("fixed", "random"):
        raise ValueError(f"unsupported policy {policy!r}; expected 'fixed' or 'random'")
    if action_space not in ("pm", "full"):
        raise ValueError(f"unsupported action_space {action_space!r}; expected 'pm' or 'full'")
    if pm_mult_range is None:
        pm_mult_range = (
            DEFAULT_PM_MULT_RANGE if action_space == "pm" else FULL_ACTION_PM_MULT_RANGE
        )
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
    sample_actions = policy == "random" or action_space == "full"
    policy_rng = make_policy_rng(problem.name, seed) if sample_actions else None

    start = time.perf_counter()
    algorithm.initialize()
    recorder.record(
        algorithm.generation,
        algorithm.nondominated_front(),
        action_with_multiplier(algorithm.current_action(), problem.n_vars),
    )
    for _ in range(generations):
        if action_space == "full":
            assert policy_rng is not None
            op_t, pm_t, es_t = sample_full_action(policy_rng, base_pm, (lo, hi))
            algorithm.step(
                mutation_prob=pm_t, mutation_operator=op_t, exploration_strength=es_t
            )
        elif policy_rng is not None:
            pm_t = sample_random_mutation_prob(policy_rng, base_pm, (lo, hi))
            algorithm.step(mutation_prob=pm_t)
        else:
            algorithm.step()
        recorder.record(
            algorithm.generation,
            algorithm.nondominated_front(),
            action_with_multiplier(algorithm.current_action(), problem.n_vars),
        )
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
        action_space=action_space,
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
        List of per-run summary dicts (also written to ``index.json``,
        together with per-problem run counts: ``n_runs``, ``n_success``
        (final HV > 0), ``n_failed``, and ``zero_hv_count`` (final
        HV == 0)).
    """
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_point = np.asarray(args.ref_point, dtype=float)

    print(
        f"Phase-0/1 dataset generation: {len(args.problems)} problems x "
        f"{len(args.seeds)} seeds x {args.generations} generations x "
        f"pop {args.pop_size}; policy={args.policy} "
        f"action_space={args.action_space} out_dir={out_dir}"
    )

    pm_mult_range = (
        (float(args.pm_mult_range[0]), float(args.pm_mult_range[1]))
        if args.pm_mult_range is not None
        else None
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
                    pm_mult_range=pm_mult_range,
                    action_space=args.action_space,
                )
            )

    per_problem: dict[str, dict[str, int]] = {}
    for summary in summaries:
        stats = per_problem.setdefault(
            summary["problem"],
            {"n_runs": 0, "n_success": 0, "n_failed": 0, "zero_hv_count": 0},
        )
        stats["n_runs"] += 1
        # Failure is defined as a final hypervolume of exactly 0 (no front
        # point dominates the reference point); a per-problem failure
        # threshold is deliberately not decided at generation time.
        if float(summary["final_hv"]) <= 0.0:
            stats["n_failed"] += 1
            stats["zero_hv_count"] += 1
        else:
            stats["n_success"] += 1

    index_path = out_dir / "index.json"
    index_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(summaries),
        "per_problem": per_problem,
        "runs": summaries,
    }
    with index_path.open("w", encoding="utf-8") as fh:
        json.dump(index_payload, fh, indent=2, ensure_ascii=False)
    print(f"[done] wrote index for {len(summaries)} runs -> {index_path}")
    return summaries


if __name__ == "__main__":
    main()
