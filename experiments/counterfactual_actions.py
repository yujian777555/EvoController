from __future__ import annotations

"""Phase 1.75: counterfactual one-step branch evaluation of controller actions.

This runner implements the causal control of ``docs/PHASE1_75_PLAN.md``
(Task 5): does the controller choose above-random actions *from the exact
same population state*? For every harvested state we branch the recorded
NSGA-II population into one generation per candidate action — the
controller-selected action plus ``--n-alternatives`` actions sampled from
the full Phase-1.5 action space — and score the one-step reward
``delta_hv + delta_igd`` (hypervolume gain plus IGD reduction vs the
snapshot state). The primary statistic is the percentile rank of the
controller action among the candidates, averaged over replicate RNG draws.

Subcommands (argparse subparsers):

* ``harvest`` — run fixed-policy NSGA-II (``OperatorConfig`` defaults) on
  held-out evaluation seeds (default 1000..1004) per problem and pickle
  ``algorithm.snapshot_state()`` plus the merged state-metric history at
  ``--states-per-run`` generations evenly spread over ``2..generations-2``
  (default: 40 states over gens 2..98 of 100) into
  ``{out_dir}/{problem}__seed{s}__gen{g}.pkl``.
* ``evaluate`` — for one problem (``--problem``, required; sharding unit),
  restore each snapshot, compute the state metrics, branch every candidate
  action with per-replicate RNG reseeding, and write
  ``{out_dir}/counterfactual_{problem}.json`` with per-state records and a
  summary (``mean_percentile_rank``, ``n_states``). The index-0 candidate
  comes from the controller selected by ``--controller-type``:
  ``multihead`` (default; ``--controller`` is a MultiHeadController
  checkpoint) or ``planning`` (Phase 2B; ``--controller`` is a
  PlanningController saved config JSON and ``--predictor`` an
  OutcomePredictor checkpoint). Both expose the same
  ``predict_action(history, encoder, pm_min, pm_max, n_vars)`` contract,
  so snapshot restore, branching, replicate RNG, and percentile ranking
  are shared unchanged.
* ``aggregate`` — merge the per-problem files into
  ``{results_dir}/counterfactual.json`` with the overall mean percentile
  rank, a deterministic bootstrap 95% CI (PCG64 seed 0, 10000 resamples),
  and an early/mid/late generation-third breakdown.
* ``evaluate-horizon`` — Phase-2B protocol-aligned branching (Task 4 of
  ``docs/PHASE2B_STABILIZATION_PLAN.md``). The Phase-2B planner scores
  candidate actions by *predicted long-horizon absolute HV*, so scoring it
  with the one-step Phase-1.75 reward measures a different objective. For
  every state this subcommand restores the snapshot, applies one candidate
  action for ``max(horizons)`` consecutive generations, and records the
  nondominated front's HV after each requested horizon, so
  ``reward_h = hv(after h generations) - hv(before the branch)``. Candidate
  index 0 is the controller action and — since Phase 2.75D, controlled by
  ``--include-default-action`` (default on) — index 1 is the **NSGA-II default
  action** tagged ``kind="default"``, which is the baseline of Target B
  ("advantage over the default evolutionary policy", Task 3 of
  ``docs/PHASE2_75D_PLAN.md``); the sampled alternatives follow, and
  ``--n-alternatives`` counts only those. ``--no-include-default-action``
  reproduces the pre-2.75D candidate set exactly. Per
  horizon it reports the controller action's percentile rank, the oracle
  regret (best candidate mean score - controller mean score), whether the
  controller action is the oracle argmax, and the within-state
  Spearman/Kendall correlation between the predictor's scores and the
  realized scores (``null`` for ``--controller-type multihead``, which has
  no outcome model, and for horizons outside the predictor's own horizons).
  Output goes to ``{out_dir}/counterfactual_horizon_{problem}.json``;
  ``--out-dir`` defaults to the Phase-2B directory
  :data:`DEFAULT_PLANNING_OUT_DIR`, so no Phase-1.75 artifact can be
  overwritten.
* ``aggregate-horizon`` — merge the per-problem horizon files into
  ``{results_dir}/counterfactual_horizon.json`` with per-horizon pooled
  statistics, the bootstrap 95% CI of the mean percentile rank, and
  early/mid/late generation-third breakdowns per horizon.

Determinism contract: identical inputs produce byte-identical outputs.
Every stochastic draw is derived from deterministic seed material
(``zlib.crc32`` of the state identity) via PCG64; output artifacts
deliberately carry no wall-clock timestamps or runtimes so that a repeated
run reproduces the files exactly (this overrides the usual runtime-record
convention for these artifacts; the full configuration is recorded
instead).

Seeding scheme (per snapshot):

* ``snapshot_hash_seed = crc32("{problem}|{seed}|{generation}")`` — seeds
  alternative sampling: alternative ``k`` uses
  ``Generator(PCG64([snapshot_hash_seed, k]))`` and
  :func:`experiments.generate_dataset.sample_full_action` (the exact
  full-action sampling of the dataset generator).
* ``branch_seed = crc32("{snapshot_hash_seed}|{action_index}")`` — seeds
  the per-replicate branch RNG: after ``restore_state``, the algorithm
  generator is replaced by ``Generator(PCG64([branch_seed, rep]))`` for
  ``rep`` in ``range(n_reps)``, so action ranking never depends on a single
  stochastic draw and no candidate inherits the harvest RNG position. The
  same branch RNG drives every generation of a branch, so an
  ``evaluate-horizon`` branch is one reproducible multi-generation
  trajectory from the snapshot state, not a fresh draw per generation.

Cost of ``evaluate-horizon`` (measured on the Phase-2B host, CPU only:
~0.39 s per NSGA-II generation at pop 100 / n_vars 30 / ZDT1, ~0.019 s at
pop 20; one hypervolume call is ~7 us):

* every branch runs ``max(horizons)`` generations, so the default
  ``--horizons "5 10 20"`` always costs 20 generations per branch; adding
  further horizons below ``max(horizons)`` is nearly free;
* generations per state = ``(1 + n_alternatives) * n_reps * max(horizons)``
  = 11 * 3 * 20 = 660 with the defaults (~4.5 min per state);
* per problem at ``--max-states 50``: ~33,000 generations, i.e. ~3.5 h;
* the sharding unit is ``--problem`` (one output file per problem), so the
  five ZDT problems are five independent workers.

Example:
    ``python experiments/counterfactual_actions.py harvest``
    ``python experiments/counterfactual_actions.py evaluate --problem zdt1 \
        --controller results/phase1_75/controller.pt --encoder results/phase1_75/encoder.json``
    ``python experiments/counterfactual_actions.py aggregate``
    ``python experiments/counterfactual_actions.py evaluate-horizon --problem zdt1 \
        --controller results/phase2_outcome/planning_controller.json \
        --controller-type planning --predictor results/phase2_outcome/predictor.pt \
        --encoder results/phase2_outcome/encoder.json``
    ``python experiments/counterfactual_actions.py aggregate-horizon``
"""

import argparse
import json
import math
import pickle
import sys
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from scipy import stats

if __package__ in (None, ""):
    # Allow ``python experiments/counterfactual_actions.py`` from the repo
    # root: the script directory (not the repo root) is on sys.path then.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.nsga2 import NSGAII, OperatorConfig
from benchmarks import get_problem
from benchmarks.base import Problem
from controller.dataset import OPERATOR_TO_INDEX, merge_state_reward
from controller.multihead_controller import MultiHeadController
from controller.outcome_predictor import OutcomePredictor
from controller.state_encoder import ProblemAwareEncoder, StateEncoder
from experiments.generate_dataset import sample_full_action
from metrics.indicators import hypervolume, igd
from trajectory.recorder import EvolutionRecorder

if TYPE_CHECKING:
    from controller.planning_controller import PlanningController

#: Default evaluation problem grid (all Phase-0 ZDT benchmarks).
DEFAULT_PROBLEMS: tuple[str, ...] = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")
#: Default held-out evaluation seeds (disjoint from all training seeds).
DEFAULT_EVAL_SEEDS: tuple[int, ...] = (1000, 1001, 1002, 1003, 1004)
#: Default snapshot output directory of ``harvest``.
DEFAULT_SNAPSHOTS_DIR = "results/phase1_75/snapshots"
#: Default output directory of ``evaluate``/``aggregate`` for the Phase-1.75
#: imitative controller (``--controller-type multihead``).
DEFAULT_OUT_DIR = "results/phase1_75"
#: Stage-isolated output directory used when ``--controller-type=planning``,
#: so Phase-2B counterfactual artifacts never overwrite Phase-1.75 results.
DEFAULT_PLANNING_OUT_DIR = "results/phase2b/counterfactual"
#: Default number of snapshot states per (problem, seed) run.
DEFAULT_STATES_PER_RUN = 40
#: Default number of alternative actions per state (plus the controller's).
DEFAULT_N_ALTERNATIVES = 19
#: Default number of replicate RNG draws per candidate action.
DEFAULT_N_REPS = 5
#: Default cap on evaluated states per problem (200 = 5 seeds x 40 states).
DEFAULT_MAX_STATES = 200
#: Deployment pm bounds as multipliers of 1/n_vars (Phase-1.5 contract).
DEFAULT_PM_MULT_RANGE: tuple[float, float] = (0.25, 8.0)
#: Bootstrap settings of ``aggregate`` (deterministic by contract).
DEFAULT_N_RESAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 0
#: Default branch horizons of ``evaluate-horizon`` in generations. The Phase-2B
#: planner scores candidates against the predictor's horizons [1, 5, 10, 20],
#: so the default covers the long-horizon part of that objective (the branch
#: always runs ``max(horizons)`` generations; shorter horizons are recorded
#: along the way).
DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)
#: Default number of alternative actions per state for ``evaluate-horizon``
#: (the controller action is candidate 0), matching the Phase-2B B2 grid.
DEFAULT_HORIZON_N_ALTERNATIVES = 10
#: Default number of replicate RNG draws per candidate for ``evaluate-horizon``.
DEFAULT_HORIZON_N_REPS = 3
#: Default cap on evaluated states per problem for ``evaluate-horizon``.
DEFAULT_HORIZON_MAX_STATES = 50


# ---------------------------------------------------------------------------
# harvest
# ---------------------------------------------------------------------------


def snapshot_generations(generations: int, states_per_run: int) -> list[int]:
    """Evenly spread snapshot generations over ``2..generations - 2``.

    Generation 0 is the initialization (no history to condition on) and the
    final generation leaves no room for the one-step branch, so snapshots
    cover the interior range; with the defaults (100 generations, 40
    states) this is 40 generations over 2..98.

    Args:
        generations: Total number of NSGA-II generations of the run; must
            be >= 4 so the interior range is non-empty.
        states_per_run: Requested number of snapshot states; capped by the
            number of available interior generations.

    Returns:
        Sorted list of unique generation indices (ascending).

    Raises:
        ValueError: If ``generations`` < 4 or ``states_per_run`` < 1.
    """
    if int(generations) < 4:
        raise ValueError(f"generations must be >= 4, got {generations}")
    if int(states_per_run) < 1:
        raise ValueError(f"states_per_run must be >= 1, got {states_per_run}")
    lo, hi = 2, int(generations) - 2
    n = min(int(states_per_run), hi - lo + 1)
    grid = np.linspace(lo, hi, n)
    return sorted({int(round(float(g))) for g in grid})


def harvest_snapshots(
    problem_name: str,
    seed: int,
    *,
    generations: int,
    pop_size: int,
    states_per_run: int,
    n_reference_points: int,
    ref_point: np.ndarray,
    out_dir: str | Path,
) -> list[Path]:
    """Run one fixed-policy NSGA-II run and pickle snapshot states.

    The run uses the ``OperatorConfig`` defaults throughout (no action
    injection); the recorder computes the state metrics of every
    generation. At each :func:`snapshot_generations` generation the pickle
    payload stores ``problem``, ``seed``, ``generation``, ``state``
    (:meth:`NSGAII.snapshot_state`), ``history`` (merged state+reward dicts
    of generations ``0..g`` — exactly what a controller may observe when
    choosing the action for the step into ``g + 1``), ``state_metrics``
    (hv/igd/diversity of the snapshot state), and a ``config`` block that
    fully determines the run.

    Args:
        problem_name: Benchmark identifier accepted by ``get_problem``.
        seed: Random seed of the run.
        generations: Number of NSGA-II generations to execute.
        pop_size: Population size.
        states_per_run: Number of snapshot generations (evenly spread).
        n_reference_points: Points sampled from the true Pareto front.
        ref_point: Hypervolume reference point, shape ``(2,)``.
        out_dir: Destination directory; files are named
            ``{problem}__seed{seed}__gen{g}.pkl``.

    Returns:
        Paths of the written snapshot files, in ascending generation order.
    """
    problem = get_problem(problem_name)
    operators = OperatorConfig()
    algorithm = NSGAII(problem, pop_size=pop_size, operators=operators, seed=seed)
    recorder = EvolutionRecorder(
        problem_name=problem.name,
        reference_front=problem.reference_front(n_points=n_reference_points),
        ref_point=ref_point,
    )
    snap_at = set(snapshot_generations(generations, states_per_run))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_pm = 1.0 / problem.n_vars

    algorithm.initialize()
    recorder.record(algorithm.generation, algorithm.nondominated_front(), algorithm.current_action())
    paths: list[Path] = []
    for _ in range(int(generations)):
        algorithm.step()
        recorder.record(
            algorithm.generation, algorithm.nondominated_front(), algorithm.current_action()
        )
        if algorithm.generation not in snap_at:
            continue
        transitions = recorder.transitions()
        payload = {
            "problem": problem.name,
            "seed": int(seed),
            "generation": int(algorithm.generation),
            "state": algorithm.snapshot_state(),
            "history": [merge_state_reward(t) for t in transitions],
            "state_metrics": {
                "hv": float(transitions[-1]["state"]["hv"]),
                "igd": float(transitions[-1]["state"]["igd"]),
                "diversity": float(transitions[-1]["state"]["diversity"]),
            },
            "config": {
                "problem": problem.name,
                "seed": int(seed),
                "algorithm": "nsga2",
                "policy": "fixed",
                "pop_size": int(pop_size),
                "generations": int(generations),
                "operators": {
                    "crossover_operator": operators.crossover_operator,
                    "crossover_prob": float(operators.crossover_prob),
                    "mutation_operator": operators.mutation_operator,
                    "mutation_probability": float(base_pm),
                    "eta_c": float(operators.eta_c),
                    "eta_m": float(operators.eta_m),
                    "gaussian_sigma": float(operators.gaussian_sigma),
                },
                "pm_mult_range": [float(DEFAULT_PM_MULT_RANGE[0]), float(DEFAULT_PM_MULT_RANGE[1])],
                "ref_point": [float(ref_point[0]), float(ref_point[1])],
                "n_reference_points": int(n_reference_points),
                "states_per_run": int(states_per_run),
            },
        }
        path = out_dir / f"{problem.name}__seed{seed}__gen{algorithm.generation}.pkl"
        with path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        paths.append(path)
    return paths


def run_harvest(args: argparse.Namespace) -> list[Path]:
    """Harvest snapshots for the full (problem, seed) grid.

    Writes one pickle per snapshot plus a deterministic
    ``harvest_config.json`` (no timestamps) into ``args.out_dir``.

    Args:
        args: Parsed arguments of the ``harvest`` subcommand.

    Returns:
        Paths of all written snapshot files.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_point = np.asarray(args.ref_point, dtype=float)
    problems = [get_problem(name).name for name in args.problems]
    seeds = [int(s) for s in args.seeds]
    print(
        f"Phase-1.75 counterfactual harvest: problems={problems}, seeds={seeds}, "
        f"gens={args.generations}, pop={args.pop_size}, "
        f"states_per_run={args.states_per_run}; out_dir={out_dir}"
    )
    paths: list[Path] = []
    for problem_name in problems:
        for seed in seeds:
            written = harvest_snapshots(
                problem_name,
                seed,
                generations=args.generations,
                pop_size=args.pop_size,
                states_per_run=args.states_per_run,
                n_reference_points=args.n_reference_points,
                ref_point=ref_point,
                out_dir=out_dir,
            )
            paths.extend(written)
            print(f"[harvest] {problem_name} seed={seed}: {len(written)} snapshots")
    config_path = out_dir / "harvest_config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "problems": problems,
                "seeds": seeds,
                "generations": int(args.generations),
                "pop_size": int(args.pop_size),
                "states_per_run": int(args.states_per_run),
                "n_reference_points": int(args.n_reference_points),
                "ref_point": [float(ref_point[0]), float(ref_point[1])],
                "policy": "fixed",
                "n_snapshots": len(paths),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[done] {len(paths)} snapshots -> {out_dir}")
    return paths


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def _snapshot_hash_seed(problem_name: str, seed: int, generation: int) -> int:
    """Deterministic seed material identifying one snapshot state."""
    return zlib.crc32(f"{problem_name}|{seed}|{generation}".encode("utf-8"))


def _branch_seed(snapshot_hash_seed: int, action_index: int) -> int:
    """Deterministic per-candidate seed material for the replicate RNGs."""
    return zlib.crc32(f"{snapshot_hash_seed}|{action_index}".encode("utf-8"))


def _load_encoder(path: str | Path) -> StateEncoder | ProblemAwareEncoder:
    """Load a fitted encoder, dispatching on the persisted payload keys.

    A :class:`ProblemAwareEncoder` payload carries ``problem_vector``; a
    plain :class:`StateEncoder` payload does not.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if "problem_vector" in payload:
        return ProblemAwareEncoder.load(path)
    return StateEncoder.load(path)


def _resolve_evaluate_out_dir(controller_type: str | None) -> str:
    """Return the stage-isolated default output directory for ``evaluate``.

    Phase-2B planning counterfactuals must never write into the Phase-1.75
    results directory, so ``--controller-type=planning`` defaults to
    :data:`DEFAULT_PLANNING_OUT_DIR` while the imitative controller keeps
    :data:`DEFAULT_OUT_DIR`.

    Args:
        controller_type: Value of ``--controller-type`` (``"multihead"`` or
            ``"planning"``; ``None`` behaves like ``"multihead"``).

    Returns:
        The default output directory for that controller family.
    """
    if str(controller_type) == "planning":
        return DEFAULT_PLANNING_OUT_DIR
    return DEFAULT_OUT_DIR


def _load_planning_controller(
    predictor: OutcomePredictor,
    path: str | Path,
    pm_mult_range: tuple[float, float],
) -> PlanningController:
    """Bind a PlanningController config JSON to ``predictor``.

    The Phase-2B planning controller holds no torch weights of its own
    (they live in the predictor checkpoint), so ``--controller`` points to
    a saved config JSON carrying the constructor settings. Keys are read
    from the nested ``"config"`` object when present (the codebase
    ``save`` convention) else from the top level; a missing
    ``pm_mult_range`` falls back to the CLI ``--pm-mult-range`` value.

    The import is deferred so the default ``multihead`` path also works in
    checkouts where ``controller/planning_controller.py`` is absent.

    Args:
        predictor: Outcome predictor used to score candidate actions.
        path: PlanningController saved config JSON.
        pm_mult_range: Fallback multiplier bounds ``(lo, hi)`` used when
            the config JSON carries no ``pm_mult_range``.

    Returns:
        The constructed PlanningController.

    Raises:
        TypeError: If required constructor settings (``n_candidates``,
            ``candidate_seed``, ``horizon_weights``) are absent from the
            config JSON and the constructor provides no defaults.
    """
    from controller.planning_controller import PlanningController

    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    config = payload.get("config", payload) if isinstance(payload, dict) else {}
    kwargs: dict[str, Any] = {}
    if "n_candidates" in config:
        kwargs["n_candidates"] = int(config["n_candidates"])
    if "candidate_seed" in config:
        kwargs["candidate_seed"] = int(config["candidate_seed"])
    if "horizon_weights" in config:
        kwargs["horizon_weights"] = [float(w) for w in config["horizon_weights"]]
    if "pm_mult_range" in config:
        kwargs["pm_mult_range"] = tuple(float(v) for v in config["pm_mult_range"])
    else:
        kwargs["pm_mult_range"] = (float(pm_mult_range[0]), float(pm_mult_range[1]))
    return PlanningController(predictor, **kwargs)


def _load_controller_and_encoder(
    args: argparse.Namespace,
) -> tuple[
    MultiHeadController | PlanningController,
    StateEncoder | ProblemAwareEncoder,
    OutcomePredictor | None,
]:
    """Load the encoder, the controller and (planning only) the predictor.

    Shared by ``evaluate`` and ``evaluate-horizon`` so both subcommands use
    identical artifact loading and dimension checks; extracting it left the
    ``evaluate`` behavior unchanged.

    Args:
        args: Parsed arguments carrying ``encoder``, ``controller``,
            ``controller_type``, ``predictor`` and ``pm_mult_range``.

    Returns:
        ``(controller, encoder, predictor)``; ``predictor`` is ``None`` for
        ``--controller-type=multihead``.

    Raises:
        ValueError: If ``--controller-type=planning`` is given without
            ``--predictor``, or the encoder dimension does not match the
            controller/predictor it was trained with.
    """
    encoder = _load_encoder(args.encoder)
    pm_mult_range = (float(args.pm_mult_range[0]), float(args.pm_mult_range[1]))
    if str(args.controller_type) == "planning":
        if args.predictor is None:
            raise ValueError(
                "--predictor is required when --controller-type=planning "
                "(path to an OutcomePredictor saved with .save())"
            )
        predictor = OutcomePredictor.load(args.predictor)
        expected_dim = int(encoder.dim) + 4
        if predictor.input_dim != expected_dim:
            raise ValueError(
                f"predictor input_dim {predictor.input_dim} does not match "
                f"encoder.dim + 4 = {expected_dim}; pass the encoder the "
                f"predictor was trained with"
            )
        controller = _load_planning_controller(
            predictor, args.controller, pm_mult_range
        )
        return controller, encoder, predictor
    controller = MultiHeadController.load(args.controller)
    if encoder.dim != controller.input_dim:
        raise ValueError(
            f"encoder dim {encoder.dim} does not match controller input_dim "
            f"{controller.input_dim}; pass the encoder the controller was trained with"
        )
    return controller, encoder, None


def percentile_rank_of_first(values: Sequence[float] | np.ndarray) -> float:
    """Midrank percentile of ``values[0]`` among all values, in ``[0, 1]``.

    ``(n_less + 0.5 * n_tied) / (n - 1)`` over the remaining entries: 1.0
    when the first value strictly exceeds every competitor, 0.0 when it is
    strictly below all of them, and 0.5 when everything ties. For a
    candidate exchangeable with the alternatives (the null of no action
    skill) the expected value is 0.5.

    Args:
        values: At least two reward values; index 0 is the candidate of
            interest.

    Returns:
        The percentile rank in ``[0, 1]``.

    Raises:
        ValueError: If fewer than two values are given.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size < 2:
        raise ValueError(f"need at least two values, got {v.size}")
    first = v[0]
    rest = v[1:]
    n_less = int(np.sum(rest < first))
    n_tied = int(np.sum(rest == first))
    return float((n_less + 0.5 * n_tied) / (v.size - 1))


#: NSGA-II default variation action (``algorithms.nsga2.OperatorConfig``
#: defaults: polynomial mutation, ``pm = 1 / n_vars`` i.e. multiplier 1.0, and
#: ``eta_m = 20``). ``evaluate-horizon`` inserts it as an explicit candidate so
#: Phase 2.75D Target B (advantage over the default evolutionary policy) can be
#: computed from intervention data.
DEFAULT_ACTION_OPERATOR = "polynomial"
DEFAULT_ACTION_MULTIPLIER = 1.0
DEFAULT_ACTION_EXPLORATION = 20.0


def _build_candidate_actions(
    *,
    controller_action: dict[str, Any],
    hash_seed: int,
    n_alternatives: int,
    base_pm: float,
    pm_mult_range: tuple[float, float],
    include_default_action: bool = False,
) -> list[dict[str, Any]]:
    """Candidate action list: controller, default, then sampled alternatives.

    Shared by ``evaluate`` and ``evaluate-horizon`` so both protocols score
    bit-identical candidate sets for the same snapshot: alternative ``k`` is
    drawn with ``Generator(PCG64([hash_seed, k]))`` through
    :func:`experiments.generate_dataset.sample_full_action`.

    Layout:

    * index 0 — the controller action;
    * index 1 — the NSGA-II default action
      (:data:`DEFAULT_ACTION_OPERATOR`, ``base_pm`` = multiplier 1.0,
      :data:`DEFAULT_ACTION_EXPLORATION`), **only** when
      ``include_default_action`` is True. Phase 2.75D Target B needs this
      candidate, because a sampled pool contains the default tuple only by
      chance;
    * the rest — the sampled alternatives, whose draws do not depend on the
      flag (the RNG seed is ``[hash_seed, k]``), so the pre-2.75D corpus is
      reproduced exactly with ``include_default_action=False``.

    Args:
        controller_action: Dict returned by the controller's
            ``predict_action``.
        hash_seed: :func:`_snapshot_hash_seed` material of the state.
        n_alternatives: Number of sampled alternatives (>= 1).
        base_pm: Baseline mutation probability ``1 / n_vars``.
        pm_mult_range: Log-uniform multiplier bounds around ``base_pm``.
        include_default_action: Insert the NSGA-II default action as the
            second candidate (default False keeps the historical behaviour).

    Returns:
        ``1 + n_alternatives`` action dicts (``2 + n_alternatives`` with the
        default candidate), each with keys ``mutation_operator``,
        ``mutation_probability`` and ``exploration_strength``.
    """
    actions: list[dict[str, Any]] = [
        {
            "mutation_operator": str(controller_action["mutation_operator"]),
            "mutation_probability": float(controller_action["mutation_probability"]),
            "exploration_strength": float(controller_action["exploration_strength"]),
        }
    ]
    if include_default_action:
        actions.append(
            {
                "mutation_operator": DEFAULT_ACTION_OPERATOR,
                "mutation_probability": float(base_pm) * DEFAULT_ACTION_MULTIPLIER,
                "exploration_strength": DEFAULT_ACTION_EXPLORATION,
            }
        )
    for k in range(1, int(n_alternatives) + 1):
        rng_k = np.random.Generator(np.random.PCG64([hash_seed, k]))
        operator, pm, exploration = sample_full_action(rng_k, base_pm, pm_mult_range)
        actions.append(
            {
                "mutation_operator": str(operator),
                "mutation_probability": float(pm),
                "exploration_strength": float(exploration),
            }
        )
    return actions


def evaluate_snapshot(
    payload: dict[str, Any],
    *,
    problem: Problem,
    reference_front: np.ndarray,
    ref_point: np.ndarray,
    controller: MultiHeadController | PlanningController,
    encoder: StateEncoder | ProblemAwareEncoder,
    n_alternatives: int,
    n_reps: int,
    pm_mult_range: tuple[float, float],
) -> dict[str, Any]:
    """Run the counterfactual one-step branch evaluation at one snapshot.

    Restores the snapshot into a fresh NSGA-II instance, recomputes the
    state metrics (hypervolume/IGD of the current nondominated front) and
    verifies them against the harvested values, then branches every
    candidate action — index 0 is the controller's ``predict_action`` on
    the stored history (the :class:`MultiHeadController` and Phase-2B
    ``PlanningController`` contracts are identical:
    ``predict_action(history, encoder, pm_min, pm_max, n_vars) -> dict``),
    indices 1..n_alternatives are sampled from the full action space via
    ``Generator(PCG64([snapshot_hash_seed, k]))``. Each candidate is
    stepped ``n_reps`` times; before every branch step the snapshot is
    restored and the algorithm RNG is replaced by
    ``Generator(PCG64([branch_seed, rep]))``. After all branches the
    snapshot state is restored once more.

    The one-step reward of a branch is ``delta_hv + delta_igd`` where
    ``delta_hv = hv_after - hv_before`` and
    ``delta_igd = igd_before - igd_after`` (improvement-positive, matching
    the trajectory recorder convention).

    Args:
        payload: Snapshot payload as written by :func:`harvest_snapshots`.
        problem: Problem instance matching ``payload["problem"]``.
        reference_front: True Pareto front samples for IGD, shape
            ``(n, 2)``.
        ref_point: Hypervolume reference point, shape ``(2,)``.
        controller: Fitted controller used for the index-0 candidate; a
            :class:`MultiHeadController` or a Phase-2B
            ``PlanningController`` — only ``predict_action`` is used.
        encoder: Fitted encoder matching the controller's input.
        n_alternatives: Number of alternative actions (>= 1).
        n_reps: Replicate RNG draws per candidate (>= 1).
        pm_mult_range: Multipliers on ``1 / n_vars`` bounding the
            controller pm output and the alternative sampling.

    Returns:
        Per-state record with the state metrics, the controller action,
        per-candidate reward details, and the controller percentile rank
        (per replicate and averaged).

    Raises:
        ValueError: If the payload problem does not match ``problem``, or
            ``n_alternatives``/``n_reps`` are invalid.
        RuntimeError: If the restored state metrics do not reproduce the
            harvested ones (the snapshot/harvest metric settings differ).
    """
    if int(n_alternatives) < 1:
        raise ValueError(f"n_alternatives must be >= 1, got {n_alternatives}")
    if int(n_reps) < 1:
        raise ValueError(f"n_reps must be >= 1, got {n_reps}")
    if str(payload["problem"]) != problem.name:
        raise ValueError(
            f"snapshot problem {payload['problem']!r} does not match {problem.name!r}"
        )
    snap = payload["state"]
    pop_size = int(snap["config"]["pop_size"])
    algorithm = NSGAII(problem, pop_size=pop_size, operators=OperatorConfig(), seed=0)
    algorithm.restore_state(snap)
    front0 = algorithm.nondominated_front()
    hv0 = float(hypervolume(front0, ref_point))
    igd0 = float(igd(front0, reference_front))
    stored = payload.get("state_metrics")
    if stored is not None and not (
        math.isclose(hv0, float(stored["hv"]), rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(igd0, float(stored["igd"]), rel_tol=0.0, abs_tol=1e-9)
    ):
        raise RuntimeError(
            f"restored state metrics (hv={hv0}, igd={igd0}) do not reproduce the "
            f"harvested ones ({stored}); check ref_point/n_reference_points"
        )

    n_vars = problem.n_vars
    pm_min = float(pm_mult_range[0]) / n_vars
    pm_max = float(pm_mult_range[1]) / n_vars
    controller_action = controller.predict_action(
        list(payload["history"]), encoder, pm_min, pm_max, n_vars=n_vars
    )
    hash_seed = _snapshot_hash_seed(
        str(payload["problem"]), int(payload["seed"]), int(payload["generation"])
    )
    base_pm = 1.0 / n_vars
    actions = _build_candidate_actions(
        controller_action=controller_action,
        hash_seed=hash_seed,
        n_alternatives=int(n_alternatives),
        base_pm=base_pm,
        pm_mult_range=pm_mult_range,
    )

    n_candidates = len(actions)
    rewards = np.zeros((n_candidates, int(n_reps)), dtype=float)
    delta_hvs = np.zeros_like(rewards)
    delta_igds = np.zeros_like(rewards)
    for k, action in enumerate(actions):
        branch = _branch_seed(hash_seed, k)
        for rep in range(int(n_reps)):
            algorithm.restore_state(snap)
            algorithm.rng = np.random.Generator(np.random.PCG64([branch, rep]))
            algorithm.step(
                mutation_prob=action["mutation_probability"],
                mutation_operator=action["mutation_operator"],
                exploration_strength=action["exploration_strength"],
            )
            front1 = algorithm.nondominated_front()
            hv1 = float(hypervolume(front1, ref_point))
            igd1 = float(igd(front1, reference_front))
            delta_hvs[k, rep] = hv1 - hv0
            delta_igds[k, rep] = igd0 - igd1
            rewards[k, rep] = delta_hvs[k, rep] + delta_igds[k, rep]
    # Leave the instance at the snapshot state (spec: restore the
    # snapshot's RNG state after all branches).
    algorithm.restore_state(snap)

    ranks_per_rep = [
        percentile_rank_of_first(rewards[:, rep]) for rep in range(int(n_reps))
    ]
    candidates = [
        {
            "index": int(k),
            "kind": "controller" if k == 0 else "alternative",
            "action": actions[k],
            "rewards": [float(v) for v in rewards[k]],
            "delta_hv": [float(v) for v in delta_hvs[k]],
            "delta_igd": [float(v) for v in delta_igds[k]],
            "mean_reward": float(rewards[k].mean()),
        }
        for k in range(n_candidates)
    ]
    return {
        "problem": str(payload["problem"]),
        "seed": int(payload["seed"]),
        "generation": int(payload["generation"]),
        "state_metrics": {"hv": hv0, "igd": igd0},
        "controller_action": actions[0],
        "candidates": candidates,
        "controller_mean_reward": float(rewards[0].mean()),
        "controller_mean_delta_hv": float(delta_hvs[0].mean()),
        "controller_mean_delta_igd": float(delta_igds[0].mean()),
        "controller_percentile_rank_per_rep": [float(r) for r in ranks_per_rep],
        "controller_percentile_rank": float(np.mean(ranks_per_rep)),
    }


def _select_snapshot_files(snapshots_dir: Path, problem: str, max_states: int) -> list[Path]:
    """Deterministically select up to ``max_states`` snapshot files.

    Files are sorted by (seed, generation) parsed from the file name; when
    more than ``max_states`` files exist, an evenly spaced subset (linspace
    indices, deduplicated, order preserved) is taken so early/mid/late
    generations stay covered.

    Raises:
        FileNotFoundError: If no snapshots exist for the problem.
    """
    def _sort_key(path: Path) -> tuple[int, int, str]:
        parts = path.stem.split("__")
        seed = int(parts[1].removeprefix("seed"))
        gen = int(parts[2].removeprefix("gen"))
        return (seed, gen, path.name)

    files = sorted(snapshots_dir.glob(f"{problem}__seed*__gen*.pkl"), key=_sort_key)
    if not files:
        raise FileNotFoundError(
            f"no snapshots for problem {problem!r} in {snapshots_dir}; run 'harvest' first"
        )
    if len(files) <= max_states:
        return files
    idx = np.linspace(0, len(files) - 1, int(max_states)).round().astype(int)
    return [files[i] for i in dict.fromkeys(int(i) for i in idx)]


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the controller action against alternatives at every state.

    Loads the fitted encoder (``--encoder``) and the controller selected
    by ``--controller-type`` — a MultiHeadController checkpoint
    (``--controller``) for ``multihead``, or an OutcomePredictor
    checkpoint (``--predictor``) bound to a PlanningController config JSON
    (``--controller``) for ``planning`` — evaluates up to ``--max-states``
    snapshots of ``--problem``, and writes
    ``{out_dir}/counterfactual_{problem}.json``.

    Args:
        args: Parsed arguments of the ``evaluate`` subcommand.

    Returns:
        The payload exactly as written to the output JSON.

    Raises:
        ValueError: If ``--controller-type=planning`` is given without
            ``--predictor``, or the encoder dimension does not match the
            controller/predictor it was trained with.
    """
    problem = get_problem(args.problem)
    controller, encoder, _ = _load_controller_and_encoder(args)
    pm_mult_range = (float(args.pm_mult_range[0]), float(args.pm_mult_range[1]))
    reference_front = problem.reference_front(n_points=args.n_reference_points)
    ref_point = np.asarray(args.ref_point, dtype=float)
    files = _select_snapshot_files(Path(args.snapshots_dir), problem.name, args.max_states)

    print(
        f"Phase-1.75 counterfactual evaluate: problem={problem.name}, "
        f"controller_type={args.controller_type}, {len(files)} states, "
        f"alternatives={args.n_alternatives}, reps={args.n_reps}"
    )
    states: list[dict[str, Any]] = []
    run_generations: int | None = None
    for path in files:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        cfg = payload.get("config", {})
        if run_generations is None:
            run_generations = int(cfg.get("generations", 0))
        elif int(cfg.get("generations", 0)) != run_generations:
            raise ValueError(
                f"snapshot {path.name} has generations={cfg.get('generations')}, "
                f"expected {run_generations}; harvest runs must share one setting"
            )
        record = evaluate_snapshot(
            payload,
            problem=problem,
            reference_front=reference_front,
            ref_point=ref_point,
            controller=controller,
            encoder=encoder,
            n_alternatives=args.n_alternatives,
            n_reps=args.n_reps,
            pm_mult_range=pm_mult_range,
        )
        states.append(record)
        print(
            f"[evaluate] {problem.name} seed={record['seed']} gen={record['generation']} "
            f"percentile_rank={record['controller_percentile_rank']:.3f}"
        )

    ranks = np.asarray([s["controller_percentile_rank"] for s in states], dtype=float)
    payload: dict[str, Any] = {
        "problem": problem.name,
        "config": {
            "controller": str(args.controller),
            "controller_type": str(args.controller_type),
            "predictor": str(args.predictor) if args.predictor is not None else None,
            "encoder": str(args.encoder),
            "snapshots_dir": str(args.snapshots_dir),
            "n_alternatives": int(args.n_alternatives),
            "n_reps": int(args.n_reps),
            "max_states": int(args.max_states),
            "pm_mult_range": [float(pm_mult_range[0]), float(pm_mult_range[1])],
            "ref_point": [float(ref_point[0]), float(ref_point[1])],
            "n_reference_points": int(args.n_reference_points),
            "generations": int(run_generations if run_generations is not None else 0),
        },
        "states": states,
        "summary": {
            "mean_percentile_rank": float(ranks.mean()) if ranks.size else None,
            "std_percentile_rank": float(ranks.std()) if ranks.size else None,
            "n_states": int(ranks.size),
            "seeds": sorted({int(s["seed"]) for s in states}),
        },
    }
    out_dir = Path(args.out_dir or _resolve_evaluate_out_dir(args.controller_type))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"counterfactual_{problem.name}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[done] wrote {ranks.size} state records -> {out_path}")
    return payload


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: np.ndarray, n_resamples: int, seed: int
) -> tuple[float, float]:
    """Deterministic bootstrap 95% CI of the mean (PCG64, percentile method).

    Args:
        values: Per-state values of shape ``(n,)``; must be non-empty.
        n_resamples: Number of bootstrap resamples.
        seed: Seed of the PCG64 resampling generator.

    Returns:
        ``(lower, upper)`` 2.5/97.5 percentiles of the resampled means.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty array")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    idx = rng.integers(0, values.size, size=(int(n_resamples), values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def run_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    """Merge per-problem counterfactual files into one summary.

    Pools the per-state controller percentile ranks of every problem file
    ``{results_dir}/counterfactual_{problem}.json`` and reports the overall
    mean, a bootstrap 95% CI, and an early/mid/late breakdown by generation
    thirds of the harvest run length (``generation / generations``).

    Args:
        args: Parsed arguments of the ``aggregate`` subcommand.

    Returns:
        The payload exactly as written to ``counterfactual.json``.

    Raises:
        FileNotFoundError: If a per-problem file is missing.
    """
    results_dir = Path(args.results_dir)
    problems = [get_problem(name).name for name in args.problems]
    per_problem: dict[str, Any] = {}
    pooled_ranks: list[float] = []
    pooled_rel_gen: list[float] = []
    for problem_name in problems:
        path = results_dir / f"counterfactual_{problem_name}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run 'evaluate --problem {problem_name}' first"
            )
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        generations = int(payload["config"]["generations"])
        if generations <= 0:
            raise ValueError(f"{path} has non-positive config.generations")
        ranks = [float(s["controller_percentile_rank"]) for s in payload["states"]]
        per_problem[problem_name] = {
            "mean_percentile_rank": float(np.mean(ranks)) if ranks else None,
            "n_states": len(ranks),
        }
        pooled_ranks.extend(ranks)
        pooled_rel_gen.extend(
            float(s["generation"]) / generations for s in payload["states"]
        )

    ranks = np.asarray(pooled_ranks, dtype=float)
    rel = np.asarray(pooled_rel_gen, dtype=float)
    if ranks.size == 0:
        raise ValueError("no states found in any per-problem file")
    ci_lo, ci_hi = _bootstrap_ci(ranks, args.n_resamples, args.bootstrap_seed)

    thirds: dict[str, Any] = {}
    for label, mask in (
        ("early", rel < 1.0 / 3.0),
        ("mid", (rel >= 1.0 / 3.0) & (rel < 2.0 / 3.0)),
        ("late", rel >= 2.0 / 3.0),
    ):
        sel = ranks[mask]
        thirds[label] = {
            "mean_percentile_rank": float(sel.mean()) if sel.size else None,
            "n_states": int(sel.size),
        }

    payload = {
        "config": {
            "problems": problems,
            "results_dir": str(results_dir),
            "n_resamples": int(args.n_resamples),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "overall": {
            "mean_percentile_rank": float(ranks.mean()),
            "std_percentile_rank": float(ranks.std()),
            "n_states": int(ranks.size),
            "bootstrap_ci_95": [ci_lo, ci_hi],
        },
        "per_problem": per_problem,
        "generation_thirds": thirds,
    }
    out_path = results_dir / "counterfactual.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(
        f"[done] overall mean percentile rank {ranks.mean():.4f} "
        f"(95% CI [{ci_lo:.4f}, {ci_hi:.4f}], n={ranks.size}) -> {out_path}"
    )
    return payload


# ---------------------------------------------------------------------------
# evaluate-horizon (Phase 2B Task 4: long-horizon counterfactual evaluation)
# ---------------------------------------------------------------------------


def _resolved_horizons(raw: Sequence[int]) -> list[int]:
    """Validate, deduplicate and sort requested branch horizons.

    Args:
        raw: Horizon values in generations, e.g. ``[5, 10, 20]``.

    Returns:
        Ascending list of unique horizons, each >= 1.

    Raises:
        ValueError: If ``raw`` is empty or contains a value < 1.
    """
    values = [int(h) for h in raw]
    if not values:
        raise ValueError("horizons must be a non-empty list of generation offsets")
    if min(values) < 1:
        raise ValueError(f"every horizon must be >= 1, got {values}")
    return sorted(set(values))


def _operator_one_hot(operator: str) -> list[float]:
    """One-hot action feature ``[is_polynomial, is_gaussian]``.

    Uses :data:`controller.dataset.OPERATOR_TO_INDEX`, the encoding
    :func:`controller.dataset.build_outcome_samples` trained the predictor
    with, so predicted scores are built from the exact input layout the
    Phase-2B :class:`~controller.planning_controller.PlanningController`
    scores its candidates with.

    Args:
        operator: Mutation operator name of a candidate action.

    Returns:
        ``[1.0, 0.0]`` for ``"polynomial"``, ``[0.0, 1.0]`` for ``"gaussian"``.

    Raises:
        ValueError: If ``operator`` is not a supported mutation operator.
    """
    if operator not in OPERATOR_TO_INDEX:
        raise ValueError(
            f"unsupported mutation_operator {operator!r}; expected one of "
            f"{sorted(OPERATOR_TO_INDEX)}"
        )
    one_hot = [0.0, 0.0]
    one_hot[OPERATOR_TO_INDEX[operator]] = 1.0
    return one_hot


def _finite_or_none(result: Any) -> float | None:
    """Finite ``statistic`` of a scipy correlation result, else ``None``."""
    value = getattr(result, "statistic", None)
    if value is None:
        value = getattr(result, "correlation", None)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _correlation_pair(
    predicted: Sequence[float] | None,
    realized: Sequence[float] | np.ndarray,
) -> tuple[float | None, float | None]:
    """Spearman and Kendall correlation of predicted vs realized scores.

    Args:
        predicted: Predicted per-candidate scores, or ``None`` when the run
            has no outcome model (``--controller-type multihead``) or the
            horizon is outside the predictor's horizons.
        realized: Realized per-candidate mean scores aligned with
            ``predicted``.

    Returns:
        ``(spearman, kendall)``; ``(None, None)`` when no prediction is
        available, fewer than two candidates were scored, either input is
        constant (scipy would report NaN), or scipy reports a non-finite
        value.
    """
    if predicted is None:
        return None, None
    x = np.asarray(list(predicted), dtype=float).reshape(-1)
    y = np.asarray(realized, dtype=float).reshape(-1)
    if x.size < 2 or x.size != y.size:
        return None, None
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None, None
    return (
        _finite_or_none(stats.spearmanr(x, y)),
        _finite_or_none(stats.kendalltau(x, y)),
    )


def _predict_candidate_rewards(
    predictor: OutcomePredictor | None,
    encoder: StateEncoder | ProblemAwareEncoder,
    history: Sequence[dict[str, Any]],
    actions: Sequence[dict[str, Any]],
    n_vars: int,
    horizons: Sequence[int],
    hv_before: float,
) -> dict[int, list[float]] | None:
    """Per-horizon predicted rewards of every candidate action.

    Builds one predictor row per candidate exactly as
    :func:`controller.dataset.build_outcome_samples` builds its training
    rows — ``encoder.transform(history)`` concatenated with
    ``[mutation_multiplier, exploration_strength, onehot_polynomial,
    onehot_gaussian]`` where ``mutation_multiplier = mutation_probability *
    n_vars`` — and converts the predicted absolute HV into a reward relative
    to the branch point: ``predicted_reward(h) = predicted_hv(h) -
    hv_before``. Within one state ``hv_before`` is constant, so the ranking
    of predicted rewards equals the ranking of predicted absolute HV.

    Args:
        predictor: Fitted outcome predictor, or ``None`` (multihead runs
            carry no outcome model).
        encoder: Fitted encoder, matching the predictor's training encoder.
        history: Merged state+reward dicts at the snapshot generation.
        actions: Candidate actions, index 0 first.
        n_vars: Decision-variable count; fixes the multiplier scale.
        horizons: Requested branch horizons in generations.
        hv_before: Hypervolume of the snapshot's nondominated front.

    Returns:
        ``{horizon: [predicted_reward per candidate]}`` restricted to the
        horizons the predictor covers, or ``None`` when no prediction is
        available (no predictor, or none of ``horizons`` is one of the
        predictor's horizons).

    Raises:
        ValueError: If the predictor input width does not match
            ``encoder.dim + 4``.
    """
    if predictor is None:
        return None
    state_block = np.asarray(encoder.transform(list(history)), dtype=np.float64)
    expected_dim = int(encoder.dim) + 4
    if int(predictor.input_dim) != expected_dim:
        raise ValueError(
            f"predictor input_dim {predictor.input_dim} does not match "
            f"encoder.dim + 4 = {expected_dim}; pass the encoder the "
            f"predictor was trained with"
        )
    rows: list[np.ndarray] = []
    for action in actions:
        features = np.asarray(
            [
                float(action["mutation_probability"]) * int(n_vars),
                float(action["exploration_strength"]),
                *_operator_one_hot(str(action["mutation_operator"])),
            ],
            dtype=np.float64,
        )
        rows.append(np.concatenate([state_block, features]))
    predictions = np.asarray(predictor.predict(np.vstack(rows)), dtype=np.float64)
    column_of = {int(h): i for i, h in enumerate(predictor.horizons)}
    predicted: dict[int, list[float]] = {}
    for horizon in horizons:
        column = column_of.get(int(horizon))
        if column is None:
            continue
        predicted[int(horizon)] = [
            float(predictions[k, column] - float(hv_before))
            for k in range(len(actions))
        ]
    return predicted or None


def evaluate_snapshot_horizon(
    payload: dict[str, Any],
    *,
    problem: Problem,
    reference_front: np.ndarray,
    ref_point: np.ndarray,
    controller: MultiHeadController | PlanningController,
    encoder: StateEncoder | ProblemAwareEncoder,
    predictor: OutcomePredictor | None,
    horizons: Sequence[int],
    n_alternatives: int,
    n_reps: int,
    pm_mult_range: tuple[float, float],
    include_default_action: bool = False,
) -> dict[str, Any]:
    """Branch every candidate for ``max(horizons)`` generations at one snapshot.

    Restores the snapshot into a fresh NSGA-II instance, reproduces and
    verifies the harvested state metrics, then builds the candidate list
    exactly as :func:`evaluate_snapshot` does (index 0 = controller action,
    indices 1..n = full-action samples). For every candidate and replicate
    it restores the snapshot, reseeds the algorithm RNG with
    ``Generator(PCG64([branch_seed, rep]))`` and steps ``max(horizons)``
    consecutive generations with that single action, recording the
    nondominated front's hypervolume after each requested horizon, so

        ``reward[k][h][rep] = hv(after h generations) - hv(before branch)``

    is the realized long-horizon gain of candidate ``k`` at horizon ``h``.

    Rewards are measured against the branch point and are never clipped, so
    a reward can be marginally negative: (mu + lambda) elitism preserves the
    non-domination rank, but when the combined first front exceeds the
    population size NSGA-II truncates it by crowding distance and can drop
    nondominated points, which slightly lowers the front's hypervolume
    (reproducible at pop_size 20 on ZDT1). Longer branches therefore
    normally accumulate hypervolume without being pointwise monotone.

    Args:
        payload: Snapshot payload as written by :func:`harvest_snapshots`.
        problem: Problem instance matching ``payload["problem"]``.
        reference_front: True Pareto front samples for the branch-point IGD
            integrity check, shape ``(n, 2)``.
        ref_point: Hypervolume reference point, shape ``(2,)``.
        controller: Controller used for the index-0 candidate (multihead or
            planning); only ``predict_action`` is used.
        encoder: Fitted encoder matching the controller/predictor input.
        predictor: Outcome predictor used for the predicted-vs-realized
            ranking correlations, or ``None`` (multihead runs).
        horizons: Branch horizons in generations (each >= 1).
        n_alternatives: Number of alternative actions (>= 1).
        n_reps: Replicate RNG draws per candidate (>= 1).
        pm_mult_range: Multipliers on ``1 / n_vars`` bounding the controller
            pm output and the alternative sampling.

    Returns:
        Per-state record with the branch-point metrics, the controller
        action, per-candidate per-horizon HV/reward details, and a
        ``per_horizon`` block holding the percentile rank, the oracle
        regret, the oracle-argmax flag, and the predicted-vs-realized
        correlations.

    Raises:
        ValueError: If the payload problem does not match ``problem``, or
            ``n_alternatives``/``n_reps``/``horizons`` are invalid.
        RuntimeError: If the restored state metrics do not reproduce the
            harvested ones (the snapshot/harvest metric settings differ).
    """
    if int(n_alternatives) < 1:
        raise ValueError(f"n_alternatives must be >= 1, got {n_alternatives}")
    if int(n_reps) < 1:
        raise ValueError(f"n_reps must be >= 1, got {n_reps}")
    horizon_list = _resolved_horizons(horizons)
    if str(payload["problem"]) != problem.name:
        raise ValueError(
            f"snapshot problem {payload['problem']!r} does not match {problem.name!r}"
        )
    snap = payload["state"]
    pop_size = int(snap["config"]["pop_size"])
    algorithm = NSGAII(problem, pop_size=pop_size, operators=OperatorConfig(), seed=0)
    algorithm.restore_state(snap)
    front0 = algorithm.nondominated_front()
    hv_before = float(hypervolume(front0, ref_point))
    igd_before = float(igd(front0, reference_front))
    stored = payload.get("state_metrics")
    if stored is not None and not (
        math.isclose(hv_before, float(stored["hv"]), rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(igd_before, float(stored["igd"]), rel_tol=0.0, abs_tol=1e-9)
    ):
        raise RuntimeError(
            f"restored state metrics (hv={hv_before}, igd={igd_before}) do not "
            f"reproduce the harvested ones ({stored}); check "
            f"ref_point/n_reference_points"
        )

    n_vars = problem.n_vars
    pm_min = float(pm_mult_range[0]) / n_vars
    pm_max = float(pm_mult_range[1]) / n_vars
    controller_action = controller.predict_action(
        list(payload["history"]), encoder, pm_min, pm_max, n_vars=n_vars
    )
    hash_seed = _snapshot_hash_seed(
        str(payload["problem"]), int(payload["seed"]), int(payload["generation"])
    )
    actions = _build_candidate_actions(
        controller_action=controller_action,
        hash_seed=hash_seed,
        n_alternatives=int(n_alternatives),
        base_pm=1.0 / n_vars,
        pm_mult_range=pm_mult_range,
        include_default_action=bool(include_default_action),
    )
    default_index = 1 if include_default_action else None
    n_candidates = len(actions)
    predicted = _predict_candidate_rewards(
        predictor,
        encoder,
        payload["history"],
        actions,
        n_vars,
        horizon_list,
        hv_before,
    )

    max_horizon = horizon_list[-1]
    horizon_index = {h: i for i, h in enumerate(horizon_list)}
    future_hv = [
        [[hv_before] * int(n_reps) for _ in horizon_list] for _ in range(n_candidates)
    ]
    for k, action in enumerate(actions):
        branch = _branch_seed(hash_seed, k)
        for rep in range(int(n_reps)):
            algorithm.restore_state(snap)
            algorithm.rng = np.random.Generator(np.random.PCG64([branch, rep]))
            for step in range(1, max_horizon + 1):
                algorithm.step(
                    mutation_prob=action["mutation_probability"],
                    mutation_operator=action["mutation_operator"],
                    exploration_strength=action["exploration_strength"],
                )
                index = horizon_index.get(step)
                if index is None:
                    continue
                future_hv[k][index][rep] = float(
                    hypervolume(algorithm.nondominated_front(), ref_point)
                )
    # Leave the instance at the snapshot state (mirrors evaluate_snapshot).
    algorithm.restore_state(snap)

    rewards = [
        [
            [future_hv[k][i][rep] - hv_before for rep in range(int(n_reps))]
            for i in range(len(horizon_list))
        ]
        for k in range(n_candidates)
    ]

    per_horizon: dict[str, Any] = {}
    for i, horizon in enumerate(horizon_list):
        # Rows are candidates, columns replicates -- the same layout
        # ``evaluate_snapshot`` uses, so ``realized[:, rep]`` is the
        # candidate vector of replicate ``rep``.
        realized = np.asarray(
            [
                [rewards[k][i][rep] for rep in range(int(n_reps))]
                for k in range(n_candidates)
            ],
            dtype=float,
        )
        mean_rewards = realized.mean(axis=1)
        ranks_per_rep = [
            percentile_rank_of_first(realized[:, rep]) for rep in range(int(n_reps))
        ]
        best_index = int(np.argmax(mean_rewards))
        best_mean = float(mean_rewards[best_index])
        controller_mean = float(mean_rewards[0])
        predicted_scores = (
            None if predicted is None else predicted.get(int(horizon))
        )
        spearman, kendall = _correlation_pair(predicted_scores, mean_rewards)
        per_horizon[str(horizon)] = {
            "n_candidates": int(n_candidates),
            "controller_percentile_rank": float(np.mean(ranks_per_rep)),
            "controller_percentile_rank_per_rep": [
                float(rank) for rank in ranks_per_rep
            ],
            "controller_mean_reward": controller_mean,
            "best_candidate_index": best_index,
            "best_mean_reward": best_mean,
            "oracle_regret": float(best_mean - controller_mean),
            "candidate_reward_spread": float(mean_rewards.max() - mean_rewards.min()),
            "planner_is_oracle_argmax": bool(best_index == 0),
            "prediction_available": bool(predicted_scores is not None),
            "spearman_predicted_vs_realized": spearman,
            "kendall_predicted_vs_realized": kendall,
        }

    candidates = [
        {
            "index": int(k),
            "kind": (
                "controller"
                if k == 0
                else "default"
                if k == default_index
                else "alternative"
            ),
            "action": actions[k],
            "future_hv": {
                str(h): [float(future_hv[k][i][rep]) for rep in range(int(n_reps))]
                for i, h in enumerate(horizon_list)
            },
            "reward": {
                str(h): [float(rewards[k][i][rep]) for rep in range(int(n_reps))]
                for i, h in enumerate(horizon_list)
            },
            "mean_reward": {
                str(h): float(np.mean(rewards[k][i]))
                for i, h in enumerate(horizon_list)
            },
            "predicted_reward": (
                None
                if predicted is None
                else {
                    str(h): float(predicted[h][k])
                    for h in horizon_list
                    if h in predicted
                }
            ),
        }
        for k in range(n_candidates)
    ]
    return {
        "problem": str(payload["problem"]),
        "seed": int(payload["seed"]),
        "generation": int(payload["generation"]),
        "state_metrics": {"hv": hv_before, "igd": igd_before},
        "controller_action": actions[0],
        "prediction_available": bool(predicted is not None),
        "candidates": candidates,
        "per_horizon": per_horizon,
    }


def _mean_or_none(values: np.ndarray) -> float | None:
    """Mean of a possibly empty array, or ``None`` when it is empty."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    return float(arr.mean()) if arr.size else None


def _horizon_arrays(
    states: Sequence[dict[str, Any]], horizon: int
) -> dict[str, np.ndarray]:
    """Pool the per-state metrics of one horizon across states.

    States that did not evaluate this horizon (e.g. pooled files produced
    with different ``--horizons``) are skipped, so every returned array has
    the same length: the number of states carrying this horizon.

    Args:
        states: Per-state records of one or more ``evaluate-horizon`` files.
        horizon: Horizon in generations.

    Returns:
        ``percentile_rank``, ``oracle_regret``, ``candidate_reward_spread``,
        ``controller_reward``, ``best_reward``, ``oracle_argmax`` (0/1) and
        the defined ``spearman`` / ``kendall`` correlations per state.
    """
    key = str(int(horizon))
    ranks: list[float] = []
    regrets: list[float] = []
    spreads: list[float] = []
    controller_rewards: list[float] = []
    best_rewards: list[float] = []
    oracle_argmax: list[float] = []
    spearman: list[float] = []
    kendall: list[float] = []
    for state in states:
        entry = state.get("per_horizon", {}).get(key)
        if entry is None:
            continue
        ranks.append(float(entry["controller_percentile_rank"]))
        regrets.append(float(entry["oracle_regret"]))
        spreads.append(float(entry["candidate_reward_spread"]))
        controller_rewards.append(float(entry["controller_mean_reward"]))
        best_rewards.append(float(entry["best_mean_reward"]))
        oracle_argmax.append(1.0 if entry["planner_is_oracle_argmax"] else 0.0)
        if entry["spearman_predicted_vs_realized"] is not None:
            spearman.append(float(entry["spearman_predicted_vs_realized"]))
        if entry["kendall_predicted_vs_realized"] is not None:
            kendall.append(float(entry["kendall_predicted_vs_realized"]))
    return {
        "percentile_rank": np.asarray(ranks, dtype=float),
        "oracle_regret": np.asarray(regrets, dtype=float),
        "candidate_reward_spread": np.asarray(spreads, dtype=float),
        "controller_reward": np.asarray(controller_rewards, dtype=float),
        "best_reward": np.asarray(best_rewards, dtype=float),
        "oracle_argmax": np.asarray(oracle_argmax, dtype=float),
        "spearman": np.asarray(spearman, dtype=float),
        "kendall": np.asarray(kendall, dtype=float),
    }


def _generation_thirds(
    ranks: np.ndarray, regrets: np.ndarray, relative_generations: np.ndarray
) -> dict[str, Any]:
    """Early/mid/late breakdown of pooled per-state horizon statistics.

    Args:
        ranks: Per-state percentile ranks.
        regrets: Per-state oracle regrets, aligned with ``ranks``.
        relative_generations: Per-state ``generation / generations`` values,
            aligned with ``ranks``.

    Returns:
        ``{"early": ..., "mid": ..., "late": ...}`` with the state count,
        mean percentile rank and mean oracle regret of each third.
    """
    thirds: dict[str, Any] = {}
    for label, mask in (
        ("early", relative_generations < 1.0 / 3.0),
        (
            "mid",
            (relative_generations >= 1.0 / 3.0)
            & (relative_generations < 2.0 / 3.0),
        ),
        ("late", relative_generations >= 2.0 / 3.0),
    ):
        selected_ranks = ranks[mask]
        selected_regrets = regrets[mask]
        thirds[label] = {
            "n_states": int(selected_ranks.size),
            "mean_percentile_rank": _mean_or_none(selected_ranks),
            "mean_oracle_regret": _mean_or_none(selected_regrets),
        }
    return thirds


def _horizon_summary(
    states: Sequence[dict[str, Any]],
    horizon: int,
    *,
    n_resamples: int | None = None,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    relative_generations: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Pooled statistics of one horizon over ``states``.

    Args:
        states: Per-state records carrying a ``per_horizon[str(horizon)]``
            block; states without it are skipped.
        horizon: Horizon in generations.
        n_resamples: When given, add the deterministic bootstrap 95% CI of
            the mean percentile rank computed with this many resamples.
        bootstrap_seed: Seed of the bootstrap generator.
        relative_generations: Optional per-state ``generation / generations``
            values aligned with ``states``; when given, an early/mid/late
            breakdown is included.

    Returns:
        Summary dict with the state count, mean/std percentile rank, mean
        oracle regret, mean candidate spread, mean controller/best rewards,
        oracle-argmax fraction and predicted-vs-realized correlations.
    """
    arrays = _horizon_arrays(states, horizon)
    ranks = arrays["percentile_rank"]
    summary: dict[str, Any] = {
        "n_states": int(ranks.size),
        "mean_percentile_rank": _mean_or_none(ranks),
        "std_percentile_rank": float(ranks.std()) if ranks.size else None,
        "mean_oracle_regret": _mean_or_none(arrays["oracle_regret"]),
        "mean_candidate_reward_spread": _mean_or_none(
            arrays["candidate_reward_spread"]
        ),
        "mean_controller_reward": _mean_or_none(arrays["controller_reward"]),
        "mean_best_reward": _mean_or_none(arrays["best_reward"]),
        "frac_planner_is_oracle_argmax": _mean_or_none(arrays["oracle_argmax"]),
        "n_states_with_correlation": int(arrays["spearman"].size),
        "mean_spearman": _mean_or_none(arrays["spearman"]),
        "mean_kendall": _mean_or_none(arrays["kendall"]),
    }
    if n_resamples is not None:
        summary["bootstrap_ci_95"] = (
            list(_bootstrap_ci(ranks, int(n_resamples), int(bootstrap_seed)))
            if ranks.size
            else None
        )
    if relative_generations is not None:
        summary["generation_thirds"] = _generation_thirds(
            ranks,
            arrays["oracle_regret"],
            np.asarray(relative_generations, dtype=float),
        )
    return summary


def _horizon_config_key(
    args: argparse.Namespace,
    problem_name: str,
    horizons: Sequence[int],
    pm_mult_range: tuple[float, float],
    ref_point: np.ndarray,
) -> dict[str, Any]:
    """Fingerprint identifying one ``evaluate-horizon`` invocation.

    A checkpoint is only reused when this fingerprint matches exactly, so
    changing any protocol knob (horizons, candidates, reps, seeds,
    controller/encoder/predictor, snapshot set) starts a fresh run instead of
    silently mixing incompatible state records.
    """
    return {
        "problem": str(problem_name),
        "controller": str(args.controller),
        "controller_type": str(args.controller_type),
        "predictor": str(args.predictor) if args.predictor is not None else None,
        "encoder": str(args.encoder),
        "snapshots_dir": str(args.snapshots_dir),
        "horizons": [int(h) for h in horizons],
        "n_alternatives": int(args.n_alternatives),
        "n_reps": int(args.n_reps),
        "include_default_action": bool(args.include_default_action),
        "max_states": int(args.max_states),
        "pm_mult_range": [float(pm_mult_range[0]), float(pm_mult_range[1])],
        "ref_point": [float(ref_point[0]), float(ref_point[1])],
        "n_reference_points": int(args.n_reference_points),
    }


def _write_horizon_checkpoint(
    path: Path,
    config_key: dict[str, Any],
    run_generations: int | None,
    done_files: set[str],
    states: list[dict[str, Any]],
) -> None:
    """Atomically persist the in-progress state records of one problem.

    The file is written to ``<path>.tmp`` and then replaced into place, so a
    crash (or power loss) mid-write cannot leave a truncated checkpoint that
    the next run would refuse to parse.
    """
    payload = {
        "config_key": config_key,
        "run_generations": run_generations,
        "done_files": sorted(done_files),
        "states": states,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    tmp_path.replace(path)


def run_evaluate_horizon(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate every candidate over ``--horizons`` generations per state.

    Loads the controller, encoder and predictor exactly as
    :func:`run_evaluate` does, evaluates up to ``--max-states`` snapshots of
    ``--problem`` with :func:`evaluate_snapshot_horizon`, and writes
    ``{out_dir}/counterfactual_horizon_{problem}.json`` — by default into
    the Phase-2B isolated directory :data:`DEFAULT_PLANNING_OUT_DIR`.

    Args:
        args: Parsed arguments of the ``evaluate-horizon`` subcommand.

    Returns:
        The payload exactly as written to the output JSON.

    Raises:
        ValueError: If ``--controller-type=planning`` is given without
            ``--predictor``, the encoder dimension does not match the
            controller/predictor, ``--horizons`` is invalid, or the
            snapshots disagree on their harvest length.
    """
    problem = get_problem(args.problem)
    controller, encoder, predictor = _load_controller_and_encoder(args)
    pm_mult_range = (float(args.pm_mult_range[0]), float(args.pm_mult_range[1]))
    horizons = _resolved_horizons(args.horizons)
    reference_front = problem.reference_front(n_points=args.n_reference_points)
    ref_point = np.asarray(args.ref_point, dtype=float)
    files = _select_snapshot_files(Path(args.snapshots_dir), problem.name, args.max_states)

    print(
        f"Phase-2B counterfactual evaluate-horizon: problem={problem.name}, "
        f"controller_type={args.controller_type}, {len(files)} states, "
        f"alternatives={args.n_alternatives}, reps={args.n_reps}, "
        f"horizons={horizons} (generations per branch={horizons[-1]})"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"counterfactual_horizon_{problem.name}.json"
    checkpoint_path = out_dir / f".checkpoint_horizon_{problem.name}.json"
    config_key = _horizon_config_key(
        args, problem.name, horizons, pm_mult_range, ref_point
    )

    # Resume support: a long run (hours) must not lose its finished states to
    # a shutdown, so each completed state is appended to a checkpoint that the
    # next invocation of the same command picks up.
    states: list[dict[str, Any]] = []
    done_files: set[str] = set()
    run_generations: int | None = None
    if checkpoint_path.is_file():
        try:
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            ckpt = {}
        if ckpt.get("config_key") == config_key:
            states = list(ckpt.get("states", []))
            done_files = set(ckpt.get("done_files", []))
            run_generations = ckpt.get("run_generations")
            print(
                f"[resume] checkpoint found: {len(states)} states already done, "
                f"{len(files) - len(done_files)} remaining"
            )
        else:
            print("[resume] checkpoint config mismatch; starting fresh")

    for path in files:
        if path.name in done_files:
            continue
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        cfg = payload.get("config", {})
        if run_generations is None:
            run_generations = int(cfg.get("generations", 0))
        elif int(cfg.get("generations", 0)) != run_generations:
            raise ValueError(
                f"snapshot {path.name} has generations={cfg.get('generations')}, "
                f"expected {run_generations}; harvest runs must share one setting"
            )
        record = evaluate_snapshot_horizon(
            payload,
            problem=problem,
            reference_front=reference_front,
            ref_point=ref_point,
            controller=controller,
            encoder=encoder,
            predictor=predictor,
            horizons=horizons,
            n_alternatives=args.n_alternatives,
            n_reps=args.n_reps,
            pm_mult_range=pm_mult_range,
            include_default_action=bool(args.include_default_action),
        )
        states.append(record)
        done_files.add(path.name)
        _write_horizon_checkpoint(
            checkpoint_path, config_key, run_generations, done_files, states
        )
        ranks = " ".join(
            f"h{h}={record['per_horizon'][str(h)]['controller_percentile_rank']:.3f}"
            for h in horizons
        )
        print(
            f"[evaluate-horizon] {problem.name} seed={record['seed']} "
            f"gen={record['generation']} {ranks} "
            f"[{len(states)}/{len(files)}]"
        )

    payload = {
        "problem": problem.name,
        "config": {
            "controller": str(args.controller),
            "controller_type": str(args.controller_type),
            "predictor": str(args.predictor) if args.predictor is not None else None,
            "encoder": str(args.encoder),
            "snapshots_dir": str(args.snapshots_dir),
            "horizons": [int(h) for h in horizons],
            "branch_generations": int(horizons[-1]),
            "n_alternatives": int(args.n_alternatives),
            "n_reps": int(args.n_reps),
            "include_default_action": bool(args.include_default_action),
            "max_states": int(args.max_states),
            "pm_mult_range": [float(pm_mult_range[0]), float(pm_mult_range[1])],
            "ref_point": [float(ref_point[0]), float(ref_point[1])],
            "n_reference_points": int(args.n_reference_points),
            "generations": int(run_generations if run_generations is not None else 0),
        },
        "states": states,
        "summary": {
            "n_states": len(states),
            "seeds": sorted({int(s["seed"]) for s in states}),
            "per_horizon": {
                str(h): _horizon_summary(states, h) for h in horizons
            },
        },
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    if checkpoint_path.is_file():
        checkpoint_path.unlink()
    print(f"[done] wrote {len(states)} state records -> {out_path}")
    return payload


def run_aggregate_horizon(args: argparse.Namespace) -> dict[str, Any]:
    """Merge the per-problem horizon files into ``counterfactual_horizon.json``.

    Pools the per-state records of every
    ``{results_dir}/counterfactual_horizon_{problem}.json`` and reports, per
    horizon: the overall mean percentile rank with a deterministic bootstrap
    95% CI, the mean oracle regret, the oracle-argmax fraction, the mean
    predicted-vs-realized correlations, and an early/mid/late breakdown by
    generation thirds of the harvest run length (``generation /
    generations``). Artifacts carry no timestamps, so identical inputs
    reproduce the output byte-for-byte.

    Args:
        args: Parsed arguments of the ``aggregate-horizon`` subcommand.

    Returns:
        The payload exactly as written to ``counterfactual_horizon.json``.

    Raises:
        FileNotFoundError: If a per-problem file is missing.
        ValueError: If a file has non-positive ``config.generations`` or no
            state carries any horizon.
    """
    results_dir = Path(args.results_dir)
    problems = [get_problem(name).name for name in args.problems]
    per_problem: dict[str, Any] = {}
    pooled: list[tuple[dict[str, Any], float]] = []
    horizons: set[int] = set()
    for problem_name in problems:
        path = results_dir / f"counterfactual_horizon_{problem_name}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run 'evaluate-horizon --problem {problem_name}' first"
            )
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        generations = int(payload["config"]["generations"])
        if generations <= 0:
            raise ValueError(f"{path} has non-positive config.generations")
        problem_horizons = [int(h) for h in payload["config"]["horizons"]]
        horizons.update(problem_horizons)
        states = payload["states"]
        pooled.extend(
            (state, float(state["generation"]) / generations) for state in states
        )
        per_problem[problem_name] = {
            str(h): _horizon_summary(states, h) for h in sorted(problem_horizons)
        }

    if not pooled:
        raise ValueError("no states found in any per-problem file")

    ordered = sorted(horizons)
    pooled_summaries: dict[str, Any] = {}
    for horizon in ordered:
        selected = [
            (state, relative)
            for state, relative in pooled
            if str(horizon) in state.get("per_horizon", {})
        ]
        pooled_summaries[str(horizon)] = _horizon_summary(
            [state for state, _ in selected],
            horizon,
            n_resamples=int(args.n_resamples),
            bootstrap_seed=int(args.bootstrap_seed),
            relative_generations=[relative for _, relative in selected],
        )

    payload = {
        "config": {
            "problems": problems,
            "results_dir": str(results_dir),
            "horizons": ordered,
            "n_resamples": int(args.n_resamples),
            "bootstrap_seed": int(args.bootstrap_seed),
        },
        "horizons": pooled_summaries,
        "per_problem": per_problem,
    }
    out_path = results_dir / "counterfactual_horizon.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    for horizon in ordered:
        entry = pooled_summaries[str(horizon)]
        mean_rank = entry["mean_percentile_rank"]
        ci = entry["bootstrap_ci_95"]
        if mean_rank is None or ci is None:
            print(f"[done] horizon {horizon}: no states")
            continue
        print(
            f"[done] horizon {horizon}: mean percentile rank {mean_rank:.4f} "
            f"(95% CI [{ci[0]:.4f}, {ci[1]:.4f}], n={entry['n_states']})"
        )
    print(f"[done] wrote {out_path}")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the counterfactual evaluation.

    Args:
        argv: Argument list to parse; ``None`` reads ``sys.argv``.

    Returns:
        Parsed namespace with ``command`` in {harvest, evaluate,
        evaluate-horizon, aggregate, aggregate-horizon} and the
        subcommand-specific settings.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Phase-1.75 counterfactual one-step branch evaluation: does the "
            "controller choose above-random actions from identical states?"
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    harvest = sub.add_parser(
        "harvest", help="Run fixed-policy NSGA-II and pickle snapshot states."
    )
    harvest.add_argument(
        "--problems", nargs="+", default=list(DEFAULT_PROBLEMS),
        help="Benchmark problems (default: %(default)s).",
    )
    harvest.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_EVAL_SEEDS),
        help="Held-out evaluation seeds (default: %(default)s).",
    )
    harvest.add_argument(
        "--generations", type=int, default=100,
        help="NSGA-II generations per run (default: %(default)s).",
    )
    harvest.add_argument(
        "--pop-size", type=int, default=100,
        help="Population size (default: %(default)s).",
    )
    harvest.add_argument(
        "--states-per-run", type=int, default=DEFAULT_STATES_PER_RUN,
        help="Snapshot generations per run, spread over 2..generations-2 "
        "(default: %(default)s).",
    )
    harvest.add_argument(
        "--n-reference-points", type=int, default=200,
        help="Points sampled from the true Pareto front for IGD (default: %(default)s).",
    )
    harvest.add_argument(
        "--ref-point", nargs=2, type=float, default=[1.1, 1.1],
        metavar=("REF_F1", "REF_F2"),
        help="Hypervolume reference point (default: %(default)s).",
    )
    harvest.add_argument(
        "--out-dir", type=str, default=DEFAULT_SNAPSHOTS_DIR,
        help="Snapshot output directory (default: %(default)s).",
    )

    evaluate = sub.add_parser(
        "evaluate", help="Branch controller vs alternative actions at each snapshot."
    )
    evaluate.add_argument(
        "--problem", required=True,
        help="Problem to evaluate (sharding unit; one file per problem).",
    )
    evaluate.add_argument(
        "--snapshots-dir", type=str, default=DEFAULT_SNAPSHOTS_DIR,
        help="Directory with harvest snapshots (default: %(default)s).",
    )
    evaluate.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for counterfactual_{problem}.json. Defaults to "
        "the Phase-2B isolated directory "
        f"({DEFAULT_PLANNING_OUT_DIR}) when --controller-type=planning, so "
        f"Phase-1.75 artifacts under {DEFAULT_OUT_DIR} are never overwritten; "
        "defaults to %(default)s-style "
        f"{DEFAULT_OUT_DIR} for --controller-type=multihead.",
    )
    evaluate.add_argument(
        "--controller", required=True,
        help="Controller artifact: a MultiHeadController saved with .save() "
        "(--controller-type multihead) or a PlanningController saved config "
        "JSON (--controller-type planning).",
    )
    evaluate.add_argument(
        "--controller-type", choices=["multihead", "planning"], default="multihead",
        help="Controller family of the index-0 candidate: 'multihead' "
        "(Phase-1.75 imitative controller, default) or 'planning' (Phase-2B "
        "candidate-action planner; requires --predictor).",
    )
    evaluate.add_argument(
        "--predictor", type=str, default=None,
        help="Path to an OutcomePredictor saved with .save(); required when "
        "--controller-type=planning, ignored otherwise.",
    )
    evaluate.add_argument(
        "--encoder", required=True,
        help="Path to the fitted encoder saved with .save() (StateEncoder or "
        "ProblemAwareEncoder; for the problem-aware encoder the file must "
        "match --problem).",
    )
    evaluate.add_argument(
        "--n-alternatives", type=int, default=DEFAULT_N_ALTERNATIVES,
        help="Alternative full actions per state (default: %(default)s).",
    )
    evaluate.add_argument(
        "--n-reps", type=int, default=DEFAULT_N_REPS,
        help="Replicate RNG draws per candidate action (default: %(default)s).",
    )
    evaluate.add_argument(
        "--max-states", type=int, default=DEFAULT_MAX_STATES,
        help="Cap on evaluated states per problem, evenly subsampled "
        "(default: %(default)s).",
    )
    evaluate.add_argument(
        "--pm-mult-range", nargs=2, type=float, default=list(DEFAULT_PM_MULT_RANGE),
        metavar=("PM_MULT_LO", "PM_MULT_HI"),
        help="Multipliers on 1/n_vars bounding pm (default: %(default)s).",
    )
    evaluate.add_argument(
        "--n-reference-points", type=int, default=200,
        help="Points sampled from the true Pareto front for IGD; must match "
        "the harvest setting (default: %(default)s).",
    )
    evaluate.add_argument(
        "--ref-point", nargs=2, type=float, default=[1.1, 1.1],
        metavar=("REF_F1", "REF_F2"),
        help="Hypervolume reference point; must match the harvest setting "
        "(default: %(default)s).",
    )

    aggregate = sub.add_parser(
        "aggregate", help="Merge per-problem files into counterfactual.json."
    )
    aggregate.add_argument(
        "--problems", nargs="+", default=list(DEFAULT_PROBLEMS),
        help="Problems to merge (default: %(default)s).",
    )
    aggregate.add_argument(
        "--results-dir", type=str, default=DEFAULT_OUT_DIR,
        help="Directory with counterfactual_{problem}.json files (default: %(default)s).",
    )
    aggregate.add_argument(
        "--n-resamples", type=int, default=DEFAULT_N_RESAMPLES,
        help="Bootstrap resamples for the 95%% CI (default: %(default)s).",
    )
    aggregate.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED,
        help="Seed of the bootstrap PCG64 generator (default: %(default)s).",
    )
    horizon = sub.add_parser(
        "evaluate-horizon",
        help="Phase-2B: branch every candidate for N generations and score "
        "the future hypervolume at each horizon.",
    )
    horizon.add_argument(
        "--problem", required=True,
        help="Problem to evaluate (sharding unit; one file per problem).",
    )
    horizon.add_argument(
        "--snapshots-dir", type=str, default=DEFAULT_SNAPSHOTS_DIR,
        help="Directory with harvest snapshots (default: %(default)s).",
    )
    horizon.add_argument(
        "--out-dir", type=str, default=DEFAULT_PLANNING_OUT_DIR,
        help="Output directory for counterfactual_horizon_{problem}.json "
        "(default: %(default)s; stage-isolated from the Phase-1.75 results).",
    )
    horizon.add_argument(
        "--controller", required=True,
        help="Controller artifact: a MultiHeadController saved with .save() "
        "(--controller-type multihead) or a PlanningController saved config "
        "JSON (--controller-type planning).",
    )
    horizon.add_argument(
        "--controller-type", choices=["multihead", "planning"], default="multihead",
        help="Controller family of the index-0 candidate: 'multihead' "
        "(Phase-1.75 imitative controller, default) or 'planning' (Phase-2B "
        "candidate-action planner; requires --predictor).",
    )
    horizon.add_argument(
        "--predictor", type=str, default=None,
        help="Path to an OutcomePredictor saved with .save(); required when "
        "--controller-type=planning, ignored otherwise.",
    )
    horizon.add_argument(
        "--encoder", required=True,
        help="Path to the fitted encoder saved with .save() (StateEncoder or "
        "ProblemAwareEncoder; for the problem-aware encoder the file must "
        "match --problem).",
    )
    horizon.add_argument(
        "--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS),
        help="Branch horizons in generations; every branch runs "
        "max(horizons) generations (default: %(default)s).",
    )
    horizon.add_argument(
        "--n-alternatives", type=int, default=DEFAULT_HORIZON_N_ALTERNATIVES,
        help="Alternative full actions per state (default: %(default)s).",
    )
    horizon.add_argument(
        "--include-default-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Insert the NSGA-II default action (polynomial mutation, "
        "pm = 1/n_vars i.e. multiplier 1.0, eta_m = "
        f"{DEFAULT_ACTION_EXPLORATION:g}) as an explicit candidate right after "
        "the controller action, tagged kind='default'. Phase 2.75D Target B "
        "(advantage over the default evolutionary policy) is only computable "
        "with this candidate; --n-alternatives still counts the sampled "
        "alternatives only. Use --no-include-default-action to reproduce the "
        "pre-2.75D candidate set exactly (default: %(default)s).",
    )
    horizon.add_argument(
        "--n-reps", type=int, default=DEFAULT_HORIZON_N_REPS,
        help="Replicate RNG draws per candidate action (default: %(default)s).",
    )
    horizon.add_argument(
        "--max-states", type=int, default=DEFAULT_HORIZON_MAX_STATES,
        help="Cap on evaluated states per problem, evenly subsampled "
        "(default: %(default)s).",
    )
    horizon.add_argument(
        "--pm-mult-range", nargs=2, type=float, default=list(DEFAULT_PM_MULT_RANGE),
        metavar=("PM_MULT_LO", "PM_MULT_HI"),
        help="Multipliers on 1/n_vars bounding pm (default: %(default)s).",
    )
    horizon.add_argument(
        "--n-reference-points", type=int, default=200,
        help="Points sampled from the true Pareto front for IGD; must match "
        "the harvest setting (default: %(default)s).",
    )
    horizon.add_argument(
        "--ref-point", nargs=2, type=float, default=[1.1, 1.1],
        metavar=("REF_F1", "REF_F2"),
        help="Hypervolume reference point; must match the harvest setting "
        "(default: %(default)s).",
    )

    aggregate_horizon = sub.add_parser(
        "aggregate-horizon",
        help="Merge counterfactual_horizon_{problem}.json files into "
        "counterfactual_horizon.json.",
    )
    aggregate_horizon.add_argument(
        "--problems", nargs="+", default=list(DEFAULT_PROBLEMS),
        help="Problems to merge (default: %(default)s).",
    )
    aggregate_horizon.add_argument(
        "--results-dir", type=str, default=DEFAULT_PLANNING_OUT_DIR,
        help="Directory with counterfactual_horizon_{problem}.json files "
        "(default: %(default)s).",
    )
    aggregate_horizon.add_argument(
        "--n-resamples", type=int, default=DEFAULT_N_RESAMPLES,
        help="Bootstrap resamples for the 95%% CI (default: %(default)s).",
    )
    aggregate_horizon.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED,
        help="Seed of the bootstrap PCG64 generator (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Any:
    """CLI entry point: dispatch to the selected subcommand.

    Args:
        argv: Optional argument list; ``None`` reads ``sys.argv``.

    Returns:
        The subcommand's result (paths or payload dict).
    """
    args = parse_args(argv)
    if args.command == "harvest":
        return run_harvest(args)
    if args.command == "evaluate":
        return run_evaluate(args)
    if args.command == "aggregate":
        return run_aggregate(args)
    if args.command == "evaluate-horizon":
        return run_evaluate_horizon(args)
    if args.command == "aggregate-horizon":
        return run_aggregate_horizon(args)
    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
