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
  stochastic draw and no candidate inherits the harvest RNG position.

Example:
    ``python experiments/counterfactual_actions.py harvest``
    ``python experiments/counterfactual_actions.py evaluate --problem zdt1 \
        --controller results/phase1_75/controller.pt --encoder results/phase1_75/encoder.json``
    ``python experiments/counterfactual_actions.py aggregate``
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

if __package__ in (None, ""):
    # Allow ``python experiments/counterfactual_actions.py`` from the repo
    # root: the script directory (not the repo root) is on sys.path then.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.nsga2 import NSGAII, OperatorConfig
from benchmarks import get_problem
from benchmarks.base import Problem
from controller.dataset import merge_state_reward
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
    actions: list[dict[str, Any]] = [
        {
            "mutation_operator": str(controller_action["mutation_operator"]),
            "mutation_probability": float(controller_action["mutation_probability"]),
            "exploration_strength": float(controller_action["exploration_strength"]),
        }
    ]
    for k in range(1, int(n_alternatives) + 1):
        rng_k = np.random.Generator(np.random.PCG64([hash_seed, k]))
        op_k, pm_k, es_k = sample_full_action(rng_k, base_pm, pm_mult_range)
        actions.append(
            {
                "mutation_operator": str(op_k),
                "mutation_probability": float(pm_k),
                "exploration_strength": float(es_k),
            }
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
    encoder = _load_encoder(args.encoder)
    pm_mult_range = (float(args.pm_mult_range[0]), float(args.pm_mult_range[1]))
    controller: MultiHeadController | PlanningController
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
    else:
        controller = MultiHeadController.load(args.controller)
        if encoder.dim != controller.input_dim:
            raise ValueError(
                f"encoder dim {encoder.dim} does not match controller input_dim "
                f"{controller.input_dim}; pass the encoder the controller was trained with"
            )
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
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the counterfactual evaluation.

    Args:
        argv: Argument list to parse; ``None`` reads ``sys.argv``.

    Returns:
        Parsed namespace with ``command`` in {harvest, evaluate, aggregate}
        and the subcommand-specific settings.
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
    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
