from __future__ import annotations

"""Phase 2.75C, Tasks 3-4: long-horizon collection and cross-problem transfer.

Two jobs live here, both writing into the isolated ``results/phase2_75c/`` tree.

**Task 3 — long-horizon intervention collection** (``--stage collect``).
``experiments/counterfactual_actions.py evaluate-horizon`` is the producer of
intervention data; this runner drives it per problem with the long-horizon
configuration of the plan (``--horizons 20 50 100 --max-states 30
--n-alternatives 10 --n-reps 3`` by default), focuses on the difficult
landscapes (``--problems zdt4 zdt6`` by default) and writes everything to
``--counterfactual-dir`` (default ``results/phase2_75c/counterfactual``).

A **manifest** (``{counterfactual_dir}/manifest.json``) records the full
configuration, a cost estimate derived from the configured grid, the status of
every problem (``pending`` / ``completed`` / ``failed`` with the error), the
measured runtime and the output file. Re-running skips problems already marked
completed with an existing artifact (``--force`` redoes them) and
``--shard/--num-shards`` splits the problem list across parallel workers, so a
~100-generation branch grid can be computed incrementally.

**Task 4 — cross-problem generalization** (``--stage train`` /
``generalize`` / ``all``). The dataset is split **by state** across all
problems (never by row), the held-out states of the *test* problems form the
evaluation set, and two arms are trained on exactly the same held-out split:

``cross_problem``
    trained only on ``--train-problems`` (default zdt1 zdt2 zdt3) — the
    zero-shot arm that never sees a zdt4/zdt6 state.
``all_problems``
    trained on every problem's training states — the in-distribution arm.

Both are evaluated on the same test states with
:func:`controller.advantage_predictor.ranking_metrics` per ``(problem,
horizon)`` (alternatives only by default, mirroring
``experiments/analyze_advantage_generalization.py``, which treats the
controller action as the arm under test rather than as a candidate), and the
result is written to ``{out_dir}/generalization.json`` together with the
training problem sets, the split statistics and the model paths.
``--feature-set {v1,v2}`` selects the 64-column Phase-2.75 dataset or the
76-column problem-aware dataset of Task 1.

Example:
    ``python experiments/run_phase2_75c.py --stage collect --problems zdt4``
    ``python experiments/run_phase2_75c.py --stage all --feature-set v2``
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    # Allow ``python experiments/run_phase2_75c.py`` from the repo root: the
    # script directory (not the repo root) is on sys.path then.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.advantage_predictor import AdvantagePredictor, ranking_metrics
from experiments import train_advantage_predictor as tap

#: Phase-2.75C root of every artifact this runner writes.
DEFAULT_OUT_DIR = "results/phase2_75c"
#: Isolated long-horizon intervention directory (Task 3).
DEFAULT_COUNTERFACTUAL_DIR = "results/phase2_75c/counterfactual"
#: Snapshot corpus consumed by ``evaluate-horizon``.
DEFAULT_SNAPSHOTS_DIR = "results/phase1_75/snapshots"
#: Controller artefacts used for the index-0 candidate of the collection.
DEFAULT_CONTROLLER = "results/phase2_outcome/planning_controller.json"
DEFAULT_PREDICTOR = "results/phase2_outcome/predictor.pt"
DEFAULT_ENCODER = "results/phase2_outcome/encoder.json"
#: Model output directory of the transfer arms.
DEFAULT_MODEL_DIR = "results/phase2_75c/models"
#: Generalization report.
DEFAULT_GENERALIZATION_OUT = "results/phase2_75c/generalization.json"
#: Phase-2.75 (v1, 64-column) dataset directory.
DEFAULT_V1_DATASET_DIR = "results/phase2_75"
#: Advantage definition directory of the v2 (76-column) dataset.
DEFAULT_ADVANTAGE_BASELINE = "state_mean"
#: Long-horizon grid of Task 3.
DEFAULT_COLLECT_HORIZONS: tuple[int, ...] = (20, 50, 100)
DEFAULT_COLLECT_PROBLEMS: tuple[str, ...] = ("zdt4", "zdt6")
DEFAULT_MAX_STATES = 30
DEFAULT_N_ALTERNATIVES = 10
DEFAULT_N_REPS = 3
#: Transfer split of Task 4.
DEFAULT_TRAIN_PROBLEMS: tuple[str, ...] = ("zdt1", "zdt2", "zdt3")
DEFAULT_TEST_PROBLEMS: tuple[str, ...] = ("zdt4", "zdt6")
DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)
#: Measured Phase-2.75 cost of one NSGA-II generation (pop 100 / n_vars 30);
#: used only for the manifest's estimate, never for control flow.
DEFAULT_SECONDS_PER_GENERATION = 0.4
#: Scalar metrics averaged into ``overall`` blocks.
_SCALAR_METRICS: tuple[str, ...] = (
    "spearman_mean",
    "kendall_mean",
    "oracle_hit_rate",
    "regret_mean",
    "oracle_gap_mean",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments of the Phase-2.75C runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2.75C: long-horizon intervention collection (Task 3) and "
            "cross-problem advantage transfer (Task 4)."
        )
    )
    parser.add_argument(
        "--stage", choices=["collect", "train", "generalize", "all"],
        default="all",
        help="'collect' gathers long-horizon interventions; 'train' fits the "
        "transfer arms; 'generalize' evaluates them; 'all' = train -> "
        "generalize (collection stays explicit, it is expensive).",
    )
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Phase-2.75C root (default: %(default)s).")
    # --- Task 3: collection -------------------------------------------------
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_COLLECT_PROBLEMS),
                        help="Problems to collect (default: %(default)s).")
    parser.add_argument("--counterfactual-dir", type=str, default=DEFAULT_COUNTERFACTUAL_DIR,
                        help="Isolated intervention output directory "
                        "(default: %(default)s).")
    parser.add_argument("--snapshots-dir", type=str, default=DEFAULT_SNAPSHOTS_DIR,
                        help="Snapshot corpus (default: %(default)s).")
    parser.add_argument("--controller", type=str, default=DEFAULT_CONTROLLER,
                        help="Controller config for the index-0 candidate "
                        "(default: %(default)s).")
    parser.add_argument("--predictor", type=str, default=DEFAULT_PREDICTOR,
                        help="Outcome predictor of the collection controller "
                        "(default: %(default)s).")
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER,
                        help="Fitted encoder used by the collection controller "
                        "and copied into --model-dir (default: %(default)s).")
    parser.add_argument("--collect-horizons", nargs="+", type=int,
                        default=list(DEFAULT_COLLECT_HORIZONS),
                        help="Long-horizon grid (default: %(default)s).")
    parser.add_argument("--max-states", type=int, default=DEFAULT_MAX_STATES,
                        help="States per problem (default: %(default)s).")
    parser.add_argument("--n-alternatives", type=int, default=DEFAULT_N_ALTERNATIVES,
                        help="Sampled alternatives per state (default: %(default)s).")
    parser.add_argument("--n-reps", type=int, default=DEFAULT_N_REPS,
                        help="Branch replicates per candidate (default: %(default)s).")
    parser.add_argument("--seconds-per-generation", type=float,
                        default=DEFAULT_SECONDS_PER_GENERATION,
                        help="Cost constant of the manifest estimate "
                        "(default: %(default)s).")
    parser.add_argument("--shard", type=int, default=0,
                        help="Shard index over the problem list (default: %(default)s).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Number of shards (default: %(default)s).")
    parser.add_argument("--force", action="store_true",
                        help="Redo problems that the manifest marks completed.")
    # --- Task 4: transfer ---------------------------------------------------
    parser.add_argument("--feature-set", choices=["v1", "v2"], default="v1",
                        help="Dataset family: v1 (64 columns, Phase 2.75) or v2 "
                        "(76 columns, problem-aware) (default: %(default)s).")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Dataset directory; default depends on "
                        "--feature-set and --advantage-baseline.")
    parser.add_argument("--advantage-baseline", type=str,
                        default=DEFAULT_ADVANTAGE_BASELINE,
                        help="v2 advantage definition sub-directory "
                        "(default: %(default)s).")
    parser.add_argument("--train-problems", nargs="+",
                        default=list(DEFAULT_TRAIN_PROBLEMS),
                        help="Problems of the zero-shot arm (default: %(default)s).")
    parser.add_argument("--test-problems", nargs="+",
                        default=list(DEFAULT_TEST_PROBLEMS),
                        help="Problems the transfer is measured on "
                        "(default: %(default)s).")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS),
                        help="Advantage horizons (default: %(default)s).")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Transfer model directory (default: %(default)s).")
    parser.add_argument("--out", type=str, default=DEFAULT_GENERALIZATION_OUT,
                        help="Generalization report (default: %(default)s).")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs per arm (default: %(default)s).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: %(default)s).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size (default: %(default)s).")
    parser.add_argument("--train-seed", type=int, default=0,
                        help="Seed of the split and the networks (default: %(default)s).")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of states held out (default: %(default)s).")
    parser.add_argument("--contrastive", choices=["on", "off"], default="on",
                        help="Paired hinge training (default: %(default)s).")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="Hinge margin (default: %(default)s).")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 128],
                        help="Hidden layer widths (default: %(default)s).")
    parser.add_argument(
        "--evaluation-candidates", choices=["alternatives", "all"],
        default="alternatives",
        help="'alternatives' (default) drops kind == 'controller' from both "
        "training and evaluation, matching "
        "experiments/analyze_advantage_generalization.py; 'all' keeps it.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Task 3: long-horizon intervention collection
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Human-readable ``H:MM:SS`` (or ``M:SS``) duration."""
    total = max(0.0, float(seconds))
    hours, remainder = divmod(int(total), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def estimate_collection(
    problems: Sequence[str],
    *,
    horizons: Sequence[int],
    n_alternatives: int,
    n_reps: int,
    states_per_problem: dict[str, int],
    seconds_per_generation: float,
) -> dict[str, Any]:
    """Cost estimate of one collection run, per problem and in total.

    Every branch runs ``max(horizons)`` generations, and a state costs
    ``(1 + n_alternatives) * n_reps`` branches, so the generation count of a
    problem is ``states * (1 + n_alternatives) * n_reps * max(horizons)``.

    Args:
        problems: Problems to collect.
        horizons: Long-horizon grid.
        n_alternatives: Sampled alternatives per state.
        n_reps: Replicates per candidate.
        states_per_problem: States actually evaluated per problem.
        seconds_per_generation: Measured cost of one generation.

    Returns:
        ``{"branch_generations", "seconds_per_generation", "per_problem":
        {problem: {"n_states", "generations", "estimate_sec"}},
        "total_generations", "total_estimate_sec"}``.

    Raises:
        ValueError: If the horizon grid is empty or the grid sizes are < 1.
    """
    horizon_list = [int(h) for h in horizons]
    if not horizon_list:
        raise ValueError("horizons must not be empty")
    if int(n_alternatives) < 1 or int(n_reps) < 1:
        raise ValueError(
            f"n_alternatives and n_reps must be >= 1, got "
            f"{n_alternatives}/{n_reps}"
        )
    branch_generations = max(horizon_list)
    per_problem: dict[str, Any] = {}
    total_generations = 0
    for problem in problems:
        states = int(states_per_problem.get(str(problem), 0))
        generations = states * (1 + int(n_alternatives)) * int(n_reps) * branch_generations
        total_generations += generations
        per_problem[str(problem)] = {
            "n_states": states,
            "generations": generations,
            "estimate_sec": float(generations) * float(seconds_per_generation),
        }
    return {
        "branch_generations": branch_generations,
        "seconds_per_generation": float(seconds_per_generation),
        "per_problem": per_problem,
        "total_generations": int(total_generations),
        "total_estimate_sec": float(total_generations) * float(seconds_per_generation),
    }


def available_states(snapshots_dir: str | Path, problem: str, cap: int) -> int:
    """States ``evaluate-horizon`` will actually use for one problem.

    The producer evenly subsamples ``min(cap, n_snapshots)`` of the harvested
    snapshots, so the estimate uses the real corpus size when it is present.
    """
    directory = Path(snapshots_dir)
    if directory.is_dir():
        count = len(list(directory.glob(f"{problem}__seed*__gen*.pkl")))
        if count:
            return int(min(int(cap), count))
    return int(cap)


def manifest_path(counterfactual_dir: str | Path) -> Path:
    """Location of the collection manifest."""
    return Path(counterfactual_dir) / "manifest.json"


def load_manifest(counterfactual_dir: str | Path) -> dict[str, Any]:
    """Read the collection manifest, or return an empty skeleton."""
    path = manifest_path(counterfactual_dir)
    if not path.is_file():
        return {"problems": {}}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    payload.setdefault("problems", {})
    return payload


def write_manifest(counterfactual_dir: str | Path, payload: dict[str, Any]) -> Path:
    """Write the collection manifest (creating the directory if needed)."""
    path = manifest_path(counterfactual_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def run_horizon_evaluation(namespace: argparse.Namespace) -> dict[str, Any]:
    """Run one ``evaluate-horizon`` invocation (indirection for tests).

    Returns:
        The payload written by ``run_evaluate_horizon``.
    """
    from experiments.counterfactual_actions import run_evaluate_horizon

    return run_evaluate_horizon(namespace)


def _evaluate_namespace(args: argparse.Namespace, problem: str) -> argparse.Namespace:
    """``evaluate-horizon`` CLI namespace of one problem."""
    from experiments.counterfactual_actions import parse_args as cfa_parse_args

    return cfa_parse_args(
        [
            "evaluate-horizon",
            "--problem", str(problem),
            "--snapshots-dir", str(args.snapshots_dir),
            "--out-dir", str(args.counterfactual_dir),
            "--controller", str(args.controller),
            "--controller-type", "planning",
            "--predictor", str(args.predictor),
            "--encoder", str(args.encoder),
            "--horizons", *[str(h) for h in args.collect_horizons],
            "--n-alternatives", str(args.n_alternatives),
            "--n-reps", str(args.n_reps),
            "--max-states", str(args.max_states),
        ]
    )


def shard_problems(
    problems: Sequence[str], shard: int, num_shards: int
) -> list[str]:
    """Deterministic shard of the problem list.

    Raises:
        ValueError: If the shard indices are inconsistent.
    """
    if int(num_shards) < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= int(shard) < int(num_shards):
        raise ValueError(f"shard must lie in [0, {int(num_shards)}), got {shard}")
    ordered = [str(problem) for problem in problems]
    if int(num_shards) == 1:
        return ordered
    return ordered[int(shard) :: int(num_shards)]


def run_collect_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Collect long-horizon interventions for the requested problems.

    The manifest is refreshed after every problem, so an interrupted or
    sharded run resumes without recomputation.

    Returns:
        The manifest payload.
    """
    problems = shard_problems(args.problems, args.shard, args.num_shards)
    counterfactual_dir = Path(args.counterfactual_dir)
    counterfactual_dir.mkdir(parents=True, exist_ok=True)
    states = {
        problem: available_states(args.snapshots_dir, problem, args.max_states)
        for problem in problems
    }
    estimate = estimate_collection(
        problems,
        horizons=args.collect_horizons,
        n_alternatives=args.n_alternatives,
        n_reps=args.n_reps,
        states_per_problem=states,
        seconds_per_generation=args.seconds_per_generation,
    )
    manifest = load_manifest(counterfactual_dir)
    manifest["config"] = {
        "problems": problems,
        "requested_problems": [str(p) for p in args.problems],
        "shard": int(args.shard),
        "num_shards": int(args.num_shards),
        "counterfactual_dir": str(counterfactual_dir),
        "snapshots_dir": str(args.snapshots_dir),
        "controller": str(args.controller),
        "predictor": str(args.predictor),
        "encoder": str(args.encoder),
        "horizons": [int(h) for h in args.collect_horizons],
        "max_states": int(args.max_states),
        "n_alternatives": int(args.n_alternatives),
        "n_reps": int(args.n_reps),
        "branch_generations": estimate["branch_generations"],
        "protocol": (
            "experiments/counterfactual_actions.py evaluate-horizon: index 0 is "
            "the controller action, 1..N are sampled full actions, every branch "
            "runs max(horizons) generations with one action"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest["estimate"] = estimate
    records = manifest.setdefault("problems", {})
    print(
        f"[collect] {len(problems)} problems, branch generations "
        f"{estimate['branch_generations']}, estimated "
        f"{format_duration(estimate['total_estimate_sec'])}"
    )
    for problem in problems:
        entry = records.setdefault(str(problem), {})
        output = counterfactual_dir / f"counterfactual_horizon_{problem}.json"
        if (
            not args.force
            and entry.get("status") == "completed"
            and output.is_file()
        ):
            print(
                f"[collect] {problem}: already completed "
                f"({entry.get('n_states', '?')} states) -> skipped"
            )
            continue
        per_problem_estimate = estimate["per_problem"][str(problem)]
        entry.update(
            {
                "status": "running",
                "estimate_sec": per_problem_estimate["estimate_sec"],
                "output": output.name,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_manifest(counterfactual_dir, manifest)
        started = time.perf_counter()
        try:
            payload = run_horizon_evaluation(_evaluate_namespace(args, problem))
        except Exception as exc:  # pragma: no cover - environment dependent
            entry.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "runtime_sec": float(time.perf_counter() - started),
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            write_manifest(counterfactual_dir, manifest)
            print(f"[collect] {problem}: FAILED ({entry['error']})")
            continue
        runtime = float(time.perf_counter() - started)
        entry.update(
            {
                "status": "completed",
                "error": None,
                "n_states": len(payload.get("states", [])),
                "n_candidates": len(
                    payload.get("states", [{}])[0].get("candidates", [])
                )
                if payload.get("states")
                else 0,
                "runtime_sec": runtime,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_manifest(counterfactual_dir, manifest)
        print(
            f"[collect] {problem}: {entry['n_states']} states, "
            f"{entry['n_candidates']} candidates, took {format_duration(runtime)} "
            f"(estimated {format_duration(per_problem_estimate['estimate_sec'])})"
        )
    completed = [
        problem
        for problem in problems
        if records.get(str(problem), {}).get("status") == "completed"
    ]
    failed = [
        problem
        for problem in problems
        if records.get(str(problem), {}).get("status") == "failed"
    ]
    measured = sum(
        float(records[str(problem)].get("runtime_sec", 0.0))
        for problem in problems
        if records.get(str(problem), {}).get("status") == "completed"
    )
    manifest["summary"] = {
        "shard_completed": completed,
        "shard_failed": failed,
        "shard_pending": [
            problem
            for problem in problems
            if records.get(str(problem), {}).get("status") not in ("completed", "failed")
        ],
        "shard_measured_sec": measured,
    }
    path = write_manifest(counterfactual_dir, manifest)
    print(
        f"[collect] completed={completed} failed={failed} "
        f"measured={format_duration(measured)} -> {path}"
    )
    return manifest


# ---------------------------------------------------------------------------
# Task 4: cross-problem transfer
# ---------------------------------------------------------------------------


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    """Dataset directory implied by ``--feature-set``/``--dataset-dir``.

    Raises:
        ValueError: If the resolved directory does not exist.
    """
    if args.dataset_dir:
        directory = Path(args.dataset_dir)
    elif str(args.feature_set) == "v1":
        directory = Path(DEFAULT_V1_DATASET_DIR)
    else:
        directory = Path(args.out_dir) / str(args.advantage_baseline)
    if not directory.is_dir():
        raise ValueError(
            f"dataset directory not found: {directory} (feature-set="
            f"{args.feature_set}); build it first "
            f"(build_intervention_dataset.py / build_intervention_dataset_v2.py)"
        )
    return directory


def load_problem_arrays(
    dataset_dir: str | Path,
    problems: Sequence[str],
    *,
    drop_controller: bool,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load and concatenate the npz files of the requested problems.

    Args:
        dataset_dir: Directory with ``intervention_dataset_*.npz``.
        problems: Problem whitelist (matched against each file's own
            ``problem`` array, so both the v1 and the v2 naming work).
        drop_controller: Drop rows with ``candidate_kind == "controller"``
            (the arm under test rather than a candidate).

    Returns:
        ``(arrays, files)`` with the concatenated columns.

    Raises:
        FileNotFoundError: If no npz matches the requested problems.
        ValueError: If a file lacks a required array.
    """
    required = (
        "X",
        "y_adv",
        "problem",
        "seed",
        "generation",
        "horizon",
        "candidate_index",
        "candidate_kind",
    )
    wanted = {str(problem) for problem in problems}
    columns: dict[str, list[np.ndarray]] = {name: [] for name in required}
    files: list[str] = []
    for path in sorted(Path(dataset_dir).glob("intervention_dataset_*.npz")):
        with np.load(path) as arrays:
            missing = [name for name in required if name not in arrays]
            if missing:
                raise ValueError(f"{path} is missing arrays {missing}")
            problem_labels = np.asarray(arrays["problem"]).astype(str)
            keep = np.isin(problem_labels, sorted(wanted))
            if drop_controller:
                keep &= np.asarray(arrays["candidate_kind"]).astype(str) != "controller"
            if not keep.any():
                continue
            for name in required:
                columns[name].append(np.asarray(arrays[name])[keep])
        files.append(path.name)
    if not files:
        raise FileNotFoundError(
            f"no intervention_dataset_*.npz for problems {sorted(wanted)} in "
            f"{dataset_dir}"
        )
    merged = {
        name: np.concatenate(values) if values else np.zeros(0)
        for name, values in columns.items()
    }
    return merged, files


def _block_problem(block: dict[str, Any]) -> str:
    """Problem name of a state block key."""
    return str(block["key"]).split("|")[0]


def _stack_blocks(blocks: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the ``X``/``y`` of several blocks."""
    if not blocks:
        return np.zeros((0, 0)), np.zeros((0, 0))
    return (
        np.vstack([block["X"] for block in blocks]),
        np.vstack([block["y"] for block in blocks]),
    )


def evaluate_transfer(
    predictor: AdvantagePredictor,
    blocks: Sequence[dict[str, Any]],
    horizons: Sequence[int],
    problems: Sequence[str],
) -> dict[str, Any]:
    """Ranking metrics per problem and horizon on a set of state blocks.

    Returns:
        ``{"per_problem": {problem: {"overall": {...}, "per_horizon": {...}}},
        "pooled": {...}}``; problems without blocks get ``n_groups = 0``.
    """
    per_problem: dict[str, Any] = {}
    pooled_predictions: list[np.ndarray] = []
    pooled_realized: list[np.ndarray] = []
    for problem in problems:
        selected = [
            block for block in blocks if _block_problem(block) == str(problem)
        ]
        entry: dict[str, Any] = {"n_states": len(selected), "per_horizon": {}}
        for column, horizon in enumerate(horizons):
            if not selected:
                entry["per_horizon"][str(int(horizon))] = {"n_groups": 0}
                continue
            predictions = np.asarray(
                [predictor.predict(block["X"])[:, column] for block in selected]
            )
            realized = np.asarray([block["y"][:, column] for block in selected])
            entry["per_horizon"][str(int(horizon))] = ranking_metrics(
                predictions, realized
            )
            pooled_predictions.append(predictions)
            pooled_realized.append(realized)
        overall: dict[str, Any] = {"n_states": len(selected)}
        for metric in _SCALAR_METRICS:
            values = [
                block[metric]
                for block in entry["per_horizon"].values()
                if block.get(metric) is not None
            ]
            overall[metric] = float(np.mean(values)) if values else None
        entry["overall"] = overall
        per_problem[str(problem)] = entry
    pooled: dict[str, Any] = {}
    if pooled_predictions:
        pooled_metrics = ranking_metrics(
            np.vstack(pooled_predictions), np.vstack(pooled_realized)
        )
        pooled = pooled_metrics
    return {"per_problem": per_problem, "pooled": pooled}


def train_arm(
    name: str,
    train_blocks: Sequence[dict[str, Any]],
    val_blocks: Sequence[dict[str, Any]],
    horizons: Sequence[int],
    args: argparse.Namespace,
    input_dim: int,
) -> dict[str, Any]:
    """Train one transfer arm (optionally contrastively) and save it.

    Args:
        name: Arm identifier (``cross_problem`` / ``all_problems``).
        train_blocks: Training states of this arm.
        val_blocks: Held-out states (used for the validation loss only).
        horizons: Horizon grid.
        args: Parsed arguments.
        input_dim: Feature width.

    Returns:
        Arm record with the model path, row counts and loss curves.
    """
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    predictor = AdvantagePredictor(
        input_dim=int(input_dim),
        horizons=[int(h) for h in horizons],
        hidden_dims=tuple(int(w) for w in args.hidden_dims),
        seed=int(args.train_seed),
        lr=float(args.lr),
    )
    x_val, y_val = _stack_blocks(val_blocks)
    if str(args.contrastive) == "on":
        x_train, y_train, x_neg, y_neg = tap.contrastive_pairs(train_blocks)
        rows = int(x_train.shape[0])
        history = predictor.fit(
            x_train,
            y_train,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            X_val=x_val,
            y_val=y_val,
            X_neg=x_neg,
            y_neg=y_neg,
            margin=float(args.margin),
        )
    else:
        x_train, y_train = _stack_blocks(train_blocks)
        rows = int(x_train.shape[0])
        history = predictor.fit(
            x_train,
            y_train,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            X_val=x_val,
            y_val=y_val,
        )
    model_path = model_dir / f"model_{name}.pt"
    predictor.save(model_path)
    print(
        f"[train] {name}: {len(train_blocks)} states / {rows} rows -> {model_path}"
    )
    return {
        "model": str(model_path),
        "n_states": len(train_blocks),
        "n_rows": rows,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
    }


def run_train_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Train the zero-shot and in-distribution transfer arms.

    The state-level split is computed over **all** problems, so the held-out
    states of the test problems are never part of either arm's training set.

    Returns:
        The training record written into ``generalization.json``.

    Raises:
        ValueError: If the dataset has no usable state block for the split.
    """
    dataset_dir = resolve_dataset_dir(args)
    problems = list(dict.fromkeys([*args.train_problems, *args.test_problems]))
    arrays, files = load_problem_arrays(
        dataset_dir,
        problems,
        drop_controller=str(args.evaluation_candidates) == "alternatives",
    )
    blocks, skipped = tap.build_state_blocks(arrays, args.horizons)
    if not blocks:
        raise ValueError(
            f"no usable state blocks for horizons {args.horizons} in {dataset_dir}"
        )
    train_blocks, val_blocks = tap.split_blocks(
        blocks, float(args.val_fraction), int(args.train_seed)
    )
    test_problems = {str(problem) for problem in args.test_problems}
    train_problems = {str(problem) for problem in args.train_problems}
    evaluation_blocks = [
        block for block in val_blocks if _block_problem(block) in test_problems
    ]
    cross_train = [
        block for block in train_blocks if _block_problem(block) in train_problems
    ]
    all_train = list(train_blocks)
    if not evaluation_blocks:
        raise ValueError(
            f"no held-out states for test problems {sorted(test_problems)}; "
            f"increase --val-fraction or check the dataset"
        )
    if not cross_train:
        raise ValueError(
            f"no training states for train problems {sorted(train_problems)}"
        )
    train_keys = {str(block["key"]) for block in all_train}
    evaluation_keys = {str(block["key"]) for block in evaluation_blocks}
    leaked = sorted(train_keys & evaluation_keys)
    if leaked:
        raise ValueError(
            f"state leak between training and evaluation sets: {leaked[:5]}"
        )
    input_dim = int(np.asarray(cross_train[0]["X"]).shape[1])
    train_record: dict[str, Any] = {
        "config": {
            "feature_set": str(args.feature_set),
            "dataset_dir": str(dataset_dir),
            "dataset_files": files,
            "advantage_baseline": (
                str(args.advantage_baseline)
                if str(args.feature_set) == "v2"
                else None
            ),
            "evaluation_candidates": str(args.evaluation_candidates),
            "train_problems": sorted(train_problems),
            "test_problems": sorted(test_problems),
            "horizons": [int(h) for h in args.horizons],
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "train_seed": int(args.train_seed),
            "val_fraction": float(args.val_fraction),
            "contrastive": str(args.contrastive),
            "margin": float(args.margin),
            "hidden_dims": [int(w) for w in args.hidden_dims],
            "input_dim": input_dim,
        },
        "split": {
            "unit": "state = (problem, seed, generation); all candidates and horizons stay together",
            "n_states_total": len(blocks),
            "n_states_train": len(train_blocks),
            "n_states_val": len(val_blocks),
            "n_states_dropped": len(skipped),
            "n_states_evaluation": len(evaluation_blocks),
            "n_states_cross_train": len(cross_train),
            "n_states_all_train": len(all_train),
            "train_state_keys": sorted(train_keys),
            "evaluation_state_keys": sorted(evaluation_keys),
            "evaluation_problem_counts": {
                problem: sum(
                    1
                    for block in evaluation_blocks
                    if _block_problem(block) == problem
                )
                for problem in sorted(test_problems)
            },
        },
        "arms": {},
    }
    for name, arm_blocks in (
        ("cross_problem", cross_train),
        ("all_problems", all_train),
    ):
        train_record["arms"][name] = train_arm(
            name, arm_blocks, evaluation_blocks, args.horizons, args, input_dim
        )
        train_record["arms"][name]["train_problems"] = sorted(
            {_block_problem(block) for block in arm_blocks}
        )
    train_record["blocks"] = {
        "evaluation": evaluation_blocks,
        "cross_train": cross_train,
        "all_train": all_train,
    }
    from controller.state_encoder import StateEncoder

    encoder_source = args.encoder
    if encoder_source is None:
        meta_path = Path(dataset_dir) / "intervention_meta.json"
        if meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as fh:
                encoder_source = json.load(fh).get("config", {}).get("encoder")
    if encoder_source is None:
        raise FileNotFoundError(
            "no encoder recorded in the dataset meta; pass --encoder"
        )
    StateEncoder.load(encoder_source).save(Path(args.model_dir) / "encoder.json")
    return train_record


def run_generalize_stage(
    args: argparse.Namespace, train_record: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Evaluate the trained arms on the held-out states of the test problems.

    Args:
        args: Parsed arguments.
        train_record: Output of :func:`run_train_stage`; when ``None`` the
            stage re-loads the dataset and the saved models.

    Returns:
        The generalization payload written to ``--out``.

    Raises:
        FileNotFoundError: If a trained arm is missing.
    """
    if train_record is None:
        train_record = run_train_stage(args)
    blocks = train_record.get("blocks")
    if blocks is None:  # re-derive when the record came from a saved dict
        dataset_dir = resolve_dataset_dir(args)
        problems = list(dict.fromkeys([*args.train_problems, *args.test_problems]))
        arrays, _files = load_problem_arrays(
            dataset_dir,
            problems,
            drop_controller=str(args.evaluation_candidates) == "alternatives",
        )
        all_blocks, _skipped = tap.build_state_blocks(arrays, args.horizons)
        evaluation_keys = set(train_record["split"]["evaluation_state_keys"])
        blocks = {
            "evaluation": [
                block for block in all_blocks if str(block["key"]) in evaluation_keys
            ]
        }
    evaluation_blocks = blocks["evaluation"]
    arms: dict[str, Any] = {}
    per_problem: dict[str, dict[str, Any]] = {
        str(problem): {} for problem in args.test_problems
    }
    for name in ("cross_problem", "all_problems"):
        record = train_record["arms"][name]
        model_path = Path(record["model"])
        if not model_path.is_file():
            raise FileNotFoundError(f"trained arm not found: {model_path}")
        predictor = AdvantagePredictor.load(model_path)
        report = evaluate_transfer(
            predictor, evaluation_blocks, args.horizons, args.test_problems
        )
        arms[name] = {
            "model": str(model_path),
            "train_problems": record["train_problems"],
            "n_states_train": record["n_states"],
            "n_rows_train": record["n_rows"],
            "pooled": report["pooled"],
        }
        for problem, entry in report["per_problem"].items():
            per_problem.setdefault(problem, {})[name] = entry
    payload: dict[str, Any] = {
        "config": train_record["config"],
        "training_problem_sets": {
            "cross_problem": train_record["arms"]["cross_problem"]["train_problems"],
            "all_problems": train_record["arms"]["all_problems"]["train_problems"],
        },
        "arms": arms,
        "per_problem": per_problem,
        "split": train_record["split"],
        "losses": {
            name: {
                "train_loss": train_record["arms"][name]["train_loss"],
                "val_loss": train_record["arms"][name]["val_loss"],
            }
            for name in arms
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[generalize] wrote {out_path}")
    for problem in args.test_problems:
        for name in ("cross_problem", "all_problems"):
            entry = per_problem.get(str(problem), {}).get(name, {})
            overall = entry.get("overall", {})
            rho = overall.get("spearman_mean")
            hit = overall.get("oracle_hit_rate")
            print(
                f"[generalize] {problem} {name}: spearman="
                f"{'n/a' if rho is None else round(rho, 3)} hit="
                f"{'n/a' if hit is None else round(hit, 3)}"
            )
    return payload


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch on ``--stage``."""
    stage = str(args.stage)
    if stage == "collect":
        return run_collect_stage(args)
    if stage == "train":
        return run_train_stage(args)
    if stage == "generalize":
        return run_generalize_stage(args)
    record = run_train_stage(args)
    return run_generalize_stage(args, record)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point."""
    return run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
