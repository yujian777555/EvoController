"""Analyze mutation-probability action dynamics of the Phase-1 controllers.

This script reads Phase-1 evaluation trajectories (one JSON per
``(arm, problem, seed)`` run, as produced by ``experiments/run_phase1.py``)
and answers three questions about the action channel of the learned
controllers:

* Q1 - Within-run variation: does the mutation probability (pm) emitted
  by a controller change across generations inside a single run? For
  every arm x problem x seed run we compute the std and range of the pm
  sequence. ``fixed`` and ``constant`` arms are expected to give 0.
* Q2 - State dependence: is the emitted pm correlated with the observed
  evolution state? For every arm and problem we pool (state feature, pm)
  pairs over all seeds and compute the Spearman rank correlation between
  pm and each of hv / igd / diversity / generation.
* Q3 - Problem dependence: does the controller behave differently per
  problem? For ``mlp_w10`` we report the pm distribution (mean / std)
  per problem and run a Kruskal-Wallis test across the five problems.

Convention: generation 0 is excluded from every statistic. Its recorded
action is the algorithm's default (``1 / n_vars``) and is identical for
all arms — it is not emitted by a controller (see the deployment
semantics documented in ``run_phase1.py``) — so keeping it would blur
the fixed/constant sanity check.

Outputs (written under ``--out-dir``, default ``results/analysis``):

* ``action_dynamics.json`` — all numeric results of Q1 / Q2 / Q3.
* ``pm_vs_generation_mlp_w10.png`` — per-problem mean pm (± std band
  across seeds) versus generation for the ``mlp_w10`` arm.
* ``pm_histograms_by_arm.png`` — pm distribution histogram per arm.

Example:
    python experiments/analyze_actions.py \
        --traj-dir results/phase1/trajectories --out-dir results/analysis
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: only write image files, never open a window

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kruskal, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJ_DIR = REPO_ROOT / "results" / "phase1" / "trajectories"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "analysis"

FILENAME_RE = re.compile(
    r"^(?P<arm>[a-z0-9_]+)_(?P<problem>zdt\d+)_seed(?P<seed>\d+)\.json$"
)

#: State features correlated against pm in Q2.
STATE_FEATURES: Tuple[str, ...] = ("hv", "igd", "diversity", "generation")

#: Display / output order of the arms.
ARM_ORDER: Tuple[str, ...] = ("fixed", "constant", "mlp_w1", "mlp_w10")

#: Arm analysed by Q3 (as specified by the task).
Q3_ARM = "mlp_w10"


class Run:
    """One evaluation run loaded from a trajectory JSON.

    Attributes:
        arm: Controller arm name, e.g. ``"mlp_w10"``.
        problem: Benchmark problem name, e.g. ``"zdt1"``.
        seed: Evaluation random seed.
        generation: Per-generation generation indices (generation >= 1).
        pm: Per-generation mutation probability (generation >= 1).
        features: Mapping from state feature name to per-generation array
            (generation >= 1).
    """

    def __init__(
        self,
        arm: str,
        problem: str,
        seed: int,
        generation: np.ndarray,
        pm: np.ndarray,
        features: Dict[str, np.ndarray],
    ) -> None:
        self.arm = arm
        self.problem = problem
        self.seed = seed
        self.generation = generation
        self.pm = pm
        self.features = features

    @property
    def key(self) -> str:
        """Unique human-readable key of this run, e.g. ``mlp_w10_zdt1_seed100``."""
        return f"{self.arm}_{self.problem}_seed{self.seed}"


def load_runs(traj_dir: Path) -> List[Run]:
    """Load every ``{arm}_{problem}_seed{seed}.json`` file in ``traj_dir``.

    Generation 0 is dropped from all arrays (see module docstring).

    Args:
        traj_dir: Directory containing the Phase-1 trajectory JSON files.

    Returns:
        One :class:`Run` per file, sorted by (arm, problem, seed).

    Raises:
        FileNotFoundError: If ``traj_dir`` does not exist or contains no
            matching JSON file.
    """
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"trajectory directory not found: {traj_dir}")
    runs: List[Run] = []
    for path in sorted(traj_dir.glob("*.json")):
        match = FILENAME_RE.match(path.name)
        if match is None:
            print(f"warning: skipping unrecognised file name: {path.name}")
            continue
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        transitions = [t for t in data["transitions"] if t["generation"] >= 1]
        generation = np.array([t["generation"] for t in transitions], dtype=float)
        pm = np.array(
            [t["action"]["mutation_probability"] for t in transitions], dtype=float
        )
        features = {
            "hv": np.array([t["state"]["hv"] for t in transitions], dtype=float),
            "igd": np.array([t["state"]["igd"] for t in transitions], dtype=float),
            "diversity": np.array(
                [t["state"]["diversity"] for t in transitions], dtype=float
            ),
            "generation": generation,
        }
        runs.append(
            Run(
                arm=match.group("arm"),
                problem=match.group("problem"),
                seed=int(match.group("seed")),
                generation=generation,
                pm=pm,
                features=features,
            )
        )
    if not runs:
        raise FileNotFoundError(f"no trajectory JSON files found in: {traj_dir}")
    runs.sort(key=lambda r: (r.arm, r.problem, r.seed))
    return runs


def q1_within_run_variation(runs: Sequence[Run]) -> Dict[str, Any]:
    """Q1: std and range of the pm sequence within each run.

    Returns:
        Nested dict with ``per_run`` (each run's std/range), ``per_arm``
        (mean/max std and mean range aggregated over runs), and
        ``per_arm_problem`` (mean std per arm x problem).
    """
    per_run: Dict[str, Dict[str, float]] = {}
    for run in runs:
        per_run[run.key] = {
            "std": float(np.std(run.pm)),
            "range": float(np.ptp(run.pm)),
        }
    per_arm: Dict[str, Dict[str, Any]] = {}
    for arm in sorted({r.arm for r in runs}):
        arm_runs = [r for r in runs if r.arm == arm]
        stds = np.array([per_run[r.key]["std"] for r in arm_runs])
        ranges = np.array([per_run[r.key]["range"] for r in arm_runs])
        per_arm[arm] = {
            "n_runs": len(arm_runs),
            "mean_std": float(np.mean(stds)),
            "max_std": float(np.max(stds)),
            "mean_range": float(np.mean(ranges)),
            "max_range": float(np.max(ranges)),
        }
    per_arm_problem: Dict[str, Dict[str, float]] = {}
    for arm in sorted({r.arm for r in runs}):
        for problem in sorted({r.problem for r in runs}):
            keys = [
                per_run[r.key]
                for r in runs
                if r.arm == arm and r.problem == problem
            ]
            if keys:
                per_arm_problem.setdefault(arm, {})[problem] = {
                    "mean_std": float(np.mean([k["std"] for k in keys])),
                    "mean_range": float(np.mean([k["range"] for k in keys])),
                }
    return {
        "per_run": per_run,
        "per_arm": per_arm,
        "per_arm_problem": per_arm_problem,
    }


def _spearman(feature: np.ndarray, pm: np.ndarray) -> Dict[str, Any]:
    """Spearman correlation of one feature against pm, with degenerate guards."""
    if np.ptp(pm) == 0.0:
        return {"spearman_rho": None, "p_value": None, "note": "zero variance in pm"}
    if np.ptp(feature) == 0.0:
        return {
            "spearman_rho": None,
            "p_value": None,
            "note": "zero variance in feature",
        }
    result = spearmanr(feature, pm)
    rho, p_value = float(result[0]), float(result[1])
    if np.isnan(rho) or np.isnan(p_value):
        return {"spearman_rho": None, "p_value": None, "note": "undefined (NaN)"}
    return {"spearman_rho": rho, "p_value": p_value}


def q2_state_dependence(runs: Sequence[Run]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Q2: Spearman correlation between pm and each state feature.

    For every arm, correlations are computed per problem (pooling all
    seeds of that arm x problem) and additionally for all problems
    pooled (key ``"all_problems_pooled"``).

    Returns:
        Mapping ``arm -> problem -> feature -> correlation result``.
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for arm in sorted({r.arm for r in runs}):
        out[arm] = {}
        problems = sorted({r.problem for r in runs})
        for problem in problems + ["all_problems_pooled"]:
            pool = [
                r
                for r in runs
                if r.arm == arm
                and (problem == "all_problems_pooled" or r.problem == problem)
            ]
            out[arm][problem] = {
                feature: _spearman(
                    np.concatenate([r.features[feature] for r in pool]),
                    np.concatenate([r.pm for r in pool]),
                )
                for feature in STATE_FEATURES
            }
            out[arm][problem]["n_points"] = int(sum(len(r.pm) for r in pool))
    return out


def q3_problem_dependence(runs: Sequence[Run]) -> Dict[str, Any]:
    """Q3: pm distribution of one arm per problem + Kruskal-Wallis test.

    Returns:
        Mapping with per-problem mean/std/n of pm and the Kruskal-Wallis
        H statistic and p-value across problems.
    """
    arm_runs = [r for r in runs if r.arm == Q3_ARM]
    problems = sorted({r.problem for r in arm_runs})
    per_problem: Dict[str, Dict[str, Any]] = {}
    groups: List[np.ndarray] = []
    for problem in problems:
        pooled = np.concatenate(
            [r.pm for r in arm_runs if r.problem == problem]
        )
        groups.append(pooled)
        per_problem[problem] = {
            "mean": float(np.mean(pooled)),
            "std": float(np.std(pooled)),
            "n": int(pooled.size),
        }
    h_stat, p_value = kruskal(*groups)
    return {
        "arm": Q3_ARM,
        "per_problem": per_problem,
        "kruskal_wallis": {
            "H": float(h_stat),
            "p_value": float(p_value),
            "n_groups": len(groups),
        },
    }


def plot_pm_vs_generation(runs: Sequence[Run], out_dir: Path) -> Path:
    """Plot 1: mean pm (± std band across seeds) vs generation for ``mlp_w10``.

    Args:
        runs: All loaded runs.
        out_dir: Directory the figure is written to.

    Returns:
        Path of the written PNG file.
    """
    arm_runs = sorted(
        [r for r in runs if r.arm == Q3_ARM], key=lambda r: (r.problem, r.seed)
    )
    problems = sorted({r.problem for r in arm_runs})
    n_gens = min(int(len(r.pm)) for r in arm_runs)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for problem in problems:
        matrix = np.array(
            [r.pm[:n_gens] for r in arm_runs if r.problem == problem]
        )
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        gens = arm_runs[0].generation[:n_gens]
        ax.plot(gens, mean, label=problem, linewidth=1.6)
        ax.fill_between(gens, mean - std, mean + std, alpha=0.15)
    ax.axhline(
        1.0 / 30.0,
        linestyle="--",
        color="gray",
        linewidth=1.0,
        label="default 1/n_vars",
    )
    ax.set_xlabel("generation")
    ax.set_ylabel("mutation probability")
    ax.set_title(f"{Q3_ARM}: mutation probability vs generation (mean ± std over seeds)")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    out_path = out_dir / f"pm_vs_generation_{Q3_ARM}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pm_histograms(runs: Sequence[Run], out_dir: Path) -> Path:
    """Plot 2: pm distribution histogram per arm (all problems pooled).

    Args:
        runs: All loaded runs.
        out_dir: Directory the figure is written to.

    Returns:
        Path of the written PNG file.
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(9, 7), sharex=True, sharey=False
    )
    for ax, arm in zip(axes.ravel(), ARM_ORDER):
        pooled = np.concatenate([r.pm for r in runs if r.arm == arm])
        ax.hist(pooled, bins=40, color="#4878d0", edgecolor="white", linewidth=0.3)
        ax.set_title(
            f"{arm}  (mean={pooled.mean():.4f}, std={pooled.std():.4f}, n={pooled.size})",
            fontsize=10,
        )
        ax.set_ylabel("count")
        ax.axvline(1.0 / 30.0, color="gray", linestyle="--", linewidth=1.0)
    for ax in axes[-1, :]:
        ax.set_xlabel("mutation probability")
    fig.suptitle("Mutation probability distribution per arm (all problems, gen >= 1)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path = out_dir / "pm_histograms_by_arm.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Phase-1 controller mutation-probability dynamics (Q1-Q3)."
    )
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=DEFAULT_TRAJ_DIR,
        help=f"directory of trajectory JSON files (default: {DEFAULT_TRAJ_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory for JSON stats and figures (default: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the full analysis and write all outputs.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.traj_dir)
    arms = sorted({r.arm for r in runs})
    problems = sorted({r.problem for r in runs})
    print(
        f"loaded {len(runs)} runs: arms={arms}, problems={problems}, "
        f"generations per run >= {min(len(r.pm) for r in runs)} (gen 0 excluded)"
    )

    q1 = q1_within_run_variation(runs)
    q2 = q2_state_dependence(runs)
    q3 = q3_problem_dependence(runs)

    report: Dict[str, Any] = {
        "meta": {
            "traj_dir": str(args.traj_dir.resolve()),
            "out_dir": str(out_dir.resolve()),
            "n_runs": len(runs),
            "arms": arms,
            "problems": problems,
            "conventions": {
                "excluded_generation_0": True,
                "exclusion_reason": (
                    "generation-0 action is the algorithm default 1/n_vars for "
                    "all arms, not emitted by a controller"
                ),
                "std_convention": "population std (numpy default, ddof=0)",
                "correlation_method": "Spearman rank correlation",
            },
        },
        "q1_within_run_variation": q1,
        "q2_state_dependence": q2,
        "q3_problem_dependence": q3,
    }
    json_path = out_dir / "action_dynamics.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {json_path}")

    plot1_path = plot_pm_vs_generation(runs, out_dir)
    print(f"wrote {plot1_path}")
    plot2_path = plot_pm_histograms(runs, out_dir)
    print(f"wrote {plot2_path}")

    print("\n=== Q1: within-run pm variation (mean of per-run std) ===")
    for arm in ARM_ORDER:
        if arm in q1["per_arm"]:
            stats = q1["per_arm"][arm]
            print(
                f"  {arm:10s} mean_std={stats['mean_std']:.6f} "
                f"max_std={stats['max_std']:.6f} mean_range={stats['mean_range']:.6f}"
            )
    print("\n=== Q3: Kruskal-Wallis across problems (mlp_w10) ===")
    print(
        f"  H={q3['kruskal_wallis']['H']:.4f} "
        f"p={q3['kruskal_wallis']['p_value']:.3e}"
    )
    for problem, stats in q3["per_problem"].items():
        print(
            f"  {problem}: mean={stats['mean']:.6f} std={stats['std']:.6f} n={stats['n']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
