from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Problem(ABC):
    """Abstract base class for continuous multi-objective benchmark problems.

    All problems in EvoController are MINIMIZATION problems: lower objective
    values are better. This convention is required by the trajectory recorder,
    where rewards are defined as hypervolume gains and IGD reductions.

    The fixed interface allows algorithms (e.g. NSGA-II) and the trajectory
    recorder to be written against a single contract independent of the
    concrete benchmark.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical lowercase problem identifier (e.g. ``"zdt1"``)."""

    @property
    @abstractmethod
    def n_vars(self) -> int:
        """Number of decision variables."""

    @property
    def n_objs(self) -> int:
        """Number of objectives. All Phase-0 benchmarks are bi-objective."""
        return 2

    @property
    @abstractmethod
    def lower_bounds(self) -> np.ndarray:
        """Lower bound of each decision variable, shape ``(n_vars,)``."""

    @property
    @abstractmethod
    def upper_bounds(self) -> np.ndarray:
        """Upper bound of each decision variable, shape ``(n_vars,)``."""

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate a single decision vector.

        Args:
            x: Decision vector of shape ``(n_vars,)``.

        Returns:
            Objective vector of shape ``(2,)`` to be minimized.

        Raises:
            ValueError: If ``x`` does not have shape ``(n_vars,)``.
        """

    @abstractmethod
    def reference_front(self, n_points: int = 200) -> np.ndarray:
        """Sample the true Pareto front of the problem.

        Used as the reference set for IGD computation during trajectory
        recording. Points lie on the optimal front (minimal ``g``).

        Args:
            n_points: Number of points to return.

        Returns:
            Array of shape ``(n_points, 2)`` of mutually nondominated
            objective vectors on the true Pareto front.
        """

    def describe(self, n_samples: int = 256, seed: int = 0) -> dict[str, Any]:
        """Estimate cheap landscape statistics by uniform random sampling.

        Draws ``n_samples`` seeded uniform random points from the decision
        space and evaluates them, summarizing the problem without running
        any search. The statistics give a controller problem-identifying
        context (Phase 1.5) at negligible cost:

        * ``n_vars``: decision-space dimension; scales the default mutation
          probability (``1 / n_vars``) and the search difficulty.
        * ``n_objs``: number of objectives (2 for all current benchmarks).
        * ``bounds_width_mean``: mean of ``upper_bounds - lower_bounds``;
          the length scale of the decision space.
        * ``f1_mean`` / ``f2_mean``: objective locations under uniform
          sampling (objective offset / scale indicator).
        * ``f1_std`` / ``f2_std``: objective spreads; proxy for how
          strongly the landscape responds to decision perturbations.
        * ``f_corr``: Pearson correlation between f1 and f2 over the
          samples; near -1 indicates strongly conflicting objectives
          (trade-off dominated landscape), near +1 aligned ones. Set to
          0.0 when either objective has zero variance (correlation
          undefined).
        * ``ideal_est`` / ``nadir_est``: per-objective min / max over the
          samples, as ``[f1, f2]`` lists; rough anchors of the
          objective-space extent (sampling estimates, not the true
          ideal/nadir points).

        Args:
            n_samples: Number of uniform random evaluation points; >= 1.
            seed: Seed of the sampling RNG. Identical seeds give identical
                results.

        Returns:
            Dict with exactly the keys listed above.

        Raises:
            ValueError: If ``n_samples`` < 1.
        """
        if int(n_samples) < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        rng = np.random.default_rng(seed)
        lower = np.asarray(self.lower_bounds, dtype=float)
        upper = np.asarray(self.upper_bounds, dtype=float)
        xs = rng.uniform(lower, upper, size=(int(n_samples), self.n_vars))
        objectives = np.asarray([self.evaluate(x) for x in xs], dtype=float)
        f1 = objectives[:, 0]
        f2 = objectives[:, 1]
        if int(n_samples) >= 2 and f1.std() > 0.0 and f2.std() > 0.0:
            f_corr = float(np.corrcoef(f1, f2)[0, 1])
        else:
            f_corr = 0.0
        return {
            "n_vars": int(self.n_vars),
            "n_objs": int(self.n_objs),
            "bounds_width_mean": float(np.mean(upper - lower)),
            "f1_mean": float(f1.mean()),
            "f1_std": float(f1.std()),
            "f2_mean": float(f2.mean()),
            "f2_std": float(f2.std()),
            "f_corr": f_corr,
            "ideal_est": [float(f1.min()), float(f2.min())],
            "nadir_est": [float(f1.max()), float(f2.max())],
        }
