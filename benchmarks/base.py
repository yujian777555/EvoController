from __future__ import annotations

from abc import ABC, abstractmethod

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
