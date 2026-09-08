from __future__ import annotations

"""Fixed-length numeric descriptors of benchmark problems.

Phase 1.5 trains one controller across problems, so the controller input
must identify which problem is being optimized. This module defines the
canonical 9-dimensional problem feature vector derived from
:meth:`benchmarks.base.Problem.describe` (seeded sampling, deterministic).
``n_objs`` is deliberately excluded: it is constant (2) across all
benchmarks and would be a zero-variance feature.
"""

import math

import numpy as np

from benchmarks.base import Problem

PROBLEM_FEATURE_NAMES: list[str] = [
    "n_vars_log",
    "bounds_width_mean",
    "f1_std",
    "f2_std",
    "f_corr",
    "ideal_est_0",
    "ideal_est_1",
    "nadir_est_0",
    "nadir_est_1",
]


def problem_feature_vector(problem: Problem) -> np.ndarray:
    """Compute the canonical 9-dimensional descriptor of a problem.

    Calls ``problem.describe()`` with default arguments (seed 0), so the
    vector is deterministic for a given problem. ``n_vars`` is encoded as
    ``log10(n_vars)`` to keep its scale comparable to the other features.
    Objective means are excluded in favor of the more informative spreads
    and ideal/nadir anchors.

    Args:
        problem: The benchmark problem to describe.

    Returns:
        Array of shape ``(9,)``, dtype ``float64``, with entries in
        :data:`PROBLEM_FEATURE_NAMES` order.
    """
    desc = problem.describe()
    ideal = desc["ideal_est"]
    nadir = desc["nadir_est"]
    return np.asarray(
        [
            math.log10(float(desc["n_vars"])),
            float(desc["bounds_width_mean"]),
            float(desc["f1_std"]),
            float(desc["f2_std"]),
            float(desc["f_corr"]),
            float(ideal[0]),
            float(ideal[1]),
            float(nadir[0]),
            float(nadir[1]),
        ],
        dtype=np.float64,
    )
