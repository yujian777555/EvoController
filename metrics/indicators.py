from __future__ import annotations

"""Quality indicators for two-objective minimization fronts.

All indicators follow the EvoController trajectory-recording conventions:
objectives are minimized, fronts are ``(k, 2)`` arrays of objective vectors,
and the true Pareto front (when available) is supplied by the benchmark.
"""

import numpy as np

__all__ = ["hypervolume", "igd", "diversity_spread"]


def _as_front(front: np.ndarray, name: str) -> np.ndarray:
    """Validate that ``front`` is a 2-D array with exactly two columns.

    Args:
        front: Candidate objective-value array.
        name: Parameter name used in the error message.

    Returns:
        The input converted to a float array.

    Raises:
        ValueError: If the array is not 2-D with shape ``(k, 2)``.
    """
    arr = np.asarray(front, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape (k, 2); got shape {arr.shape}."
        )
    return arr


def _filter_nondominated(front: np.ndarray) -> np.ndarray:
    """Reduce a 2-objective minimization front to its nondominated points.

    A point ``p`` is dominated by point ``q`` iff ``q <= p`` in both
    objectives and ``q < p`` in at least one. Exact duplicates are kept
    only once, since they contribute nothing to the dominated region.

    Args:
        front: Array of shape ``(k, 2)``.

    Returns:
        Array of shape ``(m, 2)``, ``m <= k``, with dominated points and
        duplicates removed, sorted by the first objective ascending.
    """
    front = np.unique(front, axis=0)
    keep = np.ones(len(front), dtype=bool)
    for i in range(len(front)):
        if not keep[i]:
            continue
        for j in range(len(front)):
            if i == j or not keep[j]:
                continue
            if (
                front[j, 0] <= front[i, 0]
                and front[j, 1] <= front[i, 1]
                and (front[j, 0] < front[i, 0] or front[j, 1] < front[i, 1])
            ):
                keep[i] = False
                break
    nd = front[keep]
    order = np.argsort(nd[:, 0], kind="stable")
    return nd[order]


def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Exclusive hypervolume dominated by ``front`` w.r.t. ``ref_point``.

    Computed for 2-objective minimization via the sort-and-sweep method:
    after filtering to nondominated points inside the reference region,
    sort by ``f1`` ascending and accumulate horizontal slices
    ``(f1_{i+1} - f1_i) * (ref2 - f2_i)``, where ``f1_{i+1}`` is the next
    point's first objective or ``ref1`` for the last point. Points with
    ``f1 >= ref1`` or ``f2 >= ref2`` do not dominate the reference point
    and are ignored.

    The caller may pass a front that is not nondominated; dominated points
    and duplicates are filtered internally and cannot inflate the value.

    Example:
        ``hypervolume([[0.2, 0.8], [0.8, 0.2]], [1.0, 1.0]) == 0.28``
        (slices ``0.6 * 0.2`` and ``0.2 * 0.8``).

    Args:
        front: Objective vectors, shape ``(k, 2)``. May be empty (k = 0).
        ref_point: Reference point, shape ``(2,)``; should be strictly
            worse than every front point of interest.

    Returns:
        The dominated hypervolume, a non-negative float. Returns 0.0 when
        no front point lies strictly inside the reference region.

    Raises:
        ValueError: If ``front`` is not shaped ``(k, 2)`` or ``ref_point``
            is not shaped ``(2,)``.
    """
    pts = _as_front(front, "front")
    ref = np.asarray(ref_point, dtype=float)
    if ref.ndim != 1 or ref.shape[0] != 2:
        raise ValueError(
            f"ref_point must have shape (2,); got shape {ref.shape}."
        )
    if len(pts) == 0:
        return 0.0

    inside = pts[(pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])]
    if len(inside) == 0:
        return 0.0

    nd = _filter_nondominated(inside)
    n = len(nd)
    hv = 0.0
    for i in range(n):
        f1_next = nd[i + 1, 0] if i + 1 < n else ref[0]
        hv += (f1_next - nd[i, 0]) * (ref[1] - nd[i, 1])
    return float(hv)


def igd(front: np.ndarray, reference_front: np.ndarray) -> float:
    """Inverted Generational Distance of ``front`` to ``reference_front``.

    IGD is the mean over reference points of the Euclidean distance to the
    nearest obtained front point:
    ``IGD = (1/|R|) * sum_{r in R} min_{p in F} ||r - p||_2``.
    Lower is better; 0.0 means every reference point is covered exactly.

    An empty obtained front cannot approximate anything; by convention this
    returns ``float("inf")`` rather than raising, so that trajectory
    recording never crashes on a degenerate generation.

    Args:
        front: Obtained objective vectors, shape ``(k, 2)``; may be empty.
        reference_front: True Pareto front samples, shape ``(n, 2)``; must
            contain at least one point.

    Returns:
        The IGD value (non-negative), or ``inf`` for an empty front.

    Raises:
        ValueError: If either array is not shaped ``(m, 2)``, or if
            ``reference_front`` is empty.
    """
    ref = _as_front(reference_front, "reference_front")
    pts = _as_front(front, "front")
    if len(ref) == 0:
        raise ValueError("reference_front must contain at least one point.")
    if len(pts) == 0:
        return float("inf")
    dists = np.linalg.norm(ref[:, None, :] - pts[None, :, :], axis=2)
    return float(np.mean(np.min(dists, axis=1)))


def diversity_spread(front: np.ndarray) -> float:
    """Deb et al. spread metric Delta of an obtained front.

    Computed from the obtained front alone (no true-front extreme points),
    following Deb, *Multi-Objective Optimization using Evolutionary
    Algorithms* (2001), with this no-true-front convention: the boundary
    points are the obtained points with minimum and maximum ``f1``; ``df``
    and ``dl`` are their nearest-neighbor distances within the front; for
    every remaining point ``i``, ``d_i`` is its nearest-neighbor distance
    and ``d_bar`` their mean::

        Delta = (df + dl + sum_i |d_i - d_bar|) / (df + dl + k * d_bar)

    where ``k`` is the number of non-boundary points. Delta near 0
    indicates a uniform spread; larger values indicate clustering or gaps.

    Edge-case conventions: a front with fewer than 2 points carries no
    spread information and yields 0.0; with exactly 2 points both are
    boundary points, there is nothing to compare them against, and the
    spread is defined as 0.0.

    Args:
        front: Obtained objective vectors, shape ``(k, 2)``.

    Returns:
        The spread value Delta (non-negative float); 0.0 for ``k <= 2``.

    Raises:
        ValueError: If ``front`` is not shaped ``(k, 2)``.
    """
    pts = _as_front(front, "front")
    k = len(pts)
    if k <= 2:
        return 0.0

    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    nn = np.min(dists, axis=1)

    order = np.argsort(pts[:, 0], kind="stable")
    first, last = order[0], order[-1]
    interior_mask = np.ones(k, dtype=bool)
    interior_mask[first] = False
    interior_mask[last] = False
    interior_nn = nn[interior_mask]

    df = float(nn[first])
    dl = float(nn[last])
    d_bar = float(np.mean(interior_nn))
    denom = df + dl + len(interior_nn) * d_bar
    if denom == 0.0:
        return 0.0
    delta = (df + dl + float(np.sum(np.abs(interior_nn - d_bar)))) / denom
    return float(delta)
