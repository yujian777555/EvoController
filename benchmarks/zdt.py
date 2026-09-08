from __future__ import annotations

import numpy as np

from benchmarks.base import Problem

_NONDOM_TOL = 1e-12


def _nondominated_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask of nondominated rows in a 2-objective minimization set.

    A point p dominates q iff p_i <= q_i for all objectives and p_j < q_j for
    at least one. Vectorized pairwise comparison; used to filter densely
    sampled fronts (e.g. the discontinuous ZDT3/ZDT6 fronts) down to their
    nondominated subset.
    """
    n = points.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        le = np.all(points <= points[i] + _NONDOM_TOL, axis=1)
        lt = np.any(points < points[i] - _NONDOM_TOL, axis=1)
        dominators = le & lt
        dominators[i] = False
        if np.any(dominators):
            dominated[i] = True
    return ~dominated


def _resample_front(f1: np.ndarray, f2: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a dense nondominated front to exactly ``n_points`` points.

    Linear interpolation over f1 (monotone on each nondominated segment after
    sorting) keeps all returned points on or arbitrarily close to the true
    front, which is sufficient for IGD reference sets.
    """
    order = np.argsort(f1)
    f1_sorted = f1[order]
    f2_sorted = f2[order]
    targets = np.linspace(f1_sorted[0], f1_sorted[-1], n_points)
    f2_interp = np.interp(targets, f1_sorted, f2_sorted)
    return np.column_stack([targets, f2_interp])


class _ZDTBase(Problem):
    """Shared machinery for the Zitzler-Deb-Thiele test suite.

    All ZDT problems follow f1(x), f2(x) = g(x) * h(f1, g) with minimization
    semantics. Reference: Zitzler, Deb & Thiele, "Comparison of
    Multiobjective Evolutionary Algorithms: Empirical Results",
    Evolutionary Computation 8(2), 2000.
    """

    _N_DENSE = 2000

    def _check_shape(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_vars,):
            raise ValueError(
                f"{self.name}.evaluate expected x of shape ({self.n_vars},), "
                f"got {x.shape}"
            )
        return x

    def _reference_front_continuous(self, n_points: int) -> np.ndarray:
        """Reference front for problems whose optimal front is a smooth curve.

        With g at its minimum (g = 1), f2 is a deterministic function of f1;
        a uniform grid in f1 is already nondominated for convex/concave
        fronts (f2 monotone in f1 on the optimal front).
        """
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = self._front_f2(f1)
        return np.column_stack([f1, f2])

    def _reference_front_filtered(self, f1_dense: np.ndarray, f2_dense: np.ndarray, n_points: int) -> np.ndarray:
        """Reference front for discontinuous fronts (ZDT3, ZDT6).

        Dense parametric samples are filtered to the nondominated subset,
        then resampled to exactly ``n_points`` along the surviving segments.
        """
        mask = _nondominated_mask(np.column_stack([f1_dense, f2_dense]))
        return _resample_front(f1_dense[mask], f2_dense[mask], n_points)

    def _front_f2(self, f1: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class ZDT1(_ZDTBase):
    """ZDT1: convex Pareto front.

    n = 30, x in [0, 1]^30.
    f1 = x1; g = 1 + 9 * mean(x[1:]); f2 = g * (1 - sqrt(f1 / g)).
    Optimal front (g = 1): f2 = 1 - sqrt(f1), f1 in [0, 1].
    """

    @property
    def name(self) -> str:
        return "zdt1"

    @property
    def n_vars(self) -> int:
        return 30

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.zeros(self.n_vars)

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.ones(self.n_vars)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = self._check_shape(x)
        f1 = x[0]
        g = 1.0 + 9.0 * np.mean(x[1:])
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2])

    def _front_f2(self, f1: np.ndarray) -> np.ndarray:
        return 1.0 - np.sqrt(f1)

    def reference_front(self, n_points: int = 200) -> np.ndarray:
        return self._reference_front_continuous(n_points)


class ZDT2(_ZDTBase):
    """ZDT2: nonconvex (concave) Pareto front.

    Same structure as ZDT1 but f2 = g * (1 - (f1 / g)^2).
    Optimal front (g = 1): f2 = 1 - f1^2, f1 in [0, 1].
    """

    @property
    def name(self) -> str:
        return "zdt2"

    @property
    def n_vars(self) -> int:
        return 30

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.zeros(self.n_vars)

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.ones(self.n_vars)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = self._check_shape(x)
        f1 = x[0]
        g = 1.0 + 9.0 * np.mean(x[1:])
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2])

    def _front_f2(self, f1: np.ndarray) -> np.ndarray:
        return 1.0 - f1 ** 2

    def reference_front(self, n_points: int = 200) -> np.ndarray:
        return self._reference_front_continuous(n_points)


class ZDT3(_ZDTBase):
    """ZDT3: disconnected Pareto front.

    Same g as ZDT1 but h = 1 - sqrt(f1/g) - (f1/g) * sin(10 * pi * f1).
    The sine term makes the optimal front (g = 1) discontinuous: only
    subsets of f1 in [0, 1] are nondominated. The reference front is built
    by dense sampling, nondominated filtering, and resampling per connected
    segment.
    """

    @property
    def name(self) -> str:
        return "zdt3"

    @property
    def n_vars(self) -> int:
        return 30

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.zeros(self.n_vars)

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.ones(self.n_vars)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = self._check_shape(x)
        f1 = x[0]
        g = 1.0 + 9.0 * np.mean(x[1:])
        h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
        f2 = g * h
        return np.array([f1, f2])

    def reference_front(self, n_points: int = 200) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, self._N_DENSE)
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
        mask = _nondominated_mask(np.column_stack([f1, f2]))
        f1_nd, f2_nd = f1[mask], f2[mask]

        # Split into connected segments (large f1 gaps) so interpolation does
        # not draw dominated chords across the discontinuities.
        gaps = np.where(np.diff(f1_nd) > 5.0 * (1.0 / self._N_DENSE))[0]
        seg_starts = np.concatenate([[0], gaps + 1])
        seg_ends = np.concatenate([gaps + 1, [len(f1_nd)]])
        seg_lengths = seg_ends - seg_starts
        # Largest-remainder allocation: counts sum exactly to n_points and
        # stay non-negative even when n_points < number of segments.
        raw = seg_lengths / seg_lengths.sum() * n_points
        counts = np.floor(raw).astype(int)
        leftover = n_points - int(counts.sum())
        for idx in np.argsort(-(raw - counts))[:leftover]:
            counts[idx] += 1

        parts = []
        for (s, e), c in zip(zip(seg_starts, seg_ends), counts):
            if c > 0:
                parts.append(_resample_front(f1_nd[s:e], f2_nd[s:e], int(c)))
        return np.vstack(parts)


class ZDT4(_ZDTBase):
    """ZDT4: convex front with many local fronts (multimodal g).

    n = 10, x1 in [0, 1], xi in [-5, 5] for i > 1.
    g = 1 + 10 * (n - 1) + sum(xi^2 - 10 * cos(4 * pi * xi)); f2 = g * (1 - sqrt(f1 / g)).
    g is minimized (g = 1) at xi = 0; optimal front: f2 = 1 - sqrt(f1).
    """

    @property
    def name(self) -> str:
        return "zdt4"

    @property
    def n_vars(self) -> int:
        return 10

    @property
    def lower_bounds(self) -> np.ndarray:
        lb = np.full(self.n_vars, -5.0)
        lb[0] = 0.0
        return lb

    @property
    def upper_bounds(self) -> np.ndarray:
        ub = np.full(self.n_vars, 5.0)
        ub[0] = 1.0
        return ub

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = self._check_shape(x)
        f1 = x[0]
        xi = x[1:]
        g = 1.0 + 10.0 * (self.n_vars - 1) + np.sum(xi ** 2 - 10.0 * np.cos(4.0 * np.pi * xi))
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2])

    def _front_f2(self, f1: np.ndarray) -> np.ndarray:
        return 1.0 - np.sqrt(f1)

    def reference_front(self, n_points: int = 200) -> np.ndarray:
        return self._reference_front_continuous(n_points)


class ZDT6(_ZDTBase):
    """ZDT6: nonconvex, nonuniformly spaced Pareto front.

    n = 10, x in [0, 1]^10.
    f1 = 1 - exp(-4 * x1) * sin(6 * pi * x1)^6;
    g = 1 + 9 * (mean(x[1:]))^0.25; f2 = g * (1 - (f1 / g)^2).
    The optimal x1 minimizing f1 is approximately 0.280775, so the true
    front covers f1 in roughly [0.280775, 1] with f2 = 1 - f1^2 (g = 1).
    """

    @property
    def name(self) -> str:
        return "zdt6"

    @property
    def n_vars(self) -> int:
        return 10

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.zeros(self.n_vars)

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.ones(self.n_vars)

    @staticmethod
    def _f1_of_x1(x1: np.ndarray | float) -> np.ndarray | float:
        return 1.0 - np.exp(-4.0 * x1) * np.sin(6.0 * np.pi * x1) ** 6

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = self._check_shape(x)
        f1 = self._f1_of_x1(x[0])
        g = 1.0 + 9.0 * np.mean(x[1:]) ** 0.25
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2])

    def reference_front(self, n_points: int = 200) -> np.ndarray:
        # Parametrize by x1 (f1 is not monotone in x1), then filter.
        x1 = np.linspace(0.0, 1.0, self._N_DENSE)
        f1 = np.asarray(self._f1_of_x1(x1))
        f2 = 1.0 - f1 ** 2
        return self._reference_front_filtered(f1, f2, n_points)
