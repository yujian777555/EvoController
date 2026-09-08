"""NSGA-II multi-objective evolutionary algorithm.

Faithful NumPy implementation of NSGA-II as published in:

    K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist
    multiobjective genetic algorithm: NSGA-II," IEEE Transactions on
    Evolutionary Computation, vol. 6, no. 2, pp. 182-197, Apr. 2002.
    doi:10.1109/4235.996017

Conventions (required by the EvoController trajectory contract):

* All objectives are MINIMIZED (see ``benchmarks.base.Problem``).
* One generation = offspring creation (binary tournament selection + SBX
  crossover + polynomial mutation) followed by elitist (mu + lambda)
  environmental selection using fast non-dominated sorting and crowding
  distance.
* All stochastic draws go through a single ``numpy.random.Generator``
  seeded with PCG64, so a run is bit-for-bit reproducible given the seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.base import Problem

#: Same epsilon as Deb's reference C implementation: variables closer than
#: this are treated as identical and copied without crossover.
_EPS = 1.0e-14


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True iff objective vector ``a`` Pareto-dominates ``b``.

    Minimization convention: ``a`` dominates ``b`` iff ``a`` is no worse in
    every objective and strictly better in at least one. Identical vectors
    do not dominate each other.
    """
    return bool(np.all(a <= b) and np.any(a < b))


def _fast_nondominated_sort(f: np.ndarray) -> list[list[int]]:
    """Fast non-dominated sort (Deb et al. 2002, Sec. III-A).

    Args:
        f: Objective matrix of shape ``(n, n_objs)`` (minimization).

    Returns:
        List of fronts; each front is a list of row indices into ``f``.
        Front 0 is the non-dominated (rank-1) set. Runs in O(n_objs * n^2).
    """
    n = f.shape[0]
    dominated_set: list[list[int]] = [[] for _ in range(n)]  # S_p in the paper
    domination_count = np.zeros(n, dtype=np.int64)  # n_p in the paper
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(f[p], f[q]):
                dominated_set[p].append(q)
            elif _dominates(f[q], f[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    current = fronts[0]
    while current:
        next_front: list[int] = []
        for p in current:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        current = next_front
    return fronts


def _crowding_distance(f: np.ndarray, indices: list[int]) -> np.ndarray:
    """Crowding distance of one front (Deb et al. 2002, Sec. III-B).

    For each objective, points are sorted and interior points accumulate the
    side lengths of their cuboid normalized by the objective range. Boundary
    points (min or max in any objective) receive infinite distance so that
    environmental selection always preserves them.

    Args:
        f: Objective matrix of shape ``(n, n_objs)``.
        indices: Rows of ``f`` forming a single non-dominated front.

    Returns:
        Array of shape ``(len(indices),)`` aligned with ``indices``.
    """
    k = len(indices)
    distance = np.zeros(k, dtype=np.float64)
    if k <= 2:
        distance[:] = np.inf
        return distance
    sub = f[np.asarray(indices, dtype=np.int64)]
    for m in range(sub.shape[1]):
        order = np.argsort(sub[:, m], kind="stable")
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = sub[order[-1], m] - sub[order[0], m]
        if span <= 0.0:
            continue  # all values equal in this objective: no information
        for i in range(1, k - 1):
            if np.isinf(distance[order[i]]):
                continue
            distance[order[i]] += (sub[order[i + 1], m] - sub[order[i - 1], m]) / span
    return distance


@dataclass(frozen=True)
class OperatorConfig:
    """Configuration of NSGA-II variation operators.

    Attributes:
        crossover_operator: Only ``"sbx"`` (simulated binary crossover) is
            supported in Phase 0.
        crossover_prob: Probability of applying crossover to a parent pair
            (p_c in Deb et al. 2002); parents are cloned otherwise.
        mutation_operator: Only ``"polynomial"`` mutation is supported in
            Phase 0.
        mutation_prob: Per-variable mutation probability (p_m). ``None``
            resolves to the standard default ``1.0 / n_vars``.
        eta_c: SBX distribution index; larger values produce offspring
            closer to the parents.
        eta_m: Polynomial mutation distribution index; larger values produce
            smaller perturbations.
    """

    crossover_operator: str = "sbx"
    crossover_prob: float = 0.9
    mutation_operator: str = "polynomial"
    mutation_prob: float | None = None  # None -> default 1.0 / n_vars
    eta_c: float = 20.0
    eta_m: float = 20.0


class NSGAII:
    """Elitist non-dominated sorting genetic algorithm (Deb et al. 2002).

    The implementation follows the reference algorithm: fast non-dominated
    sort, crowding distance assignment, crowded binary tournament selection,
    SBX crossover with per-variable swap probability 0.5, polynomial
    mutation, and (mu + lambda) environmental selection by rank then
    crowding distance.

    All randomness flows through one ``numpy.random.Generator(PCG64(seed))``
    stored on the instance; two instances constructed with the same problem,
    configuration, and seed produce bit-identical populations.
    """

    def __init__(
        self,
        problem: Problem,
        pop_size: int,
        operators: OperatorConfig,
        seed: int,
    ) -> None:
        """Create an NSGA-II instance (no evaluations happen until initialize()).

        Args:
            problem: Benchmark problem implementing the ``Problem`` contract
                (continuous variables, minimization objectives).
            pop_size: Population size mu; must be positive and even so that
                offspring can be produced in parent pairs.
            operators: Variation operator configuration.
            seed: Seed for the PCG64 random generator.

        Raises:
            ValueError: If ``pop_size`` is not positive, is odd, an operator
                name is unsupported, or a probability lies outside [0, 1].
        """
        if pop_size <= 0:
            raise ValueError(f"pop_size must be positive, got {pop_size}")
        if pop_size % 2 != 0:
            raise ValueError(f"pop_size must be even (parent pairs), got {pop_size}")
        if operators.crossover_operator != "sbx":
            raise ValueError(
                f"unsupported crossover_operator {operators.crossover_operator!r}; only 'sbx' is implemented"
            )
        if operators.mutation_operator != "polynomial":
            raise ValueError(
                f"unsupported mutation_operator {operators.mutation_operator!r}; only 'polynomial' is implemented"
            )
        if not 0.0 <= operators.crossover_prob <= 1.0:
            raise ValueError(f"crossover_prob must lie in [0, 1], got {operators.crossover_prob}")
        if operators.mutation_prob is not None and not 0.0 <= operators.mutation_prob <= 1.0:
            raise ValueError(f"mutation_prob must lie in [0, 1], got {operators.mutation_prob}")

        self._problem = problem
        self._pop_size = int(pop_size)
        self._operators = operators
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._mutation_prob = (
            float(operators.mutation_prob)
            if operators.mutation_prob is not None
            else 1.0 / problem.n_vars
        )
        self._lb = np.asarray(problem.lower_bounds, dtype=np.float64)
        self._ub = np.asarray(problem.upper_bounds, dtype=np.float64)
        if self._lb.shape != (problem.n_vars,) or self._ub.shape != (problem.n_vars,):
            raise ValueError(
                f"bounds must have shape (n_vars,) = ({problem.n_vars},), "
                f"got {self._lb.shape} and {self._ub.shape}"
            )
        if np.any(self._lb > self._ub):
            raise ValueError("lower_bounds must not exceed upper_bounds")

        self._generation = -1
        self._population_x: np.ndarray | None = None
        self._population_f: np.ndarray | None = None
        self._ranks: np.ndarray | None = None
        self._crowding: np.ndarray | None = None

    @property
    def generation(self) -> int:
        """Current generation counter; -1 before initialize(), 0 right after."""
        return self._generation

    @property
    def population_x(self) -> np.ndarray:
        """Decision variables of the current population, shape ``(pop_size, n_vars)``.

        Returns a defensive copy so callers cannot mutate internal state.

        Raises:
            RuntimeError: If called before initialize().
        """
        if self._population_x is None:
            raise RuntimeError("population does not exist yet; call initialize() first")
        return self._population_x.copy()

    @property
    def population_f(self) -> np.ndarray:
        """Objective values of the current population, shape ``(pop_size, n_objs)``.

        Returns a defensive copy so callers cannot mutate internal state.

        Raises:
            RuntimeError: If called before initialize().
        """
        if self._population_f is None:
            raise RuntimeError("population does not exist yet; call initialize() first")
        return self._population_f.copy()

    def initialize(self) -> None:
        """Sample the initial population uniformly in the bounds and evaluate it.

        Resets the generation counter to 0. Re-initialization reuses the same
        random generator, so a second call does not reproduce the first
        population.
        """
        self._population_x = self._rng.uniform(
            self._lb, self._ub, size=(self._pop_size, self._problem.n_vars)
        )
        self._population_f = self._evaluate(self._population_x)
        self._generation = 0
        self._update_rank_crowding()

    def step(self) -> None:
        """Advance the population by exactly one NSGA-II generation.

        Offspring of size mu are created by crowded binary tournament
        selection, SBX crossover, and polynomial mutation; the (mu + lambda)
        combined parent+offspring population is then reduced back to mu by
        non-dominated rank and crowding distance (elitist replacement).

        Raises:
            RuntimeError: If called before initialize().
        """
        if self._population_x is None or self._population_f is None:
            raise RuntimeError("population does not exist yet; call initialize() first")
        mating_indices = self._tournament_selection()
        offspring_x = self._make_offspring(mating_indices)
        offspring_f = self._evaluate(offspring_x)
        combined_x = np.vstack([self._population_x, offspring_x])
        combined_f = np.vstack([self._population_f, offspring_f])
        selected = self._environmental_selection(combined_f)
        self._population_x = combined_x[selected]
        self._population_f = combined_f[selected]
        self._generation += 1
        self._update_rank_crowding()

    def nondominated_front(self) -> np.ndarray:
        """Objective values of the current rank-1 solutions.

        Returns:
            Array of shape ``(k, n_objs)`` with ``1 <= k <= pop_size``,
            sorted by ascending f1 (f2 as tie-breaker for determinism).

        Raises:
            RuntimeError: If called before initialize().
        """
        if self._population_f is None or self._ranks is None:
            raise RuntimeError("population does not exist yet; call initialize() first")
        front = self._population_f[np.flatnonzero(self._ranks == 0)]
        order = np.lexsort((front[:, 1], front[:, 0]))
        return front[order].copy()

    def current_action(self) -> dict:
        """Resolved variation action currently applied by the algorithm.

        Returns:
            Dict with keys ``"mutation_operator"`` (str) and
            ``"mutation_probability"`` (float), the latter resolved to its
            effective per-variable value (``1.0 / n_vars`` when the config
            left it as ``None``). Matches the action schema of the Phase-0
            trajectory recorder.
        """
        return {
            "mutation_operator": self._operators.mutation_operator,
            "mutation_probability": float(self._mutation_prob),
        }

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate a population row-wise via ``problem.evaluate``."""
        return np.asarray([self._problem.evaluate(row) for row in x], dtype=np.float64)

    def _update_rank_crowding(self) -> None:
        """Recompute non-domination ranks and crowding distances of the population."""
        assert self._population_f is not None
        fronts = _fast_nondominated_sort(self._population_f)
        ranks = np.empty(self._pop_size, dtype=np.int64)
        crowding = np.zeros(self._pop_size, dtype=np.float64)
        for rank, front in enumerate(fronts):
            distances = _crowding_distance(self._population_f, front)
            for idx, dist in zip(front, distances):
                ranks[idx] = rank
                crowding[idx] = dist
        self._ranks = ranks
        self._crowding = crowding

    def _tournament_selection(self) -> np.ndarray:
        """Fill a mating pool of size mu via crowded binary tournaments.

        The crowded comparison operator (Deb et al. 2002, Sec. III-B) prefers
        the lower rank, and on equal rank the larger crowding distance;
        remaining ties are broken by a fair coin flip from the generator.
        """
        assert self._ranks is not None and self._crowding is not None
        pool = np.empty(self._pop_size, dtype=np.int64)
        for slot in range(self._pop_size):
            i, j = self._rng.integers(0, self._pop_size, size=2)
            pool[slot] = self._crowded_pick(int(i), int(j))
        return pool

    def _crowded_pick(self, i: int, j: int) -> int:
        """Return the winner index of the crowded comparison between i and j."""
        assert self._ranks is not None and self._crowding is not None
        if self._ranks[i] != self._ranks[j]:
            return i if self._ranks[i] < self._ranks[j] else j
        if self._crowding[i] != self._crowding[j]:
            return i if self._crowding[i] > self._crowding[j] else j
        return i if self._rng.random() < 0.5 else j

    def _make_offspring(self, mating_indices: np.ndarray) -> np.ndarray:
        """Generate mu offspring from consecutive parent pairs in the mating pool."""
        assert self._population_x is not None
        parents = self._population_x[mating_indices]
        offspring = np.empty_like(parents)
        for pair in range(0, self._pop_size, 2):
            p1, p2 = parents[pair], parents[pair + 1]
            if self._rng.random() <= self._operators.crossover_prob:
                c1, c2 = self._sbx(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring[pair] = self._polynomial_mutation(c1)
            offspring[pair + 1] = self._polynomial_mutation(c2)
        return offspring

    def _sbx(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simulated binary crossover (Deb & Agrawal 1995; Deb et al. 2002).

        Each variable is crossed with probability 0.5. For crossed variables,
        the spread factor beta_q is sampled from the SBX distribution
        (Eq. 10/11 in the reference C code) restricted to the variable
        bounds, one beta_q per child, and the two child values are swapped
        between the offspring with probability 0.5. Results are clipped to
        the bounds.
        """
        c1, c2 = p1.copy(), p2.copy()
        eta_c = self._operators.eta_c
        for k in range(self._problem.n_vars):
            if self._rng.random() > 0.5:
                continue
            y1, y2 = float(p1[k]), float(p2[k])
            if abs(y1 - y2) <= _EPS:
                continue
            if y1 > y2:
                y1, y2 = y2, y1
            lo, hi = float(self._lb[k]), float(self._ub[k])
            rand = self._rng.random()
            child1 = self._sbx_child(y1, y2, 2.0 * (y1 - lo), rand, eta_c, lower_side=True)
            child2 = self._sbx_child(y1, y2, 2.0 * (hi - y2), rand, eta_c, lower_side=False)
            child1 = min(max(child1, lo), hi)
            child2 = min(max(child2, lo), hi)
            if self._rng.random() <= 0.5:
                c1[k], c2[k] = child1, child2
            else:
                c1[k], c2[k] = child2, child1
        return c1, c2

    @staticmethod
    def _sbx_child(
        y1: float, y2: float, twice_margin: float, rand: float, eta_c: float, lower_side: bool
    ) -> float:
        """Sample one SBX child coordinate for y1 < y2.

        ``twice_margin`` is ``2*(y1-lo)`` for the lower child and
        ``2*(hi-y2)`` for the upper child, i.e. twice the distance from the
        nearer bound; ``lower_side`` selects which child is produced.
        """
        beta = 1.0 + twice_margin / (y2 - y1)
        alpha = 2.0 - beta ** -(eta_c + 1.0)
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
        if lower_side:
            return 0.5 * ((y1 + y2) - betaq * (y2 - y1))
        return 0.5 * ((y1 + y2) + betaq * (y2 - y1))

    def _polynomial_mutation(self, x: np.ndarray) -> np.ndarray:
        """Polynomial mutation (Deb & Goyal 1996; Deb et al. 2002).

        Each variable is perturbed with probability ``mutation_prob``
        (default ``1/n_vars``) by delta_q * (hi - lo), where delta_q follows
        the polynomial distribution with index eta_m. Mutated values are
        clipped to the bounds.
        """
        y = x.copy()
        eta_m = self._operators.eta_m
        mut_pow = 1.0 / (eta_m + 1.0)
        for k in range(self._problem.n_vars):
            if self._rng.random() > self._mutation_prob:
                continue
            lo, hi = float(self._lb[k]), float(self._ub[k])
            if hi - lo <= 0.0:
                continue  # fixed variable
            yk = min(max(float(y[k]), lo), hi)
            delta1 = (yk - lo) / (hi - lo)
            delta2 = (hi - yk) / (hi - lo)
            rnd = self._rng.random()
            if rnd <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * xy ** (eta_m + 1.0)
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * xy ** (eta_m + 1.0)
                deltaq = 1.0 - val ** mut_pow
            y[k] = min(max(yk + deltaq * (hi - lo), lo), hi)
        return y

    def _environmental_selection(self, combined_f: np.ndarray) -> np.ndarray:
        """Select mu survivors from 2*mu by rank, then crowding distance.

        Whole fronts are admitted while they fit; the first overflowing front
        is truncated by descending crowding distance (Deb et al. 2002,
        Sec. III-C). Stable sorting keeps the choice deterministic.
        """
        fronts = _fast_nondominated_sort(combined_f)
        selected: list[int] = []
        for front in fronts:
            if len(selected) + len(front) <= self._pop_size:
                selected.extend(front)
                continue
            remaining = self._pop_size - len(selected)
            distances = _crowding_distance(combined_f, front)
            order = np.argsort(-distances, kind="stable")
            selected.extend(front[idx] for idx in order[:remaining])
            break
        return np.asarray(selected, dtype=np.int64)
