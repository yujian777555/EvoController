"""NSGA-II multi-objective evolutionary algorithm.

Faithful NumPy implementation of NSGA-II as published in:

    K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist
    multiobjective genetic algorithm: NSGA-II," IEEE Transactions on
    Evolutionary Computation, vol. 6, no. 2, pp. 182-197, Apr. 2002.
    doi:10.1109/4235.996017

Conventions (required by the EvoController trajectory contract):

* All objectives are MINIMIZED (see ``benchmarks.base.Problem``).
* One generation = offspring creation (binary tournament selection + SBX
  crossover + polynomial or Gaussian mutation) followed by elitist
  (mu + lambda) environmental selection using fast non-dominated sorting
  and crowding distance.
* All stochastic draws go through a single ``numpy.random.Generator``
  seeded with PCG64, so a run is bit-for-bit reproducible given the seed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

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


def _dominance_matrix(f: np.ndarray) -> np.ndarray:
    """Pairwise Pareto-dominance matrix (vectorized ``_dominates``).

    Phase-2.75D: the scalar ``_dominates`` loop dominated the runtime of every
    evaluation (98% of a counterfactual branch step, ~10M NumPy calls per
    100-generation run). The relation is computed here in one broadcast so the
    cost is O(n_objs * n^2) NumPy work instead of n^2 Python-level calls.

    Args:
        f: Objective matrix of shape ``(n, n_objs)`` (minimization).

    Returns:
        Boolean matrix ``d`` of shape ``(n, n)`` where ``d[p, q]`` is True iff
        row ``p`` dominates row ``q`` (diagonal is False). Identical to
        ``_dominates(f[p], f[q])`` for every pair.
    """
    no_worse = np.all(f[:, None, :] <= f[None, :, :], axis=2)
    strictly_better = np.any(f[:, None, :] < f[None, :, :], axis=2)
    dominates = no_worse & strictly_better
    np.fill_diagonal(dominates, False)
    return dominates


def _fast_nondominated_sort(f: np.ndarray) -> list[list[int]]:
    """Fast non-dominated sort (Deb et al. 2002, Sec. III-A).

    Args:
        f: Objective matrix of shape ``(n, n_objs)`` (minimization).

    Returns:
        List of fronts; each front is a list of row indices into ``f``.
        Front 0 is the non-dominated (rank-1) set. Runs in O(n_objs * n^2).
    """
    n = f.shape[0]
    dominates = _dominance_matrix(f)
    # ``dominated_set`` and ``domination_count`` reproduce Deb's S_p and n_p
    # using the identical ordering the scalar implementation produced (q in
    # ascending index order), so front membership *and* intra-front order are
    # bit-identical to the pre-Phase-2.75D implementation.
    dominated_set: list[np.ndarray] = [np.flatnonzero(dominates[p]) for p in range(n)]
    domination_count = dominates.sum(axis=0).astype(np.int64)
    fronts: list[list[int]] = [[]]
    for p in range(n):
        if domination_count[p] == 0:
            fronts[0].append(p)
    current = fronts[0]
    while current:
        next_front: list[int] = []
        for p in current:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(int(q))
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
        mutation_operator: ``"polynomial"`` (Deb & Goyal 1996) or
            ``"gaussian"`` (range-scaled Gaussian perturbation).
        mutation_prob: Per-variable mutation probability (p_m). ``None``
            resolves to the standard default ``1.0 / n_vars``.
        eta_c: SBX distribution index; larger values produce offspring
            closer to the parents.
        eta_m: Polynomial mutation distribution index; larger values produce
            smaller perturbations. Acts as the exploration strength of the
            polynomial operator.
        gaussian_sigma: Standard deviation of the Gaussian mutation as a
            fraction of the variable range (``x_i += N(0, 1) * sigma *
            (xu_i - xl_i)``). Acts as the exploration strength of the
            Gaussian operator. Only used when
            ``mutation_operator == "gaussian"``.
    """

    crossover_operator: str = "sbx"
    crossover_prob: float = 0.9
    mutation_operator: str = "polynomial"
    mutation_prob: float | None = None  # None -> default 1.0 / n_vars
    eta_c: float = 20.0
    eta_m: float = 20.0
    gaussian_sigma: float = 0.1


class NSGAII:
    """Elitist non-dominated sorting genetic algorithm (Deb et al. 2002).

    The implementation follows the reference algorithm: fast non-dominated
    sort, crowding distance assignment, crowded binary tournament selection,
    SBX crossover with per-variable swap probability 0.5, polynomial or
    Gaussian mutation, and (mu + lambda) environmental selection by rank then
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
        if operators.mutation_operator not in ("polynomial", "gaussian"):
            raise ValueError(
                f"unsupported mutation_operator {operators.mutation_operator!r}; "
                "expected 'polynomial' or 'gaussian'"
            )
        if not operators.gaussian_sigma > 0.0:
            raise ValueError(f"gaussian_sigma must be positive, got {operators.gaussian_sigma}")
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
        # Action values actually used by the most recent step(); None until
        # the first step (or after re-initialization) means the resolved
        # config defaults.
        self._last_mutation_prob: float | None = None
        self._last_mutation_operator: str | None = None
        self._last_exploration_strength: float | None = None
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

    @property
    def rng(self) -> np.random.Generator:
        """The instance random generator behind every stochastic draw.

        Exposed so counterfactual branch evaluation (Phase 1.75) can reseed
        individual branch steps between ``snapshot_state()`` /
        ``restore_state()`` calls. Replacing the generator mid-generation
        would break bit-identical replay; swap it only while the instance
        sits at a snapshot boundary.
        """
        return self._rng

    @rng.setter
    def rng(self, generator: np.random.Generator) -> None:
        if not isinstance(generator, np.random.Generator):
            raise TypeError(
                f"rng must be a numpy.random.Generator, got {type(generator).__name__}"
            )
        self._rng = generator

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
        self._last_mutation_prob = None
        self._last_mutation_operator = None
        self._last_exploration_strength = None
        self._update_rank_crowding()

    def step(
        self,
        mutation_prob: float | None = None,
        mutation_operator: str | None = None,
        exploration_strength: float | None = None,
    ) -> None:
        """Advance the population by exactly one NSGA-II generation.

        Offspring of size mu are created by crowded binary tournament
        selection, SBX crossover, and mutation with the effective operator;
        the (mu + lambda) combined parent+offspring population is then
        reduced back to mu by non-dominated rank and crowding distance
        (elitist replacement).

        Args:
            mutation_prob: Optional per-variable mutation probability used
                for THIS generation only (Phase-1 controller action
                injection). ``None`` keeps the configured default
                (``operators.mutation_prob``, resolved to ``1.0 / n_vars``
                when unset).
            mutation_operator: Optional mutation operator for THIS
                generation only; ``"polynomial"`` or ``"gaussian"``.
                ``None`` keeps ``operators.mutation_operator``.
            exploration_strength: Optional exploration strength for THIS
                generation only, interpreted as the polynomial distribution
                index ``eta_m`` when the effective operator is polynomial
                and as ``sigma`` when it is Gaussian. ``None`` keeps the
                configured default of the effective operator (``eta_m`` or
                ``gaussian_sigma``). Must be positive.

        The values actually applied are stored on the instance and exposed
        via ``current_action()``.

        Raises:
            RuntimeError: If called before initialize().
            ValueError: If ``mutation_prob`` lies outside [0, 1],
                ``mutation_operator`` is unsupported, or
                ``exploration_strength`` is not positive.
        """
        if self._population_x is None or self._population_f is None:
            raise RuntimeError("population does not exist yet; call initialize() first")
        if mutation_prob is not None and not 0.0 <= mutation_prob <= 1.0:
            raise ValueError(f"mutation_prob must lie in [0, 1], got {mutation_prob}")
        if mutation_operator is not None and mutation_operator not in ("polynomial", "gaussian"):
            raise ValueError(
                f"unsupported mutation_operator {mutation_operator!r}; "
                "expected 'polynomial' or 'gaussian'"
            )
        if exploration_strength is not None and not exploration_strength > 0.0:
            raise ValueError(f"exploration_strength must be positive, got {exploration_strength}")
        pm = float(mutation_prob) if mutation_prob is not None else self._mutation_prob
        operator = (
            mutation_operator if mutation_operator is not None else self._operators.mutation_operator
        )
        strength = (
            float(exploration_strength)
            if exploration_strength is not None
            else self._default_exploration_strength(operator)
        )
        mating_indices = self._tournament_selection()
        offspring_x = self._make_offspring(mating_indices, pm, operator, strength)
        offspring_f = self._evaluate(offspring_x)
        combined_x = np.vstack([self._population_x, offspring_x])
        combined_f = np.vstack([self._population_f, offspring_f])
        selected = self._environmental_selection(combined_f)
        self._population_x = combined_x[selected]
        self._population_f = combined_f[selected]
        self._generation += 1
        self._last_mutation_prob = pm
        self._last_mutation_operator = operator
        self._last_exploration_strength = strength
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
        """Variation action actually applied by the most recent step.

        Returns:
            Dict with keys ``"mutation_operator"`` (str),
            ``"mutation_probability"`` (float), and
            ``"exploration_strength"`` (float). All values are the ones
            actually used by the last ``step()`` call — injected overrides
            or, for arguments left as ``None`` (and before any step), the
            resolved config defaults. ``exploration_strength`` is the
            ``eta_m`` (polynomial) or ``sigma`` (Gaussian) in effect for
            the effective operator. Matches the Phase-1.5 action schema of
            the trajectory recorder.
        """
        operator = (
            self._last_mutation_operator
            if self._last_mutation_operator is not None
            else self._operators.mutation_operator
        )
        pm = self._last_mutation_prob if self._last_mutation_prob is not None else self._mutation_prob
        strength = (
            self._last_exploration_strength
            if self._last_exploration_strength is not None
            else self._default_exploration_strength(operator)
        )
        return {
            "mutation_operator": operator,
            "mutation_probability": float(pm),
            "exploration_strength": float(strength),
        }

    def snapshot_state(self) -> dict[str, Any]:
        """Capture the complete state needed to replay future generations.

        The snapshot is a plain dict of deep copies (safe to pickle and to
        reuse across any number of :meth:`restore_state` calls) holding the
        population decision variables and objective values, the derived
        rank/crowding state, the generation counter, the action values of
        the most recent step (so ``current_action()`` is restored
        faithfully), the full bit-generator state of the instance RNG, and
        the resolved operator configuration actually in force (default
        mutation probability, operator, exploration strengths, crossover).
        Restoring the snapshot and re-executing the same ``step()`` call
        reproduces the next generation bit-identically.

        Returns:
            Dict with keys ``generation``, ``population_x``,
            ``population_f``, ``ranks``, ``crowding``,
            ``last_mutation_prob``, ``last_mutation_operator``,
            ``last_exploration_strength``, ``rng_state``, and ``config``
            (resolved operator settings plus ``pop_size``/``n_vars``, used
            by :meth:`restore_state` for compatibility checks).

        Raises:
            RuntimeError: If called before initialize() (there is no
                population state to replay).
        """
        if self._population_x is None or self._population_f is None:
            raise RuntimeError("no state to snapshot; call initialize() first")
        return {
            "generation": int(self._generation),
            "population_x": self._population_x.copy(),
            "population_f": self._population_f.copy(),
            "ranks": None if self._ranks is None else self._ranks.copy(),
            "crowding": None if self._crowding is None else self._crowding.copy(),
            "last_mutation_prob": self._last_mutation_prob,
            "last_mutation_operator": self._last_mutation_operator,
            "last_exploration_strength": self._last_exploration_strength,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "config": {
                "pop_size": int(self._pop_size),
                "n_vars": int(self._problem.n_vars),
                "mutation_prob": float(self._mutation_prob),
                "mutation_operator": str(self._operators.mutation_operator),
                "eta_m": float(self._operators.eta_m),
                "gaussian_sigma": float(self._operators.gaussian_sigma),
                "eta_c": float(self._operators.eta_c),
                "crossover_prob": float(self._operators.crossover_prob),
                "crossover_operator": str(self._operators.crossover_operator),
            },
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        """Restore a state previously captured with :meth:`snapshot_state`.

        Every component is deep-copied back, so later mutation of the
        passed snapshot cannot corrupt this instance, and the snapshot
        itself is never mutated — one snapshot supports any number of
        restores (counterfactual branch evaluation, Phase 1.75). The
        instance is fully usable afterwards (``step()``,
        ``current_action()``, ...); re-executing the same ``step()`` call
        as after the original snapshot reproduces the next generation
        bit-identically.

        Args:
            snapshot: Dict produced by :meth:`snapshot_state` (possibly
                pickled and reloaded).

        Raises:
            ValueError: If the snapshot is incompatible with this instance:
                missing ``config`` block, different ``pop_size``/``n_vars``,
                different resolved operator configuration, malformed
                population arrays, or a bit-generator state of a different
                generator kind.
        """
        config = snapshot.get("config")
        if not isinstance(config, dict):
            raise ValueError("snapshot is missing its 'config' block")
        if int(config["pop_size"]) != self._pop_size:
            raise ValueError(
                f"snapshot pop_size {config['pop_size']} does not match "
                f"this instance's {self._pop_size}"
            )
        if int(config["n_vars"]) != self._problem.n_vars:
            raise ValueError(
                f"snapshot n_vars {config['n_vars']} does not match "
                f"this problem's {self._problem.n_vars}"
            )
        resolved: dict[str, Any] = {
            "mutation_prob": float(self._mutation_prob),
            "mutation_operator": str(self._operators.mutation_operator),
            "eta_m": float(self._operators.eta_m),
            "gaussian_sigma": float(self._operators.gaussian_sigma),
            "eta_c": float(self._operators.eta_c),
            "crossover_prob": float(self._operators.crossover_prob),
            "crossover_operator": str(self._operators.crossover_operator),
        }
        mismatched = [k for k, v in resolved.items() if config.get(k) != v]
        if mismatched:
            raise ValueError(
                f"snapshot operator config does not match this instance; "
                f"mismatched keys: {mismatched}"
            )
        population_x = snapshot.get("population_x")
        population_f = snapshot.get("population_f")
        if population_x is None or population_f is None:
            raise ValueError("snapshot carries no population (taken before initialize())")
        x = np.array(population_x, dtype=np.float64)
        f = np.array(population_f, dtype=np.float64)
        if x.shape != (self._pop_size, self._problem.n_vars):
            raise ValueError(
                f"snapshot population_x shape {x.shape} does not match "
                f"({self._pop_size}, {self._problem.n_vars})"
            )
        if f.shape != (self._pop_size, self._problem.n_objs):
            raise ValueError(
                f"snapshot population_f shape {f.shape} does not match "
                f"({self._pop_size}, {self._problem.n_objs})"
            )
        ranks = snapshot.get("ranks")
        crowding = snapshot.get("crowding")
        if ranks is None or crowding is None:
            raise ValueError("snapshot is missing ranks/crowding state")
        rng_state = copy.deepcopy(snapshot["rng_state"])
        if rng_state.get("bit_generator") != type(self._rng.bit_generator).__name__:
            raise ValueError(
                f"snapshot RNG kind {rng_state.get('bit_generator')!r} does not match "
                f"this instance's {type(self._rng.bit_generator).__name__!r}"
            )
        self._population_x = x
        self._population_f = f
        self._ranks = np.array(ranks, dtype=np.int64)
        self._crowding = np.array(crowding, dtype=np.float64)
        self._generation = int(snapshot["generation"])
        last_pm = snapshot.get("last_mutation_prob")
        self._last_mutation_prob = None if last_pm is None else float(last_pm)
        self._last_mutation_operator = snapshot.get("last_mutation_operator")
        last_expl = snapshot.get("last_exploration_strength")
        self._last_exploration_strength = None if last_expl is None else float(last_expl)
        self._rng.bit_generator.state = rng_state

    def _default_exploration_strength(self, mutation_operator: str) -> float:
        """Config-level exploration strength of an operator (eta_m or sigma)."""
        if mutation_operator == "gaussian":
            return float(self._operators.gaussian_sigma)
        return float(self._operators.eta_m)

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

    def _make_offspring(
        self,
        mating_indices: np.ndarray,
        mutation_prob: float,
        mutation_operator: str,
        exploration_strength: float,
    ) -> np.ndarray:
        """Generate mu offspring from consecutive parent pairs in the mating pool.

        Args:
            mating_indices: Mating pool of mu parent indices.
            mutation_prob: Per-variable mutation probability applied to every
                offspring in this generation.
            mutation_operator: Effective mutation operator of this
                generation (``"polynomial"`` or ``"gaussian"``).
            exploration_strength: Effective exploration strength of this
                generation (``eta_m`` for polynomial, ``sigma`` for
                Gaussian).
        """
        assert self._population_x is not None
        parents = self._population_x[mating_indices]
        offspring = np.empty_like(parents)
        for pair in range(0, self._pop_size, 2):
            p1, p2 = parents[pair], parents[pair + 1]
            if self._rng.random() <= self._operators.crossover_prob:
                c1, c2 = self._sbx(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring[pair] = self._mutate(c1, mutation_prob, mutation_operator, exploration_strength)
            offspring[pair + 1] = self._mutate(
                c2, mutation_prob, mutation_operator, exploration_strength
            )
        return offspring

    def _mutate(
        self,
        x: np.ndarray,
        mutation_prob: float,
        mutation_operator: str,
        exploration_strength: float,
    ) -> np.ndarray:
        """Apply the effective mutation operator of this generation to one offspring."""
        if mutation_operator == "gaussian":
            return self._gaussian_mutation(x, mutation_prob, exploration_strength)
        return self._polynomial_mutation(x, mutation_prob, exploration_strength)

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

    def _polynomial_mutation(
        self, x: np.ndarray, mutation_prob: float, eta_m: float
    ) -> np.ndarray:
        """Polynomial mutation (Deb & Goyal 1996; Deb et al. 2002).

        Each variable is perturbed with probability ``mutation_prob``
        (default ``1/n_vars``) by delta_q * (hi - lo), where delta_q follows
        the polynomial distribution with index ``eta_m`` (the exploration
        strength of this operator). Mutated values are clipped to the
        bounds.
        """
        y = x.copy()
        mut_pow = 1.0 / (eta_m + 1.0)
        for k in range(self._problem.n_vars):
            if self._rng.random() > mutation_prob:
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

    def _gaussian_mutation(
        self, x: np.ndarray, mutation_prob: float, sigma: float
    ) -> np.ndarray:
        """Gaussian mutation with range-scaled perturbations.

        Each variable is perturbed with probability ``mutation_prob``
        (default ``1/n_vars``) by ``N(0, 1) * sigma * (hi - lo)`` and
        clipped to the bounds. ``sigma`` is the exploration strength of
        this operator: larger values produce larger perturbations. All
        draws go through the instance generator, in the same mask-then-
        perturbation order as polynomial mutation.
        """
        y = x.copy()
        for k in range(self._problem.n_vars):
            if self._rng.random() > mutation_prob:
                continue
            lo, hi = float(self._lb[k]), float(self._ub[k])
            if hi - lo <= 0.0:
                continue  # fixed variable
            perturbation = float(self._rng.normal()) * sigma * (hi - lo)
            y[k] = min(max(float(y[k]) + perturbation, lo), hi)
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
