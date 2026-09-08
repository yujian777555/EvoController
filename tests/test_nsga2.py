"""Tests for the NSGA-II baseline (algorithms.nsga2).

The tests use a local bi-objective stub problem implementing the
``benchmarks.base.Problem`` contract so this module can be validated
independently of the benchmarks package.
"""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.nsga2 import NSGAII, OperatorConfig

POP_SIZE = 20
N_VARS = 5
SEED = 42


class Sphere2DProblemStub:
    """Minimal 2-objective minimization problem matching the Problem contract.

    f1(x) = sum(x_i^2), f2(x) = sum((x_i - 1)^2) on [-2, 2]^5. The
    Pareto-optimal set is x_i = t for t in [0, 1], so a working NSGA-II must
    move its first front towards lower f1 + f2 values over generations.
    """

    @property
    def name(self) -> str:
        return "sphere2d_stub"

    @property
    def n_vars(self) -> int:
        return N_VARS

    @property
    def n_objs(self) -> int:
        return 2

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.full(N_VARS, -2.0)

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.full(N_VARS, 2.0)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.array([np.sum(x**2), np.sum((x - 1.0) ** 2)], dtype=np.float64)

    def reference_front(self, n_points: int = 200) -> np.ndarray:
        t = np.linspace(0.0, 1.0, n_points)
        return np.column_stack([N_VARS * t**2, N_VARS * (t - 1.0) ** 2])


def _make_algo(
    seed: int = SEED, pop_size: int = POP_SIZE, operators: OperatorConfig | None = None
) -> NSGAII:
    return NSGAII(
        problem=Sphere2DProblemStub(),
        pop_size=pop_size,
        operators=operators if operators is not None else OperatorConfig(),
        seed=seed,
    )


def test_determinism_same_seed_identical_fronts() -> None:
    """Two instances with the same seed must evolve bit-identical populations."""
    algo_a = _make_algo(seed=7)
    algo_b = _make_algo(seed=7)
    algo_a.initialize()
    algo_b.initialize()
    for _ in range(10):
        algo_a.step()
        algo_b.step()
    np.testing.assert_array_equal(algo_a.nondominated_front(), algo_b.nondominated_front())
    np.testing.assert_array_equal(algo_a.population_x, algo_b.population_x)
    np.testing.assert_array_equal(algo_a.population_f, algo_b.population_f)


def test_generation_counter_increments() -> None:
    """generation is 0 after initialize() and increases by 1 per step()."""
    algo = _make_algo()
    algo.initialize()
    assert algo.generation == 0
    for expected in range(1, 4):
        algo.step()
        assert algo.generation == expected


def test_population_size_stays_constant() -> None:
    """Elitist (mu + lambda) replacement must restore exactly mu individuals."""
    algo = _make_algo()
    algo.initialize()
    assert algo.population_x.shape == (POP_SIZE, N_VARS)
    assert algo.population_f.shape == (POP_SIZE, 2)
    for _ in range(5):
        algo.step()
        assert algo.population_x.shape == (POP_SIZE, N_VARS)
        assert algo.population_f.shape == (POP_SIZE, 2)


def test_nondominated_front_shape_and_sorted() -> None:
    """Front is (k, 2) with 1 <= k <= pop_size, sorted by ascending f1."""
    algo = _make_algo()
    algo.initialize()
    for _ in range(5):
        algo.step()
        front = algo.nondominated_front()
        assert front.ndim == 2
        assert front.shape[1] == 2
        assert 1 <= front.shape[0] <= POP_SIZE
        assert np.all(np.diff(front[:, 0]) >= 0.0)


def test_convergence_sanity() -> None:
    """After 50 generations the front's mean f1+f2 must be strictly lower."""
    algo = _make_algo(seed=123, pop_size=40)
    algo.initialize()
    initial_mean = float(np.mean(np.sum(algo.nondominated_front(), axis=1)))
    for _ in range(50):
        algo.step()
    final_mean = float(np.mean(np.sum(algo.nondominated_front(), axis=1)))
    assert final_mean < initial_mean


def test_bounds_respected_after_steps() -> None:
    """All decision variables stay inside [-2, 2] after variation operators."""
    algo = _make_algo()
    algo.initialize()
    for _ in range(20):
        algo.step()
    pop = algo.population_x
    assert np.all(pop >= -2.0)
    assert np.all(pop <= 2.0)


@pytest.mark.parametrize("pop_size", [1, 3, 21])
def test_odd_pop_size_raises_value_error(pop_size: int) -> None:
    with pytest.raises(ValueError):
        _make_algo(pop_size=pop_size)


def test_current_action_defaults_resolved() -> None:
    """Default config resolves mutation_probability to 1 / n_vars."""
    algo = _make_algo()
    action = algo.current_action()
    assert set(action.keys()) == {"mutation_operator", "mutation_probability"}
    assert action["mutation_operator"] == "polynomial"
    assert action["mutation_probability"] == pytest.approx(1.0 / N_VARS)


def test_current_action_explicit_mutation_prob() -> None:
    """An explicit mutation_prob is reported unchanged by current_action()."""
    algo = _make_algo(operators=OperatorConfig(mutation_prob=0.3))
    action = algo.current_action()
    assert action["mutation_operator"] == "polynomial"
    assert action["mutation_probability"] == pytest.approx(0.3)
