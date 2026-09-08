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
    """Default config resolves to polynomial, pm = 1/n_vars, strength = eta_m."""
    algo = _make_algo()
    action = algo.current_action()
    assert set(action.keys()) == {
        "mutation_operator",
        "mutation_probability",
        "exploration_strength",
    }
    assert action["mutation_operator"] == "polynomial"
    assert action["mutation_probability"] == pytest.approx(1.0 / N_VARS)
    assert action["exploration_strength"] == pytest.approx(20.0)


def test_current_action_gaussian_defaults_resolved() -> None:
    """A gaussian config reports its sigma as the exploration strength."""
    algo = _make_algo(
        operators=OperatorConfig(mutation_operator="gaussian", gaussian_sigma=0.2)
    )
    action = algo.current_action()
    assert set(action.keys()) == {
        "mutation_operator",
        "mutation_probability",
        "exploration_strength",
    }
    assert action["mutation_operator"] == "gaussian"
    assert action["mutation_probability"] == pytest.approx(1.0 / N_VARS)
    assert action["exploration_strength"] == pytest.approx(0.2)


def test_current_action_explicit_mutation_prob() -> None:
    """An explicit mutation_prob is reported unchanged by current_action()."""
    algo = _make_algo(operators=OperatorConfig(mutation_prob=0.3))
    action = algo.current_action()
    assert action["mutation_operator"] == "polynomial"
    assert action["mutation_probability"] == pytest.approx(0.3)
    assert action["exploration_strength"] == pytest.approx(20.0)


def test_step_override_used_exactly() -> None:
    """An injected mutation_prob is reported verbatim after the step."""
    algo = _make_algo()
    algo.initialize()
    # Before any step, current_action() returns the resolved default.
    assert algo.current_action()["mutation_probability"] == pytest.approx(1.0 / N_VARS)
    algo.step(mutation_prob=0.5)
    action = algo.current_action()
    assert action["mutation_operator"] == "polynomial"
    assert action["mutation_probability"] == pytest.approx(0.5)
    assert action["exploration_strength"] == pytest.approx(20.0)


def test_step_without_override_reverts_to_default() -> None:
    """The override applies to one generation only; the next step reverts."""
    algo = _make_algo()
    algo.initialize()
    algo.step(mutation_prob=0.5)
    algo.step()
    assert algo.current_action()["mutation_probability"] == pytest.approx(1.0 / N_VARS)


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_step_override_invalid_raises_value_error(bad: float) -> None:
    """Override probabilities outside [0, 1] are rejected before evolving."""
    algo = _make_algo()
    algo.initialize()
    with pytest.raises(ValueError):
        algo.step(mutation_prob=bad)
    # The rejected step must not have advanced the run.
    assert algo.generation == 0
    assert algo.current_action()["mutation_probability"] == pytest.approx(1.0 / N_VARS)


def test_determinism_same_seed_same_override_sequence() -> None:
    """Same seed + same override sequence -> bit-identical populations."""
    overrides = [0.1, 0.3, 0.05, 0.2]
    fronts, populations = [], []
    for _ in range(2):
        algo = _make_algo(seed=11)
        algo.initialize()
        for pm in overrides:
            algo.step(mutation_prob=pm)
        fronts.append(algo.nondominated_front())
        populations.append(algo.population_x)
    np.testing.assert_array_equal(fronts[0], fronts[1])
    np.testing.assert_array_equal(populations[0], populations[1])


def test_no_override_matches_default_step() -> None:
    """step(mutation_prob=None) is identical to the pre-change step() path."""
    algo_a = _make_algo(seed=13)
    algo_b = _make_algo(seed=13)
    algo_a.initialize()
    algo_b.initialize()
    for _ in range(5):
        algo_a.step()
        algo_b.step(mutation_prob=None)
    np.testing.assert_array_equal(algo_a.nondominated_front(), algo_b.nondominated_front())
    np.testing.assert_array_equal(algo_a.population_x, algo_b.population_x)


# --- Phase 1.5: expanded action space (operator + exploration strength) ---


def test_gaussian_mutation_reproducible() -> None:
    """Two gaussian-configured instances with one seed evolve identically."""
    operators = OperatorConfig(mutation_operator="gaussian", gaussian_sigma=0.2)
    fronts, populations = [], []
    for _ in range(2):
        algo = _make_algo(seed=17, operators=operators)
        algo.initialize()
        for _ in range(8):
            algo.step()
        fronts.append(algo.nondominated_front())
        populations.append(algo.population_x)
    np.testing.assert_array_equal(fronts[0], fronts[1])
    np.testing.assert_array_equal(populations[0], populations[1])


def test_gaussian_mutation_respects_bounds() -> None:
    """Even with a huge sigma every variable stays inside the bounds."""
    operators = OperatorConfig(mutation_operator="gaussian", gaussian_sigma=5.0)
    algo = _make_algo(operators=operators)
    algo.initialize()
    for _ in range(10):
        algo.step()
    pop = algo.population_x
    assert np.all(pop >= -2.0)
    assert np.all(pop <= 2.0)


def test_step_override_operator_only() -> None:
    """An operator override alone uses the config sigma of that operator."""
    algo = _make_algo(operators=OperatorConfig(gaussian_sigma=0.25))
    algo.initialize()
    algo.step(mutation_operator="gaussian")
    action = algo.current_action()
    assert action["mutation_operator"] == "gaussian"
    assert action["mutation_probability"] == pytest.approx(1.0 / N_VARS)
    assert action["exploration_strength"] == pytest.approx(0.25)


def test_step_override_exploration_only() -> None:
    """An exploration override applies to the effective operator (eta_m here)."""
    algo = _make_algo()
    algo.initialize()
    algo.step(exploration_strength=5.0)
    action = algo.current_action()
    assert action["mutation_operator"] == "polynomial"
    assert action["mutation_probability"] == pytest.approx(1.0 / N_VARS)
    assert action["exploration_strength"] == pytest.approx(5.0)


def test_step_override_all_three() -> None:
    """pm + operator + exploration overrides are all reported verbatim."""
    algo = _make_algo()
    algo.initialize()
    algo.step(mutation_prob=0.7, mutation_operator="gaussian", exploration_strength=0.3)
    action = algo.current_action()
    assert action["mutation_operator"] == "gaussian"
    assert action["mutation_probability"] == pytest.approx(0.7)
    assert action["exploration_strength"] == pytest.approx(0.3)


def test_step_full_overrides_revert_to_defaults() -> None:
    """All three overrides apply for one generation only, then revert."""
    algo = _make_algo()
    algo.initialize()
    algo.step(mutation_prob=0.7, mutation_operator="gaussian", exploration_strength=0.3)
    algo.step()
    action = algo.current_action()
    assert action["mutation_operator"] == "polynomial"
    assert action["mutation_probability"] == pytest.approx(1.0 / N_VARS)
    assert action["exploration_strength"] == pytest.approx(20.0)


def test_exploration_strength_maps_to_sigma() -> None:
    """exploration_strength on a gaussian step equals a config sigma change."""
    algo_a = _make_algo(
        seed=19, operators=OperatorConfig(mutation_operator="gaussian", gaussian_sigma=0.4)
    )
    algo_b = _make_algo(
        seed=19, operators=OperatorConfig(mutation_operator="gaussian", gaussian_sigma=0.1)
    )
    algo_a.initialize()
    algo_b.initialize()
    for _ in range(5):
        algo_a.step()
        algo_b.step(exploration_strength=0.4)
    np.testing.assert_array_equal(algo_a.population_x, algo_b.population_x)
    assert algo_b.current_action()["exploration_strength"] == pytest.approx(0.4)


def test_exploration_strength_maps_to_eta_m() -> None:
    """exploration_strength on a polynomial step equals a config eta_m change."""
    algo_a = _make_algo(seed=23, operators=OperatorConfig(eta_m=5.0))
    algo_b = _make_algo(seed=23)  # eta_m default 20.0
    algo_a.initialize()
    algo_b.initialize()
    for _ in range(5):
        algo_a.step()
        algo_b.step(exploration_strength=5.0)
    np.testing.assert_array_equal(algo_a.population_x, algo_b.population_x)
    assert algo_b.current_action()["exploration_strength"] == pytest.approx(5.0)


def test_invalid_config_mutation_operator_raises() -> None:
    """OperatorConfig with an unknown mutation operator is rejected."""
    with pytest.raises(ValueError):
        _make_algo(operators=OperatorConfig(mutation_operator="uniform"))


def test_invalid_config_gaussian_sigma_raises() -> None:
    """A non-positive gaussian_sigma is rejected at construction."""
    with pytest.raises(ValueError):
        _make_algo(operators=OperatorConfig(gaussian_sigma=0.0))


def test_step_override_invalid_operator_raises() -> None:
    """A step-level unknown operator is rejected before evolving."""
    algo = _make_algo()
    algo.initialize()
    with pytest.raises(ValueError):
        algo.step(mutation_operator="uniform")
    assert algo.generation == 0
    assert algo.current_action()["mutation_operator"] == "polynomial"


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
def test_step_override_invalid_exploration_raises(bad: float) -> None:
    """Non-positive exploration strengths are rejected before evolving."""
    algo = _make_algo()
    algo.initialize()
    with pytest.raises(ValueError):
        algo.step(exploration_strength=bad)
    assert algo.generation == 0
    assert algo.current_action()["exploration_strength"] == pytest.approx(20.0)


def test_determinism_same_seed_same_full_override_sequence() -> None:
    """Same seed + same (pm, operator, strength) sequence -> identical runs."""
    overrides: list[tuple[float | None, str | None, float | None]] = [
        (0.1, "polynomial", 5.0),
        (0.3, "gaussian", 0.3),
        (None, None, None),
        (0.2, "gaussian", None),
        (None, "polynomial", None),
    ]
    fronts, populations = [], []
    for _ in range(2):
        algo = _make_algo(seed=29)
        algo.initialize()
        for pm, operator, strength in overrides:
            algo.step(
                mutation_prob=pm,
                mutation_operator=operator,
                exploration_strength=strength,
            )
        fronts.append(algo.nondominated_front())
        populations.append(algo.population_x)
    np.testing.assert_array_equal(fronts[0], fronts[1])
    np.testing.assert_array_equal(populations[0], populations[1])
