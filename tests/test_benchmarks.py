from __future__ import annotations

import numpy as np
import pytest

from benchmarks import Problem, ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, get_problem

ALL_PROBLEMS = [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6]
ALL_NAMES = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"]


def _midpoint_x(problem: Problem) -> np.ndarray:
    return (problem.lower_bounds + problem.upper_bounds) / 2.0


class TestProblemInterface:
    @pytest.mark.parametrize("cls", ALL_PROBLEMS)
    def test_evaluate_shape_and_dtype(self, cls: type[Problem]) -> None:
        problem = cls()
        out = problem.evaluate(_midpoint_x(problem))
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert out.dtype == np.float64
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("cls", ALL_PROBLEMS)
    def test_n_objs_is_two(self, cls: type[Problem]) -> None:
        assert cls().n_objs == 2

    @pytest.mark.parametrize("cls", ALL_PROBLEMS)
    def test_bounds_shapes_match_n_vars(self, cls: type[Problem]) -> None:
        problem = cls()
        lb, ub = problem.lower_bounds, problem.upper_bounds
        assert lb.shape == (problem.n_vars,)
        assert ub.shape == (problem.n_vars,)
        assert np.all(lb < ub)

    @pytest.mark.parametrize("cls", ALL_PROBLEMS)
    def test_evaluate_rejects_wrong_shape(self, cls: type[Problem]) -> None:
        problem = cls()
        with pytest.raises(ValueError):
            problem.evaluate(np.zeros(problem.n_vars + 1))
        with pytest.raises(ValueError):
            problem.evaluate(np.zeros((problem.n_vars, 1)))

    @pytest.mark.parametrize("cls", ALL_PROBLEMS)
    def test_reference_front_shape_and_dtype(self, cls: type[Problem]) -> None:
        front = cls().reference_front(200)
        assert front.shape == (200, 2)
        assert front.dtype == np.float64
        assert np.all(np.isfinite(front))

    @pytest.mark.parametrize("cls", ALL_PROBLEMS)
    def test_reference_front_exact_n_points(self, cls: type[Problem]) -> None:
        for n in (1, 7, 200, 333):
            assert cls().reference_front(n).shape == (n, 2)

    @pytest.mark.parametrize("cls", ALL_PROBLEMS)
    def test_reference_front_mutually_nondominated(self, cls: type[Problem]) -> None:
        front = cls().reference_front(200)
        for i in range(front.shape[0]):
            for j in range(front.shape[0]):
                if i == j:
                    continue
                strictly_dominates = (
                    np.all(front[i] <= front[j] + 1e-6)
                    and np.any(front[i] < front[j] - 1e-6)
                )
                assert not strictly_dominates, (
                    f"{cls.__name__}: point {i} dominates point {j}"
                )


class TestKnownValues:
    @pytest.mark.parametrize("a", [0.0, 0.25, 0.5, 1.0])
    def test_zdt1_optimal_g(self, a: float) -> None:
        # x[1:] = 0 -> g = 1, so f2 = 1 - sqrt(f1) with f1 = x1 = a.
        x = np.zeros(30)
        x[0] = a
        f = ZDT1().evaluate(x)
        assert f[0] == pytest.approx(a)
        assert f[1] == pytest.approx(1.0 - np.sqrt(a))

    def test_zdt2_optimal_g(self) -> None:
        x = np.zeros(30)
        x[0] = 0.5
        f = ZDT2().evaluate(x)
        assert f[1] == pytest.approx(1.0 - 0.25)

    def test_zdt3_optimal_g_no_sine_at_zero(self) -> None:
        x = np.zeros(30)
        f = ZDT3().evaluate(x)
        assert f[0] == pytest.approx(0.0)
        assert f[1] == pytest.approx(1.0)

    def test_zdt4_optimal_g(self) -> None:
        # xi = 0 (i > 1) -> each term xi^2 - 10 cos(4 pi xi) = -10,
        # so g = 1 + 10*9 - 90 = 1.
        x = np.zeros(10)
        x[0] = 0.64
        f = ZDT4().evaluate(x)
        assert f[1] == pytest.approx(1.0 - np.sqrt(0.64))

    def test_zdt4_worst_g_at_bounds(self) -> None:
        # xi = +-5 -> g = 1 + 90 + sum(25 - 10 cos(20 pi)) = 1 + 90 + 9*15.
        x = np.full(10, 5.0)
        x[0] = 0.0
        f = ZDT4().evaluate(x)
        g_expected = 1.0 + 10.0 * 9 + 9 * (25.0 - 10.0)
        assert f[1] == pytest.approx(g_expected * 1.0)

    def test_zdt6_optimal_f1(self) -> None:
        # Global minimum of f1 over x1 in [0, 1] is approx 0.280775,
        # attained at x1 approx 0.0815 (argmax of exp(-4*x1)*sin(6*pi*x1)^6;
        # the exp term pulls the optimum below the sin lobe center 1/12).
        # With x[1:] = 0, g = 1 and f2 = 1 - f1^2.
        x = np.zeros(10)
        x[0] = 0.081458
        f = ZDT6().evaluate(x)
        assert f[0] == pytest.approx(0.280775, abs=1e-4)
        assert f[1] == pytest.approx(1.0 - f[0] ** 2, abs=1e-6)

    def test_zdt6_front_starts_near_optimal_f1(self) -> None:
        front = ZDT6().reference_front(200)
        assert front[:, 0].min() == pytest.approx(0.280775, abs=1e-3)
        assert front[:, 0].max() == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize(
        "cls", [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6], ids=ALL_NAMES
    )
    def test_reference_front_lies_on_optimal_curve(self, cls: type[Problem]) -> None:
        # With g = 1 the returned front must satisfy the analytic f2(f1).
        problem = cls()
        front = problem.reference_front(200)
        f1, f2 = front[:, 0], front[:, 1]
        if isinstance(problem, (ZDT1, ZDT4)):
            expected = 1.0 - np.sqrt(f1)
        elif isinstance(problem, ZDT2):
            expected = 1.0 - f1 ** 2
        elif isinstance(problem, ZDT6):
            expected = 1.0 - f1 ** 2
        else:  # ZDT3: analytic h with sine term
            expected = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
        # Interpolated resampling deviates slightly between dense samples.
        assert np.max(np.abs(f2 - expected)) < 2e-3


class TestGetProblem:
    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_roundtrip(self, name: str) -> None:
        problem = get_problem(name)
        assert problem.name == name

    @pytest.mark.parametrize("name", ["ZDT1", "Zdt4", "ZDT6"])
    def test_case_insensitive(self, name: str) -> None:
        assert get_problem(name).name == name.lower()

    def test_zdt5_unsupported(self) -> None:
        with pytest.raises(ValueError):
            get_problem("zdt5")

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            get_problem("unknown")
