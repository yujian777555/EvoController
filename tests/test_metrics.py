from __future__ import annotations

"""Tests for metrics.indicators (hypervolume, igd, diversity_spread).

All expected values are computed by hand from the definitions; the
hypervolume two-point case is the union of the dominated rectangles,
``0.16 + 0.16 - 0.04 = 0.28``.
"""

import numpy as np
import pytest

from metrics import diversity_spread, hypervolume, igd


class TestHypervolume:
    def test_single_point_unit_square(self) -> None:
        front = np.array([[0.5, 0.5]])
        assert hypervolume(front, np.array([1.0, 1.0])) == pytest.approx(0.25)

    def test_two_points_hand_computed(self) -> None:
        front = np.array([[0.2, 0.8], [0.8, 0.2]])
        # Rectangles 0.8*0.2 each, overlap 0.2*0.2 -> 0.16 + 0.16 - 0.04.
        assert hypervolume(front, np.array([1.0, 1.0])) == pytest.approx(0.28)

    def test_ignores_dominated_points_and_duplicates(self) -> None:
        base = np.array([[0.2, 0.8], [0.8, 0.2]])
        augmented = np.array(
            [[0.2, 0.8], [0.8, 0.2], [0.5, 0.9], [0.2, 0.8]]
        )
        ref = np.array([1.0, 1.0])
        assert hypervolume(augmented, ref) == pytest.approx(
            hypervolume(base, ref)
        )

    def test_ignores_points_outside_reference(self) -> None:
        front = np.array([[0.2, 0.8], [0.8, 0.2], [2.0, 0.1], [0.1, 5.0]])
        assert hypervolume(front, np.array([1.0, 1.0])) == pytest.approx(0.28)

    def test_all_points_outside_reference_gives_zero(self) -> None:
        front = np.array([[2.0, 0.1], [0.1, 5.0]])
        assert hypervolume(front, np.array([1.0, 1.0])) == 0.0

    def test_empty_front_gives_zero(self) -> None:
        front = np.empty((0, 2))
        assert hypervolume(front, np.array([1.0, 1.0])) == 0.0

    def test_invalid_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            hypervolume(np.array([0.5, 0.5]), np.array([1.0, 1.0]))
        with pytest.raises(ValueError):
            hypervolume(np.zeros((3, 3)), np.array([1.0, 1.0]))
        with pytest.raises(ValueError):
            hypervolume(np.zeros((2, 2)), np.array([1.0, 1.0, 1.0]))


class TestIGD:
    def test_identical_sets_give_zero(self) -> None:
        front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        assert igd(front, front.copy()) == pytest.approx(0.0, abs=1e-12)

    def test_hand_computed_example(self) -> None:
        front = np.array([[0.0, 0.0], [1.0, 1.0]])
        reference = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        # Min distances: 0, sqrt(0.5), 0 -> mean sqrt(0.5) / 3.
        expected = np.sqrt(0.5) / 3.0
        assert igd(front, reference) == pytest.approx(expected)

    def test_empty_front_gives_inf(self) -> None:
        front = np.empty((0, 2))
        reference = np.array([[0.0, 0.0]])
        assert igd(front, reference) == float("inf")

    def test_empty_reference_raises(self) -> None:
        with pytest.raises(ValueError):
            igd(np.array([[0.0, 0.0]]), np.empty((0, 2)))

    def test_invalid_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            igd(np.array([0.0, 0.0]), np.array([[0.0, 0.0]]))
        with pytest.raises(ValueError):
            igd(np.array([[0.0, 0.0]]), np.zeros((2, 3)))


class TestDiversitySpread:
    def test_evenly_spaced_front_near_zero(self) -> None:
        f1 = np.linspace(0.0, 1.0, 51)
        front = np.column_stack([f1, 1.0 - f1])
        # Uniform spacing: Delta = 2 / 51 ~= 0.039.
        assert diversity_spread(front) == pytest.approx(2.0 / 51.0, rel=1e-6)
        assert diversity_spread(front) < 0.05

    def test_clustered_front_above_half(self) -> None:
        clustered = np.column_stack(
            [np.linspace(0.0, 0.008, 9), 1.0 - np.linspace(0.0, 0.008, 9)]
        )
        front = np.vstack([clustered, [[1.0, 0.0]]])
        assert diversity_spread(front) > 0.5

    def test_fewer_than_two_points_gives_zero(self) -> None:
        assert diversity_spread(np.empty((0, 2))) == 0.0
        assert diversity_spread(np.array([[0.5, 0.5]])) == 0.0
        assert diversity_spread(np.array([[0.0, 1.0], [1.0, 0.0]])) == 0.0

    def test_invalid_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            diversity_spread(np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            diversity_spread(np.zeros((4, 3)))
