"""Tests for :mod:`controller.open_loop_controller` (Phase 1.5 baseline).

Contract coverage: statelessness (the key open-loop property), binning
determinism, hand-verified bin statistics, empty-bin fallback, global /
per-problem dispatch, ``n_vars`` handling, and save/load roundtrip.
"""

from __future__ import annotations

import copy
import json
import logging

import pytest

from controller.open_loop_controller import OpenLoopScheduleController


def make_transition(
    generation: int,
    pm: float,
    operator: str = "polynomial",
    exploration: float = 0.5,
) -> dict:
    """Build a minimal trajectory transition for the tests."""
    return {
        "generation": generation,
        "state": {},
        "action": {
            "mutation_operator": operator,
            "mutation_probability": pm,
            "exploration_strength": exploration,
        },
        "reward": {},
    }


def make_record(problem: str, n_vars: int, transitions: list) -> dict:
    """Build a minimal fit record for the tests."""
    return {
        "problem": problem,
        "n_vars": n_vars,
        "transitions": transitions,
    }


def ramp_records() -> list:
    """Two-problem corpus whose multipliers are exactly ``g + 1``.

    ``p_a`` has ``n_vars=10`` and ``pm=(g+1)/10``; ``p_b`` has
    ``n_vars=20`` and ``pm=(g+1)/20``; generations ``0..10`` with
    ``max_generation=10`` map one-per-bin into bins ``0..9`` (gen 10
    joins bin 9), so bin ``b < 9`` holds multiplier ``b + 1`` and bin 9
    holds ``{10, 11}`` (median 10.5).
    """
    return [
        make_record(
            "p_a", 10, [make_transition(g, (g + 1) / 10) for g in range(11)]
        ),
        make_record(
            "p_b", 20, [make_transition(g, (g + 1) / 20) for g in range(11)]
        ),
    ]


def test_stateless_same_generation_same_action(tmp_path):
    """Key property: identical arguments give identical actions regardless
    of the interaction history (the controller is open-loop / stateless)."""
    records = [
        make_record(
            "toy", 10, [make_transition(g, 0.01 + 0.002 * g) for g in range(101)]
        )
    ]
    controller = OpenLoopScheduleController().fit(records)

    # History A: a plain ascending sweep on a single problem.
    actions_a = [controller.predict_action(g, 100, "toy") for g in range(101)]

    # History B: a completely different interaction — reversed order,
    # queries at a different generation scale, and an unseen problem
    # interleaved between the recorded queries.
    for g in (7, 63, 30):
        controller.predict_action(g, 300, "toy")
    controller.predict_action(4, 40, "unseen_problem")
    actions_b = [
        controller.predict_action(g, 100, "toy")
        for g in range(100, -1, -1)
    ]
    actions_b.reverse()
    assert actions_a == actions_b  # exact float equality

    # A fresh, identically fitted controller agrees exactly...
    fresh = OpenLoopScheduleController().fit(copy.deepcopy(records))
    assert [fresh.predict_action(g, 100, "toy") for g in range(101)] == actions_a
    # ...and prediction did not mutate the fitted state (byte-identical
    # serialization compared to a never-used twin).
    twin = OpenLoopScheduleController().fit(copy.deepcopy(records))
    path_used = tmp_path / "used.json"
    path_twin = tmp_path / "twin.json"
    controller.save(path_used)
    twin.save(path_twin)
    assert path_used.read_text(encoding="utf-8") == path_twin.read_text(
        encoding="utf-8"
    )


def test_fit_deterministic_and_record_order_invariant(tmp_path):
    """Fitting is deterministic and independent of the record order."""
    records = ramp_records()
    paths = []
    for order in (records, copy.deepcopy(records), list(reversed(records))):
        controller = OpenLoopScheduleController(per_problem=True).fit(order)
        path = tmp_path / f"schedule_{len(paths)}.json"
        controller.save(path)
        paths.append(path)
    texts = [path.read_text(encoding="utf-8") for path in paths]
    assert texts[0] == texts[1] == texts[2]


def test_predict_pm_equals_bin_median_multiplier_over_nvars():
    """pm == bin median multiplier / n_vars, verified on a hand-computable
    corpus (multiplier of generation ``g`` is exactly ``g + 1``)."""
    records = [
        make_record(
            "toy", 10, [make_transition(g, (g + 1) / 10) for g in range(11)]
        )
    ]
    controller = OpenLoopScheduleController().fit(records)
    for g in range(9):  # one transition per bin, median = multiplier = g + 1
        assert controller.predict_action(g, 10)[
            "mutation_probability"
        ] == pytest.approx((g + 1) / 10)
    for g in (9, 10):  # bin 9 pools multipliers {10, 11} -> median 10.5
        assert controller.predict_action(g, 10)[
            "mutation_probability"
        ] == pytest.approx(10.5 / 10)
    # Exact contract check against the stored bin statistics.
    stats = controller._global_schedule[9]
    assert (
        controller.predict_action(9, 10)["mutation_probability"]
        == stats["mutation_multiplier"] / 10
    )


def test_majority_operator_and_exploration_grouping():
    """The bin's operator is the majority vote, and exploration is the
    median over that operator's group only (not over all transitions)."""
    transitions = [
        # bin 0 (gens 0-2, normalized by the record's own max gen 30):
        make_transition(0, 0.10, "polynomial", 0.1),
        make_transition(1, 0.20, "polynomial", 0.3),
        make_transition(2, 0.30, "gaussian", 0.9),
        # bin 5 (gens 15-17):
        make_transition(15, 0.10, "gaussian", 0.2),
        make_transition(16, 0.20, "polynomial", 0.7),
        make_transition(17, 0.30, "gaussian", 0.6),
        # filler establishing the record's max generation (30 -> bin 9):
        make_transition(30, 0.05, "polynomial", 0.5),
    ]
    controller = OpenLoopScheduleController().fit(
        [make_record("p", 10, transitions)]
    )
    action0 = controller.predict_action(1, 30)  # bin 0
    assert action0["mutation_operator"] == "polynomial"
    assert action0["exploration_strength"] == pytest.approx(0.2)
    assert action0["mutation_probability"] == pytest.approx(0.2)
    action5 = controller.predict_action(16, 30)  # bin 5
    assert action5["mutation_operator"] == "gaussian"
    assert action5["exploration_strength"] == pytest.approx(0.4)
    assert action5["mutation_probability"] == pytest.approx(0.2)


def test_majority_operator_tie_breaks_alphabetically():
    """A 1:1 operator tie in a bin resolves to the alphabetically first
    name, deterministically."""
    transitions = [
        make_transition(0, 0.1, "polynomial", 0.3),
        make_transition(1, 0.1, "gaussian", 0.7),
        # filler establishing max generation 19 (both gens land in bin 0):
        make_transition(19, 0.05, "polynomial", 0.5),
    ]
    controller = OpenLoopScheduleController().fit(
        [make_record("p", 10, transitions)]
    )
    action = controller.predict_action(0, 19)  # gens 0-1 both bin 0
    assert action["mutation_operator"] == "gaussian"
    assert action["exploration_strength"] == pytest.approx(0.7)


def test_empty_bin_filled_from_preceding_nonempty():
    """Interior empty bins inherit the nearest *preceding* non-empty bin."""
    transitions = [make_transition(g, (g + 1) / 10) for g in range(4)]
    transitions += [
        make_transition(g, 0.9, "gaussian", 0.9) for g in (8, 9)
    ]
    controller = OpenLoopScheduleController().fit(
        [make_record("p", 10, transitions)]
    )
    anchor = controller.predict_action(3, 10)  # bin 3 (last non-empty)
    for g in (4, 5, 6, 7):  # bins 4-7 empty -> bin 3 stats
        assert controller.predict_action(g, 10) == anchor
    assert controller.predict_action(8, 10) != anchor  # own data


def test_leading_empty_bins_use_nearest_following():
    """A leading gap (no preceding non-empty bin) inherits the nearest
    *following* non-empty bin."""
    transitions = [make_transition(g, 0.05 * (g - 4)) for g in range(5, 10)]
    controller = OpenLoopScheduleController().fit(
        [make_record("p", 10, transitions)]
    )
    anchor = controller.predict_action(5, 9)  # first non-empty bin
    for g in range(5):
        assert controller.predict_action(g, 9) == anchor


def test_global_variant_uses_queried_problem_nvars():
    """The global variant scales the pooled multiplier by the *queried*
    problem's n_vars (pm_b == pm_a / 2 for n_vars 10 vs 20)."""
    controller = OpenLoopScheduleController().fit(ramp_records())
    assert controller.name == "open_loop_global"
    pm_a = controller.predict_action(5, 10, "p_a")["mutation_probability"]
    pm_b = controller.predict_action(5, 10, "p_b")["mutation_probability"]
    assert pm_a == pytest.approx(6 / 10)  # bin 5 pooled multipliers {6, 6}
    assert pm_b == pytest.approx(6 / 20)
    assert pm_b == pytest.approx(pm_a / 2)


def test_per_problem_name_and_different_schedules():
    """The per-problem variant dispatches per problem and reports the
    documented arm name."""
    records = [
        make_record(
            "p_a", 10, [make_transition(g, (g + 1) / 10) for g in range(11)]
        ),
        make_record(
            "p_b", 10, [make_transition(g, 0.05) for g in range(11)]
        ),
    ]
    controller = OpenLoopScheduleController(per_problem=True).fit(records)
    assert controller.name == "open_loop_per_problem"
    assert controller.predict_action(8, 10, "p_a")[
        "mutation_probability"
    ] == pytest.approx(9 / 10)
    assert controller.predict_action(8, 10, "p_b")[
        "mutation_probability"
    ] == pytest.approx(0.05)
    assert controller.predict_action(8, 10, "p_a") != controller.predict_action(
        8, 10, "p_b"
    )


def test_per_problem_unknown_problem_falls_back_to_global_with_warning(
    caplog,
):
    """Unknown problem in per-problem mode falls back to the global
    schedule and logs a WARNING."""
    records = [
        make_record(
            "p_a", 10, [make_transition(g, (g + 1) / 10) for g in range(11)]
        )
    ]
    controller = OpenLoopScheduleController(per_problem=True).fit(records)
    with caplog.at_level(logging.WARNING, logger="controller.open_loop_controller"):
        unknown = controller.predict_action(5, 10, "ghost")
    assert "ghost" in caplog.text
    assert "global" in caplog.text
    # Default n_vars is the median over fitted problems (10 here), so the
    # fallback action is identical to p_a's.
    assert unknown == controller.predict_action(5, 10, "p_a")


def test_save_load_roundtrip(tmp_path):
    """save/load roundtrip restores an exactly equivalent controller."""
    controller = OpenLoopScheduleController(per_problem=True).fit(ramp_records())
    path = tmp_path / "open_loop.json"
    controller.save(path)
    restored = OpenLoopScheduleController.load(path)
    assert restored.name == controller.name
    assert restored.n_bins == controller.n_bins
    for problem in ("p_a", "p_b"):
        for g in range(0, 11, 3):
            assert restored.predict_action(
                g, 10, problem
            ) == controller.predict_action(g, 10, problem)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "open_loop_schedule_controller"
    assert payload["version"] == 1


def test_fit_returns_self_and_input_not_mutated():
    """fit returns the instance itself and leaves the input untouched."""
    records = ramp_records()
    snapshot = copy.deepcopy(records)
    controller = OpenLoopScheduleController()
    assert controller.fit(records) is controller
    assert records == snapshot


def test_predict_before_fit_raises():
    """Prediction before fit fails loudly."""
    with pytest.raises(RuntimeError):
        OpenLoopScheduleController().predict_action(1, 10, "p")


def test_fit_rejects_bad_input():
    """Empty records, transition-free records, malformed transitions, and
    non-positive n_bins/n_vars are rejected."""
    with pytest.raises(ValueError):
        OpenLoopScheduleController().fit([])
    with pytest.raises(ValueError):
        OpenLoopScheduleController().fit([make_record("p", 10, [])])
    broken = make_transition(0, 0.1)
    del broken["action"]["exploration_strength"]
    with pytest.raises(ValueError):
        OpenLoopScheduleController().fit([make_record("p", 10, [broken])])
    with pytest.raises(ValueError):
        OpenLoopScheduleController(n_bins=0)
    with pytest.raises(ValueError):
        OpenLoopScheduleController().fit([make_record("p", 0, [make_transition(0, 0.1)])])
