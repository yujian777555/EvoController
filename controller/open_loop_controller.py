"""Open-loop schedule baseline controller (Phase 1.5).

An :class:`OpenLoopScheduleController` is an ablation baseline for the
learned evolution controllers: it selects its action purely from the
normalized search progress ``generation / max_generations`` and never
observes the population state. Comparing a closed-loop learned
controller against this schedule isolates how much of the learned
controller's benefit comes from *feedback* rather than from a simple
annealing profile over the generation index.

Training corpus. ``fit`` consumes ``records`` — one entry per trajectory
run, in the pre-digested form::

    {"problem": str,
     "n_vars": int,
     "transitions": [{"generation": int,
                      "state": {...},
                      "action": {"mutation_operator": str,
                                 "mutation_probability": float,
                                 "exploration_strength": float},
                      "reward": {...}},
                     ...]}

Binning. Each record is normalized by *its own* maximum transition
generation: ``progress = generation / max_generation`` (defined as
``0.0`` when the record's maximum generation is not positive). Bins are
the ``n_bins`` (default 10) equal-width intervals ``[0.0, 0.1), ...,
[0.9, 1.0]``, with ``progress == 1.0`` falling into the last bin. Per
bin the fitted statistics are:

* the majority ``mutation_operator`` (ties break alphabetically, so the
  result is independent of record order),
* the median of the mutation *multiplier*
  ``mutation_probability * n_vars`` — the multiplier, not the raw
  probability, is pooled so records with different ``n_vars``
  aggregate correctly, and
* the median ``exploration_strength`` restricted to the transitions
  whose operator equals the bin's majority operator.

Empty-bin fallback. Empty bins inherit the statistics of the nearest
non-empty bin, preferring the *preceding* bin (pandas ``ffill``
semantics); leading empty bins inherit the nearest *following* bin. A
per-problem schedule with no data at all is not kept: prediction for
such a problem falls back to the global schedule with a WARNING.

Prediction. ``mutation_probability = multiplier / n_vars`` where
``n_vars`` is looked up from the queried ``problem_name`` — both the
global and the per-problem variant use the *queried* problem's
``n_vars`` (the multiplier is the pooled quantity; the probability is
problem-specific). If the problem is unknown, the median of the fitted
problems' ``n_vars`` is used and a WARNING is logged. The returned
probability is not clipped and may exceed 1 if the fitted multiplier
exceeds ``n_vars``.

Statelessness. ``predict_action`` is a pure function of ``(generation,
max_generations, problem_name)`` and never mutates the fitted schedule,
so the same arguments always yield exactly the same action regardless
of the call history.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["OpenLoopScheduleController"]

_LOGGER = logging.getLogger(__name__)

_SAVE_FORMAT = "open_loop_schedule_controller"
_SAVE_VERSION = 1

#: Guard against float fuzz when a progress value lands exactly on a bin
#: boundary (e.g. ``0.3 * 10`` evaluating to ``2.9999999999999996``).
_BIN_EPS = 1e-9

#: A fitted bin: majority operator, median mutation multiplier, and the
#: exploration median of the majority-operator group. ``None`` marks an
#: empty bin prior to fallback filling.
BinStats = Optional[Dict[str, Any]]

#: A full schedule: one entry per normalized-progress bin.
Schedule = List[BinStats]

#: Per-transition tuple pooled during ``fit``:
#: ``(operator, multiplier, exploration_strength)``.
_Entry = Tuple[str, float, float]

PathLike = Union[str, Path]


def _bin_index(generation: float, max_generation: float, n_bins: int) -> int:
    """Map ``(generation, max_generation)`` to a bin index in ``[0, n_bins)``.

    The normalized progress is clamped to ``[0.0, 1.0]`` before binning,
    so negative generations land in the first bin and generations beyond
    ``max_generation`` land in the last bin.

    Args:
        generation: Current generation index.
        max_generation: Normalization constant (a record's maximum
            transition generation during ``fit``; the caller-provided
            ``max_generations`` during prediction).
        n_bins: Number of equal-width progress bins.

    Returns:
        The bin index in ``[0, n_bins)``.
    """
    if max_generation <= 0:
        progress = 0.0
    else:
        progress = float(generation) / float(max_generation)
    progress = min(max(progress, 0.0), 1.0)
    return min(int(progress * n_bins + _BIN_EPS), n_bins - 1)


def _majority_operator(operators: Sequence[str]) -> str:
    """Return the most frequent operator, breaking ties alphabetically.

    Args:
        operators: Operators of all transitions pooled into one bin.

    Returns:
        The majority operator name.
    """
    counts: Dict[str, int] = {}
    for operator in operators:
        counts[operator] = counts.get(operator, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _bin_stats(entries: Sequence[_Entry]) -> Dict[str, Any]:
    """Summarize one non-empty bin.

    Args:
        entries: Pooled ``(operator, multiplier, exploration)``
            transition tuples of the bin.

    Returns:
        Dict with ``mutation_operator`` (majority), the median
        ``mutation_multiplier``, and the ``exploration_strength`` median
        over the majority-operator group only.
    """
    operator = _majority_operator([entry[0] for entry in entries])
    group = [entry[2] for entry in entries if entry[0] == operator]
    return {
        "mutation_operator": operator,
        "mutation_multiplier": float(
            np.median(np.asarray([entry[1] for entry in entries], dtype=float))
        ),
        "exploration_strength": float(np.median(np.asarray(group, dtype=float))),
    }


def _fill_empty_bins(bins: Schedule) -> Schedule:
    """Fill empty bins from the nearest non-empty bin.

    Preceding bins take priority (pandas ``ffill`` semantics); bins in a
    leading gap — with no non-empty bin before them — inherit the
    nearest following non-empty bin.

    Args:
        bins: Per-bin statistics with ``None`` marking empty bins.

    Returns:
        A new schedule of the same length. If *every* bin is empty, all
        entries remain ``None`` and the caller must handle the fallback.
    """
    filled = list(bins)
    last: BinStats = None
    for index, stats in enumerate(filled):
        if stats is not None:
            last = stats
        elif last is not None:
            filled[index] = dict(last)
    following: BinStats = None
    for index in range(len(filled) - 1, -1, -1):
        if filled[index] is not None:
            following = filled[index]
        elif following is not None:
            filled[index] = dict(following)
    return filled


def _build_schedule(
    pooled: Sequence[Sequence[_Entry]], n_bins: int
) -> Schedule:
    """Compute per-bin statistics and apply the empty-bin fallback.

    Args:
        pooled: One list of pooled transition tuples per bin.
        n_bins: Expected number of bins (for validation).

    Returns:
        The fallback-filled schedule; all-``None`` if no bin has data.

    Raises:
        ValueError: If ``len(pooled)`` does not equal ``n_bins``.
    """
    if len(pooled) != n_bins:
        raise ValueError(f"expected {n_bins} bins, got {len(pooled)}")
    schedule: Schedule = [
        _bin_stats(entries) if entries else None for entries in pooled
    ]
    return _fill_empty_bins(schedule)


class OpenLoopScheduleController:
    """Open-loop (progress-only) schedule controller.

    The controller distills a corpus of evolution trajectories into a
    per-bin action schedule over the normalized generation progress and
    replays that schedule at prediction time. It never observes
    population state, which makes it the control baseline for testing
    whether learned controllers benefit from closed-loop feedback.

    Attributes:
        per_problem: Whether per-problem schedules are used at
            prediction time (read-only property).
        n_bins: Number of normalized-progress bins (read-only property).
    """

    def __init__(self, per_problem: bool = False, n_bins: int = 10) -> None:
        """Initialize the controller.

        Args:
            per_problem: If ``True``, build one schedule per problem and
                dispatch on ``problem_name`` at prediction time;
                otherwise a single pooled global schedule is used.
            n_bins: Number of equal-width normalized-progress bins.

        Raises:
            ValueError: If ``n_bins`` is not positive.
        """
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        self._per_problem = bool(per_problem)
        self._n_bins = int(n_bins)
        self._n_vars_by_problem: Dict[str, int] = {}
        self._default_n_vars: Optional[float] = None
        self._global_schedule: Schedule = [None] * self._n_bins
        self._per_problem_schedules: Dict[str, Schedule] = {}
        self._fitted = False

    @property
    def name(self) -> str:
        """Canonical arm name of this controller."""
        return "open_loop_per_problem" if self._per_problem else "open_loop_global"

    @property
    def per_problem(self) -> bool:
        """Whether prediction dispatches on the problem name."""
        return self._per_problem

    @property
    def n_bins(self) -> int:
        """Number of normalized-progress bins."""
        return self._n_bins

    def fit(
        self, records: List[Dict[str, Any]]
    ) -> "OpenLoopScheduleController":
        """Fit the schedule(s) from a corpus of trajectory records.

        Each record is normalized by its own maximum transition
        generation; transition actions are pooled per bin as documented
        in the module docstring. The ``problem -> n_vars`` mapping is
        recorded from the first occurrence of each problem.

        Args:
            records: List of dicts of the form
                ``{"problem": str, "n_vars": int, "transitions": [...]}``.
                Records with an empty ``transitions`` list still
                contribute their ``n_vars`` mapping.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: If ``records`` is empty, contains no usable
                transitions at all, or a record/transition is missing
                required keys.
        """
        if not records:
            raise ValueError("records must be a non-empty list")
        self._n_vars_by_problem = {}
        self._default_n_vars = None
        self._per_problem_schedules = {}
        global_pooled: List[List[_Entry]] = [[] for _ in range(self._n_bins)]
        problem_pooled: Dict[str, List[List[_Entry]]] = {}
        total_transitions = 0

        for r_index, record in enumerate(records):
            try:
                problem = record["problem"]
                n_vars = int(record["n_vars"])
                transitions = record["transitions"]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"record {r_index} is malformed: {exc!r}"
                ) from exc
            if n_vars <= 0:
                raise ValueError(
                    f"record {r_index} has non-positive n_vars={n_vars}"
                )
            if problem not in self._n_vars_by_problem:
                self._n_vars_by_problem[problem] = n_vars

            if not transitions:
                continue
            max_generation = max(
                int(transition["generation"]) for transition in transitions
            )
            if problem not in problem_pooled:
                problem_pooled[problem] = [
                    [] for _ in range(self._n_bins)
                ]
            for t_index, transition in enumerate(transitions):
                try:
                    action = transition["action"]
                    operator = action["mutation_operator"]
                    multiplier = (
                        float(action["mutation_probability"]) * n_vars
                    )
                    exploration = float(action["exploration_strength"])
                except (KeyError, TypeError) as exc:
                    raise ValueError(
                        f"record {r_index} transition {t_index} is "
                        f"malformed: {exc!r}"
                    ) from exc
                entry: _Entry = (operator, multiplier, exploration)
                bin_index = _bin_index(
                    int(transition["generation"]), max_generation, self._n_bins
                )
                global_pooled[bin_index].append(entry)
                problem_pooled[problem][bin_index].append(entry)
                total_transitions += 1

        if total_transitions == 0:
            raise ValueError("records contain no transitions to fit")

        self._global_schedule = _build_schedule(global_pooled, self._n_bins)
        if self._per_problem:
            self._per_problem_schedules = {
                problem: schedule
                for problem, pooled in problem_pooled.items()
                if any(entries for entries in pooled)
                for schedule in (_build_schedule(pooled, self._n_bins),)
            }
        self._default_n_vars = float(
            np.median(sorted(self._n_vars_by_problem.values()))
        )
        self._fitted = True
        return self

    def _select_schedule(self, problem_name: Optional[str]) -> Schedule:
        """Choose the prediction schedule for ``problem_name``.

        Args:
            problem_name: Problem queried at prediction time.

        Returns:
            The per-problem schedule when available, otherwise the
            global schedule (with a WARNING in per-problem mode).
        """
        if not self._per_problem:
            return self._global_schedule
        schedule = self._per_problem_schedules.get(problem_name)
        if schedule is None:
            _LOGGER.warning(
                "no per-problem schedule for problem %r; falling back to "
                "the global schedule",
                problem_name,
            )
            return self._global_schedule
        return schedule

    def predict_action(
        self,
        generation: int,
        max_generations: int,
        problem_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Predict the action for a normalized generation progress.

        The action is looked up in the fitted schedule bin of the
        progress ``generation / max_generations``; it does not depend on
        any other state or on previous calls.

        Args:
            generation: Current generation index (clamped to
                ``[0, max_generations]`` for binning).
            max_generations: Normalization constant of the running
                experiment (``0`` or negative maps to the first bin).
            problem_name: Problem whose ``n_vars`` scales the pooled
                multiplier into a probability. ``None`` or an unknown
                name triggers the documented ``n_vars`` fallback with a
                WARNING.

        Returns:
            ``{"mutation_operator": str, "mutation_probability": float,
            "exploration_strength": float}``.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self._fitted:
            raise RuntimeError(
                "OpenLoopScheduleController.predict_action called before fit"
            )
        schedule = self._select_schedule(problem_name)
        stats = schedule[_bin_index(generation, max_generations, self._n_bins)]
        n_vars = self._n_vars_by_problem.get(problem_name)
        if n_vars is None:
            n_vars = self._default_n_vars
            _LOGGER.warning(
                "unknown problem %r; scaling with default n_vars=%s "
                "(median over fitted problems)",
                problem_name,
                n_vars,
            )
        return {
            "mutation_operator": stats["mutation_operator"],
            "mutation_probability": stats["mutation_multiplier"] / n_vars,
            "exploration_strength": stats["exploration_strength"],
        }

    def save(self, path: PathLike) -> None:
        """Serialize the fitted controller to a JSON file.

        Args:
            path: Destination file path. Parent directories are not
                created; the file is overwritten if it exists.
        """
        payload = {
            "format": _SAVE_FORMAT,
            "version": _SAVE_VERSION,
            "per_problem": self._per_problem,
            "n_bins": self._n_bins,
            "default_n_vars": self._default_n_vars,
            "n_vars_by_problem": self._n_vars_by_problem,
            "global_schedule": self._global_schedule,
            "per_problem_schedules": self._per_problem_schedules,
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: PathLike) -> "OpenLoopScheduleController":
        """Restore a controller from a JSON file written by :meth:`save`.

        Args:
            path: Source file path.

        Returns:
            The restored, ready-to-use controller instance.

        Raises:
            ValueError: If the file is not a valid saved controller of a
                compatible format version.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != _SAVE_FORMAT:
            raise ValueError(
                f"not a {_SAVE_FORMAT} file: {str(path)!r}"
            )
        if int(data.get("version", 0)) != _SAVE_VERSION:
            raise ValueError(
                f"unsupported {_SAVE_FORMAT} version "
                f"{data.get('version')!r} in {str(path)!r}"
            )
        controller = cls(
            per_problem=bool(data["per_problem"]), n_bins=int(data["n_bins"])
        )
        global_schedule = data["global_schedule"]
        if len(global_schedule) != controller._n_bins:
            raise ValueError("corrupt file: global schedule length mismatch")
        per_problem_schedules = {}
        for problem, schedule in data["per_problem_schedules"].items():
            if len(schedule) != controller._n_bins:
                raise ValueError(
                    f"corrupt file: schedule length mismatch for {problem!r}"
                )
            per_problem_schedules[problem] = schedule
        controller._n_vars_by_problem = {
            str(problem): int(n_vars)
            for problem, n_vars in data["n_vars_by_problem"].items()
        }
        controller._default_n_vars = data["default_n_vars"]
        controller._global_schedule = global_schedule
        controller._per_problem_schedules = per_problem_schedules
        controller._fitted = True
        return controller
