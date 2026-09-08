from __future__ import annotations

from benchmarks.base import Problem
from benchmarks.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6

__all__ = ["Problem", "ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "get_problem"]

_PROBLEMS: dict[str, type[Problem]] = {
    "zdt1": ZDT1,
    "zdt2": ZDT2,
    "zdt3": ZDT3,
    "zdt4": ZDT4,
    "zdt6": ZDT6,
}


def get_problem(name: str) -> Problem:
    """Instantiate a benchmark problem by name.

    Args:
        name: Problem identifier, case-insensitive. Supported: ``zdt1``,
            ``zdt2``, ``zdt3``, ``zdt4``, ``zdt6``. ``zdt5`` is intentionally
            unsupported (binary-coded problem, outside the continuous
            real-valued scope of Phase 0).

    Returns:
        A fresh :class:`Problem` instance.

    Raises:
        ValueError: If ``name`` is not a supported problem.
    """
    key = name.lower()
    if key not in _PROBLEMS:
        raise ValueError(
            f"Unknown problem {name!r}. Supported: {sorted(_PROBLEMS)} "
            "(zdt5 is binary-coded and intentionally unsupported)."
        )
    return _PROBLEMS[key]()
