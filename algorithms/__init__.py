"""Evolutionary algorithms for EvoController.

Phase 0 exposes only the NSGA-II baseline used to generate evolution
trajectory datasets; learned controllers arrive in later phases.
"""

from __future__ import annotations

from algorithms.nsga2 import NSGAII, OperatorConfig

__all__ = ["NSGAII", "OperatorConfig"]
