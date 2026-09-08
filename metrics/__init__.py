from __future__ import annotations

"""Quality indicators for two-objective minimization fronts."""

from metrics.indicators import diversity_spread, hypervolume, igd

__all__ = ["hypervolume", "igd", "diversity_spread"]
