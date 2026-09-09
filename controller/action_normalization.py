from __future__ import annotations

"""Normalized mutation action scale for Phase 1.75.

The natural NSGA-II mutation scale is ``pm_base = 1 / n_vars``, so the same
controller output means very different absolute mutation probabilities on
problems with different dimensions (e.g. ZDT1-3 with ``n_vars = 30`` versus
ZDT4/ZDT6 with ``n_vars = 10``). To remove this trivial scale confound from
the learned regression target, Phase 1.75 expresses the mutation action as a
dimensionless multiplier of the baseline::

    multiplier = mutation_probability * n_vars
    log_multiplier = log(mutation_probability * n_vars)

A multiplier of ``1.0`` (log-multiplier ``0.0``) is exactly the classic
``1 / n_vars`` default. The Phase-1.5 random policy samples this multiplier
log-uniformly from ``[0.25, 8.0]`` (see
``experiments.generate_dataset.FULL_ACTION_PM_MULT_RANGE``), which is also
the deployment clip range of multiplier-mode controllers.

All functions validate strictly and return plain Python floats.
"""

import math

__all__ = [
    "mutation_multiplier",
    "mutation_probability",
    "log_mutation_multiplier",
    "pm_from_log_multiplier",
]


def _validate_n_vars(n_vars: int) -> int:
    """Return ``n_vars`` as an int, rejecting non-positive dimensions.

    Args:
        n_vars: Number of decision variables; must be >= 1.

    Returns:
        ``int(n_vars)``.

    Raises:
        ValueError: If ``n_vars`` < 1.
    """
    n_vars = int(n_vars)
    if n_vars < 1:
        raise ValueError(f"n_vars must be >= 1, got {n_vars}")
    return n_vars


def mutation_multiplier(pm: float, n_vars: int) -> float:
    """Convert an absolute mutation probability to a baseline multiplier.

    Args:
        pm: Per-variable mutation probability; must be > 0.
        n_vars: Number of decision variables; must be >= 1.

    Returns:
        ``pm * n_vars`` as a Python float. ``1.0`` corresponds to the
        classic ``1 / n_vars`` default.

    Raises:
        ValueError: If ``pm`` <= 0 or ``n_vars`` < 1.
    """
    n_vars = _validate_n_vars(n_vars)
    pm = float(pm)
    if pm <= 0.0:
        raise ValueError(f"mutation probability must be > 0, got {pm}")
    return float(pm * n_vars)


def mutation_probability(multiplier: float, n_vars: int) -> float:
    """Convert a baseline multiplier back to an absolute probability.

    Args:
        multiplier: Multiplier of the ``1 / n_vars`` baseline; must be > 0.
        n_vars: Number of decision variables; must be >= 1.

    Returns:
        ``multiplier / n_vars`` as a Python float.

    Raises:
        ValueError: If ``multiplier`` <= 0 or ``n_vars`` < 1.
    """
    n_vars = _validate_n_vars(n_vars)
    multiplier = float(multiplier)
    if multiplier <= 0.0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")
    return float(multiplier / n_vars)


def log_mutation_multiplier(pm: float, n_vars: int) -> float:
    """Regression target: ``log`` of the normalized mutation multiplier.

    Args:
        pm: Per-variable mutation probability; must be > 0.
        n_vars: Number of decision variables; must be >= 1.

    Returns:
        ``log(pm * n_vars)`` as a Python float. The ``1 / n_vars`` default
        maps to ``0.0``.

    Raises:
        ValueError: If ``pm`` <= 0 or ``n_vars`` < 1.
    """
    return float(math.log(mutation_multiplier(pm, n_vars)))


def pm_from_log_multiplier(log_multiplier: float, n_vars: int) -> float:
    """Invert :func:`log_mutation_multiplier`.

    Args:
        log_multiplier: Log of the baseline multiplier (any real value).
        n_vars: Number of decision variables; must be >= 1.

    Returns:
        ``exp(log_multiplier) / n_vars`` as a Python float.

    Raises:
        ValueError: If ``n_vars`` < 1.
    """
    n_vars = _validate_n_vars(n_vars)
    return float(math.exp(float(log_multiplier)) / n_vars)
