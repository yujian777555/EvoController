"""Static full-action-space controller (Phase 1.75 matched baseline).

:class:`StaticFullController` emits the same action triple
``(mutation_operator, mutation_probability, exploration_strength)`` at every
generation regardless of the population state. It is the matched-action-space
static baseline of Phase 1.75: it has access to exactly the action space of
the learned multi-head controller, but no state feedback, so any gain of the
closed-loop controller over the best static tuple is attributable to
state-dependent decisions rather than to a richer action space.

The mutation probability is parameterized as a *multiplier* on the natural
NSGA-II scale ``1 / n_vars`` (``pm = multiplier / n_vars``), so a single
tuned tuple transfers across problems of different dimensionality. The
exploration strength is the polynomial distribution index ``eta_m`` when the
operator is ``"polynomial"`` and the range-scaled Gaussian ``sigma`` when it
is ``"gaussian"`` (see :class:`algorithms.OperatorConfig`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

#: Mutation operators supported by the NSGA-II backend.
SUPPORTED_OPERATORS: tuple[str, ...] = ("polynomial", "gaussian")


class StaticFullController:
    """History-free controller emitting one fixed full-action triple.

    The controller ignores the evolution history entirely; given the problem
    dimensionality it always returns the same ``(operator, multiplier /
    n_vars, exploration_strength)`` action. Instances are immutable after
    construction and serialize to plain JSON.

    Attributes:
        name: Identifier of this controller instance, derived from the
            action parameters (used in experiment records).
        operator: Mutation operator, ``"polynomial"`` or ``"gaussian"``.
        multiplier: Mutation-probability multiplier on ``1 / n_vars``.
        exploration_strength: Exploration strength of the operator
            (``eta_m`` for polynomial, ``sigma`` for Gaussian).
    """

    def __init__(
        self,
        operator: str,
        multiplier: float,
        exploration_strength: float,
    ) -> None:
        """Create a static full-action controller.

        Args:
            operator: Mutation operator; must be one of
                :data:`SUPPORTED_OPERATORS`.
            multiplier: Mutation-probability multiplier on the natural
                NSGA-II scale ``1 / n_vars``; must be positive.
            exploration_strength: Exploration strength of the operator
                (``eta_m`` or ``sigma``); must be positive.

        Raises:
            ValueError: If ``operator`` is unsupported, or ``multiplier``
                or ``exploration_strength`` is not positive.
        """
        if operator not in SUPPORTED_OPERATORS:
            raise ValueError(
                f"unsupported operator {operator!r}; expected one of {SUPPORTED_OPERATORS}"
            )
        if not float(multiplier) > 0.0:
            raise ValueError(f"multiplier must be positive, got {multiplier}")
        if not float(exploration_strength) > 0.0:
            raise ValueError(
                f"exploration_strength must be positive, got {exploration_strength}"
            )
        self.operator = str(operator)
        self.multiplier = float(multiplier)
        self.exploration_strength = float(exploration_strength)
        self.name = (
            f"static_full_{self.operator}"
            f"_mult{self.multiplier:.6g}"
            f"_expl{self.exploration_strength:.6g}"
        )

    def predict_action(
        self,
        history: Sequence[dict[str, Any]] | None = None,
        *,
        n_vars: int,
    ) -> dict[str, Any]:
        """Return the fixed action triple for a problem with ``n_vars`` variables.

        Args:
            history: Ignored; present for interface compatibility with the
                learned controllers. The static controller is history-free
                by construction.
            n_vars: Number of decision variables of the problem; must be
                >= 1. Only used to resolve ``pm = multiplier / n_vars``.

        Returns:
            Dict with keys ``"mutation_operator"`` (str),
            ``"mutation_probability"`` (``multiplier / n_vars``), and
            ``"exploration_strength"`` (float) — the same keys as
            :meth:`controller.MultiHeadController.predict_action`.

        Raises:
            ValueError: If ``n_vars`` < 1.
        """
        if int(n_vars) < 1:
            raise ValueError(f"n_vars must be >= 1, got {n_vars}")
        return {
            "mutation_operator": self.operator,
            "mutation_probability": self.multiplier / float(n_vars),
            "exploration_strength": self.exploration_strength,
        }

    def save(self, path: str | Path) -> None:
        """Serialize the controller to a JSON file.

        Args:
            path: Destination path; parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "StaticFullController",
            "name": self.name,
            "operator": self.operator,
            "multiplier": self.multiplier,
            "exploration_strength": self.exploration_strength,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "StaticFullController":
        """Load a controller saved with :meth:`save`.

        Args:
            path: Path to the JSON file written by :meth:`save`.

        Returns:
            The reconstructed controller.

        Raises:
            ValueError: If the payload is missing a required field or a
                field value fails constructor validation.
        """
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        try:
            operator = payload["operator"]
            multiplier = payload["multiplier"]
            exploration_strength = payload["exploration_strength"]
        except KeyError as exc:
            raise ValueError(f"invalid StaticFullController payload: missing {exc}") from exc
        return cls(
            operator=str(operator),
            multiplier=float(multiplier),
            exploration_strength=float(exploration_strength),
        )
