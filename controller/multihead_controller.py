from __future__ import annotations

"""Multi-head MLP controller for the Phase-1.5 full action space.

Phase 1.5 expands the controller action space from the mutation probability
alone to the triple ``(mutation_operator, mutation_probability,
exploration_strength)``. :class:`MultiHeadController` predicts all three
components from an encoded history window: a shared ``Linear -> ReLU`` trunk
feeds three heads — operator logits (two classes: ``0 = polynomial``,
``1 = gaussian``), log mutation probability, and log exploration strength.

Training minimizes the sum of the sample-weighted head losses:
cross-entropy for the operator head and MSE for the two log-space
regression heads. The advantage weights are identical to the Phase-1
dataset (see :func:`controller.dataset.build_multihead_samples`). Given the
same seed and data, construction and training are deterministic.

Deployment (:meth:`MultiHeadController.predict_action`) maps the heads to a
valid action dict: the operator is the argmax of the softmax probabilities,
the mutation probability is ``exp`` of the log-prediction clipped to
``[pm_min, pm_max]``, and the exploration strength is ``exp`` of the
log-prediction clipped to the operator-specific range
:data:`POLYNOMIAL_EXPLORATION_RANGE` (the polynomial distribution index
``eta_m``) or :data:`GAUSSIAN_EXPLORATION_RANGE` (the Gaussian ``sigma``).
These are exactly the ranges the Phase-1.5 dataset generator samples from
(``experiments.generate_dataset``).

Phase 1.75 adds the ``mutation_target="multiplier"`` mode: the pm head is
then trained on — and interpreted as — the log of the *normalized* mutation
multiplier ``pm * n_vars`` (see :mod:`controller.action_normalization`), so
the same network serves problems of different dimensions without a scale
confound. At deployment the prediction is clipped to
:data:`MULTIPLIER_RANGE` in log space and divided by the problem's
``n_vars``. The default ``mutation_target="absolute"`` reproduces the
Phase-1.5 behavior byte-for-byte, including for legacy checkpoints that
predate the mode flag.
"""

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from controller.action_normalization import pm_from_log_multiplier

#: Operator classes of the operator head, in class-index order.
OPERATOR_CLASSES: tuple[str, ...] = ("polynomial", "gaussian")

#: Deployment clip range of the exploration strength for the polynomial
#: operator (``eta_m``); matches ``ETA_M_SAMPLE_RANGE`` of the generator.
POLYNOMIAL_EXPLORATION_RANGE: tuple[float, float] = (2.0, 50.0)

#: Deployment clip range of the exploration strength for the Gaussian
#: operator (``sigma``); matches ``SIGMA_SAMPLE_RANGE`` of the generator.
GAUSSIAN_EXPLORATION_RANGE: tuple[float, float] = (0.02, 0.3)

#: Deployment clip range of the normalized mutation multiplier
#: (``pm * n_vars``) in ``mutation_target="multiplier"`` mode; matches
#: ``FULL_ACTION_PM_MULT_RANGE`` of the Phase-1.5/1.75 dataset generator.
MULTIPLIER_RANGE: tuple[float, float] = (0.25, 8.0)

#: Valid values of the ``mutation_target`` mode flag.
MUTATION_TARGETS: tuple[str, ...] = ("absolute", "multiplier")


def _clip_log(log_value: float, lo: float, hi: float) -> float:
    """Clip a log-space value to ``[log(lo), log(hi)]``.

    Args:
        log_value: Predicted log-space value.
        lo: Lower bound of the output range; must be > 0 (log-space target).
        hi: Upper bound of the output range; must be >= ``lo``.

    Returns:
        ``log_value`` clipped to ``[log(lo), log(hi)]``.

    Raises:
        ValueError: If the bounds are invalid.
    """
    if lo <= 0.0:
        raise ValueError(f"lower bound must be > 0 for log-space output, got {lo}")
    if hi < lo:
        raise ValueError(f"upper bound ({hi}) must be >= lower bound ({lo})")
    return min(max(float(log_value), math.log(lo)), math.log(hi))


class _MultiHeadNet(nn.Module):
    """Shared-trunk network with operator, log-pm, and log-exploration heads."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        """Build the trunk and the three heads.

        Args:
            input_dim: Flattened input dimension.
            hidden_dims: Hidden layer widths of the shared trunk.
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for width in hidden_dims:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.ReLU())
            prev = int(width)
        self.trunk = nn.Sequential(*layers)
        self.operator_head = nn.Linear(prev, len(OPERATOR_CLASSES))
        self.pm_head = nn.Linear(prev, 1)
        self.exploration_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(operator logits (n, 2), log pm (n,), log exploration (n,))``."""
        h = self.trunk(x)
        return (
            self.operator_head(h),
            self.pm_head(h).view(-1),
            self.exploration_head(h).view(-1),
        )


class MultiHeadController:
    """Feed-forward controller predicting the full Phase-1.5 action triple.

    The network is a stack of ``Linear -> ReLU`` blocks (the shared trunk)
    followed by three linear heads: operator logits (2 classes), log
    mutation probability, and log exploration strength. It is trained with
    the sum of the sample-weighted head losses (cross-entropy + MSE + MSE)
    using Adam. Given the same seed and data, construction and training are
    deterministic.

    Attributes:
        name: Identifier of this controller (reported in experiment
            configs); default ``"mlp2"``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        seed: int = 0,
        lr: float = 1e-3,
        name: str = "mlp2",
        mutation_target: str = "absolute",
    ) -> None:
        """Initialize the controller and reseed the torch/numpy RNGs.

        Args:
            input_dim: Flattened input dimension, e.g.
                ``StateEncoder.dim`` (``window * 6``) or
                ``ProblemAwareEncoder.dim`` (``window * 6 + 9``).
            hidden_dims: Hidden layer widths of the shared trunk.
            seed: Random seed; applied to ``torch.manual_seed`` and
                ``np.random.seed``.
            lr: Adam learning rate.
            name: Controller identifier used in experiment records.
            mutation_target: Interpretation of the pm head:
                ``"absolute"`` (default; log mutation probability, the
                Phase-1.5 behavior) or ``"multiplier"`` (log of the
                normalized multiplier ``pm * n_vars``, Phase 1.75).

        Raises:
            ValueError: If ``input_dim`` < 1 or ``mutation_target`` is
                not one of :data:`MUTATION_TARGETS`.
        """
        if int(input_dim) < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if mutation_target not in MUTATION_TARGETS:
            raise ValueError(
                f"mutation_target must be one of {MUTATION_TARGETS}, got "
                f"{mutation_target!r}"
            )
        torch.manual_seed(seed)
        np.random.seed(seed)
        self._input_dim = int(input_dim)
        self._hidden_dims = tuple(int(h) for h in hidden_dims)
        self._seed = int(seed)
        self._lr = float(lr)
        self.name = str(name)
        self._mutation_target = str(mutation_target)
        self._net = _MultiHeadNet(self._input_dim, self._hidden_dims)

    @property
    def input_dim(self) -> int:
        """Flattened input dimension."""
        return self._input_dim

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        """Hidden layer widths of the shared trunk."""
        return self._hidden_dims

    @property
    def mutation_target(self) -> str:
        """Pm-head interpretation: ``"absolute"`` or ``"multiplier"``."""
        return self._mutation_target

    def fit(
        self,
        X: np.ndarray,
        y_op: np.ndarray,
        y_logpm: np.ndarray,
        y_logexpl: np.ndarray,
        sample_weight: np.ndarray | None = None,
        epochs: int = 200,
        batch_size: int = 256,
        X_val: np.ndarray | None = None,
        yop_val: np.ndarray | None = None,
        ypm_val: np.ndarray | None = None,
        yexpl_val: np.ndarray | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train the network with the weighted three-head loss and Adam.

        The per-sample total loss is ``w * (CE(operator) + MSE(log pm) +
        MSE(log exploration))``, normalized by the weight sum per batch.
        Batches are drawn with a torch generator seeded by the controller's
        seed, so repeated runs with the same seed and data produce identical
        parameters. Reported epoch losses are full-set totals: weighted for
        the training set, unweighted for the validation set (validation
        weights are not part of the interface).

        Args:
            X: Feature matrix of shape ``(n, input_dim)``.
            y_op: Operator class indices of shape ``(n,)``; values in
                ``{0, 1}`` (``0 = polynomial``, ``1 = gaussian``).
            y_logpm: Log mutation probability targets of shape ``(n,)``.
            y_logexpl: Log exploration strength targets of shape ``(n,)``.
            sample_weight: Optional nonnegative weights of shape ``(n,)``;
                uniform weights when omitted.
            epochs: Number of training epochs.
            batch_size: Mini-batch size (clamped to ``n``).
            X_val: Optional validation features of shape ``(m, input_dim)``.
            yop_val: Optional validation operator indices of shape ``(m,)``.
            ypm_val: Optional validation log-pm targets of shape ``(m,)``.
            yexpl_val: Optional validation log-exploration targets ``(m,)``.
            verbose: Print progress every 10 epochs when True.

        Returns:
            ``{"train_loss": [...], "val_loss": [...]}`` with one entry per
            epoch; ``val_loss`` is empty unless all four validation arrays
            were given.

        Raises:
            ValueError: If ``X`` contains no samples or ``y_op`` holds a
                class index outside ``{0, 1}``.
        """
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X_t.ndim == 1:
            X_t = X_t.reshape(1, -1)
        yop_t = torch.as_tensor(np.asarray(y_op, dtype=np.int64)).view(-1)
        ypm_t = torch.as_tensor(np.asarray(y_logpm, dtype=np.float32)).view(-1)
        yexpl_t = torch.as_tensor(np.asarray(y_logexpl, dtype=np.float32)).view(-1)
        n_samples = int(X_t.shape[0])
        if n_samples == 0:
            raise ValueError("cannot fit MultiHeadController on zero samples")
        if int(yop_t.min()) < 0 or int(yop_t.max()) >= len(OPERATOR_CLASSES):
            raise ValueError(
                f"y_op class indices must lie in [0, {len(OPERATOR_CLASSES)}), "
                f"got min={int(yop_t.min())}, max={int(yop_t.max())}"
            )
        if sample_weight is None:
            w_t = torch.ones(n_samples)
        else:
            w_t = torch.as_tensor(np.asarray(sample_weight, dtype=np.float32)).view(-1)

        dataset = TensorDataset(X_t, yop_t, ypm_t, yexpl_t, w_t)
        generator = torch.Generator().manual_seed(self._seed)
        loader = DataLoader(
            dataset,
            batch_size=min(int(batch_size), n_samples),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)

        has_val = all(
            v is not None for v in (X_val, yop_val, ypm_val, yexpl_val)
        )
        val_tensors: tuple[torch.Tensor, ...] | None = None
        if has_val:
            val_tensors = (
                torch.as_tensor(np.asarray(X_val, dtype=np.float32)),
                torch.as_tensor(np.asarray(yop_val, dtype=np.int64)).view(-1),
                torch.as_tensor(np.asarray(ypm_val, dtype=np.float32)).view(-1),
                torch.as_tensor(np.asarray(yexpl_val, dtype=np.float32)).view(-1),
            )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        for epoch in range(int(epochs)):
            self._net.train()
            for xb, yob, ypb, yeb, wb in loader:
                logits, pm_pred, expl_pred = self._net(xb)
                denom = torch.sum(wb)
                loss = (
                    torch.sum(wb * F.cross_entropy(logits, yob, reduction="none")) / denom
                    + torch.sum(wb * (pm_pred - ypb) ** 2) / denom
                    + torch.sum(wb * (expl_pred - yeb) ** 2) / denom
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            history["train_loss"].append(self._total_loss(X_t, yop_t, ypm_t, yexpl_t, w_t))
            if has_val:
                assert val_tensors is not None
                history["val_loss"].append(self._total_loss(*val_tensors, None))
            if verbose and (epoch % 10 == 0 or epoch == int(epochs) - 1):
                message = (
                    f"epoch {epoch + 1}/{int(epochs)} "
                    f"train_loss={history['train_loss'][-1]:.6f}"
                )
                if has_val:
                    message += f" val_loss={history['val_loss'][-1]:.6f}"
                print(message)
        return history

    def _total_loss(
        self,
        X_t: torch.Tensor,
        yop_t: torch.Tensor,
        ypm_t: torch.Tensor,
        yexpl_t: torch.Tensor,
        w_t: torch.Tensor | None,
    ) -> float:
        """Full-set total loss in eval mode; weighted when ``w_t`` is given."""
        self._net.eval()
        with torch.no_grad():
            logits, pm_pred, expl_pred = self._net(X_t)
            ce_per = F.cross_entropy(logits, yop_t, reduction="none")
            mse_pm_per = (pm_pred - ypm_t) ** 2
            mse_expl_per = (expl_pred - yexpl_t) ** 2
            per_sample = ce_per + mse_pm_per + mse_expl_per
            if w_t is None:
                return float(torch.mean(per_sample))
            return float(torch.sum(w_t * per_sample) / torch.sum(w_t))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict operator probabilities and log-space regression outputs.

        Args:
            X: Feature matrix of shape ``(n, input_dim)`` (a 1-D array is
                treated as a single sample).

        Returns:
            Tuple ``(operator_probs, log_pm, log_expl)``: softmax operator
            probabilities of shape ``(n, 2)`` (rows sum to 1, column order
            follows :data:`OPERATOR_CLASSES`), log mutation probabilities of
            shape ``(n,)``, and log exploration strengths of shape ``(n,)``.
        """
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X_t.ndim == 1:
            X_t = X_t.reshape(1, -1)
        self._net.eval()
        with torch.no_grad():
            logits, log_pm, log_expl = self._net(X_t)
            probs = torch.softmax(logits, dim=1)
        return (
            probs.numpy().astype(float),
            log_pm.numpy().astype(float),
            log_expl.numpy().astype(float),
        )

    def predict_action(
        self,
        history: list[dict[str, Any]],
        encoder: Any,
        pm_min: float,
        pm_max: float,
        n_vars: int | None = None,
    ) -> dict[str, Any]:
        """Map an evolution history window to a full Phase-1.5 action dict.

        Args:
            history: Merged state+reward dicts, oldest first, covering
                generations strictly before the action to take.
            encoder: Fitted encoder matching ``input_dim``
                (:class:`controller.StateEncoder` or
                :class:`controller.ProblemAwareEncoder`).
            pm_min: Lower bound for the mutation probability (> 0). Used
                only in ``mutation_target="absolute"`` mode.
            pm_max: Upper bound for the mutation probability. Used only in
                ``mutation_target="absolute"`` mode.
            n_vars: Decision-variable count of the problem being solved.
                Required in ``mutation_target="multiplier"`` mode (the pm
                head output is then a log-multiplier that must be scaled by
                ``1 / n_vars``); ignored in ``"absolute"`` mode.

        Returns:
            Dict with keys ``"mutation_operator"`` (argmax of the operator
            probabilities, ``"polynomial"`` or ``"gaussian"``),
            ``"mutation_probability"``, and ``"exploration_strength"``
            (``exp`` of the log-prediction clipped to
            :data:`POLYNOMIAL_EXPLORATION_RANGE` when the chosen operator
            is polynomial, else :data:`GAUSSIAN_EXPLORATION_RANGE`). In
            ``"absolute"`` mode the mutation probability is ``exp`` of the
            log-prediction clipped to ``[pm_min, pm_max]``; in
            ``"multiplier"`` mode the log-prediction is clipped to
            ``[log 0.25, log 8.0]`` (see :data:`MULTIPLIER_RANGE`) and the
            probability is ``exp(prediction) / n_vars``.

        Raises:
            ValueError: If the pm bounds are invalid (absolute mode), or
                if ``mutation_target="multiplier"`` and ``n_vars`` is
                missing or < 1.
        """
        features = encoder.transform(history).reshape(1, -1)
        op_probs, log_pm, log_expl = self.predict(features)
        operator = OPERATOR_CLASSES[int(np.argmax(op_probs[0]))]
        if self._mutation_target == "multiplier":
            if n_vars is None:
                raise ValueError(
                    "n_vars is required when mutation_target='multiplier'"
                )
            log_multiplier = _clip_log(
                float(log_pm[0]), MULTIPLIER_RANGE[0], MULTIPLIER_RANGE[1]
            )
            pm = pm_from_log_multiplier(log_multiplier, n_vars)
        else:
            pm = math.exp(_clip_log(float(log_pm[0]), pm_min, pm_max))
        lo, hi = (
            POLYNOMIAL_EXPLORATION_RANGE
            if operator == "polynomial"
            else GAUSSIAN_EXPLORATION_RANGE
        )
        exploration = math.exp(_clip_log(float(log_expl[0]), lo, hi))
        return {
            "mutation_operator": operator,
            "mutation_probability": float(pm),
            "exploration_strength": float(exploration),
        }

    def save(self, path: str | Path) -> None:
        """Serialize config and weights with ``torch.save``.

        Args:
            path: Destination path; parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "input_dim": self._input_dim,
                "hidden_dims": list(self._hidden_dims),
                "seed": self._seed,
                "lr": self._lr,
                "name": self.name,
                "mutation_target": self._mutation_target,
            },
            "state_dict": self._net.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "MultiHeadController":
        """Load a controller saved with :meth:`save`.

        Checkpoints written before Phase 1.75 have no ``mutation_target``
        key; they load as ``"absolute"`` controllers, preserving the exact
        Phase-1.5 behavior.

        Args:
            path: Path to the file written by :meth:`save`.

        Returns:
            The controller with restored weights, in eval mode.
        """
        payload = torch.load(Path(path), map_location="cpu")
        config = payload["config"]
        controller = cls(
            input_dim=int(config["input_dim"]),
            hidden_dims=tuple(config["hidden_dims"]),
            seed=int(config["seed"]),
            lr=float(config["lr"]),
            name=str(config.get("name", "mlp2")),
            mutation_target=str(config.get("mutation_target", "absolute")),
        )
        controller._net.load_state_dict(payload["state_dict"])
        controller._net.eval()
        return controller
