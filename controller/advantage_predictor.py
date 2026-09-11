from __future__ import annotations

"""Torch MLP predicting an action's *advantage* instead of absolute future HV.

Phase 2.75, Task 4. Phase 2B's action-conditional diagnostics showed that an
:class:`controller.outcome_predictor.OutcomePredictor` trained on absolute
future hypervolume is effectively a ``state -> future HV`` regressor: shuffling
the four action features barely moves its held-out R^2, and its within-state
action ranking correlates *negatively* with the realized outcome. The learning
target, not the model capacity, is the bottleneck.

:class:`AdvantagePredictor` therefore regresses the **advantage** of a candidate
action — its outcome minus a per-state baseline, built by
``experiments/build_intervention_dataset.py`` from same-snapshot intervention
data — and can additionally be trained **contrastively**: paired
action-shuffled negatives add a hinge ranking term that forces the network to
use the action channel at all (the exact failure mode D1 measured).

Two module-level helpers are reused by Phase 2.75 experiment code:

* :func:`hinge_ranking_loss` — the contrastive term itself;
* :func:`ranking_metrics` — decision-quality metrics (Spearman/Kendall, oracle
  hit rate, regret, oracle gap) grouped by state.
"""

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["AdvantagePredictor", "hinge_ranking_loss", "ranking_metrics"]

#: Default advantage horizons (Phase 2.75 intervention grid).
DEFAULT_ADVANTAGE_HORIZONS: tuple[int, ...] = (5, 10, 20)
#: Minimum candidates per state for a rank correlation to be defined.
MIN_GROUP_SIZE = 3


def hinge_ranking_loss(
    pred_pos: torch.Tensor, pred_neg: torch.Tensor, margin: float = 0.0
) -> torch.Tensor:
    """Elementwise hinge ``relu(margin - (pred_pos - pred_neg))``.

    Paired contrastive term: row ``i`` of ``pred_neg`` is the prediction for
    the same state with a shuffled (unfavourable) action, so every positive
    entry where the negative is not beaten by ``margin`` still contributes
    gradient and the network is pushed to separate the two.

    Args:
        pred_pos: Positive-branch predictions, any shape.
        pred_neg: Negative-branch predictions, same shape.
        margin: Required prediction gap; ``0.0`` only asks for a strict
            ordering.

    Returns:
        Tensor of the same shape as the inputs, ``>= 0``.

    Raises:
        ValueError: If the two tensors have different shapes.
    """
    if pred_pos.shape != pred_neg.shape:
        raise ValueError(
            f"pred_pos shape {tuple(pred_pos.shape)} does not match "
            f"pred_neg shape {tuple(pred_neg.shape)}"
        )
    return torch.clamp(float(margin) - (pred_pos - pred_neg), min=0.0)


def _as_group_matrix(values: Any, name: str) -> np.ndarray:
    """Coerce scores to a ``(n_groups, n_candidates)`` float64 matrix.

    Raises:
        ValueError: If the array is not 1-D or 2-D.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {array.shape}")
    return array


def _scipy_stat(result: Any) -> float | None:
    """Finite ``statistic``/``correlation`` of a scipy result, else ``None``."""
    value = getattr(result, "statistic", None)
    if value is None:
        value = getattr(result, "correlation", None)
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def ranking_metrics(pred: np.ndarray, realized: np.ndarray) -> dict[str, Any]:
    """Decision-quality metrics of predicted vs realized action scores.

    Rows are states and columns are candidate actions (a 1-D input is one
    group). Per state:

    * ``spearman``/``kendall`` — rank correlation between prediction and
      realization (skipped when fewer than :data:`MIN_GROUP_SIZE`
      candidates or either side is constant);
    * ``oracle_hit`` — ``argmax(pred) == argmax(realized)`` (ties resolve to
      the earliest candidate, ``np.argmax`` semantics);
    * ``regret`` — ``realized(best) - realized(chosen)``, so a state where
      the chosen action is optimal contributes 0;
    * ``oracle_gap`` — ``realized(best) - mean(realized)``, the headroom a
      state offers over an average candidate (prediction-independent).

    Args:
        pred: Predicted scores, shape ``(n_states, n_candidates)`` or
            ``(n_candidates,)``.
        realized: Realized scores, same shape as ``pred``.

    Returns:
        Dict with ``n_groups``, the means/stds of both correlations, the
        fraction of defined correlations below 0.05, ``oracle_hit_rate``,
        ``regret_mean``/``regret_median`` and ``oracle_gap_mean``. Entries
        that are undefined are ``None``.

    Raises:
        ValueError: If the inputs are not 1-D/2-D or their shapes differ.
    """
    pred_matrix = _as_group_matrix(pred, "pred")
    realized_matrix = _as_group_matrix(realized, "realized")
    if pred_matrix.shape != realized_matrix.shape:
        raise ValueError(
            f"pred shape {pred_matrix.shape} does not match realized shape "
            f"{realized_matrix.shape}"
        )
    spearman: list[float] = []
    kendall: list[float] = []
    p_values: list[float] = []
    hits: list[float] = []
    regrets: list[float] = []
    gaps: list[float] = []
    for row_pred, row_real in zip(pred_matrix, realized_matrix):
        best = int(np.argmax(row_real))
        chosen = int(np.argmax(row_pred))
        hits.append(1.0 if best == chosen else 0.0)
        regrets.append(float(row_real[best] - row_real[chosen]))
        gaps.append(float(row_real[best] - row_real.mean()))
        if row_pred.size < MIN_GROUP_SIZE:
            continue
        if np.all(row_pred == row_pred[0]) or np.all(row_real == row_real[0]):
            continue
        spearman_result = stats.spearmanr(row_pred, row_real)
        kendall_result = stats.kendalltau(row_pred, row_real)
        rho = _scipy_stat(spearman_result)
        tau = _scipy_stat(kendall_result)
        p_value = getattr(spearman_result, "pvalue", None)
        if rho is not None:
            spearman.append(rho)
        if tau is not None:
            kendall.append(tau)
        if p_value is not None and np.isfinite(float(p_value)):
            p_values.append(float(p_value))
    return {
        "n_groups": int(pred_matrix.shape[0]),
        "spearman_mean": float(np.mean(spearman)) if spearman else None,
        "spearman_std": (
            float(np.std(spearman, ddof=1)) if len(spearman) > 1 else 0.0
            if spearman
            else None
        ),
        "kendall_mean": float(np.mean(kendall)) if kendall else None,
        "kendall_std": (
            float(np.std(kendall, ddof=1)) if len(kendall) > 1 else 0.0
            if kendall
            else None
        ),
        "significant_fraction": (
            float(np.mean([1.0 if p < 0.05 else 0.0 for p in p_values]))
            if p_values
            else None
        ),
        "oracle_hit_rate": float(np.mean(hits)) if hits else None,
        "regret_mean": float(np.mean(regrets)) if regrets else None,
        "regret_median": float(np.median(regrets)) if regrets else None,
        "oracle_gap_mean": float(np.mean(gaps)) if gaps else None,
    }


class AdvantagePredictor:
    """Feed-forward predictor of per-horizon action advantage.

    Structurally identical to
    :class:`controller.outcome_predictor.OutcomePredictor` (``Linear -> ReLU``
    stack plus a ``len(horizons)``-wide linear head, sample-weighted MSE,
    Adam, seeded batching), with two Phase-2.75 changes:

    * the target is an advantage (intervention data produced by
      ``experiments/build_intervention_dataset.py``) rather than an absolute
      future HV, and
    * :meth:`fit` accepts paired action-shuffled negatives and adds
      :func:`hinge_ranking_loss` to the objective, so the action channel
      cannot be ignored while the MSE term is fitted by the state block.

    Attributes:
        name: Identifier prefix for experiment bookkeeping.
    """

    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int] = DEFAULT_ADVANTAGE_HORIZONS,
        hidden_dims: Sequence[int] = (128, 128),
        seed: int = 0,
        lr: float = 1e-3,
    ) -> None:
        """Initialize the predictor and reseed torch/numpy RNGs.

        Args:
            input_dim: Flattened input dimension (``encoder.dim + 4`` for
                samples built from the intervention dataset).
            horizons: Advantage horizons; fixes the output width.
            hidden_dims: Hidden layer widths.
            seed: Random seed applied to ``torch.manual_seed`` and
                ``np.random.seed``.
            lr: Adam learning rate.

        Raises:
            ValueError: If ``input_dim`` < 1, ``horizons`` is empty, or a
                horizon is < 1.
        """
        if int(input_dim) < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        horizon_list = [int(h) for h in horizons]
        if not horizon_list:
            raise ValueError("horizons must be a non-empty list of generation offsets")
        if min(horizon_list) < 1:
            raise ValueError(f"every horizon must be >= 1, got {horizon_list}")
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        self._input_dim = int(input_dim)
        self._horizons = tuple(horizon_list)
        self._hidden_dims = tuple(int(h) for h in hidden_dims)
        self._seed = int(seed)
        self._lr = float(lr)
        self._margin = 0.0
        layers: list[nn.Module] = []
        previous = self._input_dim
        for width in self._hidden_dims:
            layers.append(nn.Linear(previous, width))
            layers.append(nn.ReLU())
            previous = width
        layers.append(nn.Linear(previous, len(self._horizons)))
        self._model = nn.Sequential(*layers)
        self._criterion = nn.MSELoss(reduction="none")
        self.name = f"advantage_h{'_'.join(str(h) for h in self._horizons)}"

    @property
    def input_dim(self) -> int:
        """Flattened input dimension."""
        return self._input_dim

    @property
    def horizons(self) -> tuple[int, ...]:
        """Advantage horizons."""
        return self._horizons

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        """Hidden layer widths."""
        return self._hidden_dims

    @property
    def margin(self) -> float:
        """Hinge margin used by the last :meth:`fit` call."""
        return self._margin

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        epochs: int = 300,
        batch_size: int = 256,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        X_neg: np.ndarray | None = None,
        y_neg: np.ndarray | None = None,
        margin: float = 0.0,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train with sample-weighted MSE, optionally plus a hinge term.

        Batches are drawn with a torch generator seeded by the predictor's
        seed, so repeated runs with the same seed and data produce identical
        parameters. When ``X_neg`` is given, row ``i`` must be the
        action-shuffled counterpart of row ``i`` (same state, different
        action) and the objective becomes::

            weighted MSE(pred_pos, y) + mean(relu(margin - (pred_pos - pred_neg)))

        where the hinge is averaged over rows and horizons. ``y_neg`` is
        accepted for interface symmetry with the paired dataset and is
        shape-checked, but the hinge term is preference-based: it only
        requires positives to outrank their paired negatives, it does not
        regress towards ``y_neg``.

        Args:
            X: Positive feature matrix, shape ``(n, input_dim)``.
            y: Advantage targets, shape ``(n, len(horizons))`` (1-D is
                accepted for a single horizon).
            sample_weight: Optional nonnegative weights, shape ``(n,)``.
            epochs: Number of training epochs.
            batch_size: Mini-batch size (clamped to ``n``).
            X_val: Optional validation features.
            y_val: Optional validation targets.
            X_neg: Optional paired negative features, shape ``(n, input_dim)``.
            y_neg: Optional paired negative targets, shape
                ``(n, len(horizons))``; must accompany ``X_neg``.
            margin: Hinge margin of the contrastive term.
            verbose: Print progress every 10 epochs when True.

        Returns:
            ``{"train_loss": [...], "val_loss": [...]}`` with one entry per
            epoch (``val_loss`` is empty unless ``X_val``/``y_val`` were
            given).

        Raises:
            ValueError: If shapes are inconsistent, the sample count is
                zero, or only one of ``X_neg``/``y_neg`` is provided.
        """
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X_t.ndim == 1:
            X_t = X_t.reshape(1, -1)
        n_samples = int(X_t.shape[0])
        if n_samples == 0:
            raise ValueError("cannot fit AdvantagePredictor on zero samples")
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.ndim != 2 or y_arr.shape[1] != len(self._horizons):
            raise ValueError(
                f"y must have {len(self._horizons)} columns (one per horizon), "
                f"got shape {y_arr.shape}"
            )
        if y_arr.shape[0] != n_samples:
            raise ValueError(
                f"X has {n_samples} samples but y has {y_arr.shape[0]}"
            )
        y_t = torch.as_tensor(y_arr, dtype=torch.float32)
        if sample_weight is None:
            w_t = torch.ones(n_samples)
        else:
            w_t = torch.as_tensor(
                np.asarray(sample_weight, dtype=np.float32)
            ).view(-1)
            if w_t.shape[0] != n_samples:
                raise ValueError(
                    f"sample_weight has {w_t.shape[0]} entries for "
                    f"{n_samples} samples"
                )
        if (X_neg is None) != (y_neg is None):
            raise ValueError(
                "X_neg and y_neg must be provided together (paired "
                "action-shuffled negatives)"
            )
        X_neg_t: torch.Tensor | None = None
        if X_neg is not None and y_neg is not None:
            X_neg_t = torch.as_tensor(np.asarray(X_neg, dtype=np.float32))
            if X_neg_t.ndim == 1:
                X_neg_t = X_neg_t.reshape(1, -1)
            if X_neg_t.shape != X_t.shape:
                raise ValueError(
                    f"X_neg shape {tuple(X_neg_t.shape)} does not match X "
                    f"shape {tuple(X_t.shape)}"
                )
            y_neg_arr = np.asarray(y_neg, dtype=np.float64)
            if y_neg_arr.ndim == 1:
                y_neg_arr = y_neg_arr.reshape(-1, 1)
            if y_neg_arr.shape != y_arr.shape:
                raise ValueError(
                    f"y_neg shape {y_neg_arr.shape} does not match y shape "
                    f"{y_arr.shape}"
                )
        self._margin = float(margin)

        if X_neg_t is None:
            dataset: TensorDataset = TensorDataset(X_t, y_t, w_t)
        else:
            dataset = TensorDataset(X_t, y_t, w_t, X_neg_t)
        generator = torch.Generator().manual_seed(self._seed)
        loader = DataLoader(
            dataset,
            batch_size=min(int(batch_size), n_samples),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)

        has_val = X_val is not None and y_val is not None
        X_val_t = (
            torch.as_tensor(np.asarray(X_val, dtype=np.float32))
            if has_val
            else None
        )
        y_val_t = (
            torch.as_tensor(np.asarray(y_val, dtype=np.float32)).reshape(
                -1, len(self._horizons)
            )
            if has_val
            else None
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        for epoch in range(int(epochs)):
            self._model.train()
            for batch in loader:
                xb, yb, wb = batch[0], batch[1], batch[2]
                pred = self._model(xb)
                squared_error = self._criterion(pred, yb)
                loss = torch.sum(wb.unsqueeze(1) * squared_error) / (
                    torch.sum(wb) * squared_error.size(1)
                )
                if X_neg_t is not None:
                    pred_neg = self._model(batch[3])
                    loss = loss + torch.mean(
                        hinge_ranking_loss(pred, pred_neg, self._margin)
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            history["train_loss"].append(self._weighted_mse(X_t, y_t, w_t))
            if has_val:
                history["val_loss"].append(self._plain_mse(X_val_t, y_val_t))
            if verbose and (epoch % 10 == 0 or epoch == int(epochs) - 1):
                message = (
                    f"epoch {epoch + 1}/{int(epochs)} "
                    f"train_loss={history['train_loss'][-1]:.6f}"
                )
                if has_val:
                    message += f" val_loss={history['val_loss'][-1]:.6f}"
                print(message)
        return history

    def _weighted_mse(
        self, X_t: torch.Tensor, y_t: torch.Tensor, w_t: torch.Tensor
    ) -> float:
        """Full-set weighted MSE in eval mode (hinge term excluded)."""
        self._model.eval()
        with torch.no_grad():
            squared_error = self._criterion(self._model(X_t), y_t)
            return float(
                torch.sum(w_t.unsqueeze(1) * squared_error)
                / (torch.sum(w_t) * squared_error.size(1))
            )

    def _plain_mse(self, X_t: torch.Tensor, y_t: torch.Tensor) -> float:
        """Full-set unweighted MSE in eval mode."""
        self._model.eval()
        with torch.no_grad():
            return float(torch.mean(self._criterion(self._model(X_t), y_t)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict per-horizon advantages.

        Args:
            X: Feature matrix of shape ``(n, input_dim)`` (1-D is treated as
                a single sample).

        Returns:
            Predictions of shape ``(n, len(horizons))`` as float64; column
            ``k`` belongs to ``horizons[k]``.
        """
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X_t.ndim == 1:
            X_t = X_t.reshape(1, -1)
        self._model.eval()
        with torch.no_grad():
            prediction = self._model(X_t)
        return prediction.numpy().astype(np.float64)

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
                "horizons": list(self._horizons),
                "hidden_dims": list(self._hidden_dims),
                "seed": self._seed,
                "lr": self._lr,
                "margin": self._margin,
                "name": self.name,
            },
            "state_dict": self._model.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "AdvantagePredictor":
        """Load a predictor saved with :meth:`save` (in eval mode).

        Args:
            path: Path to the file written by :meth:`save`.

        Returns:
            The predictor with restored weights.
        """
        payload = torch.load(Path(path), map_location="cpu")
        config = payload["config"]
        predictor = cls(
            input_dim=int(config["input_dim"]),
            horizons=list(config["horizons"]),
            hidden_dims=tuple(config["hidden_dims"]),
            seed=int(config["seed"]),
            lr=float(config["lr"]),
        )
        predictor.name = str(config.get("name", predictor.name))
        predictor._margin = float(config.get("margin", 0.0))
        predictor._model.load_state_dict(payload["state_dict"])
        predictor._model.eval()
        return predictor
