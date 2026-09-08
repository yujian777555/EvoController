from __future__ import annotations

"""Torch controllers mapping evolution history to mutation probability.

Both controllers regress in log space: the target is ``log`` of the
mutation probability, and :meth:`predict_action` maps a history window to
a probability via ``exp(clip(prediction, log(pm_min), log(pm_max)))``.
``MLPController`` is the Phase 1 learned baseline; ``ConstantController``
is the history-free ablation (it predicts the weighted mean of the
training targets regardless of input).
"""

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from controller.state_encoder import STATE_FEATURES, StateEncoder


def _clip_log_pm(log_pm: float, pm_min: float, pm_max: float) -> float:
    """Clip a log-space mutation probability to ``[pm_min, pm_max]``.

    Args:
        log_pm: Predicted log mutation probability.
        pm_min: Lower probability bound; must be > 0 (log-space target).
        pm_max: Upper probability bound; must be >= ``pm_min``.

    Returns:
        ``log_pm`` clipped to ``[log(pm_min), log(pm_max)]``.

    Raises:
        ValueError: If the bounds are invalid.
    """
    if pm_min <= 0.0:
        raise ValueError(f"pm_min must be > 0 for log-space target, got {pm_min}")
    if pm_max < pm_min:
        raise ValueError(f"pm_max ({pm_max}) must be >= pm_min ({pm_min})")
    return min(max(float(log_pm), math.log(pm_min)), math.log(pm_max))


class MLPController:
    """Feed-forward controller predicting log mutation probability.

    The network is a stack of ``Linear -> ReLU`` blocks followed by a
    scalar linear head, trained with sample-weighted MSE on log mutation
    probabilities using Adam. Given the same seed and data, construction
    and training are deterministic.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        seed: int = 0,
        lr: float = 1e-3,
    ) -> None:
        """Initialize the controller and reseed torch/numpy RNGs.

        Args:
            input_dim: Flattened input dimension, typically
                ``StateEncoder.dim`` (``window * 6``).
            hidden_dims: Hidden layer widths.
            seed: Random seed; applied to ``torch.manual_seed`` and
                ``np.random.seed``.
            lr: Adam learning rate.

        Raises:
            ValueError: If ``input_dim`` < 1.
        """
        if int(input_dim) < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        self._input_dim = int(input_dim)
        self._hidden_dims = tuple(int(h) for h in hidden_dims)
        self._seed = int(seed)
        self._lr = float(lr)
        layers: list[nn.Module] = []
        prev = self._input_dim
        for width in self._hidden_dims:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, 1))
        self._model = nn.Sequential(*layers)
        n_features = len(STATE_FEATURES)
        if self._input_dim % n_features == 0:
            self.name = f"mlp_w{self._input_dim // n_features}"
        else:
            self.name = "mlp"

    @property
    def input_dim(self) -> int:
        """Flattened input dimension."""
        return self._input_dim

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        """Hidden layer widths."""
        return self._hidden_dims

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        epochs: int = 200,
        batch_size: int = 256,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train the network with sample-weighted MSE and Adam.

        Batches are drawn with a torch generator seeded by the
        controller's seed, so repeated runs with the same seed and data
        produce identical parameters. Reported epoch losses are full-set
        metrics: weighted MSE for the training set, plain MSE for the
        validation set (no validation weights are part of the interface).

        Args:
            X: Feature matrix of shape ``(n, input_dim)``.
            y: Log-space targets of shape ``(n,)``.
            sample_weight: Optional nonnegative weights of shape ``(n,)``;
                uniform weights when omitted.
            epochs: Number of training epochs.
            batch_size: Mini-batch size (clamped to ``n``).
            X_val: Optional validation features of shape ``(m, input_dim)``.
            y_val: Optional validation targets of shape ``(m,)``.
            verbose: Print progress every 10 epochs when True.

        Returns:
            ``{"train_loss": [...], "val_loss": [...]}`` with one entry
            per epoch; ``val_loss`` is empty unless both ``X_val`` and
            ``y_val`` were given.

        Raises:
            ValueError: If ``X`` contains no samples.
        """
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X_t.ndim == 1:
            X_t = X_t.reshape(1, -1)
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float32)).view(-1)
        n_samples = int(X_t.shape[0])
        if n_samples == 0:
            raise ValueError("cannot fit MLPController on zero samples")
        if sample_weight is None:
            w_t = torch.ones(n_samples)
        else:
            w_t = torch.as_tensor(np.asarray(sample_weight, dtype=np.float32)).view(-1)

        dataset = TensorDataset(X_t, y_t, w_t)
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
            torch.as_tensor(np.asarray(X_val, dtype=np.float32)) if has_val else None
        )
        y_val_t = (
            torch.as_tensor(np.asarray(y_val, dtype=np.float32)).view(-1)
            if has_val
            else None
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        for epoch in range(int(epochs)):
            self._model.train()
            for xb, yb, wb in loader:
                pred = self._model(xb).view(-1)
                loss = torch.sum(wb * (pred - yb) ** 2) / torch.sum(wb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            history["train_loss"].append(self._weighted_mse(X_t, y_t, w_t))
            if has_val:
                history["val_loss"].append(self._plain_mse(X_val_t, y_val_t))
            if verbose and (epoch % 10 == 0 or epoch == int(epochs) - 1):
                message = f"epoch {epoch + 1}/{int(epochs)} train_loss={history['train_loss'][-1]:.6f}"
                if has_val:
                    message += f" val_loss={history['val_loss'][-1]:.6f}"
                print(message)
        return history

    def _weighted_mse(
        self, X_t: torch.Tensor, y_t: torch.Tensor, w_t: torch.Tensor
    ) -> float:
        """Full-set weighted MSE in eval mode."""
        self._model.eval()
        with torch.no_grad():
            pred = self._model(X_t).view(-1)
            return float(torch.sum(w_t * (pred - y_t) ** 2) / torch.sum(w_t))

    def _plain_mse(self, X_t: torch.Tensor, y_t: torch.Tensor) -> float:
        """Full-set unweighted MSE in eval mode."""
        self._model.eval()
        with torch.no_grad():
            pred = self._model(X_t).view(-1)
            return float(torch.mean((pred - y_t) ** 2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict log mutation probabilities.

        Args:
            X: Feature matrix of shape ``(n, input_dim)`` (a 1-D array is
                treated as a single sample).

        Returns:
            Predictions of shape ``(n,)`` in log space.
        """
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X_t.ndim == 1:
            X_t = X_t.reshape(1, -1)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(X_t).view(-1)
        return pred.numpy().astype(float)

    def predict_action(
        self,
        history: list[dict[str, Any]],
        encoder: StateEncoder,
        pm_min: float,
        pm_max: float,
    ) -> float:
        """Map an evolution history window to a mutation probability.

        Args:
            history: Merged state+reward dicts, oldest first, covering
                generations strictly before the action to take.
            encoder: Fitted encoder matching ``input_dim``.
            pm_min: Lower bound for the returned probability (> 0).
            pm_max: Upper bound for the returned probability.

        Returns:
            ``exp(clip(predict(history), log(pm_min), log(pm_max)))``.
        """
        features = encoder.transform(history).reshape(1, -1)
        log_pm = float(self.predict(features)[0])
        return float(math.exp(_clip_log_pm(log_pm, pm_min, pm_max)))

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
            },
            "state_dict": self._model.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "MLPController":
        """Load a controller saved with :meth:`save`.

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
        )
        controller._model.load_state_dict(payload["state_dict"])
        controller._model.eval()
        return controller


class ConstantController:
    """History-free baseline predicting a constant log mutation probability.

    The constant is the (sample-weighted) mean of the training targets,
    i.e. the best history-independent prediction under weighted MSE.
    """

    name = "constant"

    def __init__(self) -> None:
        """Initialize an unfitted controller."""
        self._constant: float | None = None

    def fit(
        self, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "ConstantController":
        """Set the constant to the weighted mean of the targets.

        Args:
            y: Log-space targets of shape ``(n,)``.
            sample_weight: Optional nonnegative weights of shape ``(n,)``;
                plain mean when omitted.

        Returns:
            The fitted controller (``self``).

        Raises:
            ValueError: If ``y`` is empty, the weights do not match ``y``,
                or the weights sum to zero.
        """
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if y_arr.size == 0:
            raise ValueError("cannot fit ConstantController on zero targets")
        if sample_weight is None:
            self._constant = float(y_arr.mean())
            return self
        w_arr = np.asarray(sample_weight, dtype=float).reshape(-1)
        if w_arr.shape != y_arr.shape:
            raise ValueError(
                f"sample_weight shape {w_arr.shape} does not match y shape {y_arr.shape}"
            )
        total = float(w_arr.sum())
        if total <= 0.0:
            raise ValueError("sample weights must sum to a positive value")
        self._constant = float(np.dot(w_arr, y_arr) / total)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the constant prediction once per row of ``X``.

        Args:
            X: Feature matrix whose first dimension gives the row count.

        Returns:
            Array of shape ``(n,)`` filled with the fitted constant.

        Raises:
            RuntimeError: If the controller has not been fitted.
        """
        self._require_fitted()
        n_rows = int(np.asarray(X).shape[0])
        return np.full(n_rows, self._constant)

    def predict_action(
        self,
        history: list[dict[str, Any]],
        encoder: StateEncoder,
        pm_min: float,
        pm_max: float,
    ) -> float:
        """Return the clipped constant, ignoring the history.

        Args:
            history: Ignored; present for interface compatibility.
            encoder: Ignored; present for interface compatibility.
            pm_min: Lower bound for the returned probability (> 0).
            pm_max: Upper bound for the returned probability.

        Returns:
            ``exp(clip(constant, log(pm_min), log(pm_max)))``.

        Raises:
            RuntimeError: If the controller has not been fitted.
        """
        self._require_fitted()
        return float(math.exp(_clip_log_pm(self._constant, pm_min, pm_max)))

    def save(self, path: str | Path) -> None:
        """Serialize the fitted constant to a JSON file.

        Args:
            path: Destination path; parent directories are created.

        Raises:
            RuntimeError: If the controller has not been fitted.
        """
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"name": self.name, "constant": self._constant}
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ConstantController":
        """Load a controller saved with :meth:`save`.

        Args:
            path: Path to the JSON file written by :meth:`save`.

        Returns:
            The fitted controller.
        """
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        controller = cls()
        controller._constant = float(payload["constant"])
        return controller

    def _require_fitted(self) -> None:
        """Raise if the controller has no fitted constant."""
        if self._constant is None:
            raise RuntimeError("ConstantController must be fitted first")
