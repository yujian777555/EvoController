from __future__ import annotations

"""Torch MLP predicting future hypervolume from (history, candidate action).

Phase-2 learning formulation. Phase 1.75 showed that imitating recorded
actions carries no causal decision signal, so the learning target moves
from the action itself to the *outcome* of the action: given the encoded
state history concatenated with a candidate action (exactly the sample
layout produced by ``controller.dataset.build_outcome_samples``), the
:class:`OutcomePredictor` regresses the absolute hypervolume at each of
the configured future horizons. A learned, action-conditioned outcome
model is the prerequisite for the planning-based controller — candidate
actions can be scored by their predicted long-run HV before being
executed on the real population.

The network is a stack of ``Linear -> ReLU`` blocks followed by a
``len(horizons)``-dimensional linear head, trained with (optionally
sample-weighted) MSE and Adam. Given the same seed and data,
construction and training are deterministic.
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["OutcomePredictor"]


class OutcomePredictor:
    """Feed-forward predictor of future-HV vectors.

    Attributes:
        name: Identifier prefix for experiment bookkeeping.
    """

    def __init__(
        self,
        input_dim: int,
        horizons: list[int] = [1, 5, 10, 20],
        hidden_dims: Sequence[int] = (128, 128),
        seed: int = 0,
        lr: float = 1e-3,
    ) -> None:
        """Initialize the predictor and reseed torch/numpy RNGs.

        Args:
            input_dim: Flattened input dimension; for samples from
                :func:`controller.dataset.build_outcome_samples` this is
                ``encoder.dim + 4``.
            horizons: Future generation offsets being predicted; fixes the
                output dimension ``len(horizons)``.
            hidden_dims: Hidden layer widths.
            seed: Random seed; applied to ``torch.manual_seed`` and
                ``np.random.seed``.
            lr: Adam learning rate.

        Raises:
            ValueError: If ``input_dim`` < 1, ``horizons`` is empty, or a
                horizon is < 1.
        """
        if int(input_dim) < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        horizons = [int(h) for h in horizons]
        if not horizons:
            raise ValueError("horizons must be a non-empty list of generation offsets")
        if min(horizons) < 1:
            raise ValueError(f"every horizon must be >= 1, got {horizons}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        self._input_dim = int(input_dim)
        self._horizons = tuple(horizons)
        self._hidden_dims = tuple(int(h) for h in hidden_dims)
        self._seed = int(seed)
        self._lr = float(lr)
        layers: list[nn.Module] = []
        prev = self._input_dim
        for width in self._hidden_dims:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, len(self._horizons)))
        self._model = nn.Sequential(*layers)
        self._criterion = nn.MSELoss(reduction="none")
        self.name = f"outcome_h{'_'.join(str(h) for h in self._horizons)}"

    @property
    def input_dim(self) -> int:
        """Flattened input dimension."""
        return self._input_dim

    @property
    def horizons(self) -> tuple[int, ...]:
        """Future generation offsets being predicted."""
        return self._horizons

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        """Hidden layer widths."""
        return self._hidden_dims

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        epochs: int = 300,
        batch_size: int = 256,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train the network with sample-weighted MSE and Adam.

        Batches are drawn with a torch generator seeded by the
        predictor's seed, so repeated runs with the same seed and data
        produce identical parameters. Reported epoch losses are full-set
        metrics: weighted MSE for the training set, plain MSE for the
        validation set (no validation weights are part of the
        interface).

        Args:
            X: Feature matrix of shape ``(n, input_dim)``.
            y: Absolute future-HV targets of shape ``(n, len(horizons))``
                (a 1-D array is accepted when there is a single horizon).
            sample_weight: Optional nonnegative weights of shape ``(n,)``;
                uniform weights when omitted.
            epochs: Number of training epochs.
            batch_size: Mini-batch size (clamped to ``n``).
            X_val: Optional validation features of shape ``(m, input_dim)``.
            y_val: Optional validation targets of shape
                ``(m, len(horizons))``.
            verbose: Print progress every 10 epochs when True.

        Returns:
            ``{"train_loss": [...], "val_loss": [...]}`` with one entry
            per epoch; ``val_loss`` is empty unless both ``X_val`` and
            ``y_val`` were given.

        Raises:
            ValueError: If ``X`` contains no samples, ``y`` does not have
                one column per horizon and one row per sample, or
                ``sample_weight`` does not match the sample count.
        """
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X_t.ndim == 1:
            X_t = X_t.reshape(1, -1)
        n_samples = int(X_t.shape[0])
        if n_samples == 0:
            raise ValueError("cannot fit OutcomePredictor on zero samples")
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
            for xb, yb, wb in loader:
                pred = self._model(xb)
                sq_err = self._criterion(pred, yb)
                loss = torch.sum(wb.unsqueeze(1) * sq_err) / (
                    torch.sum(wb) * sq_err.size(1)
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
        """Full-set weighted MSE in eval mode."""
        self._model.eval()
        with torch.no_grad():
            sq_err = self._criterion(self._model(X_t), y_t)
            return float(
                torch.sum(w_t.unsqueeze(1) * sq_err)
                / (torch.sum(w_t) * sq_err.size(1))
            )

    def _plain_mse(self, X_t: torch.Tensor, y_t: torch.Tensor) -> float:
        """Full-set unweighted MSE in eval mode."""
        self._model.eval()
        with torch.no_grad():
            return float(torch.mean(self._criterion(self._model(X_t), y_t)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict future-HV vectors.

        Args:
            X: Feature matrix of shape ``(n, input_dim)`` (a 1-D array is
                treated as a single sample).

        Returns:
            Predictions of shape ``(n, len(horizons))`` as float64; column
            ``k`` corresponds to ``horizons[k]``.
        """
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if X_t.ndim == 1:
            X_t = X_t.reshape(1, -1)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(X_t)
        return pred.numpy().astype(np.float64)

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
                "name": self.name,
            },
            "state_dict": self._model.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "OutcomePredictor":
        """Load a predictor saved with :meth:`save`.

        Args:
            path: Path to the file written by :meth:`save`.

        Returns:
            The predictor with restored weights, in eval mode.
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
        predictor._model.load_state_dict(payload["state_dict"])
        predictor._model.eval()
        return predictor
