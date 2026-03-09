"""
deep_learning.py
────────────────
Deep Learning models for disease prediction from tabular EHR data.

Models implemented:
  • MLP         — Multi-Layer Perceptron with batch norm, dropout, residual skip
  • TabNet       — Attentive interpretable tabular learning (Arik & Pfister, 2021)
  • FT-Transformer — Feature-Tokenizer + Transformer (Gorishniy et al., 2021)
  • LSTM         — Long Short-Term Memory (for longitudinal / sequential EHR)
  • 1D-CNN       — Convolutional network treating feature vector as a 1D signal

All models follow a unified scikit-learn-compatible interface:
  fit(X_train, y_train, X_val, y_val)
  predict_proba(X)
  predict(X)

Dependencies: torch, pytorch-tabnet
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Shared training utilities
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss does not improve for `patience` epochs."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


def _to_tensor(X, y=None, device=DEVICE):
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    if y is not None:
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        return X_t, y_t
    return X_t


def _make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return torch.tensor([n_neg / (n_pos + 1e-9)], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────────────────────────────────────

class _MLPNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        dropout: float,
        batch_norm: bool,
        activation: str,
    ):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU,
                  "gelu": nn.GELU, "selu": nn.SELU}[activation]

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class MLPModel:
    """
    Multi-Layer Perceptron for binary classification.

    Parameters
    ----------
    hidden_dims  : list[int]   Hidden layer widths.
    dropout      : float       Dropout rate.
    batch_norm   : bool        Apply batch normalisation after each layer.
    activation   : str         'relu' | 'leaky_relu' | 'gelu' | 'selu'
    lr           : float       Learning rate (Adam).
    weight_decay : float       L2 regularisation.
    batch_size   : int
    max_epochs   : int
    patience     : int         Early stopping patience.
    device       : str | None  'cpu', 'cuda', 'mps'; None = auto-detect.
    """

    name = "MLP"

    def __init__(
        self,
        hidden_dims: list[int] = (256, 128, 64),
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.hidden_dims  = list(hidden_dims)
        self.dropout      = dropout
        self.batch_norm   = batch_norm
        self.activation   = activation
        self.lr           = lr
        self.weight_decay = weight_decay
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.device       = torch.device(device) if device else DEVICE
        self.random_state = random_state
        self.model_       = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MLPModel":
        torch.manual_seed(self.random_state)
        in_dim = X_train.shape[1]

        self.model_ = _MLPNet(
            in_dim, self.hidden_dims, self.dropout,
            self.batch_norm, self.activation,
        ).to(self.device)

        pos_weight = _compute_pos_weight(y_train).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.Adam(self.model_.parameters(),
                                      lr=self.lr, weight_decay=self.weight_decay)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=False)

        train_loader = _make_loader(X_train, y_train, self.batch_size)
        stopper      = EarlyStopping(patience=self.patience)

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            train_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model_(Xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(yb)
            train_loss /= len(y_train)

            val_loss = train_loss
            if X_val is not None and y_val is not None:
                val_loss = self._eval_loss(X_val, y_val, criterion)

            scheduler.step(val_loss)

            if epoch % 20 == 0:
                logger.debug("[MLP] Epoch %d  train=%.4f  val=%.4f", epoch, train_loss, val_loss)

            if stopper(val_loss, self.model_):
                logger.info("[MLP] Early stop at epoch %d", epoch)
                stopper.restore_best(self.model_)
                break

        logger.info("[MLP] Training complete.")
        return self

    def _eval_loss(self, X, y, criterion):
        self.model_.eval()
        with torch.no_grad():
            Xt, yt = _to_tensor(X, y, self.device)
            loss   = criterion(self.model_(Xt), yt)
        return loss.item()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(_to_tensor(X, device=self.device)).cpu().numpy()
        prob_pos = torch.sigmoid(torch.tensor(logits)).numpy()
        return np.column_stack([1 - prob_pos, prob_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# TabNet
# ─────────────────────────────────────────────────────────────────────────────

class TabNetModel:
    """
    TabNet attentive tabular learner.

    Requires: pip install pytorch-tabnet

    Parameters
    ----------
    n_d, n_a   : int   Width of decision step and attention embedding.
    n_steps    : int   Number of sequential decision steps.
    gamma      : float Coefficient for feature reusage in masks.
    mask_type  : str   'sparsemax' | 'entmax'
    """

    name = "TabNet"

    def __init__(
        self,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 5,
        gamma: float = 1.3,
        momentum: float = 0.02,
        mask_type: str = "sparsemax",
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 256,
        lr: float = 2e-2,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError:
            raise ImportError("Install pytorch-tabnet: pip install pytorch-tabnet")

        dev = device or str(DEVICE)
        self.model_ = TabNetClassifier(
            n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
            momentum=momentum, mask_type=mask_type,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": lr},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 10, "gamma": 0.9},
            device_name=dev,
            seed=random_state,
            verbose=0,
        )
        self.max_epochs = max_epochs
        self.patience   = patience
        self.batch_size = batch_size

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TabNetModel":
        eval_set = [(X_val, y_val)] if X_val is not None else []
        self.model_.fit(
            X_train, y_train.astype(int),
            eval_set=eval_set,
            eval_metric=["auc"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.batch_size // 4,
        )
        logger.info("[TabNet] Training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_feature_importance(self) -> np.ndarray:
        return self.model_.feature_importances_


# ─────────────────────────────────────────────────────────────────────────────
# FT-Transformer
# ─────────────────────────────────────────────────────────────────────────────

class _FTTransformerNet(nn.Module):
    """Feature Tokenizer + Transformer for tabular data (Gorishniy et al., 2021)."""

    def __init__(
        self,
        in_dim: int,
        d_token: int = 64,
        n_blocks: int = 3,
        n_heads: int = 8,
        ffn_factor: float = 4.0 / 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = nn.Linear(in_dim, d_token)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=int(d_token * ffn_factor),
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.head        = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1),
        )

    def forward(self, x):
        tokens = self.tokenizer(x).unsqueeze(1)        # (B, 1, d_token)
        out    = self.transformer(tokens).squeeze(1)   # (B, d_token)
        return self.head(out).squeeze(1)               # (B,)


class FTTransformerModel:
    """
    Feature-Tokenizer Transformer for tabular binary classification.

    Reference: Gorishniy et al. (2021) "Revisiting Deep Learning Models for
    Tabular Data". NeurIPS 2021.
    """

    name = "FT-Transformer"

    def __init__(
        self,
        d_token: int = 64,
        n_blocks: int = 3,
        n_heads: int = 8,
        ffn_factor: float = 4.0 / 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.d_token      = d_token
        self.n_blocks     = n_blocks
        self.n_heads      = n_heads
        self.ffn_factor   = ffn_factor
        self.dropout      = dropout
        self.lr           = lr
        self.weight_decay = weight_decay
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.device       = torch.device(device) if device else DEVICE
        self.random_state = random_state
        self.model_       = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        torch.manual_seed(self.random_state)
        self.model_ = _FTTransformerNet(
            X_train.shape[1], self.d_token, self.n_blocks,
            self.n_heads, self.ffn_factor, self.dropout,
        ).to(self.device)

        pos_weight = _compute_pos_weight(y_train).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.AdamW(self.model_.parameters(),
                                       lr=self.lr, weight_decay=self.weight_decay)

        train_loader = _make_loader(X_train, y_train, self.batch_size)
        stopper      = EarlyStopping(patience=self.patience)

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                criterion(self.model_(Xb), yb).backward()
                optimizer.step()

            val_loss = 0.0
            if X_val is not None and y_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    Xv, yv = _to_tensor(X_val, y_val, self.device)
                    val_loss = criterion(self.model_(Xv), yv).item()

            if stopper(val_loss, self.model_):
                logger.info("[FT-Transformer] Early stop at epoch %d", epoch)
                stopper.restore_best(self.model_)
                break

        logger.info("[FT-Transformer] Training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(_to_tensor(X, device=self.device)).cpu().numpy()
        prob = torch.sigmoid(torch.tensor(logits)).numpy()
        return np.column_stack([1 - prob, prob])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM (longitudinal / sequential EHR)
# ─────────────────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, in_dim, hidden_size, num_layers, bidirectional, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0,
        )
        d = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, x):
        # x: (B, seq_len, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)


class LSTMModel:
    """
    Bidirectional LSTM for sequential/longitudinal EHR data.

    For static (non-sequential) tabular data, the feature vector is treated
    as a sequence of length 1 (each feature is one time step).

    Parameters
    ----------
    hidden_size   : int   LSTM hidden state dimension.
    num_layers    : int   Stacked LSTM depth.
    bidirectional : bool  Bidirectional LSTM.
    seq_len       : int   Expected sequence length (1 for static tabular).
    """

    name = "LSTM"

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        seq_len: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.bidirectional= bidirectional
        self.dropout      = dropout
        self.seq_len      = seq_len
        self.lr           = lr
        self.weight_decay = weight_decay
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.device       = torch.device(device) if device else DEVICE
        self.random_state = random_state
        self.model_       = None

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        """Reshape (N, F) → (N, seq_len, F//seq_len) for LSTM."""
        n, f = X.shape
        f_per_step = f // self.seq_len
        return X[:, :f_per_step * self.seq_len].reshape(n, self.seq_len, f_per_step)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        torch.manual_seed(self.random_state)
        X_seq = self._reshape(X_train)
        in_dim = X_seq.shape[2]

        self.model_ = _LSTMNet(
            in_dim, self.hidden_size, self.num_layers,
            self.bidirectional, self.dropout,
        ).to(self.device)

        pos_weight = _compute_pos_weight(y_train).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.Adam(self.model_.parameters(),
                                      lr=self.lr, weight_decay=self.weight_decay)
        train_loader = _make_loader(X_seq.reshape(len(X_seq), -1), y_train, self.batch_size)
        stopper      = EarlyStopping(patience=self.patience)

        seq_shape = X_seq.shape[1:]

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            for Xb, yb in train_loader:
                Xb = Xb.view(-1, *seq_shape).to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                criterion(self.model_(Xb), yb).backward()
                optimizer.step()

            val_loss = 0.0
            if X_val is not None and y_val is not None:
                Xv_seq = torch.tensor(self._reshape(X_val), dtype=torch.float32, device=self.device)
                yv     = torch.tensor(y_val, dtype=torch.float32, device=self.device)
                self.model_.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model_(Xv_seq), yv).item()

            if stopper(val_loss, self.model_):
                logger.info("[LSTM] Early stop at epoch %d", epoch)
                stopper.restore_best(self.model_)
                break

        logger.info("[LSTM] Training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_seq = torch.tensor(self._reshape(X), dtype=torch.float32, device=self.device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_seq).cpu().numpy()
        prob = torch.sigmoid(torch.tensor(logits)).numpy()
        return np.column_stack([1 - prob, prob])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 1D-CNN
# ─────────────────────────────────────────────────────────────────────────────

class _CNN1DNet(nn.Module):
    def __init__(self, in_channels, channels, kernel_sizes, pool_size, dropout):
        super().__init__()
        layers = []
        prev = 1  # treat feature vector as 1-channel 1D signal
        for ch, ks in zip(channels, kernel_sizes):
            layers += [
                nn.Conv1d(prev, ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout),
            ]
            prev = ch
        self.conv = nn.Sequential(*layers)
        self.head = None   # built dynamically after first forward pass

    def _build_head(self, x_sample):
        with torch.no_grad():
            out_dim = self.conv(x_sample.unsqueeze(0).unsqueeze(0)).numel()
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)   # (B, 1, F)
        x = self.conv(x)
        x = x.flatten(1)
        return self.head(x).squeeze(1)


class CNN1DModel:
    """1D Convolutional Network treating the feature vector as a 1D signal."""

    name = "1D-CNN"

    def __init__(
        self,
        channels: list[int] = (64, 128, 64),
        kernel_sizes: list[int] = (3, 3, 3),
        pool_size: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.channels     = list(channels)
        self.kernel_sizes = list(kernel_sizes)
        self.pool_size    = pool_size
        self.dropout      = dropout
        self.lr           = lr
        self.weight_decay = weight_decay
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.patience     = patience
        self.device       = torch.device(device) if device else DEVICE
        self.random_state = random_state
        self.model_       = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        torch.manual_seed(self.random_state)
        self.model_ = _CNN1DNet(
            X_train.shape[1], self.channels, self.kernel_sizes,
            self.pool_size, self.dropout,
        ).to(self.device)

        # Build head with a dummy forward pass
        sample = torch.zeros(X_train.shape[1], device=self.device)
        self.model_._build_head(sample)
        self.model_ = self.model_.to(self.device)

        pos_weight = _compute_pos_weight(y_train).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.Adam(self.model_.parameters(),
                                      lr=self.lr, weight_decay=self.weight_decay)
        train_loader = _make_loader(X_train, y_train, self.batch_size)
        stopper      = EarlyStopping(patience=self.patience)

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                criterion(self.model_(Xb), yb).backward()
                optimizer.step()

            val_loss = 0.0
            if X_val is not None and y_val is not None:
                Xv, yv = _to_tensor(X_val, y_val, self.device)
                self.model_.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model_(Xv), yv).item()

            if stopper(val_loss, self.model_):
                logger.info("[1D-CNN] Early stop at epoch %d", epoch)
                stopper.restore_best(self.model_)
                break

        logger.info("[1D-CNN] Training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(_to_tensor(X, device=self.device)).cpu().numpy()
        prob = torch.sigmoid(torch.tensor(logits)).numpy()
        return np.column_stack([1 - prob, prob])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class DeepLearningModels:
    """Registry and factory for DL models."""

    _REGISTRY = {
        "mlp":            MLPModel,
        "tabnet":         TabNetModel,
        "ft_transformer": FTTransformerModel,
        "lstm":           LSTMModel,
        "cnn_1d":         CNN1DModel,
    }

    def get(self, key: str, **kwargs):
        key = key.lower()
        if key not in self._REGISTRY:
            raise KeyError(f"Unknown DL model '{key}'. Available: {list(self._REGISTRY)}")
        return self._REGISTRY[key](**kwargs)

    def get_all(self, **kwargs) -> dict:
        import warnings
        models = {}
        for key, cls in self._REGISTRY.items():
            try:
                models[key] = cls(**kwargs)
            except ImportError as e:
                warnings.warn(f"Skipping {key}: {e}", stacklevel=2)
        return models

    @property
    def available(self) -> list[str]:
        return list(self._REGISTRY)
