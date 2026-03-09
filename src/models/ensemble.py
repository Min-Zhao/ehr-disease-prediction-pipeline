"""
ensemble.py
───────────
Ensemble models for disease prediction from tabular EHR data.

Strategies implemented:
  • Soft / Hard Voting        — combine predicted probabilities or class votes
  • Stacking                  — meta-learner trained on out-of-fold base predictions
  • Blending                  — meta-learner on a held-out blending set
  • Bayesian Model Averaging  — weighted average using approximate posterior
  • Rank Averaging            — average of probability rank transforms

All models expose:
  fit(X_train, y_train)
  predict_proba(X) → np.ndarray (n, 2)
  predict(X)       → np.ndarray (n,)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Base mixin
# ─────────────────────────────────────────────────────────────────────────────

class _BaseEnsemble:
    name: str = "BaseEnsemble"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseEnsemble":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Voting Ensemble
# ─────────────────────────────────────────────────────────────────────────────

class VotingEnsemble(_BaseEnsemble):
    """
    Soft or hard voting over a list of base estimators.

    Parameters
    ----------
    estimators  : list of (name, fitted_or_unfitted_estimator) tuples.
    voting      : 'soft' (average probabilities) | 'hard' (majority class vote).
    weights     : optional list of weights per estimator.
    """

    name = "Voting Ensemble"

    def __init__(
        self,
        estimators: list[tuple[str, Any]],
        voting: str = "soft",
        weights: Optional[list[float]] = None,
    ):
        self.estimators = estimators
        self.voting     = voting
        self.weights    = weights
        self.fitted_    = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingEnsemble":
        self.fitted_ = []
        for name, est in self.estimators:
            est_clone = clone(est) if hasattr(est, "get_params") else est
            if hasattr(est_clone, "fit"):
                est_clone.fit(X, y)
            self.fitted_.append((name, est_clone))
        logger.info("[%s] Fitted %d base estimators.", self.name, len(self.fitted_))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.voting == "soft":
            probs = [_get_proba(est, X) for _, est in self.fitted_]
            w = np.array(self.weights) if self.weights else np.ones(len(probs))
            w = w / w.sum()
            avg = sum(p * wi for p, wi in zip(probs, w))
            return avg
        else:  # hard
            votes = np.array([_get_proba(est, X)[:, 1] >= 0.5 for _, est in self.fitted_])
            majority = votes.mean(axis=0) >= 0.5
            return np.column_stack([~majority, majority]).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Stacking
# ─────────────────────────────────────────────────────────────────────────────

class StackingEnsemble(_BaseEnsemble):
    """
    Generalised stacking with out-of-fold (OOF) meta-features.

    The meta-learner is trained on OOF predictions from k-fold cross-validation
    so that it never sees training-set predictions from models trained on that fold.

    Parameters
    ----------
    base_estimators : list of (name, estimator) tuples.
    meta_learner    : sklearn-compatible estimator.
    n_splits        : int   Number of CV folds.
    use_proba       : bool  Use class probabilities as meta-features (recommended).
    passthrough     : bool  Append original features to meta-features.
    """

    name = "Stacking Ensemble"

    def __init__(
        self,
        base_estimators: list[tuple[str, Any]],
        meta_learner: Any = None,
        n_splits: int = 5,
        use_proba: bool = True,
        passthrough: bool = False,
        random_state: int = SEED,
    ):
        self.base_estimators = base_estimators
        self.meta_learner    = meta_learner or LogisticRegression(
            C=1.0, max_iter=1000, random_state=random_state)
        self.n_splits        = n_splits
        self.use_proba       = use_proba
        self.passthrough     = passthrough
        self.random_state    = random_state

        self.fitted_base_: list[list] = []   # n_estimators × n_folds
        self.meta_scaler_ = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.random_state)
        n_estimators = len(self.base_estimators)
        n_meta_cols  = n_estimators * (2 if self.use_proba else 1)

        oof_preds  = np.zeros((len(y), n_meta_cols))
        fold_models = [[] for _ in range(n_estimators)]

        for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y)):
            X_tr, X_vl = X[tr_idx], X[vl_idx]
            y_tr        = y[tr_idx]

            for i, (nm, est) in enumerate(self.base_estimators):
                est_clone = clone(est) if hasattr(est, "get_params") else est
                if hasattr(est_clone, "fit"):
                    est_clone.fit(X_tr, y_tr)
                pred = _get_meta_feature(est_clone, X_vl, self.use_proba)

                if self.use_proba:
                    oof_preds[vl_idx, i * 2:(i + 1) * 2] = pred
                else:
                    oof_preds[vl_idx, i] = pred.ravel()

                fold_models[i].append(est_clone)

        self.fitted_base_ = fold_models

        meta_X = oof_preds
        if self.passthrough:
            meta_X = np.hstack([meta_X, X])
        meta_X = self.meta_scaler_.fit_transform(meta_X)

        meta_clone = clone(self.meta_learner) \
            if hasattr(self.meta_learner, "get_params") else self.meta_learner
        meta_clone.fit(meta_X, y)
        self.meta_learner_ = meta_clone

        logger.info("[%s] Stacking complete — %d base estimators × %d folds.",
                    self.name, n_estimators, self.n_splits)
        return self

    def _make_meta_X(self, X: np.ndarray) -> np.ndarray:
        n_estimators = len(self.base_estimators)
        n_meta_cols  = n_estimators * (2 if self.use_proba else 1)
        preds = np.zeros((len(X), n_meta_cols))

        for i, fold_ests in enumerate(self.fitted_base_):
            fold_preds = [_get_meta_feature(e, X, self.use_proba) for e in fold_ests]
            avg_pred   = np.mean(fold_preds, axis=0)
            if self.use_proba:
                preds[:, i * 2:(i + 1) * 2] = avg_pred
            else:
                preds[:, i] = avg_pred.ravel()

        if self.passthrough:
            preds = np.hstack([preds, X])
        return self.meta_scaler_.transform(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_X = self._make_meta_X(X)
        return self.meta_learner_.predict_proba(meta_X)


# ─────────────────────────────────────────────────────────────────────────────
# Blending
# ─────────────────────────────────────────────────────────────────────────────

class BlendingEnsemble(_BaseEnsemble):
    """
    Blending (holdout-based stacking).

    Base models are trained on (1 − holdout_fraction) of the data;
    the meta-learner is trained on the blending (holdout) set.

    Parameters
    ----------
    base_estimators  : list of (name, estimator) tuples.
    meta_learner     : sklearn-compatible estimator.
    holdout_fraction : float  Fraction reserved for blending.
    """

    name = "Blending Ensemble"

    def __init__(
        self,
        base_estimators: list[tuple[str, Any]],
        meta_learner: Any = None,
        holdout_fraction: float = 0.2,
        random_state: int = SEED,
    ):
        self.base_estimators  = base_estimators
        self.meta_learner     = meta_learner or LogisticRegression(
            C=1.0, max_iter=1000, random_state=random_state)
        self.holdout_fraction = holdout_fraction
        self.random_state     = random_state
        self.fitted_base_     = []
        self.meta_scaler_     = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BlendingEnsemble":
        from sklearn.model_selection import train_test_split
        X_base, X_blend, y_base, y_blend = train_test_split(
            X, y, test_size=self.holdout_fraction,
            stratify=y, random_state=self.random_state,
        )

        self.fitted_base_ = []
        blend_preds = []

        for nm, est in self.base_estimators:
            est_clone = clone(est) if hasattr(est, "get_params") else est
            if hasattr(est_clone, "fit"):
                est_clone.fit(X_base, y_base)
            blend_preds.append(_get_proba(est_clone, X_blend)[:, 1:])
            self.fitted_base_.append((nm, est_clone))

        meta_X = self.meta_scaler_.fit_transform(np.hstack(blend_preds))
        meta_clone = clone(self.meta_learner) \
            if hasattr(self.meta_learner, "get_params") else self.meta_learner
        meta_clone.fit(meta_X, y_blend)
        self.meta_learner_ = meta_clone

        logger.info("[%s] Blending complete — %d base estimators.", self.name, len(self.fitted_base_))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = [_get_proba(est, X)[:, 1:] for _, est in self.fitted_base_]
        meta_X = self.meta_scaler_.transform(np.hstack(preds))
        return self.meta_learner_.predict_proba(meta_X)


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Model Averaging
# ─────────────────────────────────────────────────────────────────────────────

class BayesianModelAveraging(_BaseEnsemble):
    """
    Bayesian Model Averaging (BMA) via approximate evidence weighting.

    Weights each model by exp(AUROC) as a tractable proxy for the marginal
    likelihood, then normalises to a simplex.

    Parameters
    ----------
    estimators : list of (name, fitted_estimator) tuples.
    """

    name = "Bayesian Model Averaging"

    def __init__(
        self,
        estimators: list[tuple[str, Any]],
        n_splits: int = 5,
        random_state: int = SEED,
    ):
        self.estimators   = estimators
        self.n_splits     = n_splits
        self.random_state = random_state
        self.weights_     = None
        self.fitted_      = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianModelAveraging":
        from sklearn.metrics import roc_auc_score
        skf    = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                 random_state=self.random_state)
        aucs   = np.zeros(len(self.estimators))

        # OOF AUROC for each estimator
        for i, (nm, est) in enumerate(self.estimators):
            oof_probs = np.zeros(len(y))
            for tr, vl in skf.split(X, y):
                est_clone = clone(est) if hasattr(est, "get_params") else est
                est_clone.fit(X[tr], y[tr])
                oof_probs[vl] = _get_proba(est_clone, X[vl])[:, 1]
            aucs[i] = roc_auc_score(y, oof_probs)
            logger.info("  BMA — [%s] OOF AUROC = %.4f", nm, aucs[i])

        # Posterior weights ∝ exp(AUROC) — approximate, tractable
        log_weights  = aucs
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        self.weights_ = weights / weights.sum()
        logger.info("BMA weights: %s", dict(zip([n for n, _ in self.estimators], self.weights_.round(3))))

        # Refit on full training data
        self.fitted_ = []
        for nm, est in self.estimators:
            est_clone = clone(est) if hasattr(est, "get_params") else est
            est_clone.fit(X, y)
            self.fitted_.append((nm, est_clone))

        logger.info("[%s] Fitting complete.", self.name)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = [_get_proba(est, X) for _, est in self.fitted_]
        return sum(p * w for p, w in zip(probs, self.weights_))


# ─────────────────────────────────────────────────────────────────────────────
# Rank Averaging
# ─────────────────────────────────────────────────────────────────────────────

class RankAveragingEnsemble(_BaseEnsemble):
    """
    Rank averaging: convert each model's probabilities to ranks,
    then average the normalised ranks.

    This is robust to miscalibrated individual models.
    """

    name = "Rank Averaging"

    def __init__(self, estimators: list[tuple[str, Any]]):
        self.estimators = estimators
        self.fitted_    = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RankAveragingEnsemble":
        self.fitted_ = []
        for nm, est in self.estimators:
            est_clone = clone(est) if hasattr(est, "get_params") else est
            if hasattr(est_clone, "fit"):
                est_clone.fit(X, y)
            self.fitted_.append((nm, est_clone))
        logger.info("[%s] Fitted %d estimators.", self.name, len(self.fitted_))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        from scipy.stats import rankdata
        n = len(X)
        rank_probs = np.zeros(n)
        for _, est in self.fitted_:
            raw_prob  = _get_proba(est, X)[:, 1]
            rank_probs += rankdata(raw_prob) / n
        avg = rank_probs / len(self.fitted_)
        avg = (avg - avg.min()) / (avg.max() - avg.min() + 1e-9)  # normalise to [0,1]
        return np.column_stack([1 - avg, avg])


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleModels:
    """Factory for ensemble models."""

    _REGISTRY = {
        "voting":            VotingEnsemble,
        "stacking":          StackingEnsemble,
        "blending":          BlendingEnsemble,
        "bayesian_averaging":BayesianModelAveraging,
        "rank_averaging":    RankAveragingEnsemble,
    }

    def get(self, key: str, **kwargs) -> _BaseEnsemble:
        key = key.lower()
        if key not in self._REGISTRY:
            raise KeyError(f"Unknown ensemble '{key}'. Available: {list(self._REGISTRY)}")
        return self._REGISTRY[key](**kwargs)

    @property
    def available(self) -> list[str]:
        return list(self._REGISTRY)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_proba(estimator, X: np.ndarray) -> np.ndarray:
    """Return (n, 2) probability array from any estimator."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        return proba
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return np.column_stack([1 - scores, scores])
    else:
        raise AttributeError(f"{type(estimator)} has no predict_proba or decision_function.")


def _get_meta_feature(estimator, X: np.ndarray, use_proba: bool) -> np.ndarray:
    """Return meta-features as (n, 2) probabilities or (n, 1) binary predictions."""
    if use_proba:
        return _get_proba(estimator, X)
    else:
        return estimator.predict(X).reshape(-1, 1).astype(float)
