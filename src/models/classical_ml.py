"""
classical_ml.py
───────────────
Classical Machine Learning models for disease prediction from EHR data.

Models implemented:
  • Logistic Regression (with L1 / L2 / ElasticNet regularisation)
  • Support Vector Machine (RBF, Linear, Polynomial kernels)
  • Random Forest
  • Gradient Boosting — XGBoost, LightGBM, CatBoost
  • Decision Tree
  • K-Nearest Neighbours
  • Naïve Bayes (Gaussian)
  • Linear Discriminant Analysis

Each model class exposes:
  fit(X_train, y_train)     → self
  predict_proba(X)          → np.ndarray (n, 2)
  predict(X)                → np.ndarray (n,)
  get_feature_importance()  → pd.DataFrame | None
  hyperparameter_search()   → best estimator
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

CV_FOLDS = 5
SCORING  = "roc_auc"
N_JOBS   = -1
SEED     = 42


# ─────────────────────────────────────────────────────────────────────────────
# Base mixin
# ─────────────────────────────────────────────────────────────────────────────

class _BaseModel:
    """Shared interface for all classical ML wrappers."""

    name: str = "BaseModel"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseModel":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return None

    def _log_fitted(self):
        logger.info("[%s] Training complete.", self.name)


# ─────────────────────────────────────────────────────────────────────────────
# Individual Models
# ─────────────────────────────────────────────────────────────────────────────

class LogisticRegressionModel(_BaseModel):
    """
    Logistic Regression with elastic-net / L1 / L2 regularisation.

    Parameters
    ----------
    C            : float    Inverse regularisation strength.
    penalty      : str      'l1' | 'l2' | 'elasticnet' | 'none'
    l1_ratio     : float    ElasticNet mixing (0=L2, 1=L1); only for elasticnet.
    class_weight : str|dict 'balanced' or None.
    max_iter     : int
    """

    name = "Logistic Regression"

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        l1_ratio: Optional[float] = None,
        class_weight: Any = "balanced",
        max_iter: int = 1000,
        random_state: int = SEED,
    ):
        from sklearn.linear_model import LogisticRegression
        solver = "saga" if penalty in ("l1", "elasticnet") else "lbfgs"
        kwargs = dict(C=C, penalty=penalty, solver=solver,
                      class_weight=class_weight, max_iter=max_iter,
                      random_state=random_state, n_jobs=N_JOBS)
        if penalty == "elasticnet":
            kwargs["l1_ratio"] = l1_ratio or 0.5
        self.estimator = LogisticRegression(**kwargs)

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self._log_fitted()
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        coef = self.estimator.coef_.ravel()
        return pd.DataFrame({"importance_mean": np.abs(coef), "coefficient": coef})

    def hyperparameter_search(
        self, X, y,
        param_grid: Optional[dict] = None,
        method: str = "grid",
    ):
        if param_grid is None:
            param_grid = {
                "C":       [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
            }
        return _run_search(self.estimator, X, y, param_grid, method)


class SVMModel(_BaseModel):
    """Support Vector Machine (probability calibrated)."""

    name = "SVM"

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        class_weight: Any = "balanced",
        random_state: int = SEED,
    ):
        from sklearn.svm import SVC
        self.estimator = SVC(
            C=C, kernel=kernel, gamma=gamma,
            class_weight=class_weight, probability=True,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self._log_fitted()
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def hyperparameter_search(self, X, y, param_grid=None, method="random"):
        if param_grid is None:
            param_grid = {
                "C":      [0.1, 1, 10, 100],
                "kernel": ["rbf", "linear"],
                "gamma":  ["scale", "auto"],
            }
        return _run_search(self.estimator, X, y, param_grid, method)


class RandomForestModel(_BaseModel):
    """Random Forest Classifier."""

    name = "Random Forest"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        class_weight: Any = "balanced",
        random_state: int = SEED,
    ):
        from sklearn.ensemble import RandomForestClassifier
        self.estimator = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, class_weight=class_weight,
            random_state=random_state, n_jobs=N_JOBS,
        )

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self._log_fitted()
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            "importance_mean": self.estimator.feature_importances_,
            "importance_std":  np.std([t.feature_importances_
                                        for t in self.estimator.estimators_], axis=0),
        })

    def hyperparameter_search(self, X, y, param_grid=None, method="random"):
        if param_grid is None:
            param_grid = {
                "n_estimators":     [100, 300, 500],
                "max_depth":        [None, 5, 10, 20],
                "min_samples_split":[2, 5, 10],
            }
        return _run_search(self.estimator, X, y, param_grid, method)


class XGBoostModel(_BaseModel):
    """XGBoost Classifier."""

    name = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
        random_state: int = SEED,
    ):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("Install xgboost: pip install xgboost")

        self.estimator = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state, n_jobs=N_JOBS,
            eval_metric="logloss", verbosity=0,
        )

    def fit(self, X, y, eval_set=None, early_stopping_rounds=50):
        kwargs = {}
        if eval_set:
            kwargs["eval_set"] = eval_set
            kwargs["verbose"]  = False
            self.estimator.set_params(early_stopping_rounds=early_stopping_rounds)
        self.estimator.fit(X, y, **kwargs)
        self._log_fitted()
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({"importance_mean": self.estimator.feature_importances_})

    def hyperparameter_search(self, X, y, param_grid=None, method="random"):
        if param_grid is None:
            param_grid = {
                "n_estimators":   [100, 300, 500],
                "max_depth":      [3, 5, 7],
                "learning_rate":  [0.01, 0.05, 0.1],
                "subsample":      [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
            }
        return _run_search(self.estimator, X, y, param_grid, method)


class LightGBMModel(_BaseModel):
    """LightGBM Classifier."""

    name = "LightGBM"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        class_weight: Any = "balanced",
        random_state: int = SEED,
    ):
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("Install lightgbm: pip install lightgbm")

        self.estimator = LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, num_leaves=num_leaves,
            class_weight=class_weight, random_state=random_state,
            n_jobs=N_JOBS, verbosity=-1,
        )

    def fit(self, X, y, eval_set=None, callbacks=None):
        kwargs = {}
        if eval_set:
            kwargs["eval_set"] = eval_set
        if callbacks:
            kwargs["callbacks"] = callbacks
        self.estimator.fit(X, y, **kwargs)
        self._log_fitted()
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({"importance_mean": self.estimator.feature_importances_})

    def hyperparameter_search(self, X, y, param_grid=None, method="random"):
        if param_grid is None:
            param_grid = {
                "n_estimators":  [100, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves":    [31, 63, 127],
                "max_depth":     [-1, 5, 10],
            }
        return _run_search(self.estimator, X, y, param_grid, method)


class CatBoostModel(_BaseModel):
    """CatBoost Classifier."""

    name = "CatBoost"

    def __init__(
        self,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        auto_class_weights: str = "Balanced",
        random_state: int = SEED,
    ):
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("Install catboost: pip install catboost")

        self.estimator = CatBoostClassifier(
            iterations=iterations, depth=depth,
            learning_rate=learning_rate,
            auto_class_weights=auto_class_weights,
            random_seed=random_state, verbose=0,
        )

    def fit(self, X, y, eval_set=None, early_stopping_rounds=50):
        kwargs = dict(early_stopping_rounds=early_stopping_rounds) if eval_set else {}
        self.estimator.fit(X, y, eval_set=eval_set, **kwargs)
        self._log_fitted()
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({"importance_mean": self.estimator.get_feature_importance()})

    def hyperparameter_search(self, X, y, param_grid=None, method="random"):
        if param_grid is None:
            param_grid = {
                "iterations":    [100, 300, 500],
                "depth":         [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
            }
        return _run_search(self.estimator, X, y, param_grid, method)


class DecisionTreeModel(_BaseModel):
    """Decision Tree (interpretable baseline)."""

    name = "Decision Tree"

    def __init__(
        self,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 5,
        class_weight: Any = "balanced",
        random_state: int = SEED,
    ):
        self.estimator = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_split=min_samples_split,
            class_weight=class_weight, random_state=random_state,
        )

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self._log_fitted()
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({"importance_mean": self.estimator.feature_importances_})

    def hyperparameter_search(self, X, y, param_grid=None, method="grid"):
        if param_grid is None:
            param_grid = {
                "max_depth":         [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
            }
        return _run_search(self.estimator, X, y, param_grid, method)


class KNNModel(_BaseModel):
    """K-Nearest Neighbours Classifier."""

    name = "KNN"

    def __init__(
        self,
        n_neighbors: int = 7,
        weights: str = "distance",
        metric: str = "euclidean",
    ):
        self.estimator = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights,
            metric=metric, n_jobs=N_JOBS,
        )

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self._log_fitted()
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def hyperparameter_search(self, X, y, param_grid=None, method="grid"):
        if param_grid is None:
            param_grid = {
                "n_neighbors": [3, 5, 7, 11, 15],
                "weights":     ["uniform", "distance"],
                "metric":      ["euclidean", "manhattan"],
            }
        return _run_search(self.estimator, X, y, param_grid, method)


class NaiveBayesModel(_BaseModel):
    """Gaussian Naïve Bayes Classifier."""

    name = "Naïve Bayes"

    def __init__(self, var_smoothing: float = 1e-9):
        self.estimator = GaussianNB(var_smoothing=var_smoothing)

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self._log_fitted()
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def hyperparameter_search(self, X, y, param_grid=None, method="grid"):
        if param_grid is None:
            param_grid = {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]}
        return _run_search(self.estimator, X, y, param_grid, method)


class LDAModel(_BaseModel):
    """Linear Discriminant Analysis."""

    name = "LDA"

    def __init__(self, solver: str = "lsqr", shrinkage: Any = "auto"):
        self.estimator = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self._log_fitted()
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        coef = self.estimator.coef_.ravel()
        return pd.DataFrame({"importance_mean": np.abs(coef), "coefficient": coef})

    def hyperparameter_search(self, X, y, param_grid=None, method="grid"):
        if param_grid is None:
            param_grid = {"solver": ["lsqr"], "shrinkage": [None, "auto", 0.1, 0.5, 0.9]}
        return _run_search(self.estimator, X, y, param_grid, method)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience registry
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalMLModels:
    """
    Registry and factory for all classical ML models.

    Usage
    -----
    factory = ClassicalMLModels()
    models  = factory.get_all()          # dict: name → instantiated model
    model   = factory.get("xgboost")
    """

    _REGISTRY = {
        "logistic_regression": LogisticRegressionModel,
        "svm":                 SVMModel,
        "random_forest":       RandomForestModel,
        "xgboost":             XGBoostModel,
        "lightgbm":            LightGBMModel,
        "catboost":            CatBoostModel,
        "decision_tree":       DecisionTreeModel,
        "knn":                 KNNModel,
        "naive_bayes":         NaiveBayesModel,
        "lda":                 LDAModel,
    }

    def get(self, key: str, **kwargs) -> _BaseModel:
        key = key.lower().replace(" ", "_")
        if key not in self._REGISTRY:
            raise KeyError(f"Unknown model '{key}'. Available: {list(self._REGISTRY)}")
        return self._REGISTRY[key](**kwargs)

    def get_all(self, **kwargs) -> dict[str, _BaseModel]:
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


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_search(estimator, X, y, param_grid, method="grid", n_iter=50):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    if method == "grid":
        search = GridSearchCV(estimator, param_grid, cv=cv,
                              scoring=SCORING, n_jobs=N_JOBS, refit=True)
    elif method == "random":
        search = RandomizedSearchCV(estimator, param_grid, n_iter=n_iter,
                                    cv=cv, scoring=SCORING,
                                    n_jobs=N_JOBS, refit=True, random_state=SEED)
    else:
        raise ValueError(f"method must be 'grid' or 'random', got '{method}'")

    search.fit(X, y)
    logger.info("Best CV %s = %.4f  params=%s", SCORING, search.best_score_, search.best_params_)
    return search.best_estimator_
