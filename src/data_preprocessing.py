"""
data_preprocessing.py
─────────────────────
EHR data preprocessing utilities for disease prediction modeling.

Handles:
  • Loading and validating EHR data
  • Missing value imputation (median, mean, KNN, iterative)
  • Outlier detection and capping
  • Categorical encoding (one-hot, ordinal, target)
  • Feature scaling (standard, min-max, robust)
  • Stratified train / validation / test splitting
  • Class imbalance handling (SMOTE, ADASYN, class weights)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

try:
    from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn("imbalanced-learn not installed; oversampling disabled.", stacklevel=2)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

class EHRPreprocessor:
    """
    End-to-end preprocessing pipeline for tabular EHR data.

    Parameters
    ----------
    target_col : str
        Name of the binary outcome column.
    imputation_strategy : str
        One of 'median', 'mean', 'knn', 'iterative'.
    scaling : str
        One of 'standard', 'minmax', 'robust', 'none'.
    encoding_method : str
        One of 'onehot', 'ordinal', 'target'.
    outlier_method : str
        One of 'iqr', 'zscore', 'none'.
    outlier_threshold : float
        IQR multiplier or z-score threshold.
    random_state : int
    """

    def __init__(
        self,
        target_col: str = "superimposed_pe",
        imputation_strategy: str = "median",
        scaling: str = "standard",
        encoding_method: str = "onehot",
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        knn_neighbors: int = 5,
        random_state: int = 42,
    ):
        self.target_col          = target_col
        self.imputation_strategy = imputation_strategy
        self.scaling             = scaling
        self.encoding_method     = encoding_method
        self.outlier_method      = outlier_method
        self.outlier_threshold   = outlier_threshold
        self.knn_neighbors       = knn_neighbors
        self.random_state        = random_state

        # Fitted artefacts (populated during fit / fit_transform)
        self.imputer_    = None
        self.scaler_     = None
        self.encoders_   = {}          # {col: encoder}
        self.feature_names_out_: list[str] = []
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self._fitted     = False

    # ── I/O ──────────────────────────────────────────────────────────────────

    @staticmethod
    def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
        """Load a CSV file and return a DataFrame."""
        df = pd.read_csv(path, **kwargs)
        logger.info("Loaded %d rows × %d cols from %s", *df.shape, path)
        return df

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic schema and range checks; logs warnings on violations."""
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found.")

        # Binary target check
        unique_vals = df[self.target_col].dropna().unique()
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError(f"Target column must be binary (0/1). Found: {unique_vals}")

        missing_pct = df.isnull().mean() * 100
        high_missing = missing_pct[missing_pct > 30]
        if not high_missing.empty:
            logger.warning("Columns with >30%% missing:\n%s", high_missing.to_string())

        logger.info("Validation OK — %d rows, prevalence=%.1f%%",
                    len(df), df[self.target_col].mean() * 100)
        return df

    # ── Outlier handling ──────────────────────────────────────────────────────

    def _cap_outliers_iqr(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.outlier_threshold * iqr
            upper = q3 + self.outlier_threshold * iqr
            n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
            if n_clipped > 0:
                logger.debug("Capping %d outliers in '%s'", n_clipped, col)
            df[col] = df[col].clip(lower, upper)
        return df

    def _cap_outliers_zscore(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for col in cols:
            mean = df[col].mean()
            std  = df[col].std()
            if std == 0:
                continue
            z = (df[col] - mean) / std
            lower = mean - self.outlier_threshold * std
            upper = mean + self.outlier_threshold * std
            n_clipped = (z.abs() > self.outlier_threshold).sum()
            if n_clipped > 0:
                logger.debug("Capping %d outliers in '%s'", n_clipped, col)
            df[col] = df[col].clip(lower, upper)
        return df

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.outlier_method == "none":
            return df
        cols = self.numeric_cols_
        if self.outlier_method == "iqr":
            return self._cap_outliers_iqr(df.copy(), cols)
        elif self.outlier_method == "zscore":
            return self._cap_outliers_zscore(df.copy(), cols)
        else:
            raise ValueError(f"Unknown outlier_method: '{self.outlier_method}'")

    # ── Imputation ────────────────────────────────────────────────────────────

    def _build_imputer(self):
        if self.imputation_strategy == "median":
            return SimpleImputer(strategy="median")
        elif self.imputation_strategy == "mean":
            return SimpleImputer(strategy="mean")
        elif self.imputation_strategy == "knn":
            return KNNImputer(n_neighbors=self.knn_neighbors)
        elif self.imputation_strategy == "iterative":
            return IterativeImputer(random_state=self.random_state, max_iter=10)
        else:
            raise ValueError(f"Unknown imputation_strategy: '{self.imputation_strategy}'")

    # ── Encoding ─────────────────────────────────────────────────────────────

    def _encode_categorical(self, df: pd.DataFrame, y=None, fit: bool = True) -> pd.DataFrame:
        if not self.categorical_cols_:
            return df

        if self.encoding_method == "onehot":
            df = pd.get_dummies(df, columns=self.categorical_cols_, drop_first=False)
        elif self.encoding_method == "ordinal":
            for col in self.categorical_cols_:
                enc = self.encoders_.get(col, OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1))
                if fit:
                    df[col] = enc.fit_transform(df[[col]])
                    self.encoders_[col] = enc
                else:
                    df[col] = enc.transform(df[[col]])
        elif self.encoding_method == "target":
            if y is None and fit:
                raise ValueError("y must be provided for target encoding during fit.")
            for col in self.categorical_cols_:
                enc = self.encoders_.get(col, TargetEncoder(random_state=self.random_state))
                if fit:
                    df[col] = enc.fit_transform(df[[col]], y)
                    self.encoders_[col] = enc
                else:
                    df[col] = enc.transform(df[[col]])
        else:
            raise ValueError(f"Unknown encoding_method: '{self.encoding_method}'")
        return df

    # ── Scaling ───────────────────────────────────────────────────────────────

    def _build_scaler(self):
        if self.scaling == "standard":
            return StandardScaler()
        elif self.scaling == "minmax":
            return MinMaxScaler()
        elif self.scaling == "robust":
            return RobustScaler()
        elif self.scaling == "none":
            return None
        else:
            raise ValueError(f"Unknown scaling: '{self.scaling}'")

    # ── Main API ──────────────────────────────────────────────────────────────

    def fit_transform(
        self, df: pd.DataFrame, y_col: Optional[str] = None
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Fit preprocessor on df and return (X, y, feature_names).

        Parameters
        ----------
        df : pd.DataFrame   Full dataset including target column.
        y_col : str | None  Override target column name.

        Returns
        -------
        X : np.ndarray  shape (n_samples, n_features)
        y : np.ndarray  shape (n_samples,)
        feature_names : list[str]
        """
        target = y_col or self.target_col
        df = self.validate(df.copy())

        y = df[target].values.astype(int)
        X_df = df.drop(columns=[target])

        # Identify column types
        self.categorical_cols_ = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.numeric_cols_     = X_df.select_dtypes(include=[np.number]).columns.tolist()

        # 1. Outlier capping (numerics only, before imputation)
        X_df = self.handle_outliers(X_df)

        # 2. Imputation
        self.imputer_ = self._build_imputer()
        num_data = self.imputer_.fit_transform(X_df[self.numeric_cols_])
        X_df[self.numeric_cols_] = num_data

        # 3. Categorical encoding
        X_df = self._encode_categorical(X_df, y=y, fit=True)

        # 4. Scaling
        self.scaler_ = self._build_scaler()
        numeric_after = X_df.select_dtypes(include=[np.number]).columns.tolist()
        if self.scaler_ is not None:
            X_df[numeric_after] = self.scaler_.fit_transform(X_df[numeric_after])

        self.feature_names_out_ = X_df.columns.tolist()
        self._fitted = True
        return X_df.values.astype(np.float32), y, self.feature_names_out_

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply fitted preprocessor to new data (no target column expected)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        X_df = df.copy()
        X_df = self.handle_outliers(X_df)

        num_data = self.imputer_.transform(X_df[self.numeric_cols_])
        X_df[self.numeric_cols_] = num_data

        X_df = self._encode_categorical(X_df, fit=False)

        numeric_after = X_df.select_dtypes(include=[np.number]).columns.tolist()
        if self.scaler_ is not None:
            X_df[numeric_after] = self.scaler_.transform(X_df[numeric_after])

        return X_df[self.feature_names_out_].values.astype(np.float32)

    # ── Train / val / test splits ─────────────────────────────────────────────

    @staticmethod
    def split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.20,
        val_size: float  = 0.10,
        stratify: bool   = True,
        random_state: int = 42,
    ) -> tuple:
        """
        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        strat = y if stratify else None
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, stratify=strat, random_state=random_state)

        val_fraction = val_size / (1 - test_size)
        strat2 = y_trainval if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_fraction,
            stratify=strat2, random_state=random_state)

        logger.info("Split — train:%d  val:%d  test:%d", len(y_train), len(y_val), len(y_test))
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ── Class imbalance ───────────────────────────────────────────────────────

    @staticmethod
    def resample(
        X: np.ndarray,
        y: np.ndarray,
        method: str = "smote",
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Resample training data to handle class imbalance.

        Parameters
        ----------
        method : str
            'smote' | 'adasyn' | 'oversample' | 'undersample'
        """
        if not IMBLEARN_AVAILABLE:
            raise ImportError("Install imbalanced-learn: pip install imbalanced-learn")

        samplers = {
            "smote":       SMOTE(random_state=random_state),
            "adasyn":      ADASYN(random_state=random_state),
            "oversample":  RandomOverSampler(random_state=random_state),
            "undersample": RandomUnderSampler(random_state=random_state),
        }
        if method not in samplers:
            raise ValueError(f"Unknown resample method '{method}'. Choose from {list(samplers)}")

        sampler = samplers[method]
        X_res, y_res = sampler.fit_resample(X, y)
        logger.info("Resampled with %s: %d → %d samples", method, len(y), len(y_res))
        return X_res, y_res

    # ── Cross-validation helper ───────────────────────────────────────────────

    @staticmethod
    def stratified_kfold(
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        """Yield (X_train, X_val, y_train, y_val) tuples for k-fold CV."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, val_idx in skf.split(X, y):
            yield (X[train_idx], X[val_idx], y[train_idx], y[val_idx])

    # ── Convenience ──────────────────────────────────────────────────────────

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a summary DataFrame with missing rates, dtypes, and basic stats."""
        summary = pd.DataFrame({
            "dtype":        df.dtypes,
            "n_missing":    df.isnull().sum(),
            "pct_missing":  (df.isnull().mean() * 100).round(1),
            "n_unique":     df.nunique(),
        })
        for col in df.select_dtypes(include=[np.number]).columns:
            summary.loc[col, "mean"] = df[col].mean()
            summary.loc[col, "std"]  = df[col].std()
            summary.loc[col, "min"]  = df[col].min()
            summary.loc[col, "max"]  = df[col].max()
        return summary
