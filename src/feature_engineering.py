"""
feature_engineering.py
──────────────────────
Feature engineering utilities for EHR-based disease prediction.

Provides:
  • Clinical domain feature derivation (BP severity, lab flag composites, etc.)
  • Statistical feature selection (chi-square, mutual information, ANOVA F)
  • Dimensionality reduction (PCA, UMAP)
  • Feature importance ranking (permutation, model-based)
  • Interaction terms and polynomial features
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Feature engineering and selection for tabular EHR data.

    Parameters
    ----------
    random_state : int
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # ── Clinical domain features ──────────────────────────────────────────────

    @staticmethod
    def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive clinically interpretable composite features from raw EHR variables.

        Features added (where source columns exist):
          • pulse_pressure            — SBP - DBP (first trimester)
          • mean_arterial_pressure_t1 — (SBP + 2*DBP) / 3 (first trimester)
          • bp_severity_category      — 0=normal, 1=mild HTN, 2=severe HTN
          • hellp_risk_score          — composite of platelets, AST, LDH
          • proteinuria_flag          — protein_urine_24h ≥ 300 mg
          • severe_proteinuria_flag   — protein_urine_24h ≥ 2000 mg
          • renal_function_score      — composite of creatinine + uric acid
          • liver_enzyme_ratio        — AST / ALT
          • metabolic_risk_score      — BMI + diabetes flags composite
        """
        df = df.copy()

        # Blood pressure derived features
        if {"sbp_first_trimester", "dbp_first_trimester"}.issubset(df.columns):
            df["pulse_pressure"]         = df["sbp_first_trimester"] - df["dbp_first_trimester"]
            df["mean_arterial_pressure_t1"] = (
                df["sbp_first_trimester"] + 2 * df["dbp_first_trimester"]) / 3

        if "sbp_max_antepartum" in df.columns:
            df["bp_severity_category"] = pd.cut(
                df["sbp_max_antepartum"],
                bins=[0, 130, 160, 300],
                labels=[0, 1, 2],
            ).astype(float)

        # Proteinuria flags
        if "protein_urine_24h_mg" in df.columns:
            df["proteinuria_flag"]        = (df["protein_urine_24h_mg"] >= 300).astype(int)
            df["severe_proteinuria_flag"] = (df["protein_urine_24h_mg"] >= 2000).astype(int)

        # HELLP-risk composite (low platelets + elevated liver enzymes + elevated LDH)
        hellp_parts = []
        if "platelet_count_k_ul" in df.columns:
            hellp_parts.append((df["platelet_count_k_ul"] < 100).astype(float))
        if "ast_u_l" in df.columns:
            hellp_parts.append((df["ast_u_l"] > 70).astype(float))
        if "ldh_u_l" in df.columns:
            hellp_parts.append((df["ldh_u_l"] > 600).astype(float))
        if hellp_parts:
            df["hellp_risk_score"] = sum(hellp_parts)

        # Renal function composite
        renal_parts = []
        if "creatinine_mg_dl" in df.columns:
            renal_parts.append((df["creatinine_mg_dl"] > 1.1).astype(float))
        if "uric_acid_mg_dl" in df.columns:
            renal_parts.append((df["uric_acid_mg_dl"] > 5.5).astype(float))
        if renal_parts:
            df["renal_function_score"] = sum(renal_parts)

        # Liver enzyme ratio
        if {"alt_u_l", "ast_u_l"}.issubset(df.columns):
            df["liver_enzyme_ratio"] = df["ast_u_l"] / (df["alt_u_l"] + 1e-6)

        # Metabolic risk
        metabolic_parts = []
        if "bmi_prepregnancy" in df.columns:
            metabolic_parts.append((df["bmi_prepregnancy"] > 30).astype(float))
        if "pregestational_diabetes" in df.columns:
            metabolic_parts.append(df["pregestational_diabetes"].astype(float))
        if "gestational_diabetes" in df.columns:
            metabolic_parts.append(df["gestational_diabetes"].astype(float))
        if metabolic_parts:
            df["metabolic_risk_score"] = sum(metabolic_parts)

        n_added = len(df.columns) - len(df.columns)  # placeholder; just log
        logger.info("Clinical feature derivation complete. Total columns: %d", len(df.columns))
        return df

    # ── Statistical feature selection ─────────────────────────────────────────

    def select_by_mutual_info(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        k: int = 20,
    ) -> tuple[np.ndarray, list[str]]:
        """Select top-k features by mutual information with the target."""
        selector = SelectKBest(mutual_info_classif, k=k)
        X_sel    = selector.fit_transform(X, y)
        mask     = selector.get_support()
        selected = [name for name, kept in zip(feature_names, mask) if kept]
        logger.info("Mutual info selection: %d → %d features", X.shape[1], len(selected))
        return X_sel, selected

    def select_by_anova_f(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        k: int = 20,
    ) -> tuple[np.ndarray, list[str]]:
        """Select top-k features by ANOVA F-statistic."""
        selector = SelectKBest(f_classif, k=k)
        X_sel    = selector.fit_transform(X, y)
        mask     = selector.get_support()
        selected = [name for name, kept in zip(feature_names, mask) if kept]
        logger.info("ANOVA-F selection: %d → %d features", X.shape[1], len(selected))
        return X_sel, selected

    def select_by_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        estimator=None,
        threshold: str = "mean",
    ) -> tuple[np.ndarray, list[str]]:
        """
        Select features using a tree-based model's feature importances.

        Parameters
        ----------
        estimator : sklearn estimator (default: ExtraTreesClassifier)
        threshold : str | float   passed to SelectFromModel
        """
        if estimator is None:
            from sklearn.ensemble import ExtraTreesClassifier
            estimator = ExtraTreesClassifier(n_estimators=100, random_state=self.random_state)

        selector = SelectFromModel(estimator, threshold=threshold)
        X_sel    = selector.fit_transform(X, y)
        mask     = selector.get_support()
        selected = [name for name, kept in zip(feature_names, mask) if kept]
        logger.info("Model-based selection: %d → %d features", X.shape[1], len(selected))
        return X_sel, selected

    # ── Permutation importance ─────────────────────────────────────────────────

    @staticmethod
    def permutation_feature_importance(
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        n_repeats: int = 10,
        random_state: int = 42,
        scoring: str = "roc_auc",
    ) -> pd.DataFrame:
        """
        Compute permutation importance.

        Returns
        -------
        pd.DataFrame with columns ['feature', 'importance_mean', 'importance_std']
        sorted descending by importance_mean.
        """
        result = permutation_importance(
            estimator, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=-1,
        )
        imp_df = pd.DataFrame({
            "feature":          feature_names,
            "importance_mean":  result.importances_mean,
            "importance_std":   result.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
        return imp_df

    # ── Dimensionality reduction ───────────────────────────────────────────────

    def pca_reduction(
        self,
        X: np.ndarray,
        n_components: int | float = 0.95,
        fit: bool = True,
    ) -> tuple[np.ndarray, PCA]:
        """
        Reduce dimensionality via PCA.

        Parameters
        ----------
        n_components : int | float
            Number of components or variance fraction to retain.
        """
        if fit:
            self._pca = PCA(n_components=n_components, random_state=self.random_state)
            X_reduced = self._pca.fit_transform(X)
        else:
            X_reduced = self._pca.transform(X)
        logger.info("PCA: %d → %d components (%.1f%% variance)",
                    X.shape[1], X_reduced.shape[1],
                    self._pca.explained_variance_ratio_.sum() * 100)
        return X_reduced, self._pca

    def umap_reduction(
        self,
        X: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> np.ndarray:
        """
        Non-linear dimensionality reduction via UMAP (for visualization).

        Requires: pip install umap-learn
        """
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=self.random_state,
        )
        X_emb = reducer.fit_transform(X)
        logger.info("UMAP: %d → %d dimensions", X.shape[1], n_components)
        return X_emb

    # ── Interaction / polynomial features ────────────────────────────────────

    @staticmethod
    def add_polynomial_features(
        X: np.ndarray,
        feature_names: list[str],
        degree: int = 2,
        interaction_only: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Add polynomial / interaction features.

        Parameters
        ----------
        interaction_only : bool
            If True, only cross-product terms (no x^2, x^3, …).
        """
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only,
                                  include_bias=False)
        X_poly = poly.fit_transform(X)
        names  = poly.get_feature_names_out(feature_names)
        logger.info("Polynomial features: %d → %d", X.shape[1], X_poly.shape[1])
        return X_poly, list(names)

    # ── Feature ranking summary ───────────────────────────────────────────────

    @staticmethod
    def rank_features(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """
        Compute mutual information, ANOVA-F, and Spearman correlations
        for a quick feature-ranking summary.

        Returns
        -------
        pd.DataFrame with one row per feature, sorted by MI descending.
        """
        from scipy.stats import spearmanr

        mi    = mutual_info_classif(X, y, random_state=42)
        f_val, p_val = f_classif(X, y)

        spearman_r = []
        for i in range(X.shape[1]):
            r, _ = spearmanr(X[:, i], y)
            spearman_r.append(r)

        return (
            pd.DataFrame({
                "feature":        feature_names,
                "mutual_info":    mi,
                "anova_f":        f_val,
                "anova_p":        p_val,
                "spearman_r":     spearman_r,
                "spearman_abs_r": np.abs(spearman_r),
            })
            .sort_values("mutual_info", ascending=False)
            .reset_index(drop=True)
        )
