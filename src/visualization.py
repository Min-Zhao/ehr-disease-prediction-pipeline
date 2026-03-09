"""
visualization.py
────────────────
Visualization utilities for disease prediction modeling results.

Plots:
  • ROC curves (multi-model overlay)
  • Precision-Recall curves
  • Calibration / reliability diagrams
  • Confusion matrices
  • Feature importance (bar, SHAP summary, SHAP beeswarm, SHAP waterfall)
  • Learning curves
  • Class distribution
  • Model comparison heatmap
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ── Aesthetics ─────────────────────────────────────────────────────────────────
PALETTE = [
    "#4878CF", "#6ACC65", "#D65F5F", "#B47CC7",
    "#C4AD66", "#77BEDB", "#F0A500", "#E15759",
    "#76B7B2", "#FF9DA7",
]
sns.set_theme(style="whitegrid", font_scale=1.15)


# ─────────────────────────────────────────────────────────────────────────────

class ResultsVisualizer:
    """
    High-level visualizer for model evaluation results.

    Parameters
    ----------
    output_dir : str | Path   Directory to save figures.
    dpi        : int          Figure resolution.
    fmt        : str          File format ('png', 'pdf', 'svg').
    """

    def __init__(
        self,
        output_dir: str | Path = "results/figures",
        dpi: int  = 300,
        fmt: str  = "png",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.fmt = fmt

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.output_dir / f"{name}.{self.fmt}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        logger.info("Saved → %s", path)
        return path

    # ── ROC curves ────────────────────────────────────────────────────────────

    def plot_roc_curves(
        self,
        roc_data: dict[str, tuple],
        title: str = "ROC Curves — Model Comparison",
        filename: str = "roc_curves",
    ) -> plt.Figure:
        """
        Plot overlaid ROC curves for multiple models.

        Parameters
        ----------
        roc_data : dict  model_name → (fpr, tpr, thresholds)
        """
        fig, ax = plt.subplots(figsize=(7, 6))

        for i, (name, (fpr, tpr, _)) in enumerate(roc_data.items()):
            auc = roc_auc_score(*_get_yt_from_fpr_tpr(fpr, tpr)) if False else np.trapz(tpr, fpr)
            ax.plot(fpr, tpr, lw=2, color=PALETTE[i % len(PALETTE)],
                    label=f"{name} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
        ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    def plot_roc_with_ci(
        self,
        y_true: np.ndarray,
        y_probs: dict[str, np.ndarray],
        ci_data: Optional[dict] = None,
        title: str = "ROC Curves with 95% CI",
        filename: str = "roc_curves_ci",
    ) -> plt.Figure:
        """Plot ROC curves with bootstrap confidence bands."""
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, (name, y_prob) in enumerate(y_probs.items()):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc         = roc_auc_score(y_true, y_prob)
            color       = PALETTE[i % len(PALETTE)]
            label       = f"{name} (AUC={auc:.3f})"
            if ci_data and name in ci_data:
                lo = ci_data[name].get("roc_auc", {}).get("lower", auc)
                hi = ci_data[name].get("roc_auc", {}).get("upper", auc)
                label = f"{name} (AUC={auc:.3f}, 95% CI [{lo:.3f}–{hi:.3f}])"
            ax.plot(fpr, tpr, lw=2, color=color, label=label)

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        ax.set_xlabel("1 − Specificity", fontsize=12)
        ax.set_ylabel("Sensitivity", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8.5)
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ── Precision-Recall ──────────────────────────────────────────────────────

    def plot_pr_curves(
        self,
        y_true: np.ndarray,
        y_probs: dict[str, np.ndarray],
        title: str = "Precision-Recall Curves",
        filename: str = "pr_curves",
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 6))
        baseline = y_true.mean()
        ax.axhline(baseline, color="gray", lw=1, linestyle="--", label=f"Baseline (prevalence={baseline:.2f})")

        for i, (name, y_prob) in enumerate(y_probs.items()):
            ap = average_precision_score(y_true, y_prob)
            PrecisionRecallDisplay.from_predictions(
                y_true, y_prob, ax=ax,
                name=f"{name} (AP={ap:.3f})",
                color=PALETTE[i % len(PALETTE)],
            )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ── Calibration ───────────────────────────────────────────────────────────

    def plot_calibration(
        self,
        y_true: np.ndarray,
        y_probs: dict[str, np.ndarray],
        n_bins: int = 10,
        title: str = "Calibration (Reliability) Diagram",
        filename: str = "calibration_curves",
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

        for i, (name, y_prob) in enumerate(y_probs.items()):
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
            ax.plot(mean_pred, frac_pos, "s-", lw=2,
                    color=PALETTE[i % len(PALETTE)], label=name)

        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ── Confusion matrix ──────────────────────────────────────────────────────

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        class_names: list[str] = ("No PE", "Superimposed PE"),
        filename: Optional[str] = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=class_names,
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
        plt.tight_layout()
        fname = filename or f"confusion_matrix_{model_name.replace(' ', '_').lower()}"
        self._save(fig, fname)
        return fig

    # ── Feature importance ────────────────────────────────────────────────────

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance",
        filename: str = "feature_importance",
    ) -> plt.Figure:
        """
        Parameters
        ----------
        importance_df : pd.DataFrame
            Must have columns 'feature' and 'importance_mean'.
            Optional 'importance_std' for error bars.
        """
        df = importance_df.head(top_n).sort_values("importance_mean")
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))

        xerr = df["importance_std"] if "importance_std" in df.columns else None
        ax.barh(df["feature"], df["importance_mean"], xerr=xerr,
                color=PALETTE[0], alpha=0.85, edgecolor="white")
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: list[str],
        plot_type: str = "dot",
        title: str = "SHAP Summary Plot",
        filename: str = "shap_summary",
        max_display: int = 20,
    ) -> plt.Figure:
        """
        Plot SHAP summary (beeswarm or bar).

        Requires: pip install shap
        """
        try:
            import shap
        except ImportError:
            raise ImportError("Install shap: pip install shap")

        fig, ax = plt.subplots(figsize=(9, max(5, max_display * 0.38)))
        shap.summary_plot(
            shap_values, X,
            feature_names=feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False,
            plot_size=None,
        )
        plt.title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ── Model comparison heatmap ──────────────────────────────────────────────

    def plot_model_comparison_heatmap(
        self,
        comparison_df: pd.DataFrame,
        metric_cols: Optional[list[str]] = None,
        title: str = "Model Performance Comparison",
        filename: str = "model_comparison_heatmap",
    ) -> plt.Figure:
        """
        Heatmap of model × metric performance (numerical values only).

        Parameters
        ----------
        comparison_df : pd.DataFrame   rows=models, cols=metrics (numerical)
        """
        if metric_cols is None:
            metric_cols = [c for c in comparison_df.columns
                           if comparison_df[c].dtype in [np.float64, np.float32, float]]

        df = comparison_df[metric_cols].astype(float)

        fig, ax = plt.subplots(figsize=(max(8, len(metric_cols) * 1.2),
                                        max(4, len(df) * 0.6)))
        sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd",
                    linewidths=0.5, ax=ax,
                    cbar_kws={"label": "Score"})
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ── Learning curves ───────────────────────────────────────────────────────

    def plot_learning_curve(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        metric: str = "AUROC",
        model_name: str = "Model",
        filename: str = "learning_curve",
    ) -> plt.Figure:
        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(train_sizes, train_mean, "o-", color=PALETTE[0], label="Training")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color=PALETTE[0])
        ax.plot(train_sizes, val_mean, "s-", color=PALETTE[1], label="Validation")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color=PALETTE[1])
        ax.set_xlabel("Training Samples", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"Learning Curve — {model_name}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ── Class distribution ────────────────────────────────────────────────────

    def plot_class_distribution(
        self,
        y: np.ndarray,
        class_names: list[str] = ("No PE", "Superimposed PE"),
        title: str = "Class Distribution",
        filename: str = "class_distribution",
    ) -> plt.Figure:
        counts = np.bincount(y.astype(int))
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(class_names, counts, color=[PALETTE[0], PALETTE[2]], edgecolor="white", width=0.5)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"n={count}\n({count/sum(counts):.1%})",
                    ha="center", va="bottom", fontsize=11)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    # ── EDA helpers ───────────────────────────────────────────────────────────

    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        hue_col: Optional[str] = None,
        ncols: int = 4,
        filename: str = "feature_distributions",
    ) -> plt.Figure:
        nrows = int(np.ceil(len(numeric_cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 4, nrows * 3.5))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            if hue_col:
                for j, (val, grp) in enumerate(df.groupby(hue_col)):
                    axes[i].hist(grp[col].dropna(), bins=30, alpha=0.6,
                                 color=PALETTE[j % len(PALETTE)], label=str(val))
                axes[i].legend(fontsize=7)
            else:
                axes[i].hist(df[col].dropna(), bins=30, color=PALETTE[0], alpha=0.8)
            axes[i].set_title(col, fontsize=9)
            axes[i].tick_params(labelsize=7)

        for j in range(len(numeric_cols), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Feature Distributions", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)
        return fig

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        cols: Optional[list[str]] = None,
        method: str = "spearman",
        filename: str = "correlation_matrix",
    ) -> plt.Figure:
        subset = df[cols] if cols else df.select_dtypes(include=[np.number])
        corr   = subset.corr(method=method)
        mask   = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.6), max(7, len(corr) * 0.55)))
        sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    annot=len(corr) <= 20, fmt=".2f", linewidths=0.3, ax=ax,
                    cbar_kws={"label": f"{method.capitalize()} ρ"})
        ax.set_title(f"Feature Correlation Matrix ({method.capitalize()})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_yt_from_fpr_tpr(fpr, tpr):
    """Placeholder — real AUC comes from roc_auc_score; this path unused."""
    return np.array([0, 1]), np.array([0, 1])
