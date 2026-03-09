"""
evaluation.py
─────────────
Comprehensive model evaluation utilities for binary disease prediction.

Metrics:
  AUROC, AUPRC, accuracy, sensitivity (recall), specificity, PPV (precision),
  NPV, F1, Brier score, Matthews correlation coefficient (MCC),
  calibration error (ECE), Hosmer-Lemeshow test.

Additional capabilities:
  • Bootstrap confidence intervals for all metrics
  • DeLong test for comparing two ROC curves
  • Calibration assessment
  • Cross-validated evaluation
  • Model comparison summary table
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute a comprehensive set of binary classification metrics.

    Parameters
    ----------
    y_true    : array of {0, 1}
    y_prob    : predicted probabilities for the positive class
    threshold : decision threshold (default 0.5)

    Returns
    -------
    dict with all metrics
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "roc_auc":           roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "accuracy":          accuracy_score(y_true, y_pred),
        "sensitivity":       sensitivity,
        "specificity":       specificity,
        "ppv":               ppv,
        "npv":               npv,
        "f1":                f1_score(y_true, y_pred, zero_division=0),
        "brier_score":       brier_score_loss(y_true, y_prob),
        "mcc":               matthews_corrcoef(y_true, y_pred),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    threshold: float  = 0.5,
    random_state: int = 42,
) -> dict[str, dict]:
    """
    Bootstrap confidence intervals for all classification metrics.

    Returns
    -------
    dict  metric → {'mean', 'lower', 'upper'}
    """
    rng = np.random.default_rng(random_state)
    n   = len(y_true)
    boot_results: dict[str, list] = {}

    for _ in range(n_iterations):
        idx  = rng.integers(0, n, size=n)
        yb   = y_true[idx]
        pb   = y_prob[idx]

        # Skip iterations where only one class present
        if len(np.unique(yb)) < 2:
            continue

        m = compute_metrics(yb, pb, threshold=threshold)
        for k, v in m.items():
            boot_results.setdefault(k, []).append(v)

    alpha = 1 - confidence
    ci    = {}
    for metric, values in boot_results.items():
        arr = np.array(values)
        ci[metric] = {
            "mean":  float(np.mean(arr)),
            "lower": float(np.percentile(arr, 100 * alpha / 2)),
            "upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }
    return ci


# ─────────────────────────────────────────────────────────────────────────────
# DeLong test — compare two AUROC values
# ─────────────────────────────────────────────────────────────────────────────

def delong_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
) -> dict[str, float]:
    """
    DeLong et al. (1988) test for comparing two correlated ROC curves.

    Returns
    -------
    dict with 'auc_a', 'auc_b', 'z_statistic', 'p_value'
    """
    def _compute_midrank(x):
        j = np.argsort(x)
        z = x[j]
        n = len(z)
        T = np.zeros(n)
        i = 0
        while i < n:
            a = i
            while (a < n - 1) and (z[a] == z[a + 1]):
                a += 1
            T[i:a + 1] = 0.5 * (i + a)
            i = a + 1
        T[j] = T.copy()
        return T + 1

    def _structural_components(y_true, y_prob):
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        m, n    = len(pos_idx), len(neg_idx)

        pos_scores = y_prob[pos_idx]
        neg_scores = y_prob[neg_idx]

        all_scores = np.concatenate([pos_scores, neg_scores])
        ranks      = _compute_midrank(all_scores)

        pos_ranks  = ranks[:m]
        auc        = (pos_ranks.sum() - m * (m + 1) / 2) / (m * n)

        v_pos = (pos_ranks - np.arange(1, m + 1)) / n
        v_neg = np.zeros(n)
        for j, ns in enumerate(neg_scores):
            v_neg[j] = np.mean(pos_scores > ns) + 0.5 * np.mean(pos_scores == ns)

        return auc, v_pos, v_neg

    auc_a, v_pos_a, v_neg_a = _structural_components(y_true, y_prob_a)
    auc_b, v_pos_b, v_neg_b = _structural_components(y_true, y_prob_b)

    m = (y_true == 1).sum()
    n = (y_true == 0).sum()

    S00 = np.cov(v_pos_a, v_pos_b)[0, 1]
    S11 = np.cov(v_neg_a, v_neg_b)[0, 1]
    Var_a = np.var(v_pos_a) / m + np.var(v_neg_a) / n
    Var_b = np.var(v_pos_b) / m + np.var(v_neg_b) / n
    Cov   = S00 / m + S11 / n

    se = np.sqrt(Var_a + Var_b - 2 * Cov + 1e-12)
    z  = (auc_a - auc_b) / se
    p  = 2 * (1 - stats.norm.cdf(abs(z)))

    return {"auc_a": auc_a, "auc_b": auc_b, "z_statistic": z, "p_value": p}


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibration_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics: ECE, MCE, and reliability curve data.

    Returns
    -------
    dict with 'ece', 'mce', 'fraction_of_positives', 'mean_predicted_value'
    """
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    bin_size = len(y_true) // n_bins

    ece = float(np.mean(np.abs(fraction_pos - mean_pred)))
    mce = float(np.max(np.abs(fraction_pos - mean_pred)))

    return {
        "ece": ece,
        "mce": mce,
        "fraction_of_positives": fraction_pos.tolist(),
        "mean_predicted_value":  mean_pred.tolist(),
    }


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_groups: int = 10,
) -> dict[str, float]:
    """
    Hosmer-Lemeshow goodness-of-fit test for calibration.

    Returns
    -------
    dict with 'statistic' and 'p_value'
    """
    df_hl = pd.DataFrame({"y": y_true, "p": y_prob})
    df_hl["decile"] = pd.qcut(df_hl["p"], q=n_groups, duplicates="drop", labels=False)

    groups  = df_hl.groupby("decile")
    obs_pos = groups["y"].sum()
    obs_neg = groups["y"].apply(lambda x: (x == 0).sum())
    exp_pos = groups["p"].sum()
    exp_neg = groups.apply(lambda g: (1 - g["p"]).sum())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hl_stat = float(
            ((obs_pos - exp_pos) ** 2 / (exp_pos + 1e-9) +
             (obs_neg - exp_neg) ** 2 / (exp_neg + 1e-9)).sum()
        )

    p_val = float(1 - stats.chi2.cdf(hl_stat, df=n_groups - 2))
    return {"statistic": hl_stat, "p_value": p_val}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated evaluation
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    threshold: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Stratified k-fold cross-validation with comprehensive metrics per fold.

    Returns
    -------
    pd.DataFrame with one row per fold + 'mean' and 'std' summary rows.
    """
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows   = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        est = _clone(estimator)
        est.fit(X_tr, y_tr)

        if hasattr(est, "predict_proba"):
            y_prob = est.predict_proba(X_vl)[:, 1]
        else:
            y_prob = est.decision_function(X_vl)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)

        m = compute_metrics(y_vl, y_prob, threshold=threshold)
        m["fold"] = fold
        rows.append(m)
        logger.info("Fold %d — AUROC=%.3f  AUPRC=%.3f  Sensitivity=%.3f  Specificity=%.3f",
                    fold, m["roc_auc"], m["average_precision"],
                    m["sensitivity"], m["specificity"])

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c != "fold"]
    mean_row = df[metric_cols].mean().to_dict()
    std_row  = df[metric_cols].std().to_dict()
    mean_row["fold"] = "mean"
    std_row["fold"]  = "std"
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    return df


def _clone(estimator):
    """Safe clone without sklearn dependency at module level."""
    from sklearn.base import clone
    return clone(estimator)


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison table
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Tracks and compares multiple model evaluation results.

    Usage
    -----
    evaluator = ModelEvaluator()
    evaluator.add("Random Forest", y_test, y_prob_rf)
    evaluator.add("XGBoost",       y_test, y_prob_xgb)
    table = evaluator.comparison_table()
    """

    def __init__(
        self,
        threshold: float = 0.5,
        bootstrap: bool  = True,
        n_bootstrap: int = 1000,
        ci_level: float  = 0.95,
    ):
        self.threshold   = threshold
        self.bootstrap   = bootstrap
        self.n_bootstrap = n_bootstrap
        self.ci_level    = ci_level
        self._results: dict[str, dict] = {}

    def add(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        random_state: int = 42,
    ) -> dict:
        """Evaluate a model and store results."""
        metrics = compute_metrics(y_true, y_prob, self.threshold)

        if self.bootstrap:
            ci = bootstrap_ci(y_true, y_prob, self.n_bootstrap, self.ci_level,
                               self.threshold, random_state)
            metrics["ci"] = ci

        metrics["roc_curve"] = roc_curve(y_true, y_prob)
        self._results[model_name] = {"y_true": y_true, "y_prob": y_prob, **metrics}

        logger.info("[%s] AUROC=%.3f  AUPRC=%.3f  Sensitivity=%.3f  Specificity=%.3f",
                    model_name, metrics["roc_auc"], metrics["average_precision"],
                    metrics["sensitivity"], metrics["specificity"])
        return metrics

    def comparison_table(self, ci: bool = True) -> pd.DataFrame:
        """
        Return a summary DataFrame comparing all evaluated models.

        Columns: model, roc_auc (95%CI), average_precision (95%CI),
                 sensitivity, specificity, ppv, npv, f1, brier_score, mcc
        """
        display_metrics = [
            "roc_auc", "average_precision", "accuracy",
            "sensitivity", "specificity", "ppv", "npv",
            "f1", "brier_score", "mcc",
        ]
        rows = []
        for model, res in self._results.items():
            row = {"model": model}
            for m in display_metrics:
                val = res.get(m, np.nan)
                if ci and "ci" in res:
                    lo = res["ci"].get(m, {}).get("lower", np.nan)
                    hi = res["ci"].get(m, {}).get("upper", np.nan)
                    row[m] = f"{val:.3f} ({lo:.3f}–{hi:.3f})"
                else:
                    row[m] = round(val, 3)
            rows.append(row)

        return pd.DataFrame(rows).set_index("model")

    def pairwise_delong(self, reference_model: str) -> pd.DataFrame:
        """Compare all models against a reference using the DeLong test."""
        ref = self._results[reference_model]
        rows = []
        for name, res in self._results.items():
            if name == reference_model:
                continue
            result = delong_test(ref["y_true"], ref["y_prob"], res["y_prob"])
            rows.append({
                "model_a":      reference_model,
                "model_b":      name,
                "auc_a":        round(result["auc_a"], 3),
                "auc_b":        round(result["auc_b"], 3),
                "z_statistic":  round(result["z_statistic"], 3),
                "p_value":      round(result["p_value"], 4),
                "significant":  result["p_value"] < 0.05,
            })
        return pd.DataFrame(rows)

    def get_roc_data(self) -> dict[str, tuple]:
        """Return {model_name: (fpr, tpr, thresholds)} for plotting."""
        return {name: res["roc_curve"] for name, res in self._results.items()}

    def get_results(self) -> dict[str, dict]:
        return self._results
