"""
02_classical_ml.py
──────────────────
Pipeline step 2 — Train and evaluate all classical ML models.

Reads preprocessed splits from results/ (produced by 01_data_preprocessing.py)
and trains:
  Logistic Regression · SVM · Random Forest · XGBoost · LightGBM
  CatBoost · Decision Tree · KNN · Naïve Bayes · LDA

Outputs:
  • results/classical_ml_results.csv      — test-set metrics for all models
  • results/figures/roc_curves_classical.png
  • results/figures/pr_curves_classical.png
  • results/figures/feature_importance_<model>.png
  • results/models/<model>.pkl             — saved estimators

Usage
─────
    python pipelines/02_classical_ml.py
    python pipelines/02_classical_ml.py --tune --n-iter 30
"""

import argparse
import json
import logging
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import ModelEvaluator, cross_validate_model
from src.models.classical_ml import ClassicalMLModels
from src.visualization import ResultsVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────

def load_splits(results_dir: Path):
    return (
        np.load(results_dir / "X_train.npy"),
        np.load(results_dir / "X_val.npy"),
        np.load(results_dir / "X_test.npy"),
        np.load(results_dir / "y_train.npy"),
        np.load(results_dir / "y_val.npy"),
        np.load(results_dir / "y_test.npy"),
    )


def main(args):
    cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed         = cfg["project"]["random_seed"]
    out_cfg      = cfg["output"]
    results_dir  = ROOT / out_cfg["results_dir"]
    models_dir   = ROOT / out_cfg.get("models_dir", "results/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(results_dir)
    with open(results_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    # Combine train + val for final model training
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    logger.info("Train+val: %d  |  Test: %d", len(y_trainval), len(y_test))

    # ── Initialise models ────────────────────────────────────────────────────
    factory = ClassicalMLModels()
    models  = factory.get_all()

    evaluator = ModelEvaluator(bootstrap=True, n_bootstrap=500)
    viz       = ResultsVisualizer(
        output_dir = ROOT / out_cfg["figures_dir"],
        dpi        = out_cfg.get("figure_dpi", 150),
    )

    y_probs: dict[str, np.ndarray] = {}

    # ── Train and evaluate each model ────────────────────────────────────────
    for model_key, model in models.items():
        display_name = model.name
        logger.info("─" * 60)
        logger.info("Training: %s", display_name)

        # Optional hyperparameter tuning
        if args.tune:
            logger.info("  Hyperparameter search …")
            try:
                best_est = model.hyperparameter_search(
                    X_train, y_train, method="random"
                )
                # Refit best estimator on train+val
                try:
                    best_est.fit(X_trainval, y_trainval)
                except Exception:
                    best_est.fit(X_train, y_train)
                # Wrap in our model interface for uniform API
                model.estimator = best_est
            except Exception as e:
                logger.warning("  Tuning failed for %s: %s — using defaults.", display_name, e)
                model.fit(X_trainval, y_trainval)
        else:
            model.fit(X_trainval, y_trainval)

        # Predict on test set
        y_prob = model.predict_proba(X_test)[:, 1]
        y_probs[display_name] = y_prob
        evaluator.add(display_name, y_test, y_prob)

        # Save model
        if out_cfg.get("save_models", True):
            pkl_path = models_dir / f"{model_key}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(model, f)

        # Feature importance plot
        imp_df = model.get_feature_importance()
        if imp_df is not None:
            imp_df["feature"] = feature_names[:len(imp_df)]
            imp_df = imp_df.sort_values("importance_mean", ascending=False).head(20)
            viz.plot_feature_importance(
                imp_df,
                title=f"Feature Importance — {display_name}",
                filename=f"feature_importance_{model_key}",
            )

    # ── Cross-validated results (optional deeper analysis) ────────────────────
    if args.cv:
        logger.info("\n5-Fold Cross-Validation on full dataset …")
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.ensemble import RandomForestClassifier as RF
        for model_key, model in models.items():
            try:
                est = model.estimator
                cv_df = cross_validate_model(est, X_trainval, y_trainval)
                cv_df.to_csv(results_dir / f"cv_{model_key}.csv", index=False)
            except Exception as e:
                logger.warning("CV failed for %s: %s", model.name, e)

    # ── Comparison table ──────────────────────────────────────────────────────
    table = evaluator.comparison_table(ci=True)
    out_csv = results_dir / "classical_ml_results.csv"
    table.to_csv(out_csv)
    logger.info("\n%s\n", table.to_string())
    logger.info("Results → %s", out_csv)

    # ── Plots ──────────────────────────────────────────────────────────────────
    roc_data = evaluator.get_roc_data()
    viz.plot_roc_curves(roc_data, title="ROC Curves — Classical ML Models",
                        filename="roc_curves_classical")
    viz.plot_pr_curves(y_test, y_probs,
                       title="Precision-Recall — Classical ML Models",
                       filename="pr_curves_classical")
    viz.plot_calibration(y_test, y_probs,
                         title="Calibration — Classical ML Models",
                         filename="calibration_classical")

    logger.info("Pipeline 02 complete.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--tune",    action="store_true",
                        help="Run randomised hyperparameter search.")
    parser.add_argument("--n-iter",  type=int, default=50,
                        help="Iterations for random search (default 50).")
    parser.add_argument("--cv",      action="store_true",
                        help="Run k-fold CV for each model.")
    main(parser.parse_args())
