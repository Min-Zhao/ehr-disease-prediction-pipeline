"""
01_data_preprocessing.py
────────────────────────
Pipeline step 1 — Load, validate, preprocess, and split the EHR cohort.

Outputs (saved to results/):
  • X_train.npy, X_val.npy, X_test.npy
  • y_train.npy, y_val.npy, y_test.npy
  • feature_names.json
  • preprocessing_summary.csv
  • results/figures/class_distribution.png
  • results/figures/feature_distributions.png
  • results/figures/correlation_matrix.png

Usage
─────
    python pipelines/01_data_preprocessing.py
    python pipelines/01_data_preprocessing.py --config config/config.yaml --smote
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_preprocessing import EHRPreprocessor
from src.feature_engineering import FeatureEngineer
from src.visualization import ResultsVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg["data"]
    prep_cfg  = cfg["preprocessing"]
    out_cfg   = cfg["output"]
    target    = cfg["project"]["target_column"]
    seed      = cfg["project"]["random_seed"]

    results_dir = ROOT / out_cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    data_path = ROOT / data_cfg["synthetic_path"]
    if not data_path.exists():
        logger.warning("Synthetic data not found. Generating now …")
        import subprocess
        subprocess.run(
            [sys.executable,
             str(ROOT / "data/synthetic/generate_synthetic_data.py")],
            check=True,
        )
    df = EHRPreprocessor.load_csv(data_path)
    logger.info("Loaded dataset: %d rows × %d columns", *df.shape)

    # ── 2. Clinical feature engineering ───────────────────────────────────────
    fe = FeatureEngineer(random_state=seed)
    df = fe.add_clinical_features(df)
    logger.info("After feature engineering: %d columns", len(df.columns))

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    preprocessor = EHRPreprocessor(
        target_col           = target,
        imputation_strategy  = prep_cfg.get("imputation_strategy", "median"),
        scaling              = prep_cfg.get("scaling", "standard"),
        encoding_method      = prep_cfg.get("encoding_method", "onehot"),
        outlier_method       = prep_cfg.get("outlier_method", "iqr"),
        outlier_threshold    = prep_cfg.get("outlier_threshold", 3.0),
        knn_neighbors        = prep_cfg.get("knn_neighbors", 5),
        random_state         = seed,
    )

    summary_df = preprocessor.summary(df)
    summary_path = results_dir / "preprocessing_summary.csv"
    summary_df.to_csv(summary_path)
    logger.info("Preprocessing summary → %s", summary_path)

    X, y, feature_names = preprocessor.fit_transform(df)
    logger.info("Features after preprocessing: %d", X.shape[1])

    # ── 4. Feature ranking ────────────────────────────────────────────────────
    ranking = fe.rank_features(X, y, feature_names)
    ranking.to_csv(results_dir / "feature_ranking.csv", index=False)
    logger.info("Feature ranking → results/feature_ranking.csv")

    # ── 5. Train / val / test split ───────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = EHRPreprocessor.split(
        X, y,
        test_size    = data_cfg.get("test_size", 0.20),
        val_size     = data_cfg.get("val_size",  0.10),
        stratify     = data_cfg.get("stratify",  True),
        random_state = seed,
    )

    # ── 6. Optional SMOTE oversampling on training set ────────────────────────
    if args.smote:
        X_train, y_train = EHRPreprocessor.resample(X_train, y_train,
                                                    method="smote", random_state=seed)

    # ── 7. Save artefacts ─────────────────────────────────────────────────────
    np.save(results_dir / "X_train.npy", X_train)
    np.save(results_dir / "X_val.npy",   X_val)
    np.save(results_dir / "X_test.npy",  X_test)
    np.save(results_dir / "y_train.npy", y_train)
    np.save(results_dir / "y_val.npy",   y_val)
    np.save(results_dir / "y_test.npy",  y_test)

    with open(results_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    logger.info(
        "Saved splits — train:%d  val:%d  test:%d",
        len(y_train), len(y_val), len(y_test),
    )

    # ── 8. Visualisations ─────────────────────────────────────────────────────
    viz = ResultsVisualizer(
        output_dir = ROOT / out_cfg["figures_dir"],
        dpi        = out_cfg.get("figure_dpi", 150),
        fmt        = out_cfg.get("figure_format", "png"),
    )

    viz.plot_class_distribution(y, title="Superimposed PE — Class Distribution")

    numeric_cols = df.select_dtypes(include=["number"]).columns.drop(target, errors="ignore").tolist()
    viz.plot_feature_distributions(df, numeric_cols[:16], hue_col=target)
    viz.plot_correlation_matrix(df, cols=numeric_cols[:20])

    logger.info("Pipeline 01 complete.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--smote",  action="store_true",
                        help="Apply SMOTE oversampling to the training set.")
    main(parser.parse_args())
