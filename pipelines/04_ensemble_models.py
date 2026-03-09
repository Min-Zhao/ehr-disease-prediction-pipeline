"""
04_ensemble_models.py
──────────────────────
Pipeline step 4 — Build and evaluate ensemble models.

Combines the best classical ML + DL models into:
  • Soft Voting Ensemble
  • Stacking (5-fold OOF meta-learning)
  • Blending (holdout meta-learning)
  • Bayesian Model Averaging
  • Rank Averaging

Reads preprocessed splits from results/ and saved classical ML models.

Outputs:
  • results/ensemble_results.csv
  • results/figures/roc_curves_ensemble.png
  • results/figures/model_comparison_heatmap_all.png

Usage
─────
    python pipelines/04_ensemble_models.py
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

from src.evaluation import ModelEvaluator
from src.models.classical_ml import (
    LightGBMModel,
    LogisticRegressionModel,
    RandomForestModel,
    SVMModel,
    XGBoostModel,
)
from src.models.deep_learning import MLPModel
from src.models.ensemble import (
    BayesianModelAveraging,
    BlendingEnsemble,
    RankAveragingEnsemble,
    StackingEnsemble,
    VotingEnsemble,
)
from src.visualization import ResultsVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


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


def _load_or_build_base_estimators(models_dir: Path, seed: int):
    """
    Try to load previously saved sklearn estimators.
    Fall back to freshly built (unfitted) estimators if pkl files are missing.
    """
    estimator_defs = {
        "Logistic Regression": LogisticRegressionModel,
        "Random Forest":       RandomForestModel,
        "XGBoost":             XGBoostModel,
        "LightGBM":            LightGBMModel,
        "SVM":                 SVMModel,
    }

    estimators = []
    for name, cls in estimator_defs.items():
        key = name.lower().replace(" ", "_")
        pkl = models_dir / f"{key}.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                obj = pickle.load(f)
            # Extract sklearn estimator if wrapped
            est = getattr(obj, "estimator", obj)
            logger.info("  Loaded saved estimator: %s", name)
        else:
            obj = cls(random_state=seed)
            est = getattr(obj, "estimator", obj)
            logger.info("  Built fresh estimator: %s", name)
        estimators.append((name, est))

    return estimators


def main(args):
    cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed        = cfg["project"]["random_seed"]
    out_cfg     = cfg["output"]
    results_dir = ROOT / out_cfg["results_dir"]
    models_dir  = ROOT / out_cfg.get("models_dir", "results/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(results_dir)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    logger.info("Train+val:%d  Test:%d", len(y_trainval), len(y_test))

    # ── Base estimators ───────────────────────────────────────────────────────
    base_estimators = _load_or_build_base_estimators(models_dir, seed)

    # ── Define ensembles ──────────────────────────────────────────────────────
    ensembles = {
        "Soft Voting": VotingEnsemble(
            estimators=base_estimators, voting="soft"
        ),
        "Hard Voting": VotingEnsemble(
            estimators=base_estimators, voting="hard"
        ),
        "Stacking": StackingEnsemble(
            base_estimators=base_estimators,
            n_splits=5, use_proba=True, passthrough=False,
            random_state=seed,
        ),
        "Blending": BlendingEnsemble(
            base_estimators=base_estimators,
            holdout_fraction=0.2, random_state=seed,
        ),
        "Bayesian Averaging": BayesianModelAveraging(
            estimators=base_estimators, n_splits=5, random_state=seed,
        ),
        "Rank Averaging": RankAveragingEnsemble(
            estimators=base_estimators,
        ),
    }

    evaluator = ModelEvaluator(bootstrap=True, n_bootstrap=500)
    viz       = ResultsVisualizer(
        output_dir = ROOT / out_cfg["figures_dir"],
        dpi        = out_cfg.get("figure_dpi", 150),
    )
    y_probs: dict[str, np.ndarray] = {}

    # ── Train and evaluate ────────────────────────────────────────────────────
    for ens_name, ensemble in ensembles.items():
        logger.info("─" * 60)
        logger.info("Training ensemble: %s", ens_name)
        try:
            ensemble.fit(X_trainval, y_trainval)
            y_prob = ensemble.predict_proba(X_test)[:, 1]
            y_probs[ens_name] = y_prob
            evaluator.add(ens_name, y_test, y_prob)

            key = ens_name.lower().replace(" ", "_")
            with open(models_dir / f"ensemble_{key}.pkl", "wb") as f:
                pickle.dump(ensemble, f)
        except Exception as e:
            logger.error("  Ensemble %s failed: %s", ens_name, e)

    # ── Results ───────────────────────────────────────────────────────────────
    table = evaluator.comparison_table(ci=True)
    out_csv = results_dir / "ensemble_results.csv"
    table.to_csv(out_csv)
    logger.info("\n%s\n", table.to_string())
    logger.info("Results → %s", out_csv)

    roc_data = evaluator.get_roc_data()
    viz.plot_roc_curves(roc_data, title="ROC Curves — Ensemble Models",
                        filename="roc_curves_ensemble")
    viz.plot_pr_curves(y_test, y_probs,
                       title="Precision-Recall — Ensemble Models",
                       filename="pr_curves_ensemble")
    viz.plot_calibration(y_test, y_probs,
                         title="Calibration — Ensemble Models",
                         filename="calibration_ensemble")

    logger.info("Pipeline 04 complete.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    main(parser.parse_args())
