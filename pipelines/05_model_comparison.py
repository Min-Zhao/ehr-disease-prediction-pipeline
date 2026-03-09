"""
05_model_comparison.py
───────────────────────
Pipeline step 5 — Consolidate all results, run statistical comparisons,
compute SHAP interpretability, and generate a final report.

Reads:
  • results/classical_ml_results.csv
  • results/dl_results.csv
  • results/ensemble_results.csv
  • results/models/*.pkl  (to compute SHAP)

Outputs:
  • results/all_models_comparison.csv       — unified table
  • results/delong_tests.csv                — pairwise DeLong p-values
  • results/figures/model_comparison_heatmap_all.png
  • results/figures/roc_all_models.png
  • results/figures/shap_summary_<model>.png

Usage
─────
    python pipelines/05_model_comparison.py
    python pipelines/05_model_comparison.py --shap-model random_forest --shap-n 300
"""

import argparse
import json
import logging
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import ModelEvaluator, compute_metrics, delong_test
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


def consolidate_results(results_dir: Path) -> pd.DataFrame:
    """Merge CSV result files from all pipeline stages."""
    frames = {}
    for csv_name in ["classical_ml_results.csv", "dl_results.csv", "ensemble_results.csv"]:
        path = results_dir / csv_name
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            frames[csv_name.replace("_results.csv", "")] = df
        else:
            logger.warning("Not found: %s — run earlier pipeline steps first.", csv_name)

    if not frames:
        raise FileNotFoundError("No result CSV files found. Run pipelines 02-04 first.")

    combined = pd.concat(frames.values())
    return combined


def _load_model(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_shap(
    model,
    X: np.ndarray,
    feature_names: list[str],
    n_samples: int,
    model_name: str,
    viz: ResultsVisualizer,
):
    """Compute and plot SHAP values for a given model."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP analysis.")
        return

    idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sub = X[idx]

    # Try TreeExplainer first (faster for tree models)
    est = getattr(model, "estimator", model)
    try:
        explainer  = shap.TreeExplainer(est)
        shap_vals  = explainer.shap_values(X_sub)
        # For binary: take class-1 SHAP values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    except Exception:
        try:
            explainer = shap.Explainer(est, X_sub)
            shap_vals = explainer(X_sub).values
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
        except Exception as e:
            logger.warning("SHAP failed for %s: %s", model_name, e)
            return

    viz.plot_shap_summary(
        shap_vals, X_sub, feature_names,
        plot_type="dot",
        title=f"SHAP Summary — {model_name}",
        filename=f"shap_summary_{model_name.lower().replace(' ', '_')}",
    )
    viz.plot_shap_summary(
        shap_vals, X_sub, feature_names,
        plot_type="bar",
        title=f"SHAP Feature Importance — {model_name}",
        filename=f"shap_bar_{model_name.lower().replace(' ', '_')}",
    )
    logger.info("SHAP plots saved for %s.", model_name)


def main(args):
    cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    out_cfg     = cfg["output"]
    results_dir = ROOT / out_cfg["results_dir"]
    models_dir  = ROOT / out_cfg.get("models_dir", "results/models")

    viz = ResultsVisualizer(
        output_dir = ROOT / out_cfg["figures_dir"],
        dpi        = out_cfg.get("figure_dpi", 150),
    )

    # ── 1. Consolidate results ────────────────────────────────────────────────
    logger.info("Consolidating results from all pipeline stages …")
    combined = consolidate_results(results_dir)
    combined.to_csv(results_dir / "all_models_comparison.csv")
    logger.info("\n%s\n", combined.to_string())

    # ── 2. Load test set ──────────────────────────────────────────────────────
    _, _, X_test, _, _, y_test = load_splits(results_dir)
    with open(results_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    # ── 3. Re-evaluate models from saved pickles for ROC / DeLong ─────────────
    evaluator = ModelEvaluator(bootstrap=True, n_bootstrap=500)
    loaded_models = {}

    for pkl_path in sorted(models_dir.glob("*.pkl")):
        name = pkl_path.stem.replace("_", " ").title()
        try:
            model = _load_model(pkl_path)
            y_prob = model.predict_proba(X_test)[:, 1]
            evaluator.add(name, y_test, y_prob)
            loaded_models[pkl_path.stem] = model
            logger.info("Re-evaluated: %s  AUROC=%.3f",
                        name, roc_auc_score(y_test, y_prob))
        except Exception as e:
            logger.warning("Could not evaluate %s: %s", pkl_path.name, e)

    if not loaded_models:
        logger.warning("No saved models found. Skipping re-evaluation.")
    else:
        # ── 4. ROC comparison (all models) ─────────────────────────────────────
        viz.plot_roc_curves(
            evaluator.get_roc_data(),
            title="ROC Curves — All Models",
            filename="roc_all_models",
        )

        # ── 5. DeLong pairwise tests ───────────────────────────────────────────
        model_names = list(evaluator.get_results().keys())
        if len(model_names) >= 2:
            # Use the best-AUC model as reference
            best = max(model_names, key=lambda n: evaluator.get_results()[n]["roc_auc"])
            logger.info("DeLong reference model: %s", best)
            delong_df = evaluator.pairwise_delong(best)
            delong_df.to_csv(results_dir / "delong_tests.csv", index=False)
            logger.info("\n%s\n", delong_df.to_string(index=False))

        # ── 6. Heatmap of numerical metrics ───────────────────────────────────
        num_results = {}
        for name, res in evaluator.get_results().items():
            num_results[name] = {
                k: v for k, v in res.items()
                if isinstance(v, float) and k not in ("tp", "tn", "fp", "fn")
                and not k.startswith("ci")
            }
        num_df = pd.DataFrame(num_results).T
        viz.plot_model_comparison_heatmap(
            num_df,
            metric_cols=["roc_auc", "average_precision", "accuracy",
                          "sensitivity", "specificity", "ppv", "npv", "f1"],
            title="Model Performance Comparison",
            filename="model_comparison_heatmap_all",
        )

        # ── 7. SHAP interpretability ───────────────────────────────────────────
        shap_model_key = args.shap_model
        if shap_model_key and shap_model_key in loaded_models:
            logger.info("Computing SHAP for: %s", shap_model_key)
            compute_shap(
                loaded_models[shap_model_key],
                X_test, feature_names,
                n_samples=args.shap_n,
                model_name=shap_model_key.replace("_", " ").title(),
                viz=viz,
            )
        elif args.shap_model:
            # Auto-pick top model
            top_pkl = max(
                (p for p in models_dir.glob("*.pkl") if not p.stem.startswith("ensemble")),
                key=lambda p: p.stat().st_size, default=None,
            )
            if top_pkl:
                model = _load_model(top_pkl)
                compute_shap(model, X_test, feature_names,
                             args.shap_n, top_pkl.stem, viz)

    # ── 8. Print final summary ────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info("\n%s\n", combined.to_string())
    logger.info("Pipeline 05 complete.  All outputs in: %s", results_dir)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config/config.yaml")
    parser.add_argument("--shap-model", default="random_forest",
                        help="Model key for SHAP analysis (default: random_forest).")
    parser.add_argument("--shap-n",     type=int, default=200,
                        help="Number of samples for SHAP (default: 200).")
    main(parser.parse_args())
