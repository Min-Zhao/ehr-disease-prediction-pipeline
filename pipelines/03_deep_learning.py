"""
03_deep_learning.py
───────────────────
Pipeline step 3 — Train and evaluate deep learning models.

Models: MLP · TabNet · FT-Transformer · LSTM (1D static) · 1D-CNN

Reads preprocessed splits from results/ (produced by 01_data_preprocessing.py).

Outputs:
  • results/dl_results.csv
  • results/figures/roc_curves_dl.png
  • results/figures/pr_curves_dl.png
  • results/figures/calibration_dl.png
  • results/models/<model>.pt (PyTorch state dicts)

Usage
─────
    python pipelines/03_deep_learning.py
    python pipelines/03_deep_learning.py --models mlp tabnet --epochs 150
"""

import argparse
import json
import logging
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import ModelEvaluator
from src.models.deep_learning import (
    CNN1DModel,
    DeepLearningModels,
    FTTransformerModel,
    LSTMModel,
    MLPModel,
    TabNetModel,
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


def _build_models(cfg: dict, max_epochs: int | None, args) -> dict:
    dl_cfg  = cfg.get("deep_learning", {})
    epochs  = max_epochs or dl_cfg.get("max_epochs", 200)
    lr      = dl_cfg.get("learning_rate", 1e-3)
    dr      = dl_cfg.get("dropout_rate", 0.3)
    patience= dl_cfg.get("patience", 20)
    bs      = dl_cfg.get("batch_size", 64)
    device  = dl_cfg.get("device", None)
    seed    = cfg["project"]["random_seed"]

    all_models = {
        "mlp": MLPModel(
            hidden_dims=dl_cfg.get("mlp", {}).get("hidden_dims", [256, 128, 64]),
            dropout=dr, batch_norm=True, lr=lr,
            max_epochs=epochs, patience=patience, batch_size=bs,
            device=device, random_state=seed,
        ),
        "tabnet": TabNetModel(
            n_d=dl_cfg.get("tabnet", {}).get("n_d", 32),
            n_a=dl_cfg.get("tabnet", {}).get("n_a", 32),
            max_epochs=epochs, patience=patience, batch_size=max(bs, 256),
            device=device, random_state=seed,
        ),
        "ft_transformer": FTTransformerModel(
            d_token=dl_cfg.get("ft_transformer", {}).get("d_token", 64),
            n_blocks=dl_cfg.get("ft_transformer", {}).get("n_blocks", 3),
            dropout=dr, lr=lr, max_epochs=epochs, patience=patience,
            batch_size=bs, device=device, random_state=seed,
        ),
        "lstm": LSTMModel(
            hidden_size=dl_cfg.get("lstm", {}).get("hidden_size", 128),
            num_layers=dl_cfg.get("lstm", {}).get("num_layers", 2),
            bidirectional=True, dropout=dr, seq_len=1,
            lr=lr, max_epochs=epochs, patience=patience, batch_size=bs,
            device=device, random_state=seed,
        ),
        "cnn_1d": CNN1DModel(
            channels=dl_cfg.get("cnn_1d", {}).get("channels", [64, 128, 64]),
            kernel_sizes=dl_cfg.get("cnn_1d", {}).get("kernel_sizes", [3, 3, 3]),
            dropout=dr, lr=lr, max_epochs=epochs, patience=patience,
            batch_size=bs, device=device, random_state=seed,
        ),
    }

    requested = args.models if args.models else list(all_models.keys())
    return {k: v for k, v in all_models.items() if k in requested}


def main(args):
    cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    out_cfg     = cfg["output"]
    results_dir = ROOT / out_cfg["results_dir"]
    models_dir  = ROOT / out_cfg.get("models_dir", "results/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(results_dir)
    logger.info("Train:%d  Val:%d  Test:%d", len(y_train), len(y_val), len(y_test))

    # ── Build models ─────────────────────────────────────────────────────────
    models = _build_models(cfg, args.epochs, args)

    evaluator = ModelEvaluator(bootstrap=True, n_bootstrap=500)
    viz       = ResultsVisualizer(
        output_dir = ROOT / out_cfg["figures_dir"],
        dpi        = out_cfg.get("figure_dpi", 150),
    )
    y_probs: dict[str, np.ndarray] = {}

    # ── Train and evaluate ────────────────────────────────────────────────────
    for key, model in models.items():
        logger.info("─" * 60)
        logger.info("Training: %s", model.name)

        try:
            model.fit(X_train, y_train, X_val, y_val)
        except Exception as e:
            logger.error("  Training failed: %s", e)
            continue

        y_prob = model.predict_proba(X_test)[:, 1]
        y_probs[model.name] = y_prob
        evaluator.add(model.name, y_test, y_prob)

        # Save model
        if out_cfg.get("save_models", True):
            if hasattr(model, "model_") and hasattr(model.model_, "state_dict"):
                torch.save(model.model_.state_dict(), models_dir / f"{key}.pt")
            else:
                with open(models_dir / f"{key}.pkl", "wb") as f:
                    pickle.dump(model, f)

    if not y_probs:
        logger.error("No models trained successfully.")
        return

    # ── Comparison table ──────────────────────────────────────────────────────
    table = evaluator.comparison_table(ci=True)
    out_csv = results_dir / "dl_results.csv"
    table.to_csv(out_csv)
    logger.info("\n%s\n", table.to_string())
    logger.info("Results → %s", out_csv)

    # ── Plots ─────────────────────────────────────────────────────────────────
    roc_data = evaluator.get_roc_data()
    viz.plot_roc_curves(roc_data, title="ROC Curves — Deep Learning Models",
                        filename="roc_curves_dl")
    viz.plot_pr_curves(y_test, y_probs,
                       title="Precision-Recall — Deep Learning Models",
                       filename="pr_curves_dl")
    viz.plot_calibration(y_test, y_probs,
                         title="Calibration — Deep Learning Models",
                         filename="calibration_dl")

    logger.info("Pipeline 03 complete.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--models",  nargs="+",
                        choices=["mlp", "tabnet", "ft_transformer", "lstm", "cnn_1d"],
                        help="Subset of DL models to train (default: all).")
    parser.add_argument("--epochs",  type=int, default=None,
                        help="Override max epochs from config.")
    main(parser.parse_args())
