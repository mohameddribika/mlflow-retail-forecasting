"""
Train multiple model variants and log each one as a separate MLflow run.

Architecture
------------
At the top of this file lives MODEL_REGISTRY -- a dict mapping a unique
run name to (estimator class, hyperparameter dict, "needs scaling?" flag,
model family tag). Adding a new model variant is a one-line change.

Each call to `train_one(name, cfg)` opens its own `mlflow.start_run(...)`
block and logs:
  * Parameters: hyperparameters + dataset metadata + pipeline metadata
  * Metrics:    train/test RMSE, MAE, R^2 + training duration in seconds
  * Tags:       model_family ("linear" | "tree"), model_name
  * Artifacts:  the fitted pipeline (deployable) + a run_summary.txt

Why tree models skip StandardScaler
-----------------------------------
Tree-based models (RandomForest, GradientBoosting, XGBoost) are
scale-invariant: a feature scaled by 100 produces the same splits as
the original. Adding StandardScaler to their pipeline doesn't help
accuracy and adds latency at inference. We only scale for linear models.

Run
---
    python src/train.py                                     # train all
    python src/train.py --models random-forest-100          # train one
    python src/train.py --models random-forest-100 xgboost  # train two
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROCESSED_DIR, configure_mlflow  # noqa: E402
from preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
)

# Anything BEFORE this date is train; anything ON OR AFTER is test.
SPLIT_DATE = pd.Timestamp("2015-06-01")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "linear-regression": {
        "estimator": LinearRegression(),
        "params": {},  # LinearRegression has no real hyperparameters worth logging
        "needs_scaling": True,
        "family": "linear",
        "log_with": "sklearn",
    },
    "random-forest-100": {
        "estimator": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        "params": {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 5},
        "needs_scaling": False,
        "family": "tree",
        "log_with": "sklearn",
    },
    "random-forest-200": {
        "estimator": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ),
        "params": {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 3},
        "needs_scaling": False,
        "family": "tree",
        "log_with": "sklearn",
    },
    "gradient-boosting": {
        "estimator": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        ),
        "params": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
        "needs_scaling": False,
        "family": "tree",
        "log_with": "sklearn",
    },
    "xgboost": {
        "estimator": XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            random_state=42,
            tree_method="hist",  # fast histogram algorithm
        ),
        "params": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        "needs_scaling": False,
        "family": "tree",
        "log_with": "xgboost",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronologically split the cleaned dataset into train and test."""
    parquet = PROCESSED_DIR / "sales.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"{parquet} not found. Run `python src/data_loader.py` first."
        )
    df = pd.read_parquet(parquet)
    df["Date"] = pd.to_datetime(df["Date"])

    train_df = df[df["Date"] < SPLIT_DATE].copy()
    test_df = df[df["Date"] >= SPLIT_DATE].copy()

    return (
        train_df[FEATURE_COLUMNS],
        test_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
        test_df[TARGET_COLUMN],
    )


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def build_pipeline(estimator: Any, needs_scaling: bool) -> Pipeline:
    """Build the sklearn Pipeline for a given model.

    Tree models (RF, GB, XGB) skip StandardScaler — they're scale-invariant
    and adding the scaler only slows down inference.
    """
    steps = [("preprocess", build_preprocessor())]
    if needs_scaling:
        # with_mean=False because the OneHotEncoder output is sparse-friendly
        # even when we requested dense; safer default.
        steps.append(("scale", StandardScaler(with_mean=False)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


# ---------------------------------------------------------------------------
# Train a single model variant and log it to MLflow
# ---------------------------------------------------------------------------
def train_one(
    name: str,
    cfg: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """Train one model variant. Returns the test metrics dict."""
    pipeline = build_pipeline(cfg["estimator"], cfg["needs_scaling"])
    run_name = f"{name}-{datetime.now():%Y%m%d-%H%M%S}"
    print(f"\n[train] >>> {run_name}")

    with mlflow.start_run(run_name=run_name) as run:
        # ----- Tags (used for grouping / filtering in the UI) -----
        mlflow.set_tags({
            "model_family": cfg["family"],
            "model_name": name,
            "dataset": "rossmann-store-sales",
        })

        # ----- Parameters -----
        mlflow.log_param("model_type", type(cfg["estimator"]).__name__)
        mlflow.log_param("preprocessing", "OneHot(cat) + numeric passthrough")
        mlflow.log_param("scaled", cfg["needs_scaling"])
        mlflow.log_param("split_strategy", "time-based")
        mlflow.log_param("split_date", str(SPLIT_DATE.date()))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("n_features_raw", len(FEATURE_COLUMNS))
        for k, v in cfg["params"].items():
            mlflow.log_param(k, v)

        # ----- Fit (and time it) -----
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_seconds = time.perf_counter() - t0
        mlflow.log_metric("train_seconds", train_seconds)

        # ----- Evaluate -----
        train_metrics = evaluate(y_train, pipeline.predict(X_train))
        test_metrics = evaluate(y_test, pipeline.predict(X_test))
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # ----- Save the fitted pipeline as an MLflow model -----
        # We use the sklearn flavor for everything because the wrapping
        # Pipeline is sklearn-typed even when the inner estimator is XGBoost.
        # This makes downstream loading uniform.
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_train.head(5),
        )

        # ----- Run summary text artifact -----
        summary = (
            f"Run: {run_name}\nRun ID: {run.info.run_id}\n"
            f"Model: {type(cfg['estimator']).__name__}\n"
            f"Family: {cfg['family']}\n"
            f"Hyperparameters: {cfg['params']}\n"
            f"Train seconds: {train_seconds:.2f}\n\n"
            f"Train metrics: {train_metrics}\n"
            f"Test  metrics: {test_metrics}\n"
        )
        mlflow.log_text(summary, "run_summary.txt")

        # ----- Console output -----
        print(
            f"  fit time: {train_seconds:>6.1f}s  |  "
            f"test_rmse={test_metrics['rmse']:>8.2f}  "
            f"test_mae={test_metrics['mae']:>8.2f}  "
            f"test_r2={test_metrics['r2']:.4f}"
        )

    return test_metrics


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one or more model variants and log to MLflow."
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Subset of models to train. Defaults to all.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.models or list(MODEL_REGISTRY.keys())

    configure_mlflow()
    print("[train] Loading + splitting data ...")
    X_train, X_test, y_train, y_test = load_split()
    print(f"[train] Train rows: {len(X_train):,}   Test rows: {len(X_test):,}")
    print(f"[train] Models to train: {selected}")

    results: dict[str, dict[str, float]] = {}
    for name in selected:
        cfg = MODEL_REGISTRY[name]
        results[name] = train_one(
            name, cfg, X_train, X_test, y_train, y_test
        )

    # ----- Final leaderboard -----
    print("\n" + "=" * 78)
    print("LEADERBOARD (sorted by test RMSE, lower is better)")
    print("=" * 78)
    print(f"{'model':<22} {'test_rmse':>12} {'test_mae':>12} {'test_r2':>10}")
    print("-" * 78)
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["rmse"]):
        print(
            f"{name:<22} {m['rmse']:>12.2f} {m['mae']:>12.2f} {m['r2']:>10.4f}"
        )
    print("=" * 78)
    print("\nView all runs at: http://127.0.0.1:5555")


if __name__ == "__main__":
    main()
