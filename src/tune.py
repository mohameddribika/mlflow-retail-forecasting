"""
Hyperparameter tuning for XGBoost using Hyperopt + nested MLflow runs.

Architecture
------------
    PARENT MLflow run  (run_name = "xgboost-tuning-<ts>", tag tuning_run=true)
       |
       |-- nested run "trial-001"  (params, val_rmse)
       |-- nested run "trial-002"
       |-- ...
       |-- nested run "trial-030"
       |
       +-- final tuned model logged HERE on the parent (after refit on
           train_inner + validation, evaluated on the held-out test set)

Three-way data split (chronological, no leakage)
-----------------------------------------------
    train_inner :  Date <  2015-05-01    (~700K rows) -- used to fit each trial
    validation  :  2015-05-01 - 2015-06-01 (~85K)    -- Hyperopt's objective
    test        :  Date >= 2015-06-01    (~58K)      -- final unbiased eval

Why TPE (Tree-structured Parzen Estimator)?
------------------------------------------
TPE models which hyperparameters tend to work and biases future trials
toward promising regions. Reaches near-optimal in ~10x fewer trials than
random search. This is the standard choice for serious tuning.

Run
---
    python src/tune.py                 # default: 30 trials
    python src/tune.py --trials 50     # more trials
    python src/tune.py --trials 10     # quick smoke test
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
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROCESSED_DIR, configure_mlflow  # noqa: E402
from preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
)

# ---------------------------------------------------------------------------
# Date boundaries for the three-way split
# ---------------------------------------------------------------------------
VAL_START = pd.Timestamp("2015-05-01")
TEST_START = pd.Timestamp("2015-06-01")


# ---------------------------------------------------------------------------
# Data loading + three-way chronological split
# ---------------------------------------------------------------------------
def load_three_way_split() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]:
    parquet = PROCESSED_DIR / "sales.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"{parquet} not found. Run `python src/data_loader.py` first."
        )
    df = pd.read_parquet(parquet)
    df["Date"] = pd.to_datetime(df["Date"])

    train_df = df[df["Date"] < VAL_START].copy()
    val_df = df[(df["Date"] >= VAL_START) & (df["Date"] < TEST_START)].copy()
    test_df = df[df["Date"] >= TEST_START].copy()

    return (
        train_df[FEATURE_COLUMNS], val_df[FEATURE_COLUMNS],
        test_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN], val_df[TARGET_COLUMN],
        test_df[TARGET_COLUMN],
    )


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Hyperopt search space for XGBoost
# ---------------------------------------------------------------------------
# `hp.quniform` returns floats; we wrap with `scope.int(...)` for params that
# must be integers. `hp.loguniform(log_a, log_b)` is the right family for
# learning_rate (which lives on a log scale).
SEARCH_SPACE: dict[str, Any] = {
    "n_estimators": scope.int(hp.quniform("n_estimators", 100, 500, 50)),
    "max_depth": scope.int(hp.quniform("max_depth", 4, 12, 1)),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "subsample": hp.uniform("subsample", 0.7, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.7, 1.0),
    "min_child_weight": scope.int(
        hp.quniform("min_child_weight", 1, 10, 1)
    ),
}


# ---------------------------------------------------------------------------
# Build the pipeline for a candidate set of hyperparameters
# ---------------------------------------------------------------------------
def build_xgb_pipeline(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                XGBRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    subsample=params["subsample"],
                    colsample_bytree=params["colsample_bytree"],
                    min_child_weight=params["min_child_weight"],
                    n_jobs=-1,
                    tree_method="hist",
                    random_state=42,
                    verbosity=0,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Counter so we can name nested runs trial-001, trial-002, ...
# ---------------------------------------------------------------------------
_trial_counter = {"n": 0}


def make_objective(
    X_train: pd.DataFrame, X_val: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series,
):
    """Closure that returns the Hyperopt objective.

    Each call:
      - Opens a NESTED MLflow run.
      - Trains XGBoost with the candidate hyperparameters on train_inner.
      - Evaluates on validation.
      - Logs params + val metrics + fit time.
      - Returns {"loss": val_rmse, "status": STATUS_OK} -> Hyperopt minimizes.
    """

    def objective(params: dict[str, Any]) -> dict[str, Any]:
        _trial_counter["n"] += 1
        trial_id = _trial_counter["n"]
        run_name = f"trial-{trial_id:03d}"

        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tag("trial_id", trial_id)

            # Log all hyperparameters for this trial
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Build + train + evaluate
            t0 = time.perf_counter()
            pipeline = build_xgb_pipeline(params)
            pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - t0

            metrics = evaluate(y_val, pipeline.predict(X_val))
            mlflow.log_metric("val_rmse", metrics["rmse"])
            mlflow.log_metric("val_mae", metrics["mae"])
            mlflow.log_metric("val_r2", metrics["r2"])
            mlflow.log_metric("fit_seconds", fit_seconds)

        pretty_params = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in params.items()
        }
        print(
            f"  {run_name}  fit={fit_seconds:>5.1f}s  "
            f"val_rmse={metrics['rmse']:>8.2f}  val_r2={metrics['r2']:.4f}  "
            f"params={pretty_params}"
        )
        return {"loss": metrics["rmse"], "status": STATUS_OK}

    return objective


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperopt TPE search for XGBoost, logged to MLflow."
    )
    p.add_argument("--trials", type=int, default=30,
                   help="Number of TPE trials (default 30).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_mlflow()

    print("[tune] Loading + 3-way splitting data ...")
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_three_way_split()
    print(f"[tune] train_inner: {len(X_tr):,} rows  "
          f"validation: {len(X_val):,} rows  "
          f"test (held out): {len(X_te):,} rows")
    print(f"[tune] Running {args.trials} TPE trials ...\n")

    parent_name = f"xgboost-tuning-{datetime.now():%Y%m%d-%H%M%S}"
    with mlflow.start_run(run_name=parent_name) as parent_run:
        # ----- Tag the parent so it's discoverable in the UI -----
        mlflow.set_tags({
            "tuning_run": "true",
            "search_algo": "TPE (Hyperopt)",
            "tuned_model": "XGBRegressor",
            "dataset": "rossmann-store-sales",
        })
        mlflow.log_param("n_trials", args.trials)
        mlflow.log_param("search_space", str(SEARCH_SPACE))
        mlflow.log_param("split_strategy", "3-way chronological")
        mlflow.log_param("val_window", f"{VAL_START.date()} - "
                                       f"{TEST_START.date()}")

        # ----- Run the search -----
        trials = Trials()
        objective = make_objective(X_tr, X_val, y_tr, y_val)
        best_raw = fmin(
            fn=objective,
            space=SEARCH_SPACE,
            algo=tpe.suggest,
            max_evals=args.trials,
            trials=trials,
            show_progressbar=False,
        )

        # ----- Convert Hyperopt's raw best back to the typed param dict -----
        # quniform returns floats; we cast back to int where needed.
        best_params = {
            "n_estimators": int(best_raw["n_estimators"]),
            "max_depth": int(best_raw["max_depth"]),
            "learning_rate": float(best_raw["learning_rate"]),
            "subsample": float(best_raw["subsample"]),
            "colsample_bytree": float(best_raw["colsample_bytree"]),
            "min_child_weight": int(best_raw["min_child_weight"]),
        }

        best_val_rmse = min(t["result"]["loss"] for t in trials.trials)

        print("\n" + "=" * 78)
        print("BEST TRIAL")
        print("=" * 78)
        print(f"Best validation RMSE: {best_val_rmse:.2f}")
        print(f"Best params: {best_params}")

        # ----- Refit on train_inner + validation, evaluate on test -----
        # Now that hyperparams are chosen, use ALL non-test data for the
        # final fit. Throwing away validation post-selection would be wasteful.
        print("\n[tune] Refitting best model on train_inner + validation ...")
        X_full = pd.concat([X_tr, X_val], axis=0)
        y_full = pd.concat([y_tr, y_val], axis=0)
        final_pipeline = build_xgb_pipeline(best_params)
        t0 = time.perf_counter()
        final_pipeline.fit(X_full, y_full)
        final_fit_seconds = time.perf_counter() - t0

        test_metrics = evaluate(y_te, final_pipeline.predict(X_te))

        # ----- Log everything onto the PARENT run -----
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("best_val_rmse", best_val_rmse)
        mlflow.log_metric("test_rmse", test_metrics["rmse"])
        mlflow.log_metric("test_mae", test_metrics["mae"])
        mlflow.log_metric("test_r2", test_metrics["r2"])
        mlflow.log_metric("final_fit_seconds", final_fit_seconds)

        # The "tuned-best" tag makes this run trivially findable later when
        # we register it in the Model Registry (Step 6).
        mlflow.set_tag("tuned_best", "true")

        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
            input_example=X_full.head(5),
        )

        summary = (
            f"Tuning run: {parent_name}\n"
            f"Run ID: {parent_run.info.run_id}\n"
            f"Trials: {args.trials}\n"
            f"Best validation RMSE: {best_val_rmse:.4f}\n"
            f"Best params: {best_params}\n\n"
            f"Final test metrics (refit on train+val): {test_metrics}\n"
            f"Final fit seconds: {final_fit_seconds:.2f}\n"
        )
        mlflow.log_text(summary, "tuning_summary.txt")

        print("\n" + "=" * 78)
        print("FINAL TEST METRICS (best model, refit on train+val)")
        print("=" * 78)
        print(f"Test RMSE: {test_metrics['rmse']:.2f}   "
              f"MAE: {test_metrics['mae']:.2f}   "
              f"R^2: {test_metrics['r2']:.4f}")
        print("=" * 78)
        print(f"\nParent run: {parent_run.info.run_id}")
        print(f"View at: http://127.0.0.1:5555/#/experiments/"
              f"{parent_run.info.experiment_id}/runs/"
              f"{parent_run.info.run_id}")


if __name__ == "__main__":
    main()
