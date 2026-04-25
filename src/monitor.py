"""
Production monitoring for the deployed retail-sales-forecaster.

Simulates a stream of incoming "production" data by replaying the held-out
test period (June-July 2015) week-by-week through the deployed serving
endpoint. For each weekly window we:

    1. Call POST http://127.0.0.1:5001/invocations -> get predictions
    2. Compute per-window metrics (RMSE, MAE, R^2, mean error)
    3. Log them to MLflow as a TIME SERIES (metric.step = window_id),
       which the UI renders as a line chart automatically.
    4. Run Evidently's data-drift report comparing the window's feature
       distribution against the training reference.
    5. Save the Evidently HTML report as an MLflow artifact.
    6. Print a console alert if window RMSE exceeds an alarm threshold.

Two flavors of drift covered
----------------------------
    * Data drift   = inputs shifted (Evidently reports per-feature stats).
    * Concept drift = errors degrade over time (rising RMSE per window).

Prerequisites
-------------
    * MLflow tracking server at http://127.0.0.1:5555
    * Model serving process at http://127.0.0.1:5001
        (start with: bash scripts/serve_model.sh)
    * data/processed/sales.parquet exists

Run
---
    python src/monitor.py                # default: weekly windows
    python src/monitor.py --window-days 14
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow

# Evidently is heavyweight; suppress its noisy deprecation warnings during
# import so the script output stays readable. Also silence numpy's
# benign "invalid value in divide" warnings from Evidently's correlation
# computation (happens when a feature has zero variance in a window).
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="evidently")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
from evidently.report import Report  # noqa: E402
from evidently.metric_preset import DataDriftPreset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROCESSED_DIR, configure_mlflow  # noqa: E402
from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402

ENDPOINT = "http://127.0.0.1:5001/invocations"

# Anything BEFORE this is "training" reference; after is "production" stream.
TRAIN_END = pd.Timestamp("2015-06-01")

# RMSE threshold for the alert (1.3x the global test RMSE of ~969).
ALERT_RMSE = 1300.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_reference_and_stream() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference = training data; stream = held-out test period."""
    df = pd.read_parquet(PROCESSED_DIR / "sales.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    reference = df[df["Date"] < TRAIN_END].copy()
    stream = df[df["Date"] >= TRAIN_END].sort_values("Date").copy()
    return reference, stream


# ---------------------------------------------------------------------------
# Endpoint client
# ---------------------------------------------------------------------------
def predict_batch(X: pd.DataFrame) -> np.ndarray:
    """POST a feature batch to the live serving endpoint."""
    payload = {
        "dataframe_split": {
            "columns": X.columns.tolist(),
            "data": X.values.tolist(),
        }
    }
    try:
        r = requests.post(ENDPOINT, json=payload, timeout=60)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not reach {ENDPOINT}. Is the model server running?\n"
            f"  Start it with: bash scripts/serve_model.sh"
        )
    r.raise_for_status()
    body = r.json()
    return np.asarray(body.get("predictions", body), dtype=float)


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
def compute_drift_report(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    out_html: Path,
) -> dict:
    """Run Evidently's DataDriftPreset and save the HTML report.

    Returns a small dict with the headline drift summary so we can log
    summary metrics to MLflow alongside the full HTML artifact.
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_features, current_data=current_features)
    report.save_html(str(out_html))
    summary = report.as_dict()

    # The DataDriftPreset rolls up to: how many features drifted?
    drift_summary = summary["metrics"][0]["result"]
    return {
        "n_features": int(drift_summary["number_of_columns"]),
        "n_drifted": int(drift_summary["number_of_drifted_columns"]),
        "drift_share": float(drift_summary["share_of_drifted_columns"]),
        "dataset_drift": bool(drift_summary["dataset_drift"]),
    }


# ---------------------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay production traffic + monitor.")
    p.add_argument("--window-days", type=int, default=7,
                   help="Size of each monitoring window in days (default 7).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_mlflow()

    print("[monitor] Loading reference (train) + stream (test) ...")
    reference, stream = load_reference_and_stream()
    print(f"[monitor] reference: {len(reference):,} rows  "
          f"stream: {len(stream):,} rows")

    # Slice the stream into N-day windows
    stream_dates = np.sort(stream["Date"].dt.normalize().unique())
    window_starts = stream_dates[::args.window_days]
    print(f"[monitor] {len(window_starts)} window(s) of {args.window_days} days each")

    artifact_root = Path("artifacts") / f"monitor-{datetime.now():%Y%m%d-%H%M%S}"
    artifact_root.mkdir(parents=True, exist_ok=True)

    run_name = f"monitoring-run-{datetime.now():%Y%m%d-%H%M%S}"
    print(f"[monitor] Starting MLflow run: {run_name}\n")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "monitoring_run": "true",
            "model_under_test": "retail-sales-forecaster/Production",
            "window_days": str(args.window_days),
        })
        mlflow.log_param("endpoint", ENDPOINT)
        mlflow.log_param("alert_rmse_threshold", ALERT_RMSE)
        mlflow.log_param("window_days", args.window_days)
        mlflow.log_param("n_windows", len(window_starts))
        mlflow.log_param("reference_rows", len(reference))

        # Header
        print(f"{'win':>3}  {'start':<10}    {'end':<10}  "
              f"{'rows':>6} {'rmse':>9} {'mae':>9} {'r2':>8} "
              f"{'drift':>7}  {'alert':>7}")
        print("-" * 84)

        n_alerts = 0
        for window_id, start_raw in enumerate(window_starts, start=1):
            # Convert numpy datetime64 -> pandas Timestamp for clean ops
            start = pd.Timestamp(start_raw)
            end = start + pd.Timedelta(days=args.window_days)
            mask = (stream["Date"] >= start) & (stream["Date"] < end)
            window = stream.loc[mask].copy()
            if window.empty:
                continue

            X = window[FEATURE_COLUMNS]
            y_true = window[TARGET_COLUMN].values

            preds = predict_batch(X)
            rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
            mae = float(mean_absolute_error(y_true, preds))
            r2 = float(r2_score(y_true, preds))
            mean_err = float(np.mean(preds - y_true))

            # Log per-window metrics with step=window_id -> time series
            mlflow.log_metric("window_rmse", rmse, step=window_id)
            mlflow.log_metric("window_mae", mae, step=window_id)
            mlflow.log_metric("window_r2", r2, step=window_id)
            mlflow.log_metric("window_mean_err", mean_err, step=window_id)
            mlflow.log_metric("window_n_rows", len(window), step=window_id)

            # Drift report — compare this window's features against training
            html_path = artifact_root / f"drift_window_{window_id:02d}.html"
            drift = compute_drift_report(
                reference[FEATURE_COLUMNS], X, html_path
            )
            mlflow.log_metric("window_drifted_features",
                              drift["n_drifted"], step=window_id)
            mlflow.log_metric("window_drift_share",
                              drift["drift_share"], step=window_id)
            mlflow.log_artifact(str(html_path), artifact_path="drift_reports")

            # Alert?
            alerted = rmse > ALERT_RMSE
            if alerted:
                n_alerts += 1
                mlflow.set_tag(f"alert_window_{window_id:02d}",
                               f"rmse={rmse:.2f}")

            start_str = start.strftime("%Y-%m-%d")
            end_str = (end - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"{window_id:>3}  {start_str} -> {end_str}  "
                  f"{len(window):>6} {rmse:>9.2f} {mae:>9.2f} {r2:>8.4f} "
                  f"{drift['n_drifted']:>3}/{drift['n_features']:<3}  "
                  f"{'!!ALERT' if alerted else 'ok':>7}")

        mlflow.log_metric("total_alerts", n_alerts)

        print("-" * 80)
        if n_alerts > 0:
            print(f"[monitor] >> {n_alerts} of {len(window_starts)} "
                  f"window(s) flagged. RMSE exceeded {ALERT_RMSE}. "
                  f"Investigate.")
        else:
            print(f"[monitor] All windows within RMSE threshold {ALERT_RMSE}.")
        print(f"\n[monitor] Run ID: {run.info.run_id}")
        print(f"[monitor] View time-series charts at: http://127.0.0.1:5555/"
              f"#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        print("[monitor] Tip: in the run page, click 'Model metrics' or "
              "'Chart' tab to see RMSE over windows.")


if __name__ == "__main__":
    main()
