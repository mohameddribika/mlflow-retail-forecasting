"""
Sample client for the deployed retail-sales-forecaster.

Sends a small batch of feature rows to the running MLflow serving endpoint
and prints predictions side-by-side with the actual sales values from the
held-out test set. Useful for:

    1. Confirming the endpoint is alive and responding.
    2. Sanity-checking that predictions are in a plausible range.
    3. Demoing real-time inference in your presentation.

Prerequisites
-------------
    * MLflow tracking server running at http://127.0.0.1:5555
    * Model serving process running at http://127.0.0.1:5001
        (start with: bash scripts/serve_model.sh)
    * data/processed/sales.parquet exists

Run
---
    python src/serve_client.py
    python src/serve_client.py --rows 50  (send a larger batch)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROCESSED_DIR  # noqa: E402
from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402

ENDPOINT = "http://127.0.0.1:5001/invocations"


def make_payload(df: pd.DataFrame) -> dict:
    """Build the JSON payload in MLflow's `dataframe_split` format.

    MLflow's serving endpoint accepts several formats. `dataframe_split`
    is the most explicit: it sends column names + values separately,
    avoiding any ambiguity about column order.
    """
    return {
        "dataframe_split": {
            "columns": df.columns.tolist(),
            "data": df.values.tolist(),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a sample batch to the deployed model."
    )
    parser.add_argument("--rows", type=int, default=10,
                        help="How many rows to send (default 10).")
    args = parser.parse_args()

    # Pull a sample from the held-out test region.
    df = pd.read_parquet(PROCESSED_DIR / "sales.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    sample = df[df["Date"] >= pd.Timestamp("2015-06-01")].head(args.rows)

    X = sample[FEATURE_COLUMNS]
    y_true = sample[TARGET_COLUMN].values

    payload = make_payload(X)

    print(f"[client] Sending {len(X)} rows to {ENDPOINT} ...")
    try:
        resp = requests.post(
            ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        print(f"[client] ERROR: could not reach {ENDPOINT}.")
        print("[client] Is the model server running? Start with:")
        print("[client]   bash scripts/serve_model.sh")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"[client] ERROR: HTTP {resp.status_code}")
        print(f"[client] Response: {resp.text}")
        sys.exit(1)

    body = resp.json()
    # MLflow returns {"predictions": [...]} for sklearn models.
    preds = body.get("predictions", body)

    print(f"[client] HTTP {resp.status_code}  "
          f"received {len(preds)} predictions\n")

    # Pretty side-by-side print
    print(f"{'#':>3} {'Store':>6} {'DayOfWeek':>10} {'Promo':>6} "
          f"{'predicted':>12} {'actual':>10} {'abs_err':>10}")
    print("-" * 65)
    for i, (idx, row) in enumerate(sample.head(args.rows).iterrows()):
        pred = float(preds[i])
        actual = float(y_true[i])
        err = abs(pred - actual)
        print(f"{i + 1:>3} {row['Store']:>6} {row['DayOfWeek']:>10} "
              f"{row['Promo']:>6} {pred:>12.2f} {actual:>10.2f} "
              f"{err:>10.2f}")

    abs_errs = [abs(float(preds[i]) - float(y_true[i]))
                for i in range(len(preds))]
    mae = sum(abs_errs) / len(abs_errs)
    print("-" * 65)
    print(f"[client] Mean Absolute Error on this batch: {mae:.2f}")


if __name__ == "__main__":
    main()
