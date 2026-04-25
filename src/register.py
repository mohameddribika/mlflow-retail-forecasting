"""
Register the tuned XGBoost model in the MLflow Model Registry and walk
the version through its lifecycle:  None -> Staging -> Production.

Source-run discovery
--------------------
We don't hardcode a run ID. Instead we search the experiment for the run
that carries the tag `tuned_best=true` (set by tune.py). This means the
script keeps working even if you re-run tune.py with different trials —
the latest "best" run wins.

Lifecycle stages used
---------------------
    None      : freshly registered, not yet validated
    Staging   : passed initial validation, candidate for promotion
    Production: actively serving traffic
    Archived  : superseded by a newer Production version

Real MLOps gating
-----------------
Between Staging and Production we run a smoke test:
    - Load the staged model back from the registry by name+stage
    - Predict on a sample from the held-out test set
    - Sanity-check the predictions (positive, finite, reasonable magnitude)
If this passes, we promote to Production. If it fails, we stop with an
explicit error — better to leave the previous Production version in place
than to ship a broken model.

Run
---
    python src/register.py                     # full lifecycle (-> Production)
    python src/register.py --stop-at staging   # stop after Staging
    python src/register.py --stop-at none      # just register (no transitions)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    EXPERIMENT_NAME,
    PROCESSED_DIR,
    REGISTERED_MODEL_NAME,
    configure_mlflow,
)
from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402

STAGES_ORDERED = ["none", "staging", "production"]


# ---------------------------------------------------------------------------
# Find the source run
# ---------------------------------------------------------------------------
def find_tuned_best_run() -> str:
    """Locate the most recent run tagged tuned_best=true. Returns its ID."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="tags.tuned_best = 'true'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            "No run found with tag tuned_best=true. "
            "Run `python src/tune.py` first."
        )
    run_id = runs.iloc[0]["run_id"]
    print(f"[register] Source run: {run_id}")
    print(f"[register] test_rmse:  "
          f"{runs.iloc[0]['metrics.test_rmse']:.2f}")
    print(f"[register] test_r2:    "
          f"{runs.iloc[0]['metrics.test_r2']:.4f}")
    return run_id


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
def register_version(client: MlflowClient, run_id: str) -> str:
    """Register the run's model artifact. Returns the new version number."""
    model_uri = f"runs:/{run_id}/model"

    # Ensure the registered-model entry exists (idempotent).
    try:
        client.create_registered_model(
            name=REGISTERED_MODEL_NAME,
            description=(
                "Daily store-level sales forecaster for the Rossmann dataset. "
                "XGBoost regressor tuned via Hyperopt TPE search."
            ),
            tags={"task": "regression", "domain": "retail",
                  "framework": "xgboost"},
        )
        print(f"[register] Created registered model: {REGISTERED_MODEL_NAME}")
    except RestException as e:
        # Already exists — that's fine, we're just adding a new version.
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            print(f"[register] Registered model already exists: "
                  f"{REGISTERED_MODEL_NAME}")
        else:
            raise

    # Create a new version pointing at this run's model artifact.
    result = mlflow.register_model(model_uri=model_uri,
                                   name=REGISTERED_MODEL_NAME)
    version = result.version
    print(f"[register] Created version: v{version}")

    # Attach description + tags to this specific version.
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
        description=(
            f"Tuned XGBoost from run {run_id[:8]}. "
            f"30-trial Hyperopt TPE search, refit on train+val."
        ),
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME, version=version,
        key="source_run_id", value=run_id,
    )
    return version


# ---------------------------------------------------------------------------
# Transition stages
# ---------------------------------------------------------------------------
def transition(client: MlflowClient, version: str, stage: str,
               archive_existing: bool = False) -> None:
    print(f"[register] Transitioning v{version} -> {stage}")
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )


# ---------------------------------------------------------------------------
# Smoke test before promoting Staging -> Production
# ---------------------------------------------------------------------------
def smoke_test_staged_model() -> None:
    """Load the model from the registry by stage, predict on a sample,
    and sanity-check the output. Raise if anything looks off."""
    print("[register] Smoke-testing Staging model ...")
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/Staging"
    model = mlflow.sklearn.load_model(model_uri)

    # Pull a small sample from the held-out test region.
    df = pd.read_parquet(PROCESSED_DIR / "sales.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    sample = df[df["Date"] >= pd.Timestamp("2015-06-01")].head(20)
    X = sample[FEATURE_COLUMNS]
    y_true = sample[TARGET_COLUMN].values

    preds = model.predict(X)

    # Sanity checks: finite, non-negative-ish, roughly the right magnitude
    assert np.all(np.isfinite(preds)), "Predictions contain NaN or inf"
    assert preds.min() > -100, (
        f"Predictions implausibly negative: min={preds.min():.2f}"
    )
    assert preds.max() < 100_000, (
        f"Predictions implausibly large: max={preds.max():.2f}"
    )
    rmse = float(np.sqrt(((preds - y_true) ** 2).mean()))
    assert rmse < 5_000, (
        f"Smoke-test RMSE {rmse:.2f} is unreasonably high (>5000)"
    )

    print(f"[register]   -> sample preds range: "
          f"[{preds.min():.0f}, {preds.max():.0f}]")
    print(f"[register]   -> sample RMSE: {rmse:.2f}  (sanity threshold 5000)")
    print("[register]   -> smoke test PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Register the tuned model and walk lifecycle stages."
    )
    p.add_argument(
        "--stop-at",
        choices=STAGES_ORDERED,
        default="production",
        help="Stop the lifecycle promotion at this stage (default: production).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_mlflow()
    client = MlflowClient()

    print(f"[register] Registry name: {REGISTERED_MODEL_NAME}")
    print(f"[register] Stop at stage: {args.stop_at}\n")

    run_id = find_tuned_best_run()
    version = register_version(client, run_id)

    if args.stop_at == "none":
        print(f"\n[register] Done. v{version} registered, "
              f"left in stage 'None'.")
        return

    # None -> Staging
    transition(client, version, "Staging", archive_existing=False)

    if args.stop_at == "staging":
        print(f"\n[register] Done. v{version} promoted to Staging.")
        return

    # Smoke test before Production
    smoke_test_staged_model()

    # Staging -> Production (archive any previous Production version)
    transition(client, version, "Production", archive_existing=True)

    # Final summary
    mv = client.get_model_version(name=REGISTERED_MODEL_NAME, version=version)
    print("\n" + "=" * 70)
    print("REGISTRY STATE")
    print("=" * 70)
    print(f"Model: {REGISTERED_MODEL_NAME}")
    print(f"Version: {mv.version}")
    print(f"Current stage: {mv.current_stage}")
    print(f"Source run: {mv.source}")
    print(f"Description: {mv.description}")
    print("=" * 70)
    print(f"\nView at: http://127.0.0.1:5555/#/models/"
          f"{REGISTERED_MODEL_NAME}/versions/{version}")


if __name__ == "__main__":
    main()
