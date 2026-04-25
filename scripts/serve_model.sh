#!/usr/bin/env bash
# Serve the Production version of `retail-sales-forecaster` over HTTP.
#
# CRITICAL: We export MLFLOW_TRACKING_URI so the serve process knows where
# to find the registered model. Without this, MLflow looks at the local
# file store (./mlruns/) and reports "Registered Model not found".
#
# We also pass --env-manager local (formerly --no-conda). By default MLflow
# tries to recreate the model's conda environment for serving — slow and
# brittle. With --env-manager local we just use the active venv, which
# already has the required packages from requirements.txt.
#
# Run from the project root, with the venv activated AND MLflow server
# running on port 5555:
#
#   bash scripts/serve_model.sh
#
# Endpoint will be available at:
#   POST http://127.0.0.1:5001/invocations

set -e

export MLFLOW_TRACKING_URI="http://127.0.0.1:5555"

mlflow models serve \
  --model-uri "models:/retail-sales-forecaster/Production" \
  --host 0.0.0.0 \
  --port 5001 \
  --env-manager local
