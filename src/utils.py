"""
Shared utilities for the MLflow retail forecasting project.

Centralizes:
- Path constants (data/, models/, etc.) so modules don't hardcode paths.
- MLflow tracking URI configuration so we can switch local <-> Docker easily.
- A single source of truth for the experiment name.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

for _dir in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------
# Use environment variable if set (Docker / CI), else default to local server.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5555")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "retail-sales-forecasting")
REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME", "retail-sales-forecaster"
)


def configure_mlflow() -> None:
    """Point the MLflow client at our tracking server and ensure the
    experiment exists. Call this at the top of every entry-point script."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
