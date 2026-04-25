#!/usr/bin/env bash
# DEPRECATED: This local SQLite-backed MLflow launcher has been replaced
# by the dockerized stack. It is kept here only for emergency local-only
# testing without Docker.
#
# Primary entrypoint is now:
#
#   cd docker && docker compose up --build
#
# That stands up MLflow + PostgreSQL in containers, exposed at the same
# port (5555) the rest of the code already targets. No source changes
# needed to switch between the two.
#
# If you really need a local SQLite MLflow (e.g., your machine has no
# Docker), this script falls back to one on a DIFFERENT port (5560) so it
# never conflicts with the dockerized stack.

set -e

mkdir -p mlartifacts

echo ">>> NOTE: this is the legacy local SQLite-MLflow launcher."
echo ">>> The primary deployment is dockerized. To use it, run:"
echo ">>>   cd docker && docker compose up --build"
echo ""

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --port 5560
