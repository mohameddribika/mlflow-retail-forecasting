# MLflow Retail Sales Forecasting — End-to-End MLOps Pipeline

**Course:** AIN-3009 MLOps — Bahçeşehir University
**Author:** Mohamed Dribika
**Instructor:** Gökşin BAKIR
**Repository:** https://github.com/mohameddribika/mlflow-retail-forecasting

This project demonstrates an end-to-end Machine Learning lifecycle management
system for retail sales/demand forecasting using **MLflow** as the central
MLOps platform. It covers experiment tracking, training, hyperparameter tuning,
model registry, deployment, and post-deployment performance monitoring.

## Dataset

Real Kaggle data: **[Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)**
— 1,115 stores in Germany, daily sales 2013-01-01 to 2015-07-31, ~844K cleaned rows.
The loader (`src/data_loader.py`) handles download via `kagglehub` / `kaggle` CLI
or accepts a manually-downloaded zip dropped into `data/raw/`.

## Pipeline Results (test set, June–July 2015)

| Model | Test RMSE | Test R² | Notes |
|---|---|---|---|
| Linear Regression (baseline) | 2,671 | 0.263 | Underfits — establishes the floor |
| RandomForest (100 trees) | 1,628 | 0.726 | Healthy baseline tree model |
| RandomForest (200 trees) | 1,266 | 0.834 | Stronger but slow to train |
| GradientBoosting (default) | 1,772 | 0.676 | Default hyperparameters underperform |
| XGBoost (default) | 1,153 | 0.863 | Best out-of-the-box result |
| **XGBoost (Hyperopt-tuned)** | **969** | **0.903** | Production model |

Tuning achieved a **64% RMSE reduction** vs the baseline.

---

## Project Structure

```
mlflow-retail-forecasting/
├── data/
│   ├── raw/              # Original dataset (gitignored)
│   └── processed/        # Cleaned/feature-engineered data (gitignored)
├── src/
│   ├── data_loader.py    # Dataset download + loading
│   ├── preprocessing.py  # Feature engineering pipeline
│   ├── train.py          # Multi-model training with MLflow tracking
│   ├── tune.py           # Hyperopt-based hyperparameter tuning
│   ├── register.py       # Promote best model to Model Registry
│   ├── serve_client.py   # Sample client to call deployed model
│   ├── monitor.py        # Performance + drift monitoring
│   └── utils.py          # Shared helpers
├── notebooks/
│   └── 01_eda.ipynb      # Exploratory data analysis
├── models/               # Local model artifacts (gitignored)
├── docker/
│   ├── Dockerfile.mlflow
│   ├── Dockerfile.serve
│   └── docker-compose.yml
├── scripts/
│   └── start_mlflow.sh   # Convenience launcher for local MLflow UI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart — All-Docker Workflow (recommended)

The MLflow tracking server, the PostgreSQL backend, and the model serving
endpoint all run in containers. Local Python is only needed to execute the
training/tuning/monitoring scripts (which talk to the dockerized MLflow
over HTTP).

```bash
# 1. Create virtual environment for the training scripts
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Bring up MLflow + PostgreSQL in Docker  (in its own terminal)
cd docker
docker compose up --build
# MLflow UI now live at http://localhost:5555 (PostgreSQL-backed)

# 3. In another terminal: run the pipeline against the dockerized MLflow
cd ~/Desktop/mlflow-retail-forecasting
source venv/bin/activate
python src/data_loader.py     # Download + cache dataset
python src/train.py           # Train 5 model variants
python src/tune.py            # Hyperopt tuning of XGBoost
python src/register.py        # Promote best model to Production
python src/monitor.py         # Per-window metrics + drift detection

# 4. Bring up the serving container (after the model is in Production)
cd docker
docker compose --profile with-serve up serve --build
# Model now serving at http://localhost:5001/invocations
```

| URL                              | What it is                              |
|----------------------------------|-----------------------------------------|
| http://localhost:5555            | MLflow UI (dockerized, PostgreSQL)      |
| http://localhost:5001/invocations| Model serving REST endpoint (dockerized)|

See [`docker/README.md`](docker/README.md) for additional Docker details
(volumes, cleanup, etc.).

---

## MLflow Lifecycle Stages Demonstrated

| Stage | Module | Purpose |
|---|---|---|
| Experiment Tracking | `train.py` | Log params, metrics, artifacts for every run |
| Hyperparameter Tuning | `tune.py` | TPE search via Hyperopt, each trial as nested run |
| Model Registry | `register.py` | Versioning + None → Staging → Production transitions |
| Deployment | `docker compose --profile with-serve up serve` | REST endpoint for the Production model |
| Monitoring | `monitor.py` | Track live metrics + detect feature/prediction drift |

---

## License

For academic use as part of AIN-3009 MLOps coursework at Bahçeşehir University.
