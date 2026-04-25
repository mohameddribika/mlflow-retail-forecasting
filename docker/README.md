# Dockerized MLflow Stack — Primary Deployment

This is the **primary runtime** for the project. MLflow tracking,
PostgreSQL backend, and model serving all live in containers. The local
training scripts run in your venv and talk to the dockerized MLflow over
HTTP at `http://localhost:5555`.

## Services

| Service        | Image / Build       | Port (host -> container) | Purpose                              |
|----------------|---------------------|--------------------------|--------------------------------------|
| postgres       | postgres:16-alpine  | (internal only)          | Backend store for MLflow metadata    |
| mlflow         | Dockerfile.mlflow   | 5555 -> 5000             | MLflow tracking server (UI + API)    |
| serve          | Dockerfile.serve    | 5001 -> 5001             | Model serving (profile: with-serve)  |

## Standard run order

```bash
# 1. Bring up the tracking server + DB
cd docker
docker compose up --build

# 2. (in a separate terminal, with venv activated)
cd ~/Desktop/mlflow-retail-forecasting
source venv/bin/activate
python src/data_loader.py
python src/train.py
python src/tune.py
python src/register.py
python src/monitor.py

# 3. Bring up the serving container once a model is in Production
cd docker
docker compose --profile with-serve up serve --build
```

The serve container is in the `with-serve` Compose profile so that
`docker compose up` doesn't try to start it before a model has been
registered (it would crash on startup with "model not found").

## URLs

- **MLflow UI:** http://localhost:5555
- **Model serving endpoint:** http://localhost:5001/invocations

## Container layout under the hood

```
                  +-------------------+
                  |   Your venv       |
                  |  python src/...   |
                  +---------+---------+
                            |  HTTP (port 5555)
                            v
+---------+        +-------------------+        +----------------+
| serve   | <----- |  mlflow-server    | <----> |  postgres      |
|  :5001  |        |  (tracking API)   |        |  (metadata)    |
+---------+        +-------------------+        +----------------+
     |                       ^
     |  reads model from     |  writes artifacts to
     +---- mlflow_artifacts (shared docker volume) -+
```

The `mlflow_artifacts` volume is mounted into BOTH `mlflow-server` and
`mlflow-serve`, so the serving container can read the same model files
that the training scripts uploaded.

## Cleanup

```bash
docker compose down              # stop containers, keep volumes
docker compose down -v           # stop + delete volumes (full reset)
```

After `down -v`, all training/tuning/registration data in the dockerized
MLflow is gone. You'd re-run the pipeline to repopulate.

## Why PostgreSQL?

| Aspect              | SQLite (legacy local) | PostgreSQL (this stack)        |
|---------------------|-----------------------|--------------------------------|
| Concurrent writers  | One                   | Many                           |
| Network access      | No (file path)        | Yes (TCP)                      |
| Production-ready    | Not really            | Industry standard              |
| Setup overhead      | Zero                  | Container + credentials        |

Real ML teams run MLflow with PostgreSQL because multiple training jobs
write concurrently and the UI is accessed from multiple machines. SQLite
breaks down at any scale beyond one developer.
