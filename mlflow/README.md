# MLflow Server

Instructions assume commands are run from the repository root (`/home/manoj/projects/mlops`).

## Prepare Host Directories
```bash
mkdir -p ../data/mlops/mlruns ../logs/mlops ../temp/mlops ../data/mlops/raw
```

Place any raw datasets (for example `sample_sales.csv`) inside `../data/mlops/raw/`. Training scripts read inputs directly from `../data/mlops` so the repo stays lightweight. When running inside a container, set the `ML_DATA_ROOT` environment variable (e.g., `ML_DATA_ROOT=/app_data`) so training scripts resolve the mounted location.

## Build the Image
```bash
docker build -t mlflow-server -f docker/Dockerfile.mlflow .
```

## Run the Server
```bash
docker run -d --name mlflow-server -p 5000:5000 \
  -v /home/manoj/projects/data/mlops:/app_data \
  -v /home/manoj/projects/logs/mlops:/app_logs \
  -v /home/manoj/projects/temp/mlops:/app_temp \
  mlflow-server \
  mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri file:///app_data/mlruns \
    --default-artifact-root /app_logs
```

Visit `http://localhost:5000` to access the MLflow UI. Artifacts and backend data are written to the mounted host directories.

## Log Models to MLflow
With the server running, execute the training script (either on the host or in the Ray container) so it logs metrics and artifacts to MLflow:
```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
ML_DATA_ROOT=../data/mlops \
python3 src/models/train_sales_model.py
```
Adjust `MLFLOW_TRACKING_URI` if your server listens on a different address.

### Train the XGBoost Sales-Flag Classifier
Seed sample data if you don't already have `sales_flag.csv`, then run the classifier script:
```bash
cp data_templates/sales_flag_sample.csv ../data/mlops/raw/sales_flag.csv

MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
ML_DATA_ROOT=../data/mlops \
python3 src/models/train_sales_flag_model.py
```

## Stop and Cleanup
```bash
docker stop mlflow-server
docker rm mlflow-server
```
