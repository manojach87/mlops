# Ray Container

Commands assume you are in `/home/manoj/projects/mlops`. Ray logs should live under `/home/manoj/projects/data/mlops/ray`.

## Prepare Host Directories
```bash
mkdir -p ../data/mlops/ray ../logs/mlops ../temp/mlops
```

## Build the Image
```bash
docker build -t ray-ml -f docker/Dockerfile.ray .
```

## Run the Container
```bash
docker run -d --name ray-server \
  -p 8265:8265 -p 10001:10001 \
  -v /home/manoj/projects/data/mlops:/app_data \
  -v /home/manoj/projects/logs/mlops:/app_logs \
  -v /home/manoj/projects/temp/mlops:/app_temp \
  -v /home/manoj/projects/mlops:/app_code/mlops \
  -e RAY_TMPDIR=/app_data/ray \
  ray-ml \
  ray start --head --dashboard-host=0.0.0.0 --block --temp-dir=/app_data/ray
```

The Ray dashboard will be available at `http://localhost:8265`. Session state and logs are stored on the host in `/home/manoj/projects/data/mlops/ray`, and the repository contents are mounted inside the container at `/app_code/mlops`.

## Run the Sales Model Inside the Container
Ensure your raw dataset exists on the host at `/home/manoj/projects/data/mlops/raw/sample_sales.csv`. Then execute the training script from within the running Ray container:
```bash
docker exec -it ray-server bash -lc "cd /app_code/mlops && \
  ML_DATA_ROOT=/app_data \
  MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  python3 src/models/train_sales_model.py"
```
This uses the `/app_data` mount for inputs and logs artifacts/metrics to the MLflow server running on the host (reachable from the container as `host.docker.internal`). Model and metrics files remain on the host under `models/` and `reports/`.
> If `host.docker.internal` is unavailable on your Docker setup, replace it with the host gateway IP (commonly `172.17.0.1`).

## Stop and Cleanup
```bash
docker stop ray-server
docker rm ray-server
```
