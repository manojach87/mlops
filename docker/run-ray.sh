#!/bin/bash
docker run -d \
  -p 8265:8265 \
  -p 10001:10001 \
  -v ~/projects/temp/mlops/ray:/tmp \
  -v ~/projects/temp/mlops/ray:/app_temp \
  -v ~/projects/logs/mlops/ray:/app_logs \
  -v ~/projects/data/mlops/ray:/app_data \
  ray-ml