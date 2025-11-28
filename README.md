# MLOps Project

This is an MLOps project repository.

## Development Branch
This change is made on the dev branch.

## Test Change
This should fail to commit directly to dev.

## Running the Sales Model
1. Place your CSV dataset (for example `sample_sales.csv`) under `../data/mlops/raw/` when running locally, or under `/app_data/mlops/raw/` when running in the provided containers. The file must have the columns `marketing_spend, store_visits, online_ads, discount_rate, competitor_price, sales`.
2. From the repo root, execute the training script. When you use the default container mounts, run:
   ```bash
   ML_DATA_ROOT=/app_data/mlops python src/models/train_sales_model.py
   ```
   Omit the `ML_DATA_ROOT` variable if your dataset already lives under `../data/mlops`.
3. The trained estimator is written to `models/sales_model.joblib`, and evaluation metrics are saved to `reports/sales_metrics.json`.
