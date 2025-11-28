"""
Train a simple linear regression model that predicts daily sales volumes.

The script consumes the CSV stored under `<repo>/../data/mlops/raw/sample_sales.csv`
or the path pointed to by the `ML_DATA_ROOT` environment variable. It evaluates
the model and writes both the fitted estimator and summary metrics to disk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mlflow.data.pandas_dataset import PandasDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "data" / "mlops"
CONTAINER_DATA_ROOT = Path("/app_data")
EXTERNAL_DATA_ROOT = Path(
    os.environ.get("ML_DATA_ROOT")
    or (DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else CONTAINER_DATA_ROOT)
)
RAW_DATA_PATH = EXTERNAL_DATA_ROOT / "raw" / "sample_sales.csv"
MODEL_PATH = PROJECT_ROOT / "models/sales_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports/sales_metrics.json"
DEFAULT_EXPERIMENT = "SalesForecasting"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"

FEATURE_COLUMNS = [
    "marketing_spend",
    "store_visits",
    "online_ads",
    "discount_rate",
    "competitor_price",
]


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the CSV dataset into numpy arrays plus the original DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    missing_cols = set(FEATURE_COLUMNS + ["sales"]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_cols)}")

    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = df["sales"].to_numpy(dtype=np.float32)
    return features, targets, df


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected dataset at {RAW_DATA_PATH}. Place your raw CSV there or set ML_DATA_ROOT."
        )
    X, y, dataset_df = load_dataset(RAW_DATA_PATH)
    print(f"Training data loaded from {RAW_DATA_PATH}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }
    dataset_record = PandasDataset.from_pandas(
        dataset_df,
        source=str(RAW_DATA_PATH),
        name="sales_forecasting_dataset",
        targets="sales",
        feature_names=FEATURE_COLUMNS,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))
    with mlflow.start_run():
        mlflow.log_params(
            {
                "regressor": "LinearRegression",
                "test_size": 0.2,
                "random_state": 42,
                "data_path": str(RAW_DATA_PATH),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="sales_model")
        mlflow.log_artifact(METRICS_PATH)
        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_input(dataset_record, context="training")
        print(
            f"Logged run {mlflow.last_active_run().info.run_id} to {mlflow.get_tracking_uri()}"
        )


if __name__ == "__main__":
    main()
