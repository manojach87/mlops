"""
Train an XGBoost classifier that predicts whether sales will exceed a target
threshold ("sales_flag") using tabular marketing signals.

The script expects a CSV located under `<repo>/../data/mlops/raw/sales_flag.csv`
or whatever directory `ML_DATA_ROOT` points to. If the CSV does not exist, the
script will attempt to seed it using `data_templates/sales_flag_sample.csv`.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from xgboost import XGBClassifier

try:
    from mlflow.data.pandas_dataset import PandasDataset
    from mlflow.data.artifact_dataset_source import ArtifactDatasetSource
except ImportError:
    PandasDataset = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "data" / "mlops"
CONTAINER_DATA_ROOT = Path("/app_data")
EXTERNAL_DATA_ROOT = Path(
    os.environ.get("ML_DATA_ROOT")
    or (DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else CONTAINER_DATA_ROOT)
)
RAW_DATA_PATH = EXTERNAL_DATA_ROOT / "raw" / "sales_flag.csv"
DATA_TEMPLATE_PATH = PROJECT_ROOT / "data_templates" / "sales_flag_sample.csv"
MODEL_PATH = PROJECT_ROOT / "models/sales_flag_xgb.json"
METRICS_PATH = PROJECT_ROOT / "reports/sales_flag_metrics.json"
TUNING_REPORT_PATH = PROJECT_ROOT / "reports/sales_flag_tuning.json"
DEFAULT_EXPERIMENT = "SalesFlagClassification"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
CV_SPLITS = 3
TUNING_ITERATIONS = 20

FEATURE_COLUMNS = [
    "marketing_spend",
    "store_visits",
    "online_ads",
    "discount_rate",
    "competitor_price",
]
TARGET_COLUMN = "sales_flag"


def ensure_dataset() -> Path:
    """Ensure the classification dataset exists, seeding it from the template if available."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH
    if DATA_TEMPLATE_PATH.exists():
        shutil.copyfile(DATA_TEMPLATE_PATH, RAW_DATA_PATH)
        return RAW_DATA_PATH
    raise FileNotFoundError(
        f"Dataset not found at {RAW_DATA_PATH} and template missing at {DATA_TEMPLATE_PATH}"
    )


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int32)
    return X, y, df


def build_dataset_record(dataset_df: pd.DataFrame):
    if PandasDataset is None:
        return None
    try:
        source = ArtifactDatasetSource(str(RAW_DATA_PATH))
    except Exception:
        source = str(RAW_DATA_PATH)

    base_kwargs = {"source": source, "name": "sales_flag_dataset"}
    dataset_record = None
    try:
        dataset_record = PandasDataset(
            dataset_df,
            targets=TARGET_COLUMN,
            feature_names=FEATURE_COLUMNS,
            **base_kwargs,
        )
    except TypeError:
        try:
            dataset_record = PandasDataset(
                dataset_df, targets=TARGET_COLUMN, **base_kwargs
            )
        except TypeError:
            dataset_record = PandasDataset(dataset_df, **base_kwargs)
    return dataset_record


def run_hyperparameter_search(X_train: np.ndarray, y_train: np.ndarray):
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
    )
    param_distributions = {
        "n_estimators": [100, 150, 200, 250, 300],
        "learning_rate": [0.05, 0.075, 0.1, 0.125, 0.15],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 4, 6],
        "gamma": [0.0, 0.1, 0.2],
    }
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=TUNING_ITERATIONS,
        scoring="f1",
        n_jobs=-1,
        cv=cv,
        verbose=0,
        refit=True,
        random_state=42,
    )
    search.fit(X_train, y_train)
    best_score = float(search.best_score_)
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Best CV F1: {best_score:.3f}")

    results_df = pd.DataFrame(search.cv_results_)
    tuning_payload = {
        "best_params": search.best_params_,
        "best_cv_f1": best_score,
        "cv_splits": CV_SPLITS,
        "tuning_iterations": TUNING_ITERATIONS,
        "results": results_df.to_dict(orient="records"),
    }
    TUNING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNING_REPORT_PATH.write_text(json.dumps(tuning_payload, indent=2), encoding="utf-8")

    return search.best_estimator_, search.best_params_, best_score


def main() -> None:
    dataset_path = ensure_dataset()
    X, y, dataset_df = load_dataset(dataset_path)
    print(f"Training data loaded from {dataset_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, best_params, cv_best_f1 = run_hyperparameter_search(X_train, y_train)

    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "cv_best_f1": float(cv_best_f1),
    }

    dataset_record = build_dataset_record(dataset_df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))
    with mlflow.start_run():
        logged_params = {
            "model": "XGBClassifier",
            "test_size": 0.2,
            "cv_splits": CV_SPLITS,
            "tuning_iterations": TUNING_ITERATIONS,
        }
        logged_params.update({k: best_params.get(k) for k in best_params})
        mlflow.log_params(logged_params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="sales_flag_xgb")
        mlflow.log_artifact(METRICS_PATH)
        mlflow.log_artifact(TUNING_REPORT_PATH)
        if dataset_record is not None:
            mlflow.log_input(dataset_record, context="training")
        print(
            f"Logged run {mlflow.last_active_run().info.run_id} to {mlflow.get_tracking_uri()}"
        )


if __name__ == "__main__":
    main()
