"""Train the sales-flag classifier using Ray Tune plus MLflow logging.

This script mirrors ``train_sales_flag_model.py`` but distributes the
hyperparameter search with Ray Tune. Each Ray trial trains an XGBoost
classifier on a shared training split and reports validation metrics
back to Tune. After finding the best configuration we re-train on the
entire training set, evaluate on the held-out test split, persist the
model/metrics locally, and log the complete run to MLflow.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    from mlflow.data.pandas_dataset import PandasDataset
    from mlflow.data.artifact_dataset_source import ArtifactDatasetSource
except ImportError:  # pragma: no cover - optional dataset logging
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
MODEL_PATH = PROJECT_ROOT / "models/sales_flag_xgb_ray.json"
METRICS_PATH = PROJECT_ROOT / "reports/sales_flag_ray_metrics.json"
TUNING_REPORT_PATH = PROJECT_ROOT / "reports/sales_flag_ray_tuning.json"
DEFAULT_EXPERIMENT = "SalesFlagClassificationRay"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
TUNE_NUM_SAMPLES = 25
TUNE_VAL_SIZE = 0.2

FEATURE_COLUMNS = [
    "marketing_spend",
    "store_visits",
    "online_ads",
    "discount_rate",
    "competitor_price",
]
TARGET_COLUMN = "sales_flag"


def ensure_dataset() -> Path:
    """Ensure the dataset exists, optionally seeding it from the template."""
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


def build_dataset_record(dataset_df: pd.DataFrame):  # pragma: no cover - mlflow optional
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
            dataset_record = PandasDataset(dataset_df, targets=TARGET_COLUMN, **base_kwargs)
        except TypeError:
            dataset_record = PandasDataset(dataset_df, **base_kwargs)
    return dataset_record


def init_ray_runtime() -> None:
    """Connect to an existing Ray cluster or start a local one."""
    ray_address = os.environ.get("RAY_ADDRESS")
    init_kwargs = {"ignore_reinit_error": True}
    if ray_address:
        init_kwargs["address"] = ray_address
        print(f"Connecting to Ray cluster at {ray_address}")
    else:
        print("Starting local Ray runtime")
    ray.init(**init_kwargs)


def sanitize_config(config: Dict[str, float]) -> Dict[str, float | int]:
    sanitized = {}
    for key, value in config.items():
        if isinstance(value, (np.floating, np.float32, np.float64)):
            sanitized[key] = float(value)
        elif isinstance(value, (np.integer, np.int32, np.int64)):
            sanitized[key] = int(value)
        else:
            sanitized[key] = value
    return sanitized


def run_ray_tuning(
    X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[Dict[str, float | int], float, List[Dict[str, object]]]:
    """Launch Ray Tune to explore the XGBoost hyperparameter space."""

    X_inner_train, X_val, y_inner_train, y_val = train_test_split(
        X_train, y_train, test_size=TUNE_VAL_SIZE, stratify=y_train, random_state=21
    )
    data_bundle = (X_inner_train, y_inner_train, X_val, y_val)

    def trainable(config, data_bundle):
        X_tr, y_tr, X_v, y_v = data_bundle
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            use_label_encoder=False,
            **config,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_v)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_v, y_pred, average="binary", zero_division=0
        )
        session.report(
            {
                "f1": float(f1),
                "accuracy": float(accuracy_score(y_v, y_pred)),
                "precision": float(precision),
                "recall": float(recall),
            }
        )

    scheduler = ASHAScheduler(grace_period=1, reduction_factor=2)
    param_space = {
        "n_estimators": tune.randint(120, 320),
        "learning_rate": tune.loguniform(0.03, 0.2),
        "max_depth": tune.randint(3, 8),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
        "min_child_weight": tune.randint(1, 8),
        "gamma": tune.uniform(0.0, 0.3),
    }

    trainable_with_data = tune.with_parameters(trainable, data_bundle=data_bundle)
    tuner = tune.Tuner(
        trainable_with_data,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="f1",
            mode="max",
            num_samples=TUNE_NUM_SAMPLES,
            scheduler=scheduler,
        ),
    )

    result_grid = tuner.fit()
    best_result = result_grid.get_best_result(metric="f1", mode="max")
    best_config = sanitize_config(best_result.config)
    best_f1 = float(best_result.metrics.get("f1", 0.0))

    trial_summaries: List[Dict[str, object]] = []
    for result in result_grid:
        metrics_subset = {
            metric: (float(result.metrics[metric]) if metric in result.metrics else None)
            for metric in ("f1", "accuracy", "precision", "recall")
        }
        trial_summaries.append(
            {
                "config": sanitize_config(result.config),
                "metrics": metrics_subset,
                "log_dir": result.path,
            }
        )

    return best_config, best_f1, trial_summaries


def train_best_model(
    best_config: Dict[str, float | int], X_train: np.ndarray, y_train: np.ndarray
) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
        **best_config,
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    dataset_path = ensure_dataset()
    X, y, dataset_df = load_dataset(dataset_path)
    print(f"Training data loaded from {dataset_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    init_ray_runtime()
    best_params, best_cv_f1, tuning_trials = run_ray_tuning(X_train, y_train)
    ray.shutdown()

    model = train_best_model(best_params, X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tune_best_f1": float(best_cv_f1),
    }

    dataset_record = build_dataset_record(dataset_df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    tuning_payload = {
        "best_params": best_params,
        "best_cv_f1": best_cv_f1,
        "num_samples": TUNE_NUM_SAMPLES,
        "validation_size": TUNE_VAL_SIZE,
        "trials": tuning_trials,
    }
    TUNING_REPORT_PATH.write_text(json.dumps(tuning_payload, indent=2), encoding="utf-8")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))
    with mlflow.start_run():
        log_params = {
            "trainer": "ray_tune_xgboost",
            "test_size": 0.2,
            "val_size": TUNE_VAL_SIZE,
            "tune_num_samples": TUNE_NUM_SAMPLES,
            "ray_address": os.environ.get("RAY_ADDRESS", "local"),
        }
        log_params.update(best_params)
        mlflow.log_params(log_params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="sales_flag_xgb_ray")
        mlflow.log_artifact(METRICS_PATH)
        mlflow.log_artifact(TUNING_REPORT_PATH)
        if dataset_record is not None:
            mlflow.log_input(dataset_record, context="training")
        print(
            f"Logged run {mlflow.last_active_run().info.run_id} to {mlflow.get_tracking_uri()}"
        )


if __name__ == "__main__":
    main()
