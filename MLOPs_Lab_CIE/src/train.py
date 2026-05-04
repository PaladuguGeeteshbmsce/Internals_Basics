import os
import json
import pandas as pd
import numpy as np
import mlflow

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

DATA_PATH = "data/training_data.csv"
RESULTS_PATH = "results/step1_s1.json"


def calculate_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def main():
    # =========================
    # LOAD DATA (STRICT)
    # =========================
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    # Fix hidden whitespace issues from PDF copy
    df.columns = df.columns.str.strip()

    # Validate required column exists
    if "circuit_exec_ms" not in df.columns:
        raise ValueError(
            f"Expected column 'circuit_exec_ms' not found. Found: {list(df.columns)}"
        )

    X = df.drop(columns=["circuit_exec_ms"])
    y = df["circuit_exec_ms"]

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # MLFLOW SETUP
    # =========================
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("quantumbench-circuit-exec-ms")

    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    model_results = []
    rf_mae = gb_mae = None

    # =========================
    # TRAIN MODELS
    # =========================
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            metrics = calculate_metrics(y_test, y_pred)

            # Log ALL hyperparameters
            for param, value in model.get_params().items():
                mlflow.log_param(param, value)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Required tag
            mlflow.set_tag("project_phase", "model_selection")

            # Store results
            model_results.append(
                {
                    "name": name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "mape": metrics["mape"],
                }
            )

            if name == "RandomForest":
                rf_mae = metrics["mae"]
            else:
                gb_mae = metrics["mae"]

    # =========================
    # SELECT BEST MODEL
    # =========================
    if rf_mae < gb_mae:
        best_model = "RandomForest"
        best_mae = rf_mae
    else:
        best_model = "GradientBoosting"
        best_mae = gb_mae

    # =========================
    # FINAL JSON OUTPUT
    # =========================
    final_output = {
        "experiment_name": "quantumbench-circuit-exec-ms",
        "models": model_results,
        "best_model": best_model,
        "best_metric_name": "mae",
        "best_metric_value": best_mae,
    }

    # Save JSON
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(final_output, f, indent=4)

    print("✅ step1_s1.json generated successfully")


if __name__ == "__main__":
    main()
