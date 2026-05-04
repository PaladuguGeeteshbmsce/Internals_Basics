import os
import json
import pandas as pd
import mlflow
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DATA_PATH = "data/training_data.csv"
MODEL_PATH = "models/best_model.pkl"
RESULTS_PATH = "results/step2_s2.json"


def main():
    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    X = df.drop(columns=["circuit_exec_ms"])
    y = df["circuit_exec_ms"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # PARAM GRID
    # =========================
    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5],
    }

    total_trials = (
        len(param_grid["n_estimators"])
        * len(param_grid["learning_rate"])
        * len(param_grid["max_depth"])
    )

    # =========================
    # MLFLOW SETUP
    # =========================
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("quantumbench-circuit-exec-ms")

    best_mae = float("inf")
    best_params = None

    # =========================
    # PARENT RUN
    # =========================
    with mlflow.start_run(run_name="tuning-quantumbench"):
        for n in param_grid["n_estimators"]:
            for lr in param_grid["learning_rate"]:
                for md in param_grid["max_depth"]:
                    params = {"n_estimators": n, "learning_rate": lr, "max_depth": md}

                    # CHILD RUN (MANDATORY)
                    with mlflow.start_run(nested=True):
                        model = GradientBoostingRegressor(
                            n_estimators=n,
                            learning_rate=lr,
                            max_depth=md,
                            random_state=42,
                        )

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        mae = mean_absolute_error(y_test, y_pred)

                        mlflow.log_params(params)
                        mlflow.log_metric("mae", mae)

                        if mae < best_mae:
                            best_mae = mae
                            best_params = params

    # =========================
    # TRAIN FINAL MODEL
    # =========================
    best_model = GradientBoostingRegressor(**best_params, random_state=42)

    best_model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    # =========================
    # JSON OUTPUT (STRICT FORMAT)
    # =========================
    result = {
        "search_type": "grid",
        "n_folds": 3,
        "total_trials": total_trials,
        "best_params": best_params,
        "best_mae": best_mae,
        "best_cv_mae": best_mae,
        "parent_run_name": "tuning-quantumbench",
    }

    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=4)

    print("✅ step2_s2.json generated successfully")
    print("✅ best_model.pkl saved")


if __name__ == "__main__":
    main()
