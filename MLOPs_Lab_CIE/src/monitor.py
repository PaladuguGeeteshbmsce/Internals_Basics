import os
import json
import pandas as pd

DATA_PATH = "data/training_data.csv"
LOGS_PATH = "logs/predictions.jsonl"
RESULTS_PATH = "results/step4_s5.json"


def main():
    if not os.path.exists(LOGS_PATH):
        print("No logs found.")
        return

    # Load training data
    df_train = pd.read_csv(DATA_PATH)
    df_train.columns = df_train.columns.str.strip()

    train_qubit_mean = df_train["qubit_count"].mean()
    train_gate_mean = df_train["gate_depth"].mean()

    # Load logs
    logs = []
    preds = []

    with open(LOGS_PATH, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            logs.append(entry["input"])
            preds.append(entry["prediction"])

    df_logs = pd.DataFrame(logs)

    # Take last 15 (drifted requests)
    df_drift = df_logs.tail(15)

    live_qubit_mean = df_drift["qubit_count"].mean()
    live_gate_mean = df_drift["gate_depth"].mean()

    # Calculate shifts
    qubit_shift = abs(live_qubit_mean - train_qubit_mean)
    gate_shift = abs(live_gate_mean - train_gate_mean)

    alerts = []

    if qubit_shift > 15.28:
        alerts.append(
            {
                "feature": "qubit_count",
                "train_mean": round(train_qubit_mean, 2),
                "live_mean": round(live_qubit_mean, 2),
                "shift": round(qubit_shift, 2),
                "threshold": 15.28,
                "status": "ALERT",
            }
        )

    if gate_shift > 146.91:
        alerts.append(
            {
                "feature": "gate_depth",
                "train_mean": round(train_gate_mean, 2),
                "live_mean": round(live_gate_mean, 2),
                "shift": round(gate_shift, 2),
                "threshold": 146.91,
                "status": "ALERT",
            }
        )

    result = {
        "total_predictions": len(preds),
        "mean_prediction": sum(preds) / len(preds),
        "drift_detected": len(alerts) > 0,
        "alerts": alerts,
    }

    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=4)

    print("✅ step4_s5.json generated")


if __name__ == "__main__":
    main()
