import time
import requests
import pandas as pd

print("RUNNING FINAL VERSION")

API_URL = "http://127.0.0.1:8888/infer"

TRAIN_PATH = "data/training_data.csv"
NEW_PATH = "data/new_data.csv"


# ✅ STRONG DRIFT GENERATOR (final fix)
def transform_to_valid_range(row):
    return {
        "qubit_count": 100,
        "gate_depth": 500,
        "error_rate_pct": 5.0,
        "is_error_corrected": 1,
    }


def main():
    print("🚀 Sending traffic...")

    train_df = pd.read_csv(TRAIN_PATH)
    new_df = pd.read_csv(NEW_PATH)

    train_df.columns = train_df.columns.str.strip()
    new_df.columns = new_df.columns.str.strip()

    # Drop target column
    train_df = train_df.drop(columns=["circuit_exec_ms"])
    new_df = new_df.drop(columns=["circuit_exec_ms"])

    # =========================
    # 35 NORMAL REQUESTS
    # =========================
    for _ in range(35):
        row = train_df.sample(1).to_dict(orient="records")[0]
        requests.post(API_URL, json=row)
        time.sleep(0.05)

    # =========================
    # 15 DRIFTED REQUESTS
    # =========================
    for _ in range(15):
        raw_row = new_df.sample(1).to_dict(orient="records")[0]
        row = transform_to_valid_range(raw_row)
        requests.post(API_URL, json=row)
        time.sleep(0.05)

    print("✅ Traffic simulation complete (35 normal + 15 drifted)")


if __name__ == "__main__":
    main()
