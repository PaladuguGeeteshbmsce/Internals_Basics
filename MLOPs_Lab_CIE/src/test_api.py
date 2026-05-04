import requests
import json
import os

BASE_URL = "http://127.0.0.1:8888"

# ✅ Required test input (from question paper)
test_input = {
    "qubit_count": 36,
    "gate_depth": 326,
    "error_rate_pct": 1.8,
    "is_error_corrected": 1,
}


def main():
    # =========================
    # CALL HEARTBEAT
    # =========================
    health_response = requests.get(f"{BASE_URL}/heartbeat")
    health_json = health_response.json()

    # =========================
    # CALL INFERENCE
    # =========================
    pred_response = requests.post(f"{BASE_URL}/infer", json=test_input)
    pred_json = pred_response.json()

    prediction = pred_json["prediction"]

    # =========================
    # BUILD REQUIRED JSON
    # =========================
    result = {
        "health_endpoint": "/heartbeat",
        "predict_endpoint": "/infer",
        "port": 8888,
        "health_response": health_json,
        "test_input": test_input,
        "prediction": prediction,
    }

    # =========================
    # SAVE OUTPUT
    # =========================
    os.makedirs("results", exist_ok=True)

    with open("results/step3_s4.json", "w") as f:
        json.dump(result, f, indent=4)

    print("✅ step3_s4.json generated successfully")


if __name__ == "__main__":
    main()
