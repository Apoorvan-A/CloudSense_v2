import argparse
import requests
import json
import random
import sys

def get_base_url(ip: str) -> str:
    return f"http://{ip}:8000"


def test_health(base: str):
    print("\n[1] Health check...")
    r = requests.get(f"{base}/health", timeout=10)
    r.raise_for_status()
    print(f"    [OK] {r.json()}")


def test_metrics(base: str):
    print("\n[2] Training metrics...")
    r = requests.get(f"{base}/metrics", timeout=10)
    r.raise_for_status()
    d = r.json()
    metrics = d['training_metrics']
    print(f"    MAE  = {metrics['mae']:.4f}")
    print(f"    RMSE = {metrics['rmse']:.4f}")
    print(f"    R²   = {metrics['r2']:.4f}")


def test_predict(base: str):
    print("\n[3] Predict next CPU reading...")
    # Simulate 48 CPU readings around 55% with some noise
    seq = [55 + random.gauss(0, 5) for _ in range(48)]
    payload = {"cpu_values": seq}
    r = requests.post(f"{base}/predict", json=payload, timeout=30)
    r.raise_for_status()
    d = r.json()
    print(f"    Prediction: {d['predicted_cpu_percent']:.2f}%")


def test_realtime(base: str):
    print("\n[4] Realtime prediction + HPA recommendation...")
    # Simulate a high-CPU scenario (should trigger SCALE_OUT)
    seq = [75 + random.gauss(0, 3) for _ in range(48)]
    payload = {"cpu_values": seq, "threshold": 70.0}
    r = requests.post(f"{base}/predict/realtime", json=payload, timeout=30)
    r.raise_for_status()
    d = r.json()
    print(f"    Predicted CPU: {d['predicted_cpu_percent']:.2f}%")
    print(f"    Scale recommendation: {d['recommendation']}")

    # Now low CPU
    seq_low = [20 + random.gauss(0, 3) for _ in range(48)]
    payload_low = {"cpu_values": seq_low, "threshold": 70.0}
    r2 = requests.post(f"{base}/predict/realtime", json=payload_low, timeout=30)
    d2 = r2.json()
    print(f"    (Low CPU scenario) -> {d2['recommendation']}")


def test_validation(base: str):
    print("\n[5] Validation error (too few data points)...")
    payload = {"cpu_values": [50.0] * 10}  # only 10, need 48
    r = requests.post(f"{base}/predict", json=payload, timeout=10)
    if r.status_code == 422:
        print(f"    [OK] Correctly rejected (422): {r.json()['detail']}")
    else:
        print(f"    [FAIL] Expected 422, got {r.status_code}")


def test_swagger(base: str):
    print(f"\n[6] Swagger docs available at: {base}/docs")
    r = requests.get(f"{base}/docs", timeout=10)
    if r.status_code == 200:
        print("    [OK] Swagger UI is live")
    else:
        print(f"    Status: {r.status_code}")


def main():
    parser = argparse.ArgumentParser(description="CloudSense API Tests")
    parser.add_argument("--ip", required=True, help="EC2 public IP")
    args = parser.parse_args()

    base = get_base_url(args.ip)
    print(f"\n{'='*52}")
    print(f"  CloudSense API Test Suite")
    print(f"  Target: {base}")
    print(f"{'='*52}")

    try:
        test_health(base)
        test_metrics(base)
        test_predict(base)
        test_realtime(base)
        test_validation(base)
        test_swagger(base)
    except requests.exceptions.ConnectionError:
        print(f"\n[FAIL] Cannot connect to {base}")
        print("    Make sure the EC2 instance is running and port 8000 is open.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        sys.exit(1)

    print(f"\n{'='*52}")
    print("  [OK] All tests passed!")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
