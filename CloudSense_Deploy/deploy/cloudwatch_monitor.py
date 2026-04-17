"""
=======================================================
 CloudSense — AWS CloudWatch Auto-Scaling Integration
=======================================================
 This script:
   1. Reads CPU metrics from CloudWatch (your real EC2 instances)
   2. Sends them to CloudSense API for prediction
   3. Logs the prediction back to CloudWatch as a custom metric
   4. (Optional) Triggers Auto Scaling if predicted CPU > 70%

 Run this on a cron job or Lambda every 5 minutes.
 Cost: CloudWatch custom metrics ~$0.30/metric/month (minimal)
=======================================================
"""

import boto3
import requests
import json
import logging
from datetime import datetime, timezone, timedelta

# Config
CLOUDSENSE_API = "http://YOUR_EC2_PUBLIC_IP:8000"   # ← replace after deploy
AWS_REGION     = "ap-south-1"
TARGET_INSTANCE_ID = "i-YOUR_INSTANCE_ID"           # ← the EC2 you're monitoring
LOOK_BACK      = 48       # must match model training (48 × 5min = 4 hours history)
N_STEPS        = 6        # predict 6 × 5min = 30 min ahead
CPU_THRESHOLD  = 70.0     # % CPU to trigger scale-out

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cloudsense-monitor")

cw = boto3.client("cloudwatch", region_name=AWS_REGION)
asg = boto3.client("autoscaling",  region_name=AWS_REGION)


def fetch_cpu_history(instance_id: str, n_points: int = 50) -> list[float]:
    """Pull last n_points × 5-min CPU readings from CloudWatch."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(minutes=5 * n_points)

    resp = cw.get_metric_statistics(
        Namespace   = "AWS/EC2",
        MetricName  = "CPUUtilization",
        Dimensions  = [{"Name": "InstanceId", "Value": instance_id}],
        StartTime   = start,
        EndTime     = end,
        Period      = 300,           # 5-minute granularity
        Statistics  = ["Average"],
    )
    points = sorted(resp["Datapoints"], key=lambda x: x["Timestamp"])
    return [p["Average"] for p in points]


def call_cloudsense(cpu_sequence: list[float]) -> dict:
    """Call CloudSense /predict/realtime endpoint."""
    payload = {"cpu_sequence": cpu_sequence[-LOOK_BACK:], "n_steps": N_STEPS}
    r = requests.post(f"{CLOUDSENSE_API}/predict/realtime", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def push_prediction_to_cloudwatch(predicted_cpu: float, instance_id: str):
    """Store the prediction as a custom CloudWatch metric."""
    cw.put_metric_data(
        Namespace   = "CloudSense/Predictions",
        MetricData  = [{
            "MetricName": "PredictedCPUUtilization",
            "Dimensions": [{"Name": "InstanceId", "Value": instance_id}],
            "Timestamp":  datetime.now(timezone.utc),
            "Value":      predicted_cpu,
            "Unit":       "Percent",
        }]
    )
    log.info(f"📊 Pushed predicted CPU={predicted_cpu:.1f}% to CloudWatch")


def trigger_scale_out_if_needed(result: dict):
    """Log scale recommendation (or trigger Auto Scaling Group)."""
    recommendation = result.get("scale_recommendation", "HOLD")
    avg_pred = result.get("avg_predicted_cpu", 0)

    if recommendation == "SCALE_OUT":
        log.warning(f"⚠️  SCALE_OUT recommended — predicted avg CPU = {avg_pred:.1f}%")
        # Uncomment to actually trigger your ASG:
        # asg.set_desired_capacity(
        #     AutoScalingGroupName="cloudsense-asg",
        #     DesiredCapacity=current_capacity + 1,
        # )
    else:
        log.info(f"✅  HOLD — predicted avg CPU = {avg_pred:.1f}%")


def main():
    log.info(f"Fetching CPU history for {TARGET_INSTANCE_ID}...")
    cpu_history = fetch_cpu_history(TARGET_INSTANCE_ID, n_points=LOOK_BACK + 10)

    if len(cpu_history) < LOOK_BACK:
        log.error(f"Not enough data: got {len(cpu_history)}, need {LOOK_BACK}. "
                  "Wait for more CloudWatch data points.")
        return

    log.info(f"Last 5 CPU readings: {cpu_history[-5:]}")

    result = call_cloudsense(cpu_history)
    log.info(f"Prediction result: {result}")

    avg_predicted = result["avg_predicted_cpu"]
    push_prediction_to_cloudwatch(avg_predicted, TARGET_INSTANCE_ID)
    trigger_scale_out_if_needed(result)


if __name__ == "__main__":
    main()
