import sys
import logging
import asyncio
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import os

app = FastAPI()

EC2_IP = "15.207.242.105"
AWS_BASE_URL = f"http://{EC2_IP}:8000"

# Mock Fleet State
mock_fleet = {
    "i-main-001": {
        "id": "i-main-001",
        "name": "CloudSense-Master",
        "status": "Running",
        "type": "t3.micro",
        "is_main": True
    }
}

async def simulate_boot(instance_id: str):
    await asyncio.sleep(4)
    if instance_id in mock_fleet:
        mock_fleet[instance_id]["status"] = "Running"

@app.get("/api/fleet")
def get_fleet():
    return mock_fleet

@app.post("/api/fleet/provision")
async def provision_instance():
    new_id = "i-" + str(uuid.uuid4())[:8]
    mock_fleet[new_id] = {
        "id": new_id,
        "name": "CloudSense-Worker-" + str(uuid.uuid4())[:3],
        "status": "Pending",
        "type": "t3.micro (Free Tier)",
        "is_main": False
    }
    asyncio.create_task(simulate_boot(new_id))
    return {"status": "provisioning", "id": new_id}

@app.post("/api/fleet/{instance_id}/stop")
def stop_instance(instance_id: str):
    if instance_id in mock_fleet:
        mock_fleet[instance_id]["status"] = "Stopped"
    return {"status": "ok"}

@app.post("/api/fleet/{instance_id}/start")
async def start_instance(instance_id: str):
    if instance_id in mock_fleet:
        mock_fleet[instance_id]["status"] = "Pending"
        asyncio.create_task(simulate_boot(instance_id))
    return {"status": "ok"}

@app.post("/api/fleet/{instance_id}/terminate")
def terminate_instance(instance_id: str):
    if instance_id in mock_fleet and not mock_fleet[instance_id].get("is_main"):
        del mock_fleet[instance_id]
    return {"status": "ok"}

# Predictive UI Routes
class RealtimeRequest(BaseModel):
    cpu_values: list[float]
    threshold: float = 70.0

@app.get("/api/metrics")
def get_metrics():
    try:
        r = requests.get(f"{AWS_BASE_URL}/metrics", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def get_health():
    try:
        r = requests.get(f"{AWS_BASE_URL}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/realtime")
def predict_realtime(req: RealtimeRequest):
    try:
        payload = {"cpu_values": req.cpu_values, "threshold": req.threshold}
        r = requests.post(f"{AWS_BASE_URL}/predict/realtime", json=payload, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as file:
        return file.read()
