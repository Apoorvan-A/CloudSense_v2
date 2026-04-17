
import os, json, pickle, logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cloudsense")

# Paths
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/model_export"))

#  Model Definition  (must match exactly what was trained in Colab)
class IMFSubModel(nn.Module):
    """Single-IMF branch: Conv1d → BiLSTM → Linear"""
    def __init__(self, input_size=1, hidden_size=64, conv_filters=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(
            conv_filters, hidden_size,
            num_layers=1, batch_first=True,
            bidirectional=True, dropout=0.0
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x: (B, T, 1)
        x = x.permute(0, 2, 1)       # (B, 1, T)
        x = self.conv(x)              # (B, C, T)
        x = x.permute(0, 2, 1)       # (B, T, C)
        out, _ = self.bilstm(x)       # (B, T, H*2)
        return self.fc(out[:, -1, :]) # (B, 1)


class CEEMDANBiLSTM(nn.Module):
    """Ensemble of IMFSubModels — one per CEEMDAN IMF"""
    def __init__(self, n_imfs=6, hidden_size=64, conv_filters=32):
        super().__init__()
        self.sub_models = nn.ModuleList([
            IMFSubModel(1, hidden_size, conv_filters) for _ in range(n_imfs)
        ])

    def forward(self, imf_sequences):
        # imf_sequences: list of (B, T, 1) tensors, one per IMF
        preds = [self.sub_models[i](imf_sequences[i]) for i in range(len(self.sub_models))]
        return torch.stack(preds, dim=0).sum(dim=0)  # (B, 1)


#  Load Model at Startup
def load_artifacts():
    log.info(f"Loading model artifacts from {MODEL_DIR}")

    with open(MODEL_DIR / "model_config.json") as f:
        cfg = json.load(f)

    model = CEEMDANBiLSTM(
        n_imfs=cfg["n_imfs"],
        hidden_size=cfg["hidden_size"],
        conv_filters=cfg["conv_filters"],
    )
    state = torch.load(MODEL_DIR / "ceemdan_bilstm.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(MODEL_DIR / "imf_tail.json") as f:
        imf_tail = {int(k): np.array(v) for k, v in json.load(f).items()}

    with open(MODEL_DIR / "training_metrics.json") as f:
        train_metrics = json.load(f)

    log.info("✅ All artifacts loaded")
    return model, scaler, cfg, imf_tail, train_metrics


model, scaler, cfg, imf_tail, train_metrics = load_artifacts()
LOOK_BACK = cfg["look_back"]  # 48


#  Simple CEEMDAN (NumPy) for live signals
def fast_ceemdan(signal: np.ndarray, n_imfs: int = 6) -> List[np.ndarray]:
    """Lightweight CEEMDAN decomposition for inference."""
    n = len(signal)
    imfs, residue = [], signal.copy()
    for i in range(n_imfs - 1):
        window = max(3, n // (2 ** (i + 1)))
        kernel = np.ones(window) / window
        # full conv then trim
        smoothed = np.convolve(residue, kernel, mode='full')[:n]
        imf = residue - smoothed
        imfs.append(imf)
        residue = smoothed
    imfs.append(residue)
    return imfs


def predict_next(raw_sequence: np.ndarray, n_steps: int = 1) -> List[float]:
    """
    Given a raw CPU-utilization sequence (any length ≥ LOOK_BACK),
    predict the next n_steps values.
    """
    # Scale
    seq_2d = raw_sequence.reshape(-1, 1)
    scaled = scaler.transform(seq_2d).flatten()

    # Decompose
    live_imfs = fast_ceemdan(scaled, n_imfs=cfg["n_imfs"])

    predictions = []
    current_imfs = [imf.copy() for imf in live_imfs]

    for _ in range(n_steps):
        imf_tensors = []
        for imf in current_imfs:
            window = imf[-LOOK_BACK:]
            t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            imf_tensors.append(t)

        with torch.no_grad():
            pred_scaled = model(imf_tensors).item()

        pred_raw = scaler.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(float(pred_raw))

        # Slide windows forward
        for j, imf in enumerate(current_imfs):
            current_imfs[j] = np.append(imf, pred_scaled / cfg["n_imfs"])

    return predictions


#  FastAPI App
app = FastAPI(
    title="CloudSense API",
    description="CEEMDAN+CNN-BiLSTM Cloud Workload Predictor — VIT Vellore BCse408l",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request / Response schemas

class PredictRequest(BaseModel):
    """Send the last `look_back` CPU utilization values (%)."""
    cpu_sequence: List[float]           # e.g. [45.2, 46.1, ..., 48.0]  (48 values)
    n_steps: Optional[int] = 1          # how many future steps to predict

class PredictResponse(BaseModel):
    predictions: List[float]
    unit: str = "% CPU utilization"
    model: str = "CEEMDAN+CNN-BiLSTM"
    look_back_used: int


# Endpoints

@app.get("/health")
def health():
    return {"status": "ok", "model": "CEEMDANBiLSTM", "look_back": LOOK_BACK}


@app.get("/metrics")
def get_metrics():
    """Return the training-time evaluation metrics."""
    return {
        "dataset": "NAB AWS CloudWatch EC2",
        "model": "CEEMDAN+CNN-BiLSTM (Proposed)",
        "MAE":  train_metrics["mae"],
        "RMSE": train_metrics["rmse"],
        "R2":   train_metrics["r2"],
        "unit": "% CPU utilization",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict next N CPU utilization steps.

    Body example:
    {
        "cpu_sequence": [45.2, 46.1, 44.8, ..., 48.0],   // exactly 48 values
        "n_steps": 6
    }
    """
    if len(req.cpu_sequence) < LOOK_BACK:
        raise HTTPException(
            status_code=422,
            detail=f"cpu_sequence must have at least {LOOK_BACK} values, got {len(req.cpu_sequence)}"
        )

    arr = np.array(req.cpu_sequence, dtype=np.float32)
    preds = predict_next(arr, n_steps=req.n_steps)

    return PredictResponse(
        predictions=preds,
        look_back_used=LOOK_BACK,
    )


@app.post("/predict/realtime")
def predict_realtime(req: PredictRequest):
    """
    Same as /predict but returns a richer payload suitable for
    feeding directly into a CloudWatch dashboard or Kubernetes HPA.
    """
    if len(req.cpu_sequence) < LOOK_BACK:
        raise HTTPException(status_code=422,
            detail=f"Need ≥ {LOOK_BACK} data points.")

    arr = np.array(req.cpu_sequence, dtype=np.float32)
    preds = predict_next(arr, n_steps=req.n_steps)

    # HPA scale-out recommendation
    avg_pred = sum(preds) / len(preds)
    scale_out = avg_pred > 70.0   # threshold: 70 % CPU

    return {
        "predictions": preds,
        "avg_predicted_cpu": round(avg_pred, 2),
        "scale_recommendation": "SCALE_OUT" if scale_out else "HOLD",
        "threshold_pct": 70.0,
        "model": "CEEMDAN+CNN-BiLSTM",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
