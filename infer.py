"""
infer.py — Production-ready FastAPI inference server.

Endpoints:
  GET  /health       → liveness check
  GET  /info         → model metadata
  POST /predict      → single timestep prediction
  GET  /demo/next    → replay next row from demo dataset
  POST /demo/reset   → restart demo replay
"""
from contextlib import asynccontextmanager
from collections import deque
from typing import List
import os

import numpy as np
import torch
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import LSTMAutoencoder, CNNAutoencoder, recon_error_per_sample


# ── Global state ───────────────────────────────────────────────────────────────
_model     = None
_scaler    = None
_feat_cols = None
_threshold = None
_window    = None
_device    = None
_buffer    = None
_arch      = None

_demo_X      = None
_demo_labels = None
_demo_idx    = 0


def _load_all():
    global _model, _scaler, _feat_cols, _threshold, _window
    global _device, _buffer, _arch, _demo_X, _demo_labels, _demo_idx

    _device = torch.device("cpu")

    ckpt     = torch.load("best_model.pt", map_location=_device, weights_only=False)
    _arch    = ckpt["arch"]
    n_feat   = ckpt["n_features"]
    latent   = ckpt["latent"]
    _window  = ckpt["window"]
    n_layers = ckpt.get("layers", 2)

    if _arch == "lstm":
        _model = LSTMAutoencoder(n_feat, latent, _window, n_layers)
    else:
        _model = CNNAutoencoder(n_feat, latent, _window)

    _model.load_state_dict(ckpt["model_state"])
    _model.eval()
    _model.to(_device)

    _scaler, _feat_cols = joblib.load("scaler.pkl")
    _threshold = joblib.load("threshold.pkl")["threshold"]
    _buffer    = deque(maxlen=_window)

    if os.path.exists("demo.npz"):
        demo         = np.load("demo.npz")
        _demo_X      = demo["X"]
        _demo_labels = demo["labels"]
        _demo_idx    = 0
        print(f"Demo data loaded: {len(_demo_X)} rows")

    print(f"Model ready | arch={_arch} | features={n_feat} | "
          f"window={_window} | threshold={_threshold:.6f}")


# ── Lifespan (modern FastAPI pattern, no deprecation warning) ──────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all()
    yield


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SWaT FDI Detector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class ReadingRequest(BaseModel):
    readings: List[float]


class PredictionResponse(BaseModel):
    anomaly_score:   float
    threshold:       float
    is_anomaly:      bool
    top_sensors:     List[dict]
    buffer_fill:     int
    buffer_capacity: int
    true_label:      int = -1


# ── Core inference ─────────────────────────────────────────────────────────────
def _run_inference(scaled_row: np.ndarray, true_label: int = -1) -> PredictionResponse:
    _buffer.append(scaled_row)

    if len(_buffer) < _window:
        return PredictionResponse(
            anomaly_score=0.0, threshold=float(_threshold),
            is_anomaly=False, top_sensors=[],
            buffer_fill=len(_buffer), buffer_capacity=_window,
            true_label=true_label,
        )

    window_arr = np.stack(list(_buffer), axis=0)
    x = torch.from_numpy(window_arr.astype(np.float32)).unsqueeze(0).to(_device)

    with torch.no_grad():
        x_hat      = _model(x)
        per_sensor = recon_error_per_sample(x, x_hat)[0].cpu().numpy()
        score      = float(per_sensor.mean())

    top_idx  = np.argsort(per_sensor)[::-1][:5]
    top_sens = [
        {"sensor": _feat_cols[i], "error": round(float(per_sensor[i]), 6)}
        for i in top_idx
    ]

    return PredictionResponse(
        anomaly_score   = round(score, 6),
        threshold       = round(float(_threshold), 6),
        is_anomaly      = score >= _threshold,
        top_sensors     = top_sens,
        buffer_fill     = len(_buffer),
        buffer_capacity = _window,
        true_label      = true_label,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "arch":       _arch,
        "n_features": len(_feat_cols) if _feat_cols else 0,
        "window":     _window,
        "threshold":  _threshold,
        "feat_cols":  _feat_cols,
        "demo_rows":  len(_demo_X) if _demo_X is not None else 0,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: ReadingRequest):
    if len(req.readings) != len(_feat_cols):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(_feat_cols)} features, got {len(req.readings)}"
        )
    raw    = np.array(req.readings, dtype=np.float32).reshape(1, -1)
    scaled = _scaler.transform(raw)[0]
    return _run_inference(scaled)


@app.get("/demo/next", response_model=PredictionResponse)
def demo_next():
    global _demo_idx
    if _demo_X is None:
        raise HTTPException(status_code=404,
                            detail="No demo data. Run export_demo.py first.")
    if _demo_idx >= len(_demo_X):
        _demo_idx = 0
        _buffer.clear()

    row       = _demo_X[_demo_idx]
    label     = int(_demo_labels[_demo_idx])
    _demo_idx += 1
    return _run_inference(row, true_label=label)


@app.post("/demo/reset")
def demo_reset():
    global _demo_idx
    _demo_idx = 0
    _buffer.clear()
    return {"status": "reset", "demo_rows": len(_demo_X) if _demo_X is not None else 0}