"""
api_detective.py — GRAD-RL Framework (v7.0)
Detective Node: LSTM-AE Anomaly Detection + XGBoost Attack Classification
GRFICS Digital Twin Edition

Responsibilities:
  1. Accept a 13-float GRFICS sensor reading via POST /predict
  2. Scale with the pre-fitted MinMaxScaler
  3. Maintain a rolling 10-frame window for genuine temporal LSTM inference
  4. Flag anomaly if reconstruction MSE > P99 threshold
  5. If anomaly: classify MITRE ATT&CK technique via XGBoost
  6. Forward full payload (including scaled_features) to Defensive Node

Model artifacts expected in MODEL_DIR (_grfics_live suffix):
    scaler_grfics_live.pkl
    detective_lstm_grfics_live.keras   (or .h5 fallback)
    detective_classifier_grfics_live.pkl
    label_encoder_grfics_live.pkl
    threshold_grfics_live.json

Run locally:
    uvicorn api_detective:app --reload --port 8000
"""

import os

import collections
import json
import threading
from typing import List
import joblib
import numpy as np
import requests
import keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="GRAD-RL Detective Node",
    version="7.0 (GRFICS)",
    description="LSTM-AE anomaly detection + XGBoost attack classification for GRFICSv3",
)

# ─────────────────────────────────────────────
#  PATHS
#  Resolve relative to this file so the API works
#  regardless of the working directory uvicorn is
#  launched from.
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models"))
DEFENSE_URL = "http://127.0.0.1:8001/decide"

# ─────────────────────────────────────────────
#  GRFICS FEATURE REGISTRY
#  Exact 13 physical sensor/actuator columns from
#  grad_rl_final_dataset.csv, in CSV column order.
#  Order MUST match the MinMaxScaler fit column order.
# ─────────────────────────────────────────────
FEATURE_COLUMNS: List[str] = [
    "f1_valve",   "f1_flow",
    "f2_valve",   "f2_flow",
    "purge_valve", "purge_flow",
    "prod_valve",  "prod_flow",
    "pressure",
    "level",
    "A_purge", "B_purge", "C_purge",
]
N_FEATURES    = len(FEATURE_COLUMNS)   # 13
LSTM_TIMESTEPS = 10

# ─────────────────────────────────────────────
#  ROLLING WINDOW
#  One deque per client stream keeps genuine temporal
#  context instead of np.repeat() padding.
#  Pre-filled with zeros → valid from the first call.
# ─────────────────────────────────────────────
_WINDOW: collections.deque = collections.deque(
    [np.zeros(N_FEATURES, dtype=np.float32)] * LSTM_TIMESTEPS,
    maxlen=LSTM_TIMESTEPS,
)
_WINDOW_LOCK = threading.Lock()


# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────

def _path(kind: str) -> str:
    files = {
        "scaler":    "scaler_grfics_live.pkl",
        "lstm":      "detective_lstm_grfics_live.keras",
        "lstm_h5":   "detective_lstm_grfics_live.h5",
        "clf":       "detective_classifier_grfics_live.pkl",
        "le":        "label_encoder_grfics_live.pkl",
        "threshold": "threshold_grfics_live.json",
    }
    return os.path.join(MODEL_DIR, files[kind])


def _load_models() -> dict:
    bundle = {}

    # Scaler — required
    sp = _path("scaler")
    if not os.path.exists(sp):
        raise RuntimeError(f"Scaler not found: {sp}")
    bundle["scaler"] = joblib.load(sp)
    print(f"  ✅ Scaler:    {sp}")

    # LSTM-AE — required (.keras preferred, .h5 fallback)
    lp = _path("lstm") if os.path.exists(_path("lstm")) else _path("lstm_h5")
    if not os.path.exists(lp):
        raise RuntimeError(f"LSTM not found. Checked:\n  {_path('lstm')}\n  {_path('lstm_h5')}")
    bundle["lstm"] = keras.models.load_model(lp)
    print(f"  ✅ LSTM-AE:   {lp}")

    # XGBoost classifier — optional
    cp = _path("clf")
    bundle["clf"] = joblib.load(cp) if os.path.exists(cp) else None
    if bundle["clf"] is None:
        print(f"  ⚠️  Classifier not found — attack_type will be 'Unknown'")
    else:
        print(f"  ✅ Classifier: {cp}")

    # Label encoder — optional
    ep = _path("le")
    bundle["le"] = joblib.load(ep) if os.path.exists(ep) else None

    # Threshold — optional, default 0.05
    tp = _path("threshold")
    # if os.path.exists(tp):
    #     with open(tp) as f:
    #         bundle["threshold"] = json.load(f)["threshold"]
    #     print(f"  ✅ Threshold: {bundle['threshold']:.6f}")
    # else:
    #     bundle["threshold"] = 0.05
    #     print(f"  ⚠️  Threshold file missing — using default 0.05")

    bundle["threshold"] = 0.5
    print(f"  ✅ FORCED Threshold: {bundle['threshold']:.6f}")
    return bundle


print("\n📦 Loading GRFICS model bundle...")
try:
    _MODELS = _load_models()
    print("✅ Detective Node ready.\n")
except RuntimeError as e:
    print(f"❌ Model loading failed: {e}")
    _MODELS = None


# ─────────────────────────────────────────────
#  PYDANTIC MODELS
# ─────────────────────────────────────────────

class SensorReading(BaseModel):
    """
    One timestep of GRFICS sensor data.

    values: exactly 13 floats in FEATURE_COLUMNS order.
      [f1_valve, f1_flow, f2_valve, f2_flow,
       purge_valve, purge_flow, prod_valve, prod_flow,
       pressure, level, A_purge, B_purge, C_purge]
    """
    values: List[float]


class DetectionResult(BaseModel):
    is_anomaly:      bool
    mse_loss:        float
    attack_type:     str          # MITRE ID e.g. "T0836", or "Normal" / "Unknown"
    culprit_feature: str          # feature with highest reconstruction error
    scaled_features: List[float]  # 13 scaled values forwarded to Defensive Node


# ─────────────────────────────────────────────
#  INFERENCE ENDPOINT
# ─────────────────────────────────────────────

@app.post("/predict", response_model=DetectionResult)
def predict(reading: SensorReading):
    if _MODELS is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    scaler    = _MODELS["scaler"]
    lstm      = _MODELS["lstm"]
    clf       = _MODELS["clf"]
    le        = _MODELS["le"]
    threshold = _MODELS["threshold"]

    # ── 1. Validate & pad input ───────────────────────────────────────────
    features = np.array(reading.values, dtype=np.float32).reshape(1, -1)
    received = features.shape[1]
    if received != N_FEATURES:
        print(f"⚠️  Expected {N_FEATURES} features, got {received}. Adjusting.")
        if received < N_FEATURES:
            features = np.pad(features, ((0, 0), (0, N_FEATURES - received)))
        else:
            features = features[:, :N_FEATURES]

    # ── 2. Scale ──────────────────────────────────────────────────────────
    scaled = scaler.transform(features)   # shape: (1, 13)

    # ── 3. Rolling window → LSTM sequence ────────────────────────────────
    with _WINDOW_LOCK:
        _WINDOW.append(scaled[0])
        seq = np.array(list(_WINDOW), dtype=np.float32)   # (10, 13)
    seq_batch = seq.reshape(1, LSTM_TIMESTEPS, N_FEATURES)   # (1, 10, 13)

    # ── 4. LSTM-AE reconstruction ─────────────────────────────────────────
    reconstruction = lstm.predict(seq_batch, verbose=0)
    input_last     = seq_batch[0, -1, :]     # last timestep input
    rec_last       = reconstruction[0, -1, :]
    feat_errors    = np.power(input_last - rec_last, 2)
    mse_loss       = float(np.mean(feat_errors))
    #is_anomaly     = mse_loss > threshold
    is_anomaly     = bool(mse_loss > 0.5)

    # ── 5. Culprit feature ────────────────────────────────────────────────
    culprit_idx  = int(np.argmax(feat_errors))
    culprit_feat = FEATURE_COLUMNS[culprit_idx] if culprit_idx < N_FEATURES else "unknown"

    # ── 6. Attack classification (XGBoost) ────────────────────────────────
    attack_type = "Normal"
    if is_anomaly:
        if clf is not None:
            pred_code = clf.predict(scaled)[0]
            if le is not None:
                try:
                    attack_type = le.inverse_transform([pred_code])[0]
                except Exception:
                    attack_type = str(pred_code)
            else:
                attack_type = str(pred_code)
        else:
            attack_type = "Unknown"

    # ── 7. Forward to Defensive Node ─────────────────────────────────────
    payload = {
        "is_anomaly":      bool(is_anomaly),
        "mse_loss":        float(mse_loss),
        "attack_type":     attack_type,
        "culprit_feature": culprit_feat,
        "scaled_features": scaled[0].tolist(), # 13 floats for PPO
        "feature_errors":  feat_errors.tolist(),   
    }
    try:
        requests.post(DEFENSE_URL, json=payload, timeout=1)
    except Exception as e:
        print(f"[WARN] Defensive node unreachable: {e}")

    return payload


# ─────────────────────────────────────────────
#  UTILITY ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":         "ok",
        "models_loaded":  _MODELS is not None,
        "n_features":     N_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
        "threshold":      _MODELS["threshold"] if _MODELS else None,
    }


@app.post("/reset_window")
def reset_window():
    """
    Clear and re-zero the rolling window.
    Call after a PLC restart or any gap in data to prevent stale history
    from contaminating new readings.
    """
    with _WINDOW_LOCK:
        _WINDOW.clear()
        for _ in range(LSTM_TIMESTEPS):
            _WINDOW.append(np.zeros(N_FEATURES, dtype=np.float32))
    return {"status": "reset", "window_length": LSTM_TIMESTEPS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uvicorn api_detective:app --reload --port 8000
