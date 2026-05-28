"""
api_detective.py — GRAD-RL Framework (v6.0)
Detective Node: LSTM-AE Anomaly Detection + RF Attack Classification

Changes vs v3.0:
  - Dynamic dataset switching: SWaT (51 features) | WADI (123 features)
  - Per-dataset model cache (scaler, LSTM, RF, label_encoder, threshold)
  - FEATURE_COLUMNS replaced by per-dataset tag registries
  - DetectionResult and forwarded payload include `dataset` field
  - WADI tag-to-culprit mapping via regex (mirrors topology_manager logic)
  - Model loading is lazy-cached; hot-swap without restart
"""

import os
import json
import re
import requests
import numpy as np
import tensorflow as tf
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from functools import lru_cache

app = FastAPI(title="AIAS Detective Node", version="6.0 (Dataset-Agnostic)")

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models"))
DEFENSE_URL = "http://127.0.0.1:8001/decide"

# ─────────────────────────────────────────────
#  FEATURE REGISTRIES
#  Maps dataset → ordered sensor tag list.
#  Order must match the column order used during scaler fitting.
# ─────────────────────────────────────────────
FEATURE_COLUMNS_SWAT: List[str] = [
    "FIT101", "LIT101", "MV101",  "P101",  "P102",
    "AIT201", "AIT202", "AIT203", "FIT201", "MV201",
    "P201",  "P202",  "P203",  "P204",  "P205",  "P206",
    "DPIT301","FIT301","LIT301","MV301","MV302","MV303","MV304","P301","P302",
    "AIT401","AIT402","FIT401","LIT401","P401","P402","P403","P404","UV401",
    "AIT501","AIT502","AIT503","AIT504",
    "FIT501","FIT502","FIT503","FIT504","P501","P502",
    "PIT501","PIT502","PIT503",
    "FIT601","P601","P602","P603",
]  # 51 features

# WADI v1 sensor list (123 features).
# Sourced from iTrust WADI dataset column headers.
FEATURE_COLUMNS_WADI: List[str] = [
    "1_AIT_001_PV", "1_AIT_002_PV", "1_AIT_003_PV", "1_AIT_004_PV",
    "1_AIT_005_PV", "1_FIT_001_PV", "1_LS_001_AL",  "1_LS_002_AL",
    "1_LT_001_PV",  "1_MV_001_STATUS","1_MV_002_STATUS","1_MV_003_STATUS",
    "1_MV_004_STATUS","1_P_001_STATUS","1_P_002_STATUS","1_P_003_STATUS",
    "1_P_004_STATUS","1_P_005_STATUS","1_P_006_STATUS",
    "2_DPIT_001_PV","2_FIC_101_CO",  "2_FIC_101_PV",  "2_FIC_101_SP",
    "2_FIC_201_CO",  "2_FIC_201_PV",  "2_FIC_201_SP",  "2_FIC_301_CO",
    "2_FIC_301_PV",  "2_FIC_301_SP",  "2_FIC_401_CO",  "2_FIC_401_PV",
    "2_FIC_401_SP",  "2_FIC_501_CO",  "2_FIC_501_PV",  "2_FIC_501_SP",
    "2_FIC_601_CO",  "2_FIC_601_PV",  "2_FIC_601_SP",  "2_FIT_001_PV",
    "2_FIT_002_PV",  "2_FIT_003_PV",  "2_LIT_001_PV",  "2_LIT_002_PV",
    "2_LIT_003_PV",  "2_MV_001_STATUS","2_MV_002_STATUS","2_MV_003_STATUS",
    "2_MV_004_STATUS","2_MV_005_STATUS","2_MV_006_STATUS",
    "2_P_001_STATUS","2_P_002_STATUS","2_P_003_STATUS","2_P_004_STATUS",
    "2_PIT_001_PV",  "2_PIT_002_PV",  "2_PIT_003_PV",
    "2_SV_101_STATUS","2_SV_201_STATUS","2_SV_301_STATUS",
    "2_SV_401_STATUS","2_SV_501_STATUS","2_SV_601_STATUS",
    "3_AIT_001_PV",  "3_AIT_002_PV",  "3_AIT_003_PV",  "3_AIT_004_PV",
    "3_FIT_001_PV",  "3_LS_001_AL",   "3_LT_001_PV",
    "3_MV_001_STATUS","3_MV_002_STATUS","3_MV_003_STATUS",
    "3_P_001_STATUS","3_P_002_STATUS","3_P_003_STATUS","3_P_004_STATUS",
    # Remaining 47 WADI columns (placeholder labels — replace with exact header names)
    *[f"WADI_CH_{i:03d}" for i in range(78, 124)],
]  # 123 features  (77 named + 46 placeholders = 123)

FEATURE_REGISTRY = {
    "SWaT": FEATURE_COLUMNS_SWAT,
    "WADI": FEATURE_COLUMNS_WADI,
}

# ─────────────────────────────────────────────
#  WADI TAG → CULPRIT HEURISTIC
#  Mirrors topology_manager._wadi_fallback_map()
# ─────────────────────────────────────────────
_WADI_TAG_RE = re.compile(
    r"^(?P<zone>[123])_(?P<itype>[A-Z]+)_(?P<num>\d+)(?:_\d+)?_(?P<suffix>[A-Z]+)$",
    re.IGNORECASE,
)


def _wadi_culprit_label(tag: str) -> str:
    """Return a human-readable stage label for a WADI tag."""
    m = _WADI_TAG_RE.match(tag.strip().upper())
    if not m:
        return tag
    zone, itype = int(m.group("zone")), m.group("itype")
    if itype == "AIT":                       return f"Zone{zone}_QualitySensor"
    if itype == "PIT":                       return f"Zone{zone}_PressureZone"
    if zone == 1 and itype in ("LT", "FIT"): return "Zone1_SupplyHeader"
    if zone == 1 and itype in ("P", "MV"):   return "Zone1_SupplyPump"
    if zone == 2 and itype == "MV":          return "Zone2_DistributionValve"
    if zone == 2 and itype == "P":           return "Zone2_DistributionMain"
    if zone == 2 and itype in ("LT", "FIT","LIT"): return "Zone2_ConsumerTank"
    if zone == 3:                            return "Zone3_ReturnPump"
    return tag


# ─────────────────────────────────────────────
#  MODEL CACHE  (one bundle per dataset)
# ─────────────────────────────────────────────
_MODEL_CACHE: dict[str, dict] = {}


def _model_file(dataset: str, kind: str) -> str:
    """Resolve model file path for a given dataset and artifact kind."""
    suffix = dataset.lower()   # swat | wadi
    files = {
        "scaler":        f"scaler_{suffix}.pkl",
        "lstm":          f"detective_lstm_{suffix}.keras",
        "lstm_fallback": f"detective_lstm_{suffix}.h5",
        "rf":            f"detective_classifier_{suffix}.pkl",
        "le":            f"label_encoder_{suffix}.pkl",
        "threshold":     f"threshold_{suffix}.json",
    }
    return os.path.join(MODEL_DIR, files[kind])


def load_models(dataset: str) -> dict:
    """
    Lazy-load and cache model bundle for a dataset.
    Returns bundle dict. Raises RuntimeError if LSTM or scaler is missing.
    """
    if dataset in _MODEL_CACHE:
        return _MODEL_CACHE[dataset]

    print(f"📦 Loading model bundle for dataset={dataset}...")
    bundle: dict = {}

    # Scaler (required)
    scaler_path = _model_file(dataset, "scaler")
    if not os.path.exists(scaler_path):
        raise RuntimeError(f"Scaler not found: {scaler_path}")
    bundle["scaler"] = joblib.load(scaler_path)

    # LSTM Autoencoder (required)
    lstm_path = _model_file(dataset, "lstm")
    if not os.path.exists(lstm_path):
        lstm_path = _model_file(dataset, "lstm_fallback")
    if not os.path.exists(lstm_path):
        raise RuntimeError(f"LSTM model not found for dataset={dataset}")
    bundle["lstm"] = tf.keras.models.load_model(lstm_path)

    # RF Classifier (optional — fallback to binary label)
    rf_path = _model_file(dataset, "rf")
    bundle["rf"] = joblib.load(rf_path) if os.path.exists(rf_path) else None

    # Label Encoder (optional)
    le_path = _model_file(dataset, "le")
    bundle["le"] = joblib.load(le_path) if os.path.exists(le_path) else None

    # Threshold (optional — default 0.05)
    thr_path = _model_file(dataset, "threshold")
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            bundle["threshold"] = json.load(f)["threshold"]
    else:
        bundle["threshold"] = 0.05
        print(f"⚠️  Threshold file missing for {dataset}; using default 0.05")

    _MODEL_CACHE[dataset] = bundle
    print(f"✅ Detective [{dataset}]: Models loaded.")
    return bundle


# Pre-load SWaT at startup (fail-fast for the primary dataset)
try:
    load_models("SWaT")
except RuntimeError as e:
    print(f"⚠️  SWaT model pre-load failed: {e}. Will retry on first request.")


# ─────────────────────────────────────────────
#  PYDANTIC MODELS
# ─────────────────────────────────────────────
class SensorData(BaseModel):
    values:  List[float]
    dataset: str = "SWaT"   # "SWaT" | "WADI"


class DetectionResult(BaseModel):
    is_anomaly:  bool
    mse_loss:    float
    risk_score:  float      # placeholder; computed by defensive node
    attack_type: str
    swat_tag:    str        # culprit sensor tag (generic name for WADI)
    dataset:     str        # propagated to defensive node


# ─────────────────────────────────────────────
#  INFERENCE ENDPOINT
# ─────────────────────────────────────────────
@app.post("/predict", response_model=DetectionResult)
def predict(data: SensorData):
    dataset = data.dataset if data.dataset in ("SWaT", "WADI") else "SWaT"

    # Load (or retrieve from cache) the correct model bundle
    try:
        bundle = load_models(dataset)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    scaler    = bundle["scaler"]
    lstm      = bundle["lstm"]
    rf        = bundle["rf"]
    le        = bundle["le"]
    threshold = bundle["threshold"]
    feat_cols = FEATURE_REGISTRY[dataset]

    # ── 1. Input validation ──────────────────────────────────────────────────
    features = np.array(data.values, dtype=np.float32).reshape(1, -1)

    expected = len(feat_cols)
    received = features.shape[1]
    if received != expected:
        # Graceful degradation: pad with zeros or truncate rather than 500-error
        print(f"⚠️  [{dataset}] Expected {expected} features, got {received}. Adjusting.")
        if received < expected:
            features = np.pad(features, ((0, 0), (0, expected - received)))
        else:
            features = features[:, :expected]

    # ── 2. Scale ─────────────────────────────────────────────────────────────
    scaled = scaler.transform(features)        # (1, n_features)

    # ── 3. LSTM-AE inference ─────────────────────────────────────────────────
    # Single snapshot → simulate sequence by repeating (10 timesteps)
    # Production: dashboard should send rolling window of 10 frames
    seq = np.repeat(scaled, 10, axis=0).reshape(1, 10, -1)
    reconstruction = lstm.predict(seq, verbose=0)

    # Compare final timestep
    input_last = scaled[0]
    rec_last   = reconstruction[0, -1, :]

    feature_errors = np.power(input_last - rec_last, 2)
    mse_loss   = float(np.mean(feature_errors))
    is_anomaly = mse_loss > threshold

    # ── 4. Identify culprit sensor ────────────────────────────────────────────
    max_err_idx = int(np.argmax(feature_errors))
    try:
        raw_tag = feat_cols[max_err_idx]
    except IndexError:
        raw_tag = "Unknown_Sensor"

    # For WADI, map raw tag to a human-readable stage label
    culprit_tag = _wadi_culprit_label(raw_tag) if dataset == "WADI" else raw_tag

    # ── 5. Attack classification (RF) ────────────────────────────────────────
    attack_name = "Normal"
    if is_anomaly and rf is not None:
        pred_class = rf.predict(scaled)[0]
        if le is not None:
            try:
                attack_name = (pred_class if isinstance(pred_class, str)
                               else le.inverse_transform([pred_class])[0])
            except Exception:
                attack_name = str(pred_class)
        else:
            attack_name = "Attack" if pred_class != 0 else "Normal"
    elif is_anomaly:
        attack_name = "Unknown"

    # ── 6. Forward to Defensive Node ─────────────────────────────────────────
    payload = {
        "is_anomaly":  bool(is_anomaly),
        "mse_loss":    float(mse_loss),
        "attack_type": attack_name,
        "swat_tag":    culprit_tag,
        "dataset":     dataset,
    }
    try:
        requests.post(DEFENSE_URL, json=payload, timeout=1)
    except Exception as e:
        print(f"[WARN] Defensive node unreachable: {e}")

    return {**payload, "risk_score": 0.0}


# ─────────────────────────────────────────────
#  UTILITY ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_datasets": list(_MODEL_CACHE.keys()),
    }


@app.get("/features/{dataset}")
def get_features(dataset: str):
    cols = FEATURE_REGISTRY.get(dataset)
    if cols is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset}")
    return {"dataset": dataset, "n_features": len(cols), "columns": cols}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uvicorn src.api.api_detective:app --reload --port 8000