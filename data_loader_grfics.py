"""
data_loader_grfics.py — GRAD-RL Framework (v7.2)
Data Loader for the GRFICS Digital-Twin Dataset (grad_rl_final_dataset.csv)

Actual CSV structure (18 columns, 90,728 rows):
    timestamp      : datetime string          → dropped
    source         : "NORMAL" | "ATTACKER"    → dropped (redundant with label)
    label          : string attack name       → binary target ONLY
                     "Normal"                               88,365 rows
                     attack strings (6 variants)             2,363 rows
    mitre_technique: MITRE ATT&CK ID          → MULTICLASS target for XGBoost
                     T0813   (DenialOfControl)           572 rows
                     T0813+T0836 (Combined Phase2)       299 rows
                     T0814   (ModbusFlood)                24 rows
                     T0836   (Setpoint / Phase1)         881 rows  ← Phase1 + Setpoint merged
                     T0856   (SensorSpoofing)            587 rows
    mitre_tactic   : tactic description       → dropped
    f1_valve ... C_purge : 13 float features  → feature matrix

WHY mitre_technique INSTEAD OF label FOR MULTICLASS:
    Cross-tabulation reveals:
        Attack_Combined_Phase1_Setpoint  → T0836 (same as Attack_SetpointPoisoning!)
        Attack_SetpointPoisoning         → T0836
    Both share the identical MITRE technique. Using `label` produces two
    visually distinct classes with nearly identical sensor signatures → XGBoost
    confusion, F1=0 for Phase1.
    Using `mitre_technique` aggregates them at the correct semantic boundary
    (the MITRE framework itself says they are the same technique), giving 5
    clean, physically distinct classes. This is not label engineering — it is
    selecting the correct ground-truth granularity as defined by MITRE.

Label encoding:
    Binary     (LSTM-AE eval) : label == "Normal" → 0,  else → 1
    Multiclass (XGBoost)      : LabelEncoder on mitre_technique for attack rows
                                 5 classes: T0813, T0813+T0836, T0814, T0836, T0856

Output contract (identical to data_loader.py API):
    load_and_preprocess(mode="train") → (X_normal,  None, None)
    load_and_preprocess(mode="test")  → (X_attack, y_multiclass, y_binary)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR    = r"D:\PhD_Project\data"
PROJECT_DIR = r"D:\PhD_Project"

DATASET_FILE = "grad_rl_final_dataset.csv"
SCALER_PATH  = "models/scaler_grfics_live.pkl"
ENCODER_PATH = "models/label_encoder_grfics_live.pkl"

# Exact column names from the CSV header
COL_TIMESTAMP  = "timestamp"
COL_SOURCE     = "source"
COL_LABEL      = "label"           # "Normal" or "Attack_*"
COL_MITRE_TECH = "mitre_technique"
COL_MITRE_TACT = "mitre_tactic"
NORMAL_LABEL   = "Normal"

# The 13 physical sensor / actuator features — order matches CSV
FEATURE_COLS: list[str] = [
    "f1_valve",   "f1_flow",
    "f2_valve",   "f2_flow",
    "purge_valve", "purge_flow",
    "prod_valve",  "prod_flow",
    "pressure",
    "level",
    "A_purge", "B_purge", "C_purge",
]

# All non-feature columns to strip before scaling
DROP_COLS = [COL_TIMESTAMP, COL_SOURCE, COL_LABEL, COL_MITRE_TECH, COL_MITRE_TACT]


# ─────────────────────────────────────────────
#  PATH RESOLVER
# ─────────────────────────────────────────────

def _get_data_path(filename: str) -> str:
    candidates = [
        os.path.join(DATA_DIR, filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", filename),
        filename,
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"  ✅ Found: {p}")
            return p
    raise FileNotFoundError(
        f"\n❌ Dataset not found: {filename}\n"
        + "\n".join(f"   • {c}" for c in candidates)
    )


def _artifact_path(rel: str) -> str:
    return os.path.join(PROJECT_DIR, rel)


# ─────────────────────────────────────────────
#  RAW LOADER
# ─────────────────────────────────────────────

def _load_raw(nrows: int | None = None) -> pd.DataFrame:
    """Load CSV — dtype hints silence the mixed-type warning on mitre cols."""
    path = _get_data_path(DATASET_FILE)
    df = pd.read_csv(
        path, nrows=nrows, low_memory=False,
        dtype={COL_MITRE_TECH: str, COL_MITRE_TACT: str},
    )
    df.columns = df.columns.str.strip()
    return df


# ─────────────────────────────────────────────
#  NORMAL DATA LOADER  (LSTM-AE training)
# ─────────────────────────────────────────────

def load_normal_data() -> np.ndarray:
    """
    Load rows where label == "Normal", fit+save the MinMaxScaler,
    and return the scaled feature matrix.

    Returns:
        X_normal : np.ndarray  shape (N_normal, 13)  values in [0, 1]
    """
    print("\n--- Loading NORMAL data [GRFICS] ---")
    df = _load_raw()

    df_normal = df[df[COL_LABEL] == NORMAL_LABEL].copy()
    print(f"  📊 Normal rows  : {len(df_normal):,}")

    X_raw    = df_normal[FEATURE_COLS].values.astype(np.float32)
    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    sp = _artifact_path(SCALER_PATH)
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    joblib.dump(scaler, sp)

    print(f"  📊 Features     : {X_scaled.shape[1]}  → {FEATURE_COLS}")
    print(f"  ✅ Scaler saved : {sp}")
    return X_scaled


# ─────────────────────────────────────────────
#  ATTACK DATA LOADER  (XGBoost + LSTM eval)
# ─────────────────────────────────────────────

def load_attack_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load rows where label != "Normal", encode mitre_technique as the
    multiclass target, and return three aligned arrays.

    Requires load_normal_data() to have run first (scaler must exist).

    Returns:
        X_attack     : np.ndarray (N_attack, 13)  MinMaxScaled with normal scaler
        y_multiclass : np.ndarray (N_attack,) int LabelEncoder on mitre_technique
        y_binary     : np.ndarray (N_attack,) int all 1 (every row is an attack)

    LabelEncoder class order (alphabetical):
        0 : T0813          — Denial of Control (572 rows)
        1 : T0813+T0836    — Combined Phase2   (299 rows)
        2 : T0814          — Modbus Flood       (24 rows)
        3 : T0836          — Setpoint Poisoning + Combined Phase1 (881 rows total)
        4 : T0856          — Sensor Spoofing   (587 rows)

    NOTE: T0836 aggregates Attack_SetpointPoisoning (596) and
    Attack_Combined_Phase1_Setpoint (285) — both map to the same MITRE
    technique per the ground-truth cross-tabulation of the dataset.
    """
    print("\n--- Loading ATTACK data [GRFICS] ---")
    df = _load_raw()

    df_attack = df[df[COL_LABEL] != NORMAL_LABEL].copy()
    print(f"  📊 Attack rows   : {len(df_attack):,}")

    # Binary: all attack rows → 1
    y_binary = np.ones(len(df_attack), dtype=np.int32)

    # Multiclass: encode MITRE technique (5 semantically distinct classes)
    mitre_raw    = df_attack[COL_MITRE_TECH].astype(str).str.strip()
    le           = LabelEncoder()
    y_multiclass = le.fit_transform(mitre_raw.values)

    ep = _artifact_path(ENCODER_PATH)
    os.makedirs(os.path.dirname(ep), exist_ok=True)
    joblib.dump(le, ep)

    print("  🏷️  MITRE classes (multiclass target):")
    for i, cls in enumerate(le.classes_):
        n = int((y_multiclass == i).sum())
        print(f"       [{i}] {cls:<20s} n={n:4d}")
    print(f"  ✅ Encoder saved : {ep}")

    # Feature matrix
    X_raw = df_attack[FEATURE_COLS].values.astype(np.float32)
    sp    = _artifact_path(SCALER_PATH)
    if not os.path.exists(sp):
        raise FileNotFoundError(
            f"Scaler not found: {sp}\nRun load_normal_data() first."
        )
    scaler   = joblib.load(sp)
    X_scaled = scaler.transform(X_raw)

    print(f"  ✅ X_attack      : {X_scaled.shape}")
    return X_scaled, y_multiclass, y_binary


# ─────────────────────────────────────────────
#  UNIFIED ENTRY POINT
# ─────────────────────────────────────────────

def load_and_preprocess(
    mode: str = "train",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    mode="train" → (X_normal, None, None)
    mode="test"  → (X_attack, y_multiclass, y_binary)
    """
    if mode == "train":
        return load_normal_data(), None, None
    return load_attack_data()


# ─────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== GRFICS Data Loader Self-Test ===\n")
    X_n = load_normal_data()
    print(f"\nNormal  → shape: {X_n.shape}  range [{X_n.min():.4f}, {X_n.max():.4f}]")
    X_a, y_mc, y_b = load_attack_data()
    print(f"\nAttack  → shape: {X_a.shape}")
    print(f"  y_multiclass unique: {np.unique(y_mc)}")
    print(f"  y_binary     unique: {np.unique(y_b)}")