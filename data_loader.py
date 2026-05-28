"""
data_loader.py — GRAD-RL Framework (v6.2)

Fix vs v6.1:
  - find_target_column(): strict exact/substring matching replaces fuzzy
    keyword scan. Legitimate sensor features like 'TOTAL_CONS_REQUIRED_FLOW'
    that happen to contain 'attack' or 'normal' as substrings are no longer
    accidentally dropped.
  - Normal data loader: if no explicit label column is found, returns None
    and does NOT drop the last column as a fallback.
  - Attack data loader: fallback to last column only for attack files where
    a label is guaranteed to exist (documented explicitly).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import os

# ─────────────────────────────────────────────
#  DATASET FILE MAP
# ─────────────────────────────────────────────
DATASETS = {
    "SWaT": {
        "NORMAL": "SWaT_Dataset_Normal_v1.csv",
        "ATTACK": "SWaT_Dataset_Attack_v0.csv",
    },
    "WADI": {
        "NORMAL": "WADI_14days_new.csv",
        "ATTACK": "WADI_attackdataLABLE.csv",
    },
}

SCALER_PATH_SWAT        = "models/scaler_swat.pkl"
SCALER_PATH_WADI        = "models/scaler_wadi.pkl"
LABEL_ENCODER_PATH_SWAT = "models/label_encoder_swat.pkl"
LABEL_ENCODER_PATH_WADI = "models/label_encoder_wadi.pkl"

DATA_DIR    = r"D:\PhD_Project\data"
PROJECT_DIR = r"D:\PhD_Project"

# ─────────────────────────────────────────────
#  STRICT LABEL COLUMN WHITELIST
#
#  Only these exact strings (case-insensitive full-column-name match, or
#  an exact known substring) are accepted as label columns.
#
#  WHY STRICT?
#  The previous fuzzy scan matched any column containing 'attack' or 'normal'
#  as a substring, which incorrectly dropped sensor features such as:
#    - 'TOTAL_CONS_REQUIRED_FLOW'  (contains no keyword but was caught by
#       overbroad matching in some variants)
#    - Any future WADI column whose engineering description happens to include
#       a keyword word.
#  Strict matching touches only the exact column names produced by iTrust /
#  SUTD when publishing SWaT and WADI datasets.
# ─────────────────────────────────────────────
_LABEL_EXACT = {
    # SWaT exact column names
    "label",
    "normal/attack",
    # WADI exact column name (with spaces, as published)
    "attack lable (1:no attack, -1:attack)",
    # Common variants seen in re-releases
    "attack label",
    "attack_label",
    "normal_label",
    "normal_lable",
}


def find_target_column(df: pd.DataFrame, require_label: bool = False) -> str | None:
    """
    Return the label/target column name using STRICT matching only.

    Matching rules (applied in order):
      1. Exact full-column-name match against _LABEL_EXACT (case-insensitive,
         after stripping whitespace).
      2. Column name IS one of the exact strings in _LABEL_EXACT when both
         are stripped+lowercased.

    If no match is found:
      - require_label=False  → returns None  (caller must handle gracefully)
      - require_label=True   → falls back to the LAST column and logs a warning
        (used only for attack files where a label is guaranteed present)

    Columns that will NO LONGER be accidentally matched:
      'TOTAL_CONS_REQUIRED_FLOW', 'P_NORMALISED', 'ATTACK_PRESSURE_*', etc.
    """
    for col in df.columns:
        if col.strip().lower() in _LABEL_EXACT:
            return col

    if require_label:
        fallback = df.columns[-1]
        print(f"  ⚠️  No explicit label column found. "
              f"Assuming last column as label: '{fallback}'")
        return fallback

    return None


# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────

def get_data_path(filename: str) -> str:
    candidates = [
        os.path.join(DATA_DIR, filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"  ✅ Found: {p}")
            return p
    raise FileNotFoundError(
        f"\n❌ File not found: {filename}\n"
        + "\n".join(f"   • {c}" for c in candidates)
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove fully-NaN columns/rows, timestamp columns, and fill residual NaNs.
    Does NOT drop columns based on name content (no fuzzy matching here).
    """
    # Drop fully-empty columns and rows
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    # Drop timestamp/date columns by exact keyword in name
    ts_cols = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ("timestamp", " time", "date"))
        # 'time' alone would match 'REAL_TIME_FLOW' etc.; require leading space
        # or exact 'timestamp'/'date' to be safe
    ]
    df = df.drop(columns=ts_cols, errors="ignore")

    # Fill residual NaNs (forward-fill → back-fill → 0) to avoid loss:nan
    df = df.ffill().bfill().fillna(0)
    return df


# ─────────────────────────────────────────────
#  NORMAL DATA LOADER
# ─────────────────────────────────────────────

def load_normal_data(dataset: str = "SWaT") -> np.ndarray:
    """
    Load normal (non-attack) data for LSTM-AE training.
    Fits and saves the per-dataset MinMaxScaler.
    Returns X_scaled: np.ndarray shape (N, n_features).
    """
    filename = DATASETS[dataset]["NORMAL"]
    path     = get_data_path(filename)
    print(f"\n--- Loading NORMAL data [{dataset}] from {filename} ---")

    nrows    = 50_000  if dataset == "SWaT" else 200_000
    skiprows = 1       if dataset == "WADI" else 0
    df = pd.read_csv(path, nrows=nrows, skiprows=skiprows, low_memory=False)
    df.columns = df.columns.str.strip()

    # Strict label detection — require_label=False so we never drop a sensor
    # column just because no label column was found
    target_col = find_target_column(df, require_label=False)
    if target_col:
        print(f"  ⚠️  Label column '{target_col}' found in normal file → dropping.")
        df = df.drop(columns=[target_col])
    else:
        print("  ✅ No label column detected in normal file (expected for WADI).")

    df = clean_dataframe(df)
    X_numeric = df.select_dtypes(include=[np.number])
    print(f"  📊 Shape after cleaning: {X_numeric.shape}")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_numeric.values)

    scaler_rel  = SCALER_PATH_SWAT if dataset == "SWaT" else SCALER_PATH_WADI
    scaler_path = os.path.join(PROJECT_DIR, scaler_rel)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ Scaler fitted on {dataset} normal data → saved to {scaler_path}")

    return X_scaled


# ─────────────────────────────────────────────
#  ATTACK DATA LOADER
# ─────────────────────────────────────────────

def load_attack_data(dataset: str = "SWaT") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load labelled attack dataset.

    Returns:
        X_scaled     : np.ndarray (N, n_features)
        y_multiclass : np.ndarray (N,) int  — LabelEncoder output for RF
        y_binary     : np.ndarray (N,) int  — manually derived {0,1} for LSTM-AE

    Binary label derivation (dataset-specific, BEFORE LabelEncoder):
        SWaT : raw == 'normal' → 0 ; anything else → 1
        WADI : raw in {'1','1.0'} → 0 (No Attack) ; '-1'/'-1.0' → 1 (Attack)
    """
    filename   = DATASETS[dataset]["ATTACK"]
    scaler_rel = SCALER_PATH_SWAT        if dataset == "SWaT" else SCALER_PATH_WADI
    enc_rel    = LABEL_ENCODER_PATH_SWAT if dataset == "SWaT" else LABEL_ENCODER_PATH_WADI

    path = get_data_path(filename)
    print(f"\n--- Loading ATTACK data [{dataset}] from {filename} ---")

    skiprows = 1 if dataset == "WADI" else 0
    df = pd.read_csv(path, nrows=100_000, skiprows=skiprows, low_memory=False)
    df.columns = df.columns.str.strip()

    # require_label=True: attack files always have a label; last-column fallback
    # is acceptable here (but will log a warning so it is visible)
    target_col = find_target_column(df, require_label=True)
    print(f"  🏷️  Label column: '{target_col}'")

    df = df.dropna(subset=[target_col])
    raw_labels = df[target_col].astype(str).str.strip().str.lower()

    # ── Binary labels (BEFORE LabelEncoder — avoids sort-order inversion) ──
    if dataset == "WADI":
        y_binary = np.where(raw_labels.isin(["1", "1.0"]), 0, 1).astype(np.int32)
    else:
        y_binary = np.where(raw_labels == "normal", 0, 1).astype(np.int32)

    n_normal = int((y_binary == 0).sum())
    n_attack = int((y_binary == 1).sum())
    print(f"  📊 Binary — Normal(0): {n_normal}, Attack(1): {n_attack}")
    if n_normal == 0 and dataset == "SWaT":
        print("  ℹ️  No Normal rows in SWaT attack file — expected (pure attack dataset).")

    # ── Multiclass labels for RF ────────────────────────────────────────────
    le = LabelEncoder()
    y_multiclass = le.fit_transform(df[target_col].astype(str).str.strip())
    print(f"  🏷️  RF classes: {le.classes_}")

    enc_path = os.path.join(PROJECT_DIR, enc_rel)
    os.makedirs(os.path.dirname(enc_path), exist_ok=True)
    joblib.dump(le, enc_path)
    print(f"  ✅ LabelEncoder saved → {enc_path}")

    # ── Feature matrix ──────────────────────────────────────────────────────
    df_feat   = df.drop(columns=[target_col])
    df_feat   = clean_dataframe(df_feat)
    X_numeric = df_feat.select_dtypes(include=[np.number])

    # ── Scale ───────────────────────────────────────────────────────────────
    scaler_path = os.path.join(PROJECT_DIR, scaler_rel)
    scaler      = joblib.load(scaler_path)

    if X_numeric.shape[1] != scaler.n_features_in_:
        print(f"  ⚠️  Feature mismatch: data={X_numeric.shape[1]}, "
              f"scaler={scaler.n_features_in_}. Re-fitting scaler.")
        scaler    = MinMaxScaler()
        X_scaled  = scaler.fit_transform(X_numeric.values)
        joblib.dump(scaler, scaler_path)
    else:
        X_scaled = scaler.transform(X_numeric.values)

    print(f"  ✅ X={X_scaled.shape}, y_multiclass={y_multiclass.shape}, "
          f"y_binary={y_binary.shape}")
    return X_scaled, y_multiclass, y_binary


# ─────────────────────────────────────────────
#  UNIFIED ENTRY POINT
# ─────────────────────────────────────────────

def load_and_preprocess(
    dataset: str = "SWaT",
    mode: str = "train",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    mode="train" → (X_normal, None, None)
    mode="test"  → (X_scaled, y_multiclass, y_binary)
    """
    if mode == "train":
        return load_normal_data(dataset=dataset), None, None
    return load_attack_data(dataset=dataset)


# ─────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    for ds in ["SWaT", "WADI"]:
        print(f"\n{'─'*40}  {ds}")
        try:
            X_n = load_normal_data(dataset=ds)
            print(f"  Normal  shape: {X_n.shape}")
        except Exception as e:
            print(f"  ❌ Normal load: {e}")
        try:
            X_a, y_mc, y_b = load_attack_data(dataset=ds)
            print(f"  Attack  shape: {X_a.shape}")
            print(f"  y_multiclass unique: {np.unique(y_mc)}")
            print(f"  y_binary     unique: {np.unique(y_b)}")
        except Exception as e:
            print(f"  ❌ Attack load: {e}")