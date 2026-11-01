"""
hai_pipeline.py — GRAD-RL Framework: HAI 20.07 Dataset
Steam turbine + water treatment ICS anomaly detection

Kullanim:
    python src/hai_pipeline.py

Adimlar:
  1. Train/Test yukle (auto separator detection)
  2. LSTM-AE egit (normal data only)
  3. Threshold hesapla (P99 val)
  4. Attack data uzerinde evaluate et
  5. Sonuclari raporla
"""

import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score

try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, LSTM, Dense,
                                          RepeatVector, TimeDistributed)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
except ImportError:
    from keras.models import Model
    from keras.layers import (Input, LSTM, Dense,
                               RepeatVector, TimeDistributed)
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam

# ─────────────────────────────────────────────
PROJECT_DIR = r"D:\PhD_Project"
HAI_DIR     = r"D:\ICS Data and Papers\hai data and models\hai-20.07"
MODEL_DIR   = os.path.join(PROJECT_DIR, "models")

TRAIN_FILE  = os.path.join(HAI_DIR, "train.csv")
TEST_FILE   = os.path.join(HAI_DIR, "test.csv")

OUT_MODEL   = os.path.join(MODEL_DIR, "detective_lstm_hai.keras")
OUT_SCALER  = os.path.join(MODEL_DIR, "scaler_hai.pkl")
OUT_THRESH  = os.path.join(MODEL_DIR, "threshold_hai.json")
OUT_RESULTS = os.path.join(MODEL_DIR, "hai_results.json")

TIMESTEPS   = 10
PATIENCE    = 10
MAX_EPOCHS  = 60
SEED        = 42

LABEL_COLS  = ['attack', 'attack_P1', 'attack_P2', 'attack_P3']

np.random.seed(SEED)

# ─────────────────────────────────────────────
def detect_sep(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        line = f.readline()
    if ';' in line: return ';'
    if '\t' in line: return '\t'
    return ','

def load_hai(path, nrows=None):
    sep = detect_sep(path)
    df  = pd.read_csv(path, sep=sep, nrows=nrows,
                      low_memory=False, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    return df

def make_windows(X, w=10):
    return np.array([X[i:i+w] for i in range(len(X) - w + 1)])

def build_lstm_ae(n_features, timesteps=10):
    inp = Input(shape=(timesteps, n_features))
    x   = LSTM(64, return_sequences=True, dropout=0.2)(inp)
    x   = LSTM(32, return_sequences=False, dropout=0.2)(x)
    x   = RepeatVector(timesteps)(x)
    x   = LSTM(32, return_sequences=True, dropout=0.2)(x)
    x   = LSTM(64, return_sequences=True, dropout=0.2)(x)
    out = TimeDistributed(Dense(n_features))(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def metrics(y_true, y_pred, label=""):
    tn,fp,fn,tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    fpr  = fp/(fp+tn)*100 if (fp+tn) > 0 else 0
    rec  = tp/(tp+fn)*100 if (tp+fn) > 0 else 0
    prec = tp/(tp+fp)*100 if (tp+fp) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    acc  = (tp+tn)/len(y_true)*100
    return dict(label=label, FPR=fpr, Recall=rec,
                Precision=prec, F1=f1, Accuracy=acc,
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))

# ─────────────────────────────────────────────
print("\n" + "="*65)
print("  HAI 20.07 — GRAD-RL Pipeline")
print("="*65)

# ── 1. VERİ YUKLE ────────────────────────────
print("\n[1] Veri yukleniyor...")
df_train = load_hai(TRAIN_FILE)
df_test  = load_hai(TEST_FILE)

print(f"    Train: {df_train.shape} | sep='{detect_sep(TRAIN_FILE)}'")
print(f"    Test:  {df_test.shape}  | sep='{detect_sep(TEST_FILE)}'")

# Sensor sutunlari (label ve time haric)
time_col    = df_train.columns[0]
sensor_cols = [c for c in df_train.columns
               if c not in LABEL_COLS + [time_col]
               and df_train[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

print(f"    Sensor sutun sayisi: {len(sensor_cols)}")
print(f"    Label sutunlari: {[c for c in LABEL_COLS if c in df_train.columns]}")

# Train: sadece normal satırlar (attack=0)
attack_col = 'attack' if 'attack' in df_train.columns else None
if attack_col:
    df_normal = df_train[df_train[attack_col] == 0][sensor_cols]
    print(f"\n    Normal train: {len(df_normal):,} satir")
    print(f"    (Train attack: {(df_train[attack_col]!=0).sum():,} satir, egitimde kullanilmiyor)")
else:
    df_normal = df_train[sensor_cols]
    print(f"\n    Normal train (attack col yok): {len(df_normal):,} satir")

df_normal = df_normal.ffill().bfill().fillna(0)

# ── 2. SCALER ───────────────────────────────
print("\n[2] MinMaxScaler fit ediliyor...")
scaler    = MinMaxScaler()
X_normal  = scaler.fit_transform(df_normal.values)
joblib.dump(scaler, OUT_SCALER)
n_features = X_normal.shape[1]
print(f"    n_features = {n_features}")
print(f"    Scaler kaydedildi: {OUT_SCALER}")

# Train / Val bolumu (80/20 temporal)
split    = int(len(X_normal) * 0.8)
X_train  = X_normal[:split]
X_val    = X_normal[split:]
wins_tr  = make_windows(X_train, TIMESTEPS)
wins_val = make_windows(X_val,   TIMESTEPS)
print(f"\n    Train windows: {len(wins_tr):,} | Val windows: {len(wins_val):,}")

# ── 3. MODEL EGIT ────────────────────────────
print("\n[3] LSTM-AE egitiliyor...")
print(f"    patience={PATIENCE} | max_epochs={MAX_EPOCHS}")

model = build_lstm_ae(n_features, TIMESTEPS)
model.summary(print_fn=lambda x: None)  # sessiz

cbs = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE,
                  min_delta=1e-6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=4, min_lr=1e-5, verbose=0),
]

history = model.fit(
    wins_tr, wins_tr,
    validation_data=(wins_val, wins_val),
    epochs=MAX_EPOCHS,
    batch_size=128,
    callbacks=cbs,
    verbose=1,
)

final_train = history.history['loss'][-1]
final_val   = history.history['val_loss'][-1]
gap         = final_val / final_train if final_train > 0 else 999
best_epoch  = int(np.argmin(history.history['val_loss'])) + 1

print(f"\n    Best epoch:    {best_epoch}")
print(f"    Train loss:    {final_train:.6f}")
print(f"    Val loss:      {final_val:.6f}")
print(f"    Train-val gap: {gap:.2f}x")

# ── 4. THRESHOLD ─────────────────────────────
preds_val = model.predict(wins_val, batch_size=256, verbose=0)
mse_val   = np.mean((wins_val[:,-1,:] - preds_val[:,-1,:])**2, axis=1)
tau_off   = float(np.percentile(mse_val, 99))
tau_live  = round(tau_off * 1.2, 8)

print(f"\n[4] Threshold: tau_offline={tau_off:.6f}  tau_live={tau_live:.6f}")

model.save(OUT_MODEL)
with open(OUT_THRESH, 'w') as f:
    json.dump({'threshold': tau_live, 'tau_offline': tau_off,
               'dataset': 'HAI', 'train_val_gap': round(gap, 3),
               'best_epoch': best_epoch}, f, indent=2)

# ── 5. ATTACK DATA DEĞERLENDİRME ────────────
print("\n[5] Test (attack) verisi degerlendiriliyor...")

df_test_feat = df_test[sensor_cols].ffill().bfill().fillna(0)

# Feature sayısı kontrolü
if df_test_feat.shape[1] != n_features:
    print(f"    ⚠️  Feature mismatch: test={df_test_feat.shape[1]}, model={n_features}")
    # Ortak sutunlar
    common = [c for c in sensor_cols if c in df_test.columns]
    df_test_feat = df_test[common].ffill().bfill().fillna(0)
    print(f"    Ortak sensor: {len(common)}")

X_test  = scaler.transform(df_test_feat.values)
wins_ts = make_windows(X_test, TIMESTEPS)
y_full  = df_test['attack'].values[TIMESTEPS-1:] if 'attack' in df_test.columns else None

print(f"    Test windows: {len(wins_ts):,}")
if y_full is not None:
    print(f"    Normal: {(y_full==0).sum():,} | Attack: {(y_full!=0).sum():,} ({(y_full!=0).mean()*100:.1f}%)")

preds_ts = model.predict(wins_ts, batch_size=256, verbose=1)
mse_ts   = np.mean((wins_ts[:,-1,:] - preds_ts[:,-1,:])**2, axis=1)
y_pred   = (mse_ts > tau_live).astype(int)

if y_full is not None:
    y_true_bin = (y_full != 0).astype(int)
    m = metrics(y_true_bin, y_pred, "HAI LSTM-AE")

    print("\n" + "="*65)
    print("  HAI SONUCLARI")
    print("="*65)
    print(f"  Accuracy:  {m['Accuracy']:.2f}%")
    print(f"  FPR:       {m['FPR']:.2f}%")
    print(f"  Recall:    {m['Recall']:.2f}%")
    print(f"  Precision: {m['Precision']:.2f}%")
    print(f"  F1:        {m['F1']:.2f}%")
    print(f"  FN:        {m['FN']}  ({'ZERO!' if m['FN']==0 else 'var'})")
    print(f"  TP:        {m['TP']} | FP: {m['FP']} | TN: {m['TN']}")

    # Per-process attack breakdown
    print("\n  Process bazli saldiri breakdown:")
    for pcol in ['attack_P1', 'attack_P2', 'attack_P3']:
        if pcol in df_test.columns:
            y_p = df_test[pcol].values[TIMESTEPS-1:]
            n_att = (y_p != 0).sum()
            det   = ((y_pred==1) & (y_p!=0)).sum()
            rate  = det/n_att*100 if n_att > 0 else 0
            print(f"    {pcol}: {n_att} saldiri → {det} tespit ({rate:.1f}%)")

    # Kaydet
    results = {
        'model': 'LSTM-AE',
        'dataset': 'HAI_20.07',
        'n_sensors': n_features,
        'train_val_gap': round(gap, 3),
        'best_epoch': best_epoch,
        'tau_live': tau_live,
        **{k: round(float(v),4) if isinstance(v,float) else v
           for k,v in m.items() if k != 'label'}
    }
    with open(OUT_RESULTS, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Sonuclar: {OUT_RESULTS}")

    # Paper ozet
    print("\n" + "="*65)
    print("  PAPER ICIN OZET SATIR")
    print("="*65)
    print(f"""
HAI 20.07 (steam turbine + water treatment):
  LSTM Acc={m['Accuracy']:.2f}%  FPR={m['FPR']:.2f}%  FN={m['FN']}  F1={m['F1']/100:.3f}
  Train-val gap: {gap:.2f}x (best epoch: {best_epoch})
""")