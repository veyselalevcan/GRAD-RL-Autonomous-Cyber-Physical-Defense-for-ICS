"""
wadi_retrain_v2.py — GRAD-RL Framework
WADI LSTM-AE yeniden egitimi — data_loader kullanarak dogru preprocessing

DUZELTME: Onceki retrain script'i 128 feature aliyordu (4 tane all-NaN
kolonu clean_dataframe ile silinmiyordu). Bu script data_loader.py'i
dogrudan import ederek AYNI 124 feature'i uretir.

Scaler: v2 olarak ayri kaydedilir (orijinal bozulmaz)
Model:  detective_lstm_wadi_v2.keras
Thresh: threshold_wadi_v2.json
"""

import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import joblib

# ─────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gradrl import paths

PROJECT_DIR = str(paths.ROOT)
MODEL_DIR   = str(paths.MODEL_DIR)

# data_loader import
import gradrl.data_loader as dl

# ─────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUT_MODEL  = os.path.join(MODEL_DIR, "detective_lstm_wadi_v2.keras")
OUT_SCALER = os.path.join(MODEL_DIR, "scaler_wadi_v2.pkl")
OUT_THRESH = os.path.join(MODEL_DIR, "threshold_wadi_v2.json")

TIMESTEPS = 10
# ─────────────────────────────────────────────

def make_windows(X, w=10):
    return np.array([X[i:i+w] for i in range(len(X) - w + 1)])

def build_lstm_ae(n_features, timesteps=10):
    inp = tf.keras.Input(shape=(timesteps, n_features))
    x   = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inp)
    x   = tf.keras.layers.LSTM(32, return_sequences=False, dropout=0.2)(x)
    x   = tf.keras.layers.RepeatVector(timesteps)(x)
    x   = tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)(x)
    x   = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(x)
    return tf.keras.Model(inp, out)

# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  WADI Retrain v2 — data_loader dogrudan kullaniliyor")
print("="*60)

# 1. data_loader ile WADI normal verisini yukle
#    Bu ayni scaler_wadi.pkl'i OVERWRITE eder.
#    v2 modeli icin bunu kopyalayacagiz.
print("\n[1] data_loader.load_normal_data('WADI') calistiriliyor...")

# Gecici olarak scaler path'i v2'ye yonlendir
original_scaler_path = dl.SCALER_PATH_WADI
dl.SCALER_PATH_WADI  = "models/scaler_wadi_v2.pkl"

try:
    X_normal = dl.load_normal_data(dataset="WADI")
finally:
    dl.SCALER_PATH_WADI = original_scaler_path  # Geri al

print(f"    Shape: {X_normal.shape}")
n_features = X_normal.shape[1]
print(f"    n_features = {n_features}  (124 olmali)")

# v2 scaler'i dogrudan kaydet
scaler_v2 = joblib.load(os.path.join(PROJECT_DIR, "models/scaler_wadi_v2.pkl"))
print(f"    Scaler v2 kaydedildi: {OUT_SCALER}")

# 2. Train / Val bolumu
split      = int(len(X_normal) * 0.8)
X_train    = X_normal[:split]
X_val      = X_normal[split:]
wins_train = make_windows(X_train, TIMESTEPS)
wins_val   = make_windows(X_val,   TIMESTEPS)
print(f"\n[2] Train windows: {len(wins_train)} | Val windows: {len(wins_val)}")

# Baslangic train-val gap kontrolu
print(f"    Train MSE range: {X_train.min():.4f} – {X_train.max():.4f}")
print(f"    Val MSE range:   {X_val.min():.4f} – {X_val.max():.4f}")

# 3. Model
print(f"\n[3] LSTM-AE kuruluyor (n_features={n_features})...")
model = build_lstm_ae(n_features, TIMESTEPS)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

# 4. Callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-6,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1,
    ),
]

# 5. Egitim
print("\n[4] Egitim basliyor...")
print("    patience=10  |  ReduceLROnPlateau  |  max_epochs=60")
print("    Beklenen: val_loss steadily azalmali, gap ~1-3x olmali\n")
history = model.fit(
    wins_train, wins_train,
    validation_data=(wins_val, wins_val),
    epochs=60,
    batch_size=128,
    callbacks=callbacks,
    verbose=1,
)

# 6. Sonuc
final_train = history.history["loss"][-1]
final_val   = history.history["val_loss"][-1]
gap         = final_val / final_train if final_train > 0 else 999
best_epoch  = np.argmin(history.history["val_loss"]) + 1

print(f"\n[5] Egitim tamamlandi:")
print(f"    Best epoch:      {best_epoch}")
print(f"    Final train:     {final_train:.6f}")
print(f"    Final val:       {final_val:.6f}")
print(f"    Train-val gap:   {gap:.2f}×  (eski: 8.6×)")

# 7. Threshold
preds_val = model.predict(wins_val, batch_size=256, verbose=0)
mse_val   = np.mean((wins_val[:, -1, :] - preds_val[:, -1, :])**2, axis=1)
tau_p99   = float(np.percentile(mse_val, 99))
tau_live  = round(tau_p99 * 1.2, 8)

print(f"\n[6] Threshold hesaplandi:")
print(f"    tau_offline (P99 val): {tau_p99:.8f}")
print(f"    tau_live    (×1.2):    {tau_live:.8f}")

# Karsilastirma ile orijinal
with open(os.path.join(MODEL_DIR, "threshold_wadi.json")) as f:
    old_tau = json.load(f)["threshold"]
print(f"    Eski tau_offline:      {old_tau:.8f}")
print(f"    Oran (yeni/eski):      {tau_p99/old_tau:.2f}×  (1.0'e yakin olmali)")

# 8. Kaydet
model.save(OUT_MODEL)
with open(OUT_THRESH, "w") as f:
    json.dump({
        "threshold":      tau_live,
        "tau_offline":    tau_p99,
        "dataset":        "WADI",
        "method":         "percentile_99",
        "model":          "v2_patience10",
        "train_val_gap":  round(gap, 3),
        "best_epoch":     int(best_epoch),
    }, f, indent=2)

print(f"\n✅ Model:   {OUT_MODEL}")
print(f"✅ Scaler:  {OUT_SCALER}")
print(f"✅ Thresh:  {OUT_THRESH}")

print("\n" + "="*60)
if gap <= 3.0 and abs(tau_p99/old_tau - 1.0) < 2.0:
    print("✅ BASARILI: Gap dusuk, threshold benzer scale'da")
    print("   Sonraki adim: python src/defense/wadi_eval_v2.py")
elif gap <= 3.0:
    print("⚠️  Gap iyi ama threshold hala farkli scale'da")
    print("   Bu normal olabilir — eval calistir, FPR/Recall'a bak")
else:
    print(f"⚠️  Gap hala yuksek ({gap:.1f}x) — daha fazla epoch gerekebilir")
print("="*60)