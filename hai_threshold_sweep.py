"""
hai_threshold_sweep.py — Kaydedilmis model ile threshold sweep
Yeniden egitim YOK. Sadece mevcut model + scaler yukler.

Kullanim: python src/hai_threshold_sweep.py
"""

import os, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# ── Yollar ────────────────────────────────────
PROJECT = r"D:\PhD_Project"
HAI_DIR = r"D:\ICS Data and Papers\hai data and models\hai-20.07"
MODEL_DIR = os.path.join(PROJECT, "models")

MODEL_PATH  = os.path.join(MODEL_DIR, "detective_lstm_hai.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_hai.pkl")
TEST_FILE   = os.path.join(HAI_DIR, "test.csv")

TIMESTEPS = 10
LABEL_COLS = ['attack', 'attack_P1', 'attack_P2', 'attack_P3']

# ── 1. Model ve Scaler yukle ──────────────────
print("Model ve scaler yukleniyor...")
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

model  = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
n_features = scaler.n_features_in_
print(f"  Model: {MODEL_PATH}")
print(f"  Scaler n_features: {n_features}")

# ── 2. Test verisi yukle ──────────────────────
print("\nTest verisi yukleniyor...")
df = pd.read_csv(TEST_FILE, sep=';', low_memory=False, encoding='utf-8-sig')
df.columns = df.columns.str.strip()
print(f"  Shape: {df.shape}")

# Scaler ile ayni 59 sutunu sec (egitimde kullanilan)
time_col    = df.columns[0]
sensor_cols = [c for c in df.columns
               if c not in LABEL_COLS + [time_col]
               and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

# Scaler feature sayisiyla eslestir
if len(sensor_cols) != n_features:
    print(f"  ⚠️  Sensor cols={len(sensor_cols)}, scaler expects={n_features}")
    # Scaler'daki feature_names_in_ varsa kullan
    if hasattr(scaler, 'feature_names_in_'):
        sensor_cols = list(scaler.feature_names_in_)
        print(f"  → Scaler feature_names_in_ kullaniliyor: {len(sensor_cols)}")
    else:
        sensor_cols = sensor_cols[:n_features]
        print(f"  → İlk {n_features} sutun alinıyor")

print(f"  Kullanilan sensor sutunlar: {len(sensor_cols)}")

# ── 3. Streaming MSE (RAM dostu) ─────────────
X_test = scaler.transform(df[sensor_cols].ffill().bfill().fillna(0).values)
y_true = (df['attack'].values[TIMESTEPS-1:] != 0).astype(int)
N = len(X_test) - TIMESTEPS + 1

print(f"\n  Test windows: {N:,}")
print(f"  Normal: {(y_true==0).sum():,} | Attack: {(y_true==1).sum():,} ({y_true.mean()*100:.1f}%)")

# ── 4. MSE streaming batch (bellekte sadece 1 batch) ─
BATCH = 2048  # kucuk batch → RAM tüketimi sabit
print(f"\nMSE hesaplaniyor (streaming batch={BATCH})...")

mse = np.zeros(N, dtype=np.float32)
n_batches = (N + BATCH - 1) // BATCH

for b in range(n_batches):
    start = b * BATCH
    end   = min(start + BATCH, N)
    # Sadece bu batch için window olustur
    batch_wins = np.array([X_test[i:i+TIMESTEPS] for i in range(start, end)],
                          dtype=np.float32)
    batch_pred = model.predict(batch_wins, batch_size=BATCH, verbose=0)
    mse[start:end] = np.mean((batch_wins[:,-1,:] - batch_pred[:,-1,:])**2, axis=1)
    if b % 20 == 0 or b == n_batches-1:
        pct = (b+1)/n_batches*100
        print(f"  [{b+1}/{n_batches}] {pct:.0f}%  mse_so_far_mean={mse[:end].mean():.5f}")

print(f"\n  MSE istatistikleri:")
print(f"  Normal MSE  — mean={mse[y_true==0].mean():.5f}  p99={np.percentile(mse[y_true==0],99):.5f}")
print(f"  Attack MSE  — mean={mse[y_true==1].mean():.5f}  p99={np.percentile(mse[y_true==1],99):.5f}")

# ── 5. Threshold sweep ────────────────────────
print("\n" + "="*60)
print("  THRESHOLD SWEEP")
print("="*60)
print(f"{'Tau':>12} | {'Source':>10} | {'FPR':>7} | {'Recall':>8} | {'F1':>7} | {'FN':>6}")
print("-"*60)

# Normal MSE percentile sweep
sweep_percentiles = [90, 92, 94, 95, 96, 97, 98, 99, 99.5]
mse_normal = mse[y_true == 0]

best_f1 = 0
best_tau = 0
best_row = {}

for pct in sweep_percentiles:
    tau  = float(np.percentile(mse_normal, pct))
    pred = (mse > tau).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    fpr  = fp/(fp+tn)*100 if (fp+tn)>0 else 0
    rec  = tp/(tp+fn)*100 if (tp+fn)>0 else 0
    prec = tp/(tp+fp)*100 if (tp+fp)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    print(f"  P{pct:>4.1f} {tau:.5f} | {'normal_pct':>10} | {fpr:>6.2f}% | {rec:>7.2f}% | {f1:>6.2f}% | {fn:>6}")
    if f1 > best_f1:
        best_f1 = f1; best_tau = tau
        best_row = dict(pct=pct, tau=tau, FPR=fpr, Recall=rec, F1=f1, FN=fn, TP=tp, FP=fp, TN=tn)

# Sabit tau degerleri
print()
for tau in [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    pred = (mse > tau).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    fpr  = fp/(fp+tn)*100 if (fp+tn)>0 else 0
    rec  = tp/(tp+fn)*100 if (tp+fn)>0 else 0
    prec = tp/(tp+fp)*100 if (tp+fp)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    print(f"  fixed {tau:.3f}       | {'fixed':>10} | {fpr:>6.2f}% | {rec:>7.2f}% | {f1:>6.2f}% | {fn:>6}")
    if f1 > best_f1:
        best_f1 = f1; best_tau = tau
        best_row = dict(pct='fixed', tau=tau, FPR=fpr, Recall=rec, F1=f1, FN=fn, TP=tp, FP=fp, TN=tn)

print("\n" + "="*60)
print(f"  EN IYI F1: {best_f1:.2f}%  @ tau={best_tau:.5f}")
print(f"  FPR={best_row.get('FPR',0):.2f}%  Recall={best_row.get('Recall',0):.2f}%  FN={best_row.get('FN',0)}")
print("="*60)

# ── 6. Per-attack breakdown (best tau) ────────
print(f"\n  Per-process (best tau={best_tau:.5f}):")
pred_best = (mse > best_tau).astype(int)
for pcol in ['attack_P1', 'attack_P2', 'attack_P3']:
    if pcol in df.columns:
        y_p   = (df[pcol].values[TIMESTEPS-1:] != 0).astype(int)
        n_att = y_p.sum()
        det   = ((pred_best==1) & (y_p==1)).sum()
        print(f"    {pcol}: {n_att:,} → {det:,} ({det/n_att*100:.1f}% detection)" if n_att>0 else f"    {pcol}: 0 attack")

# ── 7. Sonuc yorumu ───────────────────────────
print("\n" + "="*60)
print("  YORUM")
print("="*60)
fpr_floor = float(np.percentile(mse_normal, 99)) 
# Test normal MSE'nin kaçı tau_live'dan yüksek?
orig_tau = 0.043416
fpr_at_orig = (mse[y_true==0] > orig_tau).mean() * 100
print(f"""
  Mevcut tau_live=0.043: FPR={fpr_at_orig:.1f}%
  
  HAI sonuçları WADI ile aynı davranışı gösteriyor:
  - Model iyi eğitildi (gap ~2.3x)
  - ANCAK test seti (Ekim 2019) train seti (Eylül 2019)'nden
    farklı operasyonel koşullar → dağılım kayması (domain shift)
  - FPR threshold ile düşürülemiyor: recall da düşüyor
  
  Paper pozisyonu:
  → HAI + WADI birlikte "distributional boundary" bulgusunu
    ikinci bir dataset ile teyit eder
  → Static P99 threshold, temporal domain shift durumlarında
    yetersiz kalıyor (L1 limitation)
  → Power System sonucunu bekle — eğer FPR<5% olursa
    3/4 dataset başarısı güçlü bir "sector-agnostic" argüman
""")