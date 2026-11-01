"""
wadi_sweep.py — Yeni model icin optimal threshold bul
Her iki model icin FPR-Recall egrisini karsilastir.
"""

import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import joblib
from sklearn.metrics import confusion_matrix

try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gradrl import paths
import gradrl.data_loader as dl

PROJECT_DIR = str(paths.ROOT)
MODEL_DIR   = str(paths.MODEL_DIR)

TIMESTEPS = 10

def make_windows(X, w=10):
    return np.array([X[i:i+w] for i in range(len(X)-w+1)])

def get_mse(model, X):
    wins  = make_windows(X, TIMESTEPS)
    preds = model.predict(wins, batch_size=256, verbose=0)
    return np.mean(np.power(wins[:,-1,:] - preds[:,-1,:], 2), axis=1)

def metrics_at_tau(mse, y, tau):
    pred = (mse > tau).astype(int)
    tn,fp,fn,tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    fpr  = fp/(fp+tn)*100 if (fp+tn) > 0 else 0
    rec  = tp/(tp+fn)*100 if (tp+fn) > 0 else 0
    prec = tp/(tp+fp)*100 if (tp+fp) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    return fpr, rec, prec, f1

print("\n" + "="*65)
print("  WADI Threshold Sweep: Eski vs Yeni Model")
print("="*65)

# Eski model
old_model = load_model(os.path.join(MODEL_DIR, "detective_lstm_wadi.keras"))
X_old, y_old = dl.load_attack_data("WADI")[:3:2]  # X, y_binary
_, _, y_old = dl.load_attack_data("WADI")
X_old, _, y_old = dl.load_attack_data("WADI")
mse_old = get_mse(old_model, X_old)
y_old_win = y_old[TIMESTEPS-1:]

# Yeni model  
dl_orig = dl.SCALER_PATH_WADI
dl.SCALER_PATH_WADI = "models/scaler_wadi_v2.pkl"
try:
    X_new, _, y_new = dl.load_attack_data("WADI")
finally:
    dl.SCALER_PATH_WADI = dl_orig
new_model = load_model(os.path.join(MODEL_DIR, "detective_lstm_wadi_v2.keras"))
mse_new = get_mse(new_model, X_new)
y_new_win = y_new[TIMESTEPS-1:]

print(f"\nEski model MSE: mean={mse_old.mean():.5f}  P95={np.percentile(mse_old,95):.5f}  P99={np.percentile(mse_old,99):.5f}")
print(f"Yeni model MSE: mean={mse_new.mean():.5f}  P95={np.percentile(mse_new,95):.5f}  P99={np.percentile(mse_new,99):.5f}")

# Sweep: P80 ile P99.9 arasi
print("\n" + "="*75)
print(f"{'Tau':>10} | {'Eski FPR':>9} {'Eski Rec':>9} {'Eski F1':>8} | {'Yeni FPR':>9} {'Yeni Rec':>9} {'Yeni F1':>8} | Yeni iyi?")
print("-"*90)

for pct in [80, 85, 90, 92, 94, 95, 96, 97, 98, 99, 99.5]:
    tau_o = np.percentile(mse_old, pct)
    tau_n = np.percentile(mse_new, pct)
    fo,ro,_,f1o = metrics_at_tau(mse_old, y_old_win, tau_o)
    fn,rn,_,f1n = metrics_at_tau(mse_new, y_new_win, tau_n)
    better = "✓ FPR" if fn < fo-1 else ("✓ F1" if f1n > f1o else "")
    print(f"P{pct:>4.1f}  o:{tau_o:.4f}/n:{tau_n:.4f} | {fo:>8.1f}% {ro:>8.1f}% {f1o:>7.2f}% | {fn:>8.1f}% {rn:>8.1f}% {f1n:>7.2f}% | {better}")

# Sabit tau sweep (eski modelin threshold'unu yeni modele uygula)
print("\n" + "="*75)
print("Eski modelin tau_live=0.0189 degeri yeni modele uygulandiginda:")
print("-"*75)
for tau in [0.010, 0.015, 0.019, 0.025, 0.030, 0.040, 0.055]:
    fo,ro,_,f1o = metrics_at_tau(mse_old, y_old_win, tau)
    fn,rn,_,f1n = metrics_at_tau(mse_new, y_new_win, tau)
    better = "← YENI KAZANIYOR" if fn < fo-1 and rn > ro-5 else ""
    print(f"  tau={tau:.3f}: Eski FPR={fo:.1f}% Rec={ro:.1f}%  |  Yeni FPR={fn:.1f}% Rec={rn:.1f}%  {better}")

# En iyi yeni model operating point
best_f1 = 0
best_tau = 0.019
for tau in np.linspace(mse_new.min(), np.percentile(mse_new,99.9), 200):
    fn,rn,_,f1n = metrics_at_tau(mse_new, y_new_win, tau)
    if f1n > best_f1 and fn < 25:  # FPR < 25% sartini koru
        best_f1 = f1n
        best_tau = tau
        best_fpr = fn
        best_rec = rn

fo,ro,_,f1o = metrics_at_tau(mse_old, y_old_win, 0.019)
print(f"\n{'='*65}")
print(f"Yeni modelin optimal noktasi (FPR < 25%):")
print(f"  tau={best_tau:.5f}: FPR={best_fpr:.2f}%  Recall={best_rec:.2f}%  F1={best_f1:.2f}%")
print(f"  Eski model ref:  FPR={fo:.2f}%  Recall={ro:.2f}%  F1={f1o:.2f}%")
if best_fpr < fo - 1:
    print(f"  ✅ Yeni model daha iyi! FPR {fo-best_fpr:.1f}% dusuk")
elif best_rec > ro + 5:
    print(f"  ✅ Yeni model recall'da daha iyi (+{best_rec-ro:.1f}%)")
else:
    print(f"  ⚠️  Yeni model belirgin avantaj saglamiyor")
