"""
msu_power_pipeline.py — GRAD-RL Framework: MSU Power System Dataset
Electric transmission ICS anomaly detection
4 PMUs × 29 features = 116 sensor columns, 120 Hz, 37 scenarios

Kullanim:
    python src/msu_power_pipeline.py

Referans:
    Adhikari, Pan, Morris et al. (2015) IEEE Trans. Smart Grid
    doi:10.1109/TSG.2015.2409775
"""

import os, sys, json, glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

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
DATA_DIR    = r"D:\ICS Data and Papers\msu power dataset"
MODEL_DIR   = os.path.join(PROJECT_DIR, "models")

OUT_MODEL   = os.path.join(MODEL_DIR, "detective_lstm_msu_power.keras")
OUT_SCALER  = os.path.join(MODEL_DIR, "scaler_msu_power.pkl")
OUT_THRESH  = os.path.join(MODEL_DIR, "threshold_msu_power.json")
OUT_RESULTS = os.path.join(MODEL_DIR, "msu_power_results.json")

TIMESTEPS   = 10      # 10 samples = 83ms @ 120Hz
PATIENCE    = 10
MAX_EPOCHS  = 60
SEED        = 42
NROWS_DIAG  = 500     # sadece diagnostik icin

# Saldiri senaryolari (README Table III)
ATTACK_SCENARIOS = {
    7,8,9,10,11,12,         # Data injection (SLG fault replay)
    15,16,17,18,19,20,      # Remote tripping command injection
    21,22,23,24,25,26,27,28,29,30,  # Relay setting change (1 relay)
    35,36,37,38,39,40       # Relay setting change (2 relays)
}
NORMAL_SCENARIOS = {1,2,3,4,5,6,13,14,41}

np.random.seed(SEED)

# ─────────────────────────────────────────────
# MARKER DECODE: {scenario}{fault_loc:03d}{load_con:03d}
def decode_marker(marker_val):
    try:
        s = str(marker_val).strip().lower()
        if s in ('natural', 'normal', 'no event'): return 0, 0, 0
        if s == 'attack': return 7, 0, 0
        m = str(int(float(marker_val)))
        load_con  = int(m[-3:]) if len(m) >= 3 else 0
        fault_loc = int(m[-6:-3]) if len(m) >= 6 else 0
        scenario  = int(m[:-6]) if len(m) > 6 else int(m[:-3]) if len(m) > 3 else int(m)
        return scenario, fault_loc, load_con
    except:
        return -1, 0, 0

def is_attack(marker_val):
    s = str(marker_val).strip().lower()
    if s == 'attack': return 1
    if s in ('natural', 'normal', 'no event'): return 0
    sc, _, _ = decode_marker(marker_val)
    return 1 if sc in ATTACK_SCENARIOS else 0

# ─────────────────────────────────────────────
def detect_sep(path):
    with open(path, 'r', errors='ignore') as f:
        line = f.readline()
    if ';' in line: return ';'
    if '\t' in line: return '\t'
    return ','

def load_file(path, nrows=None):
    sep = detect_sep(path)
    try:
        df = pd.read_csv(path, sep=sep, nrows=nrows,
                         low_memory=False, encoding='utf-8-sig',
                         on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"    ⚠️ Okuma hatasi {os.path.basename(path)}: {e}")
        return None

def make_windows(X, w=10):
    return np.array([X[i:i+w] for i in range(len(X) - w + 1)])

def build_lstm_ae(n_features, timesteps=10):
    inp = Input(shape=(timesteps, n_features))
    x   = LSTM(64, return_sequences=True,  dropout=0.2)(inp)
    x   = LSTM(32, return_sequences=False, dropout=0.2)(x)
    x   = RepeatVector(timesteps)(x)
    x   = LSTM(32, return_sequences=True,  dropout=0.2)(x)
    x   = LSTM(64, return_sequences=True,  dropout=0.2)(x)
    out = TimeDistributed(Dense(n_features))(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def metrics(y_true, y_pred, label=""):
    tn,fp,fn,tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    fpr  = fp/(fp+tn)*100 if (fp+tn)>0 else 0
    rec  = tp/(tp+fn)*100 if (tp+fn)>0 else 0
    prec = tp/(tp+fp)*100 if (tp+fp)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    acc  = (tp+tn)/len(y_true)*100
    return dict(label=label, FPR=fpr, Recall=rec,
                Precision=prec, F1=f1, Accuracy=acc,
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))

# ─────────────────────────────────────────────
print("\n" + "="*65)
print("  MSU Power System — GRAD-RL Pipeline")
print("="*65)

# ── 1. DOSYALARI BUL ─────────────────────────
patterns = [
    os.path.join(DATA_DIR, "data*.csv"),
    os.path.join(DATA_DIR, "data*.xlsx"),
    os.path.join(DATA_DIR, "*.csv"),
    os.path.join(DATA_DIR, "*.xlsx"),
]
files = []
for p in patterns:
    files.extend(glob.glob(p))
files = sorted(set(files))

print(f"\n[1] Bulunan dosyalar: {len(files)}")
for f in files[:5]:
    print(f"    {os.path.basename(f)}")
if len(files) > 5:
    print(f"    ... ({len(files)} toplam)")

if not files:
    print(f"\n❌ Dosya bulunamadi: {DATA_DIR}")
    print("   Dosya adlarini kontrol edin (data1.csv, data1.xlsx, vb.)")
    sys.exit(1)

# ── 2. TANI: İlk dosyayı incele ──────────────
print(f"\n[2] Diagnostik (ilk dosya)...")
sample_path = files[0]

# xlsx ise farklı oku
if sample_path.endswith('.xlsx'):
    df_sample = pd.read_excel(sample_path, nrows=NROWS_DIAG)
else:
    df_sample = load_file(sample_path, nrows=NROWS_DIAG)

if df_sample is None or df_sample.empty:
    print("❌ İlk dosya okunamadi!")
    sys.exit(1)

print(f"    Dosya: {os.path.basename(sample_path)}")
print(f"    Shape (ilk {NROWS_DIAG} satir): {df_sample.shape}")
print(f"    Sutunlar (ilk 10): {list(df_sample.columns[:10])}")
print(f"    Sutunlar (son 5):  {list(df_sample.columns[-5:])}")

# Marker/label sutunu bul
marker_col = None
for cand in ['marker', 'Marker', 'MARKER', df_sample.columns[-1]]:
    if cand in df_sample.columns:
        marker_col = cand
        break

print(f"    Marker sutunu: {marker_col}")
if marker_col:
    sample_vals = df_sample[marker_col].dropna().unique()[:10]
    print(f"    Ornek marker degerleri: {sample_vals}")
    # Decode etmeyi dene
    for v in sample_vals[:3]:
        s, fl, lc = decode_marker(v)
        print(f"      marker={v} → scenario={s}, fault_loc={fl}, load_con={lc}, attack={s in ATTACK_SCENARIOS}")

# PMU sensor sutunlari belirle
skip_cols = {'Date','Timestamp','date','timestamp','time',
             'control_panel_log1','control_panel_log2',
             'control_panel_log3','control_panel_log4',
             'relay1_log','relay2_log','relay3_log','relay4_log',
             'snort_log1','snort_log2','snort_log3','snort_log4',
             'marker','Marker','MARKER','fault_loc','load_con'}
# R#- ile başlayan tüm sütunlar PMU
pmu_cols = [c for c in df_sample.columns
            if c not in skip_cols
            and (c.startswith('R1') or c.startswith('R2') or
                 c.startswith('R3') or c.startswith('R4'))]

# Eğer R#- pattern yoksa, numeric olmayan ve bilinen skip dışındakileri al
if not pmu_cols:
    pmu_cols = [c for c in df_sample.columns
                if c not in skip_cols
                and df_sample[c].dtype in [np.float64, np.float32,
                                            np.int64, np.int32]]

print(f"\n    PMU sensor sutunlari: {len(pmu_cols)}")
if pmu_cols:
    print(f"    Ornek: {pmu_cols[:5]} ... {pmu_cols[-3:]}")

# ── 3. TÜM DOSYAYI YÜKLe ────────────────────
print(f"\n[3] Tum dosyalar yukleniyor...")
all_dfs = []
for fp in files:
    try:
        if fp.endswith('.xlsx'):
            df_tmp = pd.read_excel(fp)
        else:
            df_tmp = load_file(fp)
        if df_tmp is not None and not df_tmp.empty:
            all_dfs.append(df_tmp)
            print(f"    ✅ {os.path.basename(fp)}: {df_tmp.shape}")
        else:
            print(f"    ⚠️  {os.path.basename(fp)}: bos")
    except Exception as e:
        print(f"    ❌ {os.path.basename(fp)}: {e}")

if not all_dfs:
    print("❌ Hic dosya yuklenemedi!")
    sys.exit(1)

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"\n    TOPLAM: {df_all.shape[0]:,} satir × {df_all.shape[1]} sutun")

# ── 4. LABEL OLUSTUR ─────────────────────────
print(f"\n[4] Label olusturuluyor...")

if marker_col and marker_col in df_all.columns:
    df_all['_attack'] = df_all[marker_col].apply(is_attack)
    n_normal = (df_all['_attack'] == 0).sum()
    n_attack = (df_all['_attack'] == 1).sum()
    print(f"    Normal: {n_normal:,} | Attack: {n_attack:,} ({n_attack/(n_normal+n_attack)*100:.1f}%)")

    # Senaryo dağılımı
    df_all['_scenario'] = df_all[marker_col].apply(lambda x: decode_marker(x)[0])
    print(f"    Benzersiz senaryolar: {sorted(df_all['_scenario'].unique())}")
else:
    print("    ⚠️  Marker sutunu bulunamadi — son sutunu label olarak deniyoruz")
    df_all['_attack'] = 0  # fallback
    n_normal = len(df_all)
    n_attack = 0

# ── 5. SENSOR FEATURE HAZIRLA ────────────────
# pmu_cols'u df_all uzerinde yeniden belirle
pmu_cols = [c for c in pmu_cols if c in df_all.columns]
if not pmu_cols:
    pmu_cols = [c for c in df_all.columns
                if c not in skip_cols | {'_attack', '_scenario'}
                and df_all[c].dtype in [np.float64, np.float32,
                                         np.int64, np.int32]]

print(f"\n[5] PMU sensor sutunlari: {len(pmu_cols)}")

df_feat   = df_all[pmu_cols + ['_attack']].copy()
# Infinity ve cok buyuk deger temizle (PMU artifact)
df_feat[pmu_cols] = df_feat[pmu_cols].replace([np.inf, -np.inf], np.nan)
for col in pmu_cols:
    col_mean = df_feat[col].mean()
    col_std  = df_feat[col].std()
    if col_std > 0:
        df_feat[col] = df_feat[col].clip(col_mean - 10*col_std, col_mean + 10*col_std)
df_feat   = df_feat.ffill().bfill().fillna(0)
df_feat[pmu_cols] = df_feat[pmu_cols].replace([np.inf, -np.inf], 0)
inf_count = np.isinf(df_feat[pmu_cols].values).sum()
print(f"    Infinity temizlendi. Kalan inf: {inf_count}")
n_features = len(pmu_cols)

# ── 6. TRAIN/TEST AYIR ───────────────────────
# Normal = attack=0, Test = tümü (attack label ile)
df_normal = df_feat[df_feat['_attack'] == 0][pmu_cols]
print(f"\n[6] Normal (egitim): {len(df_normal):,} satir")
print(f"    Test (hepsi):    {len(df_feat):,} satir")

# Scaler
scaler   = MinMaxScaler()
X_normal = scaler.fit_transform(df_normal.values)
joblib.dump(scaler, OUT_SCALER)

split    = int(len(X_normal) * 0.8)
X_tr     = X_normal[:split]
X_val    = X_normal[split:]
wins_tr  = make_windows(X_tr,  TIMESTEPS)
wins_val = make_windows(X_val, TIMESTEPS)
print(f"    Train windows: {len(wins_tr):,} | Val windows: {len(wins_val):,}")

# ── 7. LSTM-AE ───────────────────────────────
print(f"\n[7] LSTM-AE egitiliyor ({n_features} feature)...")

model = build_lstm_ae(n_features, TIMESTEPS)
cbs = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE,
                  min_delta=1e-6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=4, min_lr=1e-5, verbose=0),
]
history = model.fit(
    wins_tr, wins_tr,
    validation_data=(wins_val, wins_val),
    epochs=MAX_EPOCHS, batch_size=128,
    callbacks=cbs, verbose=1,
)

final_tr  = history.history['loss'][-1]
final_val = history.history['val_loss'][-1]
gap       = final_val / final_tr if final_tr > 0 else 999
best_ep   = int(np.argmin(history.history['val_loss'])) + 1

print(f"\n    Best epoch: {best_ep}")
print(f"    Train loss: {final_tr:.6f}")
print(f"    Val loss:   {final_val:.6f}")
print(f"    Gap:        {gap:.2f}x")

# ── 8. THRESHOLD ─────────────────────────────
preds_val = model.predict(wins_val, batch_size=256, verbose=0)
mse_val   = np.mean((wins_val[:,-1,:] - preds_val[:,-1,:])**2, axis=1)
tau_off   = float(np.percentile(mse_val, 99))
tau_live  = round(tau_off * 1.2, 8)

print(f"\n[8] tau_offline={tau_off:.6f}  tau_live={tau_live:.6f}")
model.save(OUT_MODEL)
with open(OUT_THRESH, 'w') as f:
    json.dump({'threshold': tau_live, 'tau_offline': tau_off,
               'dataset': 'MSU_PowerSystem',
               'train_val_gap': round(gap, 3),
               'best_epoch': best_ep}, f, indent=2)

# ── 9. EVALUATE ──────────────────────────────
print(f"\n[9] Test degerlendirmesi...")
X_test  = scaler.transform(df_feat[pmu_cols].values)
wins_ts = make_windows(X_test, TIMESTEPS)
y_full  = df_feat['_attack'].values[TIMESTEPS-1:]
y_true  = (y_full != 0).astype(int)

preds_ts = model.predict(wins_ts, batch_size=256, verbose=1)
mse_ts   = np.mean((wins_ts[:,-1,:] - preds_ts[:,-1,:])**2, axis=1)
y_pred   = (mse_ts > tau_live).astype(int)

m = metrics(y_true, y_pred, "MSU Power")

print("\n" + "="*65)
print("  MSU POWER SYSTEM SONUCLARI")
print("="*65)
print(f"  Accuracy:  {m['Accuracy']:.2f}%")
print(f"  FPR:       {m['FPR']:.2f}%")
print(f"  Recall:    {m['Recall']:.2f}%")
print(f"  Precision: {m['Precision']:.2f}%")
print(f"  F1:        {m['F1']:.2f}%")
print(f"  FN:        {m['FN']}  ({'ZERO ✅' if m['FN']==0 else 'var'})")
print(f"  TP:{m['TP']} | FP:{m['FP']} | TN:{m['TN']}")

# Attack type breakdown
if marker_col:
    print(f"\n  Senaryo tipi breakdown:")
    for attack_type, scenarios in [
        ("Data Injection (SLG replay)", {7,8,9,10,11,12}),
        ("Command Injection",           {15,16,17,18,19,20}),
        ("Relay Setting Change",        {21,22,23,24,25,26,27,28,29,30,35,36,37,38,39,40}),
    ]:
        mask = df_feat['_scenario'].values[TIMESTEPS-1:].isin(scenarios)
        n_att = mask.sum()
        if n_att > 0:
            det = ((y_pred==1) & mask).sum()
            print(f"    {attack_type}: {n_att:,} → {det:,} tespit ({det/n_att*100:.1f}%)")

results = {
    'model': 'LSTM-AE', 'dataset': 'MSU_PowerSystem',
    'n_sensors': n_features,
    'train_val_gap': round(gap, 3),
    'best_epoch': best_ep,
    'tau_live': tau_live,
    **{k: round(float(v),4) if isinstance(v,float) else v
       for k,v in m.items() if k != 'label'}
}
with open(OUT_RESULTS, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Sonuclar: {OUT_RESULTS}")
print("\n" + "="*65)
print("  PAPER OZET")
print("="*65)
print(f"""
MSU Power System (electric transmission, PMU-based):
  n_features={n_features} | gap={gap:.2f}x | best_epoch={best_ep}
  Acc={m['Accuracy']:.2f}%  FPR={m['FPR']:.2f}%  FN={m['FN']}  F1={m['F1']/100:.3f}
""")