"""C5 (Reviewer 2): automatic threshold calibration instead of hand-set thresholds.

Split-conformal threshold: for calibration scores s_1..s_n from normal operation
at the deployment site and miss-coverage alpha, tau = s_(k), k = ceil((n+1)(1-alpha)).
Under exchangeability the false-alarm probability per window is <= alpha.

Part 1, WADI (model and data handling of the paper: models/detective_lstm_wadi.keras,
first 100,000 rows of the attack-period file, 93,365 normal / 6,635 attack):
  P    paper threshold: P99 of training-normal errors x 1.2
  CG   conformal, global score, calibrated on the first hour of the attack-period
       file (commissioning baseline, before the first attack)
  CM   Mondrian conformal per WADI zone (1_, 2_, 3_ tags), alpha/3 per zone
  CG+k CG with persistence: alarm if >= 3 of the last 5 windows exceed tau
Evaluation excludes the calibration hour. Event level: an attack segment counts
as detected if any window inside it raises an alarm.

Part 2, GRFICSv3 live session (telemetry_audit.csv MSE stream):
  live threshold 0.5 used in the paper vs a conformal threshold calibrated on
  the quiet minutes before the first scenario (00:13:30-00:16:30 log clock).
Output: results/c5_conformal_threshold.json
"""
import json, os, math
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, joblib
from tensorflow.keras.models import load_model

ROOT = r"D:\PhD_Project"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ALPHA = 0.01
TS = 10


def conformal_tau(s, alpha):
    s = np.sort(np.asarray(s)); n = len(s); k = math.ceil((n + 1) * (1 - alpha))
    return float(s[min(k, n) - 1])


def load_numeric(path, nrows, n_feat, label=False):
    df = pd.read_csv(path, nrows=nrows, skiprows=1, low_memory=False)
    df.columns = df.columns.str.strip(); y = None
    if label:
        lc = next((c for c in df.columns if "label" in c.lower() or "lable" in c.lower()), df.columns[-1])
        y = np.where(df[lc].astype(str).str.strip().isin(["1", "1.0"]), 0, 1); df = df.drop(columns=[lc])
    df = df.drop(columns=[c for c in df.columns if any(k in c.lower() for k in ("timestamp", " time", "date"))], errors="ignore")
    df = df.dropna(axis=1, how="all").ffill().bfill().fillna(0).select_dtypes(include=[np.number])
    v = df.values[:, :n_feat] if df.shape[1] >= n_feat else np.hstack([df.values, np.zeros((len(df), n_feat - df.shape[1]))])
    return v, y


def feat_err(model, X, bs=512):
    out = []
    for i in range(0, len(X) - TS + 1, 20000):
        j = min(i + 20000, len(X) - TS + 1)
        W = np.stack([X[k:k + TS] for k in range(i, j)]).astype(np.float32)
        P = model.predict(W, batch_size=bs, verbose=0)
        out.append(((W[:, -1, :] - P[:, -1, :]) ** 2).astype(np.float32))
    return np.vstack(out)


def metrics(y, a, seg):
    tn = int(((a == 0) & (y == 0)).sum()); fp = int(((a == 1) & (y == 0)).sum())
    tp = int(((a == 1) & (y == 1)).sum()); fn = int(((a == 0) & (y == 1)).sum())
    det = sum(int(a[s:e].any()) for s, e in seg)
    return {"FPR_%": round(100 * fp / max(fp + tn, 1), 2), "recall_%": round(100 * tp / max(tp + fn, 1), 2),
            "precision_%": round(100 * tp / max(tp + fp, 1), 2), "attacks_detected": f"{det}/{len(seg)}",
            "false_alarm_windows_per_h": round(fp / max((fp + tn) / 3600, 1e-9), 1)}


def segments(y):
    seg, s = [], None
    for i, v in enumerate(y):
        if v and s is None: s = i
        if not v and s is not None: seg.append((s, i)); s = None
    if s is not None: seg.append((s, len(y)))
    return seg


res = {"alpha": ALPHA}
# ------------------------------ WADI ------------------------------
scaler = joblib.load(os.path.join(ROOT, "models", "scaler_wadi.pkl")); nf = scaler.n_features_in_
model = load_model(os.path.join(ROOT, "models", "detective_lstm_wadi.keras"))
tau_off = json.load(open(os.path.join(ROOT, "models", "threshold_wadi.json")))["threshold"]
hdr = [c.strip() for c in pd.read_csv(os.path.join(ROOT, "data", "WADI_14days_new.csv"), nrows=0).columns]
names = [c for c in hdr if not any(k in c.lower() for k in ("timestamp", " time", "date", "label", "lable"))][:nf]
zones = {z: np.array([n.startswith(p) for n in names]) for z, p in (("Z1", "1_"), ("Z2", "2_"), ("Z3", "3_"))}
CACHE = os.path.join(OUT, "wadi_feat_err_cache.npz")
if os.path.exists(CACHE):
    z = np.load(CACHE); E, y = z["E"], z["y"]
else:
    Xa, ya = load_numeric(os.path.join(ROOT, "data", "WADI_attackdataLABLE.csv"), 100_000, nf, label=True)
    E = feat_err(model, scaler.transform(Xa)); y = ya[TS - 1:]
    np.savez_compressed(CACHE, E=E, y=y)
g = E.mean(1); zs = {z: E[:, m].mean(1) for z, m in zones.items() if m.any()}
first_attack = int(np.argmax(y == 1))
CAL = min(3600, first_attack - 60)
ev = slice(CAL, None); yv = y[ev]; seg = segments(yv)
res["wadi"] = {"rows": int(len(y)), "first_attack_window": first_attack, "calibration_windows": CAL,
               "zone_sizes": {z: int(m.sum()) for z, m in zones.items()}, "n_attack_segments_eval": len(seg)}
tauP = tau_off * 1.2
tauG = conformal_tau(g[:CAL], ALPHA)
tauZ = {z: conformal_tau(s[:CAL], ALPHA / len(zs)) for z, s in zs.items()}
aP = (g > tauP).astype(int); aG = (g > tauG).astype(int)
aZ = np.zeros_like(aG)
for z, s in zs.items(): aZ |= (s > tauZ[z]).astype(int)
def persist(a, k=3, n=5): return (np.convolve(a, np.ones(n, int), "full")[: len(a)] >= k).astype(int)
res["wadi"]["thresholds"] = {"P_paper": tauP, "CG": tauG, "CM": tauZ}
res["wadi"]["P_paper"] = metrics(yv, aP[ev], seg)
res["wadi"]["CG_conformal_global"] = metrics(yv, aG[ev], seg)
res["wadi"]["CM_mondrian_zone"] = metrics(yv, aZ[ev], seg)
res["wadi"]["CG_persist_3of5"] = metrics(yv, persist(aG)[ev], seg)
res["wadi"]["CM_persist_3of5"] = metrics(yv, persist(aZ)[ev], seg)
# drift check: false-alarm rate of CG over time (6 blocks of normal windows)
nrm = np.where(yv == 0)[0]; blocks = np.array_split(nrm, 6)
res["wadi"]["CG_FPR_by_time_block_%"] = [round(100 * float(aG[ev][b].mean()), 2) for b in blocks]
# rolling recalibration with operator feedback: every R windows the threshold is
# recomputed on the windows of the previous hour that the operator confirmed as
# normal (labels used only for past, already reviewed windows)
for R in (3600, 900):
    aR = np.zeros_like(aG); tau_t = tauG; taus = []
    for i in range(CAL, len(g)):
        if (i - CAL) % R == 0 and i > CAL:
            past = np.arange(max(0, i - 3600), i); past = past[y[past] == 0]
            if len(past) > 300: tau_t = conformal_tau(g[past], ALPHA)
            taus.append(tau_t)
        aR[i] = int(g[i] > tau_t)
    res["wadi"][f"CG_rolling_recal_every_{R}s"] = dict(metrics(yv, aR[ev], seg),
        tau_range=[round(min(taus), 5), round(max(taus), 5)] if taus else None)

# ------------------------------ GRFICSv3 live ------------------------------
t = pd.read_csv(os.path.join(ROOT, "grfics attack and normal data", "audit_logs", "telemetry_audit.csv"), parse_dates=["Timestamp"])
cal = t[(t.Timestamp >= "2026-05-17 00:13:30") & (t.Timestamp < "2026-05-17 00:16:30")].MSE.values
tauL = conformal_tau(cal, ALPHA)
OFF = pd.Timedelta(hours=3)
J = [tuple(s.strip() for s in l.split("|")) for l in open(os.path.join(ROOT, "grfics attack and normal data", "audit_logs", "experiment_journal.txt")) if l.strip()]
win = []
for k, (kind, name, ts) in enumerate(J):
    if kind != "ATTACK_START": continue
    st = pd.Timestamp(ts) + OFF
    en = next((pd.Timestamp(x[2]) + OFF for x in J[k + 1:] if x[0] == "ATTACK_END" and x[1] == name),
              next((pd.Timestamp(x[2]) + OFF for x in J[k + 1:] if x[0] == "ATTACK_START"), st + pd.Timedelta("120s")))
    win.append((name, st, en))
quiet = t[t.Timestamp >= "2026-05-17 01:53:00"]  # after the last scenario ended (01:51:12) + margin
live = {"calibration_samples": int(len(cal)), "calibration_mse_median": float(np.median(cal)),
        "tau_conformal": tauL, "tau_paper": 0.5, "tau_offline_P99": 0.0027309285942465067, "scenarios": []}
for tau, key in ((0.5, "paper"), (tauL, "conformal")):
    live[f"quiet_tail_false_alarm_%_{key}"] = round(100 * float((quiet.MSE > tau).mean()), 2)
for name, st, en in win:
    w = t[(t.Timestamp >= st) & (t.Timestamp <= en)]
    live["scenarios"].append({"scenario": name, "n": int(len(w)),
                              "alarm_%_paper": round(100 * float((w.MSE > 0.5).mean()), 1) if len(w) else None,
                              "alarm_%_conformal": round(100 * float((w.MSE > tauL).mean()), 1) if len(w) else None})
live["scenarios_detected_paper"] = sum(1 for s in live["scenarios"] if s["alarm_%_paper"])
live["scenarios_detected_conformal"] = sum(1 for s in live["scenarios"] if s["alarm_%_conformal"])
live["quiet_tail_samples"] = int(len(quiet))
res["grfics_live"] = live
json.dump(res, open(os.path.join(OUT, "c5_conformal_threshold.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
