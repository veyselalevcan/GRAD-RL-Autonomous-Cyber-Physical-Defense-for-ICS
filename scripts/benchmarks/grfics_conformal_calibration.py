"""C1c detection on the fresh GRFICSv3 instance with the paper's LSTM-AE.

The deployed threshold (tau_live = 1.2 x offline P99) flags 90 % of normal rows on the new
instance. Here the threshold is recalibrated automatically (split conformal, alpha = 0.01)
on the first minutes of normal operation of the new instance: the settle segment and the
normal lead-in of the first two C1c episodes. Evaluation on the remaining 28 episodes:
false alarms on their normal lead-ins, detection and delay per technique.
Output: results/c1c_detection_conformal.json
"""
import json, math, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, joblib
from tensorflow.keras.models import load_model

ROOT = r"D:\PhD_Project"; HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = ["f1_valve", "f1_flow", "f2_valve", "f2_flow", "purge_valve", "purge_flow",
        "prod_valve", "prod_flow", "pressure", "level", "A_purge", "B_purge", "C_purge"]
lab = pd.read_csv(os.path.join(HERE, "lab", "out", "lab_campaign.csv"), engine="python",
                  on_bad_lines=lambda f: f[:47] + [f[-1]] if len(f) == 49 else None)
lab = lab[(lab.read_ok == 1) & lab.phase.isin(["settle", "c1c"])].copy()
det = load_model(os.path.join(ROOT, "models", "detective_lstm_grfics_live.keras"))
sc = joblib.load(os.path.join(ROOT, "models", "scaler_grfics_live.pkl"))
tau_live = 1.2 * json.load(open(os.path.join(ROOT, "models", "threshold_grfics_live.json")))["threshold"]

parts = []
for (ph, ep), g in lab.groupby(["phase", "episode"], sort=False):
    X = sc.transform(g[FEAT].values.astype(np.float32)).astype(np.float32)
    mse = np.full(len(g), np.nan)
    if len(X) >= 10:
        W = np.stack([X[i:i + 10] for i in range(len(X) - 9)])
        mse[9:] = ((det.predict(W, batch_size=1024, verbose=0) - W) ** 2).mean((1, 2))
    parts.append(g.assign(mse=mse))
d = pd.concat(parts)
d["episode"] = pd.to_numeric(d.episode)
first_eps = sorted(d[d.phase == "c1c"].episode.unique())[:2]
cal = d[((d.phase == "settle") | ((d.phase == "c1c") & d.episode.isin(first_eps) & (d.segment == "normal"))) & d.mse.notna()].mse.values
n = len(cal); tau_c = float(np.sort(cal)[min(math.ceil((n + 1) * 0.99), n) - 1])
ev = d[(d.phase == "c1c") & ~d.episode.isin(first_eps) & d.mse.notna()]

out = {"calibration_rows": n, "calibration_minutes": round(n / 60, 1), "tau_live": tau_live, "tau_conformal": tau_c,
       "evaluated_episodes": int(ev.episode.nunique())}
for name, tau in (("tau_live (paper)", tau_live), ("conformal on new instance", tau_c)):
    r = {"false_alarm_rate_normal": round(float((ev[ev.segment == "normal"].mse > tau).mean()), 4), "per_technique": {}}
    for t, g in ev[ev.segment == "attack"].groupby("technique"):
        per = g.groupby("episode").apply(lambda e: e.t_seg[e.mse > tau].min())
        r["per_technique"][t] = {"episodes_detected": f"{int(per.notna().sum())}/{len(per)}",
                                 "median_delay_s": None if per.isna().all() else float(per.median()),
                                 "flagged_row_share": round(float((g.mse > tau).mean()), 3)}
    out[name] = r
json.dump(out, open(os.path.join(HERE, "results", "c1c_detection_conformal.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
