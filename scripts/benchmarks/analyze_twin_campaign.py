"""Analysis of the 2026-11-24 GRFICSv3 lab campaign (lab/out/lab_campaign.csv).

C1c  technique classification on repeated, independently executed episodes
     (1) external validation: classifier trained on the paper's dataset, tested on the new episodes
     (2) leave-one-repetition-out inside the new campaign, three feature sets:
         F13 (paper sensors), F13+temporal (10-s rolling mean/std), F13+temporal+PLC state
         (setpoint, command, HMI and run/manual registers, Modbus read latency)
     (3) detection by the paper's LSTM-AE (tau_live = 1.2 x offline P99) per episode, false alarms on normal
C4a  closed-loop defence: per run and per (technique, policy) physical outcomes
Output: results/lab_c1c_c4a.json and results/lab_c4a_runs.csv
"""
import json, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, xgboost as xgb, joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = r"D:\PhD_Project"; HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = ["f1_valve", "f1_flow", "f2_valve", "f2_flow", "purge_valve", "purge_flow",
        "prod_valve", "prod_flow", "pressure", "level", "A_purge", "B_purge", "C_purge"]
STATE = ["sp_a", "sp_pressure", "sp_override", "sp_level", "cmd_f1", "cmd_f2", "cmd_purge", "cmd_prod",
         "hmi_pressure", "hmi_level", "run_bit", "manual_mode", "latency_ms"]
PARAMS = dict(max_depth=7, learning_rate=0.02, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.5, reg_lambda=2.0,
              min_child_weight=5, gamma=0.1, n_estimators=300, verbosity=0, random_state=42)
res = {}

# rows with a failed Modbus read were written with one extra empty field (49 vs 48); drop it
lab = pd.read_csv(os.path.join(HERE, "lab", "out", "lab_campaign.csv"), engine="python",
                  on_bad_lines=lambda f: f[:47] + [f[-1]] if len(f) == 49 else None)
lab["utc"] = pd.to_datetime(lab.utc)
ok = lab[lab.read_ok == 1].copy()
res["rows"] = {"total": len(lab), "read_failures": int((lab.read_ok == 0).sum()),
               "read_failures_by_segment": {f"{k[0]}|{k[1]}|{k[2]}": int(v) for k, v in
                                            lab[lab.read_ok == 0].groupby(["phase", "technique", "segment"]).size().items()}}

# ------------------------------------------------------------------ C1c
c1 = ok[ok.phase == "c1c"].copy()
c1["rep"] = c1.groupby("technique").episode.rank(method="dense").astype(int) - 1
for f in FEAT:   # temporal features per episode, causal 10-s window
    g = c1.groupby("episode")[f]
    c1[f + "_m10"] = g.transform(lambda s: s.rolling(10, min_periods=1).mean())
    c1[f + "_s10"] = g.transform(lambda s: s.rolling(10, min_periods=2).std().fillna(0))
TEMP = [f + s for f in FEAT for s in ("_m10", "_s10")]
att = c1[c1.segment == "attack"].copy()
classes = sorted(att.technique.unique()); cid = {c: i for i, c in enumerate(classes)}
att["y"] = att.technique.map(cid)
res["c1c_episodes"] = att.groupby("technique").episode.nunique().to_dict()

# (1) external validation
orig = pd.read_csv(os.path.join(ROOT, "data", "grad_rl_final_dataset.csv"), low_memory=False, dtype={"mitre_technique": str})
osc = MinMaxScaler().fit(orig.loc[orig.label == "Normal", FEAT].values.astype(np.float32))
oa = orig[orig.label != "Normal"].copy(); oa["t"] = oa.mitre_technique.str.strip()
oa = oa[oa.t.isin(classes)]
clf = xgb.XGBClassifier(**PARAMS).fit(osc.transform(oa[FEAT].values.astype(np.float32)), oa.t.map(cid).values,
                                      sample_weight=np.clip(compute_sample_weight("balanced", oa.t.map(cid).values), 1, None))
yp = clf.predict(osc.transform(att[FEAT].values.astype(np.float32)))
ext = {"row_macro_f1": round(f1_score(att.y, yp, average="macro"), 4), "per_technique": {}}
for t in classes:
    m = (att.technique == t).values
    ext["per_technique"][t] = {"row_recall": round(float((yp[m] == cid[t]).mean()), 4),
                               "predicted_as": pd.Series([classes[v] for v in yp[m]]).value_counts().to_dict()}
ep_major = att.assign(p=yp).groupby("episode").apply(lambda g: g.p.mode()[0] == g.y.iloc[0])
ext["episode_majority_correct"] = f"{int(ep_major.sum())}/{len(ep_major)}"
res["c1c_external_validation"] = ext

# (2) leave-one-repetition-out
def lor(cols):
    yp = np.zeros(len(att), int); out = {}
    for r in sorted(att.rep.unique()):
        te = (att.rep == r).values
        sc = MinMaxScaler().fit(att.loc[~te, cols].values)
        m = xgb.XGBClassifier(**PARAMS).fit(sc.transform(att.loc[~te, cols].values), att.y[~te],
                                            sample_weight=np.clip(compute_sample_weight("balanced", att.y[~te]), 1, None))
        yp[te] = m.predict(sc.transform(att.loc[te, cols].values))
    out["row_macro_f1"] = round(f1_score(att.y, yp, average="macro"), 4)
    out["per_technique_row_recall"] = {t: round(float((yp[(att.technique == t).values] == cid[t]).mean()), 4) for t in classes}
    maj = att.assign(p=yp).groupby("episode").apply(lambda g: g.p.mode()[0] == g.y.iloc[0])
    out["episode_majority_correct"] = f"{int(maj.sum())}/{len(maj)}"
    return out
res["c1c_leave_one_repetition_out"] = {"F13": lor(FEAT), "F13+temporal": lor(FEAT + TEMP),
                                       "F13+temporal+PLC_state": lor(FEAT + TEMP + STATE)}

# (3) detection with the paper's LSTM-AE
from tensorflow.keras.models import load_model
det = load_model(os.path.join(ROOT, "models", "detective_lstm_grfics_live.keras"))
lsc = joblib.load(os.path.join(ROOT, "models", "scaler_grfics_live.pkl"))
tau = 1.2 * json.load(open(os.path.join(ROOT, "models", "threshold_grfics_live.json")))["threshold"]
flags = []
for ep, g in ok.groupby(["phase", "episode"], sort=False):
    X = lsc.transform(g[FEAT].values.astype(np.float32)).astype(np.float32)
    if len(X) < 10:
        flags.append(pd.Series(0, index=g.index)); continue
    W = np.stack([X[i:i + 10] for i in range(len(X) - 9)])
    mse = ((det.predict(W, batch_size=1024, verbose=0) - W) ** 2).mean((1, 2))
    flags.append(pd.Series(np.r_[np.zeros(9, int), (mse > tau).astype(int)], index=g.index))
ok["flag"] = pd.concat(flags)
d = {"tau": tau, "per_technique": {}}
c1f = ok[ok.phase == "c1c"]
d["false_alarm_rate_normal_segments"] = round(float(c1f[c1f.segment == "normal"].flag.mean()), 4)
for t in classes:
    eps = c1f[(c1f.technique == t) & (c1f.segment == "attack")].groupby("episode")
    hit = eps.flag.max(); delay = eps.apply(lambda g: g.t_seg[g.flag == 1].min())
    d["per_technique"][t] = {"episodes_detected": f"{int(hit.sum())}/{len(hit)}",
                             "median_delay_s": None if delay.isna().all() else float(delay.median()),
                             "flagged_row_share": round(float(eps.flag.mean().mean()), 4)}
res["c1c_detection_lstm_ae"] = d

# ------------------------------------------------------------------ C4a
# runs during which the OpenPLC runtime crashed and was restarted (lab/out/plc_events.log);
# they were repeated afterwards as "<id>r". Run 1 crashed at the last post-attack second and is kept.
CRASHED = {"9", "15", "18", "23", "28", "35", "38"}
ok["episode"] = ok.episode.astype(str)
res["c4a_plc_crashes"] = {"runs_with_crash": sorted(CRASHED, key=int) + ["1 (at end of post segment)"],
                          "events": open(os.path.join(HERE, "lab", "out", "plc_events.log")).read().splitlines()}
c4 = ok[(ok.phase == "c4a") & ok.segment.isin(["lead", "attack", "post"]) & ~ok.episode.isin(CRASHED)]
runs = []
for ep, g in c4.groupby("episode"):
    a = g[g.segment.isin(["attack", "post"])]
    acts = pd.to_numeric(g.action, errors="coerce").fillna(0).astype(int)
    rec = ok[(ok.phase == "c4a") & (ok.episode == ep) & (ok.segment == "recovery")]
    runs.append({"run": ep, "technique": g.technique.iloc[0], "policy": g.policy.iloc[0],
                 "max_pressure_kPa": round(a.pressure.max(), 1),
                 "s_above_2900": int((a.pressure > 2900).sum()), "s_above_3000": int((a.pressure > 3000).sum()),
                 "mean_product_flow": round(a.prod_flow.mean(), 1),
                 "level_min": round(a.level.min(), 1), "level_max": round(a.level.max(), 1),
                 "valve_share_normal_lead": round(float((acts[g.segment == "lead"] == 1).mean()), 3),
                 "valve_share_attack_post": round(float((acts[g.segment.isin(["attack", "post"])] == 1).mean()), 3),
                 "shutdown_share_attack_post": round(float((acts[g.segment.isin(["attack", "post"])] == 2).mean()), 3),
                 "recovery_s": int(len(rec))})
runs = pd.DataFrame(runs)
runs.to_csv(os.path.join(HERE, "results", "lab_c4a_runs.csv"), index=False)
agg = runs.groupby(["technique", "policy"]).mean(numeric_only=True).drop(columns="run", errors="ignore").round(2)
res["c4a_by_technique_policy"] = {f"{t} | {p}": r.to_dict() for (t, p), r in agg.iterrows()}
res["c4a_by_policy"] = runs.groupby("policy").mean(numeric_only=True).drop(columns="run", errors="ignore").round(2).to_dict("index")

json.dump(res, open(os.path.join(HERE, "results", "lab_c1c_c4a.json"), "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str)[:6000])
