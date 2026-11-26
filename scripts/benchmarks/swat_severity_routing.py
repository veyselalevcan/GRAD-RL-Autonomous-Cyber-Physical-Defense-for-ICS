"""C7 (Reviewer 2): SWaT Attack6 -> Attack4 confusion and the resulting
autonomous action on a high-severity attack.

Data: data/SWaT_Dataset_Attack_v0.csv (4,237 attack-phase rows, labels Attack1-6),
scaler fitted on SWaT normal data (models/scaler_swat.pkl, as in the paper).

1. classifier: paper setting (70/30 class-wise, early stopping on test) and the
   corrected setting (60/10/30, early stopping on validation);
2. Mode-1 action mapping of the paper (Attack1/5 -> monitor, Attack4 -> valve,
   Attack2/3/6 -> approval) and two fixes:
   F1 probability escalation: approval if P(high-severity classes) >= q
      (q chosen on the validation block: smallest q with no high-severity
       sample on the valve path; sweep reported);
   F2 incident-level maximum severity: rows are grouped into incidents
      (gap > 60 s starts a new incident) and the action never drops below the
      highest severity already seen in the incident;
3. leave-one-episode-out for classes with more than one episode.
Output: results/c7_swat_escalation.json
"""
import json, os
import joblib, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

ROOT = r"D:\PhD_Project"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEED = 42
d = pd.read_csv(os.path.join(ROOT, "data", "SWaT_Dataset_Attack_v0.csv"))
d.columns = d.columns.str.strip()
d["ts"] = pd.to_datetime(d["Timestamp"].str.strip(), format="%d/%m/%Y %I:%M:%S %p")
d = d.sort_values("ts").reset_index(drop=True)
scaler = joblib.load(os.path.join(ROOT, "models", "scaler_swat.pkl"))
feat = [c for c in d.columns if c not in ("Timestamp", "Label", "ts")]
X = scaler.transform(d[feat].values.astype(np.float32))
CL = sorted(d.Label.unique()); y = d.Label.map({c: i for i, c in enumerate(CL)}).values
SEV = {"Attack1": 6.5, "Attack5": 6.5, "Attack4": 7.5, "Attack2": 9.5, "Attack3": 9.5, "Attack6": 9.5}
sev = np.array([SEV[c] for c in CL]); HIGH = sev >= 9.0
def action(s): return np.where(s >= 9.0, "APPROVAL", np.where(s >= 7.0, "VALVE", "MONITOR"))
PAR = dict(max_depth=7, learning_rate=0.02, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.5,
           reg_lambda=2.0, min_child_weight=5, gamma=0.1, eval_metric="mlogloss", verbosity=0,
           random_state=SEED, n_estimators=1000)
d["ep"] = d.groupby("Label").ts.transform(lambda s: (s.diff().dt.total_seconds() > 60).cumsum())
d["episode"] = d.Label + "#" + d.ep.astype(str)


def fit(tr, va):
    present = np.unique(y[tr]); m = {c: i for i, c in enumerate(present)}
    clf = xgb.XGBClassifier(**PAR, callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)])
    keep = np.isin(y[va], present)
    clf.fit(X[tr], [m[v] for v in y[tr]], sample_weight=np.clip(compute_sample_weight("balanced", y[tr]), 1, None),
            eval_set=[(X[va][keep], [m[v] for v in y[va][keep]])], verbose=False)
    P = np.zeros((len(X), len(CL))); P[:, present] = clf.predict_proba(X)
    return clf, P


def safety(idx, act):
    yt = y[idx]
    return {"high_sev_on_valve": int(((act == "VALVE") & HIGH[yt]).sum()),
            "high_sev_on_monitor": int(((act == "MONITOR") & HIGH[yt]).sum()),
            "low_sev_escalated_to_approval": int(((act == "APPROVAL") & ~HIGH[yt]).sum()),
            "approval_share": round(float((act == "APPROVAL").mean()), 4), "n": int(len(idx))}


def incident_max(idx, s):
    out = s.copy(); t = d.ts.values[idx]; order = np.argsort(t, kind="stable")
    run, last = 0.0, None
    for j in order:
        if last is None or (t[j] - last) / np.timedelta64(1, "s") > 60: run = 0.0
        run = max(run, s[j]); out[j] = run; last = t[j]
    return out


res = {"classes": CL, "severity": SEV,
       "episodes": d.groupby("episode").agg(n=("Label", "size"), start=("ts", "min"), end=("ts", "max")).astype(str).reset_index().to_dict("records")}
tr, va, te, tr70 = [], [], [], []
for c in range(len(CL)):
    i = np.where(y == c)[0]; n = len(i)
    tr += list(i[: int(.6 * n)]); va += list(i[int(.6 * n): int(.7 * n)]); tr70 += list(i[: int(.7 * n)]); te += list(i[int(.7 * n):])
tr, va, te, tr70 = map(np.array, (tr, va, te, tr70))

for name, (a, b) in {"paper_es_on_test": (tr70, te), "es_on_validation": (tr, va)}.items():
    clf, P = fit(a, b)
    yp = P[te].argmax(1)
    cm = confusion_matrix(y[te], yp, labels=range(len(CL)))
    r = {"macro_f1": round(f1_score(y[te], yp, average="macro"), 4), "best_iteration": int(clf.best_iteration),
         "confusion": {CL[i]: {CL[j]: int(cm[i, j]) for j in range(len(CL)) if cm[i, j]} for i in range(len(CL))},
         "attack6_as_attack4": int(cm[CL.index("Attack6"), CL.index("Attack4")])}
    s0 = sev[yp]
    r["F0_paper_mapping"] = safety(te, action(s0))
    if name == "es_on_validation":
        ph_va, ph_te = P[va][:, HIGH].sum(1), P[te][:, HIGH].sum(1)
        sweep = {}
        for q in [0.05, 0.1, 0.2, 0.3, 0.5]:
            s1 = np.where(ph_te >= q, np.maximum(s0, 9.5), s0)
            sweep[str(q)] = safety(te, action(s1))
        cand = [q for q in np.arange(0.01, 0.51, 0.01) if not ((action(np.where(ph_va >= q, 9.5, sev[P[va].argmax(1)])) == "VALVE") & HIGH[y[va]]).any()]
        q_star = float(max(cand)) if cand else 0.01
        s1 = np.where(ph_te >= q_star, np.maximum(s0, 9.5), s0)
        r["F1_prob_escalation"] = dict(safety(te, action(s1)), q_selected_on_validation=round(q_star, 2), sweep=sweep)
        r["F2_incident_max"] = safety(te, action(incident_max(te, s0)))
        r["F1_plus_F2"] = safety(te, action(incident_max(te, s1)))
    res[name] = r

lo = []
for ep, g in d.groupby("episode"):
    c = CL.index(g.Label.iloc[0])
    if d[d.Label == g.Label.iloc[0]].episode.nunique() < 2: continue
    test = np.where(d.episode == ep)[0]; trn = np.where(d.episode != ep)[0]
    a, b = [], []
    for cc in np.unique(y[trn]):
        ii = trn[y[trn] == cc]; k = int(.85 * len(ii)); a += list(ii[:k]); b += list(ii[k:])
    _, P = fit(np.array(a), np.array(b)); yp = P[test].argmax(1)
    lo.append({"held_out": ep, "n": len(test), "recall": round(float((yp == c).mean()), 4),
               "predicted": pd.Series([CL[v] for v in yp]).value_counts().to_dict(),
               "high_sev_on_valve_or_monitor": int((HIGH[c] & (sev[yp] < 9)).sum())})
res["leave_one_episode_out"] = lo
json.dump(res, open(os.path.join(OUT, "c7_swat_escalation.json"), "w"), indent=1, default=str)
for k in ("paper_es_on_test", "es_on_validation"):
    r = res[k]; print(k, r["macro_f1"], r["best_iteration"], "A6->A4", r["attack6_as_attack4"])
    for f in ("F0_paper_mapping", "F1_prob_escalation", "F2_incident_max", "F1_plus_F2"):
        if f in r: print("  ", f, {kk: vv for kk, vv in r[f].items() if kk != "sweep"})
print("sweep", res["es_on_validation"]["F1_prob_escalation"]["sweep"])
print("episodes", len(res["episodes"]))
for x in lo: print("LOEO", x)
