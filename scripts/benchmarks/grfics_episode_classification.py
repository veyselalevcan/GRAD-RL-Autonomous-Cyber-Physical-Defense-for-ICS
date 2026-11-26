"""C1 (Reviewer 2, major 1; Reviewer 1, Q2): GRFICSv3 technique classification
without test leakage.

(a) in-episode split as in the paper (class-wise chronological 70/30):
    a0  early stopping on the test portion (paper setting, reproduces L8)
    a1  early stopping on a validation block carved from the training portion
        (chronological 60/10/30 per class); test portion unchanged
    a2  no early stopping, fixed 300 trees
(b) leave-one-episode-out for the classes that have two recorded episodes
    (T0813, T0814, T0836): the held-out episode is never seen in training,
    validation for early stopping is the chronological tail of the training data.
(c) live session (2026-05-16/17, separate executions, never used in training):
    technique attributed by the deployed model to anomalous events inside each
    scenario window of the experiment journal.

Outputs: results/c1_grfics_classification.json
"""
import json, os, sys
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT = r"D:\PhD_Project"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
SEED = 42
FEAT = ["f1_valve", "f1_flow", "f2_valve", "f2_flow", "purge_valve", "purge_flow",
        "prod_valve", "prod_flow", "pressure", "level", "A_purge", "B_purge", "C_purge"]
PARAMS = dict(max_depth=7, learning_rate=0.02, subsample=0.9, colsample_bytree=0.9,
              reg_alpha=0.5, reg_lambda=2.0, min_child_weight=5, gamma=0.1,
              eval_metric="mlogloss", verbosity=0, random_state=SEED)

df = pd.read_csv(os.path.join(ROOT, "data", "grad_rl_final_dataset.csv"),
                 low_memory=False, dtype={"mitre_technique": str})
df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
scaler = MinMaxScaler().fit(df.loc[df.label == "Normal", FEAT].values.astype(np.float32))
att = df[df.label != "Normal"].sort_values("ts").reset_index(drop=True)
le = LabelEncoder().fit(att.mitre_technique.str.strip())
att["y"] = le.transform(att.mitre_technique.str.strip())
X = scaler.transform(att[FEAT].values.astype(np.float32))
y = att.y.values
# episode id: new episode when the gap to the previous row of the same label > 60 s
att["ep"] = att.groupby("label").ts.transform(lambda s: (s.diff().dt.total_seconds() > 60).cumsum())
att["episode"] = att.label + "#" + att.ep.astype(str)
CLASSES = list(le.classes_)


def fit(Xtr, ytr, Xval=None, yval=None, n_estimators=1000):
    w = np.clip(compute_sample_weight("balanced", ytr), 1.0, None)
    kw = dict(PARAMS, n_estimators=n_estimators)
    if Xval is not None:
        kw["callbacks"] = [xgb.callback.EarlyStopping(rounds=50, save_best=True)]
    clf = xgb.XGBClassifier(**kw)
    # classes absent from ytr would break the encoder: remap to contiguous ids
    present = np.unique(ytr)
    m = {c: i for i, c in enumerate(present)}
    ev = [(Xval, np.array([m.get(v, -1) for v in yval]))] if Xval is not None else None
    if ev is not None and (ev[0][1] < 0).any():
        keep = ev[0][1] >= 0
        ev = [(Xval[keep], ev[0][1][keep])]
    clf.fit(Xtr, np.array([m[v] for v in ytr]), sample_weight=w,
            eval_set=ev, verbose=False)
    inv = np.array(present)
    return clf, inv


def predict(clf, inv, Xte):
    return inv[np.argmax(clf.predict_proba(Xte), axis=1)]


def report(yt, yp):
    labs = sorted(set(yt) | set(yp))
    r = classification_report(yt, yp, labels=labs, target_names=[CLASSES[i] for i in labs],
                              output_dict=True, zero_division=0)
    return {"macro_f1": f1_score(yt, yp, average="macro", labels=sorted(set(yt)), zero_division=0),
            "accuracy": accuracy_score(yt, yp),
            "per_class": {k: {m: round(v[m], 4) for m in ("precision", "recall", "f1-score", "support")}
                          for k, v in r.items() if k in CLASSES}}


res = {"classes": CLASSES,
       "episodes": att.groupby("episode").agg(technique=("mitre_technique", "first"), n=("y", "size"),
                                               start=("ts", "min"), end=("ts", "max")).astype(str)
                      .reset_index().to_dict("records")}

# ---------------- (a) in-episode split ----------------
tr, va, te, tr70 = [], [], [], []
for c in np.unique(y):
    idx = np.where(y == c)[0]
    n = len(idx)
    tr.extend(idx[: int(n * 0.6)]); va.extend(idx[int(n * 0.6): int(n * 0.7)])
    tr70.extend(idx[: int(n * 0.7)]); te.extend(idx[int(n * 0.7):])
tr, va, te, tr70 = map(np.array, (tr, va, te, tr70))
a = {}
clf, inv = fit(X[tr70], y[tr70], X[te], y[te]); a["a0_es_on_test(paper)"] = dict(report(y[te], predict(clf, inv, X[te])), best_iteration=int(clf.best_iteration))
clf, inv = fit(X[tr], y[tr], X[va], y[va]); a["a1_es_on_validation"] = dict(report(y[te], predict(clf, inv, X[te])), best_iteration=int(clf.best_iteration))
clf, inv = fit(X[tr70], y[tr70], n_estimators=300); a["a2_no_es_300_trees"] = report(y[te], predict(clf, inv, X[te]))
res["a_in_episode"] = a

# ---------------- (b) leave-one-episode-out ----------------
b = []
for ep, g in att.groupby("episode"):
    c = g.y.iloc[0]
    if att[att.y == c].episode.nunique() < 2:
        continue
    test = att.episode == ep
    trn = np.where(~test)[0]
    # chronological 85/15 per class inside the training data for early stopping
    tr_i, va_i = [], []
    for cc in np.unique(y[trn]):
        ii = trn[y[trn] == cc]; k = int(len(ii) * 0.85)
        tr_i.extend(ii[:k]); va_i.extend(ii[k:])
    clf, inv = fit(X[tr_i], y[tr_i], X[va_i], y[va_i])
    yp = predict(clf, inv, X[test.values])
    pred = pd.Series([CLASSES[v] for v in yp]).value_counts().to_dict()
    b.append({"held_out_episode": ep, "technique": CLASSES[c], "n": int(test.sum()),
              "recall": float((yp == c).mean()), "predicted_as": pred})
res["b_leave_one_episode_out"] = b

# ---------------- (c) live session ----------------
LIVE = os.path.join(ROOT, "grfics attack and normal data", "audit_logs")
inc = [json.loads(l) for l in open(os.path.join(LIVE, "xai_incidents.json"), encoding="utf-8") if l.strip()]
inc = pd.DataFrame([{"ts": pd.Timestamp(r["timestamp"]), "tech": r["details"]["technique"],
                     "conf": r["details"]["confidence"], "cvss": r["details"]["cvss"],
                     "action": r["details"]["final_action"]} for r in inc])
OFFSET = pd.Timedelta(hours=3)  # journal clock = live-log clock - 3 h (matches screenshot times)
ev = [tuple(s.strip() for s in l.split("|")) for l in open(os.path.join(LIVE, "experiment_journal.txt")) if l.strip()]
EXPECT = {"T0836": "T0836", "T0813": "T0813", "T0836_RAMP_UP": "T0836", "T0836_ULTRA_SLEY": "T0836",
          "T0836_RAMP_DOWN": "T0836", "T0856": "T0856", "T0814": "T0814", "T0836_MASS_FLOOD": "T0836",
          "T0836_MASS_RESILIENT": "T0836", "T0856_SPOOF": "T0856"}
c_rows = []
for k, (kind, name, ts) in enumerate(ev):
    if kind != "ATTACK_START":
        continue
    st = pd.Timestamp(ts) + OFFSET
    nxt = [pd.Timestamp(x[2]) + OFFSET for x in ev[k + 1:] if x[0] == "ATTACK_END" and x[1] == name]
    nxt_start = [pd.Timestamp(x[2]) + OFFSET for x in ev[k + 1:] if x[0] == "ATTACK_START"]
    en = nxt[0] if nxt else (nxt_start[0] if nxt_start else st + pd.Timedelta("120s"))
    w = inc[(inc.ts >= st) & (inc.ts <= en)]
    exp = EXPECT.get(name)
    row = {"scenario": name, "expected_class": exp or "out-of-vocabulary",
           "window_s": (en - st).total_seconds(), "end_logged": bool(nxt),
           "anomalous_events": int(len(w)),
           "predicted": w.tech.value_counts().to_dict(),
           "mean_conf": round(float(w.conf.mean()), 1) if len(w) else None}
    if exp and len(w):
        row["attribution_accuracy"] = round(float((w.tech == exp).mean()), 4)
    c_rows.append(row)
inv_rows = [r for r in c_rows if r["expected_class"] != "out-of-vocabulary" and r["anomalous_events"]]
tot = sum(r["anomalous_events"] for r in inv_rows)
res["c_live_session"] = {
    "scenarios": c_rows,
    "in_vocabulary_event_weighted_accuracy": round(sum(r["attribution_accuracy"] * r["anomalous_events"] for r in inv_rows) / tot, 4) if tot else None,
    "in_vocabulary_scenario_majority_correct": f"{sum(max(r['predicted'], key=r['predicted'].get) == r['expected_class'] for r in inv_rows)}/{len(inv_rows)}",
}

json.dump(res, open(os.path.join(OUT, "c1_grfics_classification.json"), "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in ("a_in_episode",)}, indent=1, default=str)[:3000])
for r in b: print("LOEO", r)
for r in c_rows: print("LIVE", r)
print("LIVE summary", res["c_live_session"]["in_vocabulary_event_weighted_accuracy"], res["c_live_session"]["in_vocabulary_scenario_majority_correct"])
