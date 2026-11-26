"""C2 (Reviewer 2): quantitative comparison with multivariate time-series anomaly
detectors on SWaT and WADI under one common protocol.

Protocol (as in GDN / TranAD evaluations):
  * SWaT A1 (2015): normal file without the first 6 h (stabilisation), full attack file.
    WADI (2017): 14-day normal file without the first 6 h, full attack-period file;
    four all-empty WADI columns dropped.
  * 10-s down-sampling: median of each 10 rows, label = 1 if any row in the block is an attack.
  * Min-max scaling fitted on normal training data; all features kept (constant ones shifted only).
  * Last 10 % of the normal data = validation (early stopping and threshold).
  * Threshold: P99 of validation scores (the GRAD-RL rule). Also reported, for comparison
    with published numbers only: best F1 over all thresholds (oracle) and point-adjusted F1
    at that threshold.
Detectors: LSTM-AE (GRAD-RL architecture, re-implemented), USAD, TranAD, GDN,
IsolationForest, PCA. Single seed (42), CPU.
Usage: python c2_baselines.py SWaT|WADI [method ...]
Output: results/c2_<dataset>_<method>.json
"""
import json, math, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve

ROOT = r"D:\PhD_Project"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
EVAL_BS = int(os.environ.get("EVAL_BS", "1024"))  # smaller for GDN paper settings (memory)
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)
DS = 10


# ------------------------------------------------------------------ data
def _downsample(df_feat, lab):
    n = len(df_feat) // DS * DS
    X = df_feat.values[:n].astype(np.float32).reshape(-1, DS, df_feat.shape[1])
    X = np.median(X, axis=1)
    y = lab[:n].reshape(-1, DS).max(1) if lab is not None else None
    return X, y


def load(ds):
    f = os.path.join(CACHE, f"{ds}.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True); return z["Xtr"], z["Xte"], z["yte"], list(z["names"])
    if ds == "SWaT":
        n = pd.read_csv(os.path.join(ROOT, "data", "SWaT_Dataset_Normal_v1.csv")); n.columns = n.columns.str.strip()
        a = pd.read_csv(os.path.join(ROOT, "data", "swat2015", "SWaT_Dataset_Attack_v0.csv")); a.columns = a.columns.str.strip()
        lc = "Normal/Attack"; feats = [c for c in n.columns if c not in ("Timestamp", lc)]
        n = n.iloc[21600:]
        ya = (a[lc].astype(str).str.replace(" ", "").str.lower() == "attack").astype(int).values
        Xtr, _ = _downsample(n[feats].apply(pd.to_numeric, errors="coerce").ffill().fillna(0), None)
        Xte, yte = _downsample(a[feats].apply(pd.to_numeric, errors="coerce").ffill().fillna(0), ya)
    else:
        hdr = [c.strip() for c in pd.read_csv(os.path.join(ROOT, "data", "WADI_14days_new.csv"), nrows=0).columns]
        feats = [c for c in hdr if c not in ("Row", "Date", "Time")]
        parts = []
        for ch in pd.read_csv(os.path.join(ROOT, "data", "WADI_14days_new.csv"), chunksize=100_000, low_memory=False):
            ch.columns = ch.columns.str.strip(); parts.append(ch[feats].apply(pd.to_numeric, errors="coerce").astype(np.float32))
        n = pd.concat(parts, ignore_index=True).iloc[21600:]
        a = pd.read_csv(os.path.join(ROOT, "data", "WADI_attackdataLABLE.csv"), skiprows=1, low_memory=False)
        a.columns = a.columns.str.strip(); lc = a.columns[-1]
        a = a[pd.to_numeric(a[lc], errors="coerce").notna()]
        ya = (pd.to_numeric(a[lc]) == -1).astype(int).values
        empty = [c for c in feats if n[c].isna().all() or c not in a.columns]
        feats = [c for c in feats if c not in empty]
        Xtr, _ = _downsample(n[feats].ffill().bfill().fillna(0), None)
        Xte, yte = _downsample(a[feats].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0), ya)
    # all features kept (attacks often switch actuators that are constant in training);
    # constant features are shifted only
    lo, hi = Xtr.min(0), Xtr.max(0); rng = np.where(hi > lo, hi - lo, 1.0)
    Xtr = (Xtr - lo) / rng; Xte = (Xte - lo) / rng
    names = list(feats)
    np.savez_compressed(f, Xtr=Xtr.astype(np.float32), Xte=Xte.astype(np.float32), yte=yte, names=np.array(names))
    return Xtr.astype(np.float32), Xte.astype(np.float32), yte, names


def windows(X, w):
    idx = np.arange(len(X) - w + 1)[:, None] + np.arange(w)
    return X[idx]


# ------------------------------------------------------------------ metrics
def segments(y):
    d = np.diff(np.r_[0, y, 0]); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def point_adjust(y, a):
    a = a.copy()
    for s, e in segments(y):
        if a[s:e].any(): a[s:e] = 1
    return a


def prf(y, a):
    tp = int(((a == 1) & (y == 1)).sum()); fp = int(((a == 1) & (y == 0)).sum()); fn = int(((a == 0) & (y == 1)).sum())
    tn = int(((a == 0) & (y == 0)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-12), 4),
            "fpr": round(fp / max(fp + tn, 1), 4)}


def evaluate(y, s, s_val):
    tau = float(np.percentile(s_val, 99)); a = (s > tau).astype(int)
    seg = segments(y)
    r = {"val_p99": dict(prf(y, a), attacks_detected=f"{sum(int(a[s0:e].any()) for s0, e in seg)}/{len(seg)}")}
    # commissioning calibration: split-conformal (alpha = 0.01) on the test-period
    # normal data before the first attack; evaluation on the remainder
    c = int(np.argmax(y == 1)) - 6
    sc = np.sort(s[:c]); k = math.ceil((c + 1) * 0.99); tau_c = float(sc[min(k, c) - 1])
    ac = (s[c:] > tau_c).astype(int); seg_c = segments(y[c:])
    r["commissioning_conformal"] = dict(prf(y[c:], ac), n_calibration=c,
                                        attacks_detected=f"{sum(int(ac[s0:e].any()) for s0, e in seg_c)}/{len(seg_c)}")
    p, rc, th = precision_recall_curve(y, s); f = 2 * p * rc / np.maximum(p + rc, 1e-12); i = int(np.nanargmax(f[:-1]))
    ab = (s >= th[i]).astype(int)
    r["oracle_best_f1"] = prf(y, ab)
    r["oracle_point_adjusted"] = prf(y, point_adjust(y, ab))
    return r


# ------------------------------------------------------------------ deep models
def batches(n, bs, shuffle):
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def train_loop(model, step_fn, Wtr, Wva, epochs, bs=128, lr=1e-3, patience=3):
    opt = torch.optim.Adam(model.parameters(), lr=lr); best, bad, state = 1e18, 0, None
    for ep in range(epochs):
        model.train()
        for b in batches(len(Wtr), bs, True):
            opt.zero_grad(); loss = step_fn(model, torch.from_numpy(Wtr[b]), ep, True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([step_fn(model, torch.from_numpy(Wva[b]), ep, False).item() for b in batches(len(Wva), EVAL_BS, False)]))
        if vl < best - 1e-7: best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if state: model.load_state_dict(state)
    return ep + 1


class LSTMAE(nn.Module):  # GRAD-RL architecture: 64-32 | repeat | 32-64 | dense
    def __init__(s, d, w):
        super().__init__(); s.w = w
        s.e1 = nn.LSTM(d, 64, batch_first=True); s.e2 = nn.LSTM(64, 32, batch_first=True)
        s.d1 = nn.LSTM(32, 32, batch_first=True); s.d2 = nn.LSTM(32, 64, batch_first=True); s.o = nn.Linear(64, d)
        s.dr = nn.Dropout(0.2)
    def forward(s, x):
        h, _ = s.e1(x); h = s.dr(torch.relu(h)); _, (z, _) = s.e2(h); z = torch.relu(z[-1]).unsqueeze(1).repeat(1, s.w, 1)
        h, _ = s.d1(z); h, _ = s.d2(torch.relu(h)); return s.o(torch.relu(h))


class USAD(nn.Module):
    def __init__(s, n_in, z=40):
        super().__init__()
        def enc(): return nn.Sequential(nn.Linear(n_in, n_in // 2), nn.ReLU(), nn.Linear(n_in // 2, n_in // 4), nn.ReLU(), nn.Linear(n_in // 4, z), nn.ReLU())
        def dec(): return nn.Sequential(nn.Linear(z, n_in // 4), nn.ReLU(), nn.Linear(n_in // 4, n_in // 2), nn.ReLU(), nn.Linear(n_in // 2, n_in), nn.Sigmoid())
        s.E, s.D1, s.D2 = enc(), dec(), dec()


class TranAD(nn.Module):
    def __init__(s, d, w):
        super().__init__(); m = 2 * d; nh = d if m % d == 0 and d <= 64 else next(h for h in (8, 6, 4, 3, 2, 1) if m % h == 0)
        s.d, s.w = d, w
        pe = torch.zeros(w, m); pos = torch.arange(w).unsqueeze(1); div = torch.exp(torch.arange(0, m, 2) * (-math.log(10000.0) / m))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]; s.register_buffer("pe", pe)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(m, nh, 16, 0.1, batch_first=True), 1)
        s.dec1 = nn.TransformerDecoder(nn.TransformerDecoderLayer(m, nh, 16, 0.1, batch_first=True), 1)
        s.dec2 = nn.TransformerDecoder(nn.TransformerDecoderLayer(m, nh, 16, 0.1, batch_first=True), 1)
        s.fc = nn.Sequential(nn.Linear(m, d), nn.Sigmoid())
    def encode(s, W, c, tgt):
        mem = s.enc(torch.cat([W, c], 2) * math.sqrt(s.d) + s.pe)
        t = tgt.repeat(1, 1, 2); return t, mem
    def forward(s, W):
        tgt = W[:, -1:, :]; c = torch.zeros_like(W)
        t, mem = s.encode(W, c, tgt); x1 = s.fc(s.dec1(t, mem)); x2 = s.fc(s.dec2(t, mem))
        c = (x1 - tgt) ** 2; t, mem = s.encode(W, c.repeat(1, s.w, 1), tgt); x2h = s.fc(s.dec2(t, mem))
        return x1, x2, x2h


class GDN(nn.Module):
    def __init__(s, d, w, dim=64, k=15):
        super().__init__(); s.k = min(k, d - 1); s.emb = nn.Embedding(d, dim); s.lin = nn.Linear(w, dim, bias=False)
        s.att_i = nn.Parameter(torch.randn(2 * dim) * 0.1); s.att_j = nn.Parameter(torch.randn(2 * dim) * 0.1)
        s.bn = nn.BatchNorm1d(dim); s.out = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, 1)); s.d = d
    def forward(s, x):  # x: (B, w, d) -> forecast (B, d)
        B = x.shape[0]; v = s.emb.weight; vn = v / v.norm(dim=1, keepdim=True)
        sim = vn @ vn.T; sim.fill_diagonal_(-1e9); nb = sim.topk(s.k, dim=1).indices  # (d, k)
        h = s.lin(x.transpose(1, 2))  # (B, d, dim)
        gi = torch.cat([h, v.expand(B, -1, -1)], 2)  # (B, d, 2dim)
        hj = h[:, nb]; vj = v[nb].expand(B, -1, -1, -1); gj = torch.cat([hj, vj], 3)  # (B, d, k, 2dim)
        e = torch.nn.functional.leaky_relu((gi * s.att_i).sum(-1, keepdim=True) + (gj * s.att_j).sum(-1), 0.2)  # (B,d,k)
        a = torch.softmax(e, -1).unsqueeze(-1); z = (a * hj).sum(2)
        z = torch.relu(s.bn(z.reshape(-1, z.shape[-1]))).reshape(B, s.d, -1) * v
        return s.out(z.reshape(-1, z.shape[-1])).reshape(B, s.d)


def run(method, Xtr, Xte, yte):
    nv = int(0.1 * len(Xtr)); Xfit, Xval = Xtr[:-nv], Xtr[-nv:]
    t0 = time.time(); info = {}
    if method == "IsolationForest":
        m = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=2).fit(Xfit)
        sv, st = -m.score_samples(Xval), -m.score_samples(Xte)
    elif method == "PCA":
        m = PCA(0.95, random_state=SEED).fit(Xfit); info["components"] = int(m.n_components_)
        rec = lambda X: ((X - m.inverse_transform(m.transform(X))) ** 2).mean(1)
        sv, st = rec(Xval), rec(Xte)
    else:
        w = 5 if method == "GDN" else 10
        Wf, Wv, Wt = windows(Xfit, w), windows(Xval, w), windows(Xte, w); d = Xtr.shape[1]
        if method == "LSTM-AE":
            model = LSTMAE(d, w)
            step = lambda m_, W, ep, tr: ((m_(W) - W) ** 2).mean()
            info["epochs"] = train_loop(model, step, Wf, Wv, 20, bs=64)
            score = lambda m_, W: ((m_(W) - W) ** 2).mean((1, 2))
        elif method == "USAD":
            n_in = w * d; model = USAD(n_in)
            Wf, Wv, Wt = Wf.reshape(len(Wf), -1), Wv.reshape(len(Wv), -1), Wt.reshape(len(Wt), -1)
            o1 = torch.optim.Adam(list(model.E.parameters()) + list(model.D1.parameters()))
            o2 = torch.optim.Adam(list(model.E.parameters()) + list(model.D2.parameters()))
            EP = 30
            for ep in range(EP):
                n = ep + 1
                for b in batches(len(Wf), 256, True):
                    W = torch.from_numpy(Wf[b]); z = model.E(W); w1 = model.D1(z); w2 = model.D2(z); w3 = model.D2(model.E(w1))
                    l1 = 1 / n * ((W - w1) ** 2).mean() + (1 - 1 / n) * ((W - w3) ** 2).mean()
                    l1.backward(); o1.step(); o1.zero_grad(); o2.zero_grad()
                    z = model.E(W); w1 = model.D1(z); w2 = model.D2(z); w3 = model.D2(model.E(w1))
                    l2 = 1 / n * ((W - w2) ** 2).mean() - (1 - 1 / n) * ((W - w3) ** 2).mean()
                    l2.backward(); o2.step(); o1.zero_grad(); o2.zero_grad()
            info["epochs"] = EP
            def score(m_, W):
                w1 = m_.D1(m_.E(W)); w2 = m_.D2(m_.E(w1)); return 0.5 * ((W - w1) ** 2).mean(1) + 0.5 * ((W - w2) ** 2).mean(1)
        elif method == "TranAD":
            model = TranAD(d, w)
            def step(m_, W, ep, tr):
                n = ep + 1; x1, x2, x2h = m_(W); tgt = W[:, -1:, :]
                return 1 / n * ((x1 - tgt) ** 2).mean() + (1 - 1 / n) * ((x2h - tgt) ** 2).mean()
            info["epochs"] = train_loop(model, step, Wf, Wv, 5, bs=128, lr=1e-4, patience=5)
            def score(m_, W):
                x1, _, x2h = m_(W); tgt = W[:, -1:, :]; return (0.5 * (x1 - tgt) ** 2 + 0.5 * (x2h - tgt) ** 2).mean((1, 2))
        elif method == "GDN":
            # GDN_CFG=paper: settings of the GDN paper for WADI (embedding 128, top-k 30, 50 epochs, patience 15)
            paper_cfg = os.environ.get("GDN_CFG") == "paper"
            model = GDN(d, w, dim=128 if paper_cfg else 64, k=30 if (paper_cfg or d >= 80) else 15)
            info["config"] = "GDN paper (dim 128, top-k 30, max 50 epochs, patience 15)" if paper_cfg else "default (dim 64, max 30 epochs, patience 5)"
            Yf, Yv, Yt = Xfit[w:], Xval[w:], Xte[w:]; Wf, Wv, Wt = Wf[:-1], Wv[:-1], Wt[:-1]
            Wf = np.concatenate([Wf, Yf[:, None, :]], 1); Wv = np.concatenate([Wv, Yv[:, None, :]], 1); Wt = np.concatenate([Wt, Yt[:, None, :]], 1)
            step = lambda m_, W, ep, tr: ((m_(W[:, :-1]) - W[:, -1]) ** 2).mean()
            info["epochs"] = train_loop(model, step, Wf, Wv, 50 if paper_cfg else 30, bs=128, patience=15 if paper_cfg else 5)
            model.eval()
            with torch.no_grad():
                ev = np.vstack([(model(torch.from_numpy(Wv[b][:, :-1])) - torch.from_numpy(Wv[b][:, -1])).abs().numpy() for b in batches(len(Wv), EVAL_BS, False)])
            med = np.median(ev, 0); iqr = np.subtract(*np.percentile(ev, [75, 25], 0)) + 1e-2
            def score(m_, W):
                e = (m_(W[:, :-1]) - W[:, -1]).abs().numpy(); s_ = ((e - med) / iqr).max(1); return torch.from_numpy(s_)
        model.eval()
        with torch.no_grad():
            sc = lambda W: np.concatenate([np.asarray(score(model, torch.from_numpy(W[b]))) for b in batches(len(W), EVAL_BS, False)])
            sv, st = sc(Wv), sc(Wt)
        if method == "GDN":  # moving average over 3 points as in GDN
            sm = lambda s_: np.convolve(s_, np.ones(3) / 3, "same"); sv, st = sm(sv), sm(st)
        pad = len(Xte) - len(st); st = np.r_[np.full(pad, st[0]), st]
    info["train_seconds"] = round(time.time() - t0, 1)
    np.savez_compressed(os.path.join(CACHE, f"scores_{CUR_DS}_{method}.npz"), st=st, sv=sv, y=yte)
    return evaluate(yte, st, sv), info


if __name__ == "__main__":
    ds = sys.argv[1]; methods = sys.argv[2:] or ["PCA", "IsolationForest", "LSTM-AE", "USAD", "TranAD", "GDN"]
    CUR_DS = ds
    Xtr, Xte, yte, names = load(ds)
    print(ds, "train", Xtr.shape, "test", Xte.shape, "attack share", round(float(yte.mean()), 4), "segments", len(segments(yte)), flush=True)
    for m in methods:
        r, info = run(m, Xtr, Xte, yte)
        out = {"dataset": ds, "method": m, "n_features": Xtr.shape[1], "n_train": len(Xtr), "n_test": len(Xte), **info, **r}
        tag = "_paper_cfg" if (m == "GDN" and os.environ.get("GDN_CFG") == "paper") else ""
        json.dump(out, open(os.path.join(OUT, f"c2_{ds}_{m}{tag}.json"), "w"), indent=1)
        print(m, info, r["val_p99"], "COMM", r["commissioning_conformal"], "oracleF1", r["oracle_best_f1"]["f1"], "PA-F1", r["oracle_point_adjusted"]["f1"], flush=True)
