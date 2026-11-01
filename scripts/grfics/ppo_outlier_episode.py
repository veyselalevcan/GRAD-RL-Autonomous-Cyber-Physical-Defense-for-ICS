"""Inspect the single PPO evaluation episode with return -2000."""
import os, sys, numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gradrl import paths
from gradrl.grfics_env import GrficsDefenseEnv
from stable_baselines3 import PPO
F = ["f1_valve","f1_flow","f2_valve","f2_flow","purge_valve","purge_flow","prod_valve","prod_flow","pressure","level","A_purge","B_purge","C_purge"]
df = pd.read_csv(paths.DATA_DIR / "grad_rl_final_dataset.csv", low_memory=False, dtype={"mitre_technique": str, "mitre_tactic": str})
df.columns = df.columns.str.strip()
X = joblib.load(paths.MODEL_DIR / "scaler_grfics_live.pkl").transform(df[F].values.astype(np.float32))
y = (df["label"] != "Normal").astype(np.int32).values
m = PPO.load(paths.MODEL_DIR / "rl_defender_grfics.zip")
env = GrficsDefenseEnv(sensor_data=X, labels=y, max_steps=200)
for ep in range(50):
    o, _ = env.reset(seed=ep); R = 0; idx = []; acts = []; hp = []
    while True:
        a = int(m.predict(o, deterministic=True)[0]); o, r, t, tr, info = env.step(a)
        R += r; idx.append(info["replay_idx"]); acts.append(a); hp.append(info["high_pressure"])
        if t or tr: break
    if R <= -1000:
        lab = df["label"].iloc[idx]
        print("seed", ep, "return", R, "rows", idx[0], idx[-1], "labels", lab.value_counts().to_dict(),
              "high-pressure steps", sum(hp), "actions", np.bincount(acts, minlength=3).tolist(),
              "pressure range", float(X[idx, 8].min()), float(X[idx, 8].max()))
