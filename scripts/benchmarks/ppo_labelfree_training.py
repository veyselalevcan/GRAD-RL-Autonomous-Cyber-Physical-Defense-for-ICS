"""C4b (Reviewer 2): PPO reward without attack labels.

The paper's reward treats a step as critical when normalized pressure >= 0.90 OR the
replayed row carries an attack label. Two label-free variants are trained with the
same environment, hyper-parameters, 200,000 steps, 4 environments and seed 42:
  PPO-pressure  critical = pressure >= 0.90 only
  PPO-detector  critical = pressure >= 0.90 OR LSTM-AE alarm (window MSE > tau_live =
                1.2 x offline P99), i.e. the signal the deployed pipeline actually has
All policies are evaluated on the same 50 replay episodes (seeds 0-49, 200 steps) with
label-independent metrics (high-pressure response, operational health, disruptive
actions on normal steps, valve share on attack steps) and with the return under the
paper's reward. Replay limitation unchanged: actions do not alter the recorded process.
Output: results/c4_ppo_labelfree.json, models in revision_experiments/models/
"""
import json, os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, joblib
sys.path.insert(0, r"D:\PhD_Project\src\grficsv3")
from grfics_env import GrficsDefenseEnv, PRESSURE_IDX, HIGH_PRESSURE_THRESHOLD
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from scipy.stats import wilcoxon

ROOT = r"D:\PhD_Project"; HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "models"); os.makedirs(MD, exist_ok=True)
FEAT = ["f1_valve", "f1_flow", "f2_valve", "f2_flow", "purge_valve", "purge_flow",
        "prod_valve", "prod_flow", "pressure", "level", "A_purge", "B_purge", "C_purge"]
df = pd.read_csv(os.path.join(ROOT, "data", "grad_rl_final_dataset.csv"), low_memory=False, dtype={"mitre_technique": str})
X = joblib.load(os.path.join(ROOT, "models", "scaler_grfics_live.pkl")).transform(df[FEAT].values.astype(np.float32)).astype(np.float32)
y = (df["label"] != "Normal").astype(np.int32).values

flag_path = os.path.join(HERE, "cache", "grfics_detector_flags.npy")
if os.path.exists(flag_path):
    flags = np.load(flag_path)
else:
    from tensorflow.keras.models import load_model
    m = load_model(os.path.join(ROOT, "models", "detective_lstm_grfics_live.keras"))
    W = np.stack([X[i:i + 10] for i in range(len(X) - 9)])
    mse = ((m.predict(W, batch_size=1024, verbose=0) - W) ** 2).mean((1, 2))
    tau = 1.2 * json.load(open(os.path.join(ROOT, "models", "threshold_grfics_live.json")))["threshold"]
    flags = np.r_[np.zeros(9, np.int32), (mse > tau).astype(np.int32)]
    os.makedirs(os.path.dirname(flag_path), exist_ok=True); np.save(flag_path, flags)
res = {"detector_flags": {"tau_live": "1.2 x P99", "flag_rate_on_normal_rows": float(flags[y == 0].mean()),
                          "flag_rate_on_attack_rows": float(flags[y == 1].mean())}}

signals = {"PPO-pressure": np.zeros_like(y), "PPO-detector": flags}
models = {"PPO (paper, label reward)": PPO.load(os.path.join(ROOT, "models", "rl_defender_grfics.zip"))}
for name, sig in signals.items():
    path = os.path.join(MD, name + ".zip")
    if os.path.exists(path):
        models[name] = PPO.load(path); continue
    venv = make_vec_env(lambda: GrficsDefenseEnv(sensor_data=X, labels=sig, max_steps=200), n_envs=4, seed=42)
    mdl = PPO("MlpPolicy", venv, learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10, gamma=0.99,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, seed=42, verbose=0)
    mdl.learn(total_timesteps=200_000); mdl.save(path); models[name] = mdl
    print("trained", name, flush=True)

rng = np.random.default_rng(42)
pols = {k: (lambda o, m=m: int(m.predict(o, deterministic=True)[0])) for k, m in models.items()}
pols.update({"Threshold rule": lambda o: 1 if o[PRESSURE_IDX] >= HIGH_PRESSURE_THRESHOLD else 0,
             "Always DO_NOTHING": lambda o: 0, "Always OPEN_SAFETY_VALVE": lambda o: 1,
             "Random": lambda o: int(rng.integers(0, 3))})
per = {}
for name, pol in pols.items():
    env = GrficsDefenseEnv(sensor_data=X, labels=y, max_steps=200)  # paper reward for the return column
    R_all, H, acts = [], [], np.zeros(3); hp = hp_ok = ns = nd = at = at_v = 0
    for ep in range(50):
        o, _ = env.reset(seed=ep); R = 0.0
        while True:
            a = pol(o); o, r, term, trunc, info = env.step(a); R += r; acts[a] += 1
            lab = y[info["replay_idx"]]
            if info["high_pressure"]: hp += 1; hp_ok += a == 1
            elif lab == 0: ns += 1; nd += a != 0
            if lab == 1: at += 1; at_v += a == 1
            if term or trunc: R_all.append(R); H.append(info["op_health"]); break
    per[name] = R_all
    res[name] = {"return_paper_reward": [round(float(np.mean(R_all)), 2), round(float(np.std(R_all)), 2)],
                 "mean_health": round(float(np.mean(H)), 3), "hp_response_%": round(100 * hp_ok / max(hp, 1), 1),
                 "normal_disruptive_%": round(100 * nd / max(ns, 1), 1), "attack_steps_valve_%": round(100 * at_v / max(at, 1), 1),
                 "shutdown_%": round(100 * acts[2] / acts.sum(), 2)}
for k in per:
    if k != "Threshold rule":
        d = np.array(per[k]) - np.array(per["Threshold rule"])
        res[k]["vs_threshold_rule_wilcoxon_p"] = float(wilcoxon(d).pvalue) if np.any(d) else 1.0
json.dump(res, open(os.path.join(HERE, "results", "c4_ppo_labelfree.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
