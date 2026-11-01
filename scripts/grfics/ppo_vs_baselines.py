"""PPO vs. simple baselines in the GRFICSv3 replay environment.

Uses the same environment, data, scaler, episode length (200) and seeds (0..49)
as evaluate_agent() in scripts/grfics/train_ppo.py.
"""
import json, os, sys
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gradrl import paths
from gradrl.grfics_env import GrficsDefenseEnv, PRESSURE_IDX, HIGH_PRESSURE_THRESHOLD
from stable_baselines3 import PPO
from scipy.stats import wilcoxon

FEATURES = ["f1_valve", "f1_flow", "f2_valve", "f2_flow", "purge_valve", "purge_flow",
            "prod_valve", "prod_flow", "pressure", "level", "A_purge", "B_purge", "C_purge"]
df = pd.read_csv(paths.DATA_DIR / "grad_rl_final_dataset.csv", low_memory=False,
                 dtype={"mitre_technique": str, "mitre_tactic": str})
df.columns = df.columns.str.strip()
X = joblib.load(paths.MODEL_DIR / "scaler_grfics_live.pkl").transform(df[FEATURES].values.astype(np.float32))
y = (df["label"] != "Normal").astype(np.int32).values
model = PPO.load(paths.MODEL_DIR / "rl_defender_grfics.zip")

rng = np.random.default_rng(42)
policies = {
    "PPO": lambda o: int(model.predict(o, deterministic=True)[0]),
    "Threshold rule (valve if p>=0.90)": lambda o: 1 if o[PRESSURE_IDX] >= HIGH_PRESSURE_THRESHOLD else 0,
    "Always DO_NOTHING": lambda o: 0,
    "Always OPEN_SAFETY_VALVE": lambda o: 1,
    "Random": lambda o: int(rng.integers(0, 3)),
}

out, per_ep = {}, {}
for name, pol in policies.items():
    env = GrficsDefenseEnv(sensor_data=X, labels=y, max_steps=200)
    acts = np.zeros(3); rewards = []; health = []; hp = hp_ok = 0; normal_steps = normal_disrupt = 0
    for ep in range(50):
        obs, _ = env.reset(seed=ep); R = 0.0
        while True:
            a = pol(obs)
            obs, r, term, trunc, info = env.step(a)
            R += r; acts[a] += 1
            if info["high_pressure"]:
                hp += 1; hp_ok += (a == 1)
            elif y[info["replay_idx"]] == 0:
                normal_steps += 1; normal_disrupt += (a != 0)
            if term or trunc:
                rewards.append(R); health.append(info["op_health"]); break
    per_ep[name] = rewards
    out[name] = {"mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
                 "mean_health": float(np.mean(health)),
                 "hp_response_pct": 100 * hp_ok / max(hp, 1), "hp_steps": hp,
                 "normal_disruptive_pct": 100 * normal_disrupt / max(normal_steps, 1),
                 "action_pct": (100 * acts / acts.sum()).round(1).tolist()}
for name in per_ep:
    if name != "PPO":
        d = np.array(per_ep["PPO"]) - np.array(per_ep[name])
        out[name]["ppo_minus_this_mean"] = float(d.mean())
        out[name]["wilcoxon_p"] = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
print(json.dumps(out, indent=2))
out["_per_episode_rewards"] = per_ep
json.dump(out, open(paths.RESULTS_DIR / "grfics" / "ppo_vs_baselines.json", "w"), indent=2)
