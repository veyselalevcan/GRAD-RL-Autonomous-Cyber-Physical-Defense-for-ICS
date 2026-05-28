"""
train_rl_agent.py — GRAD-RL Framework (v7.0)
PPO Reinforcement Learning Agent Training for GRFICS Defense

Trains a Proximal Policy Optimization (PPO) agent on the GrficsDefenseEnv
using real sensor sequences from grad_rl_final_dataset.csv.

Architecture position:
    [LSTM-AE: anomaly detection]  →  [XGBoost: attack classification]
                                              ↓
                                   [PPO Agent: mitigation action]  ← THIS FILE
                                              ↓
                              [Action: DO_NOTHING / OPEN_VALVE / SHUTDOWN]

Saved artifacts:
    models/rl_defender_grfics.zip          ← trained PPO policy
    docs/results/rl_training_rewards.png   ← reward curve
    docs/results/rl_evaluation_report.txt  ← final evaluation summary

Usage:
    pip install stable-baselines3
    python train_rl_agent.py
"""

import os
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnRewardThreshold,
    CheckpointCallback, BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from grfics_env import GrficsDefenseEnv, OBS_IDX, PRESSURE_IDX

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
SEED         = 42
DATA_PATH    = r"D:\PhD_Project\data\grad_rl_final_dataset.csv"
SCALER_PATH  = r"D:\PhD_Project\models\scaler_grfics_live.pkl"
MODEL_DIR    = "models"
RESULTS_DIR  = "docs/results"
AGENT_PATH   = os.path.join(MODEL_DIR, "rl_defender_grfics")

TOTAL_TIMESTEPS = 200_000   # increase to 500_000 for better convergence
EPISODE_STEPS   = 200
N_ENVS          = 4         # parallel environments for faster training
EVAL_FREQ       = 10_000    # evaluate every N steps
EVAL_EPISODES   = 20

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
#  DATA PREPARATION
# ─────────────────────────────────────────────

def load_replay_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load and scale the full CSV for replay-mode training.

    The RL agent trains on real sensor sequences rather than purely synthetic
    observations — this grounds the policy in actual process dynamics.

    Returns:
        sensor_data : np.ndarray (N, 13)  MinMaxScaled features
        labels      : np.ndarray (N,)     binary anomaly labels (0/1)
    """
    print("  Loading replay data from CSV...")
    df = pd.read_csv(
        DATA_PATH, low_memory=False,
        dtype={"mitre_technique": str, "mitre_tactic": str},
    )
    df.columns = df.columns.str.strip()

    FEATURE_COLS = [
        "f1_valve", "f1_flow", "f2_valve", "f2_flow",
        "purge_valve", "purge_flow", "prod_valve", "prod_flow",
        "pressure", "level", "A_purge", "B_purge", "C_purge",
    ]

    # Scale with the pre-fitted normal-data scaler
    scaler      = joblib.load(SCALER_PATH)
    sensor_data = scaler.transform(df[FEATURE_COLS].values.astype(np.float32))

    # Binary labels: Normal=0, any attack=1
    labels = (df["label"] != "Normal").astype(np.int32).values

    print(f"  Replay data: {sensor_data.shape}  "
          f"Normal={int((labels==0).sum()):,}  Attack={int((labels==1).sum()):,}")
    return sensor_data, labels


# ─────────────────────────────────────────────
#  REWARD LOGGING CALLBACK
# ─────────────────────────────────────────────

class RewardLogger(BaseCallback):
    """Tracks mean episode reward for plotting."""
    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        self._ep_reward += float(reward)
        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
        return True


# ─────────────────────────────────────────────
#  ENVIRONMENT FACTORY
# ─────────────────────────────────────────────

def make_env(sensor_data: np.ndarray, labels: np.ndarray):
    """Return a callable that creates a monitored GrficsDefenseEnv."""
    def _init():
        env = GrficsDefenseEnv(
            sensor_data=sensor_data,
            labels=labels,
            max_steps=EPISODE_STEPS,
        )
        return Monitor(env)
    return _init


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────

def train_ppo(sensor_data: np.ndarray, labels: np.ndarray) -> PPO:
    print("\n─── PPO Agent Training ───")

    # Vectorised parallel environments for faster sample collection
    vec_env = make_vec_env(
        make_env(sensor_data, labels),
        n_envs=N_ENVS,
        seed=SEED,
    )

    # Evaluation environment (single, unmonitored)
    eval_env = Monitor(GrficsDefenseEnv(
        sensor_data=sensor_data,
        labels=labels,
        max_steps=EPISODE_STEPS,
    ))

    # Callbacks
    reward_logger = RewardLogger()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=RESULTS_DIR,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(EVAL_FREQ // N_ENVS, 1),
        save_path=MODEL_DIR,
        name_prefix="rl_defender_grfics_ckpt",
        verbose=0,
    )

    # PPO hyperparameters — tuned for short-horizon ICS episodes
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=512,           # rollout length per env
        batch_size=64,
        n_epochs=10,
        gamma=0.99,            # discount — long-term health preservation
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # entropy bonus: prevents premature convergence
        verbose=1,
        seed=SEED,
        
    )

    print(f"  Policy  : MlpPolicy (obs=13, act=3)")
    print(f"  Steps   : {TOTAL_TIMESTEPS:,}  |  N_envs: {N_ENVS}")
    print(f"  Eval    : every {EVAL_FREQ:,} steps  ({EVAL_EPISODES} episodes)")
    print()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[reward_logger, eval_callback, checkpoint_cb],
        progress_bar=True,
    )

    # Save final model
    model.save(AGENT_PATH)
    print(f"\n  ✅ Final PPO agent saved → {AGENT_PATH}.zip")

    return model, reward_logger


# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────

def evaluate_agent(
    model: PPO,
    sensor_data: np.ndarray,
    labels: np.ndarray,
    n_episodes: int = 50,
) -> dict:
    """
    Run the trained agent deterministically and compute aggregate metrics.

    Returns:
        dict with mean_reward, action_distribution, high_press_handled,
        avg_op_health, anomaly_detection_rate
    """
    print("\n─── Final Agent Evaluation ───")

    env = GrficsDefenseEnv(sensor_data=sensor_data, labels=labels,
                           max_steps=EPISODE_STEPS)

    action_counts = {0: 0, 1: 0, 2: 0}
    episode_rewards, op_healths = [], []
    high_press_steps = correct_valve_steps = 0
    total_attack_steps = detected_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            action     = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            action_counts[action] += 1

            if info["high_pressure"]:
                high_press_steps += 1
                if action == 1:   # OPEN_SAFETY_VALVE
                    correct_valve_steps += 1

            if labels is not None:
                replay_idx = info.get("replay_idx", 0)
                if replay_idx < len(labels) and labels[replay_idx] == 1:
                    total_attack_steps += 1
                    if action != 0:   # any mitigation during attack
                        detected_steps += 1

            if terminated or truncated:
                episode_rewards.append(ep_reward)
                op_healths.append(info["op_health"])
                break

    total_actions = sum(action_counts.values())
    action_pct    = {k: v/total_actions*100 for k, v in action_counts.items()}

    results = {
        "n_episodes":             n_episodes,
        "mean_reward":            float(np.mean(episode_rewards)),
        "std_reward":             float(np.std(episode_rewards)),
        "mean_op_health":         float(np.mean(op_healths)),
        "action_pct_do_nothing":  round(action_pct[0], 1),
        "action_pct_valve":       round(action_pct[1], 1),
        "action_pct_shutdown":    round(action_pct[2], 1),
        "high_press_handled_pct": round(
            correct_valve_steps / max(high_press_steps, 1) * 100, 1),
        "attack_response_rate":   round(
            detected_steps / max(total_attack_steps, 1) * 100, 1),
    }

    print(f"  Episodes           : {n_episodes}")
    print(f"  Mean reward        : {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean op. health    : {results['mean_op_health']:.3f}")
    print(f"  Action dist        : DO_NOTHING={results['action_pct_do_nothing']}%  "
          f"VALVE={results['action_pct_valve']}%  "
          f"SHUTDOWN={results['action_pct_shutdown']}%")
    print(f"  High-press handled : {results['high_press_handled_pct']}%")
    print(f"  Attack response    : {results['attack_response_rate']}%")

    return results


# ─────────────────────────────────────────────
#  PLOTS & REPORT
# ─────────────────────────────────────────────

def save_reward_plot(reward_logger: RewardLogger):
    rewards = reward_logger.episode_rewards
    if not rewards:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.4, color="steelblue", label="Episode reward")
    # Rolling mean
    window = max(1, len(rewards) // 20)
    rolling = pd.Series(rewards).rolling(window, min_periods=1).mean()
    plt.plot(rolling, color="navy", linewidth=2, label=f"Rolling mean (w={window})")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("PPO Training — Episode Rewards [GRFICS]")
    plt.xlabel("Episode"); plt.ylabel("Total Reward")
    plt.legend(); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "rl_training_rewards.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Reward plot saved → {path}")


def save_text_report(results: dict):
    path = os.path.join(RESULTS_DIR, "rl_evaluation_report.txt")
    lines = [
        "=" * 60,
        "  GRAD-RL: PPO AGENT EVALUATION REPORT [GRFICS]",
        "=" * 60,
        f"  Episodes evaluated : {results['n_episodes']}",
        f"  Mean episode reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}",
        f"  Mean op. health    : {results['mean_op_health']:.3f}",
        "",
        "  Action Distribution:",
        f"    DO_NOTHING          : {results['action_pct_do_nothing']}%",
        f"    OPEN_SAFETY_VALVE   : {results['action_pct_valve']}%",
        f"    EMERGENCY_SHUTDOWN  : {results['action_pct_shutdown']}%",
        "",
        "  Safety Metrics:",
        f"    High-pressure events handled correctly : {results['high_press_handled_pct']}%",
        f"    Attack steps with mitigation response  : {results['attack_response_rate']}%",
        "=" * 60,
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✅ Text report saved → {path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  GRAD-RL: PPO RL AGENT TRAINING [GRFICS]")
    print(f"  Timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Env steps : {EPISODE_STEPS} per episode")
    print(f"{'='*60}\n")

    # Load real sensor data for replay-mode training
    sensor_data, labels = load_replay_data()

    # Train
    model, reward_logger = train_ppo(sensor_data, labels)

    # Evaluate
    results = evaluate_agent(model, sensor_data, labels, n_episodes=50)

    # Save outputs
    save_reward_plot(reward_logger)
    save_text_report(results)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Agent → {AGENT_PATH}.zip")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()