"""
grfics_env.py — GRAD-RL Framework (v7.1)
Gymnasium RL Environment for GRFICSv3 Digital-Twin Defense

Observation Space : Box(13,) — 13 MinMaxScaled physical sensor readings
                    Feature order (matches FEATURE_COLS in data_loader_grfics.py):
                      [0] f1_valve    [1] f1_flow    [2] f2_valve    [3] f2_flow
                      [4] purge_valve [5] purge_flow  [6] prod_valve  [7] prod_flow
                      [8] pressure    [9] level       [10] A_purge
                      [11] B_purge    [12] C_purge

Action Space      : Discrete(3)
                      0 : DO_NOTHING             — no intervention
                      1 : OPEN_SAFETY_VALVE      — relieve overpressure
                      2 : EMERGENCY_PUMP_SHUTDOWN — stop production pump

High-pressure threshold:
    Raw `pressure` range: [97.76, 3200.0].
    MinMaxScaled: (2900 - 97.76) / (3200 - 97.76) ≈ 0.904
    HIGH_PRESSURE_THRESHOLD = 0.90  (conservative safety margin)

Connecting to live SCADA:
    Override _get_live_observation() to call scada_monitor.py / Detective API.
"""

import random
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ─────────────────────────────────────────────
#  FEATURE INDEX MAP  (matches data_loader_grfics.FEATURE_COLS order)
# ─────────────────────────────────────────────
OBS_IDX: dict[str, int] = {
    "f1_valve":    0,
    "f1_flow":     1,
    "f2_valve":    2,
    "f2_flow":     3,
    "purge_valve": 4,
    "purge_flow":  5,
    "prod_valve":  6,
    "prod_flow":   7,
    "pressure":    8,   # ← key safety sensor
    "level":       9,
    "A_purge":    10,
    "B_purge":    11,
    "C_purge":    12,
}

N_SENSORS  = 13
MAX_STEPS  = 200

# Actions
DO_NOTHING          = 0
OPEN_SAFETY_VALVE   = 1
EMERGENCY_SHUTDOWN  = 2

# Pressure column index and normalised threshold.
# Raw pressure stats from CSV: min=97.76, max=3200.0, mean=2683.4
# After MinMaxScaler: 0.90 ≈ raw value of ~2890 psi — near-critical level.
PRESSURE_IDX           = OBS_IDX["pressure"]
HIGH_PRESSURE_THRESHOLD = 0.90


# ─────────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────────

class GrficsDefenseEnv(gym.Env):
    """
    Gymnasium environment wrapping the GRFICSv3 digital twin process.

    Modes:
        replay — steps through pre-loaded sensor_data array (N, 13).
        live   — calls _get_live_observation() for real-time SCADA data.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        sensor_data: Optional[np.ndarray] = None,
        labels:      Optional[np.ndarray] = None,
        max_steps:   int = MAX_STEPS,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            sensor_data : Pre-recorded MinMaxScaled matrix (N, 13).
                          If None, synthetic observations are generated.
            labels      : Binary anomaly labels (N,) aligned with sensor_data.
                          0 = Normal, 1 = Attack. Used for reward computation.
            max_steps   : Episode horizon (steps).
            render_mode : "human" for console output.
        """
        super().__init__()
        self.render_mode = render_mode
        self.max_steps   = max_steps
        self._sensor_data = sensor_data
        self._labels      = labels
        self._use_replay  = sensor_data is not None

        # ── Spaces ────────────────────────────────────────────────────────
        # 13 MinMaxScaled features from the GRFICS process — all in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_SENSORS,), dtype=np.float32
        )
        # Three physical interventions available via OpenPLC ladder logic
        self.action_space = spaces.Discrete(3)

        # Episode state
        self._obs:          np.ndarray = np.zeros(N_SENSORS, dtype=np.float32)
        self._step_count:   int   = 0
        self._replay_idx:   int   = 0
        self._op_health:    float = 1.0
        self._total_reward: float = 0.0

    # ──────────────────────────────────────────
    #  RESET
    # ──────────────────────────────────────────

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count   = 0
        self._op_health    = 1.0
        self._total_reward = 0.0

        if self._use_replay and self._sensor_data is not None:
            n_rows     = len(self._sensor_data)
            max_start  = max(0, n_rows - self.max_steps - 1)
            self._replay_idx = int(self.np_random.integers(0, max(1, max_start)))
            self._obs  = self._sensor_data[self._replay_idx].astype(np.float32)
        else:
            self._obs  = self._synthetic_normal_obs()

        return self._obs.copy(), self._build_info(action=None, reward=0.0)

    # ──────────────────────────────────────────
    #  STEP
    # ──────────────────────────────────────────

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"
        self._step_count += 1

        # ── Advance observation ────────────────────────────────────────────
        if self._use_replay:
            self._replay_idx = min(self._replay_idx + 1, len(self._sensor_data) - 1)
            self._obs = self._sensor_data[self._replay_idx].astype(np.float32)
        else:
            self._obs = self._get_live_observation()

        pressure   = float(self._obs[PRESSURE_IDX])
        high_press = pressure >= HIGH_PRESSURE_THRESHOLD
        is_attack  = self._current_label()

        # ── Reward ────────────────────────────────────────────────────────
        reward = self._compute_reward(action, high_press, is_attack)
        self._total_reward += reward

        # ── Operational health ─────────────────────────────────────────────
        if high_press and action == DO_NOTHING:
            # Ignoring overpressure degrades process integrity rapidly
            self._op_health = max(0.0, self._op_health - 0.10)
        elif action == OPEN_SAFETY_VALVE:
            # Correct mitigation — minor process disruption
            self._op_health = max(0.0, self._op_health - 0.01)
        elif action == EMERGENCY_SHUTDOWN:
            # Hard stop — significant operational cost
            self._op_health = max(0.0, self._op_health - 0.15)
        else:
            # Normal operation — gradual recovery
            self._op_health = min(1.0, self._op_health + 0.01)

        terminated = self._op_health <= 0.0
        truncated  = self._step_count >= self.max_steps
        info       = self._build_info(action, reward)

        if self.render_mode == "human":
            self._render_console(action, reward, pressure, high_press)

        return self._obs.copy(), reward, terminated, truncated, info

    # ──────────────────────────────────────────
    #  REWARD FUNCTION
    # ──────────────────────────────────────────

    def _compute_reward(self, action: int, high_press: bool, is_attack: bool) -> float:
        """
        Reward rules:

        Anomalous / high-pressure state:
          DO_NOTHING         → −10   (dangerous: threat ignored)
          OPEN_SAFETY_VALVE  → +20   (correct: pressure relieved)
          EMERGENCY_SHUTDOWN → +5    (works but over-aggressive)

        Normal state:
          DO_NOTHING         → +1    (correct: no action needed)
          any other action   → −2    (unnecessary process disruption)
        """
        if high_press or is_attack:
            if action == DO_NOTHING:        return -10.0
            if action == OPEN_SAFETY_VALVE: return +20.0
            if action == EMERGENCY_SHUTDOWN:return  +5.0
        else:
            if action == DO_NOTHING:        return  +1.0
            return -2.0   # unnecessary action during normal state
        return 0.0

    # ──────────────────────────────────────────
    #  OBSERVATION SOURCES
    # ──────────────────────────────────────────

    def _get_live_observation(self) -> np.ndarray:
        """
        Override to connect to the live SCADA bridge
        (scada_monitor.py → Detective API → this method).
        Default: returns a synthetic observation.
        """
        return self._synthetic_normal_obs()

    def _synthetic_normal_obs(self) -> np.ndarray:
        """
        Synthetic normal-operation observation in the GRFICS operating band.

        Based on CSV statistics (MinMaxScaled):
          pressure (idx 8) : normally ≈ 0.84  (raw ≈ 2683 / 3200)
          f1_valve (idx 0) : normally ≈ 1.0   (always 100% open in normal)
          level    (idx 9) : normally ≈ 0.61  (raw ≈ 44.1 / 75.25)
        """
        obs = np.zeros(N_SENSORS, dtype=np.float32)

        # Set realistic normal-operation baselines
        obs[OBS_IDX["f1_valve"]]   = float(np.random.uniform(0.95, 1.00))
        obs[OBS_IDX["f1_flow"]]    = float(np.random.uniform(0.90, 1.00))
        obs[OBS_IDX["f2_valve"]]   = float(np.random.uniform(0.00, 0.05))
        obs[OBS_IDX["f2_flow"]]    = float(np.random.uniform(0.00, 0.05))
        obs[OBS_IDX["purge_valve"]]= float(np.random.uniform(0.00, 0.50))
        obs[OBS_IDX["purge_flow"]] = float(np.random.uniform(0.00, 0.50))
        obs[OBS_IDX["prod_valve"]] = float(np.random.uniform(0.20, 0.90))
        obs[OBS_IDX["prod_flow"]]  = float(np.random.uniform(0.20, 0.90))
        obs[OBS_IDX["pressure"]]   = float(np.random.uniform(0.80, 0.88))
        obs[OBS_IDX["level"]]      = float(np.random.uniform(0.55, 0.70))
        obs[OBS_IDX["A_purge"]]    = float(np.random.uniform(0.95, 1.00))
        obs[OBS_IDX["B_purge"]]    = float(np.random.uniform(0.28, 0.35))
        obs[OBS_IDX["C_purge"]]    = float(np.random.uniform(0.85, 0.95))

        # Inject high-pressure event with 10% probability
        if random.random() < 0.10:
            obs[PRESSURE_IDX] = float(np.random.uniform(0.91, 0.98))

        return obs

    def _current_label(self) -> bool:
        """True if current timestep is labelled as attack (replay) or high-pressure (live)."""
        if self._labels is not None and self._replay_idx < len(self._labels):
            return bool(self._labels[self._replay_idx])
        return float(self._obs[PRESSURE_IDX]) >= HIGH_PRESSURE_THRESHOLD

    # ──────────────────────────────────────────
    #  INFO DICT
    # ──────────────────────────────────────────

    def _build_info(self, action: Any, reward: float) -> dict:
        return {
            "step":          self._step_count,
            "op_health":     round(self._op_health, 4),
            "pressure":      round(float(self._obs[PRESSURE_IDX]),            4),
            "f1_valve":      round(float(self._obs[OBS_IDX["f1_valve"]]),     4),
            "prod_flow":     round(float(self._obs[OBS_IDX["prod_flow"]]),    4),
            "level":         round(float(self._obs[OBS_IDX["level"]]),        4),
            "action":        action,
            "reward":        reward,
            "total_reward":  round(self._total_reward, 4),
            "high_pressure": float(self._obs[PRESSURE_IDX]) >= HIGH_PRESSURE_THRESHOLD,
            "replay_idx":    self._replay_idx,
        }

    # ──────────────────────────────────────────
    #  RENDERING
    # ──────────────────────────────────────────

    def _render_console(self, action: int, reward: float,
                        pressure: float, high_press: bool):
        names = {DO_NOTHING: "DO_NOTHING       ",
                 OPEN_SAFETY_VALVE: "OPEN_SAFETY_VALVE",
                 EMERGENCY_SHUTDOWN: "EMERG_SHUTDOWN   "}
        icon = "🔴" if high_press else "🟢"
        print(f"[{self._step_count:04d}] {icon} "
              f"P={pressure:.3f} | {names.get(action,'?')} | "
              f"R={reward:+6.1f} | Health={self._op_health:.3f}")

    def render(self):
        if self.render_mode == "human":
            p = float(self._obs[PRESSURE_IDX])
            self._render_console(None, 0.0, p, p >= HIGH_PRESSURE_THRESHOLD)

    def close(self):
        pass


# ─────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== GrficsDefenseEnv Self-Test ===\n")

    # Synthetic mode
    env = GrficsDefenseEnv(max_steps=20, render_mode="human")
    obs, info = env.reset(seed=42)
    print(f"obs shape : {obs.shape}")
    print(f"obs space : {env.observation_space}")
    print(f"act space : {env.action_space}\n")
    print(f"Feature index map: {OBS_IDX}\n")

    total_r = 0.0
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward
        if terminated or truncated:
            break
    print(f"\nEpisode done — total_reward={total_r:.1f}, op_health={info['op_health']}")

    # Replay mode with real feature distribution
    print("\n── Replay mode (high-pressure injection) ──")
    n = 100
    fake_data = np.zeros((n, N_SENSORS), dtype=np.float32)
    fake_data[:, OBS_IDX["f1_valve"]]   = 0.98
    fake_data[:, OBS_IDX["pressure"]]   = 0.84   # normal pressure
    fake_data[:, OBS_IDX["level"]]      = 0.62
    fake_data[40:55, OBS_IDX["pressure"]] = 0.95  # attack: overpressure
    fake_labels = np.zeros(n, dtype=np.int32)
    fake_labels[40:55] = 1

    env2 = GrficsDefenseEnv(sensor_data=fake_data, labels=fake_labels, max_steps=60)
    obs2, _ = env2.reset(seed=0)
    for _ in range(60):
        action = (OPEN_SAFETY_VALVE
                  if obs2[PRESSURE_IDX] >= HIGH_PRESSURE_THRESHOLD
                  else DO_NOTHING)
        obs2, r, terminated, truncated, _ = env2.step(action)
        if terminated or truncated:
            break
    print(f"Replay done — steps={env2._step_count}, health={env2._op_health:.3f}")
    print("✅ Self-test passed.")