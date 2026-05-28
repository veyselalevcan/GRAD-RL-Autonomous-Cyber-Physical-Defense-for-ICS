"""
scada_env.py — GRAD-RL Framework (v6.0)
Custom Gymnasium Environment for ICS Autonomous Defense

Changes vs v5.0:
  - EpisodeMetrics integrated into step() and reset()
  - Resilience Bonus in reward function (mitigate without shutdown)
  - step() info dict now exports CWDD, RuA, PLT per step
  - under_attack flag derived from risk_score heuristic (no GT label needed at runtime)
  - dataset parameter forwarded to TopologyManager
  - Latency guard: metrics ops are O(1) per step, meeting <50 ms constraint
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time

from metrics import EpisodeMetrics

# ─────────────────────────────────────────────
#  Action cost fraction for health calculation
#  (must match metrics.ACTION_HEALTH_COST)
# ─────────────────────────────────────────────
ACTION_COST_FRACTION = {0: 0.00, 1: 0.05, 2: 0.30, 3: 1.00, 4: 0.02}


class ScadaDefensiveEnv(gym.Env):
    """
    PhD-Grade Custom Environment for Cyber-Physical System Defense.

    Objective:
        The RL agent must balance Security Risk vs Operational Continuity.
        Penalised heavily for false positives (unnecessary shutdowns) and
        for missed critical threats.

    Action Space (Discrete 5):
        0: MONITOR        — Cost: 0       (Do nothing)
        1: BLOCK_IP       — Cost: −2      (Minimal disruption)
        2: ISOLATE_ZONE   — Cost: −5      (Moderate disruption)
        3: SYSTEM_SHUTDOWN— Cost: −100    (Last resort)
        4: ALERT_OPERATOR — Cost: −1      (Human-in-the-loop)

    Observation Space (Box 7):
        [MSE_Loss, Risk_Score, Cascade_Impact, Op_Health, CWDD_est,
         Dataset_Flag, Placeholder]
        Note: shape expanded from (5,) to (7,) to include new metric signals.
        Dataset_Flag: 0.0 = SWaT, 1.0 = WADI

    Metrics tracked per episode (exported in info dict):
        - CWDD  : Criticality-Weighted Detection Delay
        - RuA   : Resilience-Under-Attack (fraction of steps health ≥ 0.9)
        - PLT   : Proactive Lead-Time (steps before safety gate)
        - Op_Health: Current operational health [0, 1]

    XAI Fidelity is tracked externally (requires GT labels from dataset);
    use EpisodeMetrics.record_xai() from your training loop.
    """

    metadata = {"render_modes": []}

    # Betweenness centrality estimates per dataset (pre-computed defaults).
    # Override by passing a centrality dict at construction.
    DEFAULT_CENTRALITY_SWAT = {
        "Stage_P1": 0.05, "Stage_P2": 0.10, "Stage_P3": 0.15,
        "Stage_P4": 0.20, "Stage_P5": 0.25, "Stage_P6": 0.10,
        "PLC_P1": 0.30,   "PLC_P2": 0.40,   "PLC_P3": 0.50,
        "PLC_P4": 0.55,   "PLC_P5": 0.60,   "PLC_P6": 0.35,
        "HMI_Main": 0.80,
    }
    DEFAULT_CENTRALITY_WADI = {
        "SUPPLY_HEADER":     0.70,  # bridge node (single supply path)
        "SUPPLY_PUMP":       0.65,
        "DISTRIBUTION_MAIN": 0.85,  # highest: all flow passes through
        "DISTRIBUTION_VALVE":0.55,
        "CONSUMER_TANK_A":   0.20,
        "CONSUMER_TANK_B":   0.20,
        "PRESSURE_ZONE":     0.30,
        "QUALITY_SENSOR":    0.50,
        "RETURN_PUMP":       0.40,
        "TELEMETRY":         0.60,
        "REMOTE_IO":         0.45,
        "HMI_Main":          0.90,
    }

    def __init__(
        self,
        dataset: str = "SWaT",
        centrality: dict | None = None,
        episode_steps: int = 100,
    ):
        """
        Args:
            dataset:        "SWaT" or "WADI" — controls centrality defaults
                            and dataset_flag in the observation.
            centrality:     Optional override. Pass topology_manager.centrality
                            for accurate betweenness values.
            episode_steps:  Steps per episode (default 100).
        """
        super().__init__()

        self.dataset        = dataset
        self.episode_steps  = episode_steps

        # Select centrality table
        if centrality is not None:
            self._centrality = centrality
        elif dataset == "WADI":
            self._centrality = self.DEFAULT_CENTRALITY_WADI
        else:
            self._centrality = self.DEFAULT_CENTRALITY_SWAT

        self._dataset_flag = 1.0 if dataset == "WADI" else 0.0

        # ── Action Space ──────────────────────────────────────────
        self.action_space = spaces.Discrete(5)

        # ── Observation Space ─────────────────────────────────────
        # [MSE, Risk, Cascade, Op_Health, CWDD_signal, Dataset_flag, Pad]
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(7,), dtype=np.float32
        )

        # ── Internal State ────────────────────────────────────────
        self.state:         np.ndarray = None
        self.steps_done:    int = 0
        self.op_health:     float = 1.0
        self._attack_start: int | None = None
        self._detected:     bool = False

        # ── PhD Metrics ───────────────────────────────────────────
        self.metrics = EpisodeMetrics(self._centrality)

        # Node pool for simulated attack attribution
        self._node_pool = list(self._centrality.keys())

    # ──────────────────────────────────────────
    #  RESET
    # ──────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.metrics.reset()
        self.steps_done    = 0
        self.op_health     = 1.0
        self._attack_start = None
        self._detected     = False

        # Baseline state: tiny MSE, zero risk, full health
        self.state = np.array(
            [0.04, 0.0, 0.0, 1.0, 0.0, self._dataset_flag, 0.0],
            dtype=np.float32,
        )
        return self.state, {}

    # ──────────────────────────────────────────
    #  STEP
    # ──────────────────────────────────────────

    def step(self, action: int):
        t0 = time.perf_counter()  # latency guard

        self.steps_done += 1
        current_mse   = float(self.state[0])
        current_risk  = float(self.state[1])
        cascade       = float(self.state[2])

        # ── 1. Attack / Normal determination (heuristic without GT) ──
        under_attack = current_risk >= 4.0

        # ── 2. CWDD: mark attack start and simulate detection event ──
        if under_attack and self._attack_start is None:
            self._attack_start = self.steps_done
            # Attribute to a node proportional to risk (random from pool)
            attacked_node = random.choice(self._node_pool)
            self.metrics.mark_attack_start(self.steps_done, attacked_node)
            self._detected = False

        # Detection event: LSTM-AE crosses threshold → MSE proxy > 0.05
        if under_attack and not self._detected and current_mse > 0.05:
            self.metrics.mark_detection(self.steps_done)
            self._detected = True

        if not under_attack:
            self._attack_start = None
            self._detected     = False

        # ── 3. RuA & PLT metric updates ──────────────────────────
        self.metrics.step(action, current_risk, under_attack, self.steps_done)

        # ── 4. Operational Health update ─────────────────────────
        action_cost        = ACTION_COST_FRACTION.get(action, 0.0)
        residual_damage    = 0.0
        if under_attack and action not in (1, 2, 3):
            residual_damage = (current_risk / 10.0) * 0.05
        self.op_health = max(0.0, min(
            1.0,
            self.op_health - action_cost - residual_damage + (0.02 if not under_attack else 0.0)
        ))

        # ── 5. REWARD FUNCTION ────────────────────────────────────
        reward = self._compute_reward(action, current_risk, cascade)

        # ── 6. Next State ─────────────────────────────────────────
        self.state = self._get_random_state()

        # ── 7. Build info dict ────────────────────────────────────
        info = {
            "op_health":    round(self.op_health, 4),
            "under_attack": under_attack,
            "dataset":      self.dataset,
            # Live metric snapshots (full summary at episode end via metrics.summary())
            "cwdd_current": round(self.metrics.cwdd.get_cwdd(), 4),
            "rua_current":  self.metrics.rua.get_rua(),
            "plt_steps":    self.metrics.plt.get_plt_steps(),
        }

        # Latency check (log if > 10 ms — metric overhead budget)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 10:
            info["metric_latency_ms"] = round(elapsed_ms, 2)

        terminated = self.steps_done >= self.episode_steps
        if terminated:
            info["episode_metrics"] = self.metrics.summary()

        return self.state, reward, terminated, False, info

    # ──────────────────────────────────────────
    #  REWARD FUNCTION  (v6.0 with Resilience Bonus)
    # ──────────────────────────────────────────

    def _compute_reward(self, action: int, risk: float, cascade: float) -> float:
        """
        Base reward logic (v5.0 risk tiers) augmented with:

        Resilience Bonus (+15):
            Awarded when the agent successfully mitigates a Medium/High risk
            (risk ∈ [4, 9)) without triggering SYSTEM_SHUTDOWN (action ≠ 3).
            Models the ICS requirement to preserve operational continuity.

        Cascade Penalty (−cascade × 2):
            Proportional penalty for number of downstream assets affected.
            Encourages early intervention before propagation.

        Health Coupling (±health × 5):
            Ties reward to operational health state, incentivising the agent
            to maintain uptime even under attack.
        """
        reward: float = 0.0

        # Base penalty: high risk is always costly
        reward -= risk * 0.5

        # ── Risk-tier action evaluation ──────────────────────────
        if risk < 4.0:
            if action == 0:   reward += 10          # correct: MONITOR
            elif action == 4: reward -= 2            # minor annoyance
            else:             reward -= 10           # false positive

        elif risk < 7.0:  # Medium
            if action == 1:   reward += 15           # ideal: BLOCK_IP
            elif action == 4: reward += 10           # safe choice
            elif action == 0: reward -= 10           # negligence
            elif action == 2: reward -= 5            # overreaction
            elif action == 3: reward -= 50           # major overreaction

        elif risk < 9.0:  # High
            if action == 2:   reward += 25           # ideal: ISOLATE_ZONE
            elif action == 4: reward += 15           # acceptable
            elif action == 0: reward -= 50           # dangerous negligence
            elif action == 3: reward -= 20           # too aggressive (if isolation works)

        else:             # Critical (≥9)
            if action == 3:   reward += 50           # ideal: SHUTDOWN
            elif action == 4: reward += 20           # call for help
            elif action == 2: reward -= 10           # insufficient
            elif action == 0: reward -= 100          # catastrophic failure

        # ── Resilience Bonus ─────────────────────────────────────
        # Awarded when agent handles elevated risk without full shutdown
        if 4.0 <= risk < 9.0 and action in (1, 2, 4):
            reward += 15.0  # preserve continuity bonus

        # ── Cascade Penalty ──────────────────────────────────────
        reward -= cascade * 2.0

        # ── Health Coupling ───────────────────────────────────────
        # Positive when healthy, negative when health degraded
        reward += (self.op_health - 0.5) * 5.0

        return reward

    # ──────────────────────────────────────────
    #  STATE GENERATOR
    # ──────────────────────────────────────────

    def _get_random_state(self) -> np.ndarray:
        """
        Simulates diverse threat scenarios.
        80 % normal, 20 % attack.  Includes correlated MSE/risk for realism.
        """
        if random.random() < 0.8:
            risk    = np.random.uniform(0.0, 3.0)
            mse     = np.random.uniform(0.0, 0.05)
            cascade = 0.0
        else:
            risk    = np.random.uniform(4.0, 10.0)
            mse     = np.random.uniform(0.05, 1.0)
            cascade = float(np.random.randint(1, 10))

        # CWDD signal: normalised current CWDD (capped at 10 for obs range)
        cwdd_signal = min(self.metrics.cwdd.get_cwdd(), 10.0)

        return np.array(
            [mse, risk, cascade, self.op_health, cwdd_signal, self._dataset_flag, 0.0],
            dtype=np.float32,
        )

    # ──────────────────────────────────────────
    #  UTILITY
    # ──────────────────────────────────────────

    def get_episode_summary(self) -> dict:
        """Call at episode end to retrieve full PhD metrics summary."""
        return self.metrics.summary()

    def switch_dataset(self, dataset: str, centrality: dict | None = None):
        """
        Hot-swap between SWaT and WADI without re-instantiating.
        Useful for curriculum training (train on SWaT, evaluate on WADI).
        """
        self.dataset       = dataset
        self._dataset_flag = 1.0 if dataset == "WADI" else 0.0
        if centrality is not None:
            self._centrality = centrality
        elif dataset == "WADI":
            self._centrality = self.DEFAULT_CENTRALITY_WADI
        else:
            self._centrality = self.DEFAULT_CENTRALITY_SWAT
        self.metrics = EpisodeMetrics(self._centrality)
        self._node_pool = list(self._centrality.keys())