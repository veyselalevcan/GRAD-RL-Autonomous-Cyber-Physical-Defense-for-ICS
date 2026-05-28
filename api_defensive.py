"""
api_defensive.py — GRAD-RL Framework (v6.0)
Defensive Node: PPO Autonomous Response + Graph-Based Risk Assessment + XAI

Changes vs v3.0:
  - IncidentReport now accepts `dataset` field (SWaT | WADI)
  - GenericAssetMapper instantiated per-request with correct dataset
  - PPO obs vector updated to 7-dim: [MSE, Risk, Cascade, Op_Health,
    CWDD_est, Dataset_Flag, Pad] — matches scada_env v6.0 training space
  - Dataset_Flag: SWaT=0.0, WADI=1.0
  - API-mode defaults: Op_Health=1.0, CWDD_est=0.0, Pad=0.0
  - Topology mapper instances are LRU-cached per dataset to avoid
    repeated graph construction (betweenness centrality is expensive)
  - Existing HUMAN_IN_THE_LOOP and Safety Gate (Risk ≥ 9.0) logic unchanged
"""

import sys
import os
import numpy as np
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from stable_baselines3 import PPO

# ─────────────────────────────────────────────
#  IMPORT PATH + SIBLING MODULES
# ─────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from topology_manager import GenericAssetMapper
    _TOPOLOGY_AVAILABLE = True
    print("✅ Topology Manager: module found.")
except ImportError as e:
    print(f"❌ Topology Manager import failed: {e}")
    GenericAssetMapper = None
    _TOPOLOGY_AVAILABLE = False

# Actuator layer (graceful no-op if not wired)
try:
    from actuator_controller import block_ip, isolate_zone, system_shutdown, alert_operator
    _ACTUATOR_AVAILABLE = True
except ImportError:
    _ACTUATOR_AVAILABLE = False
    def block_ip():       print("[ACT] block_ip (stub)")
    def isolate_zone():   print("[ACT] isolate_zone (stub)")
    def system_shutdown():print("[ACT] system_shutdown (stub)")
    def alert_operator(): print("[ACT] alert_operator (stub)")

app = FastAPI(title="AIAS Defensive Brain", version="6.0 (Dataset-Agnostic)")

# ─────────────────────────────────────────────
#  PATHS & POLICY CONFIG
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "..", "..", "models")
AGENT_PATH  = os.path.join(MODEL_DIR, "rl_defender_agent.zip")

HUMAN_IN_THE_LOOP  = True
CRITICAL_THRESHOLD = 9.0   # IEC 61511 Safety Gate

# ─────────────────────────────────────────────
#  DATASET FLAGS
# ─────────────────────────────────────────────
DATASET_FLAG = {"SWaT": 0.0, "WADI": 1.0}

# ─────────────────────────────────────────────
#  ACTION MAP
# ─────────────────────────────────────────────
ACTION_MAP = {
    0: "MONITOR",
    1: "BLOCK_IP",
    2: "ISOLATE_ZONE",
    3: "SYSTEM_SHUTDOWN",
    4: "ALERT_OPERATOR",
}

# ─────────────────────────────────────────────
#  TOPOLOGY MANAGER CACHE
#  lru_cache keyed on dataset string avoids repeated
#  nx.betweenness_centrality() computation on every request.
# ─────────────────────────────────────────────
@lru_cache(maxsize=2)   # one entry per dataset
def _get_topology(dataset: str):
    if not _TOPOLOGY_AVAILABLE:
        return None
    try:
        mapper = GenericAssetMapper(dataset=dataset)
        print(f"✅ Topology [{dataset}]: graph built and cached.")
        return mapper
    except Exception as e:
        print(f"❌ Topology [{dataset}] init failed: {e}")
        return None


# ─────────────────────────────────────────────
#  RL AGENT LOAD
# ─────────────────────────────────────────────
rl_agent = None
try:
    if os.path.exists(AGENT_PATH):
        rl_agent = PPO.load(AGENT_PATH)
        print("✅ RL Agent: Loaded Successfully.")
    else:
        print(f"⚠️  RL Agent not found at {AGENT_PATH}. Rule-based fallback active.")
except Exception as e:
    print(f"❌ RL Loading Error: {e}")


# ─────────────────────────────────────────────
#  PYDANTIC MODELS
# ─────────────────────────────────────────────
class IncidentReport(BaseModel):
    is_anomaly:  bool
    mse_loss:    float
    attack_type: str = "Unknown"
    swat_tag:    str = "Unknown"   # culprit asset tag (SWaT or WADI stage label)
    dataset:     str = "SWaT"      # "SWaT" | "WADI"
    # Optional pre-computed PhD metric signals (forwarded from training loop)
    op_health:   float = 1.0       # Operational Health [0, 1]
    cwdd_est:    float = 0.0       # CWDD signal (0 = no delay observed)


class DefenseResponse(BaseModel):
    action:           str
    confidence:       float
    final_rationale:  str
    risk_score:       float
    dataset:          str
    # XAI fields
    xai_explanation:  str
    mitigation_plan:  List[str]
    propagation_path: List[str]
    # PhD metrics reflected in response
    obs_vector:       List[float]   # 7-dim vector sent to PPO (transparency)
    is_bridge_node:   bool


# ─────────────────────────────────────────────
#  DECISION ENDPOINT
# ─────────────────────────────────────────────
@app.post("/decide", response_model=DefenseResponse)
def decide(incident: IncidentReport):
    dataset     = incident.dataset if incident.dataset in ("SWaT", "WADI") else "SWaT"
    ds_flag     = DATASET_FLAG.get(dataset, 0.0)

    # ── 1. Normal operation fast-path ─────────────────────────────────────
    if not incident.is_anomaly:
        obs_vec = [incident.mse_loss, 0.0, 0.0,
                   incident.op_health, incident.cwdd_est, ds_flag, 0.0]
        return DefenseResponse(
            action="MONITOR",
            confidence=1.0,
            final_rationale="System operating within normal parameters.",
            risk_score=0.0,
            dataset=dataset,
            xai_explanation="No active threats detected.",
            mitigation_plan=[],
            propagation_path=[],
            obs_vector=obs_vec,
            is_bridge_node=False,
        )

    # ── 2. Topology / Graph Risk Assessment ───────────────────────────────
    topology = _get_topology(dataset)

    if topology is not None:
        try:
            impact      = topology.assess_impact(incident.swat_tag, incident.attack_type)
            risk_score  = impact["risk_score"]
            xai_text    = impact["explanation"]
            mitigation  = impact.get("mitigation", [])
            path        = impact.get("propagation_path", [])
            cascade_cnt = float(len(path))              # proxy for cascade size
            is_bridge   = impact.get("is_bridge_node", False)
        except Exception as e:
            print(f"⚠️  Topology assess_impact failed: {e}")
            risk_score, xai_text, mitigation, path = 5.0, "Topology error.", [], []
            cascade_cnt, is_bridge = 0.0, False
    else:
        risk_score  = min(incident.mse_loss * 10.0, 10.0)  # rough heuristic
        xai_text    = "Topology Module Unavailable. Risk estimated from MSE."
        mitigation  = ["Manual operator review required"]
        path        = []
        cascade_cnt = 0.0
        is_bridge   = False

    # ── 3. Build 7-dim PPO observation ────────────────────────────────────
    #
    #  Index | Feature        | Source
    #  ------+----------------+---------------------------------------------
    #    0   | MSE_Loss       | incident.mse_loss  (LSTM-AE output)
    #    1   | Risk_Score     | topology.assess_impact → risk_score
    #    2   | Cascade_Count  | len(propagation_path)
    #    3   | Op_Health      | incident.op_health (default 1.0 for API mode)
    #    4   | CWDD_est       | incident.cwdd_est  (default 0.0 for API mode)
    #    5   | Dataset_Flag   | 0.0 = SWaT, 1.0 = WADI
    #    6   | Pad            | 0.0 (reserved for future features)
    #
    obs = np.array([
        incident.mse_loss,
        risk_score,
        cascade_cnt,
        incident.op_health,   # 1.0 at inference time (stateless API)
        incident.cwdd_est,    # 0.0 unless training loop pushes a value
        ds_flag,
        0.0,                  # Pad
    ], dtype=np.float32)

    # ── 4. RL Agent / Rule-Based Decision ────────────────────────────────
    if rl_agent is not None:
        try:
            action_idx, _ = rl_agent.predict(obs, deterministic=True)
            action_idx    = int(action_idx)
            confidence    = 0.95
        except Exception as e:
            print(f"⚠️  PPO predict error: {e}. Falling back to rule-based.")
            action_idx, confidence = _rule_based(risk_score)
    else:
        action_idx, confidence = _rule_based(risk_score)

    ai_action    = ACTION_MAP.get(action_idx, "MONITOR")
    final_action = ai_action
    prefix       = "Autonomous"

    # ── 5. Human-in-the-Loop Safety Gate (IEC 61511) ─────────────────────
    if HUMAN_IN_THE_LOOP:
        if risk_score >= CRITICAL_THRESHOLD or ai_action == "SYSTEM_SHUTDOWN":
            final_action = "REQUIRE_HUMAN_APPROVAL"
            prefix       = "CRITICAL ADVISORY"
            xai_text    += (
                f"\n\n⚠️ **SAFETY GATE [{dataset}]:** Risk ({risk_score:.2f}) ≥ "
                f"{CRITICAL_THRESHOLD}. Autonomous shutdown blocked — awaiting "
                f"operator authorisation (IEC 61511 SIL compliance)."
            )

    # ── 6. Actuate ────────────────────────────────────────────────────────
    if final_action != "REQUIRE_HUMAN_APPROVAL":
        _actuate(final_action)

    # ── 7. Compose response ───────────────────────────────────────────────
    rationale = (
        f"{prefix} [{dataset}]: '{ai_action}' selected. "
        f"Risk={risk_score:.2f}/10.0, "
        f"Cascade={int(cascade_cnt)} nodes, "
        f"Bridge={'Yes' if is_bridge else 'No'}."
    )

    return DefenseResponse(
        action=final_action,
        confidence=confidence,
        final_rationale=rationale,
        risk_score=risk_score,
        dataset=dataset,
        xai_explanation=xai_text,
        mitigation_plan=mitigation,
        propagation_path=path,
        obs_vector=obs.tolist(),
        is_bridge_node=is_bridge,
    )


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def _rule_based(risk_score: float) -> tuple[int, float]:
    """Deterministic rule-based fallback when PPO agent is unavailable."""
    if risk_score >= CRITICAL_THRESHOLD: return 3, 0.50   # SHUTDOWN
    if risk_score >= 7.0:                return 2, 0.50   # ISOLATE_ZONE
    if risk_score >= 4.0:                return 1, 0.50   # BLOCK_IP
    return 4, 0.50                                        # ALERT_OPERATOR


def _actuate(action: str):
    """Dispatch to actuator layer."""
    dispatch = {
        "BLOCK_IP":        block_ip,
        "ISOLATE_ZONE":    isolate_zone,
        "SYSTEM_SHUTDOWN": system_shutdown,
        "ALERT_OPERATOR":  alert_operator,
    }
    fn = dispatch.get(action)
    if fn:
        fn()


# ─────────────────────────────────────────────
#  UTILITY ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":         "ok",
        "rl_agent":       rl_agent is not None,
        "topology_swat":  _get_topology.cache_info().currsize > 0,
        "hitl_enabled":   HUMAN_IN_THE_LOOP,
        "safety_gate":    CRITICAL_THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

# uvicorn src.api.api_defensive:app --reload --port 8001