"""
api_defensive.py — GRAD-RL Framework (v7.0)
Defensive Node: PPO Autonomous Response + CVSS 4.0 Risk Scoring + XAI
GRFICS Digital Twin Edition

Responsibilities:
  1. Receive anomaly payload from Detective Node (POST /decide)
  2. Score attack severity with CVSS 4.0 base scores (MITRE-mapped)
  3. Feed 13 scaled sensor features directly to PPO agent → action
  4. Apply IEC 61511 Safety Gate:
       CVSS >= 7.0  OR  RL Action == EMERGENCY_SHUTDOWN
       → final_action = REQUIRE_HUMAN_APPROVAL
  5. Return structured XAI explanation for every decision

PPO Observation Space: Box(13,) — exact same 13 scaled GRFICS features
  forwarded by the Detective Node.

Action Space: Discrete(3)
  0: DO_NOTHING
  1: OPEN_SAFETY_VALVE
  2: EMERGENCY_SHUTDOWN  (always requires human approval)

CVSS 4.0 Base Scores (MITRE ATT&CK for ICS):
  T0813            : 6.5   — Denial of Control
  T0813+T0836      : 9.5   — Combined (worst-case)
  T0814            : 5.0   — Modbus Flood
  T0836            : 9.5   — Setpoint Poisoning  (Critical)
  T0856            : 7.5   — Sensor Spoofing
  Unknown/default  : 7.0   — Assume High on unknown

Run locally:
    uvicorn api_defensive:app --reload --port 8001
"""

import os

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from stable_baselines3 import PPO

app = FastAPI(
    title="GRAD-RL Defensive Node",
    version="7.0 (GRFICS)",
    description="PPO autonomous defense with CVSS 4.0 risk scoring and IEC 61511 safety gate",
)

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models"))
AGENT_PATH = os.path.join(MODEL_DIR, "rl_defender_grfics.zip")

# ─────────────────────────────────────────────
#  CVSS 4.0 BASE SCORE TABLE
#  Scores sourced from MITRE ATT&CK for ICS
#  technique impact assessments.
#
#  Rationale per technique:
#    T0813 (Denial of Control):
#      AV:N / AC:L / AT:N / PR:L / UI:N / VC:N / VI:H / VA:H / SC:N / SI:N / SA:L
#      → CVSS 4.0 Base: 6.5 (High) — disrupts control but no direct safety impact
#
#    T0836 (Setpoint Poisoning):
#      AV:N / AC:L / AT:N / PR:L / UI:N / VC:N / VI:H / VA:H / SC:H / SI:H / SA:H
#      → CVSS 4.0 Base: 9.5 (Critical) — physical process parameters altered;
#        can cause overpressure, equipment damage, or safety incidents
#
#    T0856 (Sensor Spoofing):
#      AV:N / AC:H / AT:N / PR:L / UI:N / VC:H / VI:H / VA:N / SC:N / SI:L / SA:L
#      → CVSS 4.0 Base: 7.5 (High) — corrupts situational awareness; operator
#        and automated safety responses may act on false data
#
#    T0814 (Modbus Flood):
#      AV:N / AC:L / AT:N / PR:N / UI:N / VC:N / VI:N / VA:H / SC:N / SI:N / SA:N
#      → CVSS 4.0 Base: 5.0 (Medium) — availability impact only; no data/safety
#        integrity compromise
#
#    T0813+T0836 (Combined Campaign):
#      Worst-case of constituent techniques; inherits T0836 critical score.
#      → CVSS 4.0 Base: 9.5 (Critical)
# ─────────────────────────────────────────────
CVSS_TABLE: dict[str, float] = {
    "T0813":        6.5,
    "T0813+T0836":  9.5,
    "T0814":        5.0,
    "T0836":        9.5,
    "T0856":        7.5,
}
CVSS_DEFAULT        = 7.0   # conservative: assume High on unknown attack type
CVSS_GATE_THRESHOLD = 7.0   # IEC 61511: require human approval above this

# ─────────────────────────────────────────────
#  CVSS SEVERITY LABELS  (CVSS 4.0 ranges)
# ─────────────────────────────────────────────
def _cvss_severity(score: float) -> str:
    if score >= 9.0: return "Critical"
    if score >= 7.0: return "High"
    if score >= 4.0: return "Medium"
    if score >  0.0: return "Low"
    return "None"


# ─────────────────────────────────────────────
#  CVSS LOOKUP
# ─────────────────────────────────────────────
def calculate_cvss(attack_type: str) -> float:
    """
    Return CVSS 4.0 base score for a MITRE ATT&CK for ICS technique.

    Accepts both exact MITRE IDs (e.g. 'T0836') and human-readable labels
    produced by the XGBoost classifier (e.g. 'Attack_SetpointPoisoning').
    Falls back to CVSS_DEFAULT (7.0) for unknown attack types.
    """
    # Direct lookup
    if attack_type in CVSS_TABLE:
        return CVSS_TABLE[attack_type]

    # Label → technique heuristic
    label_lower = attack_type.lower()
    if "setpoint" in label_lower or "poisoning" in label_lower:
        return CVSS_TABLE["T0836"]
    if "denial" in label_lower or "doc" in label_lower or "t0813" in label_lower:
        return CVSS_TABLE["T0813"]
    if "spoofing" in label_lower or "sensor" in label_lower:
        return CVSS_TABLE["T0856"]
    if "flood" in label_lower or "modbus" in label_lower or "t0814" in label_lower:
        return CVSS_TABLE["T0814"]
    if "combined" in label_lower:
        return CVSS_TABLE["T0813+T0836"]
    if attack_type == "Normal":
        return 0.0

    return CVSS_DEFAULT


# ─────────────────────────────────────────────
#  ACTION MAP  (Discrete(3) — matches grfics_env.py)
# ─────────────────────────────────────────────
ACTION_MAP = {
    0: "DO_NOTHING",
    1: "OPEN_SAFETY_VALVE",
    2: "EMERGENCY_SHUTDOWN",
}

ACTION_DESC = {
    "DO_NOTHING":          "Continue monitoring; no physical intervention required.",
    "OPEN_SAFETY_VALVE":   "Activate safety relief valve to reduce process pressure.",
    "EMERGENCY_SHUTDOWN":  "Halt production pump immediately; requires operator confirmation.",
}

# ─────────────────────────────────────────────
#  PPO AGENT LOADING
# ─────────────────────────────────────────────
rl_agent = None
try:
    if os.path.exists(AGENT_PATH):
        rl_agent = PPO.load(AGENT_PATH)
        print(f"✅ PPO Agent loaded: {AGENT_PATH}")
    else:
        print(f"⚠️  PPO Agent not found at {AGENT_PATH}. Rule-based fallback active.")
except Exception as e:
    print(f"❌ PPO load error: {e}. Rule-based fallback active.")


# ─────────────────────────────────────────────
#  PYDANTIC MODELS
# ─────────────────────────────────────────────
class AnomalyPayload(BaseModel):
    """
    Forwarded by the Detective Node after anomaly detection + classification.

    scaled_features: 13 MinMaxScaled GRFICS sensor values.
      These are fed directly to the PPO agent as its observation vector.
    """
    is_anomaly:      bool
    mse_loss:        float
    attack_type:     str   = "Unknown"
    culprit_feature: str   = "Unknown"
    scaled_features: List[float]   # exactly 13 values


class DefenseResponse(BaseModel):
    rl_action:        str            # DO_NOTHING | OPEN_SAFETY_VALVE | EMERGENCY_SHUTDOWN
    final_action:     str            # rl_action or REQUIRE_HUMAN_APPROVAL
    cvss_score:       float
    cvss_severity:    str
    xai_explanation:  str            # structured human-readable XAI report
    iec61511_policy:  str            # "Autonomous Execution" or "Human Approval Required"
    obs_vector:       List[float]    # 13-dim vector sent to PPO (transparency)
    action_rationale: str


# ─────────────────────────────────────────────
#  RULE-BASED FALLBACK
# ─────────────────────────────────────────────
def _rule_based_action(cvss: float) -> int:
    """Deterministic fallback when PPO agent is unavailable."""
    if cvss >= 9.0: return 2   # EMERGENCY_SHUTDOWN
    if cvss >= 6.0: return 1   # OPEN_SAFETY_VALVE
    return 0                   # DO_NOTHING


# ─────────────────────────────────────────────
#  DECISION ENDPOINT
# ─────────────────────────────────────────────
@app.post("/decide", response_model=DefenseResponse)
def decide(payload: AnomalyPayload):

    # ── 1. Normal fast-path ───────────────────────────────────────────────
    if not payload.is_anomaly:
        obs = np.array(payload.scaled_features[:13], dtype=np.float32)
        if len(obs) < 13:
            obs = np.pad(obs, (0, 13 - len(obs)))
        return DefenseResponse(
            rl_action="DO_NOTHING",
            final_action="DO_NOTHING",
            cvss_score=0.0,
            cvss_severity="None",
            xai_explanation=(
                "GRAD-RL XAI Report\n"
                "──────────────────\n"
                "Status       : No anomaly detected\n"
                "MSE Loss     : {:.6f} (below threshold)\n"
                "Policy       : Normal operation — no intervention required.".format(
                    payload.mse_loss
                )
            ),
            iec61511_policy="Autonomous Execution",
            obs_vector=obs.tolist(),
            action_rationale="System within normal operating parameters.",
        )

    # ── 2. CVSS 4.0 Risk Scoring ─────────────────────────────────────────
    cvss     = calculate_cvss(payload.attack_type)
    severity = _cvss_severity(cvss)

    # ── 3. PPO Observation → Action ───────────────────────────────────────
    # The GRFICS PPO agent was trained on Box(13,) — the same 13 scaled sensor
    # features produced by the Detective Node's scaler. Feed them directly.
    obs = np.array(payload.scaled_features[:13], dtype=np.float32)
    if len(obs) < 13:
        obs = np.pad(obs, (0, 13 - len(obs)))

    if rl_agent is not None:
        try:
            action_idx, _ = rl_agent.predict(obs, deterministic=True)
            action_idx    = int(action_idx)
            method        = "PPO RL Agent"
        except Exception as e:
            print(f"⚠️  PPO predict error: {e}. Using rule-based fallback.")
            action_idx = _rule_based_action(cvss)
            method     = "Rule-based fallback"
    else:
        action_idx = _rule_based_action(cvss)
        method     = "Rule-based fallback (PPO unavailable)"

    rl_action     = ACTION_MAP.get(action_idx, "DO_NOTHING")
    action_detail = ACTION_DESC.get(rl_action, "")

    # ── 4. IEC 61511 Safety Gate ──────────────────────────────────────────
    # Condition A: CVSS score >= 7.0 (High / Critical)
    # Condition B: RL agent recommends EMERGENCY_SHUTDOWN (irreversible)
    # Either condition blocks autonomous execution and escalates to operator.
    gate_triggered = (cvss >= CVSS_GATE_THRESHOLD) or (action_idx == 2)

    if gate_triggered:
        final_action  = "REQUIRE_HUMAN_APPROVAL"
        iec_policy    = "Human Approval Required"
        gate_reason   = (
            f"CVSS {cvss} >= {CVSS_GATE_THRESHOLD} ({severity})"
            if cvss >= CVSS_GATE_THRESHOLD
            else "RL agent recommended EMERGENCY_SHUTDOWN (irreversible action)"
        )
    else:
        final_action  = rl_action
        iec_policy    = "Autonomous Execution"
        gate_reason   = f"CVSS {cvss} < {CVSS_GATE_THRESHOLD} and action is reversible"

    # ── 5. XAI Report ─────────────────────────────────────────────────────
    xai = (
        "GRAD-RL XAI Decision Report\n"
        "════════════════════════════════════════════════\n"
        f"Detected Attack    : {payload.attack_type}\n"
        f"Culprit Feature    : {payload.culprit_feature}  "
        f"(MSE Loss = {payload.mse_loss:.6f})\n"
        "\n"
        "CVSS 4.0 Risk Assessment\n"
        "────────────────────────\n"
        f"  Base Score       : {cvss:.1f} / 10.0\n"
        f"  Severity Level   : {severity}\n"
        f"  Technique Basis  : MITRE ATT&CK for ICS ({payload.attack_type})\n"
        "\n"
        "RL Agent Recommendation\n"
        "────────────────────────\n"
        f"  Method           : {method}\n"
        f"  Recommended      : {rl_action}\n"
        f"  Action Detail    : {action_detail}\n"
        "\n"
        "IEC 61511 Safety Gate\n"
        "────────────────────────\n"
        f"  Gate Triggered   : {'Yes' if gate_triggered else 'No'}\n"
        f"  Reason           : {gate_reason}\n"
        f"  Final Policy     : {iec_policy}\n"
        f"  Final Action     : {final_action}\n"
        "════════════════════════════════════════════════"
    )

    rationale = (
        f"{iec_policy}: '{rl_action}' "
        f"[CVSS {cvss}/{severity}] "
        f"via {method}."
    )

    return DefenseResponse(
        rl_action=rl_action,
        final_action=final_action,
        cvss_score=cvss,
        cvss_severity=severity,
        xai_explanation=xai,
        iec61511_policy=iec_policy,
        obs_vector=obs.tolist(),
        action_rationale=rationale,
    )


# ─────────────────────────────────────────────
#  UTILITY ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":             "ok",
        "ppo_agent_loaded":   rl_agent is not None,
        "action_space":       ACTION_MAP,
        "cvss_table":         CVSS_TABLE,
        "iec61511_gate":      CVSS_GATE_THRESHOLD,
        "obs_dimensions":     13,
    }


@app.get("/cvss/{attack_type}")
def cvss_lookup(attack_type: str):
    """Look up CVSS score and severity for a MITRE technique ID."""
    score    = calculate_cvss(attack_type)
    severity = _cvss_severity(score)
    return {
        "attack_type": attack_type,
        "cvss_score":  score,
        "severity":    severity,
        "gate":        score >= CVSS_GATE_THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

# uvicorn api_defensive:app --reload --port 8001
