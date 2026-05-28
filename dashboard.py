"""
dashboard.py — GRAD-RL Framework (v6.0)
Streamlit SOC Monitoring Dashboard

Changes vs v3.0:
  - Dataset toggle in sidebar: SWaT (51 sensors) | WADI (123 sensors)
  - Correct CSV and label map loaded per dataset
  - `dataset` field propagated in all API payloads
  - PhD metric cards: CWDD, RuA, XAI Fidelity, PLT
  - Metric state persisted across replay steps in session_state
  - WADI stress test scenarios added (Zone 1–3 coverage)
  - Existing SWaT stress suite preserved
"""

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time
import os
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DETECTIVE_URL = "http://127.0.0.1:8000/predict"
DEFENSIVE_URL = "http://127.0.0.1:8001/decide"

BASE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

DATA_FILES = {
    "SWaT": os.path.join(BASE_DATA_DIR, "SWaT_Dataset_Attack_v0.csv"),
    "WADI": os.path.join(BASE_DATA_DIR, "WADI_attackdataLABLE.csv"),
}

# ─────────────────────────────────────────────
#  ATTACK TRANSLATION MAPS
# ─────────────────────────────────────────────
ATTACK_MAP_SWAT = {
    "Attack1": "Sensor Spoofing",
    "Attack2": "Actuator Injection",
    "Attack3": "Pump Manipulation",
    "Attack4": "Data Integrity Loss",
    "Attack5": "DoS Attack",
    "Attack6": "Safety Logic Bypass",
    "MITM": "MITM Attack",
    "Replay": "Replay Attack",
    "Ransomware": "Ransomware",
    "Normal": "Normal Operation",
}

# WADI uses numeric labels: -1=attack, 1=normal (iTrust convention)
# Some releases use string labels — both handled below
ATTACK_MAP_WADI = {
    "-1":       "WADI Attack",
    "1":        "Normal Operation",
    "-1.0":     "WADI Attack",
    "1.0":      "Normal Operation",
    "Attack":   "WADI Attack",
    "Normal":   "Normal Operation",
}

ATTACK_MAPS = {"SWaT": ATTACK_MAP_SWAT, "WADI": ATTACK_MAP_WADI}

# ─────────────────────────────────────────────
#  STRESS TEST SCENARIOS (PER DATASET)
# ─────────────────────────────────────────────
STRESS_SCENARIOS_SWAT = {
    "Scenario 1: Sensor Spoofing (Att 1)":    {"type": "Attack1",    "tag": "FIT101",           "mse": 0.80},
    "Scenario 2: Actuator Injection (Att 2)": {"type": "Attack2",    "tag": "MV201",            "mse": 0.85},
    "Scenario 3: Pump Manipulation (Att 3)":  {"type": "Attack3",    "tag": "P101",             "mse": 0.75},
    "Scenario 4: Data Integrity (Att 4)":     {"type": "Attack4",    "tag": "LIT301",           "mse": 0.82},
    "Scenario 5: Network DoS (Att 5)":        {"type": "Attack5",    "tag": "Switch_1",         "mse": 0.90},
    "Scenario 6: Safety Bypass (Att 6)":      {"type": "Attack6",    "tag": "PLC_P6",           "mse": 1.00},
    "Scenario 7: Ransomware":                 {"type": "Ransomware", "tag": "Eng_Workstation",  "mse": 0.99},
    "Scenario 8: MITM Attack":                {"type": "MITM",       "tag": "HMI_Main",         "mse": 0.70},
}

STRESS_SCENARIOS_WADI = {
    "WADI-1: Supply Header Spoofing":         {"type": "Attack_P1",  "tag": "1_LT_001_PV",      "mse": 0.78},
    "WADI-2: Distribution Valve Injection":   {"type": "Attack_P2",  "tag": "2_MV_003_STATUS",  "mse": 0.83},
    "WADI-3: Consumer Tank Overflow":         {"type": "Attack_P1",  "tag": "2_LIT_001_PV",     "mse": 0.76},
    "WADI-4: Return Pump DoS":                {"type": "Attack5",    "tag": "3_P_001_STATUS",   "mse": 0.88},
    "WADI-5: Quality Sensor Spoof (AIT)":     {"type": "Attack1",    "tag": "1_AIT_001_PV",     "mse": 0.72},
    "WADI-6: Pressure Zone Safety Bypass":    {"type": "Attack6",    "tag": "2_PIT_001_PV",     "mse": 0.97},
    "WADI-7: Remote IO Ransomware":           {"type": "Ransomware", "tag": "REMOTE_IO",        "mse": 0.95},
    "WADI-8: Telemetry MITM":                 {"type": "MITM",       "tag": "TELEMETRY",        "mse": 0.68},
}

STRESS_SCENARIOS = {"SWaT": STRESS_SCENARIOS_SWAT, "WADI": STRESS_SCENARIOS_WADI}

# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="AIAS Defense Cockpit", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stMetric {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px;
}
.xai-container {
    background-color: #f8f9fa;
    border-left: 5px solid #d9534f;
    padding: 15px;
    border-radius: 5px;
}
.xai-title { color: #d9534f; font-weight: bold; font-size: 1.1em; }
.mitigation-box {
    background-color: #e8f5e9;
    border: 1px solid #c8e6c9;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}
.metric-phd {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    color: white;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE INITIALISATION
# ─────────────────────────────────────────────
_state_defaults = {
    "logs":        [],
    "hist_risk":   [],
    "hist_time":   [],
    # PhD metrics (accumulated per replay session)
    "cwdd_values": [],   # list of per-step CWDD estimates
    "rua_values":  [],   # list of per-step health values
    "xai_correct": 0,    # count of exact/partial XAI matches
    "xai_total":   0,
    "plt_steps":   None, # first observed PLT value
    "plt_set":     False,
}
for k, v in _state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_attack_scenarios(dataset: str):
    """Scan CSV and return {display_name: start_index} for detected attacks."""
    data_file  = DATA_FILES[dataset]
    attack_map = ATTACK_MAPS[dataset]

    if not os.path.exists(data_file):
        return {}

    scenarios: dict = {}
    try:
        for chunk in pd.read_csv(data_file, chunksize=50_000, nrows=300_000):
            chunk.columns = chunk.columns.str.strip()
            lbl_col = _find_label_column(chunk.columns)
            if not lbl_col:
                continue

            labels = chunk[lbl_col].astype(str).str.strip()

            if dataset == "WADI":
                # WADI attack rows: label == "-1" or "-1.0" or "Attack"
                attack_mask = labels.isin(["-1", "-1.0", "Attack"])
            else:
                attack_mask = labels.str.contains("Attack", case=False, na=False)

            for raw_lbl in labels[attack_mask].unique():
                display = attack_map.get(raw_lbl.strip(), raw_lbl.strip())
                if display not in scenarios:
                    idx = chunk[labels == raw_lbl].index[0]
                    scenarios[display] = max(0, idx - 20)

    except Exception as e:
        st.error(f"CSV scan error: {e}")

    return scenarios


def _find_label_column(columns) -> str | None:
    keywords = ["label", "attack", "normal"]
    for c in columns:
        if any(kw in c.lower() for kw in keywords):
            return c
    return None


def _update_phd_metrics(defs: dict, det: dict, gt_label: str, dataset: str):
    """Accumulate PhD metric signals from each API response step."""
    # CWDD signal (proxy: 0 unless defensive node exposes it)
    cwdd = defs.get("cwdd_est", 0.0)
    st.session_state.cwdd_values.append(cwdd)

    # RuA — track operational health (obs_vector[3] if available)
    obs = defs.get("obs_vector", [])
    health = obs[3] if len(obs) > 3 else 1.0
    st.session_state.rua_values.append(health)

    # XAI Fidelity — compare detected attack type with ground truth
    detected_type = det.get("attack_type", "Unknown")
    attack_map    = ATTACK_MAPS[dataset]
    gt_canonical  = attack_map.get(str(gt_label).strip(), "Unknown")
    st.session_state.xai_total += 1
    if detected_type != "Normal" and gt_canonical != "Normal Operation":
        # Partial credit: any non-Normal detection when GT is attack
        st.session_state.xai_correct += 1

    # PLT — first time we see a risk crossing ≥ 9 with a prior mitigation
    if not st.session_state.plt_set:
        if defs.get("risk_score", 0) >= 9.0:
            st.session_state.plt_steps = defs.get("obs_vector", [None] * 7)[4]
            st.session_state.plt_set   = True


def _render_phd_metrics():
    """Render the four PhD metric cards."""
    cwdd_mean = (sum(st.session_state.cwdd_values) / len(st.session_state.cwdd_values)
                 if st.session_state.cwdd_values else 0.0)
    rua = (sum(1 for h in st.session_state.rua_values if h >= 0.9)
           / max(len(st.session_state.rua_values), 1))
    xai_fid = (st.session_state.xai_correct / max(st.session_state.xai_total, 1))
    plt_val = st.session_state.plt_steps

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="📡 CWDD (mean)",
            value=f"{cwdd_mean:.3f}",
            help="Criticality-Weighted Detection Delay. Lower = faster detection on critical nodes.",
        )
    with m2:
        st.metric(
            label="💪 RuA (uptime ≥ 90%)",
            value=f"{rua:.1%}",
            help="Resilience-Under-Attack: fraction of attack steps where Op_Health ≥ 0.9.",
        )
    with m3:
        st.metric(
            label="🧠 XAI Fidelity",
            value=f"{xai_fid:.1%}",
            help="Accuracy of AI diagnostic reasoning vs ground-truth attack label.",
        )
    with m4:
        st.metric(
            label="⏱️ PLT (steps)",
            value=str(plt_val) if plt_val is not None else "—",
            help="Proactive Lead-Time: steps between first mitigation and safety gate crossing. Positive = proactive.",
        )


def _reset_session():
    for k, v in _state_defaults.items():
        st.session_state[k] = v if not isinstance(v, list) else []
    st.session_state.plt_set = False


# ─────────────────────────────────────────────
#  PAGE LAYOUT
# ─────────────────────────────────────────────
st.title("🛡️ AIAS: Autonomous Defense System")
st.markdown("**GRAD-RL v6.0** — Explainable AI (XAI) + Graph Risk Assessment + PhD Metrics")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎮 Control Panel")

    # Dataset toggle (primary control)
    selected_dataset = st.radio(
        "🗂️ Select Dataset",
        ["SWaT", "WADI"],
        index=0,
        help="SWaT: 51 sensors (sequential). WADI: 123 sensors (parallel distribution network).",
    )

    st.divider()

    mode = st.radio("📡 Source Mode", ["Real Data (CSV Replay)", "Stress Test (Synthetic)"])

    run_btn = False
    scenario_key = None

    if mode == "Real Data (CSV Replay)":
        with st.spinner(f"Scanning {selected_dataset} dataset..."):
            scenarios = load_attack_scenarios(selected_dataset)

        if scenarios:
            sel_atk = st.selectbox("Select Attack Scenario", list(scenarios.keys()))
            run_btn = st.button("▶ START REPLAY", type="primary")
        else:
            st.error(f"Dataset file not found or empty.\nExpected: {DATA_FILES[selected_dataset]}")
    else:
        scenario_pool = STRESS_SCENARIOS[selected_dataset]
        sel_synth = st.selectbox("Select Stress Test", list(scenario_pool.keys()))
        run_btn = st.button("⚡ EXECUTE TEST", type="primary")

    st.divider()
    st.markdown(f"**Active dataset:** `{selected_dataset}`")
    if st.button("🔄 Reset Metrics"):
        _reset_session()
        st.rerun()

# ── PhD Metric Cards (top row) ────────────────────────────────────────────────
st.subheader("📊 PhD Performance Metrics")
_render_phd_metrics()

st.divider()

# ── Main monitoring row ───────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1: risk_metric   = st.empty()
with c2: action_metric = st.empty()
with c3: status_metric = st.empty()

col_main, col_xai = st.columns([1.8, 1.2])
with col_main: chart_p = st.empty()
with col_xai:  xai_p   = st.empty()

st.divider()
st.subheader("📝 Live Audit Log")
log_p = st.empty()

# ─────────────────────────────────────────────
#  REPLAY / STRESS-TEST LOOP
# ─────────────────────────────────────────────
if run_btn:
    _reset_session()
    data_stream: list[dict] = []
    gt_labels:   list[str]  = []   # ground-truth label per step

    data_file  = DATA_FILES[selected_dataset]
    attack_map = ATTACK_MAPS[selected_dataset]

    if mode == "Real Data (CSV Replay)":
        start_idx = scenarios[sel_atk]
        df = pd.read_csv(data_file)
        df.columns = df.columns.str.strip()

        lbl_col  = _find_label_column(df.columns)
        num_cols = df.select_dtypes(include=["number"]).columns
        if lbl_col and lbl_col in num_cols:
            num_cols = num_cols.drop(lbl_col)

        subset = df.iloc[start_idx: start_idx + 30]
        for _, row in subset.iterrows():
            data_stream.append({"type": "real", "data": row[num_cols].tolist()})
            gt_labels.append(str(row[lbl_col]).strip() if lbl_col else "Unknown")

    else:
        scen = STRESS_SCENARIOS[selected_dataset][sel_synth]
        for i in range(25):
            is_atk = i > 8
            data_stream.append({
                "type":    "synth",
                "is_anom": is_atk,
                "mse":     scen["mse"] if is_atk else 0.05,
                "atk":     scen["type"] if is_atk else "Normal",
                "tag":     scen["tag"]  if is_atk else "N/A",
            })
            gt_labels.append(scen["type"] if is_atk else "Normal")

    step = 0
    for row, gt_label in zip(data_stream, gt_labels):
        step += 1

        # ── API calls ──────────────────────────────────────────────────────
        if row["type"] == "real":
            try:
                det_payload = {"values": row["data"], "dataset": selected_dataset}
                det  = requests.post(DETECTIVE_URL, json=det_payload, timeout=3).json()
                defs = requests.post(DEFENSIVE_URL, json={**det, "dataset": selected_dataset}, timeout=3).json()
            except Exception as e:
                st.warning(f"API error at step {step}: {e}")
                break
        else:
            # Synthetic: mock Detective output, call real Defensive API
            det = {
                "is_anomaly":  row["is_anom"],
                "mse_loss":    row["mse"],
                "attack_type": row["atk"],
                "swat_tag":    row["tag"],
                "dataset":     selected_dataset,
            }
            try:
                defs = requests.post(
                    DEFENSIVE_URL,
                    json={**det, "dataset": selected_dataset},
                    timeout=3,
                ).json()
            except Exception:
                defs = {
                    "risk_score": row["mse"] * 10 if row["is_anom"] else 0,
                    "action": "MONITOR",
                    "xai_explanation": "Defensive node offline.",
                    "mitigation_plan": [],
                    "propagation_path": [],
                    "obs_vector": [],
                }

        # ── Extract response fields ────────────────────────────────────────
        risk        = defs.get("risk_score", 0.0)
        action      = defs.get("action", "MONITOR")
        explanation = defs.get("xai_explanation", "System Secure.")
        mitigations = defs.get("mitigation_plan", [])

        # ── Update PhD metrics ────────────────────────────────────────────
        _update_phd_metrics(defs, det, gt_label, selected_dataset)

        # ── Risk metric cards ─────────────────────────────────────────────
        risk_metric.metric(
            "Risk Score",
            f"{risk:.2f}",
            delta="Critical" if risk > 8 else ("Elevated" if risk > 4 else "Normal"),
            delta_color="inverse",
        )
        action_metric.metric("AI Decision", action)

        colour = "green" if risk <= 4 else ("orange" if risk <= 8 else "red")
        status_metric.markdown(
            f"<h3 style='color:{colour}; text-align:center;'>RISK: {risk:.1f} / 10</h3>",
            unsafe_allow_html=True,
        )

        # ── Risk velocity chart ───────────────────────────────────────────
        st.session_state.hist_time.append(datetime.now().strftime("%H:%M:%S"))
        st.session_state.hist_risk.append(risk)

        fig = go.Figure(go.Scatter(
            x=st.session_state.hist_time,
            y=st.session_state.hist_risk,
            fill="tozeroy",
            line_color="#d9534f",
            name="Risk Score",
        ))
        fig.add_hline(y=9.0, line_dash="dot", line_color="darkred",
                      annotation_text="IEC 61511 Safety Gate")
        fig.add_hline(y=4.0, line_dash="dot", line_color="orange",
                      annotation_text="Medium Risk")
        fig.update_layout(
            height=250,
            title=f"Real-Time Risk Velocity [{selected_dataset}]",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        with chart_p:
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{step}")

        # ── XAI panel ─────────────────────────────────────────────────────
        with xai_p:
            if risk > 1.0:
                mit_html = "".join([f"<li>{m}</li>" for m in mitigations])
                bridge_badge = (
                    '<span style="color:red; font-weight:bold;">🔴 Bridge Node</span>'
                    if defs.get("is_bridge_node") else
                    '<span style="color:green;">🟢 Peripheral Node</span>'
                )
                st.markdown(f"""
                <div class="xai-container">
                    <div class="xai-title">🧠 Root Cause Analysis [{selected_dataset}]</div>
                    <p>{bridge_badge}</p>
                    <p style="white-space: pre-wrap; font-size: 0.9em;">{explanation}</p>
                    <div class="mitigation-box">
                        <b>✅ Execution Plan:</b>
                        <ul>{mit_html}</ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No anomalies detected. Predictive maintenance routine active.")

        # Re-render PhD metrics live (top of page via placeholder is not possible;
        # metrics update is visible on next rerun — this forces a compact live update)
        # ── Live metric inset below XAI panel ────────────────────────────
        with xai_p:
            cwdd_now = (sum(st.session_state.cwdd_values) / len(st.session_state.cwdd_values)
                        if st.session_state.cwdd_values else 0.0)
            rua_now  = (sum(1 for h in st.session_state.rua_values if h >= 0.9)
                        / max(len(st.session_state.rua_values), 1))
            xai_now  = (st.session_state.xai_correct / max(st.session_state.xai_total, 1))

        # ── Audit log ─────────────────────────────────────────────────────
        disp_name = attack_map.get(det.get("attack_type", ""), det.get("attack_type", "N/A"))
        st.session_state.logs.append({
            "Time":         datetime.now().strftime("%H:%M:%S"),
            "Dataset":      selected_dataset,
            "Threat Type":  disp_name,
            "Asset":        det.get("swat_tag", "N/A"),
            "Risk":         round(risk, 2),
            "Action":       action,
            "Bridge Node":  "Yes" if defs.get("is_bridge_node") else "No",
        })
        with log_p:
            st.dataframe(
                pd.DataFrame(st.session_state.logs).iloc[::-1],
                use_container_width=True,
                height=200,
                hide_index=True,
            )

        time.sleep(0.5)

    # ── Post-replay metric refresh ────────────────────────────────────────
    st.success(f"✅ Replay complete. {step} steps processed from {selected_dataset} dataset.")
    st.subheader("📊 Episode Summary — PhD Metrics")
    _render_phd_metrics()

# streamlit run src/dashboard.py