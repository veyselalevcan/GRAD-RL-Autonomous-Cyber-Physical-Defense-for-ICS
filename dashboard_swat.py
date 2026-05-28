import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time
import os
from datetime import datetime

# --- CONFIGURATION ---
DETECTIVE_URL = "http://127.0.0.1:8000/predict"
DEFENSIVE_URL = "http://127.0.0.1:8001/decide"
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "SWaT_Dataset_Attack_v0.csv")

st.set_page_config(page_title="AIAS Defense Cockpit", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; }
    .xai-container { background-color: #f8f9fa; border-left: 5px solid #d9534f; padding: 15px; border-radius: 5px; }
    .xai-title { color: #d9534f; font-weight: bold; font-size: 1.1em; }
    .mitigation-box { background-color: #e8f5e9; border: 1px solid #c8e6c9; padding: 10px; border-radius: 5px; margin-top: 10px;}
    </style>
""", unsafe_allow_html=True)

# --- COMPLETE ATTACK TRANSLATION MAP (1-6) ---
ATTACK_MAP = {
    "Attack1": "Sensor Spoofing",
    "Attack2": "Actuator Injection",
    "Attack3": "Pump Manipulation",
    "Attack4": "Data Integrity Loss",
    "Attack5": "DoS Attack",
    "Attack6": "Safety Logic Bypass",
    "MITM": "MITM Attack",
    "Replay": "Replay Attack",
    "Ransomware": "Ransomware",
    "Normal": "Normal Operation"
}

if 'logs' not in st.session_state: st.session_state.logs = []
if 'hist_risk' not in st.session_state: st.session_state.hist_risk = []
if 'hist_time' not in st.session_state: st.session_state.hist_time = []

@st.cache_data
def load_attack_scenarios():
    if not os.path.exists(DATA_FILE): return {}
    scenarios = {}
    try:
        # Scan CSV for real attacks
        for chunk in pd.read_csv(DATA_FILE, chunksize=50000, nrows=300000):
            chunk.columns = chunk.columns.str.strip()
            lbl_col = next((c for c in chunk.columns if 'Label' in c or 'Attack' in c), None)
            if lbl_col:
                attacks = chunk[chunk[lbl_col].astype(str).str.contains("Attack", case=False, na=False)]
                for raw_lbl in attacks[lbl_col].unique():
                    clean_lbl = raw_lbl.strip()
                    generic_name = ATTACK_MAP.get(clean_lbl, clean_lbl)
                    if generic_name not in scenarios:
                        idx = attacks[attacks[lbl_col] == raw_lbl].index[0]
                        scenarios[generic_name] = max(0, idx - 20)
    except: pass
    return scenarios

# --- EXPANDED STRESS TEST SCENARIOS (FULL SUITE) ---
STRESS_SCENARIOS = {
    "Scenario 1: Sensor Spoofing (Att 1)":   {"type": "Attack1", "tag": "FIT101", "mse": 0.8},
    "Scenario 2: Actuator Injection (Att 2)":{"type": "Attack2", "tag": "MV201", "mse": 0.85},
    "Scenario 3: Pump Manipulation (Att 3)": {"type": "Attack3", "tag": "P101", "mse": 0.75},
    "Scenario 4: Data Integrity (Att 4)":    {"type": "Attack4", "tag": "LIT301", "mse": 0.82},
    "Scenario 5: Network DoS (Att 5)":       {"type": "Attack5", "tag": "Switch_1", "mse": 0.9},
    "Scenario 6: Safety Bypass (Att 6)":     {"type": "Attack6", "tag": "PLC_P6", "mse": 1.0},
    "Scenario 7: Ransomware Attack":         {"type": "Ransomware", "tag": "Eng_Workstation", "mse": 0.99},
    "Scenario 8: MITM Attack":               {"type": "MITM", "tag": "HMI_Main", "mse": 0.7}
}

st.title("🛡️ AIAS: Autonomous Defense System")
st.markdown("**PhD Module:** Explainable AI (XAI) & Root Cause Analysis")

with st.sidebar:
    st.header("🎮 Operation Mode")
    mode = st.radio("Source", ["Real Data (CSV Replay)", "Stress Test (Synthetic)"])
    
    run_btn = False
    if mode == "Real Data (CSV Replay)":
        with st.spinner("Loading & Translating Dataset (Attacks 1-6)..."):
            scenarios = load_attack_scenarios()
        
        if scenarios:
            sel_atk = st.selectbox("Select Attack Scenario", list(scenarios.keys()))
            run_btn = st.button("▶ START REPLAY", type="primary")
        else:
            st.error("Dataset not found or empty.")
    else:
        # Updated to use the expanded dictionary
        sel_synth = st.selectbox("Select Stress Test", list(STRESS_SCENARIOS.keys()))
        run_btn = st.button("⚡ EXECUTE TEST", type="primary")

c1, c2, c3 = st.columns(3)
with c1: risk_metric = st.empty()
with c2: action_metric = st.empty()
with c3: status_metric = st.empty()

col_main, col_xai = st.columns([1.8, 1.2])
with col_main: chart_p = st.empty()
with col_xai: xai_p = st.empty()

st.divider()
st.subheader("📝 Live Audit Log")
log_p = st.empty()

if run_btn:
    st.session_state.logs = []
    st.session_state.hist_risk = []
    st.session_state.hist_time = []
    
    data_stream = []
    if mode == "Real Data (CSV Replay)":
        start_idx = scenarios[sel_atk]
        df = pd.read_csv(DATA_FILE)
        raw_label = next((k for k, v in ATTACK_MAP.items() if v == sel_atk), "Attack1")
        
        lbl_col = [c for c in df.columns if 'Label' in c or 'Attack' in c][0]
        num_cols = df.select_dtypes(include=['number']).columns.drop(lbl_col, errors='ignore')
        
        subset = df.iloc[start_idx : start_idx+30]
        for _, row in subset.iterrows():
            data_stream.append({"type": "real", "data": row[num_cols].tolist()})
    else:
        scen = STRESS_SCENARIOS[sel_synth]
        for i in range(25):
            is_atk = i > 8
            data_stream.append({
                "type": "synth", 
                "is_anom": is_atk, 
                "mse": scen["mse"] if is_atk else 0.05,
                "atk": scen["type"] if is_atk else "Normal", 
                "tag": scen["tag"] if is_atk else "N/A"
            })

    step = 0
    for row in data_stream:
        step += 1
        
        if row["type"] == "real":
            try:
                det = requests.post(DETECTIVE_URL, json={"values": row["data"]}).json()
                defs = requests.post(DEFENSIVE_URL, json=det).json()
            except: break
        else:
            # Mock Detective Output for Synthetic Mode
            det = {"is_anomaly": row['is_anom'], "mse_loss": row['mse'], "attack_type": row['atk'], "swat_tag": row['tag']}
            try: defs = requests.post(DEFENSIVE_URL, json=det).json()
            except: defs = {}

        risk = defs.get('risk_score', 0)
        action = defs.get('action', 'MONITOR')
        explanation = defs.get('xai_explanation', 'System Secure.')
        mitigations = defs.get('mitigation_plan', [])
        
        risk_metric.metric("Risk Score", f"{risk:.2f}", delta="Critical" if risk > 8 else "Normal", delta_color="inverse")
        action_metric.metric("AI Decision", action)
        
        status_color = "green"
        if risk > 8: status_color = "red"
        elif risk > 4: status_color = "orange"
        status_metric.markdown(f"<h3 style='color:{status_color}; text-align:center;'>RISK LEVEL: {risk:.1f}</h3>", unsafe_allow_html=True)
        
        st.session_state.hist_time.append(datetime.now().strftime("%H:%M:%S"))
        st.session_state.hist_risk.append(risk)
        fig = go.Figure(go.Scatter(x=st.session_state.hist_time, y=st.session_state.hist_risk, fill='tozeroy', line_color='#d9534f'))
        fig.update_layout(height=250, title="Real-Time Risk Velocity", margin=dict(l=0, r=0, t=30, b=0))
        with chart_p: st.plotly_chart(fig, use_container_width=True, key=f"c_{step}")
        
        with xai_p:
            if risk > 1.0:
                mit_html = "".join([f"<li>{m}</li>" for m in mitigations])
                st.markdown(f"""
                <div class="xai-container">
                    <div class="xai-title">🧠 Root Cause Analysis</div>
                    <p style="white-space: pre-wrap; font-size: 0.95em;">{explanation}</p>
                    <div class="mitigation-box">
                        <b>✅ Execution Plan:</b>
                        <ul>{mit_html}</ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No anomalies detected. Predictive maintenance routine active.")

        # Show CLEAN names in log
        disp_name = ATTACK_MAP.get(det.get('attack_type'), det.get('attack_type'))
        st.session_state.logs.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Threat Type": disp_name,
            "Asset": det.get('swat_tag', 'N/A'),
            "Risk": risk,
            "Action": action
        })
        with log_p: st.dataframe(pd.DataFrame(st.session_state.logs).iloc[::-1], use_container_width=True, height=200, hide_index=True)
        
        time.sleep(0.5)