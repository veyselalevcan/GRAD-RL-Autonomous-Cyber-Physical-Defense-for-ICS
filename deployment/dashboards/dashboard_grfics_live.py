"""
dashboard_grfics.py — GRAD-RL Framework v7.5 (Live Event-Driven Edition)
AIAS Defense Cockpit | PhD Module: XAI & Root Cause Analysis

Run:
    streamlit run dashboard_grfics.py
"""

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time
import os
import json
import csv
from datetime import datetime
from pymodbus.client import ModbusTcpClient as _PlcClient

# ─────────────────────────────────────────────
#  CONFIGURATION & LIVE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GRAD-RL Defense Cockpit",
    page_icon="🛡️",
    layout="wide"
)

DETECTIVE_URL = "http://127.0.0.1:8000/predict"
DEFENSIVE_URL = "http://127.0.0.1:8001/decide"
_ROOT         = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH  = os.path.join(os.environ.get("GRADRL_DATA_DIR", os.path.join(_ROOT, "data")), "grad_rl_final_dataset.csv")
P99_THRESHOLD = 0.002731   

FEATURE_COLUMNS = [
    "f1_valve", "f1_flow", "f2_valve", "f2_flow",
    "purge_valve", "purge_flow", "prod_valve",  "prod_flow",
    "pressure", "level", "A_purge", "B_purge", "C_purge"
]

LIVE_PLC_HOST  = "127.0.0.1"
LIVE_PLC_PORT  = 502
LIVE_SCALES    = [100.0, 500.0, 100.0, 500.0, 100.0, 500.0, 100.0, 500.0, 3200.0, 100.0, 100.0, 100.0, 100.0]

def read_live_plc_single_shot() -> tuple[list[float] | None, dict | None]:
    """Reads registers cleanly without locking socket loops."""
    client = _PlcClient(LIVE_PLC_HOST, port=LIVE_PLC_PORT, timeout=3)
    try:
        if not client.connect():
            return None, None
        r = client.read_input_registers(address=100, count=13)
        if r.isError():
            return None, None
        raw = r.registers
        features = [round(raw[i] / 65535.0 * LIVE_SCALES[i], 4) for i in range(13)]
        row_dict = dict(zip(FEATURE_COLUMNS, features))
        return features, row_dict
    except:
        return None, None
    finally:
        client.close()

# ─────────────────────────────────────────────
#  CSS & LOGGING SYSTEM
# ─────────────────────────────────────────────
st.markdown("""
<style>
.xai-container { background-color: #fff5f5; border-left: 5px solid #d9534f; padding: 18px; border-radius: 8px; margin-bottom: 16px; }
.xai-title { color: #d9534f; font-weight: bold; font-size: 1.15em; margin-bottom: 10px; }
.mitigation-box { background-color: #e8f5e9; border: 1px solid #c8e6c9; padding: 14px; border-radius: 8px; margin-top: 12px; }
.normal-box { background-color: #f0f4ff; border-left: 4px solid #4a90d9; padding: 14px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

LOG_DIR       = os.path.join(_ROOT, "results", "grfics_live", "new_session")
TELEMETRY_LOG = os.path.join(LOG_DIR, "telemetry_audit.csv")
XAI_LOG       = os.path.join(LOG_DIR, "xai_incidents.json")
os.makedirs(LOG_DIR, exist_ok=True)

def log_telemetry(ts, scenario, pressure, mse, risk, action):
    exists = os.path.exists(TELEMETRY_LOG)
    with open(TELEMETRY_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists: w.writerow(["Timestamp","Scenario","Pressure","MSE","CVSS","Action"])
        w.writerow([ts, scenario, pressure, round(mse,6), round(risk,2), action])

def log_incident(ts, scenario, data: dict):
    with open(XAI_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": ts, "scenario": scenario, "details": data}) + "\n")

# ─────────────────────────────────────────────
#  SESSION STATE MANAGMENT (Kritik Alan)
# ─────────────────────────────────────────────
for k, v in [("logs",[]), ("hist_data",[]), ("latest_xai", None), ("running", False), ("current_idx", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    for col in FEATURE_COLUMNS: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df
df = load_data()

# ─────────────────────────────────────────────
#  SIDEBAR CONTROL INTERFACE
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Operation Mode")
    mode = st.radio("Data Source", ["Real Data (CSV Replay)", "Stress Test (Synthetic ×1.4)", "🔴 Live GRFICSv3 (Real-Time)"])
    scenario = st.selectbox("Attack Scenario", ["Baseline (Normal Operation)", "T0836: Setpoint Poisoning", "T0856: Sensor Spoofing", "T0813: Denial of Control", "Combined Attack (T0813+T0836)"])
    
    SCENARIO_IDX = {"Baseline (Normal Operation)": 0, "T0836: Setpoint Poisoning": 1135, "T0856: Sensor Spoofing": 524, "T0813: Denial of Control": 0, "Combined Attack (T0813+T0836)": 90135}
    speed = st.slider("Replay Speed (s/step)", 0.1, 2.0, 0.5)
    
    # İki ayrı buton ile döngüyü asenkron kontrol ediyoruz
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("▶ START", type="primary", use_container_width=True):
            st.session_state.running = True
            st.session_state.current_idx = SCENARIO_IDX[scenario]
            st.session_state.logs = []
            st.session_state.hist_data = []
            try: requests.post("http://127.0.0.1:8000/reset_window", timeout=2)
            except: pass
    with col_b2:
        if st.button("🛑 STOP", type="secondary", use_container_width=True):
            st.session_state.running = False
            
    st.divider()
    if mode == "🔴 Live GRFICSv3 (Real-Time)":
        st.info("Live Monitoring Active.\nTrigger attacks from MobaXterm terminal window.")

# ─────────────────────────────────────────────
#  REAL-TIME UI LAYOUT
# ─────────────────────────────────────────────
st.title("🛡️ GRAD-RL: Autonomous Cyber-Physical Defense")
st.markdown("**AIAS Defense Cockpit** | LSTM-AE + XGBoost + PPO | GRFICSv3 Digital Twin")
st.divider()

c1, c2, c3 = st.columns(3)
with c1: m_risk   = st.empty()
with c2: m_action = st.empty()
with c3: m_sev    = st.empty()

col_l, col_r = st.columns([1.8, 1.2])
with col_l:
    st.markdown("**Real-Time Telemetry: Pressure & CVSS Risk**")
    chart_ph = st.empty()
    st.markdown("**Live Audit Log**")
    log_ph   = st.empty()
with col_r:
    st.markdown("**Incident Response Panel (XAI)**")
    xai_ph   = st.empty()

def build_chart(data):
    d = pd.DataFrame(data)
    if d.empty: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["time"], y=d["pressure"], name="Pressure", line=dict(color="#2196F3", width=2)))
    fig.add_trace(go.Scatter(x=d["time"], y=d["risk"], name="CVSS Risk", fill="tozeroy", line=dict(color="#d9534f", width=2)))
    fig.update_layout(height=280, margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified", legend=dict(orientation="h", y=1.1))
    return fig

def build_xai(threat: str, culprit: str, mse: float, row: pd.Series) -> tuple:
    t = threat.upper()
    if "T0856" in t:
        valve = float(row.get("f1_valve", 1.0))
        flow  = float(row.get("f1_flow",  0.0))
        if valve < 0.1 and flow > 0.05:
            diag = f"Physical contradiction on <b>{culprit}</b>: f1_valve CLOSED ({valve:.2f}) yet f1_flow reports active flow ({flow:.3f})."
            fore = "PLC processing falsified values. Risk: chemical imbalance or tank overflow."
        else:
            diag = f"Telemetry spoofing on <b>{culprit}</b>. Readings violate physical invariants (MSE {mse:.4f})."
            fore = "Spoofed industrial sensor telemetry forcing incorrect valve actuation."
        return diag, fore
    if "T0836" in t:
        press = float(row.get("pressure", 0.0))
        if press > 2750:
            diag = f"Malicious setpoint injection on <b>{culprit}</b>. Reactor pressure forced CRITICAL HIGH ({press:.1f} PSI)."
            fore = "Catastrophic breach expected within ~45 s."
        else:
            diag = f"Malicious setpoint injection on <b>{culprit}</b>. Pressure forced LOW ({press:.1f} PSI)."
            fore = "Process stall and pump cavitation imminent."
        return diag, fore
    if "T0813" in t and "T0836" not in t:
        return f"DoS on <b>{culprit}</b>. Control traffic blocked.", "Loss of Control bounds."
    
    return f"Anomaly on <b>{culprit}</b>. (MSE {mse:.4f}).", "IEC 61511 fallback provisions active."

def confidence_pct(mse: float) -> float:
    return min(99.9, 50.0 + (mse / P99_THRESHOLD) * 0.01)

# ─────────────────────────────────────────────
#  DYNAMIC STEP EXECUTION (Bütün kilitleri çözen yer)
# ─────────────────────────────────────────────
if st.session_state.running:
    # 1. VERİ KAYNAĞI SEÇİMİ
    if mode == "🔴 Live GRFICSv3 (Real-Time)":
        vals, row_raw = read_live_plc_single_shot()
        if vals is None:
            time.sleep(1)
            st.rerun()
        row = pd.Series(row_raw)
        current_scenario_text = "LIVE"
    else:
        idx = st.session_state.current_idx
        if idx >= len(df):
            st.session_state.running = False
            st.st.rerun()
        row = df.iloc[idx]
        vals = row[FEATURE_COLUMNS].tolist()
        if mode == "Stress Test (Synthetic ×1.4)": vals = [v * 1.4 for v in vals]
        row = pd.Series(dict(zip(FEATURE_COLUMNS, vals)))
        st.session_state.current_idx += 1
        current_scenario_text = scenario

    # 2. PIPELINE PIPELINE ÇAĞRILARI
    try:
        res_det = requests.post(DETECTIVE_URL, json={"values": vals}, timeout=5).json()
        res_def = requests.post(DEFENSIVE_URL, json={
            "is_anomaly": res_det.get("is_anomaly"), "mse_loss": res_det.get("mse_loss"),
            "attack_type": res_det.get("attack_type"), "culprit_feature": res_det.get("culprit_feature"),
            "scaled_features": res_det.get("scaled_features"),
        }, timeout=5).json()

        ts_full = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_disp = time.strftime("%H:%M:%S")

        risk = float(res_def.get("cvss_score", 0.0))
        sev  = res_def.get("cvss_severity", "None")
        action = res_def.get("final_action", "DO_NOTHING")
        rl_act = res_def.get("rl_action", "DO_NOTHING")
        threat = res_det.get("attack_type", "Normal")
        mse    = float(res_det.get("mse_loss", 0.0))
        culprit = res_det.get("culprit_feature", "unknown")

        log_telemetry(ts_full, current_scenario_text, float(row["pressure"]), mse, risk, action)

        # UI YENİLEME
        m_risk.metric("CVSS 4.0 Risk", f"{risk:.2f}", delta="⚠ CRITICAL" if risk >= 7 else "✓ Normal", delta_color="inverse")
        m_action.metric("PPO → Blue Agent", action)
        sc = {"Critical":"red","High":"orange","Medium":"blue","Low":"green"}.get(sev,"gray")
        m_sev.markdown(f"<h2 style='color:{sc};text-align:center;border:2px solid {sc};border-radius:8px;padding:4px'>{sev}</h2>", unsafe_allow_html=True)

        st.session_state.hist_data.append({"time": ts_disp, "pressure": float(row["pressure"]), "risk": risk})
        if len(st.session_state.hist_data) > 50: st.session_state.hist_data.pop(0)
        
        # FIX 1: use_container_width kaldırıldı, yerine yeni 2026 standardı width='stretch' eklendi
        chart_ph.plotly_chart(build_chart(st.session_state.hist_data), width='stretch')

        entry = f"[{ts_disp}] THREAT={threat} | ASSET={culprit} | CVSS={risk:.1f} | ACT={action}"
        st.session_state.logs.insert(0, entry)
        if len(st.session_state.logs) > 6: st.session_state.logs.pop()
        log_ph.code("\n".join(st.session_state.logs))

        if res_det.get("is_anomaly"):
            feat_errs = res_det.get("feature_errors", [0.0] * len(FEATURE_COLUMNS))
            df_xai = pd.DataFrame({"Sensor": FEATURE_COLUMNS, "Attribution": feat_errs}).sort_values("Attribution", ascending=True).tail(5)
            
            fig_xai = go.Figure(go.Bar(
                x=df_xai["Attribution"], y=df_xai["Sensor"], orientation="h",
                marker=dict(color="#d9534f", line=dict(color="#800000", width=1)),
                text=df_xai["Attribution"].apply(lambda v: f"{v:.4f}"), textposition="auto"
            ))
            fig_xai.update_layout(title="<b>Local Reconstruction Error Attribution (εⱼ)</b>", height=200, margin=dict(l=0,r=0,t=30,b=0), xaxis_title="Contribution to Anomaly", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

            diag, fore = build_xai(threat, culprit, mse, row)
            conf = confidence_pct(mse)

            if action == "REQUIRE_HUMAN_APPROVAL":
                rat  = f"Human Approval Required — PPO recommended <b>{rl_act}</b>, CVSS {risk:.1f} ≥ 7.0."
                gate = "<span style='color:#d9534f;font-weight:bold'>IEC 61511: TRIGGERED (Lockout)</span>"
            elif action == "OPEN_SAFETY_VALVE":
                rat  = "Autonomous Mitigation — pressure relief engaged. CVSS &lt; 7.0, autonomous execution permitted."
                gate = "<span style='color:#5cb85c;font-weight:bold'>IEC 61511: COMPLIANT (Autonomous)</span>"
            else:
                rat  = "Proactive monitoring — intervention not yet mathematically optimal."
                gate = "IEC 61511: Standby"

            log_incident(ts_full, current_scenario_text, {
                "technique": threat, "mse": round(mse, 6), "confidence": round(conf, 1),
                "cvss": risk, "final_action": action, "rl_action": rl_act,
                "diagnosis": diag.replace("<b>","").replace("</b>",""), "forecast": fore,
                "top5_sensors": df_xai.to_dict("records"),
            })

            st.session_state.latest_xai = f"""
<div class="xai-container">
    <div class="xai-title">🧠 Physics-Informed Root Cause Analysis</div>
    <p><b>🚨 TECHNIQUE:</b> {threat} &nbsp;|&nbsp; MSE: {mse:.4f} &nbsp;|&nbsp; Confidence: {conf:.1f}%</p>
    <p><b>🔍 DIAGNOSIS:</b> {diag}</p>
    <p><b>⚠ FORECAST:</b> {fore}</p>
    <div class="mitigation-box">
        <b>🛡️ Autonomous Defense Decision:</b>
        <ul>
            <li><b>PPO Action:</b> {rl_act}</li>
            <li><b>Final Action:</b> {action}</li>
            <li><b>Rationale:</b> {rat}</li>
            <li><b>Standard:</b> {gate}</li>
        </ul>
    </div>
</div>"""

        if st.session_state.latest_xai:
            xai_ph.empty()
            with xai_ph.container():
                st.markdown(st.session_state.latest_xai, unsafe_allow_html=True)
                if res_det.get("is_anomaly"): 
                    # FIX 1: XAI bar grafiği de use_container_width yerine yeni standarda çekildi
                    st.plotly_chart(fig_xai, width='stretch', key=f"xai_{ts_disp.replace(':','')}")
        else:
            xai_ph.markdown('<div class="normal-box">✅ <b>Predictive monitoring active.</b><br>All parameters within digital-twin bounds.</div>', unsafe_allow_html=True)

    except Exception as e:
        st.sidebar.error(f"Pipeline error: {e}")

    # ADIM SONU: Kontrolü 0.5 saniyeliğine sisteme bırakıp sayfayı baştan tetikliyoruz. Websocket nefes alıyor!
    time.sleep(speed)
    st.rerun()

else:
    # Durdurulduğunda bekleme ekranı
    xai_ph.markdown('<div class="normal-box">⏸️ <b>Cockpit In Standby Mode.</b><br>Select mode and click START to capture live industrial telemetry streams.</div>', unsafe_allow_html=True)