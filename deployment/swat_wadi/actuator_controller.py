import json
import os
import requests
from datetime import datetime


DETECTIVE_API_URL = "http://127.0.0.1:8000/predict"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "system_state.json")

DEFAULT_STATE = {
    "network_status": "NORMAL",
    "zone_isolated": False,
    "pumps": {
        "P601": "ON",
        "P602": "ON",
        "P603": "ON"
    },
    "alerts": []
}


def load_state():
    if not os.path.exists(STATE_FILE):
        save_state(DEFAULT_STATE)
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def log_alert(message):
    state = load_state()
    state["alerts"].append({
        "timestamp": datetime.now().isoformat(),
        "message": message
    })
    save_state(state)


# ---------- ACTIONS ----------

def block_ip():
    state = load_state()
    state["network_status"] = "THREAT_BLOCKED"
    save_state(state)
    log_alert("Firewall rule deployed by AI.")


def isolate_zone():
    state = load_state()
    state["zone_isolated"] = True
    save_state(state)
    log_alert("Compromised zone isolated.")


def system_shutdown():
    state = load_state()
    for pump in state["pumps"]:
        state["pumps"][pump] = "OFF"
    save_state(state)
    log_alert("Emergency shutdown executed by AI.")


def alert_operator(message="Operator attention required"):
    log_alert(message)

def execute_action(action: str, target_tag: str):
    print(f"[ACTUATOR] Executing {action} on {target_tag}")
    return True

def block_ip():
    s = _load()
    s["network"] = "BLOCKED"
    _save(s)
    _alert("AI blocked malicious traffic.")
    reobserve_system()

def isolate_zone():
    s = _load()
    s["zone_isolated"] = True
    _save(s)
    _alert("AI isolated affected zone.")
    reobserve_system()

def system_shutdown():
    s = _load()
    for p in s["pumps"]:
        s["pumps"][p] = "OFF"
    _save(s)
    _alert("AI executed emergency shutdown.")
    reobserve_system()

def alert_operator():
    _alert("AI requests operator review.")
    reobserve_system()

    
def reobserve_system():
    """
    After an action, simulate new sensor reading and send back to Detective.
    This creates the closed-loop defense cycle.
    """
    state = _load()

    # Simple simulation logic:
    # If pumps are OFF, pressure drops
    values = [1.0] * 51  # Default SWaT feature length

    if state["zone_isolated"]:
        values[0] = 0.5  # FIT101 reduced flow

    if state["pumps"]["P601"] == "OFF":
        values[10] = 0.2  # Example pressure drop

    try:
        requests.post(DETECTIVE_API_URL, json={"values": values}, timeout=1)
    except Exception as e:
        print(f"[WARN] Re-observation failed: {e}")


