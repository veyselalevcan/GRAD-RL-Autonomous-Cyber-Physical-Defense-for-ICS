"""
live_bridge.py — GRFICSv3 Modbus-to-API Live Bridge  v7.3
Data source: OpenPLC FC=4 Input Registers, address=100, count=13
Scale factors verified from original dataset collection scripts.
"""

import time
import requests
from pymodbus.client import ModbusTcpClient

# ── Configuration ──────────────────────────────────────────────────────────────
PLC_IP        = "127.0.0.1"
PLC_PORT      = 502
DETECTIVE_URL = "http://127.0.0.1:8000/predict"
DEFENSIVE_URL = "http://127.0.0.1:8001/decide"
POLL_INTERVAL = 1.0
API_TIMEOUT   = 5.0

FEATURE_NAMES = [
    "f1_valve","f1_flow","f2_valve","f2_flow",
    "purge_valve","purge_flow","prod_valve","prod_flow",
    "pressure","level","A_purge","B_purge","C_purge",
]

# Physical unit scale factors (raw / 65535 * scale = physical value)
# Verified against grfics_normal_part2.csv collection script
SCALE_FACTORS = {
    "f1_valve":    100.0,   # 0-100 %
    "f1_flow":     500.0,   # 0-500 (flow units)
    "f2_valve":    100.0,
    "f2_flow":     500.0,
    "purge_valve": 100.0,
    "purge_flow":  500.0,
    "prod_valve":  100.0,
    "prod_flow":   500.0,
    "pressure":   3200.0,   # 0-3200 PSI
    "level":       100.0,   # 0-100 %
    "A_purge":     100.0,   # 0-100 %
    "B_purge":     100.0,
    "C_purge":     100.0,
}
SCALES = [SCALE_FACTORS[n] for n in FEATURE_NAMES]
# ───────────────────────────────────────────────────────────────────────────────


def read_process_values(client: ModbusTcpClient) -> list[float] | None:
    """
    FC=4 Input Registers, address=100, count=13.
    Each register: 16-bit unsigned (0-65535) → physical unit via scale factor.
    """
    try:
        result = client.read_input_registers(address=100, count=13)
    except Exception as e:
        print(f"[BRIDGE] Transport error: {e}")
        return None

    if result.isError():
        print(f"[BRIDGE] Modbus error (FC=4, addr=100, count=13): {result}")
        return None

    return [round(raw / 65535.0 * scale, 4)
            for raw, scale in zip(result.registers, SCALES)]


def post_to_detective(features: list[float]) -> dict | None:
    try:
        r = requests.post(DETECTIVE_URL, json={"values": features}, timeout=API_TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        print("[DETECTIVE] Unreachable — uvicorn api_detective running on :8000?")
    except requests.exceptions.Timeout:
        print("[DETECTIVE] Timeout.")
    return None


def post_to_defensive(det: dict) -> dict | None:
    if not det.get("is_anomaly"):
        return None
    try:
        r = requests.post(DEFENSIVE_URL, json=det, timeout=API_TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except requests.exceptions.ConnectionError:
        print("[DEFENSIVE] Unreachable — uvicorn api_defensive running on :8001?")
    return None


def run_bridge():
    print("=" * 60)
    print("  GRAD-RL Live Bridge v7.3 | FC=4 addr=100 | 13 features")
    print("=" * 60)

    client = ModbusTcpClient(PLC_IP, port=PLC_PORT, timeout=3)
    if not client.connect():
        print(f"[BRIDGE] Cannot connect to {PLC_IP}:{PLC_PORT}")
        return
    print(f"[BRIDGE] Connected. Polling every {POLL_INTERVAL}s\n")

    cycle = 0
    try:
        while True:
            cycle += 1
            features = read_process_values(client)

            if features is None:
                # Reconnect on next cycle
                client.close()
                time.sleep(POLL_INTERVAL)
                client.connect()
                continue

            pressure = features[8]
            level    = features[9]
            print(f"[{cycle:05d}] P={pressure:.1f}PSI L={level:.1f}%  ", end="")

            det_resp = post_to_detective(features)
            if det_resp is None:
                print("→ detective unreachable")
                time.sleep(POLL_INTERVAL)
                continue

            dfn_resp = post_to_defensive(det_resp)
            anomaly  = det_resp.get("is_anomaly", False)
            attack   = det_resp.get("attack_type", "Normal")
            action   = dfn_resp.get("final_action", "—") if dfn_resp else "—"
            print(f"→ anomaly={anomaly} type={attack:12s} action={action}")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[BRIDGE] Stopped.")
    finally:
        client.close()


if __name__ == "__main__":
    run_bridge()