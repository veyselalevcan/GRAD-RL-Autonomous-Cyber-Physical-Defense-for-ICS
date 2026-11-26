"""C3 / C2 ablation (Reviewers 1 and 2): effect of the graph module and of the
safety weight w_s on response decisions, on the SWaT full-campaign stack
(FL-WGAN-GP-GRAD-RL/GRAD-RL, pipeline v2.1 with incident manager).

Variants (reference: results/swat/live, w_s = 1.5, graph on = live_analysis_v2.1.json):
  graph_off  betweenness C_B = 0 for every asset and no propagation along the
             topology (risk from the directly attacked asset only); the PPO policy
             and every other component unchanged
  ws_1.0     w_s = 1.0 (no safety overweighting)
  ws_2.0     w_s = 2.0
The whole attack period is streamed exactly as scripts/05_live_test.py does
(simulated operator approves gated actions after 10 s), then analysed with
scripts/07_live_analysis.py.
Usage: python c3_graph_ablation.py graph_off|ws_1.0|ws_2.0|ws_1.5
"""
import csv, importlib.util, json, sys, time
from pathlib import Path

import numpy as np

STACK = Path(r"D:\PhD_Project\FL-WGAN-GP-GRAD-RL\GRAD-RL")
sys.path.insert(0, str(STACK))
from gradrl import k3_graph  # noqa: E402
from gradrl.config import load_config  # noqa: E402
from gradrl.data import load_frame  # noqa: E402
from gradrl.gate import HUMAN  # noqa: E402
from gradrl.pipeline import GradRL  # noqa: E402

variant = sys.argv[1]
cfg = load_config("swat")
if variant == "graph_off":
    _init = k3_graph.Topology.__init__

    def init(self, path):
        _init(self, path)
        self.betweenness = {a: 0.0 for a in self.betweenness}
    k3_graph.Topology.__init__ = init
    k3_graph.Topology.most_probable_paths = lambda self, src: {src: (1.0, [src])}
elif variant.startswith("ws_"):
    cfg.raw["k3"]["safety_weight"] = float(variant[3:])
else:
    raise SystemExit("unknown variant")

live = f"live_ablation_{variant}"
out = cfg.results_dir / live
out.mkdir(parents=True, exist_ok=True)
df, _ = load_frame(cfg)
g = GradRL(cfg)
if "_file" in df:
    df = df[df["_file"] == "attack"].reset_index(drop=True)
X = df[g.features].to_numpy(dtype=np.float64)
pending, t0 = {}, time.time()
with open(out / "decisions.csv", "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["row", "t", "label", "attack", "cls", "split", "status", "k1_alarm", "k1_ratio", "invariants",
                "technique", "confidence", "target", "risk", "hazard", "action", "mode", "approved", "latency_ms"])
    for i in range(len(df)):
        s = time.perf_counter()
        d = g.step(X[i], str(df["_t"].iloc[i]))
        lat = (time.perf_counter() - s) * 1000
        approved = []
        for a, due in list(pending.items()):
            if i >= due:
                g.execute(a); approved.append(a); del pending[a]
        if d is None:
            continue
        if d.gate.mode == HUMAN and d.action != "MONITOR" and d.action not in pending:
            pending[d.action] = i + 10
        w.writerow([i, d.timestamp, df["_label"].iloc[i], int(df["_attack"].iloc[i]), df["_cls"].iloc[i],
                    df["_split"].iloc[i], d.status, int(d.extra.get("k1_alarm", d.k1_alarm)), round(d.k1_ratio, 3),
                    "|".join(v.id for v in d.invariants), d.technique, round(d.k2_conf, 3), d.target or "",
                    round(d.risk, 2), round(d.hazard, 3), d.action, d.gate.mode, "|".join(approved), round(lat, 2)])
        if i % 50000 == 0:
            print(f"  row {i}/{len(df)} {time.time() - t0:.0f}s", flush=True)
print(f"stream done {time.time() - t0:.0f}s", flush=True)

spec = importlib.util.spec_from_file_location("la", STACK / "scripts" / "07_live_analysis.py")
la = importlib.util.module_from_spec(spec); spec.loader.exec_module(la)
la.main("swat", 300, 300, live, f"live_analysis_ablation_{variant}.json")
