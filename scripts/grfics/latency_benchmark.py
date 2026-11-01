# latency_benchmark.py - GRAD-RL pipeline latency measurement
# Run: python latency_benchmark.py
# Outputs: docs/results/latency_report.txt, docs/results/latency_boxplot.png


import time
import json
import os
import numpy as np
import joblib
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# ---------------------------------------------
#  PATHS
# ---------------------------------------------
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gradrl import paths
PROJECT_DIR  = str(paths.ROOT)
MODEL_DIR    = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR  = str(paths.RESULTS_DIR / "grfics")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_WARMUP  = 20     # warmup iterations (JIT compilation etc.)
N_MEASURE = 200    # measurement iterations

# ---------------------------------------------
#  LOAD MODELS
# ---------------------------------------------
print("Loading models...")
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_grfics_live.pkl"))
lstm   = tf.keras.models.load_model(
             os.path.join(MODEL_DIR, "detective_lstm_grfics_live.keras"))
clf    = joblib.load(
             os.path.join(MODEL_DIR, "detective_classifier_grfics_live.pkl"))
ppo    = PPO.load(os.path.join(MODEL_DIR, "rl_defender_grfics.zip"))

with open(os.path.join(MODEL_DIR, "threshold_grfics_live.json")) as f:
    threshold = json.load(f)["threshold"]

N_FEATURES   = 13
TIME_STEPS   = 10

# ---------------------------------------------
#  SYNTHETIC SINGLE-SAMPLE INPUT
#  Represents one real-time Modbus poll arriving
#  at the detective API after 10 frames are buffered
# ---------------------------------------------
rng          = np.random.default_rng(42)
raw_sample   = rng.uniform(0, 1, (1, N_FEATURES)).astype(np.float32)
scaled       = scaler.transform(raw_sample)              # (1, 13)
seq          = np.repeat(scaled, TIME_STEPS, axis=0)     # simulate rolling window
seq_batch    = seq.reshape(1, TIME_STEPS, N_FEATURES)    # (1, 10, 13)
ppo_obs      = scaled[0]                                 # (13,)


# ---------------------------------------------
#  MEASUREMENT HELPER
# ---------------------------------------------
def measure(fn, n_warmup=N_WARMUP, n_measure=N_MEASURE):
    """Run fn() and return list of latencies in ms."""
    for _ in range(n_warmup):
        fn()
    lats = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        fn()
        lats.append((time.perf_counter() - t0) * 1000.0)
    return lats


# ---------------------------------------------
#  COMPONENT BENCHMARKS
# ---------------------------------------------
print("Benchmarking scaler...")
lat_scaler = measure(lambda: scaler.transform(raw_sample))

print("Benchmarking LSTM-AE forward pass...")
lat_lstm = measure(
    lambda: lstm.predict(seq_batch, verbose=0))

print("Benchmarking MSE computation + threshold...")
def _mse_thresh():
    recons = lstm.predict(seq_batch, verbose=0)
    mse    = float(np.mean(np.power(seq_batch - recons, 2)))
    return mse > threshold
lat_mse = measure(_mse_thresh)

print("Benchmarking XGBoost classify...")
lat_xgb = measure(lambda: clf.predict(scaled))

print("Benchmarking PPO inference...")
lat_ppo = measure(lambda: ppo.predict(ppo_obs, deterministic=True))

print("Benchmarking end-to-end pipeline...")
def _end_to_end():
    s       = scaler.transform(raw_sample)
    sq      = np.repeat(s, TIME_STEPS, axis=0).reshape(1, TIME_STEPS, N_FEATURES)
    recons  = lstm.predict(sq, verbose=0)
    mse     = float(np.mean(np.power(sq - recons, 2)))
    is_anom = mse > threshold
    if is_anom:
        clf.predict(s)
    ppo.predict(s[0], deterministic=True)
lat_e2e = measure(_end_to_end)


# ---------------------------------------------
#  STATISTICS
# ---------------------------------------------
def stats(lats):
    arr = np.array(lats)
    return {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "p50":    float(np.percentile(arr, 50)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
    }

components = {
    "Scaler (MinMaxScaler)":       lat_scaler,
    "LSTM-AE forward pass":        lat_lstm,
    "MSE + threshold decision":    lat_mse,
    "XGBoost classify":            lat_xgb,
    "PPO policy inference":        lat_ppo,
    "End-to-end pipeline":         lat_e2e,
}



# ---------------------------------------------
#  HELPER
# ---------------------------------------------
def _get_cpu_info():
    try:
        import platform
        return platform.processor() or platform.machine()
    except Exception:
        return "Unknown CPU"

# ---------------------------------------------
#  TEXT REPORT
# ---------------------------------------------
lines = [
    "=" * 65,
    "  GRAD-RL: Inference Latency Benchmark",
    f"  Hardware: {_get_cpu_info()}",
    f"  Warmup: {N_WARMUP}  |  Measurements: {N_MEASURE}",
    "=" * 65,
    f"\n  {'Component':<35s}  {'Mean':>7s}  {'Std':>6s}  {'P50':>7s}  {'P95':>7s}",
    f"  {'-'*63}",
]

for name, lats in components.items():
    s = stats(lats)
    lines.append(
        f"  {name:<35s}  {s['mean']:>6.2f}ms"
        f"  {s['std']:>5.2f}  {s['p50']:>6.2f}ms  {s['p95']:>6.2f}ms"
    )

# Paper-ready table format
lines += [
    "\n",
    "  LaTeX-ready table rows (for Table II in paper):",
    "  " + "-" * 60,
]
for name, lats in list(components.items())[:-1]:  # exclude e2e row
    s = stats(lats)
    lines.append(
        f"  {name:<33s} & {s['mean']:.0f} & {s['std']:.1f} \\\\"
    )
e2e = stats(lat_e2e)
lines.append(
    f"  \\textbf{{Total (end-to-end)}} & "
    f"\\textbf{{{e2e['mean']:.0f}}} & \\textbf{{{e2e['std']:.1f}}} \\\\"
)

report_text = "\n".join(lines)
print(report_text)

report_path = os.path.join(RESULTS_DIR, "latency_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\n✅ Report → {report_path}")


# ---------------------------------------------
#  BOX PLOT FIGURE
# ---------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#161b22")

labels_short = [
    "Scaler", "LSTM-AE\nforward", "MSE+\nthreshold",
    "XGBoost", "PPO\npolicy", "End-to-\nend"
]
data = [np.array(v) for v in components.values()]
bp = ax.boxplot(
    data, labels=labels_short, patch_artist=True,
    medianprops=dict(color="#f0883e", linewidth=2),
    whiskerprops=dict(color="#8b949e"),
    capprops=dict(color="#8b949e"),
    flierprops=dict(marker=".", color="#8b949e", alpha=0.3),
)

colors = ["#238636", "#238636", "#238636", "#238636", "#238636", "#58a6ff"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(1000, color="#f85149", linestyle="--", linewidth=1.2,
           label="1000ms (1 Hz Modbus polling window)", alpha=0.8)
ax.set_title("GRAD-RL Pipeline Inference Latency (GRFICSv3, CPU-only)",
             color="#e6edf3", fontsize=11)
ax.set_ylabel("Latency (ms)", color="#8b949e")
ax.tick_params(colors="#8b949e")
ax.legend(fontsize=8, labelcolor="#e6edf3",
          facecolor="#161b22", edgecolor="#30363d")
ax.set_ylim(bottom=0, top=200)  # focus on sub-200ms range
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")

fig_path = os.path.join(RESULTS_DIR, "latency_boxplot.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Figure → {fig_path}")