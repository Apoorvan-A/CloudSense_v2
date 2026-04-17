
import sys, os
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader    import load_dataset
from models_torch   import (LSTMModel, CNNLSTMModel, BiLSTMModel,
                             TransformerModel, CEEMDANBiLSTM)
from train_evaluate import (train_model, evaluate_model,
                             train_ceemdan_model, evaluate_ceemdan_model,
                             DEVICE)

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

LOOK_BACK  = 48
EPOCHS     = 150
LR         = 1e-3
BATCH      = 64
PATIENCE   = 15

sns.set_theme(style="whitegrid", palette="muted")

print("=" * 60)
print("  CloudSense  |  Cloud Resource Usage Analytics")
print("  Apoorvan A  |  23BCE0755  |  VIT Vellore")
print(f"  Device: {DEVICE}")
print("=" * 60)

# Step 1: Data
print("\n[1/4] Loading dataset …")
data = load_dataset(data_dir="data", look_back=LOOK_BACK)
scaler = data["scaler"]

# Step 2: Train all models
print("\n[2/4] Training models …")

MODELS = [
    ("LSTM (Bansal 2023)",
     LSTMModel(hidden_size=64, num_layers=2, dropout=0.2, look_back=LOOK_BACK)),
    ("CNN-LSTM (Bi 2023)",
     CNNLSTMModel(cnn_filters=32, hidden_size=64, num_layers=2,
                  dropout=0.2, look_back=LOOK_BACK)),
    ("Bi-LSTM (Xing 2024)",
     BiLSTMModel(hidden_size=64, num_layers=2, dropout=0.2, look_back=LOOK_BACK)),
    ("Transformer (Lackinger 2024)",
     TransformerModel(d_model=64, nhead=4, num_layers=2,
                      dim_feedforward=128, dropout=0.1, look_back=LOOK_BACK)),
    ("CEEMDAN+CNN-BiLSTM (Proposed)",
     CEEMDANBiLSTM(n_imfs=6, look_back=LOOK_BACK, hidden=32)),
]

results    = {}
histories  = {}

for name, model in MODELS:
    print(f"\n  → {name}")
    if "CEEMDAN" in name:
        model, hist = train_ceemdan_model(
            model, data, epochs=EPOCHS, lr=LR,
            batch_size=BATCH, patience=PATIENCE, verbose=True)
        preds, y_true, metrics = evaluate_ceemdan_model(model, data, scaler)
    else:
        model, hist = train_model(
            model, data, epochs=EPOCHS, lr=LR,
            batch_size=BATCH, patience=PATIENCE, verbose=True)
        preds, y_true, metrics = evaluate_model(model, data, scaler)

    results[name]   = {"metrics": metrics, "preds": preds, "y_true": y_true}
    histories[name] = hist
    print(f"     MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}"
          f"  MAPE={metrics['MAPE']:.2f}%  R²={metrics['R2']:.4f}")

# Step 3: Save metrics
print("\n[3/4] Saving metrics …")
rows = [{"Model": n, **r["metrics"]} for n, r in results.items()]
pd.DataFrame(rows).to_csv("results/metrics.csv", index=False)

with open("results/summary.txt", "w") as f:
    f.write("=" * 65 + "\n")
    f.write("  CloudSense  –  Comparative Results\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"{'Model':<42} {'MAE':>6} {'RMSE':>6} {'MAPE%':>7} {'R²':>7}\n")
    f.write("-" * 65 + "\n")
    for row in rows:
        star = " ★" if "Proposed" in row["Model"] else "  "
        f.write(f"{row['Model'][:42]:<42} "
                f"{row['MAE']:>6.3f} {row['RMSE']:>6.3f} "
                f"{row['MAPE']:>7.2f} {row['R2']:>7.4f}{star}\n")
    f.write("\n★ = Proposed model\n")

print("  → results/metrics.csv")
print("  → results/summary.txt")

# Step 4: Figures
print("\n[4/4] Generating figures …")

MODEL_COLORS = {
    "LSTM (Bansal 2023)"           : "#e76f51",
    "CNN-LSTM (Bi 2023)"           : "#2a9d8f",
    "Bi-LSTM (Xing 2024)"          : "#e9c46a",
    "Transformer (Lackinger 2024)" : "#457b9d",
    "CEEMDAN+CNN-BiLSTM (Proposed)": "#d62828",
    "Actual"                       : "#264653",
}

def savefig(name):
    path = f"figures/{name}"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {path}")

# Fig 1 – Raw data
df = data["raw_df"]
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(df["cpu_util"].values[:2016], color="#264653", lw=0.7)
ax.fill_between(range(2016), df["cpu_util"].values[:2016], alpha=0.1, color="#264653")
ax.set_title("Real AWS EC2 CPU Utilisation (Numenta NAB Dataset) – 7-day snapshot",
             fontweight="bold")
ax.set_xlabel("Time step (5-min intervals)")
ax.set_ylabel("CPU Utilisation (%)")
savefig("fig1_raw_data.png")

# Fig 2 – CEEMDAN decomposition
from data_loader import _ceemdan_decompose
seg  = data["train_s"][:2016]
imfs = _ceemdan_decompose(seg, n_imfs=5)
fig, axes = plt.subplots(len(imfs) + 1, 1, figsize=(13, 2.2 * (len(imfs) + 1)),
                          sharex=True)
fig.suptitle("CEEMDAN Decomposition of CPU Utilisation Signal", fontweight="bold")
axes[0].plot(seg, color="#264653", lw=0.7)
axes[0].set_ylabel("Original")
colors = ["#e76f51", "#f4a261", "#2a9d8f", "#457b9d", "#e9c46a", "#6d6875"]
labels = [f"IMF {i+1}" for i in range(len(imfs)-1)] + ["Residue (Trend)"]
for ax, imf, lbl, col in zip(axes[1:], imfs, labels, colors):
    ax.plot(imf, color=col, lw=0.7)
    ax.set_ylabel(lbl, fontsize=8)
    ax.axhline(0, color="#ccc", lw=0.5, ls="--")
axes[-1].set_xlabel("Time step")
plt.tight_layout()
savefig("fig2_ceemdan.png")

# Fig 3 – Predictions overlay
proposed_key = "CEEMDAN+CNN-BiLSTM (Proposed)"
n_show = min(576, len(results[proposed_key]["y_true"]))
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(results[proposed_key]["y_true"][:n_show],
        color=MODEL_COLORS["Actual"], lw=1.8, label="Actual", zorder=10)
for name, col in MODEL_COLORS.items():
    if name == "Actual" or name not in results: continue
    lw = 1.8 if name == proposed_key else 0.9
    ls = "-" if name == proposed_key else "--"
    ax.plot(results[name]["preds"][:n_show], color=col, lw=lw, ls=ls,
            alpha=1.0 if name == proposed_key else 0.65,
            label=name, zorder=11 if name == proposed_key else 9)
ax.set_title("CPU Utilisation: Actual vs Predicted — All Models (2-day window)",
             fontweight="bold")
ax.set_xlabel("Time step (5-min intervals)")
ax.set_ylabel("CPU (%)")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
savefig("fig3_predictions_overlay.png")

# Fig 4 – Loss curves
fig, ax = plt.subplots(figsize=(11, 5))
for name, hist in histories.items():
    val_loss = hist if isinstance(hist[0], float) else \
               [np.mean(h) for h in hist]  # CEEMDAN returns list of lists
    ax.plot(val_loss, label=name,
            color=MODEL_COLORS.get(name, "#888"),
            lw=2.0 if "Proposed" in name else 1.2,
            ls="-" if "Proposed" in name else "--")
ax.set_title("Validation Loss Convergence — All Models", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss (normalised)")
ax.legend(fontsize=9)
plt.tight_layout()
savefig("fig4_loss_curves.png")

# Figs 5–8 — Metric bars
metrics_list = ["MAE", "RMSE", "MAPE", "R2"]
metric_titles = {
    "MAE" : "MAE — Mean Absolute Error (lower is better)",
    "RMSE": "RMSE — Root Mean Squared Error (lower is better)",
    "MAPE": "MAPE % — Mean Abs. Percentage Error (lower is better)",
    "R2"  : "R² — Coefficient of Determination (higher is better)",
}
names  = list(results.keys())
colors_b = [MODEL_COLORS.get(n, "#888") for n in names]
for i, metric in enumerate(metrics_list):
    fig, ax = plt.subplots(figsize=(11, 5))
    vals = [results[n]["metrics"][metric] for n in names]
    bars = ax.bar(range(len(names)), vals, color=colors_b, edgecolor="white")
    prop_idx = names.index(proposed_key) if proposed_key in names else -1
    if prop_idx >= 0:
        bars[prop_idx].set_edgecolor("#d62828")
        bars[prop_idx].set_linewidth(2.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names],
                       rotation=15, ha="right", fontsize=9)
    ax.set_title(metric_titles[metric], fontweight="bold")
    plt.tight_layout()
    savefig(f"fig{5+i}_{metric.lower()}_bars.png")

print("\n" + "=" * 60)
print("  DONE!  →  figures/  and  results/")
print("=" * 60)

print("\nFINAL RESULTS:")
print(f"{'Model':<42} {'MAE':>6} {'RMSE':>6} {'MAPE%':>7} {'R²':>7}")
print("-" * 65)
for row in rows:
    star = " ★ PROPOSED" if "Proposed" in row["Model"] else ""
    print(f"{row['Model'][:42]:<42} "
          f"{row['MAE']:>6.3f} {row['RMSE']:>6.3f} "
          f"{row['MAPE']:>7.2f} {row['R2']:>7.4f}{star}")
