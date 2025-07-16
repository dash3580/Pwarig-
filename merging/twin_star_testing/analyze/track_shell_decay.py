# track_shell_decay.py
# Analyze shell detection lifespan and energy transitions from summary CSV

import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Config ---
LIFETIME_FILE = "./output/twist_shell_structure_over_time.csv"
ENERGY_FILE = "./output/cycle_log.csv"
OUTPUT_PLOT_LIFETIME = "./output/shell_lifetimes.png"
OUTPUT_PLOT_ENERGY = "./output/total_twist_energy_annotated.png"

# --- Load data ---
df_life = pd.read_csv(LIFETIME_FILE)
df_energy = pd.read_csv(ENERGY_FILE)

# --- Detect shell columns ---
shell_cols = [col for col in df_life.columns if col.startswith("r_peak_")]
if not shell_cols:
    raise ValueError("No 'r_peak_N' columns found. Available: " + ", ".join(df_life.columns))

# --- Track shell presence ---
shell_lifetimes = {}
for col in shell_cols:
    mask = df_life[col].notna()
    if mask.any():
        first = df_life.loc[mask, 'step'].iloc[0]
        last = df_life.loc[mask, 'step'].iloc[-1]
        shell_lifetimes[col] = (first, last)

# --- Plot shell lifetimes ---
plt.figure(figsize=(10, 3))
for i, (shell, (start, end)) in enumerate(shell_lifetimes.items()):
    plt.hlines(y=i, xmin=start, xmax=end, label=shell, linewidth=4)
plt.yticks(range(len(shell_lifetimes)), list(shell_lifetimes.keys()))
plt.xlabel("Simulation Step")
plt.title("Detected Shell Presence Over Time")
plt.grid(True)
plt.tight_layout()
os.makedirs("./output", exist_ok=True)
plt.savefig(OUTPUT_PLOT_LIFETIME)

# --- Plot total twist energy and annotate shell transitions ---
plt.figure(figsize=(10, 5))
plt.plot(df_energy['step'], df_energy['twist_energy'], label="Twist Energy")
colors = ['red', 'orange', 'green']
for i, (shell, (start, end)) in enumerate(shell_lifetimes.items()):
    plt.axvline(x=end, color=colors[i % len(colors)], linestyle='--', label=f"{shell} ends")

plt.xlabel("Simulation Step")
plt.ylabel("Total Twist Energy")
plt.title("Twist Energy and Shell Collapse Events")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT_ENERGY)

print("✅ Saved shell lifetimes and total twist energy plot to ./output")
