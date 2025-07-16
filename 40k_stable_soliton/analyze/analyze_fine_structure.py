# analyze_fine_structure.py
# Estimate PWARI-G analog of fine-structure constant from alpha_pwari and emitted energy

import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Config ---
INPUT_CSV = "./output/cycle_log.csv"
PLOT_OUTPUT = "./output/fine_structure_alpha_pwari.png"

# --- Load Data ---
df = pd.read_csv(INPUT_CSV)
df = df.sort_values("step")

# --- Rolling average (optional consistency)
WINDOW_SIZE = 1000
alpha_avg = df["alpha_pwari"].rolling(window=WINDOW_SIZE, min_periods=1).mean()

# --- Final emitted vs final soliton energy ratio ---
final_emitted = df["emitted_energy_cumulative"].iloc[-1]
final_soliton = df["soliton_energy"].iloc[-1]
emitted_ratio = final_emitted / final_soliton

print(f"📊 Final emitted / soliton energy ratio: {emitted_ratio:.6f}")
print(f"📊 Average alpha_PWARI: {alpha_avg.mean():.6f}")

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["alpha_pwari"], alpha=0.3, label="Raw α_PWARI")
plt.plot(df["step"], alpha_avg, color="red", label=f"{WINDOW_SIZE}-step Average")

plt.axhline(1/137, color="green", linestyle="--", label="1/137 ≈ α (known)")
plt.xlabel("Step")
plt.ylabel("$\\alpha_{\\text{PWARI}} = E_{\\text{twist}} / E_{\\text{soliton}}$")
plt.title("PWARI-G Analog of Fine-Structure Constant Over Time")
plt.grid(True)
plt.legend()

os.makedirs("./output", exist_ok=True)
plt.tight_layout()
plt.savefig(PLOT_OUTPUT)
print(f"✅ Saved plot to {PLOT_OUTPUT}")