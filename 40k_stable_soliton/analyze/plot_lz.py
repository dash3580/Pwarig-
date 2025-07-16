# plot_twist_wave_Lz.py
# Plot total angular momentum Lz in twist_wave vs simulation time

import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Config ---
CSV_FILE = "./output/twist_wave_Lz_over_time.csv"
OUTPUT_PLOT = "./output/Lz_total_vs_time.png"

# --- Load data ---
df = pd.read_csv(CSV_FILE)

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(df['step'], df['Lz_total'], color='orange', label="Total Lz (twist_wave)")
plt.xlabel("Simulation Step")
plt.ylabel("Total Angular Momentum $L_z$")
plt.title("Total Angular Momentum in Twist Wave Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()

# --- Save ---
os.makedirs("./output", exist_ok=True)
plt.savefig(OUTPUT_PLOT)
print(f"✅ Saved: {OUTPUT_PLOT}")
