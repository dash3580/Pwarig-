# plot_twist_energy_bands.py
# Step 3: Plot twist energy distribution across radial bands over time

import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Config ---
CSV_FILE = "./output/twist_energy_radial_bands.csv"
OUTPUT_PLOT = "./output/twist_energy_bands_vs_time.png"

# --- Load data ---
df = pd.read_csv(CSV_FILE)

# --- Select radial band columns ---
band_cols = [col for col in df.columns if col.startswith("R")]

# --- Plot ---
plt.figure(figsize=(12, 6))
for col in band_cols:
    plt.plot(df['step'], df[col], label=col)

plt.xlabel("Simulation Step")
plt.ylabel("Twist Energy in Band")
plt.title("Radial Twist Energy Distribution Over Time")
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()

# --- Save ---
os.makedirs("./output", exist_ok=True)
plt.savefig(OUTPUT_PLOT)
print(f"✅ Saved: {OUTPUT_PLOT}")
