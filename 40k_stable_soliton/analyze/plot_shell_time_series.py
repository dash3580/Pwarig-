import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Config ---
INPUT_CSV = "output/twist_wave_phase_by_shell.csv"
OUTPUT_PNG = "output/shell_phase_amplitude_vs_time.png"

# --- Load Data ---
df = pd.read_csv(INPUT_CSV)

# --- Plot ---
plt.figure(figsize=(10, 6))
for col in df.columns:
    if col != "time":
        plt.plot(df["time"], df[col], label=col)

plt.xlabel("Time")
plt.ylabel("Average Twist Wave Amplitude")
plt.title("Twist Wave Amplitude Over Time by Shell Radius")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()

# --- Save ---
os.makedirs("output", exist_ok=True)
plt.savefig(OUTPUT_PNG)
print(f"Saved: {OUTPUT_PNG}")