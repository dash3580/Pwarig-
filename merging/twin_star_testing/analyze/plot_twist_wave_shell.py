import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Paths ---
INPUT_CSV = "output/twist_wave_phase_by_shell.csv"

OUTPUT_DIR = "analyze/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(INPUT_CSV)

# --- Plot Raw Signal ---
plt.figure(figsize=(12, 6))
for col in df.columns[1:]:
    plt.plot(df["time"], df[col], label=col)
plt.title("Raw Shell Phase Amplitudes (θ)")
plt.xlabel("Time")
plt.ylabel("Average θ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "raw_theta_shells.png"))
plt.close()

# --- Plot Mean-Subtracted Signal ---
plt.figure(figsize=(12, 6))
for col in df.columns[1:]:
    centered = df[col] - df[col].mean()
    plt.plot(df["time"], centered, label=col)
plt.title("Mean-Subtracted Twist Field θ by Shell Radius")
plt.xlabel("Time")
plt.ylabel("θ - mean(θ)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "centered_theta_shells.png"))
plt.close()

print("Saved: raw_theta_shells.png and centered_theta_shells.png to analyze/output/")
