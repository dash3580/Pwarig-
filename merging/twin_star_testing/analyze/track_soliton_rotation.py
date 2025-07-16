# track_theta_phase_rotation.py
# Track twist phase field (theta) at the soliton core over time

import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from numpy import unwrap, angle

# --- Config ---
NPY_DIR = "../npy_cycle"
CENTER = (48, 48, 48)
OUTPUT_CSV = "./output/theta_phase_vs_time.csv"
OUTPUT_PLOT = "./output/theta_phase_unwrapped.png"

# --- Load files ---
theta_files = sorted(glob(os.path.join(NPY_DIR, "theta_*.npy")))
steps = [int(f.split("_")[-1].split(".")[0]) for f in theta_files]

# --- Extract theta value at center ---
phase_vals = []
for f in theta_files:
    theta = np.load(f)
    val = theta[CENTER]
    phase = np.angle(np.exp(1j * val))  # wrap to [-pi, pi]
    phase_vals.append(phase)

# --- Unwrap and save ---
unwrapped = unwrap(phase_vals)
np.savetxt(OUTPUT_CSV, np.column_stack([steps, unwrapped]), delimiter=",", header="step,theta_unwrapped", comments="")

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(steps, unwrapped, label="Core theta phase (unwrapped)")
plt.xlabel("Simulation Step")
plt.ylabel("Unwrapped Phase $\\theta$")
plt.title("Twist Field Core Phase Rotation Over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
print(f"✅ Saved: {OUTPUT_PLOT}")