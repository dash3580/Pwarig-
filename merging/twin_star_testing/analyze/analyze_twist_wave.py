# analyze_twist_wave.py
import numpy as np
import os
import re
import csv

# --- Parameters ---
DX = 0.2
CENTER = (48, 48, 48)
DR = 0.2
R_MAX = 50.0

NGRID = 96
GRID_SHAPE = (NGRID, NGRID, NGRID)

BASE_DIR = "../npy_cycle"
OUTPUT_DIR = "output"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "twist_wave_shell_profile_analysis.csv")

# --- Ensure output folder exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Build radial grid ---
z, y, x = np.indices(GRID_SHAPE)
r = np.sqrt((x - CENTER[0])**2 + (y - CENTER[1])**2 + (z - CENTER[2])**2) * DX
r_bins = np.arange(0, R_MAX + DR, DR)
r_indices = np.digitize(r.ravel(), r_bins)

# --- Collect twist_wave_*.npy files ---
files = sorted(
    [f for f in os.listdir(BASE_DIR) if re.match(r"twist_wave_\d+\.npy", f)],
    key=lambda f: int(re.findall(r"\d+", f)[0])
)

# --- Main analysis ---
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "radius", "twist_wave_avg"])

    for fname in files:
        step = int(re.findall(r"\d+", fname)[0])
        full_path = os.path.join(BASE_DIR, fname)
        twist_wave = np.load(full_path)
        twist_abs = np.abs(twist_wave).ravel()

        for i in range(1, len(r_bins)):
            mask = r_indices == i
            avg_val = np.mean(twist_abs[mask]) if np.any(mask) else np.nan
            writer.writerow([step, r_bins[i - 1], avg_val])

        print(f"[✓] Processed {fname}")
