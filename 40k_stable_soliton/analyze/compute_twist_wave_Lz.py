# compute_twist_wave_Lz.py
# Compute angular momentum Lz from twist_wave field over time

import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import csv

# --- Config ---
NPY_DIR = "../npy_cycle"
OUTPUT_CSV = "./output/twist_wave_Lz_over_time.csv"
OUTPUT_IMG_DIR = "./output/Lz_maps"
SLICE_Z = 48  # middle of the grid (z-axis)
DX = 0.2

# --- Setup ---
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
twist_files = sorted(glob(os.path.join(NPY_DIR, "twist_wave_*.npy")))

# --- Grid setup ---
GRID_SIZE = 96
x = np.linspace(0, GRID_SIZE-1, GRID_SIZE) * DX
X, Y = np.meshgrid(x, x)
x_centered = X - np.mean(x)
y_centered = Y - np.mean(x)

# --- Prepare CSV log ---
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["step", "Lz_total"])

    for path in twist_files:
        step = int(path.split("_")[-1].split(".")[0])
        twist = np.load(path)

        # --- 2D slice at fixed Z ---
        psi = twist[:, :, SLICE_Z]
        dpsi_dx = np.gradient(psi, DX, axis=0)
        dpsi_dy = np.gradient(psi, DX, axis=1)

        # --- Angular momentum density (Lz)
        Lz_density = x_centered * dpsi_dy - y_centered * dpsi_dx
        Lz_total = np.sum(Lz_density)

        # --- Save to CSV ---
        writer.writerow([step, Lz_total])

        # --- Optional heatmap per step ---
        plt.figure(figsize=(5, 4))
        plt.imshow(Lz_density, extent=[x[0], x[-1], x[0], x[-1]], origin='lower', cmap='RdBu')
        plt.colorbar(label="Lz Density")
        plt.title(f"Lz Density (Step {step})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_IMG_DIR}/Lz_density_{step:05d}.png")
        plt.close()

print(f"✅ Finished. Results saved to {OUTPUT_CSV} and per-step maps in {OUTPUT_IMG_DIR}")