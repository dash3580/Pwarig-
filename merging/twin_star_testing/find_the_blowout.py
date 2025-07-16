import os
import numpy as np
import csv

# --- Config ---
INPUT_DIR = "../npy_cycle"  # Update this
OUTPUT_CSV = "npy_grid_dump.csv"

# --- Gather all .npy files ---
npy_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npy")]
npy_files.sort()

# --- Open CSV for writing ---
with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file", "x", "y", "z", "value"])

    for fname in npy_files:
        try:
            data = np.load(os.path.join(INPUT_DIR, fname))
            if data.ndim != 3:
                print(f"Skipping {fname} (not 3D)")
                continue

            nx, ny, nz = data.shape
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        val = data[x, y, z]
                        writer.writerow([fname, x, y, z, val])

        except Exception as e:
            print(f"Error reading {fname}: {e}")
