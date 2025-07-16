# track_twist_energy_field.py
import numpy as np
import os
import re
import csv

# --- Config ---
INPUT_FOLDER = "../npy_cycle"
OUTPUT_CSV = "output/twist_energy_outer_region.csv"
RADIUS_CUTOFF = 1.5  # changed this but numbers are to small to register, need better code, to see futher out
DX = 0.2  # Grid spacing
GRID_CENTER = (48, 48, 48)

# --- Setup ---
os.makedirs("output", exist_ok=True)

twist_files = sorted(
    [f for f in os.listdir(INPUT_FOLDER) if re.match(r"twist_energy_\d+\.npy", f)],
    key=lambda x: int(re.findall(r"\d+", x)[0])
)

if not twist_files:
    print("❌ No twist_energy_*.npy files found in:", INPUT_FOLDER)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "outer_twist_energy"])

    for fname in twist_files:
        step = int(re.findall(r"\d+", fname)[0])
        full_path = os.path.join(INPUT_FOLDER, fname)

        try:
            data = np.load(full_path)
            z, y, x = np.indices(data.shape)
            r = np.sqrt(
                (x - GRID_CENTER[0])**2 +
                (y - GRID_CENTER[1])**2 +
                (z - GRID_CENTER[2])**2
            ) * DX

            mask = r > RADIUS_CUTOFF
            outer_energy = np.sum(data[mask])

            writer.writerow([step, outer_energy])
            print(f"✅ Processed {fname:25} | Step {step:5d} | Energy = {outer_energy:.3e}")
        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")
