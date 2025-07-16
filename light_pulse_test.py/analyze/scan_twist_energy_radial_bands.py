import os
import numpy as np
import re
import csv

# --- Configuration ---
INPUT_FOLDER = "../npy_cycle"  # Local relative path to folder with .npy files
OUTPUT_CSV = "analyze/output/twist_energy_radial_bands.csv"
DX = 0.2
GRID_CENTER = (48, 48, 48)
BAND_WIDTH = 1.0
R_MIN = 0.0
R_MAX = 10.0
PRECISION = 12  # Decimal places in output

# --- Prepare Bands ---
radii = np.arange(R_MIN, R_MAX, BAND_WIDTH)
bands = [(r, r + BAND_WIDTH) for r in radii]

# --- Ensure output folder exists ---
os.makedirs("analyze/output", exist_ok=True)

# --- Get list of twist energy files ---
twist_files = sorted(
    [f for f in os.listdir(INPUT_FOLDER) if re.match(r"twist_energy_\d+\.npy", f)],
    key=lambda x: int(re.findall(r"\d+", x)[0])
)

# --- Scan and Log ---
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["step"] + [f"R{low:.1f}-{high:.1f}" for (low, high) in bands]
    writer.writerow(header)

    for fname in twist_files:
        step = int(re.findall(r"\d+", fname)[0])
        data = np.load(os.path.join(INPUT_FOLDER, fname))

        z, y, x = np.indices(data.shape)
        r = np.sqrt((x - GRID_CENTER[0])**2 +
                    (y - GRID_CENTER[1])**2 +
                    (z - GRID_CENTER[2])**2) * DX

        band_energies = []
        for (low, high) in bands:
            mask = (r >= low) & (r < high)
            energy = np.sum(data[mask])
            band_energies.append(f"{energy:.{PRECISION}e}")

        writer.writerow([step] + band_energies)
        print(f"[✓] {fname} analyzed")
