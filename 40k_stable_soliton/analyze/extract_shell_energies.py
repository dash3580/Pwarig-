# extract_shell_energies.py
# Extract twist energy inside shell bands from twist_energy_*.npy files

import numpy as np
import os
import re
import csv

# --- Configuration ---
DATA_DIR = "../npy_cycle"  # Change this to your actual data folder
OUTPUT_CSV = "./output/shell_energy_bands.csv"
DX = 0.2
CENTER = (48, 48, 48)
RADIUS_BINS = {
    "n1": (0.2, 0.4),
    "n2": (0.6, 0.8),
    "n3": (1.0, 1.2),
}

# --- Build radius grid ---
Z, Y, X = np.indices((96, 96, 96))
r = np.sqrt((X - CENTER[0])**2 + (Y - CENTER[1])**2 + (Z - CENTER[2])**2) * DX

# --- Collect energy data ---
pattern = re.compile(r"twist_energy_(\d+).npy")
results = []

for fname in sorted(os.listdir(DATA_DIR)):
    match = pattern.match(fname)
    if not match:
        continue
    step = int(match.group(1))
    fpath = os.path.join(DATA_DIR, fname)
    data = np.load(fpath)

    row = {"step": step}
    for shell, (rmin, rmax) in RADIUS_BINS.items():
        mask = (r >= rmin) & (r < rmax)
        row[shell] = float(np.sum(data[mask]))

    results.append(row)

# --- Save to CSV ---
os.makedirs("./output", exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["step"] + list(RADIUS_BINS.keys()))
    writer.writeheader()
    writer.writerows(results)

print(f"✅ Saved: {OUTPUT_CSV}")
