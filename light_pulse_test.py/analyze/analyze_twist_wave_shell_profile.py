# analyze_twist_wave_shell_profile.py
import numpy as np
import os
import re
import csv

# --- Config ---
GRID_CENTER = (48, 48, 48)
DX = 0.2
R_MAX = 50
DR = 0.2
INPUT_FOLDER = "../pulse_output"
OUTPUT_CSV = "output/twist_wave_shell_profile_analysis.csv"

def compute_radial_profile(data, fname):
    shape = data.shape
    z, y, x = np.indices(shape)
    r = np.sqrt((x - GRID_CENTER[0])**2 + (y - GRID_CENTER[1])**2 + (z - GRID_CENTER[2])**2) * DX
    r_bins = np.arange(0, R_MAX + DR, DR)

    avg_profile = np.zeros(len(r_bins) - 1)
    max_profile = np.zeros(len(r_bins) - 1)

    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
        if np.any(mask):
            values = data[mask]
            avg_profile[i] = np.mean(np.abs(values))
            max_profile[i] = np.max(np.abs(values))
        else:
            avg_profile[i] = np.nan
            max_profile[i] = np.nan

    if np.all(np.isnan(avg_profile)):
        print(f"⚠️  WARNING: No valid data in any radial bin ({fname})")
    elif np.max(max_profile) < 1e-6:
        print(f"⚠️  Low signal in {fname} (max < 1e-6)")

    return r_bins[:-1], avg_profile, max_profile

def scan_twist_wave_files():
    os.makedirs("output", exist_ok=True)
    files = [f for f in os.listdir(INPUT_FOLDER) if re.match(r"twist_wave_\d+\.npy", f)]
    files.sort(key=lambda f: int(re.findall(r"\d+", f)[0]))

    if not files:
        print("❌ No twist_wave_*.npy files found in", INPUT_FOLDER)
        return

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "step", "radius", "twist_wave_avg", "twist_wave_max"])

        for fname in files:
            print(f"🔍 Processing {fname}")
            step = int(re.findall(r"\d+", fname)[0])
            data = np.load(os.path.join(INPUT_FOLDER, fname))
            radii, avg_vals, max_vals = compute_radial_profile(data, fname)
            for r, a, m in zip(radii, avg_vals, max_vals):
                writer.writerow([fname, step, r, a, m])

    print(f"✅ Output written to: {OUTPUT_CSV}")

if __name__ == "__main__":
    scan_twist_wave_files()
