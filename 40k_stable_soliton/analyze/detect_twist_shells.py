# detect_twist_shells.py
# Full radial twist shell structure detection for PWARI-G hydrogen soliton

import os
import numpy as np
import csv
from scipy.signal import find_peaks
from scipy.ndimage import center_of_mass

# --- Config ---
INPUT_FOLDER = "../npy_cycle"
OUTPUT_FOLDER = "./output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "twist_shell_structure_over_time.csv")
DX = 0.2
CENTER_METHOD = "COM"  # Options: "fixed" or "COM"
FIXED_CENTER = (48, 48, 48)  # used if CENTER_METHOD == "fixed"
MAX_RADIUS = 50.0
DR = 0.2
PEAK_MIN_HEIGHT_RATIO = 0.05  # minimum peak height relative to max
PEAK_MIN_DISTANCE = 3  # in radial bins

# --- Prepare radial bins ---
r_bins = np.arange(0, MAX_RADIUS + DR, DR)

# --- Get twist energy files ---
twist_files = [f for f in os.listdir(INPUT_FOLDER) if f.startswith("twist_energy_") and f.endswith(".npy")]
twist_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

results = []

for fname in twist_files:
    try:
        step = int(''.join(filter(str.isdigit, fname)))
        data = np.load(os.path.join(INPUT_FOLDER, fname))

        # Determine center of soliton
        if CENTER_METHOD == "COM":
            phi = np.load(os.path.join(INPUT_FOLDER, f"phi_{step:04d}.npy"))
            center = center_of_mass(phi)
        else:
            center = FIXED_CENTER

        z, y, x = np.indices(data.shape)
        r = np.sqrt(
            (x - center[0])**2 +
            (y - center[1])**2 +
            (z - center[2])**2
        ) * DX

        r_indices = np.digitize(r.ravel(), r_bins)
        energy_flat = data.ravel()

        radial_energy = np.zeros(len(r_bins))
        counts = np.zeros(len(r_bins))

        for i in range(len(energy_flat)):
            idx = r_indices[i]
            if 0 <= idx < len(r_bins):
                radial_energy[idx] += energy_flat[i]
                counts[idx] += 1

        profile = np.where(counts > 0, radial_energy / counts, 0.0)
        height_thresh = profile.max() * PEAK_MIN_HEIGHT_RATIO
        peaks, props = find_peaks(profile, height=height_thresh, distance=PEAK_MIN_DISTANCE)

        row = {"step": step}
        for i, p in enumerate(peaks):
            if i >= 5: break  # limit to first 5 peaks
            row[f"r_peak_{i+1}"] = round(r_bins[p], 3)
            row[f"height_{i+1}"] = round(props["peak_heights"][i], 6)

        row["n_peaks"] = len(peaks)
        row["twist_total"] = float(np.sum(data))
        results.append(row)
        print(f"[✓] Step {step:5d}: {len(peaks)} peak(s) detected")

    except Exception as e:
        print(f"[!] Error in {fname}: {e}")

# --- Save CSV ---
if results:
    keys = sorted(set(k for row in results for k in row.keys()))
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\n✅ Shell detection complete. Output written to {OUTPUT_CSV}")
else:
    print("❌ No results to write. Check input folder and parameters.")
