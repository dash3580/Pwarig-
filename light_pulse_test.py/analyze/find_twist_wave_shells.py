# find_twist_wave_shells.py
import os
import numpy as np
import csv
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# --- Constants ---
DX = 0.2
INPUT_DIR = "../pulse_output"
OUTPUT_DIR = "output"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "shell_structure_detected.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Get twist_wave files ---
files = [f for f in os.listdir(INPUT_DIR) if f.startswith("theta_") and f.endswith(".npy")]
files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

# --- Storage ---
results = []

for fname in files:
    try:
        full_path = os.path.join(INPUT_DIR, fname)
        data = np.load(full_path)

        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError("NaNs or Infs detected in array.")

        grad = np.gradient(data, DX)
        grad_mag_sq = sum(g**2 for g in grad)

        grid = np.indices(data.shape).astype(np.float32)
        center = np.array(data.shape)[:, None, None, None] / 2.0
        r = np.sqrt(np.sum((grid - center)**2, axis=0)) * DX
        max_radius = np.linalg.norm(np.array(data.shape)) / 2 * DX

        r_bins = np.linspace(0, np.max(r), 300)
        r_indices = np.digitize(r.flatten(), r_bins)
        energy_flat = grad_mag_sq.flatten()

        radial_energy = np.zeros(len(r_bins))
        counts = np.zeros(len(r_bins))

        for i in range(len(energy_flat)):
            radial_energy[r_indices[i] - 1] += energy_flat[i]
            counts[r_indices[i] - 1] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            profile = np.where(counts > 0, radial_energy / counts, 0.0)

        smoothed = gaussian_filter1d(profile, sigma=2.0)
        peaks, props = find_peaks(smoothed, height=1e-5, distance=3, prominence=1e-6, width=2)

        peak_radii = r_bins[peaks]
        safe_mask = peak_radii < max_radius
        step = int(''.join(filter(str.isdigit, fname)))

        if np.any(safe_mask):
            row = {"step": step}
            for i, idx in enumerate(np.where(safe_mask)[0]):
                row[f"radius_{i+1}"] = round(peak_radii[idx], 3)
                row[f"height_{i+1}"] = round(props["peak_heights"][idx], 6)
                row[f"prominence_{i+1}"] = round(props["prominences"][idx], 6)
                row[f"width_{i+1}"] = round(props["widths"][idx], 3)
        else:
            row = {"step": step, "note": "no valid peaks"}

        results.append(row)
        print(f"[✓] Step {step}: {np.sum(safe_mask)} shell(s)")

    except Exception as e:
        print(f"[!] Error in {fname}: {e}")

# --- Write CSV ---
if results:
    keys = sorted(set(k for row in results for k in row.keys()))
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

print(f"\n✅ Twist-wave shell scan complete. Output saved to: {OUTPUT_CSV}")
