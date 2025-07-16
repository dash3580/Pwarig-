# analyze_shell_structure.py
# Detects radial shell structure in twist_wave energy using twist_energy_*.npy files

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import center_of_mass
from scipy.signal import find_peaks

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "npy_cycle"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "shell_structure_detected.csv"

# --- Shell Detection ---
def analyze_twist_energy(step):
    try:
        twist_energy = np.load(DATA_DIR / f"twist_energy_{step:04d}.npy")
        phi = np.load(DATA_DIR / f"phi_{step:04d}.npy")
        center = np.array(center_of_mass(phi)).astype(int)

        grid = np.indices(phi.shape)
        r2 = sum((grid[i] - center[i])**2 for i in range(3))
        r = np.sqrt(r2).astype(int)
        r_max = r.max()
        radial_energy = np.zeros(r_max + 1)
        counts = np.zeros(r_max + 1)

        for radius in range(r_max + 1):
            mask = (r == radius)
            radial_energy[radius] = twist_energy[mask].sum()
            counts[radius] = mask.sum()

        profile = radial_energy / (counts + 1e-12)
        peaks, _ = find_peaks(profile, distance=4, height=profile.max() * 0.1)

        return {
            "step": step,
            "n_peaks": len(peaks),
            "peak_r1": peaks[0] if len(peaks) > 0 else -1,
            "peak_r2": peaks[1] if len(peaks) > 1 else -1,
            "peak_r3": peaks[2] if len(peaks) > 2 else -1,
            "twist_total": twist_energy.sum()
        }
    except Exception as e:
        print(f"[Failed at step {step}]: {e}")
        return None

# --- Main Loop ---
def main():
    steps = []
    for fname in os.listdir(DATA_DIR):
        if fname.startswith("twist_energy_") and fname.endswith(".npy"):
            try:
                step = int(fname.split("_")[2].split(".")[0])
                steps.append(step)
            except:
                continue

    steps = sorted(set(steps))
    results = []

    for step in steps:
        result = analyze_twist_energy(step)
        if result:
            results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved shell structure data to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
