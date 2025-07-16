# compute_metrics.py
# Computes soliton, twist, gravity, and shell-specific energy metrics for each step

import numpy as np
import pandas as pd
import os
import re
import csv

from scipy.ndimage import center_of_mass

# --- Config ---
INPUT_FOLDER = "../pulse_output"
SHELLS_CSV = "./output/twist_shell_structure_over_time.csv"
OUTPUT_CSV = "./output/pwari_metrics_summary.csv"
DX = 0.2

# --- Load shell radii from previous step ---
shell_df = pd.read_csv(SHELLS_CSV)
shell_df.set_index("step", inplace=True)

# --- List relevant files ---
phi_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.startswith("phi_")])

results = []

for phi_file in phi_files:
    try:
        step = int(re.findall(r"\d+", phi_file)[0])
        phi = np.load(os.path.join(INPUT_FOLDER, f"phi_{step:04d}.npy"))
        phi_dot = np.load(os.path.join(INPUT_FOLDER, f"phi_dot_{step:04d}.npy"))
        theta_dot = np.load(os.path.join(INPUT_FOLDER, f"theta_dot_{step:04d}.npy"))
        gravity = np.load(os.path.join(INPUT_FOLDER, f"gravity_{step:04d}.npy"))

        # --- Coordinates ---
        z, y, x = np.indices(phi.shape)
        com = center_of_mass(phi)
        r = np.sqrt(
            (x - com[0])**2 +
            (y - com[1])**2 +
            (z - com[2])**2
        ) * DX

        # --- Total energy components ---
        grad_phi_sq = sum(np.gradient(phi, DX, axis=i)**2 for i in range(3))
        rho_phi = 0.5 * (phi_dot**2 + grad_phi_sq)
        rho_theta = 0.5 * phi**2 * theta_dot**2
        rho_gravity = 0.5 * (1 - gravity)**2

        E_phi = np.sum(rho_phi)
        E_theta = np.sum(rho_theta)
        E_grav = np.sum(rho_gravity)
        alpha = E_theta / E_phi if E_phi > 0 else 0.0

        row = {
            "step": step,
            "E_soliton": E_phi,
            "E_twist": E_theta,
            "E_gravity": E_grav,
            "alpha_pwari": alpha
        }

        # --- Shell-specific energy ---
        if step in shell_df.index:
            r_peaks = [col for col in shell_df.columns if col.startswith("r_peak_")]
            for i, col in enumerate(r_peaks):
                if not np.isnan(shell_df.at[step, col]):
                    r0 = shell_df.at[step, col]
                    dr = 0.25
                    mask = (r >= r0 - dr) & (r <= r0 + dr)
                    E_shell = np.sum(rho_theta[mask])
                    row[f"E_shell_{i+1}"] = E_shell

        results.append(row)
        print(f"[✓] Step {step:5d} | E_soliton={E_phi:.4e}, E_twist={E_theta:.4e}, alpha={alpha:.3e}")

    except Exception as e:
        print(f"[!] Error in step {phi_file}: {e}")

# --- Save ---
if results:
    keys = sorted(set(k for row in results for k in row.keys()))
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n✅ Metrics computed and saved to: {OUTPUT_CSV}")
else:
    print("❌ No results written. Check for missing NPY files or shell data.")