# analyze_shell_quantization.py
# Compare PWARI-G shell radii to Bohr quantization: r ~ n^2 => E ~ -13.6/n^2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# --- Config ---
INPUT_CSV = "./output/twist_shell_structure_over_time.csv"
PLOT_OUTPUT = "./output/bohr_comparison.png"

# --- Load data ---
df = pd.read_csv(INPUT_CSV)
df = df.sort_values("step")
df = df.reset_index(drop=True)

# --- Extract radius columns ---
radius_cols = [col for col in df.columns if col.startswith("r_peak_")]

# --- Build (n_est, E_n) points ---
data = []
for _, row in df.iterrows():
    r1 = row.get("r_peak_1", np.nan)
    if np.isnan(r1) or r1 <= 0:
        continue  # skip if no ground shell
    for col in radius_cols:
        r = row.get(col, np.nan)
        if not np.isnan(r) and r > 0:
            n_est = np.sqrt(r / r1)
            E_n = -13.6 / (n_est**2)
            data.append({
                "step": row["step"],
                "shell": col,
                "r": r,
                "r1": r1,
                "n_est": n_est,
                "E_bohr": E_n
            })

quant_df = pd.DataFrame(data)

# --- Fit Rydberg constant from E_n vs n_est ---
def rydberg_fit(n, R):
    return -R / n**2

popt, _ = curve_fit(rydberg_fit, quant_df["n_est"], quant_df["E_bohr"])
R_fit = popt[0]
print(f"🔬 Fitted Rydberg constant: R = {R_fit:.12f} eV")

# --- Plot ---
plt.figure(figsize=(10, 6))
for shell in quant_df["shell"].unique():
    df_shell = quant_df[quant_df["shell"] == shell]
    plt.scatter(df_shell["n_est"], df_shell["E_bohr"], label=f"{shell}", s=12)

# Overlay Bohr model and fitted curve
n_vals = np.linspace(1, 5, 500)
E_bohr = -13.6 / n_vals**2
E_fit = -R_fit / n_vals**2
plt.plot(n_vals, E_bohr, 'k--', label="Bohr Model (-13.6/n²)")
plt.plot(n_vals, E_fit, 'r-', label=f"Fitted Rydberg (R = {R_fit:.4f} eV)")

plt.xlabel("Estimated Quantum Number $n_{\text{est}}$")
plt.ylabel("Bohr Energy Level $E_n$ (eV)")
plt.title("PWARI-G Shell Radii vs Bohr Quantization")
plt.legend()
plt.grid(True)
os.makedirs("./output", exist_ok=True)
plt.tight_layout()
plt.savefig(PLOT_OUTPUT)
print(f"✅ Saved Bohr comparison plot to {PLOT_OUTPUT}")