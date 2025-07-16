# analyze_shell_phase.py
# 1. PHYSICAL CHECK: Sample twist_wave field at r = 2, 4, 6 across time
# Confirm structure and frequency stability using sine fit and raw plots

import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.optimize import curve_fit

# --- Config ---
NPY_DIR = "../npy_cycle"
OUTPUT_PLOT = "./output/twist_wave_shell_oscillations.png"
OUTPUT_FREQ_LOG = "./output/fitted_shell_frequencies.txt"
CENTER = np.array([48, 48, 48])
FIT_STEPS = 10
SHELL_RADII = [2, 4, 6]  # Based on FFT peaks: 0.30, 0.70, 1.10 with DX = 0.2
print(f"[i] Using direct shell radii: {SHELL_RADII}")

# --- Locate twist_wave field files ---
twist_files = sorted(glob(os.path.join(NPY_DIR, "twist_wave_*.npy")))
print(f"[i] Found {len(twist_files)} twist_wave files")
steps = [int(f.split("_")[-1].split(".")[0]) for f in twist_files]

# --- Initialize trackers ---
shell_traces = {r: [] for r in SHELL_RADII}
step_trace = []

# --- Helper to get index at given radius ---
def shell_index(center, r):
    x = center[0] + r
    return tuple([int(x)] + list(center[1:]))

# --- Extract values ---
for fpath, step in zip(twist_files, steps):
    twist = np.load(fpath)
    for r in SHELL_RADII:
        idx = shell_index(CENTER, r)
        try:
            shell_traces[r].append(twist[idx])
        except IndexError:
            print(f"[!] Index out of bounds for radius r = {r}")
    step_trace.append(step)

# --- Sine fitting ---
def sine_func(t, A, omega, phi, offset):
    return A * np.sin(omega * t + phi) + offset

with open(OUTPUT_FREQ_LOG, "w") as log_file:
    for r, trace in shell_traces.items():
        if len(trace) == 0:
            log_file.write(f"r = {r} | no data extracted\n")
            print(f"[!] No data extracted for r = {r}")
            continue
        if len(trace) < FIT_STEPS:
            log_file.write(f"r = {r} | not enough data for fitting (has {len(trace)} points)\n")
            print(f"[!] Not enough data to fit r = {r}, only {len(trace)} points")
            continue

        t_fit = np.array(step_trace[:FIT_STEPS])
        y_fit = np.array(trace[:FIT_STEPS])
        print(f"\n[~] Radius {r} preview: {y_fit[:10]}")
        try:
            popt, _ = curve_fit(sine_func, t_fit, y_fit, p0=[0.1, 0.1, 0, 0])
            A, omega, phi, offset = popt
            freq = omega / (2 * np.pi)
            log_file.write(f"r = {r} | fitted frequency = {freq:.6f} Hz\n")
            print(f"[✓] Fitted shell r = {r}: frequency ≈ {freq:.6f} Hz")
        except Exception as e:
            log_file.write(f"r = {r} | fit failed: {e}\n")
            print(f"[!] Fit failed for r = {r}: {e}")

# --- Plot ---
plt.figure(figsize=(12, 6))
for r, trace in shell_traces.items():
    if len(trace) > 0:
        plt.plot(step_trace, trace, label=f"r = {r}")

plt.xlabel("Step")
plt.ylabel("Twist Wave Field")
plt.title("PWARI-G Twist Wave Shell Oscillations (Unsmoothed)")
plt.legend()
plt.grid(True)
os.makedirs("./output", exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
print(f"✅ Saved shell phase plot to {OUTPUT_PLOT}")