# track_shell_frequency_over_time.py
# Step 4: Test robustness of fitted frequency at fixed radius across time

import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.optimize import curve_fit

# --- Config ---
NPY_DIR = "../npy_cycle"
CENTER = np.array([48, 48, 48])
RADIUS = 2
WINDOW = 10
DX = 0.2

# --- Helper functions ---
def shell_index(center, r):
    x = center[0] + r
    return tuple([int(x)] + list(center[1:]))

def sine_func(t, A, omega, phi, offset):
    return A * np.sin(omega * t + phi) + offset

# --- Load files ---
twist_files = sorted(glob(os.path.join(NPY_DIR, "twist_wave_*.npy")))
steps = [int(f.split("_")[-1].split(".")[0]) for f in twist_files]
values = []

# --- Extract single shell trace ---
for f in twist_files:
    twist = np.load(f)
    val = twist[shell_index(CENTER, RADIUS)]
    values.append(val)

# --- Sliding window sine fit ---
frequencies = []
fit_steps = []
for i in range(len(steps) - WINDOW):
    t_fit = np.array(steps[i:i+WINDOW])
    y_fit = np.array(values[i:i+WINDOW])
    try:
        popt, _ = curve_fit(sine_func, t_fit, y_fit, p0=[0.1, 0.1, 0, 0])
        omega = popt[1]
        freq = omega / (2 * np.pi)
        frequencies.append(freq)
        fit_steps.append(steps[i + WINDOW // 2])
    except:
        frequencies.append(np.nan)
        fit_steps.append(steps[i + WINDOW // 2])

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(fit_steps, frequencies, label=f"r = {RADIUS}")
plt.xlabel("Simulation Step")
plt.ylabel("Fitted Frequency (Hz)")
plt.title("Fitted Twist Frequency vs Time (Sliding Window)")
plt.grid(True)
plt.tight_layout()
os.makedirs("./output", exist_ok=True)
plt.savefig("./output/shell_frequency_vs_time.png")
print("✅ Saved: ./output/shell_frequency_vs_time.png")
