# scan_twist_energy_shells.py
# Step 2: Analyze twist energy radial profile at fixed timestep (e.g., step 40000)

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

# --- Config ---
STEP = 40000
FILENAME = f"../npy_cycle/twist_wave_{STEP:04d}.npy"
CENTER = np.array([48, 48, 48])
DX = 0.2  # grid spacing
NBINS = 100

# --- Load twist_wave field ---
print(f"[i] Loading: {FILENAME}")
twist = np.load(FILENAME)
shape = twist.shape

# --- Compute radial coordinates ---
zz, yy, xx = np.indices(shape)
coords = np.stack([xx, yy, zz], axis=0)
radii = np.sqrt(np.sum((coords - CENTER[:, None, None, None])**2, axis=0)) * DX

# --- Bin twist energy radially ---
energy_density = twist**2
r_flat = radii.flatten()
e_flat = energy_density.flatten()

r_bins = np.linspace(0, r_flat.max(), NBINS+1)
r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
hist = np.zeros_like(r_centers)

for i in range(NBINS):
    in_bin = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
    hist[i] = np.sum(e_flat[in_bin])

# --- Smooth and normalize ---
hist_smooth = gaussian_filter1d(hist, sigma=1)
hist_norm = hist_smooth / np.max(hist_smooth)

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(r_centers, hist_norm, label=f"Step {STEP}")
plt.xlabel("Radius r")
plt.ylabel("Normalized Twist Energy")
plt.title("Radial Twist Energy Profile at Step 40000")
plt.grid(True)
plt.legend()
os.makedirs("./output", exist_ok=True)
plt.tight_layout()
plt.savefig("./output/radial_twist_energy_40000.png")
print("✅ Saved: ./output/radial_twist_energy_40000.png")