# overlay_field_slices.py
# Final visualization script: enhanced clarity with midplane theta contrast, overlays, and field contours

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Config ---
STEP = 2000  # Change this to the step number you want to visualize
SLICE_INDEX = 48
DATA_DIR = "../npy_cycle"

# --- File paths ---
phi_file = os.path.join(DATA_DIR, f"phi_{STEP:04d}.npy")
theta_file = os.path.join(DATA_DIR, f"theta_{STEP:04d}.npy")
twist_file = os.path.join(DATA_DIR, f"twist_wave_{STEP:04d}.npy")

# --- Load fields ---
phi = np.load(phi_file)
theta = np.load(theta_file)
twist = np.load(twist_file)

# --- Extract slices ---
phi_slice = phi[:, :, SLICE_INDEX]
theta_slice = theta[:, :, SLICE_INDEX]
twist_slice = twist[:, :, SLICE_INDEX]

# Orthogonal theta slices
theta_x = theta[SLICE_INDEX, :, :]
theta_y = theta[:, SLICE_INDEX, :]
theta_z = theta[:, :, SLICE_INDEX]

# --- Clamped diverging normalization for midplanes ---
def symmetric_clamp(field, limit=None):
    vmax = np.max(np.abs(field)) if limit is None else limit
    return np.clip(field, -vmax, vmax), -vmax, vmax

# --- Create RGB composite with enhanced contrast ---
def normalize_contrast(field):
    centered = field - np.mean(field)
    scaled = centered / (np.max(np.abs(centered)) + 1e-8)
    return 0.5 + 0.5 * scaled  # scale to [0, 1]

phi_norm = normalize_contrast(phi_slice)
theta_norm = normalize_contrast(theta_slice)
twist_norm = normalize_contrast(twist_slice)

rgb = np.zeros((*phi_slice.shape, 3))
rgb[..., 0] = phi_norm
rgb[..., 1] = theta_norm
rgb[..., 2] = twist_norm

# --- Plot enhanced visualization ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Overlay RGB composite
axs[0, 0].imshow(rgb, origin='lower')
axs[0, 0].set_title(f"RGB Overlay (φ=R, θ=G, Twist=B) - Step {STEP}")
axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])

# Theta midplanes with symmetric clamping
theta_x_c, vmin, vmax = symmetric_clamp(theta_x)
theta_y_c, _, _ = symmetric_clamp(theta_y, limit=vmax)
theta_z_c, _, _ = symmetric_clamp(theta_z, limit=vmax)

im1 = axs[0, 1].imshow(theta_x_c, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
axs[0, 1].set_title("θ X-midplane")
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])
fig.colorbar(im1, ax=axs[0, 1], shrink=0.8)

im2 = axs[1, 0].imshow(theta_y_c, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
axs[1, 0].set_title("θ Y-midplane")
axs[1, 0].set_xticks([])
axs[1, 0].set_yticks([])
fig.colorbar(im2, ax=axs[1, 0], shrink=0.8)

im3 = axs[1, 1].imshow(theta_z_c, cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
axs[1, 1].set_title("θ Z-midplane")
axs[1, 1].set_xticks([])
axs[1, 1].set_yticks([])
fig.colorbar(im3, ax=axs[1, 1], shrink=0.8)

plt.suptitle(f"PWARI-G Atomic Shell Visualization – Step {STEP}", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs("./output", exist_ok=True)
plt.savefig(f"./output/final_field_visual_step{STEP:05d}.png", dpi=400)
plt.show()
print(f"✅ Saved upgraded visualization to ./output/final_field_visual_step{STEP:05d}.png")
