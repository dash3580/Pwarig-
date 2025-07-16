# two_soliton_merge_sim.py
# PWARI-G two-soliton binding experiment (full dynamics)

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Grid and simulation parameters ---
GRID_SIZE = (96, 96, 96)
DX = 0.2
DT = 0.005
STEPS = 40000

# --- Snap parameters ---
SNAP_THRESHOLD = 5e-16
SNAP_ALPHA = 0.3

# --- Field arrays ---
phi = np.zeros(GRID_SIZE)
phi_dot = np.zeros(GRID_SIZE)
theta = np.zeros(GRID_SIZE)
theta_dot = np.zeros(GRID_SIZE)
twist_wave = np.zeros(GRID_SIZE)
twist_energy = np.zeros(GRID_SIZE)
phi_energy = np.zeros(GRID_SIZE)
wave_out = np.zeros(GRID_SIZE)

# --- Grid and center coordinates ---
grid = np.indices(GRID_SIZE).astype(np.float32)
center = np.array(GRID_SIZE)[:, None, None, None] / 2.0

# --- Two soliton core initialization ---
center1 = center.copy()
center1[0] -= 10
center2 = center.copy()
center2[0] += 10

r2a = np.sum((grid - center1)**2, axis=0) * DX**2
r2b = np.sum((grid - center2)**2, axis=0) * DX**2

phi += np.exp(-r2a * 2.0) + np.exp(-r2b * 2.0)
phi_init = np.copy(phi)

# --- Twist initialization ---
x, y = grid[0] - center[0], grid[1] - center[1]
theta = np.arctan2(y, x) * 0.5

# --- Utilities ---
def laplacian(f):
    return sum(np.gradient(np.gradient(f, DX, axis=i), DX, axis=i) for i in range(3))

def grad_squared(f):
    return sum(np.gradient(f, DX, axis=i)**2 for i in range(3))

def strain_ratio(phi, theta, theta_dot):
    return (grad_squared(theta) + theta_dot**2) / (phi**2 + 1e-8)

def apply_snap(theta_dot, phi, theta):
    ratio = strain_ratio(phi, theta, theta_dot)
    snap_zone = ratio > SNAP_THRESHOLD
    theta_dot[snap_zone] -= SNAP_ALPHA * ratio[snap_zone]
    return theta_dot, snap_zone

def compute_energies():
    twist_energy[:] = phi**2 * (theta_dot**2 + grad_squared(theta))
    phi_energy[:] = phi**2 + phi_dot**2

# --- Evolution step ---
def evolve(phi, phi_dot, theta, theta_dot):
    phi_ddot = laplacian(phi) - phi**3 - phi * theta_dot**2
    recoil = 0.01 * (phi_init - phi)
    phi_dot += DT * (phi_ddot + recoil)
    phi += DT * phi_dot

    theta_ddot = laplacian(theta) + 0.01 * phi_dot
    theta_dot += DT * theta_ddot
    theta_dot, snap_zone = apply_snap(theta_dot, phi, theta)
    theta += DT * theta_dot

    twist_wave[snap_zone] = theta_dot[snap_zone]
    wave_out[snap_zone] += theta_dot[snap_zone]**2
    compute_energies()
    return phi, phi_dot, theta, theta_dot

# --- Output directory ---
os.makedirs("merged_npy", exist_ok=True)

# --- Main loop ---
for step in range(STEPS + 1):
    phi, phi_dot, theta, theta_dot = evolve(phi, phi_dot, theta, theta_dot)

    if step % 100 == 0:
        np.save(f"merged_npy/phi_{step:05d}.npy", phi)
        np.save(f"merged_npy/theta_{step:05d}.npy", theta)
        np.save(f"merged_npy/twist_wave_{step:05d}.npy", twist_wave)
        np.save(f"merged_npy/twist_energy_{step:05d}.npy", twist_energy)
        np.save(f"merged_npy/phi_energy_{step:05d}.npy", phi_energy)
        print(f"Saved step {step}")

print("✅ Two-soliton merge simulation complete.")
