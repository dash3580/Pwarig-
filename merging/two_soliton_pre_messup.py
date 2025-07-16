# two_soliton_photon_test.py — PWARI-G Hydrogen Bonding Test with Photon Excitation

import numpy as np
import os

# --- Simulation Parameters ---
GRID_SIZE = (96, 96, 96)
DX = 0.2
DT = 0.001
STEPS = 10000
START_STEP = 39950
PHOTON_INJECTION_STEP = 50
OUTPUT_INTERVAL = 50
SNAP_THRESHOLD = 5e-16
SNAP_ALPHA = 0.3
SEPARATION = 30  # grid units between two solitons

# --- Load Base Field from Single Soliton State ---
def load_field(name):
    return np.load(f"{name}_{START_STEP}.npy")

base_phi = load_field("phi")
base_phi_dot = load_field("phi_dot")
base_theta = load_field("theta")
base_theta_dot = load_field("theta_dot")
base_gravity = load_field("gravity")

# --- Duplicate and Offset Fields for Two Solitons ---
def place_two(field):
    shifted = np.zeros_like(field)
    center = np.array(GRID_SIZE) // 2
    offset = np.array([SEPARATION // 2, 0, 0])
    slc = tuple(slice(center[i]-12, center[i]+12) for i in range(3))
    tgt1 = tuple(slice(center[i]-12 - offset[i], center[i]+12 - offset[i]) for i in range(3))
    tgt2 = tuple(slice(center[i]-12 + offset[i], center[i]+12 + offset[i]) for i in range(3))
    shifted[tgt1] += field[slc]
    shifted[tgt2] += field[slc]
    return shifted

phi = place_two(base_phi)
phi_dot = place_two(base_phi_dot)
theta = place_two(base_theta)
theta_dot = place_two(base_theta_dot)
gravity = place_two(base_gravity)

# --- Utilities ---
def laplacian(f):
    return sum(np.gradient(np.gradient(f, DX, axis=i), DX, axis=i) for i in range(3))

def compute_snap_pressure(phi):
    snap_core = np.abs(phi) > SNAP_THRESHOLD
    return snap_core.astype(float)

def inject_photon(phi, step):
    if step == PHOTON_INJECTION_STEP:
        center = np.array(GRID_SIZE) // 2
        offsets = [np.array([-SEPARATION // 2, 0, 0]), np.array([SEPARATION // 2, 0, 0])]
        for shift in offsets:
            pos = tuple(center + shift)
            x, y, z = np.indices(GRID_SIZE)
            mask = (x - pos[0])**2 + (y - pos[1])**2 + (z - pos[2])**2 < 3**2
            phi[mask] += 0.5
        print(f"Photon pulses injected at step {START_STEP + step}")

# --- Diagnostics ---
lz_trace = []
twist_trace = []
soliton_energy_trace = []
step_trace = []

# --- Save snapshots ---
def save_fields(step):
    np.save(f"phi_{START_STEP + step}.npy", phi)
    np.save(f"theta_{START_STEP + step}.npy", theta)
    np.save(f"twist_energy_{START_STEP + step}.npy", phi**2 * theta_dot**2)

def compute_lz(phi, theta_dot):
    return np.sum(phi**2 * theta_dot) * DX**3

def compute_twist(phi, theta_dot):
    return np.sum(phi**2 * theta_dot**2) * DX**3

def compute_soliton_energy(phi, phi_dot):
    return np.sum(phi**2 + phi_dot**2) * DX**3

# --- Evolution Loop ---
for step in range(STEPS):
    inject_photon(phi, step)

    lap_phi = laplacian(phi)
    lap_theta = laplacian(theta)

    snap_pressure = compute_snap_pressure(phi)

    phi_ddot = lap_phi - phi + gravity * phi**3 - 0.01 * phi_dot
    theta_ddot = gravity * lap_theta + 0.01 * phi_dot - SNAP_ALPHA * snap_pressure

    phi_dot += DT * phi_ddot
    phi += DT * phi_dot

    theta_dot += DT * theta_ddot
    theta += DT * theta_dot

    gravity += DT * (phi**2 - gravity)

    # --- Diagnostics ---
    lz = compute_lz(phi, theta_dot)
    twist = compute_twist(phi, theta_dot)
    soliton_energy = compute_soliton_energy(phi, phi_dot)

    lz_trace.append(lz)
    twist_trace.append(twist)
    soliton_energy_trace.append(soliton_energy)
    step_trace.append(START_STEP + step)

    if step % OUTPUT_INTERVAL == 0:
        print(f"Step {START_STEP + step} | Lz: {lz:.5e} | Twist: {twist:.5e} | Soliton: {soliton_energy:.5e}")
        save_fields(step)

# --- Save Results ---
results = np.column_stack((step_trace, lz_trace, twist_trace, soliton_energy_trace))
np.savetxt("two_soliton_photon_response.csv", results, delimiter=",", header="step,Lz,twist,soliton_energy", comments="")
print("✅ Two-soliton test complete and diagnostics saved.")
