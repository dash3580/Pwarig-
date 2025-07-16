# atom_dual.py
# Simulate two hydrogen-like solitons in PWARI-G, interacting via shared twist wave

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Simulation parameters ---
GRID_SIZE = (96, 96, 96)
DX = 0.2
DT = 0.005
STEPS = 200001
GRAVITY_ALPHA = 1.0
SNAP_THRESHOLD = 5e-16
SNAP_ALPHA = 0.3
OFFSET = 3  # in grid units (0.6 sim units)

# --- Utilities ---
def laplacian(f):
    return sum(np.gradient(np.gradient(f, DX, axis=i), DX, axis=i) for i in range(3))

def make_absorbing_mask(shape, edge_width=8, strength=6.0):
    mask = np.ones(shape)
    for axis in range(3):
        for i in range(edge_width):
            decay = np.exp(-strength * (1 - (i + 1) / edge_width)**2)
            front = [slice(None)] * 3
            back = [slice(None)] * 3
            front[axis] = i
            back[axis] = -1 - i
            mask[tuple(front)] *= decay
            mask[tuple(back)] *= decay
    return mask

def generate_twist_pulse(theta_dot, grad_theta, snap_zone):
    pulse = np.zeros_like(theta_dot)
    norm = np.sqrt(sum(g**2 for g in grad_theta)) + 1e-10
    for i in range(3):
        pulse += grad_theta[i] * theta_dot * snap_zone / norm
    pulse *= 0.07
    return pulse

def evolve_twist_wave(twist_wave, phi_A, phi_B, theta_dot_A, theta_dot_B):
    wave_ddot = laplacian(twist_wave)
    twist_wave += DT * wave_ddot
    anchor_zone = (phi_A + phi_B) > 0.5
    combined_theta_dot = 0.5 * (theta_dot_A + theta_dot_B)
    twist_wave[anchor_zone] -= 0.02 * (twist_wave[anchor_zone] - combined_theta_dot[anchor_zone])
    return twist_wave

def compute_angular_momentum(phi, theta):
    grad_theta = np.gradient(theta, DX)
    x = np.arange(GRID_SIZE[0]) - GRID_SIZE[0] // 2
    y = np.arange(GRID_SIZE[1]) - GRID_SIZE[1] // 2
    X, Y = np.meshgrid(x, y, indexing='ij')
    mid_z = GRID_SIZE[2] // 2
    theta_x = grad_theta[0][:, :, mid_z]
    theta_y = grad_theta[1][:, :, mid_z]
    phi_sq = phi[:, :, mid_z]**2
    Lz = np.sum((X * theta_y - Y * theta_x) * phi_sq) * DX**3
    return Lz

def apply_spin_snap(theta, theta_dot, snap_zone):
    theta_dot[snap_zone] = 0.0
    theta[snap_zone] *= -1.0
    return theta, theta_dot

# --- Initialization ---
def initialize_two_atoms():
    global phi_init_A, phi_init_B
    grid = np.indices(GRID_SIZE).astype(np.float32)
    center = np.array(GRID_SIZE)[:, None, None, None] // 2

    center_A = center.copy()
    center_B = center.copy()
    center_A[0] -= OFFSET
    center_B[0] += OFFSET

    r2_A = np.sum((grid - center_A)**2, axis=0) * DX**2
    r2_B = np.sum((grid - center_B)**2, axis=0) * DX**2

    phi_A = np.exp(-r2_A * 2.0)
    phi_B = np.exp(-r2_B * 2.0)
    phi_dot_A = np.zeros_like(phi_A)
    phi_dot_B = np.zeros_like(phi_B)

    theta_A = np.zeros_like(phi_A)
    theta_B = np.zeros_like(phi_B)
    theta_dot_A = np.zeros_like(phi_A)
    theta_dot_B = np.zeros_like(phi_B)

    twist_wave = np.zeros_like(phi_A)

    grad_phi = np.gradient(phi_A + phi_B, DX)
    rho_init = 0.5 * (phi_dot_A**2 + phi_dot_B**2) + 0.5 * sum(g**2 for g in grad_phi)
    gravity = np.exp(-0.5 * rho_init)

    phi_init_A = np.copy(phi_A)
    phi_init_B = np.copy(phi_B)
    return phi_A, phi_dot_A, theta_A, theta_dot_A, phi_B, phi_dot_B, theta_B, theta_dot_B, gravity, twist_wave

# --- Evolution rules (shared) ---
def evolve_phi(phi, phi_dot, gravity, theta_dot, phi_init, phi_lag=None, TAU=None):
    ELASTIC_COEFF = 0.5
    mismatch = theta_dot**2
    base_phi_ddot = (1.0 / gravity) * (laplacian(phi) - phi**3 - phi * mismatch)
    if phi_lag is not None and TAU is not None:
        lag_term = (phi - phi_lag) / TAU
        elastic_pull = ELASTIC_COEFF * (phi_init - phi)
        phi_ddot = base_phi_ddot + elastic_pull - lag_term + elastic_pull
    else:
        phi_ddot = base_phi_ddot
    phi_dot += DT * phi_ddot
    phi += DT * phi_dot
    return phi, phi_dot

def evolve_theta(phi, phi_dot, theta, theta_dot, gravity, twist_wave):
    grad_theta = np.gradient(theta, DX)
    lap_theta = laplacian(theta)
    grad_theta_sq = sum(g**2 for g in grad_theta)
    twist_strain = phi**2 * (grad_theta_sq + theta_dot**2)

    grid = np.indices(GRID_SIZE).astype(np.float32)
    center = np.array(GRID_SIZE)[:, None, None, None] / 2.0
    r2 = np.sum((grid - center)**2, axis=0) * DX**2
    snap_bias = 1.0 / (1.0 + r2)
    snap_zone = (twist_strain * snap_bias) > SNAP_THRESHOLD
    snap_pressure = np.zeros_like(theta)
    snap_pressure[snap_zone] = theta_dot[snap_zone]

    theta_ddot = gravity * lap_theta + 0.384 * phi_dot - SNAP_ALPHA * snap_pressure
    discarded_energy = np.sum(twist_strain[snap_zone]) * DX**3

    if np.any(snap_zone):
        emission_pulse = generate_twist_pulse(theta_dot, grad_theta, snap_zone)
        twist_wave[snap_zone] += emission_pulse[snap_zone]
        theta_dot[snap_zone] = -0.265
        theta, theta_dot = apply_spin_snap(theta, theta_dot, snap_zone)

    theta_dot *= 0.995
    SPIN_CAP = 2.0 * phi
    theta_dot = np.clip(theta_dot, -SPIN_CAP, SPIN_CAP)
    theta_dot += DT * theta_ddot
    theta += DT * theta_dot
    return theta, theta_dot, discarded_energy, snap_zone, twist_wave

# --- Main Simulation ---
def run():
    phi_A, phi_dot_A, theta_A, theta_dot_A, phi_B, phi_dot_B, theta_B, theta_dot_B, gravity, twist_wave = initialize_two_atoms()

    delay_B = 50  # steps to delay B
    damping_mask = make_absorbing_mask(GRID_SIZE, edge_width=12, strength=6.0)
    os.makedirs("frames_cycle", exist_ok=True)
    os.makedirs("npy_cycle", exist_ok=True)
    log = open("cycle_log.csv", "w")
    log.write("step,twist_energy_total,discarded_total,Lz_total\n")

    for step in range(STEPS):
        # --- Snap-triggered outer wipe zone ---
        # --- Snap-triggered outer wipe zone ---
        grid = np.indices(GRID_SIZE).astype(np.float32)
        center = np.array(GRID_SIZE)[:, None, None, None] / 2.0
        r2 = np.sum((grid - center)**2, axis=0) * DX**2
        deletion_radius = 2.0
        wipe_zone = r2 > deletion_radius**2
        phi_A, phi_dot_A = evolve_phi(phi_A, phi_dot_A, gravity, theta_dot_A, phi_init_A)
        if step >= delay_B:
            phi_B, phi_dot_B = evolve_phi(phi_B, phi_dot_B, gravity, theta_dot_B, phi_init_B)

        twist_wave = evolve_twist_wave(twist_wave, phi_A, phi_B, theta_dot_A, theta_dot_B)
        theta_A, theta_dot_A, disc_A, snap_A, twist_wave = evolve_theta(phi_A, phi_dot_A, theta_A, theta_dot_A, gravity, twist_wave)
        if step >= delay_B:
            theta_B, theta_dot_B, disc_B, snap_B, twist_wave = evolve_theta(phi_B, phi_dot_B, theta_B, theta_dot_B, gravity, twist_wave)
        else:
            disc_B, snap_B = 0.0, np.zeros_like(theta_B)

        # Wipe phi fields outside deletion radius after snap
        if np.any(snap_A):
            phi_A[wipe_zone] = 0.0
            phi_dot_A[wipe_zone] = 0.0
        if step >= delay_B and np.any(snap_B):
            phi_B[wipe_zone] = 0.0
            phi_dot_B[wipe_zone] = 0.0

        phi_recoil_zone = ((phi_A < phi_init_A) & (phi_dot_A < 0)) | ((phi_B < phi_init_B) & (phi_dot_B < 0))
        gravity[phi_recoil_zone] += 0.005 * (1.0 - gravity[phi_recoil_zone])
        gravity = np.clip(gravity - DT * GRAVITY_ALPHA * (phi_A + phi_B)**2, 0.0, 1.0)

        for field in [phi_A, phi_dot_A, theta_A, theta_dot_A, phi_B, phi_dot_B, theta_B, theta_dot_B, twist_wave]:
            field *= damping_mask

        if step % 25 == 0:
            twist_energy = (phi_A**2 * theta_dot_A**2 + phi_B**2 * theta_dot_B**2)
            twist_total = np.sum(twist_energy)
            discarded = disc_A + disc_B
            Lz_A = compute_angular_momentum(phi_A, theta_A)
            Lz_B = compute_angular_momentum(phi_B, theta_B)
            Lz_total = Lz_A + Lz_B
            log.write(f"{step},{twist_total:.5e},{discarded:.5e},{Lz_total:.5e}\n")
            log.flush()

            mid = GRID_SIZE[2] // 2
            plt.imshow(twist_wave[:, :, mid], cmap='inferno')
            plt.title(f"Shared Twist Wave (step {step})")
            plt.colorbar()
            plt.savefig(f"frames_cycle/wave_{step:04d}.png")
            plt.close()

            np.save(f"npy_cycle/twist_wave_{step:04d}.npy", twist_wave)

    log.close()
    print("Simulation complete. Log saved as cycle_log.csv")

if __name__ == "__main__":
    run()
