# light_pulse_full_test_updated.py — PWARI-G Hydrogen Atom Pulse Test with Full Mechanics and Corrections

import numpy as np
import os
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
GRID_SIZE = (96, 96, 96)
DX = 0.2
DT = 0.004
STEPS = 12000
START_STEP = 40000
PHOTON_INJECTION_STEP = 50
OUTPUT_INTERVAL = 100
SNAP_THRESHOLD = 5e-16
SNAP_ALPHA = 0.3
GRAVITY_ALPHA = 1.0
GRAVITY_RELAX = 0.005
ELASTIC_COEFF = 0.5
TAU_STEPS = 1

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

def compute_lz(phi, theta):
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

# --- Load Base Fields ---
def load_field(name):
    return np.load(f"{name}_{START_STEP}.npy")

phi = load_field("phi")
phi_dot = load_field("phi_dot")
theta = load_field("theta")
theta_dot = load_field("theta_dot")
gravity = load_field("gravity")
twist_wave = np.zeros_like(phi)
phi_init = np.copy(phi)

# --- Evolution Mechanics ---
def evolve_phi(phi, phi_dot, gravity, theta_dot, phi_lag=None, TAU=None):
    mismatch = theta_dot**2
    base_phi_ddot = (1.0 / np.maximum(gravity, 1e-6)) * (laplacian(phi) - phi**3 - phi * mismatch)
    if phi_lag is not None and TAU is not None:
        lag_term = (phi - phi_lag) / TAU
        elastic_pull = ELASTIC_COEFF * (phi_init - phi)
        phi_ddot = base_phi_ddot + elastic_pull - lag_term + elastic_pull
    else:
        phi_ddot = base_phi_ddot
    phi_dot += DT * phi_ddot
    phi += DT * phi_dot
    return phi, phi_dot

def evolve_twist_wave(twist_wave):
    twist_wave += DT * laplacian(twist_wave)
    return twist_wave

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
    emitted_energy = 0.0

    if np.any(snap_zone):
        emission_pulse = generate_twist_pulse(theta_dot, grad_theta, snap_zone)
        twist_wave[snap_zone] += emission_pulse[snap_zone]
        emitted_energy = np.sum((emission_pulse[snap_zone])**2) * DX**3
        theta_dot[snap_zone] = -0.265
        theta, theta_dot = apply_spin_snap(theta, theta_dot, snap_zone)

    theta_dot *= 0.995
    theta_dot += DT * theta_ddot
    theta += DT * theta_dot
    return theta, theta_dot, snap_zone, twist_wave, discarded_energy, emitted_energy

def evolve_gravity(gravity, phi, phi_dot, theta_dot, theta, snap_zone=None):
    gravity += DT * (phi**2 - gravity)
    return gravity

# --- Photon Pulse Injection ---
def inject_photon(phi, step):
    if step == PHOTON_INJECTION_STEP:
        coords = np.indices(GRID_SIZE).astype(np.float32)
        center = np.array(GRID_SIZE)[:, None, None, None] / 2.0
        r2 = np.sum((coords - center)**2, axis=0) * DX**2
        pulse = 0.5 * (r2 < (3.0**2)).astype(np.float32)
        phi += pulse
        print(f"Photon pulse injected into phi at step {START_STEP + step}")

# --- Run Simulation ---
damping_mask = make_absorbing_mask(GRID_SIZE)
os.makedirs("pulse_output", exist_ok=True)
log = open("pulse_output/diagnostics.csv", "w")
log.write("step,Lz,twist,soliton_energy,alpha_pwari,discarded_energy,emitted_energy_cumulative,twist_max\n")

phi_buffer = []
emitted_energy_cumulative = 0.0

for step in range(STEPS):
    inject_photon(phi, step)
    phi_lag = phi_buffer[0] if len(phi_buffer) >= TAU_STEPS else None
    phi, phi_dot = evolve_phi(phi, phi_dot, gravity, theta_dot, phi_lag, TAU_STEPS * DT)
    twist_wave = evolve_twist_wave(twist_wave)
    theta, theta_dot, snap_zone, twist_wave, discarded_energy, emitted_energy = evolve_theta(phi, phi_dot, theta, theta_dot, gravity, twist_wave)
    gravity = evolve_gravity(gravity, phi, phi_dot, theta_dot, theta, snap_zone)

    if snap_zone is not None and np.any(snap_zone):
        grid = np.indices(GRID_SIZE).astype(np.float32)
        center = np.array(GRID_SIZE)[:, None, None, None] / 2.0
        r2 = np.sum((grid - center)**2, axis=0) * DX**2
        wipe_zone = r2 > 2.0**2
        phi[wipe_zone] = 0.0
        phi_dot[wipe_zone] = 0.0

    for field in [phi, phi_dot, theta, theta_dot, twist_wave]:
        field *= damping_mask

    twist_energy = phi**2 * theta_dot**2
    twist = np.sum(twist_energy) * DX**3
    twist_max = np.max(twist_energy)
    soliton_energy = np.sum(phi**2 + phi_dot**2) * DX**3
    alpha_pwari = twist / (soliton_energy + 1e-10)
    emitted_energy_cumulative += emitted_energy
    lz = compute_lz(phi, theta_dot)

    if not np.isfinite(twist) or not np.isfinite(soliton_energy) or not np.isfinite(lz):
        print(f"NaN or overflow detected at step {step}")
        break

    log.write(f"{START_STEP + step},{lz:.5e},{twist:.5e},{soliton_energy:.5e},{alpha_pwari:.8f},{discarded_energy:.5e},{emitted_energy_cumulative:.5e},{twist_max:.5e}\n")

    if step % OUTPUT_INTERVAL == 0:
        np.save(f"pulse_output/phi_{START_STEP + step}.npy", phi)
        np.save(f"pulse_output/theta_{START_STEP + step}.npy", theta)
        np.save(f"pulse_output/twist_wave_{START_STEP + step}.npy", twist_wave)
        print(f"Saved step {START_STEP + step}")

    phi_buffer.append(np.copy(phi))
    if len(phi_buffer) > TAU_STEPS:
        phi_buffer.pop(0)

log.close()
print("\u2705 Light pulse test with full physics complete.")