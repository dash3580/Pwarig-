# light_pulse_test.py
# PWARI-G Hydrogen Atom Light Pulse Injection Test

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Simulation parameters ---
GRID_SIZE = (96, 96, 96)
DX = 0.2
DT = 0.005
STEPS = 50000  # 50 pre-injection + 10,000 post-injection
GRAVITY_ALPHA = 1.0
GRAVITY_RELAX = 0.005
SNAP_THRESHOLD = 5e-16
SNAP_ALPHA = 0.3
PULSE_STEP = 50 + 40000  # Inject pulse 50 steps after loading atom
ELASTIC_COEFF = 0.5  # added for recoil control

# --- Utilities ---
def laplacian(f):
    return sum(np.gradient(np.gradient(f, DX, axis=i), DX, axis=i) for i in range(3))

def make_absorbing_mask(shape, edge_width=12, strength=6.0):
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

def inject_light_pulse(theta_dot):
    center = np.array(GRID_SIZE) // 2
    for dz in range(-2, 3):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                r2 = dx**2 + dy**2 + dz**2
                amplitude = 0.5 * np.exp(-r2 / 2.0)
                z, y, x = center + np.array([dz, dy, dx])
                theta_dot[z, y, x] += amplitude

# --- Field Initialization from .npy ---
def load_fields(step):
    base = f"npy_cycle"
    phi = np.load(f"{base}/phi_{step:04d}.npy")
    phi_dot = np.load(f"{base}/phi_dot_{step:04d}.npy")
    theta = np.load(f"{base}/theta_{step:04d}.npy")
    theta_dot = np.load(f"{base}/theta_dot_{step:04d}.npy")
    gravity = np.load(f"{base}/gravity_{step:04d}.npy")
    twist_wave = np.load(f"{base}/twist_wave_{step:04d}.npy")
    return phi, phi_dot, theta, theta_dot, gravity, twist_wave

# --- Evolution Functions ---
def evolve_phi(phi, phi_dot, gravity, theta_dot, phi_lag=None, TAU=None):
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
    snap_zone = (twist_strain / (1.0 + r2)) > SNAP_THRESHOLD
    snap_pressure = np.zeros_like(theta)
    snap_pressure[snap_zone] = theta_dot[snap_zone]
    theta_ddot = gravity * lap_theta + 0.384 * phi_dot - SNAP_ALPHA * snap_pressure
    if np.any(snap_zone):
        emission_pulse = generate_twist_pulse(theta_dot, grad_theta, snap_zone)
        twist_wave[snap_zone] += emission_pulse[snap_zone]
        theta_dot[snap_zone] = -0.265
        theta_dot *= damping_mask
        twist_wave *= damping_mask
        phi_dot *= damping_mask
    theta_dot *= 0.995
    theta_dot += DT * theta_ddot
    theta += DT * theta_dot
    return theta, theta_dot, snap_zone, twist_wave

def evolve_gravity(gravity, phi, phi_dot, theta_dot, theta, snap_zone=None):
    grid = np.indices(GRID_SIZE).astype(np.float32)
    center = np.array(GRID_SIZE)[:, None, None, None] / 2.0
    r2 = sum((grid[i] - center[i])**2 for i in range(3)) * DX**2
    W = 1.0 / (1.0 + r2)
    grad_phi = np.gradient(phi, DX)
    grad_theta = np.gradient(theta, DX)
    rho_phi = 0.5 * phi_dot**2 + 0.5 * sum(g**2 for g in grad_phi)
    rho_twist = 0.5 * phi**2 * (theta_dot**2 + sum(g**2 for g in grad_theta))
    rho_total = rho_phi + W * rho_twist
    gravity -= DT * GRAVITY_ALPHA * rho_total
    if snap_zone is not None:
        gravity[snap_zone] += GRAVITY_RELAX * (1.0 - gravity[snap_zone])
    phi_recoil_zone = (phi < phi_init) & (phi_dot < 0)
    gravity[phi_recoil_zone] += GRAVITY_RELAX * (1.0 - gravity[phi_recoil_zone])
    return gravity

# --- Main ---
def run():
    global phi_init, damping_mask
    phi, phi_dot, theta, theta_dot, gravity, twist_wave = load_fields(40000)
    phi_init = np.copy(phi)
    damping_mask = make_absorbing_mask(GRID_SIZE)
    os.makedirs("pulse_cycle", exist_ok=True)
    log = open("pulse_cycle/cycle_log.csv", "w")
    log.write("step,phi_max,twist_energy_max,soliton_energy,twist_energy,gravity_energy,Lz,alpha_pwari\n")

    phi_buffer = []
    for step in range(STEPS):
        phi_lag = phi_buffer[0] if len(phi_buffer) >= 1 else None
        phi, phi_dot = evolve_phi(phi, phi_dot, gravity, theta_dot, phi_lag, DT)
        twist_wave = evolve_twist_wave(twist_wave)
        theta, theta_dot, snap_zone, twist_wave = evolve_theta(phi, phi_dot, theta, theta_dot, gravity, twist_wave)
        gravity = evolve_gravity(gravity, phi, phi_dot, theta_dot, theta, snap_zone)

        if step == PULSE_STEP:
            inject_light_pulse(theta_dot)
            print("Light pulse injected.")

        twist_energy = phi**2 * theta_dot**2
        soliton_energy = np.sum(phi**2 + phi_dot**2)
        twist_total = np.sum(twist_energy)
        gravity_energy = np.sum((1 - gravity)**2)
        Lz = np.sum(theta_dot * np.gradient(theta, DX)[2]) * DX**3
        alpha_pwari = twist_total / (soliton_energy + 1e-10)
        log.write(f"{step},{np.max(phi):.5f},{np.max(twist_energy):.5e},{soliton_energy:.5e},{twist_total:.5e},{gravity_energy:.5e},{Lz:.5e},{alpha_pwari:.8f}")

        if snap_zone is not None and np.any(snap_zone):
            grid = np.indices(GRID_SIZE).astype(np.float32)
            center = np.array(GRID_SIZE)[:, None, None, None] / 2.0
            r2 = np.sum((grid - center)**2, axis=0) * DX**2
            deletion_radius = 2.0
            wipe_zone = r2 > deletion_radius**2
            phi[wipe_zone] = 0.0
            phi_dot[wipe_zone] = 0.0

        if step % 100 == 0:
            np.save(f"pulse_cycle/phi_{step:04d}.npy", phi)
            np.save(f"pulse_cycle/theta_{step:04d}.npy", theta)
            np.save(f"pulse_cycle/gravity_{step:04d}.npy", gravity)
            np.save(f"pulse_cycle/phi_dot_{step:04d}.npy", phi_dot)
            np.save(f"pulse_cycle/theta_dot_{step:04d}.npy", theta_dot)
            np.save(f"pulse_cycle/twist_wave_{step:04d}.npy", twist_wave)

        phi_buffer.append(np.copy(phi))
        if len(phi_buffer) > 1:
            phi_buffer.pop(0)

    log.close()
    print("Light pulse test complete.")

if __name__ == "__main__":
    run()
