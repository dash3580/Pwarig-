import numpy as np
import os
import re
import csv

# --- Config ---
INPUT_FOLDER = "../npy_cycle"
OUTPUT_FOLDER = "./output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

GRID_CENTER = (48, 48, 48)
DX = 0.2
DT = 0.005
SHELL_RADIUS_LIST = [0.3, 0.7, 1.1]  # Shells to track
RADIUS_WIDTH = 0.4
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "twist_wave_phase_by_shell.csv")

# --- Get sorted twist_wave files ---
twist_files = sorted(
    [f for f in os.listdir(INPUT_FOLDER) if re.match(r"twist_wave_\d+\.npy", f)],
    key=lambda x: int(re.findall(r"\d+", x)[0])
)

# --- Initialize ---
twist_data = {r: [] for r in SHELL_RADIUS_LIST}
times = []

# --- Loop through files ---
for t_file in twist_files:
    step = int(re.findall(r"\d+", t_file)[0])
    time = step * DT
    twist_wave = np.load(os.path.join(INPUT_FOLDER, t_file))

    z, y, x = np.indices(twist_wave.shape)
    r = np.sqrt(
        (x - GRID_CENTER[0])**2 +
        (y - GRID_CENTER[1])**2 +
        (z - GRID_CENTER[2])**2
    ) * DX

    times.append(time)
    for r_shell in SHELL_RADIUS_LIST:
        mask = (r >= r_shell - RADIUS_WIDTH) & (r <= r_shell + RADIUS_WIDTH)
        avg_value = np.mean(twist_wave[mask])
        twist_data[r_shell].append(avg_value)

# --- Save CSV ---
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time"] + [f"twist_r={r:.2f}" for r in SHELL_RADIUS_LIST])
    for i, t in enumerate(times):
        row = [t] + [twist_data[r][i] for r in SHELL_RADIUS_LIST]
        writer.writerow(row)

print(f"[✓] Phase amplitude saved to: {OUTPUT_CSV}")
