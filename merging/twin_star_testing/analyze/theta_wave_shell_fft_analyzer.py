# twist_wave_shell_fft_analyzer.py
import numpy as np
import csv
import os

INPUT_CSV = "output/twist_wave_phase_by_shell.csv"
OUTPUT_CSV = "output/twist_wave_shell_fft_analysis.csv"

def load_shell_time_series():
    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        time_series = {name: [] for name in reader.fieldnames if name != "time"}
        times = []

        for row in reader:
            times.append(float(row["time"]))
            for name in time_series:
                val = row[name]
                if val.strip() == "" or val.strip().lower() == "nan":
                    time_series[name].append(np.nan)
                else:
                    time_series[name].append(float(val))

    return np.array(times), time_series

def compute_fft(times, values):
    values = np.array(values)
    if np.all(np.isnan(values)) or len(values[~np.isnan(values)]) < 10:
        return None  # Not enough data
    values = np.nan_to_num(values)
    dt = times[1] - times[0]
    freqs = np.fft.rfftfreq(len(values), dt)
    fft_vals = np.abs(np.fft.rfft(values))
    peak_index = np.argmax(fft_vals)
    return freqs[peak_index], fft_vals[peak_index]

if not os.path.exists(INPUT_CSV):
    print(f"[ERROR] Missing input file: {INPUT_CSV}")
    exit()

times, time_series = load_shell_time_series()
fft_rows = []

for shell, values in time_series.items():
    result = compute_fft(times, values)
    if result is not None:
        freq, amplitude = result
        fft_rows.append({"shell": shell, "peak_frequency": freq, "peak_amplitude": amplitude})
    else:
        print(f"[WARN] Skipping {shell}: insufficient or invalid data")

if fft_rows:
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fft_rows[0].keys())
        writer.writeheader()
        writer.writerows(fft_rows)
    print(f"[OK] FFT analysis complete. Saved to {OUTPUT_CSV}")
else:
    print("[FAIL] No usable shell data found. Check twist_wave_phase_by_shell.csv for errors.")
