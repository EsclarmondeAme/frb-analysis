# gps_spectral_test.py
# real GPS clock drift spectral analysis:
# look for power at sidereal (1/23.934h) vs solar (1/24h) frequencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lombscargle

# -----------------------------------------
# 1. load real converted data
# -----------------------------------------
df = pd.read_csv("gps_clock_drift_real.csv")

# keep only rows with proper time + fractional frequency
df = df.dropna(subset=["time_days", "frac_freq"])
t = df["time_days"].values         # days
y = df["frac_freq"].values         # fractional frequency deviation
y = y - np.mean(y)                 # remove offset

# convert time to seconds (Lomb-Scargle prefers seconds)
t_sec = t * 86400.0

# -----------------------------------------
# 2. define frequency scan
# -----------------------------------------
# search around the two interesting frequencies:
# solar day  = 86400 s  -> f_solar   = 1/86400 Hz  ≈ 1.157e-5 rad/s
# sidereal   = 86164 s  -> f_sid     = 1/86164 Hz  ≈ 1.160e-5 rad/s

# but we scan a wider range
f_min = 5e-6
f_max = 5e-5
N = 5000

freqs = np.linspace(f_min, f_max, N)
angular_freqs = 2 * np.pi * freqs  # lombscargle uses angular frequency

# -----------------------------------------
# 3. compute lomb-scargle power
# -----------------------------------------
power = lombscargle(t_sec, y, angular_freqs)

# -----------------------------------------
# 4. extract power at solar and sidereal
# -----------------------------------------
f_solar = 1 / 86400.0
f_sidereal = 1 / 86164.0

# find nearest indices
idx_solar = np.argmin(np.abs(freqs - f_solar))
idx_sid   = np.argmin(np.abs(freqs - f_sidereal))

power_solar = power[idx_solar]
power_sid   = power[idx_sid]

# -----------------------------------------
# 5. print results
# -----------------------------------------
print("===============================================================")
print("gps atomic clock spectral test")
print("sidereal vs solar modulation")
print("===============================================================")
print(f"power at solar frequency (24h):     {power_solar:.4e}")
print(f"power at sidereal frequency (23h56m): {power_sid:.4e}")
print(f"ratio (sidereal / solar):            {power_sid/power_solar:.4f}")
print("---------------------------------------------------------------")

if power_sid > 3 * power_solar:
    print("POSSIBLE COSMIC-AXIS SIGNATURE: sidereal peak dominates.")
elif power_sid > power_solar:
    print("small sidereal preference — weak, but could be interesting.")
else:
    print("no sidereal preference — consistent with instrumental noise.")

print("===============================================================")

# -----------------------------------------
# 6. plot spectrum
# -----------------------------------------
plt.figure(figsize=(10,6))
plt.plot(freqs, power, linewidth=1)
plt.axvline(f_solar, color="red", linestyle="--", label="solar frequency")
plt.axvline(f_sidereal, color="green", linestyle="--", label="sidereal frequency")

plt.xlabel("frequency (Hz)")
plt.ylabel("Lomb–Scargle power")
plt.title("GPS fractional-frequency drift spectrum")
plt.legend()
plt.tight_layout()
plt.show()
