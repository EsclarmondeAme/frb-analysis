import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt

print("="*60)
print("fast spectral test (real GPS clock drift)")
print("="*60)

# ------------------------------------------------------------
# 1. load real clock data
# ------------------------------------------------------------
df = pd.read_csv("gps_clock_drift_real.csv")

# accept time_days instead of mjd
if "time_days" in df.columns:
    t = df["time_days"].values
else:
    raise ValueError("csv missing time_days column")

if "frac_freq" not in df.columns:
    raise ValueError("csv missing frac_freq column")

y = df["frac_freq"].values
y = y - np.mean(y)

# convert days → seconds
t_sec = t * 86400.0

# ------------------------------------------------------------
# 2. define target frequencies
# ------------------------------------------------------------
f_solar = 1.0 / 86400.0
f_sidereal = 1.0 / 86164.0905

# frequency scan
scan = np.linspace(0.5/86400.0, 2.0/86400.0, 20000)

# ------------------------------------------------------------
# 3. lomb–scargle
# ------------------------------------------------------------
print("computing Lomb–Scargle (fast)...")

ls = LombScargle(t_sec, y)
power = ls.power(scan, normalization='psd')

# ------------------------------------------------------------
# 4. extract values
# ------------------------------------------------------------
idx_solar = np.argmin(np.abs(scan - f_solar))
idx_sid   = np.argmin(np.abs(scan - f_sidereal))

P_solar = power[idx_solar]
P_sid   = power[idx_sid]

print()
print("results:")
print("--------")
print(f"solar day frequency    = {f_solar:.10e} Hz")
print(f"sidereal day frequency = {f_sidereal:.10e} Hz")
print()
print(f"power at solar    = {P_solar:.6e}")
print(f"power at sidereal = {P_sid:.6e}")
print(f"ratio sidereal/solar = {P_sid / P_solar:.4f}")

print()
print("="*60)
if P_sid > 3 * P_solar:
    print("strong sidereal component → possible cosmic-direction dependence!")
elif P_sid > P_solar:
    print("weak sidereal preference.")
else:
    print("no sidereal preference → no cone-axis signal detected.")
print("="*60)
