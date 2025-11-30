import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

# ----------------------------------------------------------
# load NIST–PTB real clock comparison data
# ----------------------------------------------------------
df = pd.read_csv("nist_ptb_clock.csv")

t_mjd = df["mjd"].to_numpy()
y = df["frac_freq"].to_numpy()

# convert MJD to seconds relative to start
t_sec = (t_mjd - t_mjd.min()) * 86400.0

# remove mean (Lomb–Scargle likes zero-mean)
y = y - np.mean(y)

# ----------------------------------------------------------
# frequencies to test: solar day vs sidereal day
# ----------------------------------------------------------
f_solar = 1.0 / 86400.0          # Hz
f_sidereal = 1.0 / 86164.0905    # Hz

# build frequency grid around them
freqs = np.linspace(5e-6, 2e-5, 5000)

# ----------------------------------------------------------
# Lomb–Scargle spectrum
# ----------------------------------------------------------
ls = LombScargle(t_sec, y)
power = ls.power(freqs)

# extract exact-location powers
power_solar = ls.power(f_solar)
power_sidereal = ls.power(f_sidereal)

# ratio as diagnostic
ratio = power_sidereal / power_solar if power_solar > 0 else np.nan

# ----------------------------------------------------------
# fit sinusoid at sidereal frequency
# model: y = A*sin(2πft + φ) + C
# ----------------------------------------------------------
omega = 2 * np.pi * f_sidereal
sin_term = np.sin(omega * t_sec)
cos_term = np.cos(omega * t_sec)

# linear regression for A*sin + B*cos + C
M = np.column_stack([sin_term, cos_term, np.ones_like(t_sec)])
params, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
A_sin, B_cos, C = params

# convert (A,B) to amplitude and phase
amp = np.sqrt(A_sin**2 + B_cos**2)
phase = np.arctan2(B_cos, A_sin)

# ----------------------------------------------------------
# output
# ----------------------------------------------------------
print("===============================================================")
print("   NIST–PTB sidereal modulation test")
print("===============================================================")
print(f"solar-day freq    = {f_solar:.12e} Hz")
print(f"sidereal-day freq = {f_sidereal:.12e} Hz")
print()
print(f"Lomb–Scargle power at solar    = {power_solar:.4e}")
print(f"Lomb–Scargle power at sidereal = {power_sidereal:.4e}")
print(f"sidereal / solar power ratio   = {ratio:.4f}")
print()
print("sinusoid fit at sidereal frequency:")
print(f"  amplitude (Δf/f)   = {amp:.3e}")
print(f"  phase (rad)        = {phase:.3f}")
print(f"  offset C           = {C:.3e}")
print()
print("amplitude in units of 10^-16:", amp / 1e-16)
print("===============================================================")

# interpretation
print("interpretation:")
if amp < 3e-16:
    print(" • no significant sidereal modulation detected (< 3e-16).")
elif amp < 1e-15:
    print(" • marginal sidereal signal (3e-16 – 1e-15) — could hint at weak cone-axis influence.")
else:
    print(" • strong sidereal modulation — would indicate orientation-dependent frequency shift.")
print("===============================================================")
