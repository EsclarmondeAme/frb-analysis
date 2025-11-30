import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

print("==============================================================")
print("atomic clock anomaly test (gps clock drift – sidereal modulation)")
print("==============================================================")
print()

# -----------------------------------------------------------------------------
# real data CSV file: user must supply this in same folder
#   columns required: 'mjd' (float), 'frac_freq' (float Δf/f), 'sat_id' (optional string)
# -----------------------------------------------------------------------------

FILENAME = "gps_clock_drift_real.csv"
if not os.path.exists(FILENAME):
    raise FileNotFoundError(f"Required file '{FILENAME}' not found – please supply your real GPS/clock drift dataset")

df = pd.read_csv(FILENAME)

if "mjd" in df.columns:
    t_days = df["mjd"].values.astype(float) - df["mjd"].min()
elif "time_days" in df.columns:
    t_days = df["time_days"].values.astype(float)
else:
    raise ValueError("CSV file must contain column 'mjd' or 'time_days'")

if "frac_freq" not in df.columns:
    raise ValueError("CSV file must contain column 'frac_freq' = Δf/f")

y = df["frac_freq"].values.astype(float)

print(f"loaded {len(t_days)} data points from '{FILENAME}'")
print()

# -----------------------------------------------------------------------------
# sidereal sinusoid model: period = 0.99726957 days
# -----------------------------------------------------------------------------
def sidereal_sine(t_days, A, phi, C):
    P_sid = 0.99726957
    omega = 2.0 * np.pi / P_sid
    return A * np.sin(omega * t_days + phi) + C

def fit_sinusoid(t_days, y):
    guess = [1e-15, 0.0, np.mean(y)]
    popt, pcov = curve_fit(
        sidereal_sine,
        t_days,
        y,
        p0=guess,
        maxfev=20000
    )
    return popt, pcov

popt, pcov = fit_sinusoid(t_days, y)
A_fit, phi_fit, C_fit = popt
A_err = np.sqrt(np.diag(pcov))[0]

print("fitted sidereal sinusoid:")
print(f"  amplitude A         = {A_fit:.3e}")
print(f"  1-sigma error dA    = {A_err:.3e}")
print(f"  phase (radians)     = {phi_fit:.3f}")
print(f"  offset C            = {C_fit:.3e}")
print()
print(f"amplitude in units of 10^-14:  {A_fit / 1e-14:.2f}")
print(f"signal-to-noise ratio (A/dA): {A_fit / A_err:.2f}")
print()

print("interpretation:")
print("---------------")
if abs(A_fit) < 2 * A_err:
    print("• no significant sidereal modulation detected (|A| < 2σ)")
    print("• consistent with no detectable frequency-layer effect at this sensitivity.")
else:
    print("• significant sidereal modulation detected (|A| ≥ 2σ)")
    print("• might indicate tiny dependence of clock rate on orientation relative to a fixed cosmic axis.")
print()

# -----------------------------------------------------------------------------
# optional plot
# -----------------------------------------------------------------------------
try:
    tt = t_days
    y_fit = sidereal_sine(tt, A_fit, phi_fit, C_fit)

    plt.figure(figsize=(8,4))
    plt.plot(tt, y, '.', label='data')
    plt.plot(tt, y_fit, '-', label='sidereal fit')
    plt.xlabel("time [days]")
    plt.ylabel("fractional frequency deviation (Δf/f)")
    plt.title("GPS/atomic clock drift vs time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("plotting failed:", e)
