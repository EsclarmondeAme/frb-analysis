import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# helper: sidereal-frequency sine
# ---------------------------------------------------------
def sidereal_sine(t_days, A, phi, C):
    """
    t_days: time in days (any zero-point)
    A     : amplitude of fractional frequency modulation
    phi   : phase in radians
    C     : constant offset
    """
    # sidereal period (days)
    P_sid = 0.99726957
    omega = 2.0 * np.pi / P_sid
    return A * np.sin(omega * t_days + phi) + C


def fit_sinusoid(t_days, y):
    """
    fit y(t) with a sidereal sine.
    returns (A, phi, C), cov
    """
    # initial guesses
    A0 = 1e-15
    phi0 = 0.0
    C0 = np.mean(y)

    popt, pcov = curve_fit(
        sidereal_sine,
        t_days,
        y,
        p0=[A0, phi0, C0],
        maxfev=10000
    )
    return popt, pcov


# ---------------------------------------------------------
# mode 1: try to load real data if available
# ---------------------------------------------------------
real_csv = "gps_clock_drift_real.csv"
use_real = os.path.exists(real_csv)

if use_real:
    print("==============================================================")
    print("atomic clock test (REAL data from gps_clock_drift_real.csv)")
    print("==============================================================")

    df = pd.read_csv(real_csv)

    # expected columns:
    #   time_days      : time in days (float) or
    #   mjd            : modified julian date (float)
    #   frac_freq      : fractional frequency deviation (Δf/f)
    #
    # we try to be flexible:
    if "time_days" in df.columns:
        t_days = df["time_days"].values.astype(float)
    elif "mjd" in df.columns:
        # re-center for numerical stability
        mjd = df["mjd"].values.astype(float)
        t_days = mjd - mjd.min()
    else:
        raise ValueError(
            "real-data mode: need a 'time_days' or 'mjd' column "
            "in gps_clock_drift_real.csv"
        )

    if "frac_freq" not in df.columns:
        raise ValueError(
            "real-data mode: need a 'frac_freq' column in gps_clock_drift_real.csv "
            "(fractional frequency deviation Δf/f)."
        )

    y = df["frac_freq"].values.astype(float)

else:
    print("================================================================")
    print("atomic clock anomaly test (SYNTHETIC toy data – NOT real clocks)")
    print("================================================================")

    # synthetic time axis: 30 days sampled hourly
    n_days = 30
    dt_hours = 1.0
    t_hours = np.arange(0.0, n_days * 24.0, dt_hours)
    t_days = t_hours / 24.0

    # true synthetic parameters (for testing the pipeline)
    A_true = 4e-15         # 0.4 × 10^-14
    phi_true = 0.2
    C_true = 5e-16

    # generate clean sidereal sinusoid
    y_clean = sidereal_sine(t_days, A_true, phi_true, C_true)

    # add small gaussian noise
    noise = np.random.normal(scale=1e-15, size=len(t_days))
    y = y_clean + noise

# ---------------------------------------------------------
# fit and interpret
# ---------------------------------------------------------
popt, pcov = fit_sinusoid(t_days, y)
A_fit, phi_fit, C_fit = popt
A_err = np.sqrt(np.diag(pcov))[0] if pcov.shape == (3, 3) else np.nan

print()
print("fitted sidereal sinusoid:")
print(f"  amplitude A         = {A_fit:.3e}")
print(f"  1-sigma error dA    = {A_err:.3e}")
print(f"  phase (radians)     = {phi_fit:.3f}")
print(f"  offset C            = {C_fit:.3e}")
print()
print(f"amplitude in units of 10^-14:  {A_fit / 1e-14:.2f}")
print(f"signal-to-noise (A/dA):        {A_fit / A_err:.2f}")

# simple interpretation
print()
print("interpretation:")
print("---------------")

if np.abs(A_fit) < 2 * A_err:
    print("• no significant sidereal modulation detected (|A| < 2σ).")
    print("• consistent with no frequency-layer effect at this sensitivity.")
else:
    print("• significant sidereal modulation detected (|A| ≥ 2σ).")
    print("• if this is REAL clock data and t is sidereal-aligned,")
    print("  it could indicate a tiny dependence of clock frequency")
    print("  on Earth's orientation relative to a fixed cosmic direction.")
    print("  (a possible signature of a cone / layer axis.)")

# ---------------------------------------------------------
# optional quick plot
# ---------------------------------------------------------
try:
    # build model on same time grid
    y_fit = sidereal_sine(t_days, A_fit, phi_fit, C_fit)

    plt.figure()
    plt.title("atomic clock fractional frequency vs time")
    plt.plot(t_days, y, ".", label="data")
    plt.plot(t_days, y_fit, "-", label="fit")
    plt.xlabel("time [days]")
    plt.ylabel("fractional frequency deviation (Δf/f)")
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print()
    print("plotting failed (this is optional):", e)
