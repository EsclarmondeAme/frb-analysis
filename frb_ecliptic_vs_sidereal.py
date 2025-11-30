#!/usr/bin/env python3
"""
frb_ecliptic_vs_sidereal.py
------------------------------------------------------------
Tests whether FRB ecliptic longitude (λ) or ecliptic latitude (β)
correlate with sidereal phase computed from MJD.

Produces:
• histogram comparison (λ-phase vs sidereal phase)
• scatter plot (λ-phase vs sidereal phase)
• harmonic decomposition + Monte Carlo p-values
• correlation coefficients
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic

# ------------------------------------------------------------
# sidereal phase (same trusted version)
# ------------------------------------------------------------
def sidereal_phase_from_mjd(mjd):
    t = Time(mjd, format="mjd", scale="utc")
    gmst = t.sidereal_time("mean", "greenwich")
    return (gmst.to(u.rad).value / (2*np.pi)) % 1.0


# ------------------------------------------------------------
# harmonic functions
# ------------------------------------------------------------
def harmonic_amplitudes(phases, n):
    A = np.mean(np.cos(2*np.pi*n*phases))
    B = np.mean(np.sin(2*np.pi*n*phases))
    R = np.sqrt(A*A + B*B)
    return A, B, R

def harmonic_pvalue(phases, n, R_obs, trials=20000):
    N = len(phases)
    R_rand = []
    for _ in range(trials):
        rand = np.random.rand(N)
        _, _, Rn = harmonic_amplitudes(rand, n)
        R_rand.append(Rn)
    return np.mean(np.array(R_rand) >= R_obs)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    print("============================================================")
    print("FRB ecliptic longitude vs sidereal-phase test")
    print("============================================================")

    df = pd.read_csv("frbs.csv")
    df = df.dropna(subset=["mjd", "ra", "dec"])
    print(f"FRBs usable: {len(df)}")

    # convert to ecliptic coordinates
    sky = SkyCoord(ra=df["ra"].values*u.deg,
                   dec=df["dec"].values*u.deg,
                   frame="icrs")
    ecl = sky.transform_to(GeocentricTrueEcliptic())

    df["lambda"] = ecl.lon.deg      # ecliptic longitude
    df["beta"]   = ecl.lat.deg      # ecliptic latitude

    df["lambda_phase"] = (df["lambda"] / 360.0) % 1.0
    df["sidereal"]     = sidereal_phase_from_mjd(df["mjd"].values)

    # correlation
    r = np.corrcoef(df["lambda_phase"], df["sidereal"])[0,1]
    print("------------------------------------------------------------")
    print(f"correlation (ecliptic λ-phase vs sidereal phase): r = {r:.4f}")
    print("------------------------------------------------------------")

    print("harmonic amplitudes (ecliptic λ vs sidereal)")
    print("n    set      A_n        B_n        R_n        p(R_rand >= R_n)")
    print("------------------------------------------------------------")

    for n in [1,2,3,4]:
        A_lam, B_lam, R_lam = harmonic_amplitudes(df["lambda_phase"], n)
        p_lam = harmonic_pvalue(df["lambda_phase"], n, R_lam)

        A_sid, B_sid, R_sid = harmonic_amplitudes(df["sidereal"], n)
        p_sid = harmonic_pvalue(df["sidereal"], n, R_sid)

        print(f"{n:<4} LAM  {A_lam:+.4f}   {B_lam:+.4f}   {R_lam:.4f}   {p_lam:.4f}")
        print(f"     SID  {A_sid:+.4f}   {B_sid:+.4f}   {R_sid:.4f}   {p_sid:.4f}")

    # ------------------------------------------------------------
    # histogram plot
    # ------------------------------------------------------------
    plt.figure(figsize=(11,5))
    plt.hist(df["sidereal"], bins=30, density=True, alpha=0.5, label="sidereal phase")
    plt.hist(df["lambda_phase"], bins=30, density=True, alpha=0.5, label="ecliptic λ-phase")
    plt.xlabel("phase (0..1)")
    plt.ylabel("probability density")
    plt.title("FRB ecliptic-longitude-phase vs sidereal-phase distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("frb_ecliptic_vs_sidereal.png")

    # ------------------------------------------------------------
    # scatter plot
    # ------------------------------------------------------------
    plt.figure(figsize=(6,6))
    plt.scatter(df["lambda_phase"], df["sidereal"], s=10)
    plt.xlabel("ecliptic longitude phase (λ / 360)")
    plt.ylabel("sidereal phase")
    plt.title("FRB ecliptic longitude vs sidereal phase")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("frb_ecliptic_vs_sidereal_scatter.png")

    print("saved → frb_ecliptic_vs_sidereal.png")
    print("saved → frb_ecliptic_vs_sidereal_scatter.png")
    print("============================================================")
    print("done.")
    print("============================================================")

if __name__ == "__main__":
    main()
