#!/usr/bin/env python3
"""
frb_galactic_vs_sidereal.py
------------------------------------------------------------
Tests whether FRB galactic longitude (l) or galactic latitude (b)
correlates with sidereal phase from the MJD timestamps.

Produces:
• histogram comparison (galactic l-phase vs sidereal phase)
• scatter plot (l-phase vs sidereal phase)
• harmonic decomposition for both
• Monte Carlo p-values
• prints correlation coefficients
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord

# ------------------------------------------------------------
# sidereal phase function (same as your working version)
# ------------------------------------------------------------
def sidereal_phase_from_mjd(mjd):
    t = Time(mjd, format="mjd", scale="utc")
    gmst = t.sidereal_time("mean", "greenwich")
    return (gmst.to(u.rad).value / (2*np.pi)) % 1.0


# ------------------------------------------------------------
# compute Fourier coefficients
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
    R_rand = np.array(R_rand)
    return np.mean(R_rand >= R_obs)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    print("============================================================")
    print("FRB galactic-l vs sidereal-phase test (frbs.csv)")
    print("============================================================")

    df = pd.read_csv("frbs.csv")

    # ensure valid
    df = df.dropna(subset=["mjd", "ra", "dec"])
    print(f"FRBs usable: {len(df)}")

    # compute galactic coordinates
    coords = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg, frame="icrs")
    df["l"] = coords.galactic.l.deg
    df["b"] = coords.galactic.b.deg

    # convert to phases
    df["sidereal"] = sidereal_phase_from_mjd(df["mjd"].values)
    df["l_phase"] = (df["l"] / 360.0) % 1.0

    # correlation
    r = np.corrcoef(df["l_phase"], df["sidereal"])[0,1]
    print("------------------------------------------------------------")
    print(f"correlation between galactic-l-phase and sidereal phase: r = {r:.4f}")
    print("------------------------------------------------------------")

    # harmonic analysis
    print("harmonic amplitudes (galactic-l vs sidereal)")
    print("n    set      A_n        B_n        R_n        p(R_rand >= R_n)")
    print("------------------------------------------------------------")

    for n in [1,2,3,4]:
        A_l, B_l, R_l = harmonic_amplitudes(df["l_phase"], n)
        p_l = harmonic_pvalue(df["l_phase"], n, R_l)

        A_s, B_s, R_s = harmonic_amplitudes(df["sidereal"], n)
        p_s = harmonic_pvalue(df["sidereal"], n, R_s)

        print(f"{n:<4} L    {A_l:+.4f}   {B_l:+.4f}   {R_l:.4f}   {p_l:.4f}")
        print(f"{''*4} SID  {A_s:+.4f}   {B_s:+.4f}   {R_s:.4f}   {p_s:.4f}")

    # ------------------------------------------------------------
    # plots
    # ------------------------------------------------------------

    # histogram
    plt.figure(figsize=(11,5))
    plt.hist(df["sidereal"], bins=30, density=True, alpha=0.5, label="sidereal phase")
    plt.hist(df["l_phase"], bins=30, density=True, alpha=0.5, label="galactic l-phase")
    plt.xlabel("phase (0..1)")
    plt.ylabel("probability density")
    plt.title("FRB galactic-l-phase vs sidereal-phase distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("frb_galactic_vs_sidereal.png")

    # scatter
    plt.figure(figsize=(6,6))
    plt.scatter(df["l_phase"], df["sidereal"], s=10)
    plt.xlabel("galactic longitude phase (l/360)")
    plt.ylabel("sidereal phase")
    plt.title("FRB galactic longitude vs sidereal phase")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("frb_galactic_vs_sidereal_scatter.png")

    print("saved → frb_galactic_vs_sidereal.png")
    print("saved → frb_galactic_vs_sidereal_scatter.png")
    print("============================================================")
    print("done.")
    print("============================================================")


if __name__ == "__main__":
    main()
