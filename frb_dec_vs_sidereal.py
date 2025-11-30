#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRB declination vs sidereal-phase correlation test
--------------------------------------------------
This script checks whether the strong FRB sidereal-phase
modulation is actually explained by sky-location (declination)
bias in CHIME's sensitivity.

Outputs:
- correlation coefficient r
- harmonic amplitudes for declination distribution
- harmonic amplitudes for sidereal distribution (for reference)
- Monte Carlo p-values
- two plots:
    • dec distribution vs sidereal distribution
    • scatter of dec vs sidereal phase
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
from scipy.stats import pearsonr

# ------------------------------------------------------------
# utilities
# ------------------------------------------------------------
def sidereal_phase_from_mjd(mjd):
    """convert MJD → sidereal phase in [0,1)."""
    t = Time(mjd, format="mjd", scale="utc")
    gmst = t.sidereal_time("mean", "greenwich")
    phi = (gmst.hour / 24.0) % 1.0
    return phi


def harmonic_coeffs(phases, nmax=4):
    """return A_n, B_n, R_n arrays for harmonics up to nmax."""
    A = []
    B = []
    R = []
    for n in range(1, nmax+1):
        A_n = np.mean(np.cos(2*np.pi*n*phases))
        B_n = np.mean(np.sin(2*np.pi*n*phases))
        R_n = np.sqrt(A_n**2 + B_n**2)
        A.append(A_n)
        B.append(B_n)
        R.append(R_n)
    return np.array(A), np.array(B), np.array(R)


def harmonic_mc_pvals(phases, R_obs, nmax=4, Nmc=20000):
    """Monte Carlo p-values for harmonic amplitudes R_n."""
    pvals = []
    N = len(phases)
    for n in range(1, nmax+1):
        R_rand = []
        for _ in range(Nmc):
            rand = np.random.rand(N)
            A_n = np.mean(np.cos(2*np.pi*n*rand))
            B_n = np.mean(np.sin(2*np.pi*n*rand))
            R_rand.append(np.sqrt(A_n*A_n + B_n*B_n))
        R_rand = np.array(R_rand)
        p = np.mean(R_rand >= R_obs[n-1])
        pvals.append(p)
    return np.array(pvals)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    print("============================================================")
    print("FRB declination vs sidereal-phase test (frbs.csv)")
    print("============================================================")

    # load data
    frb = pd.read_csv("frbs.csv")
    # must contain dec + mjd
    frb = frb.dropna(subset=["dec", "mjd"])
    print(f"FRBs with valid dec & MJD: {len(frb)}")

    dec_deg = frb["dec"].values
    mjd = frb["mjd"].values

    # convert
    print("[INFO] computing sidereal phases...")
    sid = sidereal_phase_from_mjd(mjd)

    # normalize declination into phase (optional)
    # but better: use declination in radians for correlation
    dec_norm = (dec_deg - np.min(dec_deg)) / (np.max(dec_deg) - np.min(dec_deg))

    # ------------------------------------------------------------
    # correlation
    # ------------------------------------------------------------
    r, p_r = pearsonr(dec_norm, sid)
    print("------------------------------------------------------------")
    print(f"correlation between declination and sidereal phase: r = {r:.4f},  p = {p_r:.4f}")
    print("------------------------------------------------------------")

    # ------------------------------------------------------------
    # harmonic analysis
    # ------------------------------------------------------------
    print("harmonic amplitudes (declination vs sidereal)")
    print("n    set      A_n        B_n        R_n        p(R_rand >= R_n)")
    print("------------------------------------------------------------")

    # sidereal
    A_sid, B_sid, R_sid = harmonic_coeffs(sid)
    p_sid = harmonic_mc_pvals(sid, R_sid)

    # declination (use normalized phase-like version)
    A_dec, B_dec, R_dec = harmonic_coeffs(dec_norm)
    p_dec = harmonic_mc_pvals(dec_norm, R_dec)

    for n in range(1, 5):
        print(f"{n}   DEC  {A_dec[n-1]:+0.4f}  {B_dec[n-1]:+0.4f}  {R_dec[n-1]:0.4f}   {p_dec[n-1]:0.4f}")
        print(f"    SID  {A_sid[n-1]:+0.4f}  {B_sid[n-1]:+0.4f}  {R_sid[n-1]:0.4f}   {p_sid[n-1]:0.4f}")

    # ------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------
    # histogram comparison
    plt.figure(figsize=(12,5))
    plt.hist(sid, bins=40, density=True, alpha=0.6, label="sidereal phase")
    plt.hist(dec_norm, bins=40, density=True, alpha=0.6, label="declination (normalized)")
    plt.xlabel("phase (0..1)")
    plt.ylabel("probability density")
    plt.title("FRB declination-phase vs sidereal-phase distributions")
    plt.legend()
    plt.savefig("frb_dec_vs_sidereal.png", dpi=150)
    print("saved → frb_dec_vs_sidereal.png")

    # scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(dec_norm, sid, s=12, alpha=0.7)
    plt.xlabel("declination phase (normalized)")
    plt.ylabel("sidereal phase from MJD")
    plt.title("FRB declination vs sidereal phase")
    plt.grid(True, alpha=0.3)
    plt.savefig("frb_dec_vs_sidereal_scatter.png", dpi=150)
    print("saved → frb_dec_vs_sidereal_scatter.png")

    print("============================================================")
    print("done.")
    print("============================================================")


if __name__ == "__main__":
    main()
