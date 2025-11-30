#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.time import Time

import sys

def sidereal_phase_from_mjd(mjd):
    t = Time(mjd, format="mjd", scale="utc")
    gmst = t.sidereal_time("mean", "greenwich")
    return (gmst.to_value(u.hourangle) / 24.0) % 1.0


def harmonic_amp(phases, n):
    A_n = np.mean(np.cos(2 * np.pi * n * phases))
    B_n = np.mean(np.sin(2 * np.pi * n * phases))
    R_n = np.sqrt(A_n**2 + B_n**2)
    return A_n, B_n, R_n


def monte_carlo_pvalue(R_obs, n, N=20000, Ndata=600):
    R_rand = []
    for _ in range(N):
        uniform = np.random.rand(Ndata)
        _, _, Rn = harmonic_amp(uniform, n)
        R_rand.append(Rn)
    R_rand = np.array(R_rand)
    return np.mean(R_rand >= R_obs)


def main():
    print("============================================================")
    print("FRB ecliptic latitude vs sidereal-phase test")
    print("============================================================")

    frb = pd.read_csv("frbs.csv")

    mask = (~frb["ra"].isna()) & (~frb["dec"].isna()) & (~frb["mjd"].isna())
    frb = frb[mask]
    print(f"FRBs usable: {len(frb)}")

    coords = SkyCoord(ra=frb["ra"].values * u.deg,
                      dec=frb["dec"].values * u.deg,
                      frame="icrs")

    ecl = coords.barycentrictrueecliptic

    beta = ecl.lat.to_value(u.deg)
    beta_phase = (beta + 90.0) / 180.0

    phi_sid = sidereal_phase_from_mjd(frb["mjd"].values)

    print("------------------------------------------------------------")
    r = np.corrcoef(beta_phase, phi_sid)[0, 1]
    print(f"correlation (ecliptic β-phase vs sidereal phase): r = {r:+.4f}")
    print("------------------------------------------------------------")

    print("harmonic amplitudes (ecliptic β vs sidereal)")
    print("n    set      A_n        B_n        R_n        p(R_rand >= R_n)")
    print("------------------------------------------------------------")

    for n in [1, 2, 3, 4]:
        A, B, R = harmonic_amp(beta_phase, n)
        p = monte_carlo_pvalue(R, n, Ndata=len(frb))
        print(f"{n:<4} BET  {A:+.4f}   {B:+.4f}   {R:.4f}   {p:.4f}")

        A2, B2, R2 = harmonic_amp(phi_sid, n)
        p2 = monte_carlo_pvalue(R2, n, Ndata=len(frb))
        print(f"     SID  {A2:+.4f}   {B2:+.4f}   {R2:.4f}   {p2:.4f}")

    plt.figure(figsize=(7,7))
    plt.scatter(beta_phase, phi_sid, s=12, alpha=0.5)
    plt.xlabel("ecliptic latitude phase β/180°")
    plt.ylabel("sidereal phase")
    plt.title("FRB ecliptic latitude vs sidereal phase")
    plt.grid(True)
    plt.savefig("frb_ecliptic_lat_vs_sidereal_scatter.png", dpi=150)

    print("saved → frb_ecliptic_lat_vs_sidereal_scatter.png")
    print("============================================================")
    print("done.")
    print("============================================================")


if __name__ == "__main__":
    main()
