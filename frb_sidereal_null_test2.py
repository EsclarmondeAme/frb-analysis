#!/usr/bin/env python
"""
frb_sidereal_null_test2.py

test whether the frb sidereal dipole is:

- a true time-dependent effect, or
- an artifact of sky geometry + beam pattern + mjd distribution

procedure:
1. compute exact local sidereal time for each FRB (same as harmonics file)
2. compute real dipole amplitude R1
3. randomly shuffle mjds (destroys true time structure but preserves sky distribution)
4. recompute dipole amplitude many times
5. compare real R1 to null distribution

this version matches:
- frb_sidereal_harmonics2.py
- frb_sidereal_axis_refined2.py
"""

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# CHIME observatory location
# ------------------------------------------------------------
CHIME = EarthLocation.from_geodetic(
    lon=-119.62 * u.deg,
    lat=49.32 * u.deg,
    height=0.0 * u.m,
)


# ------------------------------------------------------------
# harmonic dipole amplitude (matches your harmonics file)
# ------------------------------------------------------------
def dipole_amplitude(phases_rad):
    """
    compute R1 using the same normalization:

        A1 = (2/N) Σ cos(θ)
        B1 = (2/N) Σ sin(θ)
        R1 = sqrt(A1^2 + B1^2)
    """
    N = len(phases_rad)
    A1 = (2.0 / N) * np.sum(np.cos(phases_rad))
    B1 = (2.0 / N) * np.sum(np.sin(phases_rad))
    R1 = np.sqrt(A1 * A1 + B1 * B1)
    return R1


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    print("======================================================================")
    print("                     frb sidereal null test (refined)                ")
    print("======================================================================")

    # ---------------------------------------------------------
    # load FRB table
    # ---------------------------------------------------------
    frb = pd.read_csv("frbs.csv")

    if "mjd" not in frb.columns:
        raise SystemExit("frbs.csv must contain 'mjd' column")

    frb = frb.dropna(subset=["mjd"])
    mjd = frb["mjd"].astype(float).values
    N = len(mjd)

    print(f"frbs loaded: {N}")

    # ---------------------------------------------------------
    # compute true sidereal phases
    # ---------------------------------------------------------
    t = Time(mjd, format="mjd", scale="utc")
    lst = t.sidereal_time("mean", longitude=CHIME.lon)
    phases = lst.to_value(u.rad) % (2.0 * np.pi)

    # real dipole amplitude
    R_real = dipole_amplitude(phases)
    print(f"real dipole amplitude R1 = {R_real:.5f}")

    # ---------------------------------------------------------
    # monte carlo: time scrambling
    # ---------------------------------------------------------
    n_mc = 5000
    R_null = np.zeros(n_mc)

    print("\ncomputing null distribution (time-scramble)...")

    rng = np.random.default_rng(12345)

    for i in range(n_mc):
        # scramble mjds → destroys real timing structure
        mjd_scr = rng.permutation(mjd)

        t_scr = Time(mjd_scr, format="mjd", scale="utc")
        lst_scr = t_scr.sidereal_time("mean", longitude=CHIME.lon)
        phases_scr = lst_scr.to_value(u.rad) % (2.0 * np.pi)

        R_null[i] = dipole_amplitude(phases_scr)

    # stats
    mean_null = np.mean(R_null)
    std_null = np.std(R_null)
    z = (R_real - mean_null) / std_null

    print(f"null mean R1 = {mean_null:.5f}")
    print(f"null std     = {std_null:.5f}")
    print(f"significance = {z:.2f} σ")

    # ---------------------------------------------------------
    # histogram plot
    # ---------------------------------------------------------
    plt.figure(figsize=(10,6))
    # if null distribution has zero width, make a simple bar plot
    if np.allclose(R_null, R_null[0]):
        plt.bar([R_real], [1], width=0.01, alpha=0.6, label="null (scrambled)")
        plt.axvline(R_real, color="black", linestyle="--", linewidth=2,
                    label=f"real R1 = {R_real:.4f}")
    else:
        plt.hist(R_null, bins=50, alpha=0.6, label="null (scrambled)")
        plt.axvline(R_real, color="black", linestyle="--", linewidth=2,
                    label=f"real R1 = {R_real:.4f}")

    plt.xlabel("dipole amplitude R1")
    plt.ylabel("count")
    plt.title("FRB sidereal dipole: real vs scrambled null")
    plt.legend()
    plt.tight_layout()
    plt.savefig("frb_sidereal_null_test2.png", dpi=200)

    print("\nplot saved: frb_sidereal_null_test2.png")
    print("======================================================================")
    print("done.")
    print("======================================================================")


if __name__ == "__main__":
    main()
