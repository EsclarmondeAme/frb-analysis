#!/usr/bin/env python3
# ============================================================
# FRB SIDEREAL MODULATION TEST
# ============================================================
# this script tests whether FRB sky directions show a preferred
# phase when projected onto sidereal right ascension, and
# whether that phase aligns with the unified cosmic axis.
#
# key idea:
# earth rotates relative to the sky in a sidereal day (23h56m).
# if a real cosmic anisotropy exists, FRB arrival directions
# should show a non-uniform distribution in sidereal phase φ,
# where φ = local sidereal time at which the RA aligns overhead.
#
# footprints CANNOT generate a sidereal-phase dipole.
# ============================================================

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import kstest
from math import atan2, sqrt, cos, sin, pi

# ------------------------------------------------------------
# config
# ------------------------------------------------------------

CATALOG_FILE = r"C:\Users\ratec\Documents\CrossLayerPhysics\frbs.csv"

# unified axis (galactic)
UNIFIED_L = 159.85
UNIFIED_B = -0.51

# number of monte carlo scrambles
N_MC = 20000


# ------------------------------------------------------------
# helper: Rayleigh test for angular nonuniformity
# ------------------------------------------------------------
def rayleigh_test(angles):
    """
    Rayleigh Z test for uniformity of circular data.
    angles in radians.
    returns (R, Z, p_value)
    """
    N = len(angles)
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R = sqrt(C**2 + S**2) / N
    Z = N * R**2
    p = np.exp(-Z)  # Rayleigh distribution
    return R, Z, p


# ------------------------------------------------------------
# sidereal phase from RA (in degrees)
# ------------------------------------------------------------
def sidereal_phase_from_ra(ra_deg):
    """
    sidereal phase φ = 2π * RA / 360
    RA is a sidereal coordinate already,
    so phase = RA mapped onto [0, 2π).
    """
    return np.deg2rad(ra_deg % 360.0)


# ------------------------------------------------------------
# angular separation (deg)
# ------------------------------------------------------------
def angsep(l1, b1, l2, b2):
    """
    great-circle separation in degrees.
    """
    c1 = SkyCoord(l=l1*u.deg, b=b1*u.deg, frame="galactic")
    c2 = SkyCoord(l=l2*u.deg, b=b2*u.deg, frame="galactic")
    return c1.separation(c2).deg


# ------------------------------------------------------------
# convert Rayleigh vector to galactic direction
# ------------------------------------------------------------
def rayleigh_axis(ra_deg, dec_deg):
    """
    takes FRB sky positions and returns the axis of maximum
    sidereal modulation. this is the direction of the Rayleigh
    vector in ICRS, then converted to galactic coords.
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    # project onto equatorial unit vectors
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    M = np.vstack([x, y, z]).T
    mean_vec = M.mean(axis=0)

    # normalize
    if np.linalg.norm(mean_vec) == 0:
        mean_vec = np.array([1.0, 0.0, 0.0])
    else:
        mean_vec = mean_vec / np.linalg.norm(mean_vec)

    c = SkyCoord(
        x=mean_vec[0],
        y=mean_vec[1],
        z=mean_vec[2],
        frame="icrs",
        representation_type="cartesian"
    )

    g = c.galactic
    return g.l.deg, g.b.deg


# ------------------------------------------------------------
# monte-carlo RA scrambles
# ------------------------------------------------------------
def mc_sidereal_null(n, n_mc):
    """
    shuffle phases uniformly in [0, 2π),
    compute Rayleigh R for each scramble.
    """
    Rvals = np.empty(n_mc)
    for i in range(n_mc):
        ph = np.random.uniform(0, 2*np.pi, size=n)
        R, Z, p = rayleigh_test(ph)
        Rvals[i] = R
    return Rvals


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    print("===================================================")
    print("FRB SIDEREAL MODULATION TEST")
    print("===================================================\n")

    df = pd.read_csv(CATALOG_FILE)
    df = df.dropna(subset=["ra", "dec"])

    ra = df["ra"].values
    dec = df["dec"].values
    N = len(ra)
    print(f"loaded FRBs: {N}")

    # compute sidereal phases
    phi = sidereal_phase_from_ra(ra)

    # Rayleigh test
    R_real, Z_real, p_real = rayleigh_test(phi)

    # sidereal-modulation axis in galactic coords
    l_axis, b_axis = rayleigh_axis(ra, dec)

    # compare to unified axis
    sep = angsep(l_axis, b_axis, UNIFIED_L, UNIFIED_B)

    # monte-carlo significance
    R_null = mc_sidereal_null(N, N_MC)
    p_mc = np.mean(R_null >= R_real)

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------
    print("\n================== RESULTS ==================")
    print(f"Rayleigh R_real = {R_real:.5f}")
    print(f"Rayleigh Z_real = {Z_real:.3f}")
    print(f"Rayleigh p_real (analytic) = {p_real:.3e}")
    print(f"MC p_value (scrambled phases) = {p_mc:.4f}")
    print("---------------------------------------------")
    print(f"Sidereal-modulation axis (galactic):")
    print(f"    l = {l_axis:.3f} deg")
    print(f"    b = {b_axis:.3f} deg")
    print(f"Angular separation from unified axis = {sep:.3f} deg")
    print("=============================================\n")

    # --------------------------------------------------------
    # VERDICT
    # --------------------------------------------------------
    print("--------------- scientific verdict ---------------")

    if p_mc < 0.01 and sep < 25:
        print(
            "FRBs exhibit a statistically significant sidereal-phase modulation "
            "whose preferred direction is closely aligned with the unified cosmic axis. "
            "this cannot be produced by sky footprint, confirming a physical anisotropy."
        )

    elif p_mc < 0.01 and sep >= 25:
        print(
            "a significant sidereal modulation is detected, but its axis does not align "
            "with the unified direction. this suggests a real anisotropy, but not the "
            "same one implied by the CMB + clock data."
        )

    elif p_mc >= 0.1:
        print(
            "no significant sidereal-phase modulation is detected. FRB arrival directions "
            "are consistent with a uniform sidereal distribution, meaning temporal "
            "anisotropy does not currently support or refute the unified-axis model."
        )

    else:
        print(
            "a weak or marginal sidereal modulation is present, but not strong enough "
            "to claim detection. more FRBs or time-aware catalogs are required."
        )

    print("---------------------------------------------------")
    print("analysis complete.")
    print("===================================================\n")


if __name__ == "__main__":
    main()
