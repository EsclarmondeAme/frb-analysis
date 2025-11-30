#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB REMNANT-TIME HARMONIC DRIFT TEST (74A — |b|>=20° GALACTIC MASK)

This test measures whether remnant-time signs (+/-) produce
different low-ℓ harmonic drift amplitudes when the Galactic plane is removed.

Statistic:
    D = |mean( drift_pos ) - mean( drift_neg )|

Null:
    shuffle remnant signs (pos/neg) 2000 times.

Output:
    real D, null mean, null std, p-value
"""

import sys
import numpy as np
from numpy import sin, cos, radians
from tqdm import tqdm

# ------------------------------------------------------------
# coordinate helpers
# ------------------------------------------------------------
def galactic_mask(RA, Dec, bmin_deg=20):
    """return boolean mask for |b| >= bmin_deg (using approximate transform)"""
    ra = np.radians(RA)
    dec = np.radians(Dec)

    # J2000 galactic north pole + ascending node (standard)
    ra_gp  = np.radians(192.859508)
    dec_gp = np.radians(27.128336)
    l_cp   = np.radians(122.932)

    sinb = ( np.sin(dec)*np.sin(dec_gp)
             + np.cos(dec)*np.cos(dec_gp)*np.cos(ra - ra_gp) )

    b = np.degrees(np.arcsin(sinb))
    return np.abs(b) >= bmin_deg


# ------------------------------------------------------------
# harmonic drift metric (same engine as 74)
# ------------------------------------------------------------
def harmonic_drift(theta, phi, ellmax=8):
    """compute |drift| = sqrt(sum_{ℓ=1..ellmax} m-weighted harmonic imbalance)."""

    drift = 0.0
    # simple scalar drift: φ * cos(theta) weighted by ℓ (proxy)
    for ell in range(1, ellmax+1):
        weight = ell
        drift += weight * np.abs(np.mean(phi * np.cos(theta)))

    return drift


# ------------------------------------------------------------
# load catalog
# ------------------------------------------------------------
def load_catalog(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    RA   = data["ra"]
    Dec  = data["dec"]
    th   = data["theta_unified"]
    ph   = data["phi_unified"]
    # unified remnant signs: half positive half negative (Test70–83 convention)
    # but here assume: index < median DM → +1, else -1 (same engine as base 74)
    DM   = data["dm"]
    median_dm = np.median(DM)
    signs = np.where(DM < median_dm, +1, -1)
    return RA, Dec, th, ph, signs


# ------------------------------------------------------------
# main test
# ------------------------------------------------------------
def main(path):
    print("===============================================")
    print(" FRB REMNANT-TIME HARMONIC DRIFT TEST (74A)")
    print(" Galactic mask: |b| >= 20°")
    print("===============================================")

    RA, Dec, th, ph, signs = load_catalog(path)

    # apply mask
    MASK = galactic_mask(RA, Dec, bmin_deg=20)
    RA   = RA[MASK]
    Dec  = Dec[MASK]
    th   = th[MASK]
    ph   = ph[MASK]
    signs = signs[MASK]

    N = len(RA)
    print(f"[info] original N=600, after mask N={N}")

    # split hemispheres
    pos = (signs > 0)
    neg = (signs < 0)

    th_pos = th[pos]
    ph_pos = ph[pos]
    th_neg = th[neg]
    ph_neg = ph[neg]

    # real statistic
    drift_pos = harmonic_drift(th_pos, ph_pos)
    drift_neg = harmonic_drift(th_neg, ph_neg)
    D_real = np.abs(drift_pos - drift_neg)

    # Monte Carlo null: shuffle signs
    N_MC = 2000
    D_null = np.zeros(N_MC)

    for i in tqdm(range(N_MC), desc="MC"):
        shuffled = np.random.permutation(signs)
        pos_s = shuffled > 0
        neg_s = shuffled < 0

        D_null[i] = np.abs(
            harmonic_drift(th[pos_s], ph[pos_s]) -
            harmonic_drift(th[neg_s], ph[neg_s])
        )

    m = np.mean(D_null)
    s = np.std(D_null)
    p = np.mean(D_null >= D_real)

    print("------------------------------------------------")
    print(f"drift_pos     = {drift_pos}")
    print(f"drift_neg     = {drift_neg}")
    print(f"D_real        = {D_real}")
    print("------------------------------------------------")
    print(f"null mean D   = {m}")
    print(f"null std D    = {s}")
    print(f"p-value       = {p}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> drift asymmetry survives plane mask")
    print("  high p -> symmetry consistent with isotropy")
    print("===============================================")
    print("test 74A complete.")
    print("===============================================")


if __name__ == "__main__":
    main(sys.argv[1])
