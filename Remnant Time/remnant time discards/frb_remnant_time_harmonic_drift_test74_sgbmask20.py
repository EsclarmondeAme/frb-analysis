#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB REMNANT-TIME HARMONIC DRIFT TEST (74B — |SGB|>=20° SUPERGALACTIC MASK)

Identical engine to 74A, but masking the supergalactic plane.
"""

import sys
import numpy as np
from numpy import sin, cos, radians
from tqdm import tqdm

# ------------------------------------------------------------
# convert RA/Dec → supergalactic lat SGB
# ------------------------------------------------------------
def supergalactic_latitude(RA, Dec):
    # de Vaucouleurs 1991 SG frame
    ra = np.radians(RA)
    dec = np.radians(Dec)

    ra_p  = np.radians(283.763)   # SG north pole RA
    dec_p = np.radians(15.705)    # SG north pole Dec
    lon0  = np.radians(47.37)     # SG zero-longitude node

    sinSGB = ( np.sin(dec)*np.sin(dec_p)
               + np.cos(dec)*np.cos(dec_p)*np.cos(ra - ra_p) )

    SGB = np.degrees(np.arcsin(sinSGB))
    return SGB


def sgb_mask(RA, Dec, sgbmin_deg=20):
    SGB = supergalactic_latitude(RA, Dec)
    return np.abs(SGB) >= sgbmin_deg


# ------------------------------------------------------------
# harmonic drift engine (same as 74)
# ------------------------------------------------------------
def harmonic_drift(theta, phi, ellmax=8):
    drift = 0.0
    for ell in range(1, ellmax+1):
        drift += ell * np.abs(np.mean(phi * np.cos(theta)))
    return drift


# ------------------------------------------------------------
# load catalog & signs (same convention as 74)
# ------------------------------------------------------------
def load_catalog(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    RA   = data["ra"]
    Dec  = data["dec"]
    th   = data["theta_unified"]
    ph   = data["phi_unified"]
    DM   = data["dm"]

    med = np.median(DM)
    signs = np.where(DM < med, +1, -1)
    return RA, Dec, th, ph, signs


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main(path):
    print("===============================================")
    print(" FRB REMNANT-TIME HARMONIC DRIFT TEST (74B)")
    print(" Supergalactic mask: |SGB| >= 20°")
    print("===============================================")

    RA, Dec, th, ph, signs = load_catalog(path)

    MASK = sgb_mask(RA, Dec, sgbmin_deg=20)
    RA   = RA[MASK]
    Dec  = Dec[MASK]
    th   = th[MASK]
    ph   = ph[MASK]
    signs = signs[MASK]

    N = len(RA)
    print(f"[info] original N=600, after SGB mask N={N}")

    pos = signs > 0
    neg = signs < 0

    th_pos = th[pos]
    ph_pos = ph[pos]
    th_neg = th[neg]
    ph_neg = ph[neg]

    drift_pos = harmonic_drift(th_pos, ph_pos)
    drift_neg = harmonic_drift(th_neg, ph_neg)
    D_real = np.abs(drift_pos - drift_neg)

    N_MC = 2000
    D_null = np.zeros(N_MC)

    for i in tqdm(range(N_MC), desc="MC"):
        shuf = np.random.permutation(signs)
        pos_s = shuf > 0
        neg_s = shuf < 0

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
    print("  low p  -> drift asymmetry survives SGB mask")
    print("  high p -> symmetry consistent with isotropy")
    print("===============================================")
    print("test 74B complete.")
    print("===============================================")


if __name__ == "__main__":
    main(sys.argv[1])
