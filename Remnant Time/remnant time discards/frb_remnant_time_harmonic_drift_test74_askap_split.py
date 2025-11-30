#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB REMNANT-TIME HARMONIC DRIFT TEST (74C — ASKAP split)
Uses ASKAP FITS pointing centers + unified FRB CSV.

Same physics engine as Test 74.
"""

import sys, os
import numpy as np
from numpy import sin, cos, radians
from astropy.io import fits
from tqdm import tqdm

ASKAP_DIR = "data/positions"     # your verified directory
MATCH_TOL = 2.0                  # deg, from nearest-distance diagnostic


# ------------------------------------------------------------
# load unified CSV
# ------------------------------------------------------------
def load_catalog(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    RA   = data["ra"]
    Dec  = data["dec"]
    th   = data["theta_unified"]
    ph   = data["phi_unified"]
    DM   = data["dm"]
    med  = np.median(DM)
    signs = np.where(DM < med, +1, -1)
    return RA, Dec, th, ph, signs


# ------------------------------------------------------------
# load ASKAP FITS pointing centers
# ------------------------------------------------------------
def load_askap_fits_positions():
    RAs, DECs = [], []
    for f in os.listdir(ASKAP_DIR):
        if not f.lower().endswith(".fits"):
            continue
        path = os.path.join(ASKAP_DIR, f)
        try:
            hd = fits.open(path)
            h  = hd[0].header
            ra  = h.get("CRVAL1", None)
            dec = h.get("CRVAL2", None)
            hd.close()
            if ra is not None and dec is not None:
                RAs.append(float(ra))
                DECs.append(float(dec))
        except Exception:
            pass
    return np.array(RAs), np.array(DECs)


# ------------------------------------------------------------
# haversine angular separation (deg)
# ------------------------------------------------------------
def angsep(ra1, dec1, ra2, dec2):
    ra1  = radians(ra1);  dec1 = radians(dec1)
    ra2  = radians(ra2);  dec2 = radians(dec2)
    return np.degrees(
        np.arccos(
            sin(dec1)*sin(dec2) +
            cos(dec1)*cos(dec2)*cos(ra1 - ra2)
        )
    )


# ------------------------------------------------------------
# match FRBs to ASKAP pointing centers
# ------------------------------------------------------------
def match_askap(RA, Dec, ask_ra, ask_dec, tol):
    is_askap = np.zeros(len(RA), dtype=bool)
    for i in range(len(RA)):
        d = angsep(RA[i], Dec[i], ask_ra, ask_dec)
        if np.min(d) <= tol:
            is_askap[i] = True
    return is_askap


# ------------------------------------------------------------
# 74 harmonic drift engine
# ------------------------------------------------------------
def harmonic_drift(theta, phi, ellmax=8):
    drift = 0.0
    for ell in range(1, ellmax+1):
        drift += ell * np.abs(np.mean(phi * np.cos(theta)))
    return drift


# ------------------------------------------------------------
# test execution
# ------------------------------------------------------------
def run_subset(th, ph, signs):
    pos = signs > 0
    neg = signs < 0

    drift_pos = harmonic_drift(th[pos], ph[pos])
    drift_neg = harmonic_drift(th[neg], ph[neg])
    D_real = np.abs(drift_pos - drift_neg)

    N_MC = 2000
    D_null = np.zeros(N_MC)

    for i in range(N_MC):
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
    return D_real, m, s, p


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main(path):
    print("===============================================================")
    print(" FRB REMNANT-TIME HARMONIC DRIFT TEST (74C — ASKAP split)")
    print("===============================================================")

    RA, Dec, th, ph, signs = load_catalog(path)

    ask_ra, ask_dec = load_askap_fits_positions()
    print(f"[info] loaded {len(ask_ra)} ASKAP pointing centers")

    is_askap = match_askap(RA, Dec, ask_ra, ask_dec, MATCH_TOL)

    th_A  = th[is_askap]
    ph_A  = ph[is_askap]
    sg_A  = signs[is_askap]

    th_N  = th[~is_askap]
    ph_N  = ph[~is_askap]
    sg_N  = signs[~is_askap]

    print(f"ASKAP count     = {len(th_A)}")
    print(f"non-ASKAP count = {len(th_N)}")

    # run both subsets
    D_A, mA, sA, pA = run_subset(th_A, ph_A, sg_A) if len(th_A) > 0 else (np.nan, np.nan, np.nan, 1.0)
    D_N, mN, sN, pN = run_subset(th_N, ph_N, sg_N)

    print("---------------------------------------------------------------")
    print(f"ASKAP:     D_real={D_A}, null_mean={mA}, null_std={sA}, p={pA}")
    print(f"nonASKAP:  D_real={D_N}, null_mean={mN}, null_std={sN}, p={pN}")
    print("---------------------------------------------------------------")
    print("interpretation:")
    print("  low p  -> drift asymmetry in subset")
    print("  high p -> subset consistent with isotropy")
    print("===============================================================")
    print("test 74C complete.")
    print("===============================================================")


if __name__ == "__main__":
    main(sys.argv[1])
