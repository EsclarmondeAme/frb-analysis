#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB REMNANT-TIME NULL GEODESIC DISTORTION TEST (75C — ASKAP split)
Corrected: vectorised angular-distance for ASKAP matching.
"""

import sys, os
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from math import radians, sin, cos, acos

ASKAP_DIR = "data/positions"
MATCH_TOL = 2.0   # deg

# ------------------------------------------------------------
# scalar angular separation
# ------------------------------------------------------------
def angsep_scalar(ra1, dec1, ra2, dec2):
    ra1 = radians(ra1);  dec1 = radians(dec1)
    ra2 = radians(ra2);  dec2 = radians(dec2)
    return np.degrees(
        acos(
            sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra1 - ra2)
        )
    )

# ------------------------------------------------------------
# vectorised FRB-to-many-ASKAP calculation
# ------------------------------------------------------------
def angsep_FRB_to_ASKAP(RA, Dec, ask_ra, ask_dec):
    """
    RA,Dec   → scalars
    ask_ra   → array (20)
    ask_dec  → array (20)

    returns array of separations (len = number of ASKAP fits)
    """
    ra1 = radians(RA)
    dec1 = radians(Dec)

    ra2  = np.radians(ask_ra)
    dec2 = np.radians(ask_dec)

    vals = (np.sin(dec1)*np.sin(dec2) +
            np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2))
    vals = np.clip(vals, -1.0, 1.0)
    return np.degrees(np.arccos(vals))

# ------------------------------------------------------------
# kNN angular distortion
# ------------------------------------------------------------
def angsep_matrix(RA, Dec):
    ra  = np.radians(RA)
    dec = np.radians(Dec)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    xyz = np.vstack([x, y, z]).T
    dot = np.clip(xyz @ xyz.T, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def distortion_proxy(RA, Dec, k=5):
    N = len(RA)
    if N <= k:
        return np.zeros(N)
    D = angsep_matrix(RA, Dec)
    np.fill_diagonal(D, 1e9)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    return np.take_along_axis(D, idx, axis=1).mean(axis=1)

# ------------------------------------------------------------
# load unified catalog
# ------------------------------------------------------------
def load_catalog(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    RA   = data["ra"]
    Dec  = data["dec"]
    DM   = data["dm"]
    med = np.median(DM)
    signs = np.where(DM < med, +1, -1)
    return RA, Dec, signs

# ------------------------------------------------------------
# load ASKAP FITS pointing centers
# ------------------------------------------------------------
def load_askap_fits_positions():
    RAs, DECs = [], []
    for f in os.listdir(ASKAP_DIR):
        if not f.lower().endswith(".fits"):
            continue
        p = os.path.join(ASKAP_DIR, f)
        try:
            hd = fits.open(p)
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
# match FRBs to ASKAP footprint
# ------------------------------------------------------------
def match_askap(RA, Dec, ask_ra, ask_dec, tol):
    N = len(RA)
    is_ask = np.zeros(N, dtype=bool)
    for i in range(N):
        dvec = angsep_FRB_to_ASKAP(RA[i], Dec[i], ask_ra, ask_dec)
        if np.min(dvec) <= tol:
            is_ask[i] = True
    return is_ask

# ------------------------------------------------------------
# run distortion test on subset
# ------------------------------------------------------------
def run_subset(RA, Dec, signs):
    N = len(RA)
    if N < 10:
        return np.nan, np.nan, np.nan, 1.0

    distort = distortion_proxy(RA, Dec, k=5)
    pos = signs > 0
    neg = signs < 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan, np.nan, np.nan, 1.0

    G_real = distort[pos].mean() - distort[neg].mean()

    n_mc = 2000
    G_null = np.zeros(n_mc)
    for i in range(n_mc):
        sh = np.random.permutation(signs)
        p2 = sh > 0
        n2 = sh < 0
        if p2.sum() == 0 or n2.sum() == 0:
            G_null[i] = 0.0
        else:
            G_null[i] = distort[p2].mean() - distort[n2].mean()

    return G_real, G_null.mean(), G_null.std(), np.mean(np.abs(G_null) >= np.abs(G_real))

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main(path):
    print("================================================")
    print("FRB REMNANT-TIME NULL GEODESIC DISTORTION TEST (75C — ASKAP split)")
    print("================================================")

    RA, Dec, signs = load_catalog(path)
    ask_ra, ask_dec = load_askap_fits_positions()
    print(f"[info] loaded {len(ask_ra)} ASKAP pointing centers")

    is_ask = match_askap(RA, Dec, ask_ra, ask_dec, MATCH_TOL)

    RA_A, Dec_A, s_A = RA[is_ask], Dec[is_ask], signs[is_ask]
    RA_N, Dec_N, s_N = RA[~is_ask], Dec[~is_ask], signs[~is_ask]

    print(f"ASKAP count     = {len(RA_A)}")
    print(f"non-ASKAP count = {len(RA_N)}")

    GA, mA, sA, pA = run_subset(RA_A, Dec_A, s_A)
    GN, mN, sN, pN = run_subset(RA_N, Dec_N, s_N)

    print("------------------------------------------------")
    print(f"ASKAP:    G_real={GA}, null_mean={mA}, null_std={sA}, p={pA}")
    print(f"nonASKAP: G_real={GN}, null_mean={mN}, null_std={sN}, p={pN}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> distortion asymmetry in subset")
    print("  high p -> subset consistent with isotropy")
    print("================================================")
    print("test 75C complete.")
    print("================================================")


if __name__ == "__main__":
    main(sys.argv[1])
