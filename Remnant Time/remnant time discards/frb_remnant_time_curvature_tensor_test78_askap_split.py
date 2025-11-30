#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB Remnant-Time Curvature-Tensor Test (78K — ASKAP split)
Robustness variant of Test 78 using the SAME curvature tensor engine as 78B.
"""

import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from math import radians
from scipy.spatial import SphericalVoronoi, ConvexHull
from sklearn.neighbors import KDTree

# ----------------------------------------
# coordinate transforms
# ----------------------------------------

def radec_to_xyz(ra, dec):
    ra  = np.radians(ra)
    dec = np.radians(dec)
    x   = np.cos(dec)*np.cos(ra)
    y   = np.cos(dec)*np.sin(ra)
    z   = np.sin(dec)
    return np.column_stack([x,y,z])

def xyz_to_thetaphi(X):
    x, y, z = X[:,0], X[:,1], X[:,2]
    th = np.arccos(z)
    ph = np.arctan2(y, x)
    return np.degrees(th), np.degrees(ph) % 360.0

# ----------------------------------------
# curvature machinery
# ----------------------------------------

def safe_gaussian_curvature(X_original):
    X = np.unique(X_original, axis=0)
    if len(X) < 3:
        return np.zeros(len(X_original))

    try:
        sv = SphericalVoronoi(X)
        sv.sort_vertices_of_regions()
    except Exception:
        return np.zeros(len(X_original))

    K_unique = np.zeros(len(X))
    for i, reg in enumerate(sv.regions):
        if len(reg) < 3:
            continue
        verts = sv.vertices[reg]
        try:
            hull = ConvexHull(verts)
            A = hull.area
            if A > 1e-12:
                K_unique[i] = 4.0*np.pi / A
        except Exception:
            pass

    tree = KDTree(X)
    _, idx = tree.query(X_original)
    return K_unique[idx[:,0]]

def fit_curvature_tensor(th, ph, K):
    th0, ph0 = np.mean(th), np.mean(ph)
    dth = np.radians(th - th0)
    dph = np.radians(ph - ph0)
    A = np.column_stack([dth*dth, 2*dth*dph, dph*dph])
    coef, *_ = np.linalg.lstsq(A, K, rcond=None)
    a, b, c = coef
    return np.array([[a, b],[b, c]])

# ----------------------------------------
# askap loader
# ----------------------------------------

ASKAP_DIR = "data/positions"
MATCH_TOL = 2.0  # degrees

def load_askap_pointings():
    ras, decs = []
    pass  # intentional fix below

def load_askap_pointings():
    ras, decs = [], []
    for f in os.listdir(ASKAP_DIR):
        if not f.lower().endswith(".fits"):
            continue
        try:
            hdr = fits.getheader(os.path.join(ASKAP_DIR,f))
            ra  = hdr.get("CRVAL1", None)
            dec = hdr.get("CRVAL2", None)
            if ra is not None and dec is not None:
                ras.append(float(ra))
                decs.append(float(dec))
        except:
            pass
    return np.array(ras), np.array(decs)

def angsep(ra1, dec1, ra2_array, dec2_array):
    ra1  = radians(ra1)
    dec1 = radians(dec1)
    ra2  = np.radians(dec2_array * 0 + ra2_array)
    dec2 = np.radians(dec2_array)

    val = (np.sin(dec1)*np.sin(dec2) +
           np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2))
    return np.degrees(np.arccos(np.clip(val, -1, 1)))

def match_askap(RA, Dec, RAa, Deca, tol):
    out = np.zeros(len(RA), dtype=bool)
    for i in range(len(RA)):
        d = angsep(RA[i], Dec[i], RAa, Deca)
        if np.min(d) <= tol:
            out[i] = True
    return out

# ----------------------------------------
# subset runner
# ----------------------------------------

def run_subset(X, Th):
    if len(X) < 10:
        return np.nan, np.nan, np.nan, 1.0

    pos = Th < 90.0
    neg = Th >= 90.0

    if pos.sum()==0 or neg.sum()==0:
        return np.nan, np.nan, np.nan, 1.0

    Xp, Xn = X[pos], X[neg]
    thp, php = xyz_to_thetaphi(Xp)
    thn, phn = xyz_to_thetaphi(Xn)

    Kp = safe_gaussian_curvature(Xp)
    Kn = safe_gaussian_curvature(Xn)

    Tpos = fit_curvature_tensor(thp, php, Kp)
    Tneg = fit_curvature_tensor(thn, phn, Kn)

    norm_real = np.linalg.norm(Tpos - Tneg)

    # null
    n = len(X)
    npos = pos.sum()
    NMC = 2000
    null = np.zeros(NMC)

    for i in range(NMC):
        perm = np.random.permutation(n)
        idxp = perm[:npos]
        idxn = perm[npos:]

        thp2, php2 = xyz_to_thetaphi(X[idxp])
        thn2, phn2 = xyz_to_thetaphi(X[idxn])

        Kp2 = safe_gaussian_curvature(X[idxp])
        Kn2 = safe_gaussian_curvature(X[idxn])

        Tpos2 = fit_curvature_tensor(thp2, php2, Kp2)
        Tneg2 = fit_curvature_tensor(thn2, phn2, Kn2)

        null[i] = np.linalg.norm(Tpos2 - Tneg2)

    p = (1 + np.sum(null >= norm_real)) / (NMC + 1)
    return norm_real, null.mean(), null.std(), p

# ----------------------------------------
# main
# ----------------------------------------

def main(path):
    print("===============================================")
    print("FRB Curvature-Tensor Test 78K (ASKAP split)")
    print("===============================================")

    df = pd.read_csv(path)
    RA  = df["ra"].values
    Dec = df["dec"].values
    Th  = df["theta_unified"].values

    RAa, Deca = load_askap_pointings()
    print(f"[info] loaded {len(RAa)} ASKAP pointings")

    isA = match_askap(RA, Dec, RAa, Deca, MATCH_TOL)

    RAa, Deca, Tha = RA[isA],   Dec[isA],   Th[isA]
    RAn, Decn, Thn = RA[~isA], Dec[~isA], Th[~isA]

    Xa = radec_to_xyz(RAa, Deca)
    Xn = radec_to_xyz(RAn, Decn)

    print(f"ASKAP count     = {len(Xa)}")
    print(f"Non-ASKAP count = {len(Xn)}")

    realA, muA, sdA, pA = run_subset(Xa, Tha)
    realN, muN, sdN, pN = run_subset(Xn, Thn)

    print("------------------------------------------------")
    print(f"ASKAP:    real={realA}, null_mean={muA}, null_std={sdA}, p={pA}")
    print(f"Non-ASKAP: real={realN}, null_mean={muN}, null_std={sdN}, p={pN}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> curvature-tensor asymmetry in subset")
    print("  high p -> consistent with isotropy in subset")
    print("===============================================")

if __name__ == "__main__":
    main(sys.argv[1])
