#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB Remnant-Time Curvature-Tensor Test (78S — |SGB|>=20° Supergalactic Mask)
Robustness variant of Test 78 using the SAME curvature tensor engine as 78B.
"""

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
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

# supergalactic transform
SGL_NP_RA  = np.radians(283.25)
SGL_NP_DEC = np.radians(15.70)
SGL_LON_ZERO = np.radians(47.37)

def eq_to_sgb(ra_deg, dec_deg):
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    sinb = (np.sin(dec)*np.sin(SGL_NP_DEC) +
            np.cos(dec)*np.cos(SGL_NP_DEC)*np.cos(ra - SGL_NP_RA))
    b = np.arcsin(sinb)

    y = np.cos(dec)*np.sin(ra - SGL_NP_RA)
    x = (np.sin(dec)*np.cos(SGL_NP_DEC) -
         np.cos(dec)*np.sin(SGL_NP_DEC)*np.cos(ra - SGL_NP_RA))

    l = np.arctan2(y, x) + SGL_LON_ZERO
    return np.degrees(l) % 360.0, np.degrees(b)

# ----------------------------------------
# curvature tensor machinery
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

def xyz_to_thetaphi(X):
    x, y, z = X[:,0], X[:,1], X[:,2]
    th = np.arccos(z)
    ph = np.arctan2(y, x)
    return np.degrees(th), np.degrees(ph) % 360.0

def fit_curvature_tensor(th, ph, K):
    th0, ph0 = np.mean(th), np.mean(ph)
    dth = np.radians(th - th0)
    dph = np.radians(ph - ph0)
    A = np.column_stack([dth*dth, 2*dth*dph, dph*dph])
    coef, *_ = np.linalg.lstsq(A, K, rcond=None)
    a, b, c = coef
    return np.array([[a, b],[b, c]])

# ----------------------------------------
# main
# ----------------------------------------

def main(path):
    print("=================================================")
    print("FRB Curvature-Tensor Test 78S (|SGB|>=20° Mask)")
    print("=================================================")

    df = pd.read_csv(path)
    RA  = df["ra"].values
    Dec = df["dec"].values
    Th  = df["theta_unified"].values

    _, SGB = eq_to_sgb(RA, Dec)
    mask = np.abs(SGB) >= 20.0

    RA, Dec, Th = RA[mask], Dec[mask], Th[mask]
    print(f"[info] after SGB mask N = {len(RA)}")

    X = radec_to_xyz(RA, Dec)

    pos = Th < 90.0
    neg = Th >= 90.0

    Xp, Xn = X[pos], X[neg]
    thp, php = xyz_to_thetaphi(Xp)
    thn, phn = xyz_to_thetaphi(Xn)

    Kp = safe_gaussian_curvature(Xp)
    Kn = safe_gaussian_curvature(Xn)

    Tpos = fit_curvature_tensor(thp, php, Kp)
    Tneg = fit_curvature_tensor(thn, phn, Kn)

    D = Tpos - Tneg
    norm_real = np.linalg.norm(D)

    # null
    n = len(X)
    npos = pos.sum()
    NMC = 2000
    null = np.zeros(NMC)

    for i in tqdm(range(NMC), desc="MC"):
        perm = np.random.permutation(n)
        idxp = perm[:npos]
        idxn = perm[npos:]

        thp2, php2 = xyz_to_thetaphi(X[idxp])
        thn2, phn2 = xyz_to_thetaphi(X[idxn])

        Kp2 = safe_gaussian_curvature(X[idxp])
        Kn2 = safe_gaussian_curvature(X[idxn])

        Tp2 = fit_curvature_tensor(thp2, php2, Kp2)
        Tn2 = fit_curvature_tensor(thn2, phn2, Kn2)

        null[i] = np.linalg.norm(Tp2 - Tn2)

    p = (1 + np.sum(null >= norm_real)) / (NMC + 1)

    print("-------------------------------------------------")
    print("||T_pos - T_neg|| =", norm_real)
    print("null mean =", null.mean())
    print("null std  =", null.std())
    print("p-value   =", p)
    print("-------------------------------------------------")
    print("interpretation:")
    print("  low p  -> curvature-tensor asymmetry survives SGB mask")
    print("  high p -> consistent with isotropy")
    print("=================================================")

if __name__ == "__main__":
    main(sys.argv[1])
