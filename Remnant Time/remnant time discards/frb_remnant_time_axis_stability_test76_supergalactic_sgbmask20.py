#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# coordinate transforms
# ============================================================

def radec_to_xyz(ra, dec):
    ra  = np.radians(ra)
    dec = np.radians(dec)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x,y,z])

# supergalactic conversion
SGL_NP_RA  = np.radians(283.25)
SGL_NP_DEC = np.radians(15.70)
SGL_LON0   = np.radians(47.37)

def eq_to_sgb(ra_deg, dec_deg):
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    sinB = (np.sin(dec)*np.sin(SGL_NP_DEC) +
            np.cos(dec)*np.cos(SGL_NP_DEC)*np.cos(ra - SGL_NP_RA))
    B = np.arcsin(sinB)
    y = np.cos(dec)*np.sin(ra - SGL_NP_RA)
    x = (np.sin(dec)*np.cos(SGL_NP_DEC) -
         np.cos(dec)*np.sin(SGL_NP_DEC)*np.cos(ra - SGL_NP_RA))
    L = np.arctan2(y, x) + SGL_LON0
    return np.degrees(L)%360.0, np.degrees(B)

def random_unit_vectors(n):
    v = np.random.normal(size=(n,3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a,b), -1.0, 1.0)))

# ============================================================
# deformation metrics (same as original 76)
# ============================================================

def density_asymmetry(X, a):
    R = X @ a
    return abs(np.mean(R>0) - np.mean(R<0))

def shell_asymmetry(X, a):
    th = np.degrees(np.arccos(X @ a))
    s1 = ((th>=17.5)&(th<32.5))
    s2 = ((th>=32.5)&(th<47.5))
    R = X @ a
    d1 = abs(np.mean(R[s1]>0) - np.mean(R[s1]<0)) if np.sum(s1)>0 else 0
    d2 = abs(np.mean(R[s2]>0) - np.mean(R[s2]<0)) if np.sum(s2)>0 else 0
    return d1 + d2

def manifold_dilation(X, a):
    return abs(np.mean(X @ a))

def causal_collapse_score(X, a):
    return abs(np.mean(X @ a))

def temporal_curvature_score(X, a):
    R = X @ a
    return abs(np.var(R))

def score_all(X, a):
    return (density_asymmetry(X,a) +
            shell_asymmetry(X,a) +
            manifold_dilation(X,a) +
            causal_collapse_score(X,a) +
            temporal_curvature_score(X,a))

# ============================================================
# MAIN
# ============================================================

def main(path):

    print("===================================================")
    print(" Test 76B — Axis Stability under Supergalactic Mask |SGB|≥20°")
    print("===================================================")

    df = pd.read_csv(path)
    RA  = df["ra"].values
    Dec = df["dec"].values

    _, SGB = eq_to_sgb(RA, Dec)
    mask = np.abs(SGB) >= 20.0
    RA, Dec = RA[mask], Dec[mask]

    print(f"[info] N after |SGB|≥20° mask: {len(RA)}")

    X = radec_to_xyz(RA, Dec)

    # unified axis
    l_u, b_u = 159.8, -0.5
    lu = np.radians(l_u)
    bu = np.radians(b_u)
    aU = np.array([np.cos(bu)*np.cos(lu),
                   np.cos(bu)*np.sin(lu),
                   np.sin(bu)])

    # axis scan
    Naxes = 300
    A = random_unit_vectors(Naxes)
    S = np.array([score_all(X, a) for a in A])
    a_best = A[np.argmax(S)]
    d_real = angle_between(a_best, aU)

    # Monte Carlo
    NMC = 2000
    null = np.zeros(NMC)

    for i in tqdm(range(NMC), desc="MC", ncols=80):
        Xiso = random_unit_vectors(len(X))
        Siso = np.array([score_all(Xiso, a) for a in A])
        a_iso = A[np.argmax(Siso)]
        null[i] = angle_between(a_iso, aU)

    p = (1 + np.sum(null <= d_real)) / (NMC + 1)

    print("---------------------------------------------------")
    print(f"best-axis separation from unified axis = {d_real:.3f} deg")
    print(f"null mean = {null.mean():.3f} deg")
    print(f"null std  = {null.std():.3f} deg")
    print(f"p-value   = {p:.5f}")
    print("---------------------------------------------------")
    print("interpretation:")
    print(" low p  → unified axis emerges even with SGB mask (robust).")
    print(" high p → axis not preferred under mask (consistent with isotropy).")
    print("===================================================")

if __name__ == "__main__":
    main(sys.argv[1])
