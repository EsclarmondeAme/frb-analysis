#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# coordinate transforms
# ============================================================

def radec_to_galactic(ra, dec):
    ra  = np.radians(ra)
    dec = np.radians(dec)

    ra_gp  = np.radians(192.85948)
    dec_gp = np.radians(27.12825)
    l_omega = np.radians(32.93192)

    b = np.arcsin(np.sin(dec)*np.sin(dec_gp) +
                  np.cos(dec)*np.cos(dec_gp)*np.cos(ra - ra_gp))

    y = np.cos(dec)*np.sin(ra - ra_gp)
    x = (np.sin(dec)*np.cos(dec_gp) -
         np.cos(dec)*np.sin(dec_gp)*np.cos(ra - ra_gp))

    l = np.arctan2(y, x) + l_omega
    return (np.degrees(l) % 360.0), np.degrees(b)

def radec_to_xyz(ra, dec):
    ra  = np.radians(ra)
    dec = np.radians(dec)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x,y,z])

def random_unit_vectors(n):
    v = np.random.normal(size=(n,3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a,b), -1.0, 1.0)))

# ============================================================
#  Test 76 deformation metrics (same as original)
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
    R = X @ a
    return abs(np.mean(R))

def causal_collapse_score(X, a):
    # simple proxy retained exactly as original
    R = X @ a
    return abs(np.mean(R))

def temporal_curvature_score(X, a):
    R = X @ a
    return abs(np.var(R))

def score_all(X, a):
    s70 = density_asymmetry(X, a)
    s71 = shell_asymmetry(X, a)
    s72 = manifold_dilation(X, a)
    s74 = causal_collapse_score(X, a)
    s75 = temporal_curvature_score(X, a)
    return s70 + s71 + s72 + s74 + s75

# ============================================================
#  MAIN
# ============================================================

def main(path):

    print("===================================================")
    print(" Test 76A — Axis Stability under Galactic Mask |b|≥20°")
    print("===================================================")

    df = pd.read_csv(path)
    RA  = df["ra"].values
    Dec = df["dec"].values

    # mask
    _, b = radec_to_galactic(RA, Dec)
    mask = np.abs(b) >= 20.0
    RA, Dec = RA[mask], Dec[mask]

    print(f"[info] N after |b|≥20° mask: {len(RA)}")

    X = radec_to_xyz(RA, Dec)

    # unified axis
    l_u, b_u = 159.8, -0.5
    lu = np.radians(l_u)
    bu = np.radians(b_u)
    aU = np.array([np.cos(bu)*np.cos(lu),
                   np.cos(bu)*np.sin(lu),
                   np.sin(bu)])

    # axis-scan
    Naxes = 300
    A = random_unit_vectors(Naxes)
    S = np.array([score_all(X, a) for a in A])
    a_best = A[np.argmax(S)]
    d_real = angle_between(a_best, aU)

    # Monte Carlo null with isotropic skies
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
    print(" low p  → unified axis emerges even with |b| mask (robust).")
    print(" high p → axis not preferred under mask (consistent with isotropy).")
    print("===================================================")

if __name__ == "__main__":
    main(sys.argv[1])
