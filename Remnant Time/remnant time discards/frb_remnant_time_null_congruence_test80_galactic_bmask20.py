#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KDTree

# ============================================================
# coordinate utilities
# ============================================================

def radec_to_xyz(ra, dec):
    ra  = np.radians(ra)
    dec = np.radians(dec)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])

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
    return (np.degrees(l)%360.0), np.degrees(b)

def random_unit_vectors(n):
    v = np.random.normal(size=(n,3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)

# ============================================================
# optical scalars (copied exactly from base test80)
# ============================================================

def compute_optical_scalars(X, k=20):
    tree = KDTree(X)
    _, idx = tree.query(X, k=k+1)

    # neighbor displacement vectors
    disp = X[idx[:,1:]] - X[:,None,:]      # shape (N, k, 3)

    # compute per-point Jacobian as mean outer product
    J = np.zeros((len(X), 3, 3))
    for i in range(len(X)):
        Ni = disp[i]                       # (k,3)
        J[i] = np.mean(np.einsum("ai,aj->aij", Ni, Ni), axis=0)

    # optical scalars
    divJ = np.trace(J, axis1=1, axis2=2)
    shear = np.sqrt(np.sum((J - np.eye(3)*divJ[:,None,None]/3)**2, axis=(1,2)))
    twist = np.sqrt(np.sum((J - J.transpose(0,2,1))**2, axis=(1,2)))

    return divJ, shear, twist

# ============================================================
# hemisphere difference
# ============================================================

def hemisphere_difference(divJ, shear, twist, R):
    pos = (R > 0)
    neg = (R < 0)
    if np.sum(pos)==0 or np.sum(neg)==0:
        return np.nan
    return (np.mean(divJ[pos]) - np.mean(divJ[neg]) +
            np.mean(shear[pos]) - np.mean(shear[neg]) +
            np.mean(twist[pos]) - np.mean(twist[neg]))

# ============================================================
# main
# ============================================================

def main(path):
    print("===================================================")
    print(" Test 80A — Null-Congruence under Galactic Mask |b|>=20°")
    print("===================================================")

    df = pd.read_csv(path)
    RA  = df["ra"].values
    Dec = df["dec"].values

    _, b = radec_to_galactic(RA, Dec)
    mask = np.abs(b) >= 20
    RA, Dec = RA[mask], Dec[mask]

    print(f"[info] N after |b|>=20 mask: {len(RA)}")

    X = radec_to_xyz(RA, Dec)

    # unified axis
    lu = np.radians(159.8)
    bu = np.radians(-0.5)
    aU = np.array([
        np.cos(bu)*np.cos(lu),
        np.cos(bu)*np.sin(lu),
        np.sin(bu)
    ])
    R = X @ aU

    # compute real statistic
    divJ, shear, twist = compute_optical_scalars(X)
    real = hemisphere_difference(divJ, shear, twist, R)

    # Monte Carlo null
    NMC = 2000
    null = np.zeros(NMC)

    print("[info] running null MC (2000 skies)...")
    for i in tqdm(range(NMC)):
        Xiso = random_unit_vectors(len(X))
        divJ_i, shear_i, twist_i = compute_optical_scalars(Xiso)
        R_i = Xiso @ aU
        null[i] = hemisphere_difference(divJ_i, shear_i, twist_i, R_i)

    p = (1 + np.sum(null >= real)) / (NMC + 1)

    print("---------------------------------------------------")
    print(f"real statistic = {real}")
    print(f"null mean      = {np.mean(null)}")
    print(f"null std       = {np.std(null)}")
    print(f"p-value        = {p}")
    print("---------------------------------------------------")
    print("interpretation:")
    print(" low p  → null-congruence asymmetry survives masking")
    print(" high p → consistent with isotropy")
    print("===================================================")

if __name__ == "__main__":
    main(sys.argv[1])
