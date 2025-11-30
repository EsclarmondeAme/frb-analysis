#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, numpy as np, pandas as pd
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

# ============================================================
# correct optical-scalar engine (3×3 Jacobian per FRB)
# ============================================================

def compute_optical_scalars(X, k=20):
    tree = KDTree(X)
    _, idx = tree.query(X, k=k+1)

    divJ  = np.zeros(len(X))
    shear = np.zeros(len(X))
    twist = np.zeros(len(X))

    for i in range(len(X)):
        neigh = X[idx[i,1:]] - X[i]
        J = np.einsum("ni,nj->ij", neigh, neigh)
        J /= len(neigh)

        tr = np.trace(J)
        divJ[i] = tr

        sym = 0.5*(J + J.T)
        asym = 0.5*(J - J.T)

        shear[i] = np.linalg.norm(sym - np.eye(3)*tr/3)
        twist[i] = np.linalg.norm(asym)

    return divJ, shear, twist

# ============================================================
# hemisphere statistic
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
    print(" Test 80B — Null-Congruence under Supergalactic Mask |SGB|>=20°")
    print("===================================================")

    df = pd.read_csv(path)
    RA  = df["ra"].values
    Dec = df["dec"].values

    _, SGB = eq_to_sgb(RA, Dec)
    mask = np.abs(SGB) >= 20
    RA, Dec = RA[mask], Dec[mask]

    print(f"[info] N after |SGB|>=20 mask: {len(RA)}")

    X = radec_to_xyz(RA, Dec)

    lu = np.radians(159.8)
    bu = np.radians(-0.5)
    aU = np.array([np.cos(bu)*np.cos(lu),
                   np.cos(bu)*np.sin(lu),
                   np.sin(bu)])

    R = X @ aU

    divJ, shear, twist = compute_optical_scalars(X)
    real = hemisphere_difference(divJ, shear, twist, R)

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
    print(" low p  → null-congruence asymmetry survives SGB mask")
    print(" high p → consistent with isotropy")
    print("===================================================")

if __name__ == "__main__":
    main(sys.argv[1])
