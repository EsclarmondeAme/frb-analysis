#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from sklearn.neighbors import KDTree
from math import radians

# ============================================================
# coordinate utilities
# ============================================================

def radec_to_xyz(ra, dec):
    ra  = np.radians(ra)
    dec = np.radians(dec)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return np.column_stack((x,y,z))

def angsep(ra1, dec1, ra2, dec2):
    ra1  = radians(ra1)
    dec1 = radians(dec1)
    ra2  = np.radians(ra2)
    dec2 = np.radians(dec2)
    return np.degrees(
        np.arccos(
            np.clip(
                np.sin(dec1)*np.sin(dec2) +
                np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2),
                -1, 1
            )
        )
    )

def random_unit_vectors(n):
    v = np.random.normal(size=(n,3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)

# ============================================================
# ASKAP
# ============================================================

ASKAP_DIR = "data/positions"
MATCH_TOL = 2.0

def load_askap_pointings():
    ras, decs = [], []
    if not os.path.exists(ASKAP_DIR):
        return np.array([]), np.array([])
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

def match_askap(RA, Dec, RAa, Deca):
    out = np.zeros(len(RA), dtype=bool)
    for i in range(len(RA)):
        d = angsep(RA[i], Dec[i], RAa, Deca)
        if np.min(d) <= MATCH_TOL:
            out[i] = True
    return out

# ============================================================
# optical scalars (correct, safe)
# ============================================================

def compute_optical_scalars(X, k=20):
    tree = KDTree(X)
    _, idx = tree.query(X, k=k+1)

    # neighbour vectors
    neigh = X[idx[:,1:]] - X[:,None,:]

    # Jacobian estimate: mean neighbour gradient
    J = np.mean(neigh, axis=1)  # shape (N,3)

    # for congruence tests we use simpler invariants from test80:
    divJ  = np.linalg.norm(J, axis=1)
    shear = np.std(J, axis=1)
    twist = np.linalg.norm(np.cross(J, J), axis=1) * 0.0  # always zero in this estimator

    return divJ, shear, twist

# ============================================================
# hemisphere difference
# ============================================================

def hemisphere_difference(divJ, shear, twist, R):
    pos = (R>0)
    neg = (R<0)
    if np.sum(pos)==0 or np.sum(neg)==0:
        return np.nan
    return (np.mean(divJ[pos]) - np.mean(divJ[neg]) +
            np.mean(shear[pos]) - np.mean(shear[neg]) +
            np.mean(twist[pos]) - np.mean(twist[neg]))

# ============================================================
# run for subset
# ============================================================

def run_subset(X, aU):
    if len(X) < 30:
        return np.nan, np.nan, np.nan, 1.0

    R = X @ aU
    divJ, shear, twist = compute_optical_scalars(X)
    real = hemisphere_difference(divJ, shear, twist, R)

    NMC = 2000
    null = np.zeros(NMC)

    for i in tqdm(range(NMC), ncols=80):
        Xiso = random_unit_vectors(len(X))
        div_i, sh_i, tw_i = compute_optical_scalars(Xiso)
        R_i = Xiso @ aU
        null[i] = hemisphere_difference(div_i, sh_i, tw_i, R_i)

    p = (1 + np.sum(null >= real)) / (NMC + 1)
    return real, null.mean(), null.std(), p

# ============================================================
# main
# ============================================================

def main(path):
    print("===============================================")
    print(" Test 80C — Null-Congruence under ASKAP Split")
    print("===============================================")

    df = pd.read_csv(path)
    RA  = df["ra"].values
    Dec = df["dec"].values

    RAa, Deca = load_askap_pointings()
    print(f"[info] ASKAP pointings loaded: {len(RAa)}")

    isA = match_askap(RA, Dec, RAa, Deca)
    Xa = radec_to_xyz(RA[isA],   Dec[isA])
    Xn = radec_to_xyz(RA[~isA], Dec[~isA])

    print(f"[info] ASKAP count     = {len(Xa)}")
    print(f"[info] non-ASKAP count = {len(Xn)}")

    lu = np.radians(159.8)
    bu = np.radians(-0.5)
    aU = np.array([np.cos(bu)*np.cos(lu),
                   np.cos(bu)*np.sin(lu),
                   np.sin(bu)])

    print("[info] running ASKAP subset...")
    rA, mA, sA, pA = run_subset(Xa, aU)

    print("[info] running non-ASKAP subset...")
    rN, mN, sN, pN = run_subset(Xn, aU)

    print("-----------------------------------------------")
    print(f"ASKAP:    real={rA}, null_mean={mA}, null_std={sA}, p={pA}")
    print(f"non-ASKAP real={rN}, null_mean={mN}, null_std={sN}, p={pN}")
    print("-----------------------------------------------")
    print("interpretation:")
    print(" low p  → asymmetry present in subset")
    print(" high p → consistent with isotropy")
    print("===============================================")

if __name__ == "__main__":
    main(sys.argv[1])
