#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
from math import radians

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

def random_unit_vectors(n):
    v = np.random.normal(size=(n,3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a,b), -1.0, 1.0)))

# ============================================================
# ASKAP utilities
# ============================================================

ASKAP_DIR = "data/positions"
MATCH_TOL = 2.0

def load_askap_pointings():
    ras, decs = [], []
    if not os.path.exists(ASKAP_DIR):
        return np.array(ras), np.array(decs)
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

def angsep(ra1, dec1, ra2, dec2):
    ra1  = radians(ra1)
    dec1 = radians(dec1)
    ra2  = np.radians(ra2)
    dec2 = np.radians(dec2)
    v = (np.sin(dec1)*np.sin(dec2) +
         np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2))
    return np.degrees(np.arccos(np.clip(v, -1, 1)))

def match_askap(RA, Dec, RAa, Deca, tol):
    out = np.zeros(len(RA), dtype=bool)
    for i in range(len(RA)):
        d = angsep(RA[i], Dec[i], RAa, Deca)
        if np.min(d) <= tol:
            out[i] = True
    return out

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
# axis-stability runner for a subset
# ============================================================

def run_subset(X, aU):
    if len(X) < 10:
        return np.nan, np.nan, np.nan, 1.0

    Naxes = 300
    A = random_unit_vectors(Naxes)
    S = np.array([score_all(X, a) for a in A])
    a_best = A[np.argmax(S)]
    d_real = angle_between(a_best, aU)

    NMC = 2000
    null = np.zeros(NMC)

    for i in range(NMC):
        Xiso = random_unit_vectors(len(X))
        Siso = np.array([score_all(Xiso, a) for a in A])
        a_iso = A[np.argmax(Siso)]
        null[i] = angle_between(a_iso, aU)

    p = (1 + np.sum(null <= d_real)) / (NMC + 1)
    return d_real, null.mean(), null.std(), p

# ============================================================
# MAIN
# ============================================================

def main(path):

    print("===============================================")
    print(" Test 76C — Axis Stability under ASKAP Split")
    print("===============================================")

    df = pd.read_csv(path)
    RA  = df["ra"].values
    Dec = df["dec"].values

    RAa, Deca = load_askap_pointings()
    print(f"[info] ASKAP pointings loaded: {len(RAa)}")

    isA = match_askap(RA, Dec, RAa, Deca, MATCH_TOL)

    Xa = radec_to_xyz(RA[isA],   Dec[isA])
    Xn = radec_to_xyz(RA[~isA], Dec[~isA])

    print(f"[info] ASKAP count     = {len(Xa)}")
    print(f"[info] non-ASKAP count = {len(Xn)}")

    # unified axis
    l_u, b_u = 159.8, -0.5
    lu = np.radians(l_u)
    bu = np.radians(b_u)
    aU = np.array([np.cos(bu)*np.cos(lu),
                   np.cos(bu)*np.sin(lu),
                   np.sin(bu)])

    dA, muA, sdA, pA = run_subset(Xa, aU)
    dN, muN, sdN, pN = run_subset(Xn, aU)

    print("-----------------------------------------------")
    print(f"ASKAP:    d={dA}, null_mean={muA}, null_std={sdA}, p={pA}")
    print(f"non-ASKAP d={dN}, null_mean={muN}, null_std={sdN}, p={pN}")
    print("-----------------------------------------------")
    print("interpretation:")
    print(" low p  → unified axis preferred even in subset")
    print(" high p → axis not preferred in subset (isotropic)")
    print("===============================================")

if __name__ == "__main__":
    main(sys.argv[1])
