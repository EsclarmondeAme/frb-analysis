#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import pandas as pd
from astropy.io import fits

# ================================================
# ASKAP — identical pattern as Test 80C
# ================================================

ASKAP_DIR = "data/positions"
MATCH_TOL = 2.0   # degrees — same as 80C

def angsep(ra1, dec1, ra2, dec2):
    r1 = np.radians(ra1)
    d1 = np.radians(dec1)
    r2 = np.radians(ra2)
    d2 = np.radians(dec2)
    return np.degrees(
        np.arccos(
            np.clip(
                np.sin(d1)*np.sin(d2) +
                np.cos(d1)*np.cos(d2)*np.cos(r1-r2),
                -1, 1
            )
        )
    )

def load_askap_pointings():
    ras, decs = [], []
    if not os.path.exists(ASKAP_DIR):
        return np.array([]), np.array([])

    for f in os.listdir(ASKAP_DIR):
        if not f.lower().endswith(".fits"):
            continue
        try:
            hdr = fits.getheader(os.path.join(ASKAP_DIR, f))
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

# ================================================
# Joint entropy (Test 91 logic)
# ================================================

def bin_var(x, edges):
    idx = np.digitize(x, edges) - 1
    idx[(x < edges[0]) | (x >= edges[-1])] = -1
    return idx

def shannon_entropy_joint(theta_u, rt_sign, phi_h):
    theta_edges = np.array([0, 20, 35, 50, 90, 180])
    rt_edges    = np.array([-2, 0, 2])
    phi_edges   = np.linspace(0, 2*np.pi, 13)

    phi_h = np.mod(phi_h, 2*np.pi)

    bT = bin_var(theta_u, theta_edges)
    bR = bin_var(rt_sign, rt_edges)
    bP = bin_var(phi_h, phi_edges)

    ok = (bT >= 0) & (bR >= 0) & (bP >= 0)
    if not np.any(ok):
        return np.nan

    C = np.zeros((5,2,12), dtype=int)
    for t, r, p in zip(bT[ok], bR[ok], bP[ok]):
        C[t,r,p] += 1

    P = C[C>0].astype(float) / np.sum(C)
    return -np.sum(P*np.log(P))

def test91_stat(theta_u, rt_sign, phi_h, n_null=2000, seed=1):
    rng = np.random.default_rng(seed)
    H_real = shannon_entropy_joint(theta_u, rt_sign, phi_h)

    rt_vals  = rt_sign.copy()
    ph_vals  = phi_h.copy()

    Hnull = np.zeros(n_null)
    for i in range(n_null):
        rt_s = rng.permutation(rt_vals)
        ph_s = rng.permutation(ph_vals)
        Hnull[i] = shannon_entropy_joint(theta_u, rt_s, ph_s)

    return (H_real,
            Hnull.mean(),
            Hnull.std(ddof=1),
            np.mean(Hnull <= H_real))

# ================================================
# Test 95 — ASKAP split
# ================================================

def main(path):

    print("===============================================")
    print(" Test 95 — Joint Entropy (Test 91) Under ASKAP Split")
    print("===============================================")

    df = pd.read_csv(path)
    RA  = df["ra"].values
    Dec = df["dec"].values

    theta = df["theta_u"].values
    rt    = df["rt_sign"].values
    phi   = df["phi_h"].values

    # load ASKAP FITS positions
    RAa, Deca = load_askap_pointings()
    print(f"[info] ASKAP pointings loaded: {len(RAa)}")

    # match ASKAP subset
    isA = match_askap(RA, Dec, RAa, Deca)
    isN = ~isA

    print(f"[info] ASKAP subset size    = {np.sum(isA)}")
    print(f"[info] non-ASKAP subset size= {np.sum(isN)}")

    # full-sample (for reference)
    Hf, mf, sf, pf = test91_stat(theta, rt, phi)

    # ASKAP subset
    Ha, ma, sa, pa = test91_stat(theta[isA], rt[isA], phi[isA])

    # non-ASKAP
    Hn, mn, sn, pn = test91_stat(theta[isN], rt[isN], phi[isN])

    # print results
    print("-----------------------------------------------")
    print(f"FULL:     H={Hf:.6f}, null_mean={mf:.6f}, null_std={sf:.6f}, p={pf:.6f}")
    print(f"ASKAP:    H={Ha:.6f}, null_mean={ma:.6f}, null_std={sa:.6f}, p={pa:.6f}")
    print(f"non-ASKAP H={Hn:.6f}, null_mean={mn:.6f}, null_std={sn:.6f}, p={pn:.6f}")
    print("-----------------------------------------------")
    print("interpretation:")
    print(" low p  → joint-entropy deficit present in subset")
    print(" high p → subset consistent with isotropy")
    print("===============================================")

if __name__ == "__main__":
    main(sys.argv[1])
