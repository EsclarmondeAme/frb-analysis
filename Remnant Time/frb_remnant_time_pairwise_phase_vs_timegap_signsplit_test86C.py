#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB REMNANT-TIME: PAIRWISE PHASE VS TIME-SEPARATION — SIGN SPLIT (TEST 86C)
This test checks whether phase alignment depends on observation-time differences
*within* remnant-time hemispheres and *across* hemispheres.

Expected for geometric remnant-time field:
    - same hemisphere:   rho_same ≈ 0
    - opposite hemisphere: rho_opp ≠ 0 (weak but significant)
"""

import sys
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from scipy.special import sph_harm
from scipy.stats import pearsonr

# unified axis (from previous tests)
AXIS_L = np.deg2rad(159.8)
AXIS_B = np.deg2rad(-0.5)

def load_catalog(path):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    return df

def ra_dec_to_galactic(ra, dec):
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    l_ncp = np.deg2rad(122.932)
    ra_gp = np.deg2rad(192.85948)
    dec_gp = np.deg2rad(27.12825)

    b = np.arcsin(np.sin(dec)*np.sin(dec_gp) +
                  np.cos(dec)*np.cos(dec_gp)*np.cos(ra - ra_gp))

    l = np.arctan2(np.sin(dec)*np.cos(dec_gp) -
                   np.cos(dec)*np.sin(dec_gp)*np.cos(ra - ra_gp),
                   np.cos(dec)*np.sin(ra - ra_gp)) + l_ncp
    l = (l + 2*np.pi) % (2*np.pi)
    return l, b

def time_to_seconds(tstr):
    try:
        dt = datetime.fromisoformat(str(tstr))
        return dt.timestamp()
    except Exception:
        return np.nan

def compute_phase_matrix(l, b, lmax=8):
    theta = np.pi/2 - b
    phi = l
    Y = []
    for ell in range(1, lmax+1):
        for m in range(-ell, ell+1):
            Y.append(sph_harm(m, ell, phi, theta))
    Y = np.vstack(Y).T
    phases = np.angle(Y)
    return phases

def pairwise_indices(N):
    i_idx = []
    j_idx = []
    for i in range(N):
        for j in range(i+1, N):
            i_idx.append(i)
            j_idx.append(j)
    return np.array(i_idx), np.array(j_idx)

def corr_coef(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return pearsonr(x, y)[0]

def compute_signs(l, b):
    ux = np.cos(AXIS_b:=AXIS_B) * np.cos(AXIS_l:=AXIS_L)
    uy = np.cos(AXIS_B) * np.sin(AXIS_L)
    uz = np.sin(AXIS_B)
    ax = ux; ay = uy; az = uz

    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)

    dot = x*ax + y*ay + z*az
    return np.where(dot >= 0, +1, -1)

def main(path):

    print("="*60)
    print("PAIRWISE PHASE VS TIME-SEPARATION — SIGN SPLIT (TEST 86C)")
    print("="*60)

    df = load_catalog(path)

    if not all(k in df.columns for k in ["ra", "dec"]):
        print("[fatal] RA/Dec missing.")
        sys.exit(1)

    # convert time column
    if "utc" in df.columns:
        tvals = df["utc"].apply(time_to_seconds).values
        if np.any(np.isnan(tvals)):
            print("[warn] some UTC values could not parse.")
    elif "mjd" in df.columns:
        tvals = df["mjd"].astype(float).values * 86400.0
    else:
        print("[fatal] no usable time column.")
        sys.exit(1)

    # galactic
    l, b = ra_dec_to_galactic(df["ra"].values, df["dec"].values)

    N = len(df)
    print(f"[info] N_FRB = {N}")

    # phases
    phases = compute_phase_matrix(l, b, lmax=8)

    # pairwise indexing
    i_idx, j_idx = pairwise_indices(N)

    # phase alignment G
    dphi = phases[j_idx] - phases[i_idx]
    G = np.mean(np.cos(dphi), axis=1)

    # time separation
    dt = np.abs(tvals[j_idx] - tvals[i_idx])

    # signs
    signs = compute_signs(l, b)

    same = signs[i_idx] == signs[j_idx]
    opp  = ~same

    # real correlations
    rho_same = corr_coef(dt[same], G[same])
    rho_opp  = corr_coef(dt[opp], G[opp])

    print("---------------------------------------------------")
    print(f"[info] rho_same = {rho_same:.6e}")
    print(f"[info] rho_opp  = {rho_opp:.6e}")
    print("---------------------------------------------------")

    # null distributions (shuffle signs)
    n_null = 2000
    null_same = []
    null_opp  = []

    for _ in range(n_null):
        s = np.random.permutation(signs)
        sh_same = s[i_idx] == s[j_idx]
        sh_opp  = ~sh_same

        null_same.append(corr_coef(dt[sh_same], G[sh_same]))
        null_opp.append(corr_coef(dt[sh_opp],  G[sh_opp]))

    null_same = np.array(null_same)
    null_opp  = np.array(null_opp)

    p_same = (np.sum(np.abs(null_same) >= np.abs(rho_same)) + 1) / (n_null+1)
    p_opp  = (np.sum(np.abs(null_opp)  >= np.abs(rho_opp))  + 1) / (n_null+1)

    print("NULL DISTRIBUTION")
    print(f"  same: mean={null_same.mean():.3e}, std={null_same.std():.3e}, p={p_same:.6f}")
    print(f"  opp:  mean={null_opp.mean():.3e},  std={null_opp.std():.3e},  p={p_opp:.6f}")

    print("---------------------------------------------------")
    print("interpretation:")
    print("  - rho_same ≈ 0 with high p → phase-memory independent of time inside hemisphere")
    print("  - rho_opp  ≠ 0 with low p → crossing hemispheres carries geometric phase structure")
    print("---------------------------------------------------")
    print("test 86C complete.")
    print("="*60)

if __name__ == "__main__":
    main(sys.argv[1])
