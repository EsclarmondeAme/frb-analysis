#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from math import radians, sin, cos, acos, pi
from tqdm import tqdm

# ============================================================
# utilities
# ============================================================

def ang2vec(ra, dec):
    ra = np.radians(ra)
    dec = np.radians(dec)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack((x, y, z))

def dot(a, b):
    return np.sum(a*b, axis=1)

def angsep(ra1, dec1, ra2, dec2):
    ra1 = np.radians(ra1); dec1 = np.radians(dec1)
    ra2 = np.radians(ra2); dec2 = np.radians(dec2)
    return np.degrees(
        np.arccos(
            np.sin(dec1)*np.sin(dec2) +
            np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
        )
    )

# ============================================================
# supergalactic conversion
# ============================================================

# J2000 → supergalactic north pole
# de Vaucouleurs definition
SGL_NP_RA  = radians(283.25)
SGL_NP_DEC = radians(15.70)
SGL_LON_ZERO = radians(47.37)

def eq_to_supergalactic(ra_deg, dec_deg):
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    sinb = ( np.sin(dec)*np.sin(SGL_NP_DEC)
            + np.cos(dec)*np.cos(SGL_NP_DEC)*np.cos(ra - SGL_NP_RA) )
    b = np.arcsin(sinb)
    y = np.cos(dec)*np.sin(ra - SGL_NP_RA)
    x = ( np.sin(dec)*np.cos(SGL_NP_DEC)
        - np.cos(dec)*np.sin(SGL_NP_DEC)*np.cos(ra - SGL_NP_RA) )
    l = np.arctan2(y, x) + SGL_LON_ZERO
    return np.degrees(l) % 360.0, np.degrees(b)

# ============================================================
# curvature–drift ENGINE (matches your 76)
# ============================================================

def compute_curvature(vecs):
    # local curvature proxy = |mean of pairwise dot products|
    # same engine as 76
    N = vecs.shape[0]
    if N < 3:
        return np.nan
    D = vecs @ vecs.T
    return np.mean(np.abs(D - np.eye(N)))

def compute_drift(vecs):
    # drift proxy = |mean nearest-neighbour angular difference|
    N = vecs.shape[0]
    if N < 2:
        return np.nan
    drift = []
    for i in range(N):
        vi = vecs[i]
        dots = vecs @ vi
        dots[i] = -1  # exclude self
        j = np.argmax(dots)
        ang = acos(max(-1, min(1, dots[j])))
        drift.append(ang)
    return np.mean(drift)

# ============================================================
# main
# ============================================================

def main(path):
    print("===============================================")
    print("FRB REMNANT-TIME CURVATURE–DRIFT TEST (76B)")
    print("Supergalactic mask: |SGB| >= 20°")
    print("===============================================")

    df = pd.read_csv(path)
    RA  = df["ra"].values
    Dec = df["dec"].values
    th  = df["theta_unified"].values  # remnant-time sign comes from unified-axis θ
    vec = ang2vec(RA, Dec)

    # compute SGB mask
    _, SGB = eq_to_supergalactic(RA, Dec)
    mask = np.abs(SGB) >= 20
    RA   = RA[mask]
    Dec  = Dec[mask]
    th   = th[mask]
    vec  = vec[mask]

    print(f"[info] original N=600, after SGB mask N={len(RA)}")

    # hemispheres
    pos = th < 90
    neg = th >= 90
    Vp = vec[pos]
    Vn = vec[neg]

    # real metrics
    curv_pos = compute_curvature(Vp)
    curv_neg = compute_curvature(Vn)
    drift_pos = compute_drift(Vp)
    drift_neg = compute_drift(Vn)
    D_real = (drift_pos - drift_neg) + (curv_pos - curv_neg)

    # Monte Carlo: random hemisphere assignments
    N = len(vec)
    Np = np.sum(pos)
    Nm = np.sum(neg)
    mc = []

    for _ in range(2000):
        perm = np.random.permutation(N)
        rnd_pos = vec[perm[:Np]]
        rnd_neg = vec[perm[Np:]]
        cpos = compute_curvature(rnd_pos)
        cneg = compute_curvature(rnd_neg)
        dpos = compute_drift(rnd_pos)
        dneg = compute_drift(rnd_neg)
        mc.append( (dpos - dneg) + (cpos - cneg) )

    mc = np.array(mc)
    mean_mc = np.mean(mc)
    std_mc  = np.std(mc)
    # full-precision p-value
    p = np.mean(mc <= D_real) if D_real < mean_mc else np.mean(mc >= D_real)

    print("------------------------------------------------")
    print(f"mean curvature (R>0) = {curv_pos}")
    print(f"mean curvature (R<0) = {curv_neg}")
    print(f"mean drift     (R>0) = {drift_pos}")
    print(f"mean drift     (R<0) = {drift_neg}")
    print(f"D_real               = {D_real}")
    print("------------------------------------------------")
    print(f"null mean D          = {mean_mc}")
    print(f"null std D           = {std_mc}")
    print(f"p-value              = {p:.12f}")   # full precision
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> curvature–drift differs between hemispheres")
    print("            even after removing the supergalactic plane.")
    print("  high p -> symmetric; consistent with isotropy.")
    print("===============================================")
    print("test 76B complete.")
    print("===============================================")


if __name__ == "__main__":
    main(sys.argv[1])
