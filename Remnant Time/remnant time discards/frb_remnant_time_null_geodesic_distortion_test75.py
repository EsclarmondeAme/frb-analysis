#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time null geodesic distortion test (test 75)
--------------------------------------------------------

goal:
    detect whether the effective null geodesics (shortest angular paths)
    between FRB pairs are distorted in a directionally asymmetric way
    relative to the remnant-time field.

physics motivation:
    if higher-dimensional time compression exists, null-like propagation
    on the 3D celestial sphere will experience different "curvature"
    depending on time-direction. this test looks for that signature.

metric:
    for each hemisphere (R>0, R<0):
       1. compute spherical distance matrix D (exact)
       2. compute graph-geodesic approximation via kNN graph G (approx)
       3. define geodesic deviation Δd = G - D
       4. compute mean deviation per hemisphere
    score:
       G_real = mean(Δd_Rpos) - mean(Δd_Rneg)

null:
    2000 isotropic skies → same procedure → null distribution.

interpretation:
    low p -> directional temporal curvature (null-geodesic distortion)
    high p -> symmetric geodesics; no directional time effect
"""

import numpy as np
import csv
import sys
from tqdm import tqdm
import math

# ============================================================
# catalog utilities
# ============================================================

def detect_columns(fieldnames):
    low = [c.lower() for c in fieldnames]
    def find(*c):
        for n in c:
            if n.lower() in low:
                return fieldnames[low.index(n.lower())]
        return None
    ra = find("ra_deg","ra","raj2000","ra (deg)")
    dec = find("dec_deg","dec","dej2000","dec (deg)")
    if ra is None or dec is None:
        raise KeyError("could not detect RA/Dec columns")
    return ra, dec


def load_catalog(path):
    with open(path,"r",encoding="utf-8") as f:
        R = csv.DictReader(f)
        fields = R.fieldnames
        ra, dec = detect_columns(fields)
        RA,Dec = [],[]
        for row in R:
            RA.append(float(row[ra]))
            Dec.append(float(row[dec]))
    return np.array(RA), np.array(Dec)

# ============================================================
# coordinate transforms
# ============================================================

def radec_to_equatorial_xyz(RA,Dec):
    RA  = np.radians(RA)
    Dec = np.radians(Dec)
    x = np.cos(Dec)*np.cos(RA)
    y = np.cos(Dec)*np.sin(RA)
    z = np.sin(Dec)
    return np.vstack([x,y,z]).T


def equatorial_to_galactic_matrix():
    return np.array([
        [-0.054875539390, -0.873437104725, -0.483834991775],
        [ 0.494109453633, -0.444829594298,  0.746982248696],
        [-0.867666135681, -0.198076389622,  0.455983794523],
    ])


def radec_to_galactic_xyz(RA,Dec):
    Xeq = radec_to_equatorial_xyz(RA,Dec)
    M = equatorial_to_galactic_matrix()
    Xgal = Xeq @ M.T
    n = np.linalg.norm(Xgal,axis=1,keepdims=True)+1e-15
    return Xgal/n


def galactic_lb_to_xyz(l_deg,b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)
    v = np.array([x,y,z])
    return v/(np.linalg.norm(v)+1e-15)

# ============================================================
# remnant time functions
# ============================================================

def remnant_sign(X,axis):
    axis = axis/(np.linalg.norm(axis)+1e-15)
    R = X @ axis
    s = np.ones_like(R)
    s[R<0] = -1
    return s

# ============================================================
# geodesic / graph utilities
# ============================================================

def spherical_dist_matrix(X):
    dots = X @ X.T
    np.clip(dots,-1,1,out=dots)
    return np.degrees(np.arccos(dots))


def build_knn_graph(D,k=10):
    N = D.shape[0]
    idx = np.argpartition(D,kth=k,axis=1)[:, :k]
    G = np.full((N,N), np.inf, dtype=float)
    for i in range(N):
        G[i, idx[i]] = D[i, idx[i]]
    return G


def floyd_warshall(dist):
    """
    classical Floyd-Warshall APSP.
    n ~ 300 per hemisphere => OK.
    """
    D = dist.copy()
    N = D.shape[0]
    for k in range(N):
        D = np.minimum(D, D[:,k,None] + D[k,None,:])
    return D

# ============================================================
# temporal geodesic distortion metric
# ============================================================

def geodesic_distortion(X):
    """
    compute average geodesic deviation:
       Δd = G_apsp - D_true
    """
    D_true = spherical_dist_matrix(X)
    np.fill_diagonal(D_true,0)

    # build graph geodesic approx
    G = build_knn_graph(D_true, k=10)

    # all-pairs shortest paths
    G_apsp = floyd_warshall(G)

    # deviation
    Dev = G_apsp - D_true
    Dev = Dev[np.isfinite(Dev)]
    return float(np.mean(Dev))


def compute_temporal_curvature(Xgal,axis):
    sign = remnant_sign(Xgal,axis)
    Xpos = Xgal[sign>0]
    Xneg = Xgal[sign<0]

    dpos = geodesic_distortion(Xpos)
    dneg = geodesic_distortion(Xneg)

    G = dpos - dneg
    return G, dpos, dneg


def random_isotropic(N):
    u = np.random.uniform(-1,1,N)
    phi = np.random.uniform(0,2*np.pi,N)
    st = np.sqrt(1-u*u)
    x = st*np.cos(phi)
    y = st*np.sin(phi)
    z = u
    X = np.vstack([x,y,z]).T
    n = np.linalg.norm(X,axis=1,keepdims=True)+1e-15
    return X/n

# ============================================================
# main
# ============================================================

def main(path):
    print("[info] loading frb catalog...")
    RA,Dec = load_catalog(path)
    N = len(RA)
    print(f"[info] N_FRB = {N}")

    print("[info] converting to galactic coords...")
    Xgal = radec_to_galactic_xyz(RA,Dec)

    axis = galactic_lb_to_xyz(159.8,-0.5)

    print("[info] computing real temporal curvature...")
    G_real, dpos, dneg = compute_temporal_curvature(Xgal,axis)

    print("[info] building MC null (2000 skies)...")
    G_null = []
    for _ in tqdm(range(2000), desc="MC"):
        Xmc = random_isotropic(N)
        Gmc,_,_ = compute_temporal_curvature(Xmc,axis)
        G_null.append(Gmc)
    G_null = np.array(G_null)

    mu = float(np.mean(G_null))
    sd = float(np.std(G_null))
    p = (1 + np.sum(G_null >= G_real))/(len(G_null)+1)

    print("================================================")
    print(" frb remnant-time null geodesic distortion test (test 75)")
    print("================================================")
    print(f"mean distortion (R>0) = {dpos:.6f}")
    print(f"mean distortion (R<0) = {dneg:.6f}")
    print(f"G_real                = {G_real:.6f}")
    print("------------------------------------------------")
    print(f"null mean G           = {mu:.6f}")
    print(f"null std G            = {sd:.6f}")
    print(f"p-value               = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  - low p  -> null-geodesics bend more on one remnant-time side,")
    print("             consistent with directional temporal compression.")
    print("  - high p -> geodesic structure is symmetric; no temporal curvature.")
    print("================================================")
    print("test 75 complete.")
    print("================================================")


if __name__ == "__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_null_geodesic_distortion_test75.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
