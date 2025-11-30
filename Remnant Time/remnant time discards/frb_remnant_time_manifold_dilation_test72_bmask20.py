#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB remnant-time manifold dilation test (72A — fixed)
Galactic plane mask |b| >= 20°, adaptive-K KNN thickness without modifying Test 72 engine
"""

import numpy as np
import csv
import sys
from tqdm import tqdm
import frb_remnant_time_manifold_dilation_test72 as T72  # your engine


UNIFIED_L = 159.8
UNIFIED_B = -0.5


# ============================================================
# coordinate utilities
# ============================================================

def galactic_lb_to_xyz(l_deg, b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)
    v = np.array([x, y, z])
    return v/np.linalg.norm(v)


def detect_columns(fields):
    low = [c.lower() for c in fields]
    def find(*cands):
        for c in cands:
            if c.lower() in low:
                return fields[low.index(c.lower())]
        return None
    ra = find("ra","ra_deg","raj2000","ra(deg)","ra(degrees)")
    dec = find("dec","dec_deg","dej2000","dec(deg)","dec(degrees)")
    if ra is None or dec is None:
        raise KeyError("RA/Dec columns not found")
    return ra, dec


def load_catalog(path):
    RA, Dec = [], []
    with open(path,"r",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ra_key, dec_key = detect_columns(reader.fieldnames)
        for row in reader:
            RA.append(float(row[ra_key]))
            Dec.append(float(row[dec_key]))
    return np.array(RA), np.array(Dec)


def radec_to_equatorial_xyz(RA, Dec):
    RA = np.radians(RA)
    Dec = np.radians(Dec)
    x = np.cos(Dec)*np.cos(RA)
    y = np.cos(Dec)*np.sin(RA)
    z = np.sin(Dec)
    return np.vstack([x,y,z]).T


def equatorial_to_galactic_matrix():
    return np.array([
        [-0.054875539390,-0.873437104725,-0.483834991775],
        [ 0.494109453633,-0.444829594298, 0.746982248696],
        [-0.867666135681,-0.198076389622, 0.455983794523],
    ])


def radec_to_galactic_xyz(RA, Dec):
    Xeq = radec_to_equatorial_xyz(RA, Dec)
    M = equatorial_to_galactic_matrix()
    Xgal = Xeq @ M.T
    return Xgal/np.linalg.norm(Xgal,axis=1,keepdims=True)


def galactic_latitudes_from_xyz(Xgal):
    return np.degrees(np.arcsin(np.clip(Xgal[:,2],-1,1)))


def random_axis():
    phi = np.random.uniform(0,2*np.pi)
    u   = np.random.uniform(-1,1)
    th  = np.arccos(u)
    x = np.sin(th)*np.cos(phi)
    y = np.sin(th)*np.sin(phi)
    z = np.cos(th)
    return np.array([x,y,z])


# ============================================================
# adaptive-k patch mechanism
# ============================================================

def compute_dilation_with_adaptive_k(Xgal, axis_xyz):
    """
    temporarily patches T72.knn_thickness with adaptive-k version,
    runs compute_dilation_metrics unchanged,
    restores original function afterwards.
    """

    # hemisphere split to determine k
    dots = Xgal @ axis_xyz
    pos = Xgal[dots >= 0]
    neg = Xgal[dots < 0]

    Np, Nn = len(pos), len(neg)
    if Np < 5 or Nn < 5:
        return np.nan, np.nan, np.nan, np.nan, Np, Nn

    k = min(
        max(4, Np//3),
        max(4, Nn//3),
        12,
    )

    # original function
    original_knn = T72.knn_thickness

    # patched version with adaptive k
    def knn_thickness_patched(X):
        D = np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(axis=2))
        np.fill_diagonal(D,np.inf)
        idx = np.argpartition(D, kth=k, axis=1)[:, :k]
        return D[np.arange(len(X))[:,None], idx].mean(axis=1).mean()

    # patch
    T72.knn_thickness = knn_thickness_patched

    # compute using original engine
    S_real, d_pair, d_knn, d_h = T72.compute_dilation_metrics(Xgal, axis_xyz)

    # restore
    T72.knn_thickness = original_knn

    return S_real, d_pair, d_knn, d_h, Np, Nn


# ============================================================
# main
# ============================================================

def main(path, n_mc=2000):
    print("==============================================================")
    print("FRB REMNANT-TIME MANIFOLD DILATION TEST (72A — fixed)")
    print("Galactic mask |b|>=20°, adaptive-K without modifying engine")
    print("==============================================================")

    RA, Dec = load_catalog(path)
    print(f"[info] N_FRB original = {len(RA)}")

    Xgal_full = radec_to_galactic_xyz(RA,Dec)
    b = galactic_latitudes_from_xyz(Xgal_full)
    mask = np.abs(b) >= 20
    Xgal = Xgal_full[mask]
    print(f"[info] N_after_mask = {len(Xgal)}")

    axis_unified = galactic_lb_to_xyz(UNIFIED_L,UNIFIED_B)

    print("[info] computing real...")
    S_real, d_pair, d_knn, d_h, Np, Nn = compute_dilation_with_adaptive_k(Xgal,axis_unified)

    print("[info] building null (rotating axis)...")
    S_null = np.empty(n_mc)
    for i in tqdm(range(n_mc),ncols=80):
        ax = random_axis()
        S_null[i],_,_,_,_,_ = compute_dilation_with_adaptive_k(Xgal,ax)

    null_mean = np.nanmean(S_null)
    null_std  = np.nanstd(S_null)
    p_value   = np.mean(S_null >= S_real)

    print("==============================================================")
    print("FRB remnant-time manifold dilation test (72A — fixed)")
    print("==============================================================")
    print(f"N_pos = {Np}, N_neg = {Nn}, N_total = {len(Xgal)}")
    print("--------------------------------------------------------------")
    print(f"delta_pairwise = {d_pair:.6f}")
    print(f"delta_knn      = {d_knn:.6f}")
    print(f"delta_harmonic = {d_h:.6f}")
    print("--------------------------------------------------------------")
    print(f"S_real         = {S_real:.6f}")
    print(f"null mean S    = {null_mean:.6f}")
    print(f"null std S     = {null_std:.6f}")
    print(f"p-value        = {p_value:.6f}")
    print("==============================================================")
    print("test 72A complete.")
    print("==============================================================")



if __name__=="__main__":
    path = sys.argv[1]
    n_mc = int(sys.argv[2]) if len(sys.argv)>2 else 2000
    main(path,n_mc)
