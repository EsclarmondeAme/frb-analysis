#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time phase-memory vs axis-distance test (test 85)
----------------------------------------------------------------

goal:
    measure how the phase-memory statistic (from test 81) varies
    as a function of angular distance from the unified axis.

motivation:
    in the remnant-time hypothesis, the "time compression" is not
    uniform across the sky. events closer to the unified axis are
    less compressed in the projection, while those farther away
    are more compressed. this should produce a gradient:

        Z(theta-bin) should vary systematically with theta.

    this test bins FRBs by theta_unified and computes the harmonic
    phase memory statistic Z separately in each bin, using a
    2000-realisation permutation null for robustness.

design:
    - unified axis fixed (no scan)
    - tangent-plane spin-0 harmonic-phase field (test 81 logic)
    - bins: [0–20°], [20–40°], [40–60°], [60–180°]
    - null: remnant-sign shuffle only (footprint-preserving)
    - output: Z_real_in_bin, Z_null_distribution_in_bin
"""

import numpy as np
import csv
import sys
import math
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.special import sph_harm


# ============================================================
# catalog utilities
# ============================================================

def detect_columns(fieldnames):
    low = [c.lower() for c in fieldnames]
    def find(*names):
        for n in names:
            if n.lower() in low:
                return fieldnames[low.index(n.lower())]
        return None
    ra = find("ra_deg", "ra", "raj2000", "ra (deg)")
    dec = find("dec_deg", "dec", "dej2000", "dec (deg)")
    if ra is None or dec is None:
        raise KeyError("could not detect RA/Dec columns")
    return ra, dec


def load_catalog(path):
    with open(path, "r", encoding="utf-8") as f:
        R = csv.DictReader(f)
        ra, dec = detect_columns(R.fieldnames)
        RA, Dec = [], []
        for row in R:
            RA.append(float(row[ra]))
            Dec.append(float(row[dec]))
    return np.array(RA), np.array(Dec)


# ============================================================
# coordinate transforms
# ============================================================

def radec_to_equatorial_xyz(RA, Dec):
    RA = np.radians(RA)
    Dec = np.radians(Dec)
    x = np.cos(Dec) * np.cos(RA)
    y = np.cos(Dec) * np.sin(RA)
    z = np.sin(Dec)
    return np.vstack([x, y, z]).T


def equatorial_to_galactic_matrix():
    return np.array([
        [-0.054875539390, -0.873437104725, -0.483834991775],
        [ 0.494109453633, -0.444829594298,  0.746982248696],
        [-0.867666135681, -0.198076389622,  0.455983794523],
    ])


def radec_to_galactic_xyz(RA, Dec):
    Xeq = radec_to_equatorial_xyz(RA, Dec)
    M = equatorial_to_galactic_matrix()
    Xgal = Xeq @ M.T
    return Xgal / (np.linalg.norm(Xgal, axis=1, keepdims=True) + 1e-15)


def galactic_lb_to_xyz(l_deg, b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    v = np.array([
        np.cos(b)*np.cos(l),
        np.cos(b)*np.sin(l),
        np.sin(b)
    ])
    return v / (np.linalg.norm(v) + 1e-15)


# ============================================================
# remnant geometry
# ============================================================

def angle_from_axis(X, axis_vec):
    axis = axis_vec / (np.linalg.norm(axis_vec) + 1e-15)
    dots = np.clip(X @ axis, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def remnant_sign(X, axis_vec):
    axis = axis_vec / (np.linalg.norm(axis_vec) + 1e-15)
    v = X @ axis
    s = np.ones(len(X), dtype=int)
    s[v < 0] = -1
    return s


# ============================================================
# harmonic phase memory (test 81 core)
# ============================================================

def compute_Ylm(X, lmax=10):
    N = len(X)
    Y = np.zeros((N, (lmax+1)*(lmax+1)), dtype=complex)

    th = np.arccos(np.clip(X[:,2], -1, 1))
    ph = np.arctan2(X[:,1], X[:,0])
    if np.any(ph < 0):
        ph = np.mod(ph, 2*np.pi)

    idx = 0
    for l in range(lmax+1):
        for m in range(-l, l+1):
            Y[:, idx] = sph_harm(m, l, ph, th)
            idx += 1
    return Y


def phase_memory_stat(X, sgn, lmax=10):
    Y = compute_Ylm(X, lmax=lmax)

    pos = (sgn > 0)
    neg = (sgn < 0)
    if np.sum(pos) < 5 or np.sum(neg) < 5:
        return np.nan

    ph_pos = np.angle(np.sum(Y[pos], axis=0))
    ph_neg = np.angle(np.sum(Y[neg], axis=0))

    dphi = np.angle(np.exp(1j*(ph_pos - ph_neg)))
    Z = np.sum(np.abs(dphi)) / len(dphi)
    return Z


# ============================================================
# main binning test
# ============================================================

def compute_85_results(X, theta_deg, sgn,
                       bins=[(0,20),(20,40),(40,60),(60,180)],
                       lmax=10,
                       n_null=2000,
                       seed=42):
    rng = np.random.RandomState(seed)
    results = []

    for (a,b) in bins:
        print(f"[info] theta-bin {a}°–{b}°")

        mask = (theta_deg >= a) & (theta_deg < b)
        if np.sum(mask) < 20:
            print("[warn] fewer than 20 FRBs in this bin — skip.")
            results.append({
                "bin": (a,b),
                "Z_real": np.nan,
                "null_mean": np.nan,
                "null_std": np.nan,
                "p": np.nan
            })
            continue

        Xb = X[mask]
        sb = sgn[mask]

        Z_real = phase_memory_stat(Xb, sb, lmax=lmax)

        null_vals = []
        for _ in tqdm(range(n_null),
                      desc=f"null bin {a}-{b}",
                      leave=False):
            s_shuf = np.array(sb, copy=True)
            rng.shuffle(s_shuf)
            val = phase_memory_stat(Xb, s_shuf, lmax=lmax)
            if not np.isfinite(val):
                val = 0.0
            null_vals.append(val)

        null_vals = np.array(null_vals)
        mu = float(np.mean(null_vals))
        sigma = float(np.std(null_vals))
        p = (1 + np.sum(null_vals >= Z_real)) / (len(null_vals) + 1)

        print(f"  Z_real = {Z_real:.6f}")
        print(f"  null mean = {mu:.6f}, null std = {sigma:.6f}, p = {p:.6f}")

        results.append({
            "bin": (a,b),
            "Z_real": float(Z_real),
            "null_mean": mu,
            "null_std": sigma,
            "p": float(p)
        })

    return results


# ============================================================
# main driver
# ============================================================

def main(path):
    print("[info] loading FRB catalog...")
    RA, Dec = load_catalog(path)
    N = len(RA)
    print(f"[info] N_FRB = {N}")

    print("[info] converting to galactic xyz...")
    Xgal = radec_to_galactic_xyz(RA, Dec)

    axis = galactic_lb_to_xyz(159.8, -0.5)

    print("[info] computing axis angles + remnant signs...")
    theta = angle_from_axis(Xgal, axis)
    sgn = remnant_sign(Xgal, axis)

    print("[info] running test 85 core...")
    results = compute_85_results(Xgal, theta, sgn,
                                 bins=[(0,20),(20,40),(40,60),(60,180)],
                                 lmax=10,
                                 n_null=2000,
                                 seed=42)

    print("================================================")
    print(" REMNANT-TIME PHASE-MEMORY VS AXIS-DISTANCE (TEST 85)")
    print("================================================")
    for r in results:
        a,b = r["bin"]
        print(f"{a:3d}–{b:3d}°   Z_real={r['Z_real']:.6f}   "
              f"null_mean={r['null_mean']:.6f}   "
              f"p={r['p']:.6f}")
    print("================================================")
    print("interpretation:")
    print("  variation of Z(theta-bin) would indicate a")
    print("  gradient in temporal coherence strength,")
    print("  consistent with a non-uniform projection")
    print("  of the remnant-time field across the sky.")
    print("================================================")
    print("test 85 complete.")
    print("================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_phase_memory_thetabins_test85.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
