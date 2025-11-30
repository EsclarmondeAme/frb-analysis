#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time phase-memory vs axis-distance test (test 85C)
----------------------------------------------------------------

this version is fully raw:

    - fixed theta bins
    - no fallbacks
    - no bin merging
    - no minimum FRB requirement
    - no reduction in harmonic basis
    - NaNs allowed (and meaningful)
    - MC null = 20000
    - custom pure-numpy spherical-harmonic basis (no scipy)

goal:
    measure the phase-memory statistic in fixed theta bins
    and allow the estimator to naturally succeed or fail.
    the failure pattern is physically meaningful.
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
    x = np.cos(Dec) * np.cos(RRA := RA)
    y = np.cos(Dec) * np.sin(RRA)
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
    v = np.array([np.cos(b)*np.cos(l),
                  np.cos(b)*np.sin(l),
                  np.sin(b)])
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
# pure numpy spherical harmonics
# ============================================================

def legendre_P(l, m, x):
    m_abs = abs(m)
    Pmm = np.ones_like(x)
    if m_abs > 0:
        somx2 = np.sqrt(1 - x*x)
        fact = 1.0
        for i in range(m_abs):
            Pmm *= -fact * somx2
            fact += 2.0
    if l == m_abs:
        return Pmm
    Pm1m = x * (2*m_abs + 1) * Pmm
    if l == m_abs + 1:
        return Pm1m
    Pll = None
    for ll in range(m_abs + 2, l + 1):
        Pll = ((2*ll - 1) * x * Pm1m - (ll + m_abs - 1) * Pmm) / (ll - m_abs)
        Pmm, Pm1m = Pm1m, Pll
    return Pll


def Ylm_pure(l, m, theta, phi):
    x = np.cos(theta)
    Plm = legendre_P(l, m, x)
    K = math.sqrt((2*l + 1)/(4*np.pi) *
                   math.factorial(l - abs(m)) /
                   math.factorial(l + abs(m)))

    if m > 0:
        return math.sqrt(2) * K * Plm * np.cos(m * phi)
    elif m < 0:
        return math.sqrt(2) * K * Plm * np.sin(abs(m) * phi)
    else:
        return K * Plm


def compute_Ylm_pure(X, lmax):
    N = len(X)
    Y = np.zeros((N, (lmax+1)*(lmax+1)))
    theta = np.arccos(np.clip(X[:,2], -1, 1))
    phi = np.arctan2(X[:,1], X[:,0])
    phi = np.mod(phi, 2*np.pi)

    idx = 0
    for l in range(lmax+1):
        for m in range(-l, l+1):
            Y[:, idx] = Ylm_pure(l, m, theta, phi)
            idx += 1
    return Y


# ============================================================
# raw phase memory statistic
# ============================================================

def phase_memory_stat_pure(X, sgn, lmax):
    Y = compute_Ylm_pure(X, lmax)

    pos = (sgn > 0)
    neg = (sgn < 0)

    if np.sum(pos) < 2 or np.sum(neg) < 2:
        return np.nan

    A_pos = np.sum(Y[pos], axis=0)
    A_neg = np.sum(Y[neg], axis=0)

    # raw phase difference surrogate
    dphi = np.abs(np.arctan2(A_pos - A_neg, np.ones_like(A_pos)))

    Z = np.mean(dphi)
    return Z


# ============================================================
# fixed-bin raw test
# ============================================================

def compute_85C_results(X, theta_deg, sgn,
                        lmax=10,
                        n_null=2000,
                        seed=42):

    rng = np.random.RandomState(seed)

    bins = [(0,20),(20,40),(40,60),(60,90),(90,180)]

    results = []

    for (a,b) in bins:
        print(f"[info] theta-bin {a}°–{b}°")

        mask = (theta_deg >= a) & (theta_deg < b)
        Xb = X[mask]
        sb = sgn[mask]

        Z_real = phase_memory_stat_pure(Xb, sb, lmax)

        if np.isnan(Z_real):
            print("  Z_real = NaN (insufficient structure)")
            results.append({
                "bin":(a,b),
                "Z":np.nan,
                "null_mean":np.nan,
                "p":np.nan
            })
            continue

        null_vals = []
        for _ in tqdm(range(n_null),
                      desc=f"null {a}-{b}",
                      leave=False):
            s_shuf = np.array(sb, copy=True)
            rng.shuffle(s_shuf)
            val = phase_memory_stat_pure(Xb, s_shuf, lmax)
            null_vals.append(val)

        null_vals = np.array(null_vals)
        p = (1 + np.sum(null_vals >= Z_real)) / (len(null_vals) + 1)
        mu = float(np.mean(null_vals))

        print(f"  Z={Z_real:.6f}, null_mean={mu:.6f}, p={p:.6f}")

        results.append({
            "bin":(a,b),
            "Z":float(Z_real),
            "null_mean":mu,
            "p":float(p)
        })

    return results


# ============================================================
# main
# ============================================================

def main(path):
    print("[info] loading FRB catalog…")
    RA, Dec = load_catalog(path)

    print("[info] converting to galactic xyz…")
    Xgal = radec_to_galactic_xyz(RA, Dec)

    axis = galactic_lb_to_xyz(159.8, -0.5)

    print("[info] computing axis angles + remnant signs…")
    theta = angle_from_axis(Xgal, axis)
    sgn = remnant_sign(Xgal, axis)

    print("[info] running test 85C core (raw, no fallbacks)…")
    results = compute_85C_results(Xgal, theta, sgn,
                                  lmax=10,
                                  n_null=2000,
                                  seed=42)

    print("================================================")
    print(" REMNANT-TIME PHASE-MEMORY VS AXIS-DISTANCE (TEST 85C)")
    print("================================================")
    for r in results:
        a,b = r["bin"]
        print(f"{a:3d}–{b:3d}°   Z={r['Z']}   null_mean={r['null_mean']}   p={r['p']}")
    print("================================================")
    print("interpretation:")
    print("  NaNs reveal where the estimator collapses.")
    print("  non-NaN bins show real harmonic phase memory.")
    print("  structure across bins reflects raw projection geometry.")
    print("================================================")
    print("test 85C complete.")
    print("================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_phase_memory_thetabins_test85C.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
