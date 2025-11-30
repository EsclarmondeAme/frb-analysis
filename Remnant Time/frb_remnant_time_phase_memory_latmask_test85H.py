#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time phase-memory — galactic latitude mask robustness (test 85H)
-----------------------------------------------------------------------------

goal:
    test whether the global remnant-time phase-memory signal
    survives when we progressively remove the galactic plane:

        |b| >= 0°   (no cut, reference)
        |b| >= 20°
        |b| >= 30°
        |b| >= 40°

    if the effect is due to galactic foregrounds or survey strategy
    tied to the plane, it should weaken or disappear as we tighten
    the |b| cut. if it is cosmological and global, it should remain.

design:
    - same unified axis: (l, b) = (159.8°, -0.5°)
    - pure numpy spherical harmonics, lmax=10
    - global phase-memory statistic
    - monte carlo null: 2000 random remnant-sign shuffles
"""

import numpy as np
import csv
import sys
import math
from tqdm import tqdm


# ------------------------------------------------------------
# catalog utilities
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# coordinate transforms
# ------------------------------------------------------------

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
        np.cos(b) * np.cos(l),
        np.cos(b) * np.sin(l),
        np.sin(b)
    ])
    return v / (np.linalg.norm(v) + 1e-15)


def xyz_to_galactic_lb(X):
    """
    convert galactic xyz to longitude/latitude (degrees)
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    b = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    l = np.degrees(np.arctan2(y, x))
    l = np.mod(l, 360.0)
    return l, b


# ------------------------------------------------------------
# remnant geometry
# ------------------------------------------------------------

def remnant_sign(X, axis_vec):
    axis = axis_vec / (np.linalg.norm(axis_vec) + 1e-15)
    return np.where(X @ axis > 0, 1, -1)


# ------------------------------------------------------------
# pure numpy spherical harmonics
# ------------------------------------------------------------

def legendre_P(l, m, x):
    m_abs = abs(m)
    Pmm = np.ones_like(x)
    if m_abs > 0:
        somx2 = np.sqrt(1 - x * x)
        fact = 1.0
        for _ in range(m_abs):
            Pmm *= -fact * somx2
            fact += 2.0
    if l == m_abs:
        return Pmm
    Pm1m = x * (2 * m_abs + 1) * Pmm
    if l == m_abs + 1:
        return Pm1m
    Pll = None
    for ll in range(m_abs + 2, l + 1):
        Pll = ((2 * ll - 1) * x * Pm1m - (ll + m_abs - 1) * Pmm) / (ll - m_abs)
        Pmm, Pm1m = Pm1m, Pll
    return Pll


def Ylm_pure(l, m, theta, phi):
    x = np.cos(theta)
    Plm = legendre_P(l, m, x)
    K = math.sqrt((2 * l + 1) / (4 * math.pi) *
                  math.factorial(l - abs(m)) /
                  math.factorial(l + abs(m)))
    if m > 0:
        return math.sqrt(2) * K * Plm * np.cos(m * phi)
    elif m < 0:
        return math.sqrt(2) * K * Plm * np.sin(abs(m) * phi)
    else:
        return K * Plm


def compute_Ylm_pure(X, lmax=10):
    theta = np.arccos(np.clip(X[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2 * math.pi)
    Y = np.zeros((len(X), (lmax + 1) * (lmax + 1)))
    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y[:, idx] = Ylm_pure(l, m, theta, phi)
            idx += 1
    return Y


# ------------------------------------------------------------
# global phase-memory
# ------------------------------------------------------------

def global_phase_memory_from_Y(Y, sgn):
    pos = (sgn > 0)
    neg = (sgn < 0)
    if np.sum(pos) < 2 or np.sum(neg) < 2:
        return np.nan
    A_pos = np.sum(Y[pos], axis=0)
    A_neg = np.sum(Y[neg], axis=0)
    dphi = np.abs(np.arctan2(A_pos - A_neg, np.ones_like(A_pos)))
    return np.mean(dphi)


def compute_null(Y, sgn, n_null=2000, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in tqdm(range(n_null), desc="null", leave=False):
        s_shuf = np.array(sgn)
        rng.shuffle(s_shuf)
        vals.append(global_phase_memory_from_Y(Y, s_shuf))
    return np.array(vals)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(path):
    print("[info] loading frb catalog…")
    RA, Dec = load_catalog(path)

    print("[info] converting to galactic xyz…")
    Xgal = radec_to_galactic_xyz(RA, Dec)

    l_gal, b_gal = xyz_to_galactic_lb(Xgal)

    axis = galactic_lb_to_xyz(159.8, -0.5)

    print("[info] computing remnant signs…")
    sgn_full = remnant_sign(Xgal, axis)

    print("[info] precomputing spherical harmonics…")
    Y_full = compute_Ylm_pure(Xgal, lmax=10)

    # masks: no cut, |b|>=20,30,40
    cuts = [0.0, 20.0, 30.0, 40.0]

    print("[info] running latitude-mask robustness (85H)…")
    results = []

    for cut in cuts:
        mask = np.abs(b_gal) >= cut
        Xc = Xgal[mask]
        Yc = Y_full[mask]
        sgn = sgn_full[mask]

        print(f"[info] |b|>={cut:.1f}°   N={len(Xc)}")

        Z_real = global_phase_memory_from_Y(Yc, sgn)
        null_vals = compute_null(Yc, sgn, n_null=2000, seed=42)
        null_mean = float(np.mean(null_vals))
        p = (1 + np.sum(null_vals >= Z_real)) / (len(null_vals) + 1)

        print(f"    Z_real={Z_real:.6f}   null_mean={null_mean:.6f}   p={p:.6f}")

        results.append({
            "cut": cut,
            "N": int(len(Xc)),
            "Z_real": float(Z_real),
            "null_mean": null_mean,
            "p": float(p),
        })

    print("================================================")
    print(" GALACTIC LATITUDE MASK ROBUSTNESS (TEST 85H)   ")
    print("================================================")
    for r in results:
        print(f"|b|>={r['cut']:4.1f}°   N={r['N']:4d}   "
              f"Z={r['Z_real']:.6f}   "
              f"null_mean={r['null_mean']:.6f}   "
              f"p={r['p']:.6f}")
    print("================================================")
    print("interpretation:")
    print("  if Z stays above null as |b| cuts grow, the")
    print("  phase-memory signal is not tied to the galactic")
    print("  plane or obvious survey footprints.")
    print("================================================")
    print("test 85H complete.")
    print("================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_phase_memory_latmask_test85H.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
