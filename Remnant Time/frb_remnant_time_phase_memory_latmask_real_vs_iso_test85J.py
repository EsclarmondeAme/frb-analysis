#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time phase-memory — real vs isotropic geometry for |b| masks (test 85J)
------------------------------------------------------------------------------------

goal:
    after 85H (real |b| masks) and 85I (single isotropic sky),
    we now directly compare:

        Z_real(|b| >= cut)

    against a distribution of:

        Z_iso(|b| >= cut)

    where each isotropic sky has:
        - the same N (after mask)
        - random positions on the sphere
        - the same unified axis definition
        - hemisphere sign from axis

    this isolates geometry:
        if Z_real looks typical within the isotropic distribution,
        then the phase-memory is a geometric artifact of axis+hemisphere.
        if Z_real sits in the extreme tail, there is extra structure
        in the real FRB sky beyond pure geometry.

design:
    - cuts: |b| >= 0, 20, 30, 40
    - for each cut:
        - compute Z_real on real FRBs
        - run n_iso isotropic skies (default 1000)
        - compute Z_iso for each
        - compute geometric p-value:
              p_geom = (1 + # {Z_iso >= Z_real}) / (n_iso + 1)
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


def xyz_to_galactic_lb(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    b = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    l = np.degrees(np.arctan2(y, x))
    l = np.mod(l, 360.0)
    return l, b


def galactic_lb_to_xyz(l_deg, b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    v = np.array([
        np.cos(b) * np.cos(l),
        np.cos(b) * np.sin(l),
        np.sin(b)
    ])
    return v / (np.linalg.norm(v) + 1e-15)


# ------------------------------------------------------------
# remnant sign
# ------------------------------------------------------------

def remnant_sign(X, axis_vec):
    axis = axis_vec / (np.linalg.norm(axis_vec) + 1e-15)
    return np.where(X @ axis > 0, 1, -1)


# ------------------------------------------------------------
# random isotropic positions
# ------------------------------------------------------------

def random_isotropic_xyz(N, rng):
    u = rng.uniform(-1.0, 1.0, size=N)
    phi = rng.uniform(0.0, 2*np.pi, size=N)
    x = np.sqrt(1 - u*u) * np.cos(phi)
    y = np.sqrt(1 - u*u) * np.sin(phi)
    z = u
    return np.vstack([x, y, z]).T


# ------------------------------------------------------------
# spherical harmonics (pure numpy)
# ------------------------------------------------------------

def legendre_P(l, m, x):
    m_abs = abs(m)
    Pmm = np.ones_like(x)
    if m_abs > 0:
        somx2 = np.sqrt(1 - x*x)
        fact = 1.0
        for _ in range(m_abs):
            Pmm *= -fact * somx2
            fact += 2.0
    if l == m_abs:
        return Pmm
    Pm1m = x * (2*m_abs + 1) * Pmm
    if l == m_abs + 1:
        return Pm1m
    Pll = None
    for ll in range(m_abs + 2, l + 1):
        Pll = ((2*ll - 1)*x*Pm1m - (ll + m_abs - 1)*Pmm) / (ll - m_abs)
        Pmm, Pm1m = Pm1m, Pll
    return Pll


def Ylm_pure(l, m, theta, phi):
    x = np.cos(theta)
    Plm = legendre_P(l, m, x)
    K = math.sqrt((2*l + 1)/(4*math.pi) *
                  math.factorial(l - abs(m)) /
                  math.factorial(l + abs(m)))
    if m > 0:
        return math.sqrt(2)*K*Plm*np.cos(m*phi)
    elif m < 0:
        return math.sqrt(2)*K*Plm*np.sin(abs(m)*phi)
    else:
        return K*Plm


def compute_Ylm_matrix(X, lmax=8):
    theta = np.arccos(np.clip(X[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2*np.pi)
    Y = np.zeros((len(X), (lmax+1)*(lmax+1)))
    idx = 0
    for l in range(lmax+1):
        for m in range(-l, l+1):
            Y[:, idx] = Ylm_pure(l, m, theta, phi)
            idx += 1
    return Y


# ------------------------------------------------------------
# phase-memory
# ------------------------------------------------------------

def phase_memory(Y, sgn):
    pos = (sgn > 0)
    neg = (sgn < 0)
    if np.sum(pos) < 2 or np.sum(neg) < 2:
        return np.nan
    A_pos = np.sum(Y[pos], axis=0)
    A_neg = np.sum(Y[neg], axis=0)
    dphi = np.abs(np.arctan2(A_pos - A_neg, np.ones_like(A_pos)))
    return np.mean(dphi)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(path, n_iso=1000, seed=123):
    print("[info] loading real catalog…")
    RA, Dec = load_catalog(path)

    print("[info] converting to galactic xyz…")
    Xgal = radec_to_galactic_xyz(RA, Dec)
    l_gal, b_gal = xyz_to_galactic_lb(Xgal)

    axis = galactic_lb_to_xyz(159.8, -0.5)

    # define |b| cuts
    cuts = [0.0, 20.0, 30.0, 40.0]

    results = []
    rng = np.random.RandomState(seed)

    for cut in cuts:
        print(f"[info] --- |b|>={cut:.1f}° ---")
        mask = np.abs(b_gal) >= cut
        Xc = Xgal[mask]
        N = len(Xc)
        if N < 10:
            print(f"[warn] too few FRBs after mask: N={N}, skipping")
            continue

        print(f"[info] real-sky N={N}")
        sgn_real = remnant_sign(Xc, axis)
        Y_real = compute_Ylm_matrix(Xc, lmax=8)
        Z_real = phase_memory(Y_real, sgn_real)
        print(f"[info] real-sky Z={Z_real:.6f}")

        # isotropic ensemble
        Z_iso_list = []
        for _ in tqdm(range(n_iso), desc=f"iso |b|>={cut:.1f}", leave=False):
            X_iso = random_isotropic_xyz(N, rng)
            sgn_iso = remnant_sign(X_iso, axis)
            Y_iso = compute_Ylm_matrix(X_iso, lmax=8)
            Z_iso = phase_memory(Y_iso, sgn_iso)
            Z_iso_list.append(Z_iso)

        Z_iso_arr = np.array(Z_iso_list)
        mean_iso = float(np.mean(Z_iso_arr))
        std_iso = float(np.std(Z_iso_arr))

        p_geom = (1 + np.sum(Z_iso_arr >= Z_real)) / (len(Z_iso_arr) + 1)

        print(f"[info] iso mean Z={mean_iso:.6f}, std={std_iso:.6f}, "
              f"geom p={p_geom:.6f}")

        results.append({
            "cut": cut,
            "N": N,
            "Z_real": float(Z_real),
            "iso_mean": mean_iso,
            "iso_std": std_iso,
            "p_geom": float(p_geom),
        })

    print("============================================================")
    print(" REAL VS ISOTROPIC GEOMETRY FOR |b| MASKS (TEST 85J)       ")
    print("============================================================")
    for r in results:
        print(f"|b|>={r['cut']:4.1f}°  N={r['N']:4d}  "
              f"Z_real={r['Z_real']:.6f}  "
              f"iso_mean={r['iso_mean']:.6f}  "
              f"iso_std={r['iso_std']:.6f}  "
              f"p_geom={r['p_geom']:.6f}")
    print("============================================================")
    print("interpretation:")
    print("  if p_geom is large, Z_real is typical of isotropic skies,")
    print("  meaning the phase-memory is dominated by geometry.")
    print("  if p_geom is tiny, the real sky has extra structure beyond")
    print("  pure isotropic geometry at that |b| cut.")
    print("============================================================")
    print("test 85J complete.")
    print("============================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_phase_memory_latmask_real_vs_iso_test85J.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
