#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time shell-weighted rotational memory test (test 84)
----------------------------------------------------------------

goal:
    extend the remnant-time suite by jointly probing:
        - the shell structure seen in test 71
        - the rotational orientation field seen in test 83

    the question:
        do the 25°/40° unified-axis shells carry a stronger
        rotational-memory difference between remnant-time
        hemispheres than expected under:
            (i) footprint-preserving remnant-sign shuffles, and
            (ii) fully isotropic sky redraws?

    this test therefore evaluates:
        A_shell(k) = | <z>_{R>0,shell} - <z>_{R<0,shell} |
    for k ∈ {5,10,20,40,80}, where z = exp(2i psi) is the
    spin-2 orientation extracted from local neighbourhoods.

design:
    - unified axis fixed at (l,b)=(159.8,-0.5)
    - no axis scan
    - shells: 17.5–32.5° and 32.5–47.5°
    - two null models:
         1. sign-shuffle null  (preserves footprint)
         2. isotropic-sky null (redraws sky uniformly)
    - designed to be compatible with follow-up robustness tests:
         84A (galactic mask), 84B (supergalactic mask),
         84C (ASKAP split), 84_jackknife20
"""

import numpy as np
import csv
import sys
import math
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


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
# remnant field geometry
# ============================================================

def angle_from_axis(X, axis_vec):
    axis = axis_vec / (np.linalg.norm(axis_vec) + 1e-15)
    dots = np.clip(X @ axis, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def remnant_sign(X, axis_vec):
    axis = axis_vec / (np.linalg.norm(axis_vec) + 1e-15)
    R = X @ axis
    s = np.ones(len(X), dtype=int)
    s[R < 0] = -1
    return s


def shell_weights(theta_deg,
                  shell1=(17.5, 32.5),
                  shell2=(32.5, 47.5)):
    in_s1 = (theta_deg >= shell1[0]) & (theta_deg < shell1[1])
    in_s2 = (theta_deg >= shell2[0]) & (theta_deg < shell2[1])
    w = np.zeros_like(theta_deg, dtype=float)
    w[in_s1 | in_s2] = 1.0
    return w


# ============================================================
# tangent basis + spin-2 orientation extraction
# ============================================================

def tangent_basis(x):
    z = x / (np.linalg.norm(x) + 1e-15)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, z)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = tmp - np.dot(tmp, z)*z
    e1 /= (np.linalg.norm(e1) + 1e-15)
    e2 = np.cross(z, e1)
    e2 /= (np.linalg.norm(e2) + 1e-15)
    return e1, e2


def orientation_spin2(X, k, anis_threshold=0.1):
    N = len(X)
    if N == 0:
        return np.zeros(0, dtype=complex), np.zeros(0, dtype=bool)

    nbr = NearestNeighbors(n_neighbors=min(k+1, N),
                           algorithm="ball_tree").fit(X)
    dist, idx = nbr.kneighbors(X)

    psi = np.zeros(N)
    valid = np.zeros(N, dtype=bool)

    for i in range(N):
        e1, e2 = tangent_basis(X[i])
        nbrs = idx[i, 1:]
        if len(nbrs) < 3:
            continue

        U = []
        for j in nbrs:
            dx = X[j] - X[i]
            U.append([np.dot(dx, e1), np.dot(dx, e2)])
        U = np.array(U)
        if U.shape[0] < 3:
            continue

        C = np.cov(U.T)
        C = 0.5*(C + C.T)
        w, v = np.linalg.eigh(C)

        im = np.argmax(w)
        lam1 = w[im]
        lam2 = w[1-im]
        if lam1 + lam2 <= 0:
            continue

        anis = (lam1 - lam2)/(lam1 + lam2)
        if anis < anis_threshold:
            continue

        vmain = v[:, im]
        psi[i] = math.atan2(vmain[1], vmain[0])
        valid[i] = True

    z = np.exp(2j * psi)
    z[~valid] = 0.0
    return z, valid


# ============================================================
# shell-weighted memory statistic
# ============================================================

def shell_weighted_memory_stat(z, valid, sgn, w_shell):
    mask = valid & (w_shell > 0)

    m_pos = mask & (sgn > 0)
    m_neg = mask & (sgn < 0)

    def wmean(mask):
        if not np.any(mask):
            return 0.0 + 0.0j
        w = w_shell[mask]
        zsel = z[mask]
        ws = np.sum(w)
        if ws == 0:
            return 0.0 + 0.0j
        return np.sum(w * zsel) / ws

    Spos = wmean(m_pos)
    Sneg = wmean(m_neg)
    return abs(Spos - Sneg)


# ============================================================
# main null + real calculation
# ============================================================

def compute_84_results(X, sgn, theta_deg,
                       K=(5,10,20,40,80),
                       n_null=1000,
                       seed=42):
    rng = np.random.RandomState(seed)
    N = len(X)
    w_shell = shell_weights(theta_deg)

    axis_unified = galactic_lb_to_xyz(159.8, -0.5)

    results = []

    for k in K:
        print(f"[info] scale k = {k}")

        z, valid = orientation_spin2(X, k)
        if np.sum(valid & (w_shell > 0)) < 10:
            print(f"[warn] insufficient valid orientations at k={k}")
            results.append({
                "k": k,
                "A_real": np.nan,
                "shuffle_mean": np.nan,
                "shuffle_p": np.nan,
                "iso_mean": np.nan,
                "iso_p": np.nan,
            })
            continue

        # -------------------------
        # real statistic
        # -------------------------
        A_real = shell_weighted_memory_stat(z, valid, sgn, w_shell)

        # -------------------------
        # null 1: sign-shuffle null
        # -------------------------
        shuffle_vals = []
        for _ in tqdm(range(n_null), desc=f"shuffle null k={k}", leave=False):
            s_shuf = np.array(sgn, copy=True)
            rng.shuffle(s_shuf)
            shuffle_vals.append(
                shell_weighted_memory_stat(z, valid, s_shuf, w_shell)
            )
        shuffle_vals = np.array(shuffle_vals)
        shuffle_mean = float(np.mean(shuffle_vals))
        shuffle_p = (1 + np.sum(shuffle_vals >= A_real)) / (len(shuffle_vals) + 1)

        # -------------------------
        # null 2: isotropic sky null
        # -------------------------
        iso_vals = []
        for _ in tqdm(range(n_null), desc=f"isotropic null k={k}", leave=False):
            # isotropic draw
            u = rng.uniform(-1, 1, size=N)
            phi = rng.uniform(0, 2*np.pi, size=N)
            sin_t = np.sqrt(1 - u*u)
            Xiso = np.vstack([sin_t*np.cos(phi),
                              sin_t*np.sin(phi),
                              u]).T
            Xiso /= (np.linalg.norm(Xiso, axis=1, keepdims=True) + 1e-15)

            # recompute shell weights
            theta_iso = angle_from_axis(Xiso, axis_unified)
            w_iso = shell_weights(theta_iso)

            # reuse orientation algorithm
            zI, validI = orientation_spin2(Xiso, k)

            # random assignment of signs (same +1/-1 count)
            s_iso = np.array(sgn, copy=True)
            rng.shuffle(s_iso)

            Aiso = shell_weighted_memory_stat(zI, validI, s_iso, w_iso)
            iso_vals.append(Aiso)

        iso_vals = np.array(iso_vals)
        iso_mean = float(np.mean(iso_vals))
        iso_p = (1 + np.sum(iso_vals >= A_real)) / (len(iso_vals) + 1)

        print(f"  A_real = {A_real:.6f}")
        print(f"  shuffle null: mean={shuffle_mean:.6f}, p={shuffle_p:.6f}")
        print(f"  isotropic:    mean={iso_mean:.6f},   p={iso_p:.6f}\n")

        results.append({
            "k": k,
            "A_real": float(A_real),
            "shuffle_mean": shuffle_mean,
            "shuffle_p": float(shuffle_p),
            "iso_mean": iso_mean,
            "iso_p": float(iso_p),
        })

    return results


# ============================================================
# main driver
# ============================================================

def main(path):
    print("[info] loading frb catalog...")
    RA, Dec = load_catalog(path)
    N = len(RA)
    print(f"[info] N_FRB = {N}")

    print("[info] converting to galactic xyz...")
    Xgal = radec_to_galactic_xyz(RA, Dec)

    axis = galactic_lb_to_xyz(159.8, -0.5)

    print("[info] computing axis angles + remnant signs...")
    theta = angle_from_axis(Xgal, axis)
    sgn = remnant_sign(Xgal, axis)

    print("[info] running test 84 core...")
    results = compute_84_results(Xgal, sgn, theta,
                                 K=(5,10,20,40,80),
                                 n_null=1000,
                                 seed=42)

    print("================================================")
    print(" FRB REMNANT-TIME SHELL-WEIGHTED ROTATIONAL MEMORY (TEST 84)")
    print("================================================")
    for r in results:
        print(f"k={r['k']:3d}   A_real={r['A_real']:.6f}   "
              f"[shuffle mean={r['shuffle_mean']:.6f}, p={r['shuffle_p']:.6f}]   "
              f"[isotropic mean={r['iso_mean']:.6f}, p={r['iso_p']:.6f}]")
    print("================================================")
    print("interpretation:")
    print("  - agreement across shuffle + isotropic nulls indicates")
    print("      shell-weighted rotational memory is consistent with")
    print("      random remnant labels on both the real footprint and")
    print("      isotropic skies.")
    print("  - significant deviation (especially at k>=20) would indicate")
    print("      joint shell + spin-2 remnant-time coherence.")
    print("================================================")
    print("test 84 complete.")
    print("================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_shell_weighted_rotational_memory_test84.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
