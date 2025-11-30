#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB remnant-time shell asymmetry test (71) — 20-region jackknife
----------------------------------------------------------------

Jackknife scheme:
  - work in galactic coords
  - define 20 longitude slices in l (each 18 deg wide)
  - for each slice r = 0..19:
        drop that slice
        recompute S_total and its p-value
  - report how stable the signal is to removing any one region
"""

import numpy as np
import csv
import sys
from tqdm import tqdm

# ============================================================
# catalog + coords (copied from main 71) 
# ============================================================

def detect_columns(fieldnames):
    low = [c.lower() for c in fieldnames]

    def find(*candidates):
        for c in candidates:
            if c.lower() in low:
                return fieldnames[low.index(c.lower())]
        return None

    ra_key  = find("ra_deg","ra","raj2000","ra_deg_","ra (deg)")
    dec_key = find("dec_deg","dec","dej2000","dec_deg","dec (deg)")

    if ra_key is None or dec_key is None:
        raise KeyError("could not detect ra/dec column names")

    return ra_key, dec_key


def load_catalog(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        ra_key, dec_key = detect_columns(fields)

        RA, Dec = [], []
        for row in reader:
            RA.append(float(row[ra_key]))
            Dec.append(float(row[dec_key]))

    return np.array(RA), np.array(Dec)


def radec_to_equatorial_xyz(RA, Dec):
    RA  = np.radians(RA)
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
    norms = np.linalg.norm(Xgal, axis=1, keepdims=True) + 1e-15
    return Xgal / norms


def galactic_lb_to_xyz(l_deg, b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    x = np.cos(b) * np.cos(l)
    y = np.cos(b) * np.sin(l)
    z = np.sin(b)
    v = np.array([x, y, z], dtype=float)
    return v / (np.linalg.norm(v) + 1e-15)


def xyz_to_lb(X):
    """
    convert galactic xyz to (l,b) in degrees.
    """
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    l = np.degrees(np.arctan2(y, x)) % 360.0
    b = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return l, b

# ============================================================
# shell asymmetry pieces (as in 71 main)
# ============================================================

def angle_from_axis(X, axis_vec):
    axis_vec = axis_vec / (np.linalg.norm(axis_vec) + 1e-15)
    dots = X @ axis_vec
    np.clip(dots, -1.0, 1.0, out=dots)
    return np.degrees(np.arccos(dots))


def remnant_sign(X, axis_vec):
    axis_vec = axis_vec / (np.linalg.norm(axis_vec) + 1e-15)
    R = X @ axis_vec
    sign = np.ones_like(R, dtype=int)
    sign[R < 0] = -1
    return sign


def shell_counts(theta_deg, sign, shell_min, shell_max):
    mask_shell = (theta_deg >= shell_min) & (theta_deg < shell_max)
    in_shell = sign[mask_shell]
    n_plus = int(np.sum(in_shell > 0))
    n_minus = int(np.sum(in_shell < 0))
    return n_plus, n_minus


def total_shell_asymmetry(theta_deg, sign,
                          shell1=(17.5, 32.5),
                          shell2=(32.5, 47.5)):
    s1_min, s1_max = shell1
    s2_min, s2_max = shell2

    n1_plus, n1_minus = shell_counts(theta_deg, sign, s1_min, s1_max)
    n2_plus, n2_minus = shell_counts(theta_deg, sign, s2_min, s2_max)

    delta1 = n1_plus - n1_minus
    delta2 = n2_plus - n2_minus
    delta_total = delta1 + delta2

    S1 = abs(delta1)
    S2 = abs(delta2)
    S_total = abs(delta_total)

    stats = {
        "shell1_n_plus": n1_plus,
        "shell1_n_minus": n1_minus,
        "shell1_delta": delta1,
        "shell1_S": S1,
        "shell2_n_plus": n2_plus,
        "shell2_n_minus": n2_minus,
        "shell2_delta": delta2,
        "shell2_S": S2,
        "delta_total": delta_total,
        "S_total": S_total,
    }
    return stats, S_total


def random_isotropic(N):
    u = np.random.uniform(-1.0, 1.0, size=N)
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=N)
    sin_theta = np.sqrt(1.0 - u * u)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = u
    X = np.vstack([x, y, z]).T
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-15
    return X / norms

# ============================================================
# core helper: run test 71 on an arbitrary subset Xsub
# ============================================================

def run_shell_test(Xsub, axis_unified, Nmc=2000):
    N = len(Xsub)
    theta_deg = angle_from_axis(Xsub, axis_unified)
    sign = remnant_sign(Xsub, axis_unified)
    stats_real, S_real = total_shell_asymmetry(theta_deg, sign)

    S_null = []
    for _ in range(Nmc):
        X_mc = random_isotropic(N)
        theta_mc = angle_from_axis(X_mc, axis_unified)
        sign_mc = remnant_sign(X_mc, axis_unified)
        _, S_mc = total_shell_asymmetry(theta_mc, sign_mc)
        S_null.append(S_mc)
    S_null = np.array(S_null, dtype=float)

    mu = float(np.mean(S_null))
    sd = float(np.std(S_null))
    p  = (1.0 + np.sum(S_null >= S_real)) / (len(S_null) + 1.0)

    return stats_real, S_real, mu, sd, p

# ============================================================
# main: full sample + 20-region jackknife
# ============================================================

def main(path):
    print("[info] loading frb catalog...")
    RA, Dec = load_catalog(path)
    print(f"[info] N_FRB = {len(RA)}")

    print("[info] converting to galactic xyz...")
    Xgal = radec_to_galactic_xyz(RA, Dec)
    l, b = xyz_to_lb(Xgal)

    # unified axis
    axis_unified = galactic_lb_to_xyz(159.8, -0.5)

    print("[info] running full-sample test 71...")
    stats_full, S_full, mu_full, sd_full, p_full = run_shell_test(Xgal, axis_unified)

    print("================================================")
    print(" test 71 full-sample result")
    print("================================================")
    print(f"S_total (full)    = {S_full}")
    print(f"null mean S_total = {mu_full:.3f}")
    print(f"null std S_total  = {sd_full:.3f}")
    print(f"p-value (full)    = {p_full:.6f}")
    print("================================================")

    # jackknife 20 longitude slices
    print("[info] running 20-region longitude jackknife...")
    width = 360.0 / 20.0
    region_idx = (l / width).astype(int)
    region_idx = np.clip(region_idx, 0, 19)

    results = []

    for r in range(20):
        mask_keep = (region_idx != r)
        X_jk = Xgal[mask_keep]
        Nj = len(X_jk)
        if Nj < 50:
            print(f"[warn] region {r}: too few FRBs after removal (Nj={Nj}), skipping.")
            results.append((r, Nj, np.nan, np.nan))
            continue

        print(f"[info] jackknife region {r}: removing slice {r}, remaining N={Nj}")
        _, S_jk, mu_jk, sd_jk, p_jk = run_shell_test(X_jk, axis_unified)
        results.append((r, Nj, S_jk, p_jk))

    print("================================================")
    print(" 20-region jackknife summary for test 71")
    print(" region  N_keep   S_total_jk   p_jk")
    for r, Nj, S_jk, p_jk in results:
        print(f"  {r:2d}    {Nj:4d}    {S_jk:9.3f}   {p_jk:7.5f}")
    print("================================================")
    print("interpretation:")
    print(" if all jackknife p-values stay small, the shell asymmetry")
    print(" is not dominated by any single longitude slice.")
    print("================================================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_shell_asymmetry_test71_jackknife20.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
