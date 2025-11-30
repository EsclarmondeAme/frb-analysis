#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test 101B — Energy Conservation Drift Test (Corrected)
------------------------------------------------------

this version is fully consistent with the logic of remnant-time tests
(85F, 85D, 85G, 85I, 85N).

Key change:
    - remnant-time sign (rt_sign) is computed internally from geometry:
        sign_i = +1 if X_i · axis > 0 else -1
      exactly like all phase-memory tests.

Scientific question:
    Does FRB energy (fluence) show dependence on:
        a) unified-axis alignment (anisotropic vacuum)
        b) remnant-time hemisphere (manifold)
        c) observation time (time-translation symmetry broken)

Regression:
    E = a*cos(theta_u) + b*rt_sign + c*t_norm + d

Null:
    2000 Monte Carlo runs, shuffling:
        - sky angles
        - remnant signs
        - times
        - energies

Interpretation:
    low p(a) → directional vacuum anisotropy
    low p(b) → manifold energy imbalance
    low p(c) → time-dependent geometry
"""

import sys
import csv
import math
import random
import statistics
from time import time
import numpy as np


# ------------------------------------------------------------
# utilities
# ------------------------------------------------------------

def fit_regression(E, x1, x2, x3):
    """
    Fit linear regression:
        E = a*x1 + b*x2 + c*x3 + d
    returns (a,b,c,d)
    """
    n = len(E)
    Sx1 = sum(x1)
    Sx2 = sum(x2)
    Sx3 = sum(x3)

    Sx1x1 = sum(v*v for v in x1)
    Sx2x2 = sum(v*v for v in x2)
    Sx3x3 = sum(v*v for v in x3)

    Sx1x2 = sum(x1[i]*x2[i] for i in range(n))
    Sx1x3 = sum(x1[i]*x3[i] for i in range(n))
    Sx2x3 = sum(x2[i]*x3[i] for i in range(n))

    SE    = sum(E)
    Sx1E  = sum(x1[i]*E[i] for i in range(n))
    Sx2E  = sum(x2[i]*E[i] for i in range(n))
    Sx3E  = sum(x3[i]*E[i] for i in range(n))

    M = np.array([
        [Sx1x1, Sx1x2, Sx1x3, Sx1],
        [Sx1x2, Sx2x2, Sx2x3, Sx2],
        [Sx1x3, Sx2x3, Sx3x3, Sx3],
        [Sx1,   Sx2,   Sx3,   n  ]
    ], dtype=float)

    Y = np.array([Sx1E, Sx2E, Sx3E, SE], dtype=float)

    try:
        sol = np.linalg.solve(M, Y)
        return sol[0], sol[1], sol[2], sol[3]
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def load_catalog(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        R = csv.DictReader(f)
        for r in R:
            try:
                flu = float(r["fluence"])
                theta_u = float(r["theta_unified"])
                mjd = float(r["mjd"])
                ra = float(r["ra"])
                dec = float(r["dec"])
            except:
                continue
            rows.append((flu, theta_u, mjd, ra, dec))
    return rows


# ------------------------------------------------------------
# geometry utilities
# ------------------------------------------------------------

def radec_to_xyz(ra_deg, dec_deg):
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    x = math.cos(dec)*math.cos(ra)
    y = math.cos(dec)*math.sin(ra)
    z = math.sin(dec)
    return np.array([x,y,z])


def galactic_lb_to_xyz(l_deg, b_deg):
    l = math.radians(l_deg)
    b = math.radians(b_deg)
    v = np.array([
        math.cos(b)*math.cos(l),
        math.cos(b)*math.sin(l),
        math.sin(b)
    ])
    return v / np.linalg.norm(v)


def compute_rt_sign(rows, axis_vec):
    sgn = []
    for _,_,_,ra,dec in rows:
        v = radec_to_xyz(ra, dec)
        s = 1 if np.dot(v, axis_vec) > 0 else -1
        sgn.append(s)
    return sgn


# ------------------------------------------------------------
# regression components
# ------------------------------------------------------------

def build_predictors(rows, rt_sign):
    E  = [r[0] for r in rows]                         # fluence
    x1 = [math.cos(math.radians(r[1])) for r in rows] # axis alignment
    x2 = rt_sign                                      # remnant-time sign

    mjd_vals = [r[2] for r in rows]
    mn, mx = min(mjd_vals), max(mjd_vals)
    x3 = [2*(t - mn)/(mx - mn) - 1 for t in mjd_vals] # normalized time

    return E, x1, x2, x3


def compute_coeffs(E,x1,x2,x3):
    return fit_regression(E,x1,x2,x3)


# ------------------------------------------------------------
# null
# ------------------------------------------------------------

def run_null(E, x1, x2, x3, n_mc):
    A, B, C = [], [], []
    N = len(E)

    E0  = list(E)
    x10 = list(x1)
    x20 = list(x2)
    x30 = list(x3)

    for _ in range(n_mc):
        random.shuffle(E0)
        random.shuffle(x10)
        random.shuffle(x20)
        random.shuffle(x30)

        a,b,c,d = fit_regression(E0, x10, x20, x30)
        A.append(abs(a))
        B.append(abs(b))
        C.append(abs(c))

    return A, B, C


def p_value(real, null):
    return sum(1 for v in null if v >= abs(real)) / len(null)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("usage: python frb_energy_conservation_drift_test101B.py frbs_unified.csv [N_null]")
        sys.exit(1)

    path = sys.argv[1]
    N_null = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    print("===================================================")
    print(" Test 101B — FRB Energy Conservation Drift Test")
    print("===================================================")
    print(f"[info] loading: {path}")

    rows = load_catalog(path)
    N = len(rows)
    print(f"[info] N_FRB = {N}")

    # unified axis (galactic)
    axis = galactic_lb_to_xyz(159.8, -0.5)

    # compute remnant-time sign internally
    print("[info] computing remnant-time signs internally...")
    rt = compute_rt_sign(rows, axis)

    # build predictors
    E, x1, x2, x3 = build_predictors(rows, rt)

    print("[info] computing REAL regression...")
    a_real, b_real, c_real, d_real = compute_coeffs(E,x1,x2,x3)
    print(f"  a (axis)        = {a_real:.6f}")
    print(f"  b (manifold)    = {b_real:.6f}")
    print(f"  c (time drift)  = {c_real:.6f}")
    print(f"  d (intercept)   = {d_real:.6f}")

    print("[info] running null ensemble...")
    t0 = time()
    A, B, C = run_null(E,x1,x2,x3,N_null)
    dt = time() - t0
    print(f"[info] null completed in {dt:.2f} s")

    # statistics
    meanA, stdA = statistics.mean(A), statistics.stdev(A)
    meanB, stdB = statistics.mean(B), statistics.stdev(B)
    meanC, stdC = statistics.mean(C), statistics.stdev(C)

    pA = p_value(a_real, A)
    pB = p_value(b_real, B)
    pC = p_value(c_real, C)

    print("---------------------------------------------------")
    print(" RESULTS")
    print("---------------------------------------------------")

    print("axis term (a):")
    print(f"  real = {a_real:.6f}")
    print(f"  null_mean = {meanA:.6f}, null_std = {stdA:.6f}")
    print(f"  p-value = {pA:.6f}")
    print("---------------------------------------------------")

    print("manifold term (b):")
    print(f"  real = {b_real:.6f}")
    print(f"  null_mean = {meanB:.6f}, null_std = {stdB:.6f}")
    print(f"  p-value = {pB:.6f}")
    print("---------------------------------------------------")

    print("time-drift term (c):")
    print(f"  real = {c_real:.6f}")
    print(f"  null_mean = {meanC:.6f}, null_std = {stdC:.6f}")
    print(f"  p-value = {pC:.6f}")
    print("---------------------------------------------------")

    print("interpretation:")
    print("  low p(a) → directional vacuum anisotropy")
    print("  low p(b) → manifold hemispheric energy imbalance")
    print("  low p(c) → time-dependent geometry (broken time-translation)")
    print("  if any p < 0.05 → classical energy conservation violated")
    print("===================================================")
    print(" test 101B complete.")
    print("===================================================")


if __name__ == "__main__":
    main()
