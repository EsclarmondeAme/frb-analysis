#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test 106B — 3D Harmonic–Radial Mode Occupation Spectrum
--------------------------------------------------------

Goal:
    Full modal decomposition of FRB field into spherical harmonic Y_lm
    AND radial Bessel modes j_l(k_n r), giving coefficients:

        a_lmn = sum_i F_i * j_l(k_n r_i) * Y_lm*(theta_i,phi_i)

    And testing for anomalous mode occupation relative to isotropic null.

Radial basis:
    k_n = n * pi / Rmax,   n = 1..K
    j_l is spherical Bessel function.

Interpretation:
    p < 0.05   → over-occupied mode (angular+radial resonance)
    p < 0.001  → strong resonant 3D eigenmode candidate

Usage:
    python frb_harmonic_radial_mode_occupation_test106B.py frbs_unified.csv [lmax] [K] [N_null]
"""

import sys
import csv
import math
import random
import numpy as np
from time import time
from scipy.special import sph_harm, spherical_jn


# ------------------------------------------------------------
# load catalog
# ------------------------------------------------------------
def load_catalog(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        R = csv.DictReader(f)
        for r in R:
            try:
                ra = float(r["ra"])
                dec = float(r["dec"])
                flu = float(r["fluence"])
                z  = float(r["z_est"]) if r["z_est"] != "" else 0.2
            except:
                continue
            rows.append((ra, dec, flu, z))
    return rows


# ------------------------------------------------------------
# cosmology: convert z_est to comoving distance r (Mpc)
# ------------------------------------------------------------
c = 299792.458
def E(z):
    return 1.0 / np.sqrt(0.3*(1+z)**3 + 0.7)

def comoving_distance(z):
    if z <= 0:
        return 0.0
    N = 200
    zs = np.linspace(0, z, N)
    dz = z / (N-1)
    integral = np.sum(E(zs)) * dz
    return (c/70.0) * integral   # Mpc


# ------------------------------------------------------------
# RA,DEC → theta,phi
# ------------------------------------------------------------
def radec_to_thetaphi(ra_deg, dec_deg):
    ra  = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    theta = math.radians(90.0 - dec)
    phi   = ra
    return theta, phi


# ------------------------------------------------------------
# compute a_lmn
# ------------------------------------------------------------
def compute_almn(rows, lmax, K, k_values):
    # unpack geometry first
    N = len(rows)
    thetas = np.zeros(N)
    phis   = np.zeros(N)
    F      = np.zeros(N)
    rvals  = np.zeros(N)
    for i,(ra,dec,flu,z) in enumerate(rows):
        th,ph = radec_to_thetaphi(ra,dec)
        thetas[i] = th
        phis[i]   = ph
        F[i]      = flu
        rvals[i]  = comoving_distance(z)

    almn = {}

    for l in range(lmax+1):
        # precompute j_l(k_n r_i) for all n
        jL = np.zeros((K, N))
        for n in range(K):
            k = k_values[n]
            jL[n,:] = spherical_jn(l, k*rvals)

        for m in range(-l, l+1):
            Y = sph_harm(m, l, phis, thetas)
            Yc = np.conjugate(Y)

            for n in range(K):
                a = np.sum(F * jL[n,:] * Yc)
                almn[(l,m,n)] = a

    return almn


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_harmonic_radial_mode_occupation_test106B.py frbs_unified.csv [lmax] [K] [N_null]")
        sys.exit(1)

    path = sys.argv[1]
    lmax = int(sys.argv[2]) if len(sys.argv)>2 else 6
    K    = int(sys.argv[3]) if len(sys.argv)>3 else 5
    N_null = int(sys.argv[4]) if len(sys.argv)>4 else 1500

    print("==========================================================")
    print(" Test 106B — 3D Harmonic–Radial Mode Spectrum")
    print("==========================================================")
    print("[info] loading:", path)
    rows = load_catalog(path)
    N = len(rows)
    print("[info] N_FRB =", N)
    print("[info] lmax   =", lmax)
    print("[info] K_radial =", K)
    print("[info] N_null =", N_null)

    # compute r_max
    z_vals = [r[3] for r in rows]
    r_vals = [comoving_distance(z) for z in z_vals]
    Rmax = max(r_vals) * 1.1  # 10% padding
    print("[info] Rmax (Mpc) =", Rmax)

    # radial k-values
    k_vals = [(n+1)*math.pi/Rmax for n in range(K)]
    print("[info] k-values:", k_vals)

    # REAL a_lmn
    print("[info] computing REAL a_lmn...")
    almn_real = compute_almn(rows, lmax, K, k_vals)
    nlmn_real = {k: abs(v)**2 for k,v in almn_real.items()}

    # NULL
    print("[info] running null ensemble...")
    t0 = time()
    fluences = [r[2] for r in rows]
    null_nlmn = { (l,m,n): [] for l in range(lmax+1)
                             for m in range(-l,l+1)
                             for n in range(K) }

    for _ in range(N_null):
        random.shuffle(fluences)
        rows_sh = [(rows[i][0], rows[i][1], fluences[i], rows[i][3]) for i in range(N)]

        a_null = compute_almn(rows_sh, lmax, K, k_vals)
        for key in null_nlmn:
            null_nlmn[key].append(abs(a_null[key])**2)

    dt = time()-t0
    print(f"[info] null completed in {dt:.1f} s")

    # print results
    print("----------------------------------------------------------")
    print(" RESULTS (n_lmn and p-values)")
    print("----------------------------------------------------------")

    for l in range(lmax+1):
        for m in range(-l, l+1):
            for n in range(K):
                nR = nlmn_real[(l,m,n)]
                nulls = null_nlmn[(l,m,n)]
                p = sum(1 for v in nulls if v >= nR) / len(nulls)
                print(f" (l,m,n)=({l:2d},{m:3d},{n}) | n_lmn={nR:.3e} | p={p:.5f}")

    print("----------------------------------------------------------")
    print(" interpretation:")
    print("   p < 0.05   → over-occupied 3D mode (angular+radial)")
    print("   p < 0.001  → strong resonant eigenmode candidate")
    print("==========================================================")
    print(" test 106B complete")
    print("==========================================================")


if __name__ == "__main__":
    main()
