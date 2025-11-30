#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test 105 — Energy Flux Budget Test
-----------------------------------

Scientific question:
    Does any region of the unified-axis manifold produce statistically
    significant excess energy per unit time, beyond isotropic expectation?

Interpretation:
    - low p (p < 0.05): region shows excess energy output (possible anomaly)
    - high p: consistent with standard astrophysical distribution

Energy model:
    E = 4π D_L(z)^2 * fluence
    using standard ΛCDM luminosity distance

Null model:
    Shuffle fluences across sky positions many times, recompute flux.
"""

import sys
import csv
import math
import random
import numpy as np
from time import time
import statistics

# ------------------------------------------------------------
# Cosmology utilities (flat ΛCDM, H0=70, Ωm=0.3)
# ------------------------------------------------------------
c = 299792.458  # km/s

def E_z(z):
    return 1.0 / math.sqrt(0.3*(1+z)**3 + 0.7)

def luminosity_distance(z):
    """Simple numerical integral for D_L in Mpc."""
    if z <= 0:
        return 0.0
    N = 200
    zs = np.linspace(0, z, N)
    dz = z / (N - 1)
    integral = np.sum([E_z(zp) for zp in zs]) * dz
    D_C = (c / 70.0) * integral
    return (1 + z) * D_C  # Mpc

# ------------------------------------------------------------
# loading
# ------------------------------------------------------------
def load_catalog(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        R = csv.DictReader(f)
        for r in R:
            try:
                ra = float(r["ra"])
                dec = float(r["dec"])
                mjd = float(r["mjd"])
                flu = float(r["fluence"])
                z = float(r["z_est"]) if r["z_est"] != "" else 0.2
                theta_u = float(r["theta_unified"])
            except:
                continue
            rows.append((ra, dec, mjd, flu, z, theta_u))
    return rows

# ------------------------------------------------------------
# coordinate utilities
# ------------------------------------------------------------
def radec_to_xyz(ra_deg, dec_deg):
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    return np.array([
        math.cos(dec)*math.cos(ra),
        math.cos(dec)*math.sin(ra),
        math.sin(dec)
    ])

def galactic_lb_to_xyz(l_deg, b_deg):
    l = math.radians(l_deg)
    b = math.radians(b_deg)
    v = np.array([
        math.cos(b)*math.cos(l),
        math.cos(b)*math.sin(l),
        math.sin(b)
    ])
    return v / np.linalg.norm(v)

# ------------------------------------------------------------
# region identification
# ------------------------------------------------------------
def compute_regions(rows, axis_vec, n_shells=4):
    hemis = []
    shells = []
    theta_edges = np.linspace(0, 180, n_shells + 1)
    for ra, dec, *_ , theta_u in rows:
        v = radec_to_xyz(ra, dec)
        s = 1 if np.dot(v, axis_vec) > 0 else -1
        hemis.append(s)
        k = np.searchsorted(theta_edges, theta_u) - 1
        if k < 0: k = 0
        if k >= n_shells: k = n_shells - 1
        shells.append(k)
    return hemis, shells, theta_edges

# ------------------------------------------------------------
# compute energy flux in region
# ------------------------------------------------------------
def energy_flux(rows, hemis, shells, region_type, region_index, dt_days):
    total_E = 0.0
    for i, (_, _, _, flu, z, _) in enumerate(rows):
        if region_type == "hemisphere":
            if hemis[i] != region_index:
                continue
        else:
            if shells[i] != region_index:
                continue
        DL = luminosity_distance(z)  # Mpc
        E_iso = 4.0 * math.pi * (DL*3.086e22)**2 * flu  # Joules (fluence in Jy ms assumed)
        total_E += E_iso
    # convert dt_days → seconds
    flux = total_E / (dt_days * 86400.0)
    return flux

def p_value(real, null):
    return sum(1 for v in null if v >= real) / len(null)

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_energy_flux_budget_test105.py frbs_unified.csv [N_null]")
        sys.exit(1)

    path = sys.argv[1]
    N_null = int(sys.argv[2]) if len(sys.argv)>2 else 2000

    print("=======================================================")
    print(" Test 105 — Energy Flux Budget Test")
    print("=======================================================")
    print("[info] loading catalog:", path)
    rows = load_catalog(path)
    N = len(rows)
    print("[info] N_FRB =", N)

    # time span
    mjd_vals = [r[2] for r in rows]
    dt = max(mjd_vals) - min(mjd_vals)
    print(f"[info] MJD span = {dt:.3f} days")

    # unified axis
    axis = galactic_lb_to_xyz(159.8, -0.5)
    hemis, shells, edges = compute_regions(rows, axis, n_shells=4)

    # real fluxes
    print("[info] computing REAL energy fluxes...")
    F_plus  = energy_flux(rows, hemis, shells, "hemisphere", +1, dt)
    F_minus = energy_flux(rows, hemis, shells, "hemisphere", -1, dt)
    F_shell = [energy_flux(rows, hemis, shells, "shell", k, dt) for k in range(4)]

    # null fluxes
    print("[info] running null ensemble...")
    t0 = time()
    fluences = [r[3] for r in rows]
    null_plus = []
    null_minus = []
    null_shells = [[] for _ in range(4)]
    for _ in range(N_null):
        random.shuffle(fluences)
        # assign shuffled fluence back temporarily
        rows_sh = [(r[0], r[1], r[2], fluences[i], r[4], r[5]) for i, r in enumerate(rows)]
        null_plus.append(energy_flux(rows_sh, hemis, shells, "hemisphere", +1, dt))
        null_minus.append(energy_flux(rows_sh, hemis, shells, "hemisphere", -1, dt))
        for k in range(4):
            null_shells[k].append(energy_flux(rows_sh, hemis, shells, "shell", k, dt))

    print(f"[info] null completed in {time()-t0:.2f} s")

    # p-values
    p_plus  = p_value(F_plus,  null_plus)
    p_minus = p_value(F_minus, null_minus)
    p_shell = [p_value(F_shell[k], null_shells[k]) for k in range(4)]

    print("-------------------------------------------------------")
    print(" RESULTS — Energy Fluxes (J/s) and p-values")
    print("-------------------------------------------------------")
    print(f"hemisphere +1:   flux = {F_plus:.3e},  p = {p_plus:.4f}")
    print(f"hemisphere -1:   flux = {F_minus:.3e}, p = {p_minus:.4f}")
    for k in range(4):
        lo, hi = edges[k], edges[k+1]
        print(f"shell {k} ({lo:.1f}°–{hi:.1f}°): flux = {F_shell[k]:.3e}, p = {p_shell[k]:.4f}")

    print("-------------------------------------------------------")
    print(" interpretation:")
    print("   p < 0.05 → region outputs more energy than isotropic null,")
    print("              potential signature of an energetic anomaly.")
    print("   p ≥ 0.05 → consistent with standard astrophysical processes.")
    print("=======================================================")
    print(" test 105 complete")
    print("=======================================================")


if __name__ == "__main__":
    main()
