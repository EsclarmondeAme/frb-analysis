#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test 104 — Manifold Population-Inversion Test
---------------------------------------------

Scientific question:
    Does the FRB manifold exhibit population inversion analogous
    to negative temperature? i.e., in certain regions of the
    unified-axis geometry, are high-energy FRBs over-populated
    relative to mid-energy ones?

Idea:
    - Use fluence as energy proxy.
    - Define geometric regions:
        * remnant-time hemispheres (sign = ±1)
        * axis-distance shells (bins in theta_u)
    - Bin fluence into quantile levels:
        low, mid, high, very high (top 5%).
    - For each region R, compute inversion ratio:

            I_R = N_high(R) / N_mid(R)

      If I_R_real >> I_R_null, region shows population inversion.

Null:
    Shuffle fluence values across positions many times (default 5000),
    recompute inversion ratios, measure p-values.

Interpretation:
    - p < 0.05 → statistically significant population inversion:
                 analog of negative temperature in that region.
    - p >= 0.05 → consistent with equilibrium-like monotonic energy population.
"""

import sys
import csv
import math
import random
from time import time
import numpy as np
import statistics


# ------------------------------------------------------------
# load catalog
# ------------------------------------------------------------
def load_catalog(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        R = csv.DictReader(f)
        for r in R:
            try:
                flu = float(r["fluence"])
                ra = float(r["ra"])
                dec = float(r["dec"])
                theta_u = float(r["theta_unified"])
            except:
                continue
            rows.append((flu, ra, dec, theta_u))
    return rows


# ------------------------------------------------------------
# vector utilities
# ------------------------------------------------------------
def galactic_lb_to_xyz(l_deg, b_deg):
    l = math.radians(l_deg)
    b = math.radians(b_deg)
    v = np.array([
        math.cos(b)*math.cos(l),
        math.cos(b)*math.sin(l),
        math.sin(b)
    ])
    return v/np.linalg.norm(v)


def radec_to_xyz(ra_deg, dec_deg):
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    return np.array([
        math.cos(dec)*math.cos(ra),
        math.cos(dec)*math.sin(ra),
        math.sin(dec)
    ])


# ------------------------------------------------------------
# region identification
# ------------------------------------------------------------
def compute_regions(rows, axis_vec, n_shells=4):
    """
    returns:
        hemisphere: list of +1 / -1
        shell:     list of shell index 0..(n_shells-1)
    """
    hemis = []
    shells = []
    # define theta bins: equal-width bins from 0° to 180°
    theta_edges = np.linspace(0, 180, n_shells+1)

    for _, ra, dec, theta_u in rows:
        v = radec_to_xyz(ra, dec)
        s = 1 if np.dot(v, axis_vec) > 0 else -1
        hemis.append(s)

        # theta_u already provided in CSV (deg)
        k = np.searchsorted(theta_edges, theta_u) - 1
        if k < 0:
            k = 0
        if k >= n_shells:
            k = n_shells-1
        shells.append(k)

    return hemis, shells, theta_edges


# ------------------------------------------------------------
# quantile fluence bins
# ------------------------------------------------------------
def compute_flunce_levels(fluences):
    """
    quantiles:
        low:    bottom 40%
        mid:    40-80%
        high:   80-95%
        vhigh:  top 5%
    """
    q40 = np.quantile(fluences, 0.40)
    q80 = np.quantile(fluences, 0.80)
    q95 = np.quantile(fluences, 0.95)

    levels = []
    for f in fluences:
        if f < q40:
            levels.append(0)   # low
        elif f < q80:
            levels.append(1)   # mid
        elif f < q95:
            levels.append(2)   # high
        else:
            levels.append(3)   # very-high
    return levels, (q40, q80, q95)


# ------------------------------------------------------------
# population inversion ratio
# ------------------------------------------------------------
def inversion_ratio(levels, hemis, shells, region_type, region_index):
    """
    region_type : 'hemisphere' or 'shell'
    region_index: +1 / -1 for hemisphere, 0..n_shells-1 for shells

    levels: list of 0..3 (low/mid/high/vhigh)
    """
    high = 0
    mid = 0
    N = len(levels)
    for i in range(N):
        if region_type == "hemisphere":
            if hemis[i] != region_index:
                continue
        else:  # shell
            if shells[i] != region_index:
                continue

        if levels[i] == 1:  # mid
            mid += 1
        elif levels[i] >= 2:  # high or vhigh
            high += 1

    if mid == 0:
        return 0.0
    return high/mid


# ------------------------------------------------------------
# null ensemble
# ------------------------------------------------------------
def run_null(fluences, hemis, shells, N_null):
    N = len(fluences)
    base = list(fluences)
    I_null_hemi_plus = []
    I_null_hemi_minus = []
    I_null_shells = [[] for _ in range(max(shells)+1)]

    for _ in range(N_null):
        random.shuffle(base)
        # assign new fluence-levels under shuffle
        lv, _ = compute_flunce_levels(base)

        # hemispheres
        I_null_hemi_plus.append(inversion_ratio(lv, hemis, shells, "hemisphere", +1))
        I_null_hemi_minus.append(inversion_ratio(lv, hemis, shells, "hemisphere", -1))

        # shells
        for k in range(len(I_null_shells)):
            I_null_shells[k].append(inversion_ratio(lv, hemis, shells, "shell", k))

    return I_null_hemi_plus, I_null_hemi_minus, I_null_shells


def p_value(real, null):
    return sum(1 for v in null if v >= real) / len(null)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_population_inversion_test104.py frbs_unified.csv [N_null]")
        sys.exit(1)

    path = sys.argv[1]
    N_null = int(sys.argv[2]) if len(sys.argv)>2 else 5000

    print("=======================================================")
    print(" Test 104 — Population-Inversion / Negative-Temperature")
    print("=======================================================")

    print(f"[info] loading: {path}")
    rows = load_catalog(path)
    N = len(rows)
    print(f"[info] N_FRB = {N}")

    # unified axis in galactic coords
    axis = galactic_lb_to_xyz(159.8, -0.5)

    print("[info] computing regions (hemispheres, shells)...")
    hemis, shells, edges = compute_regions(rows, axis, n_shells=4)

    # fluence levels
    fluences = [r[0] for r in rows]
    levels, qs = compute_flunce_levels(fluences)
    print("[info] fluence quantiles (q40, q80, q95) =", qs)

    # real inversion ratios
    I_plus  = inversion_ratio(levels, hemis, shells, "hemisphere", +1)
    I_minus = inversion_ratio(levels, hemis, shells, "hemisphere", -1)
    I_shell = [inversion_ratio(levels, hemis, shells, "shell", k) for k in range(4)]

    print("[info] running null ensemble...")
    t0 = time()
    I_null_plus, I_null_minus, I_null_shells = run_null(fluences, hemis, shells, N_null)
    dt = time() - t0
    print(f"[info] null completed in {dt:.2f} s")

    # compute p-values
    p_plus  = p_value(I_plus,  I_null_plus)
    p_minus = p_value(I_minus, I_null_minus)
    p_shell = [p_value(I_shell[k], I_null_shells[k]) for k in range(4)]

    print("-------------------------------------------------------")
    print(" RESULTS — Population Inversion Ratios")
    print("-------------------------------------------------------")
    print(f"hemisphere +1:   I = {I_plus:.4f},  p = {p_plus:.4f}")
    print(f"hemisphere -1:   I = {I_minus:.4f}, p = {p_minus:.4f}")
    for k in range(4):
        lo, hi = edges[k], edges[k+1]
        print(f"shell {k} ({lo:.1f}°–{hi:.1f}°):  I = {I_shell[k]:.4f}, p = {p_shell[k]:.4f}")

    print("-------------------------------------------------------")
    print(" interpretation:")
    print("   p < 0.05 → significant population inversion:")
    print("       region has more high-energy than mid-energy events")
    print("       analog of negative temperature in statistical mechanics.")
    print("   p ≥ 0.05 → no evidence for inversion; consistent with monotonic populations.")
    print("=======================================================")
    print(" test 104 complete")
    print("=======================================================")


if __name__ == "__main__":
    main()
