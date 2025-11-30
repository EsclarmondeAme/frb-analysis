#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test 102 — Remnant-Time Detailed-Balance Test
---------------------------------------------

Scientific question:
    Does the FRB remnant-time + harmonic-phase manifold obey
    equilibrium-like detailed balance, or does it behave like a 
    non-equilibrium thermodynamic system with an arrow of time?

Principle:
    In thermodynamic equilibrium, macrostate transitions obey:
        P(A → B) ≈ P(B → A)
    up to noise.

    If detailed balance is violated significantly, the system
    has an intrinsic directional "flow" in macrostate space,
    signaling non-equilibrium structure.

Construction:
    1. Sort FRBs by observation time (MJD).
    2. For each FRB, compute:
            - remnant-time sign (from unified axis)
            - phase φ = arctan2(y, x) from galactic coords
              (same φ used in tests 85 series)
            - phase bin index
            - macrostate m_i = (sign_i, phase_bin_i)
    3. Build transition counts N(A→B) for consecutive FRBs.
    4. Compute detailed-balance violation:
            D_real = Σ_{A,B} (T(A→B) - T(B→A))²
    5. Null: scramble time order many times (default 2000)
       and recompute D_null.
    6. p-value = fraction of D_null ≥ D_real.

Interpretation:
    - p << 0.05 → strong detailed-balance violation:
                  non-equilibrium thermodynamic behaviour.
    - p ~ 0.5   → consistent with equilibrium-like dynamics.
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
                ra = float(r["ra"])
                dec = float(r["dec"])
                mjd = float(r["mjd"])
            except:
                continue
            rows.append((ra, dec, mjd))
    return rows


# ------------------------------------------------------------
# coordinate transforms
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
    return v/np.linalg.norm(v)


# ------------------------------------------------------------
# macrostate construction
# ------------------------------------------------------------
def compute_macrostates(rows, axis_vec, n_phase_bins=12):
    """
    returns:
        states  : list of macrostate indices for each FRB
        mapping : list of (rt_sign, phase_bin) tuples
    """
    states = []
    mapping = []

    # define phase bins
    phase_edges = np.linspace(0, 2*np.pi, n_phase_bins+1)

    for ra, dec, _ in rows:
        v = radec_to_xyz(ra, dec)

        # remnant-time sign (same as 85-series)
        s = 1 if np.dot(v, axis_vec) > 0 else -1

        # phase angle
        phi = np.arctan2(v[1], v[0])   # 0 to 2π
        if phi < 0:
            phi += 2*np.pi

        # phase bin index
        k = np.searchsorted(phase_edges, phi) - 1
        if k < 0:
            k = 0
        if k >= n_phase_bins:
            k = n_phase_bins-1

        states.append((s, k))

    # enumerate unique macrostates
    unique = sorted(set(states))
    mapping = unique

    # convert each (s,k) to integer macrostate index
    index_map = {state:i for i,state in enumerate(unique)}
    encoded = [index_map[s] for s in states]

    return encoded, mapping


# ------------------------------------------------------------
# transition matrix + detailed balance measure
# ------------------------------------------------------------
def compute_transition_matrix(encoded_states, n_states):
    """
    encoded_states: list of integers (macrostate indices) in time order
    return: T (n_states x n_states) transition frequency matrix
    """
    T = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(encoded_states)-1):
        a = encoded_states[i]
        b = encoded_states[i+1]
        T[a,b] += 1

    # normalize each row to probabilities
    row_sums = T.sum(axis=1)
    for i in range(n_states):
        if row_sums[i] > 0:
            T[i] /= row_sums[i]
    return T


def detailed_balance_stat(T):
    """
    compute D = sum_{A,B} (T(A→B) - T(B→A))²
    """
    n = T.shape[0]
    D = 0.0
    for i in range(n):
        for j in range(n):
            D += (T[i,j] - T[j,i])**2
    return float(D)


# ------------------------------------------------------------
# null ensemble
# ------------------------------------------------------------
def run_null(encoded_states, n_states, n_mc):
    Dvals = []
    N = len(encoded_states)
    seq = list(encoded_states)
    for _ in range(n_mc):
        random.shuffle(seq)
        Tn = compute_transition_matrix(seq, n_states)
        Dn = detailed_balance_stat(Tn)
        Dvals.append(Dn)
    return Dvals


# ------------------------------------------------------------
# p-value
# ------------------------------------------------------------
def p_value(real, null):
    return sum(1 for v in null if v >= real) / len(null)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_thermodynamic_detailed_balance_test102.py frbs_unified.csv [N_null]")
        sys.exit(1)

    path = sys.argv[1]
    N_null = int(sys.argv[2]) if len(sys.argv)>2 else 2000

    print("==========================================================")
    print(" Test 102 — Remnant-Time Detailed-Balance Thermodynamic Test")
    print("==========================================================")
    print(f"[info] loading catalog: {path}")

    rows = load_catalog(path)
    N = len(rows)
    print(f"[info] N_FRB = {N}")

    # sort by time
    rows.sort(key=lambda r: r[2])

    # unified axis (galactic)
    axis = galactic_lb_to_xyz(159.8, -0.5)

    print("[info] encoding macrostates (sign, phase_bin)...")
    encoded, mapping = compute_macrostates(rows, axis, n_phase_bins=12)
    n_states = len(mapping)
    print(f"[info] number of macrostates = {n_states}")

    print("[info] computing REAL transition matrix...")
    T_real = compute_transition_matrix(encoded, n_states)
    D_real = detailed_balance_stat(T_real)
    print(f"[info] D_real (detailed-balance violation) = {D_real:.6f}")

    print("[info] running null ensemble...")
    t0 = time()
    D_null = run_null(encoded, n_states, N_null)
    dt = time() - t0
    print(f"[info] null completed in {dt:.2f} s")

    meanD = statistics.mean(D_null)
    stdD  = statistics.stdev(D_null)
    pD    = p_value(D_real, D_null)

    print("----------------------------------------------------------")
    print(" RESULTS")
    print("----------------------------------------------------------")
    print(f"D_real = {D_real:.6f}")
    print(f"D_null_mean = {meanD:.6f}, D_null_std = {stdD:.6f}")
    print(f"p-value = {pD:.6f}")
    print("----------------------------------------------------------")
    print(" interpretation:")
    print("   - low p (p < 0.05) → detailed balance violated:")
    print("       → non-equilibrium thermodynamics")
    print("       → intrinsic directional flow in remnant-time/phase manifold")
    print("   - high p → consistent with equilibrium-like behaviour")
    print("==========================================================")
    print(" test 102 complete")
    print("==========================================================")


if __name__ == "__main__":
    main()
