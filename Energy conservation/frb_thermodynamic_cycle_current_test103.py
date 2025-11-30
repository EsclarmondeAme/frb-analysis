#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test 103 — Thermodynamic Cycle-Current Test
-------------------------------------------

Goal:
    Detect non-equilibrium "cycle currents" in the remnant-time / phase
    macrostate dynamics, using the same macrostates as Test 102.

Idea:
    - states i are defined by (rt_sign, phase_bin)
    - transition matrix T(i->j) estimated from time-ordered FRBs
    - for each triple (i,j,k), define cycle current:

        J_ijk = T(i->j) T(j->k) T(k->i) - T(i->k) T(k->j) T(j->i)

    - in equilibrium, these cycle currents vanish (detailed balance extended).
    - non-zero J_ijk indicate driven, non-conservative dynamics.

Statistic:
    J_tot = sum_{i<j<k} |J_ijk|

Null:
    scramble the time order many times, recompute T and J_tot.

Interpretation:
    - low p(J_tot) -> strong evidence for non-equilibrium cycle currents:
                      manifold supports "work-like" cycles.
    - high p       -> consistent with equilibrium-like dynamics.
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
# coordinate utilities
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
# macrostate encoding (reuse Test 102 logic)
# ------------------------------------------------------------
def compute_macrostates(rows, axis_vec, n_phase_bins=12):
    states = []
    phase_edges = np.linspace(0, 2*np.pi, n_phase_bins+1)

    for ra, dec, _ in rows:
        v = radec_to_xyz(ra, dec)

        # remnant-time sign from unified axis
        s = 1 if np.dot(v, axis_vec) > 0 else -1

        # phase angle
        phi = np.arctan2(v[1], v[0])
        if phi < 0:
            phi += 2*np.pi

        k = np.searchsorted(phase_edges, phi) - 1
        if k < 0:
            k = 0
        if k >= n_phase_bins:
            k = n_phase_bins - 1

        states.append((s, k))

    unique = sorted(set(states))
    index_map = {st: i for i, st in enumerate(unique)}
    encoded = [index_map[s] for s in states]

    return encoded, unique


def compute_transition_matrix(encoded_states, n_states):
    T = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(encoded_states)-1):
        a = encoded_states[i]
        b = encoded_states[i+1]
        T[a, b] += 1.0

    row_sums = T.sum(axis=1)
    for i in range(n_states):
        if row_sums[i] > 0:
            T[i] /= row_sums[i]
    return T


# ------------------------------------------------------------
# cycle-current statistic
# ------------------------------------------------------------
def cycle_current_total(T):
    n = T.shape[0]
    J_tot = 0.0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # forward loop i->j->k->i
                fwd = T[i,j]*T[j,k]*T[k,i]
                # reverse loop i->k->j->i
                rev = T[i,k]*T[k,j]*T[j,i]
                J = fwd - rev
                J_tot += abs(J)
    return float(J_tot)


# ------------------------------------------------------------
# null ensemble
# ------------------------------------------------------------
def run_null(encoded, n_states, n_mc):
    J_list = []
    seq = list(encoded)
    for _ in range(n_mc):
        random.shuffle(seq)
        Tn = compute_transition_matrix(seq, n_states)
        Jn = cycle_current_total(Tn)
        J_list.append(Jn)
    return J_list


def p_value(real, null):
    return sum(1 for v in null if v >= real) / len(null)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_thermodynamic_cycle_current_test103.py frbs_unified.csv [N_null]")
        sys.exit(1)

    path = sys.argv[1]
    N_null = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    print("========================================================")
    print(" Test 103 — Thermodynamic Cycle-Current Test")
    print("========================================================")
    print(f"[info] loading catalog: {path}")

    rows = load_catalog(path)
    N = len(rows)
    print(f"[info] N_FRB = {N}")

    # sort by time
    rows.sort(key=lambda r: r[2])

    # unified axis in galactic coords
    axis = galactic_lb_to_xyz(159.8, -0.5)

    print("[info] encoding macrostates (sign, phase_bin)...")
    encoded, mapping = compute_macrostates(rows, axis, n_phase_bins=12)
    n_states = len(mapping)
    print(f"[info] number of macrostates = {n_states}")

    print("[info] computing REAL transition matrix...")
    T_real = compute_transition_matrix(encoded, n_states)
    J_real = cycle_current_total(T_real)
    print(f"[info] J_real (total cycle current) = {J_real:.6e}")

    print("[info] running null ensemble...")
    t0 = time()
    J_null = run_null(encoded, n_states, N_null)
    dt = time() - t0
    print(f"[info] null completed in {dt:.2f} s")

    meanJ = statistics.mean(J_null)
    stdJ  = statistics.stdev(J_null)
    pJ    = p_value(J_real, J_null)

    print("--------------------------------------------------------")
    print(" RESULTS")
    print("--------------------------------------------------------")
    print(f"J_real = {J_real:.6e}")
    print(f"J_null_mean = {meanJ:.6e}, J_null_std = {stdJ:.6e}")
    print(f"p-value = {pJ:.6f}")
    print("--------------------------------------------------------")
    print(" interpretation:")
    print("   - low p (p < 0.05) → strong non-equilibrium cycle currents:")
    print("       manifold supports directed loops in macrostate space")
    print("       (thermodynamic cycles that can, in principle, do work).")
    print("   - high p → cycles consistent with equilibrium fluctuations.")
    print("========================================================")
    print(" test 103 complete")
    print("========================================================")


if __name__ == '__main__':
    main()
