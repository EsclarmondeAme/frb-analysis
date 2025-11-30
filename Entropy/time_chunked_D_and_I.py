#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
time_chunked_D_and_I.py

Time-chunked persistence of:
    - detailed-balance violation D (Test 102)
    - population inversion ratios I (Test 104)

Goal:
    Check whether the non-equilibrium flow (D) and the population inversion
    (I_shell) are persistent over time, rather than one-off artefacts.

Usage:
    python time_chunked_D_and_I.py frbs_unified.csv [N_chunks] [N_null_per_chunk]

    N_chunks          : number of equal-sized time chunks (default: 6)
    N_null_per_chunk  : null realisations per chunk (default: 1000)

Per chunk:
    1. Take time-ordered subset of FRBs.
    2. Compute D_real via the same macrostate construction as Test 102:
        - macrostates m = (rt_sign, phase_bin)
    3. Compute D_null by shuffling time order inside the chunk.
    4. Compute population inversion ratios I_shell_k as in Test 104:
        - hemisphere/sign from unified axis
        - shells in theta_unified: [0-45], [45-90], [90-135], [135-180]
        - fluence quantile levels
    5. Compute I_null_shell_k by shuffling fluence across FRBs in the chunk.
    6. Record p-values and summary.

Summary metrics:
    - fraction of chunks with p_D < 0.01
    - fraction of chunks with p_I_shell2 < 0.01 (the previously inverted shell)
"""

import sys
import csv
import math
import random
from time import time
import numpy as np
import statistics


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
                theta_u = float(r["theta_unified"])
            except Exception:
                continue
            rows.append((ra, dec, mjd, flu, theta_u))
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
    return np.array([x, y, z])


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
# macrostates for D (Test 102 logic)
# ------------------------------------------------------------
def compute_macrostates(rows_chunk, axis_vec, n_phase_bins=12):
    """
    rows_chunk: list of (ra, dec, mjd, flu, theta_u)
    returns:
        encoded_states: list of integer macrostate indices
        mapping: list of (sign, phase_bin)
    """
    phase_edges = np.linspace(0, 2*np.pi, n_phase_bins + 1)
    states = []

    for ra, dec, _, _, _ in rows_chunk:
        v = radec_to_xyz(ra, dec)
        s = 1 if np.dot(v, axis_vec) > 0 else -1
        phi = math.atan2(v[1], v[0])
        if phi < 0:
            phi += 2*math.pi
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
    for i in range(len(encoded_states) - 1):
        a = encoded_states[i]
        b = encoded_states[i + 1]
        T[a, b] += 1.0
    row_sums = T.sum(axis=1)
    for i in range(n_states):
        if row_sums[i] > 0:
            T[i] /= row_sums[i]
    return T


def detailed_balance_stat(T):
    """
    D = sum_{A,B} (T(A->B) - T(B->A))^2
    """
    n = T.shape[0]
    D = 0.0
    for i in range(n):
        for j in range(n):
            D += (T[i, j] - T[j, i])**2
    return float(D)


def run_null_D(encoded_states, n_states, n_mc):
    D_null = []
    seq = list(encoded_states)
    for _ in range(n_mc):
        random.shuffle(seq)
        Tn = compute_transition_matrix(seq, n_states)
        Dn = detailed_balance_stat(Tn)
        D_null.append(Dn)
    return D_null


# ------------------------------------------------------------
# inversion I (Test 104 logic, per chunk)
# ------------------------------------------------------------
def compute_regions(rows_chunk, axis_vec, n_shells=4):
    hemis = []
    shells = []
    theta_edges = np.linspace(0, 180, n_shells + 1)

    for ra, dec, _, _, theta_u in rows_chunk:
        v = radec_to_xyz(ra, dec)
        s = 1 if np.dot(v, axis_vec) > 0 else -1
        hemis.append(s)

        k = np.searchsorted(theta_edges, theta_u) - 1
        if k < 0:
            k = 0
        if k >= n_shells:
            k = n_shells - 1
        shells.append(k)

    return hemis, shells, theta_edges


def compute_fluence_levels(fluences):
    """
    quantiles:
        low:   < q40
        mid:   q40–q80
        high:  q80–q95
        vhigh: >= q95
    """
    q40 = np.quantile(fluences, 0.40)
    q80 = np.quantile(fluences, 0.80)
    q95 = np.quantile(fluences, 0.95)

    levels = []
    for f in fluences:
        if f < q40:
            levels.append(0)
        elif f < q80:
            levels.append(1)
        elif f < q95:
            levels.append(2)
        else:
            levels.append(3)
    return levels, (q40, q80, q95)


def inversion_ratio(levels, hemis, shells, region_type, region_index):
    high = 0
    mid = 0
    N = len(levels)
    for i in range(N):
        if region_type == "hemisphere":
            if hemis[i] != region_index:
                continue
        else:
            if shells[i] != region_index:
                continue
        if levels[i] == 1:
            mid += 1
        elif levels[i] >= 2:
            high += 1
    if mid == 0:
        return 0.0
    return high / mid


def run_null_I(fluences, hemis, shells, n_mc, target_shell=2):
    """
    Compute null for I in:
        - hemisphere +1
        - hemisphere -1
        - shell target_shell
    in this chunk.
    """
    N = len(fluences)
    base = list(fluences)
    I_plus_null = []
    I_minus_null = []
    I_shell_null = []

    for _ in range(n_mc):
        random.shuffle(base)
        levels, _ = compute_fluence_levels(base)
        I_plus_null.append(inversion_ratio(levels, hemis, shells, "hemisphere", +1))
        I_minus_null.append(inversion_ratio(levels, hemis, shells, "hemisphere", -1))
        I_shell_null.append(inversion_ratio(levels, hemis, shells, "shell", target_shell))

    return I_plus_null, I_minus_null, I_shell_null


# ------------------------------------------------------------
# utilities
# ------------------------------------------------------------
def p_value(real, null):
    return sum(1 for v in null if v >= real) / len(null)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python time_chunked_D_and_I.py frbs_unified.csv [N_chunks] [N_null_per_chunk]")
        sys.exit(1)

    path = sys.argv[1]
    N_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    N_null = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    print("=====================================================")
    print(" Time-chunked D and I persistence test (102 + 104)")
    print("=====================================================")
    print(f"[info] loading: {path}")
    rows = load_catalog(path)
    N = len(rows)
    print(f"[info] N_FRB total = {N}")

    # sort by time
    rows.sort(key=lambda r: r[2])
    mjd_all = [r[2] for r in rows]
    print(f"[info] MJD range: {min(mjd_all):.3f} – {max(mjd_all):.3f}")

    # unified axis (galactic)
    axis = galactic_lb_to_xyz(159.8, -0.5)

    # chunking by equal counts
    chunk_size = N // N_chunks
    if chunk_size < 30:
        print("[warn] very small chunk size; results may be noisy.")

    results = []

    print(f"[info] splitting into {N_chunks} chunks of ~{chunk_size} FRBs")
    print("-----------------------------------------------------")

    for c in range(N_chunks):
        start = c * chunk_size
        end = (c + 1) * chunk_size if c < N_chunks - 1 else N
        rows_chunk = rows[start:end]
        if len(rows_chunk) < 20:
            print(f"[warn] chunk {c} has very few FRBs ({len(rows_chunk)}); skipping.")
            continue

        mjd_vals = [r[2] for r in rows_chunk]
        print(f"[chunk {c}] N = {len(rows_chunk)}, MJD {min(mjd_vals):.3f} – {max(mjd_vals):.3f}")

        # ---------------- D in this chunk ----------------
        encoded, mapping = compute_macrostates(rows_chunk, axis, n_phase_bins=12)
        n_states = len(mapping)
        T_real = compute_transition_matrix(encoded, n_states)
        D_real = detailed_balance_stat(T_real)

        D_null = run_null_D(encoded, n_states, N_null)
        meanD = statistics.mean(D_null)
        stdD = statistics.stdev(D_null) if len(D_null) > 1 else 0.0
        pD = sum(1 for v in D_null if v >= D_real) / len(D_null)

        # ---------------- I in this chunk ----------------
        hemis, shells, theta_edges = compute_regions(rows_chunk, axis, n_shells=4)
        fluences = [r[3] for r in rows_chunk]
        levels, qs = compute_fluence_levels(fluences)

        # focus on shell 2 (90–135 degrees) as in global test 104
        I_shell2 = inversion_ratio(levels, hemis, shells, "shell", 2)

        I_plus = inversion_ratio(levels, hemis, shells, "hemisphere", +1)
        I_minus = inversion_ratio(levels, hemis, shells, "hemisphere", -1)

        I_plus_null, I_minus_null, I_shell2_null = run_null_I(
            fluences, hemis, shells, N_null, target_shell=2
        )

        pI_shell2 = p_value(I_shell2, I_shell2_null)
        pI_plus = p_value(I_plus, I_plus_null)
        pI_minus = p_value(I_minus, I_minus_null)

        print(f"  D_real       = {D_real:.4f}, null_mean = {meanD:.4f}, null_std = {stdD:.4f}, p_D = {pD:.4f}")
        print(f"  I_shell2     = {I_shell2:.4f}, p_I_shell2 = {pI_shell2:.4f}")
        print(f"  I_hemi(+1)   = {I_plus:.4f},  p_I_plus    = {pI_plus:.4f}")
        print(f"  I_hemi(-1)   = {I_minus:.4f}, p_I_minus   = {pI_minus:.4f}")
        print("-----------------------------------------------------")

        results.append({
            "chunk": c,
            "N": len(rows_chunk),
            "mjd_min": min(mjd_vals),
            "mjd_max": max(mjd_vals),
            "D_real": D_real,
            "D_mean": meanD,
            "D_std": stdD,
            "pD": pD,
            "I_shell2": I_shell2,
            "pI_shell2": pI_shell2
        })

    # summary
    if not results:
        print("[error] no valid chunks processed.")
        return

    frac_pD_001 = sum(1 for r in results if r["pD"] < 0.01) / len(results)
    frac_pI_001 = sum(1 for r in results if r["pI_shell2"] < 0.01) / len(results)

    print("=====================================================")
    print(" SUMMARY PERSISTENCE METRICS")
    print("=====================================================")
    print(f"chunks processed: {len(results)}")
    print(f"fraction of chunks with p_D < 0.01        = {frac_pD_001:.3f}")
    print(f"fraction of chunks with p_I_shell2 < 0.01 = {frac_pI_001:.3f}")
    print("=====================================================")
    print(" done.")
    print("=====================================================")


if __name__ == "__main__":
    main()
