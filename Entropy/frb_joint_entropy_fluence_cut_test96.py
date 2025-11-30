#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test 96 — Fluence-limited robustness of the joint entropy deficit (Test 91)

For a sequence of fluence cuts, we keep only the brightest N bursts and
recompute the Test 91 joint-entropy statistic:

    H_real, null_mean, null_std, p_deficit

This checks whether the Test 91 signal is driven by faint, near-threshold
bursts, or whether it persists in the bright, high-S/N population.

pure version:
    - no fallbacks
    - no apply_filter
    - requires frbs_unified_for_test91.csv with columns:
        ra, dec, theta_u, phi_h, rt_sign, fluence
"""

import argparse
import logging
import numpy as np
import pandas as pd


# -------------------------------------------------------------
# utilities
# -------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )


def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"theta_u", "phi_h", "rt_sign", "fluence"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"catalog missing required columns: {sorted(missing)}")
    return df


def bin_var(x, edges):
    idx = np.digitize(x, edges) - 1
    idx[(x < edges[0]) | (x >= edges[-1])] = -1
    return idx


def shannon_entropy_joint(theta_u, rt_sign, phi_h):
    theta_edges = np.array([0, 20, 35, 50, 90, 180])
    rt_edges    = np.array([-2, 0, 2])
    phi_edges   = np.linspace(0, 2*np.pi, 12 + 1)

    phi_h = np.mod(phi_h, 2*np.pi)

    b_theta = bin_var(theta_u, theta_edges)
    b_rt    = bin_var(rt_sign, rt_edges)
    b_phi   = bin_var(phi_h,  phi_edges)

    valid = (b_theta >= 0) & (b_rt >= 0) & (b_phi >= 0)
    if not np.any(valid):
        raise RuntimeError("no points in joint bins")

    bt = b_theta[valid]
    br = b_rt[valid]
    bp = b_phi[valid]

    C = np.zeros((5, 2, 12), dtype=int)
    for t, r, p in zip(bt, br, bp):
        C[t, r, p] += 1

    P = C.astype(float) / C.sum()
    P = P[P > 0]
    return float(-np.sum(P * np.log(P)))


def test91_stat(theta_u, rt_sign, phi_h, n_null=2000, seed=1):
    rng = np.random.default_rng(seed)

    H_real = shannon_entropy_joint(theta_u, rt_sign, phi_h)

    rt_vals  = rt_sign.copy()
    phi_vals = phi_h.copy()

    H_null = np.empty(n_null, dtype=float)
    for i in range(n_null):
        rt_shuf  = rng.permutation(rt_vals)
        phi_shuf = rng.permutation(phi_vals)
        H_null[i] = shannon_entropy_joint(theta_u, rt_shuf, phi_shuf)

    null_mean = float(np.mean(H_null))
    null_std  = float(np.std(H_null, ddof=1))
    p_deficit = float(np.mean(H_null <= H_real))

    return H_real, null_mean, null_std, p_deficit


# -------------------------------------------------------------
# Test 96 — fluence-limited cuts
# -------------------------------------------------------------

def run_test96(df: pd.DataFrame, n_null: int, seed: int):

    # sort by fluence descending (brightest first)
    df_sorted = df.sort_values("fluence", ascending=False).reset_index(drop=True)
    N_total = len(df_sorted)

    # choose a set of N_keep values
    # (only use those with enough bursts to be meaningful)
    target_keeps = [600, 500, 400, 300, 200, 150, 100]
    results = []

    for N_keep in target_keeps:
        if N_keep > N_total:
            continue
        if N_keep < 50:
            continue

        sub = df_sorted.iloc[:N_keep]

        theta_u = sub["theta_u"].to_numpy(float)
        rt_sign = sub["rt_sign"].to_numpy(float)
        phi_h   = sub["phi_h"].to_numpy(float)

        H_real, null_mean, null_std, p = test91_stat(
            theta_u, rt_sign, phi_h,
            n_null=n_null,
            seed=seed + N_keep
        )

        results.append((N_keep, len(sub), H_real, null_mean, null_std, p))

    return results


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Test 96 — Fluence-limited robustness of the joint entropy deficit (Test 91)"
    )
    p.add_argument("catalog", type=str,
                   help="frbs_unified_for_test91.csv (must include 'fluence')")
    p.add_argument("--n-null", type=int, default=2000,
                   help="number of null shuffles per fluence cut")
    p.add_argument("--seed", type=int, default=1,
                   help="rng seed base")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    df = load_catalog(args.catalog)
    logging.info(f"loaded catalog: {args.catalog}, N={len(df)}")

    results = run_test96(df, args.n_null, args.seed)

    print("=" * 70)
    print("Test 96 — Fluence-limited robustness for joint entropy deficit (Test 91)")
    print("=" * 70)
    print("N_keep   N_used   H_real      null_mean    null_std    p_deficit")
    for N_keep, N_used, H_real, null_mean, null_std, p in results:
        print(f"{N_keep:6d}  {N_used:6d}  {H_real:10.6f}  "
              f"{null_mean:11.6f}  {null_std:10.6f}  {p:10.6f}")
    print("=" * 70)
    print("interpretation:")
    print("  - if p_deficit remains small even for bright-only subsets (small N_keep),")
    print("    the Test 91 joint-entropy deficit is not driven by faint, near-threshold")
    print("    bursts or fluence-dependent selection biases.")
    print("  - if the deficit disappears when restricting to bright bursts, the")
    print("    effect may be tied to survey incompleteness at low fluence.")
    print("=" * 70)


if __name__ == "__main__":
    main()
