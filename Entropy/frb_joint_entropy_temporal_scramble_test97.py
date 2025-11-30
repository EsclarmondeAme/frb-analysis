#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test 97 — Temporal–Scrambling Robustness of the Joint Entropy Deficit (Test 91)

Purpose:
    Determine whether the Test 91 signal depends on the actual observation
    times of the FRBs, or whether it persists when those times are randomly
    permuted. If the deficit is due to survey scheduling or time–window
    structure, temporal scrambling should erase it. If the deficit is tied
    to geometry and intrinsic field structure, the real sample should remain
    an extreme outlier.

Requirements:
    - frbs_unified_for_test91.csv must contain:
          theta_u, phi_h, rt_sign, time_column
      where time_column is the observation timestamp used for remnant-time.

Pure version:
    - no fallbacks
    - no apply_filter
"""

import argparse
import logging
import numpy as np
import pandas as pd


# -------------------------------------------------------------
# utilities
# -------------------------------------------------------------

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")


def load_catalog(path: str, time_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"theta_u", "phi_h", "rt_sign", time_column}
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

    bT = bin_var(theta_u, theta_edges)
    bR = bin_var(rt_sign, rt_edges)
    bP = bin_var(phi_h,  phi_edges)

    ok = (bT >= 0) & (bR >= 0) & (bP >= 0)
    if not np.any(ok):
        raise RuntimeError("no points fall into joint bins")

    C = np.zeros((5,2,12), dtype=int)
    for t, r, p in zip(bT[ok], bR[ok], bP[ok]):
        C[t,r,p] += 1

    P = C[C>0].astype(float) / np.sum(C)
    return -np.sum(P * np.log(P))


def test91_core(theta_u, rt_sign, phi_h, n_null=2000, seed=1):
    rng = np.random.default_rng(seed)

    H_real = shannon_entropy_joint(theta_u, rt_sign, phi_h)

    rt_vals  = rt_sign.copy()
    phi_vals = phi_h.copy()

    H_null = np.empty(n_null)
    for i in range(n_null):
        rt_s  = rng.permutation(rt_vals)
        phi_s = rng.permutation(phi_vals)
        H_null[i] = shannon_entropy_joint(theta_u, rt_s, phi_s)

    return (H_real,
            float(np.mean(H_null)),
            float(np.std(H_null, ddof=1)),
            float(np.mean(H_null <= H_real)))


# -------------------------------------------------------------
# Test 97 — temporal scrambling
# -------------------------------------------------------------

def run_test97(df: pd.DataFrame, time_column: str, n_scramble: int,
               n_null: int, seed: int):

    rng = np.random.default_rng(seed)

    theta_u = df["theta_u"].to_numpy(float)
    phi_h   = df["phi_h"].to_numpy(float)
    rt_real = df["rt_sign"].to_numpy(float)

    # real-sample statistic
    H_real, null_mean, null_std, p_real = test91_core(
        theta_u, rt_real, phi_h, n_null=n_null, seed=seed
    )

    # temporal-scramble ensemble
    H_scr  = np.empty(n_scramble)
    p_scr  = np.empty(n_scramble)

    for i in range(n_scramble):
        # scramble times
        time_shuffled = rng.permutation(df[time_column].to_numpy(float))

        # recompute remnant-time sign from time ordering relative to median
        median_t = np.median(time_shuffled)
        rt_s = np.where(time_shuffled >= median_t, +1.0, -1.0)

        # compute entropy deficit for the scrambled sample
        Hs, nm, ns, ps = test91_core(
            theta_u, rt_s, phi_h,
            n_null=n_null,
            seed=seed + 1000 + i
        )

        H_scr[i] = Hs
        p_scr[i] = ps

    return (H_real, null_mean, null_std, p_real,
            H_scr, p_scr)


def parse_args():
    p = argparse.ArgumentParser(
        description="Test 97 — Temporal–Scrambling Robustness for Test 91"
    )
    p.add_argument("catalog", type=str,
                   help="frbs_unified_for_test91.csv")
    p.add_argument("--time-column", type=str, default="mjd",
                   help="column containing observation times")
    p.add_argument("--n-scramble", type=int, default=500,
                   help="number of temporal-scramble realisations")
    p.add_argument("--n-null", type=int, default=2000,
                   help="null shuffles for each statistic")
    p.add_argument("--seed", type=int, default=1,
                   help="rng seed base")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    df = load_catalog(args.catalog, args.time_column)
    logging.info(f"loaded catalog: {args.catalog}, N={len(df)}")
    logging.info(f"time column used: {args.time_column}")

    (H_real, null_mean, null_std, p_real,
     H_scr, p_scr) = run_test97(
        df, args.time_column, args.n_scramble,
        args.n_null, args.seed
    )

    print("="*70)
    print("Test 97 — Temporal–Scrambling Robustness for Joint Entropy Deficit")
    print("="*70)
    print(f"real-sample:")
    print(f"  H_real      = {H_real:.6f}")
    print(f"  null_mean   = {null_mean:.6f}")
    print(f"  null_std    = {null_std:.6f}")
    print(f"  p_deficit   = {p_real:.6f}")
    print("-"*70)
    print(f"temporal-scramble ensemble (N={len(H_scr)}):")
    print(f"  mean(H_scr) = {np.mean(H_scr):.6f}")
    print(f"  std(H_scr)  = {np.std(H_scr, ddof=1):.6f}")
    print(f"  mean(p_scr) = {np.mean(p_scr):.6f}")
    print("="*70)
    print("interpretation:")
    print("  - If real-sample p_deficit is far smaller than typical temporal-scramble")
    print("    values, the joint entropy deficit is not tied to observational")
    print("    scheduling or time-window selection. It depends on geometric and")
    print("    phase-remnant structure, not on telescope uptime.")
    print("="*70)


if __name__ == "__main__":
    main()
