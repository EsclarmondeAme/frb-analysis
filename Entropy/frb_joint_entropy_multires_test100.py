#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test 100 — Multi-Resolution Binning Robustness for the Joint Entropy Deficit (Test 91)

Purpose:
    Evaluate whether the Test 91 result depends on the specific choice of
    binning resolution in (theta_u, rt_sign, phi_h). We test a grid of
    resolutions and repeat the full entropy/null calculation for each.

    If the entropy deficit persists across all bin choices, it is not a
    binning artifact but a genuine structural feature.

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


def load_catalog(path):
    df = pd.read_csv(path)
    required = {"theta_u", "phi_h", "rt_sign"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"catalog missing required columns: {sorted(missing)}")
    return df


def bin_var(x, edges):
    idx = np.digitize(x, edges) - 1
    idx[(x < edges[0]) | (x >= edges[-1])] = -1
    return idx


def compute_entropy(theta_u, rt_sign, phi_h, n_theta, n_phi):
    # theta bins from 0–180 linearly
    theta_edges = np.linspace(0, 180, n_theta + 1)

    # rt_sign has exactly 2 bins: -1, +1
    rt_edges = np.array([-2, 0, 2])

    # phi bins over [0, 2π)
    phi_edges = np.linspace(0, 2*np.pi, n_phi + 1)

    phi = np.mod(phi_h, 2*np.pi)

    bT = bin_var(theta_u, theta_edges)
    bR = bin_var(rt_sign, rt_edges)
    bP = bin_var(phi,       phi_edges)

    ok = (bT>=0)&(bR>=0)&(bP>=0)
    if not np.any(ok):
        return np.nan

    C = np.zeros((n_theta, 2, n_phi), dtype=int)
    for t, r, p in zip(bT[ok], bR[ok], bP[ok]):
        C[t,r,p] += 1

    P = C[C>0].astype(float) / np.sum(C)
    return -np.sum(P * np.log(P))


def test91_generic(theta_u, rt_sign, phi_h, n_theta, n_phi, n_null, seed):
    rng = np.random.default_rng(seed)

    H_real = compute_entropy(theta_u, rt_sign, phi_h, n_theta, n_phi)

    rt_vals  = rt_sign.copy()
    phi_vals = phi_h.copy()

    H_null = np.empty(n_null)
    for i in range(n_null):
        rt_s  = rng.permutation(rt_vals)
        phi_s = rng.permutation(phi_vals)
        H_null[i] = compute_entropy(theta_u, rt_s, phi_s, n_theta, n_phi)

    return (H_real,
            float(np.mean(H_null)),
            float(np.std(H_null, ddof=1)),
            float(np.mean(H_null <= H_real)))


def run_test100(df, theta_grid, phi_grid, n_null, seed):

    theta_u = df["theta_u"].to_numpy(float)
    rt_sign = df["rt_sign"].to_numpy(float)
    phi_h   = df["phi_h"].to_numpy(float)

    results = []

    for i, n_theta in enumerate(theta_grid):
        for j, n_phi in enumerate(phi_grid):
            H, m, s, p = test91_generic(
                theta_u, rt_sign, phi_h,
                n_theta=n_theta,
                n_phi=n_phi,
                n_null=n_null,
                seed=seed + 1000*i + 10*j
            )

            results.append((n_theta, n_phi, H, m, s, p))

    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="Test 100 — Multi-Resolution Binning Robustness"
    )
    p.add_argument("catalog", type=str,
                   help="frbs_unified_for_test91.csv")
    p.add_argument("--n-null", type=int, default=2000,
                   help="null shuffles per configuration")
    p.add_argument("--seed", type=int, default=1,
                   help="rng base seed")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    df = load_catalog(args.catalog)
    logging.info(f"loaded catalog: {args.catalog}, N={len(df)}")

    # range of bin resolutions to test
    theta_grid = [4, 5, 6, 7]
    phi_grid   = [8, 12, 16, 24]

    results = run_test100(df, theta_grid, phi_grid, args.n_null, args.seed)

    print("="*80)
    print("Test 100 — Multi-Resolution Binning Robustness")
    print("="*80)
    print("n_theta  n_phi      H_real      null_mean    null_std    p_deficit")
    print("-"*80)
    for nT, nP, H, m, s, p in results:
        print(f"{nT:7d} {nP:6d}  {H:10.6f}  {m:11.6f}  {s:10.6f}  {p:10.6f}")
    print("="*80)
    print("interpretation:")
    print("  - If the entropy deficit persists across binning resolutions,")
    print("    it is not a bin-artifact and the structure is scale-invariant.")
    print("="*80)


if __name__ == "__main__":
    main()
