#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test 91A — RA jackknife robustness for joint entropy deficit (test 91)

for each RA slice, we remove that slice and recompute the
test 91 statistic on the remaining sky:

    H_real_k, null_mean_k, null_std_k, p_deficit_k

pure version:
    - no fallbacks
    - no apply_filter
    - requires frbs_unified_for_test91.csv with:
        ra, theta_u, phi_h, rt_sign
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
    required = {"ra", "theta_u", "phi_h", "rt_sign"}
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

    total = C.sum()
    P = C.astype(float) / total
    P = P[P > 0]
    return float(-np.sum(P * np.log(P)))


def test91_stat(theta_u, rt_sign, phi_h, n_null=2000, seed=1):
    rng = np.random.default_rng(seed)

    # real
    H_real = shannon_entropy_joint(theta_u, rt_sign, phi_h)

    # null: shuffle rt_sign and phi_h
    n = len(theta_u)
    rt_vals  = rt_sign.copy()
    phi_vals = phi_h.copy()

    H_null = np.empty(n_null, dtype=float)
    for i in range(n_null):
        rt_shuf  = rng.permutation(rt_vals)
        phi_shuf = rng.permutation(phi_vals)
        H_null[i] = shannon_entropy_joint(theta_u, rt_shuf, phi_shuf)

    null_mean = float(np.mean(H_null))
    null_std  = float(np.std(H_null, ddof=1)) if n_null > 1 else 0.0
    p_deficit = float(np.mean(H_null <= H_real))

    return H_real, null_mean, null_std, p_deficit


# -------------------------------------------------------------
# jackknife
# -------------------------------------------------------------

def run_jackknife(df: pd.DataFrame, n_slices: int, n_null: int, seed: int):

    ra = df["ra"].to_numpy(float)
    theta_u = df["theta_u"].to_numpy(float)
    phi_h   = df["phi_h"].to_numpy(float)
    rt_sign = df["rt_sign"].to_numpy(float)

    # full-sample reference
    H_full, null_mean_full, null_std_full, p_full = test91_stat(
        theta_u, rt_sign, phi_h,
        n_null=n_null, seed=seed
    )

    results = []

    for k in range(n_slices):
        ra_min = 360.0 * k / n_slices
        ra_max = 360.0 * (k + 1) / n_slices

        mask_keep = ~((ra >= ra_min) & (ra < ra_max))
        n_keep = np.sum(mask_keep)

        if n_keep < 50:
            raise RuntimeError(f"slice {k}: too few FRBs after removal")

        H_k, mean_k, std_k, p_k = test91_stat(
            theta_u[mask_keep],
            rt_sign[mask_keep],
            phi_h[mask_keep],
            n_null=n_null,
            seed=seed + k + 1,
        )

        results.append((k, ra_min, ra_max, n_keep, H_k, mean_k, std_k, p_k))

    return H_full, null_mean_full, null_std_full, p_full, results


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Test 91A — RA jackknife robustness for joint entropy deficit"
    )
    p.add_argument("catalog", type=str,
                   help="enhanced catalog (frbs_unified_for_test91.csv)")
    p.add_argument("--n-slices", type=int, default=10,
                   help="number of RA slices for jackknife (default=10)")
    p.add_argument("--n-null", type=int, default=2000,
                   help="number of null shuffles per jackknife run")
    p.add_argument("--seed", type=int, default=1,
                   help="rng seed base")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    df = load_catalog(args.catalog)

    H_full, mean_full, std_full, p_full, results = run_jackknife(
        df, n_slices=args.n_slices, n_null=args.n_null, seed=args.seed
    )

    print("=" * 60)
    print("Test 91A — RA jackknife robustness for joint entropy deficit")
    print("=" * 60)
    print(f"full-sample H_real   = {H_full:.6f}")
    print(f"full-sample null_mean= {mean_full:.6f}")
    print(f"full-sample null_std = {std_full:.6f}")
    print(f"full-sample p_deficit= {p_full:.6f}")
    print("-" * 60)
    print("per-slice results (removing RA slice [ra_min, ra_max)):")
    print("slice  ra_min  ra_max    N_keep   H_real     null_mean   null_std   p_deficit")
    for k, ra_min, ra_max, n_keep, H_k, mean_k, std_k, p_k in results:
        print(f"{k:5d}  {ra_min:6.1f}  {ra_max:6.1f}  {n_keep:7d}  "
              f"{H_k:8.6f}  {mean_k:10.6f}  {std_k:9.6f}  {p_k:9.6f}")
    print("=" * 60)
    print("interpretation:")
    print("  - if p_deficit remains small across all slices, the joint entropy")
    print("    deficit of Test 91 is not driven by any single RA region.")
    print("  - if removing a particular slice destroys the deficit, that region")
    print("    carries a disproportionate share of the signal.")
    print("=" * 60)


if __name__ == "__main__":
    main()
