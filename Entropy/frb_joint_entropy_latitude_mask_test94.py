#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test 94 — galactic-latitude cut robustness for Test 91

for each galactic-latitude threshold b_cut (e.g. 0°, 10°, 20°, 30°),
select only bursts with |b| >= b_cut and recompute the Test 91
joint-entropy statistic:

    H_real_cut, null_mean_cut, null_std_cut, p_deficit_cut

pure version:
    - no fallbacks
    - no apply_filter
    - requires frbs_unified_for_test91.csv with columns:
        ra, dec, theta_u, phi_h, rt_sign
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
    # require RA/Dec for coordinate transforms
    required = {"ra", "dec", "theta_u", "phi_h", "rt_sign"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"catalog missing required columns: {sorted(missing)}")
    return df


# basic rotation from equatorial (ra,dec) to galactic (l,b)
# (pure, explicit; no astropy)
def radec_to_gal(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    # J2000 rotation matrix (IAU-defined)
    R = np.array([
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [ 0.4941094279, -0.4448296300,  0.7469822445],
        [-0.8676661490, -0.1980763734,  0.4559837762],
    ])

    cosd = np.cos(dec)
    x = cosd * np.cos(ra)
    y = cosd * np.sin(ra)
    z = np.sin(dec)
    v_eq = np.stack([x, y, z], axis=-1)

    v_gal = v_eq @ R.T
    xg, yg, zg = v_gal[:,0], v_gal[:,1], v_gal[:,2]

    b  = np.arcsin(zg)
    l  = np.arctan2(yg, xg)
    l  = np.mod(l, 2*np.pi)

    return np.rad2deg(l), np.rad2deg(b)


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
# test 94: latitude cuts
# -------------------------------------------------------------

def run_test94(df, b_cuts, n_null, seed):

    # convert once
    l, b = radec_to_gal(df["ra"].to_numpy(float),
                        df["dec"].to_numpy(float))

    theta_u = df["theta_u"].to_numpy(float)
    rt_sign = df["rt_sign"].to_numpy(float)
    phi_h   = df["phi_h"].to_numpy(float)

    results = []

    for b_cut in b_cuts:

        mask = np.abs(b) >= b_cut
        n_keep = np.sum(mask)

        if n_keep < 50:
            raise RuntimeError(f"b_cut={b_cut}: too few FRBs after masking")

        H_real, null_mean, null_std, p = test91_stat(
            theta_u[mask], rt_sign[mask], phi_h[mask],
            n_null=n_null, seed=seed + int(b_cut)
        )

        results.append((b_cut, n_keep, H_real, null_mean, null_std, p))

    return results


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Test 94 — galactic-latitude cut robustness for Test 91"
    )
    p.add_argument("catalog", type=str,
                   help="enhanced catalog (frbs_unified_for_test91.csv)")
    p.add_argument("--b-cuts", type=float, nargs="+",
                   default=[0.0, 10.0, 20.0, 30.0],
                   help="list of |b| cuts in degrees")
    p.add_argument("--n-null", type=int, default=2000,
                   help="number of null shuffles per mask")
    p.add_argument("--seed", type=int, default=1,
                   help="rng seed base")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    df = load_catalog(args.catalog)
    results = run_test94(df, args.b_cuts, args.n_null, args.seed)

    print("="*60)
    print("Test 94 — galactic-latitude cut robustness for Test 91")
    print("="*60)
    print("b_cut   N_keep    H_real      null_mean    null_std    p_deficit")
    for b_cut, n_keep, H_real, null_mean, null_std, p in results:
        print(f"{b_cut:5.1f}  {n_keep:7d}  {H_real:10.6f}  "
              f"{null_mean:11.6f}  {null_std:10.6f}  {p:10.6f}")
    print("="*60)
    print("interpretation:")
    print("  - if p_deficit remains small for all |b| cuts, the Test 91")
    print("    joint-entropy deficit is not confined to low-latitude or")
    print("    high-latitude sky regions.")
    print("="*60)


if __name__ == "__main__":
    main()
