#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test 93 — conditional entropy and mutual information decomposition
for the (theta_u, rt_sign, phi_h) field.

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
    required = {"theta_u", "phi_h", "rt_sign"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"catalog missing required columns: {sorted(missing)}")
    return df


def bin_var(x, edges):
    idx = np.digitize(x, edges) - 1
    idx[(x < edges[0]) | (x >= edges[-1])] = -1
    return idx


def entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        raise RuntimeError("empty counts in entropy_from_counts")
    p = counts.astype(float) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# -------------------------------------------------------------
# core decomposition
# -------------------------------------------------------------

def run_test93(df: pd.DataFrame):

    theta = df["theta_u"].to_numpy(float)
    rt    = df["rt_sign"].to_numpy(float)
    phi   = df["phi_h"].to_numpy(float)

    # binning identical to test 91
    theta_edges = np.array([0, 20, 35, 50, 90, 180])
    rt_edges    = np.array([-2, 0, 2])
    phi_edges   = np.linspace(0, 2*np.pi, 12 + 1)

    phi = np.mod(phi, 2*np.pi)

    b_theta = bin_var(theta, theta_edges)
    b_rt    = bin_var(rt,    rt_edges)
    b_phi   = bin_var(phi,   phi_edges)

    valid = (b_theta >= 0) & (b_rt >= 0) & (b_phi >= 0)
    if not np.any(valid):
        raise RuntimeError("no points in joint bins")

    bt = b_theta[valid]
    br = b_rt[valid]
    bp = b_phi[valid]

    Nθ = len(theta_edges) - 1   # 5
    Nrt = len(rt_edges) - 1     # 2
    Nφ = len(phi_edges) - 1     # 12

    # joint counts
    C_trp = np.zeros((Nθ, Nrt, Nφ), dtype=int)
    for t, r, p in zip(bt, br, bp):
        C_trp[t, r, p] += 1

    # pairwise and marginal counts
    C_theta = C_trp.sum(axis=(1, 2))
    C_rt    = C_trp.sum(axis=(0, 2))
    C_phi   = C_trp.sum(axis=(0, 1))

    C_theta_rt  = C_trp.sum(axis=2)
    C_theta_phi = C_trp.sum(axis=1)
    C_rt_phi    = C_trp.sum(axis=0)

    # entropies
    H_trp       = entropy_from_counts(C_trp)
    H_theta     = entropy_from_counts(C_theta)
    H_rt        = entropy_from_counts(C_rt)
    H_phi       = entropy_from_counts(C_phi)
    H_theta_rt  = entropy_from_counts(C_theta_rt)
    H_theta_phi = entropy_from_counts(C_theta_phi)
    H_rt_phi    = entropy_from_counts(C_rt_phi)

    # mutual informations
    I_theta_rt  = H_theta + H_rt  - H_theta_rt
    I_theta_phi = H_theta + H_phi - H_theta_phi
    I_rt_phi    = H_rt    + H_phi - H_rt_phi

    # total correlation (multi-information)
    T_total = H_theta + H_rt + H_phi - H_trp

    # conditional entropies
    H_rt_phi_given_theta = H_trp - H_theta
    H_theta_rt_given_phi = H_trp - H_phi
    H_theta_phi_given_rt = H_trp - H_rt

    return {
        "H_trp": H_trp,
        "H_theta": H_theta,
        "H_rt": H_rt,
        "H_phi": H_phi,
        "H_theta_rt": H_theta_rt,
        "H_theta_phi": H_theta_phi,
        "H_rt_phi": H_rt_phi,
        "I_theta_rt": I_theta_rt,
        "I_theta_phi": I_theta_phi,
        "I_rt_phi": I_rt_phi,
        "T_total": T_total,
        "H_rt_phi_given_theta": H_rt_phi_given_theta,
        "H_theta_rt_given_phi": H_theta_rt_given_phi,
        "H_theta_phi_given_rt": H_theta_phi_given_rt,
        "Nθ": Nθ,
        "Nrt": Nrt,
        "Nφ": Nφ,
    }


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Test 93 — conditional entropy and mutual information of (theta_u, rt_sign, phi_h)"
    )
    p.add_argument("catalog", type=str,
                   help="enhanced catalog (frbs_unified_for_test91.csv)")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    df = load_catalog(args.catalog)
    res = run_test93(df)

    print("=" * 60)
    print("Test 93 — conditional entropy and mutual information")
    print("=" * 60)
    print(f"H(theta, rt, phi)      = {res['H_trp']:.6f}")
    print(f"H(theta)               = {res['H_theta']:.6f}")
    print(f"H(rt)                  = {res['H_rt']:.6f}")
    print(f"H(phi)                 = {res['H_phi']:.6f}")
    print("-" * 60)
    print(f"H(theta, rt)           = {res['H_theta_rt']:.6f}")
    print(f"H(theta, phi)          = {res['H_theta_phi']:.6f}")
    print(f"H(rt, phi)             = {res['H_rt_phi']:.6f}")
    print("-" * 60)
    print(f"I(theta; rt)           = {res['I_theta_rt']:.6f}")
    print(f"I(theta; phi)          = {res['I_theta_phi']:.6f}")
    print(f"I(rt; phi)             = {res['I_rt_phi']:.6f}")
    print(f"Total correlation T    = {res['T_total']:.6f}")
    print("-" * 60)
    print(f"H(rt, phi | theta)     = {res['H_rt_phi_given_theta']:.6f}")
    print(f"H(theta, rt | phi)     = {res['H_theta_rt_given_phi']:.6f}")
    print(f"H(theta, phi | rt)     = {res['H_theta_phi_given_rt']:.6f}")
    print("-" * 60)
    print(f"bins: theta={res['Nθ']}, rt={res['Nrt']}, phi={res['Nφ']}")
    print("=" * 60)
    print("interpretation (qualitative):")
    print("  - large I(rt; phi)     → strong coupling between remnant-time sign and phase.")
    print("  - large I(theta; rt)   → remnant-time polarity depends on axis distance.")
    print("  - large I(theta; phi)  → phase depends on axis distance.")
    print("  - large T_total        → strong three-way dependence beyond marginals.")
    print("=" * 60)


if __name__ == "__main__":
    main()
