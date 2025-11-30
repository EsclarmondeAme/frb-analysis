#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test 91 — pure joint-entropy deficit test

variables used:
    - theta_u      (angular distance to unified axis, degrees)
    - rt_sign      (remnant-time sign: +1 or -1)
    - phi_h        (harmonic phase angle in radians, typically from Y_lm, l<=8)

goal:
    measure the 3D joint entropy H(theta_u_bin, rt_sign_bin, phi_h_bin)
    and compare to null models that preserve:
        - theta_u distribution      (same bins, same counts)
        - rt_sign distribution      (same number of + and -)
        - phi_h global distribution (same histogram)

null randomizations:
    - shuffle rt_sign among bursts
    - shuffle phi_h among bursts
    - keep theta_u bins fixed

this tests whether the *combination* of geometry + remnant-time + phase
is in an unusually ordered state that cannot arise from independent fields.

strict version:
    - no fallbacks
    - no apply_filter
    - no alternate metrics
    - axis must be provided (for theta_u)
    - theta_u must exist or be computed directly
    - harmonic phase must be provided as phi_h
    - remnant-time sign must be provided as rt_sign
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# utilities
# ----------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )


def load_catalog(path: str) -> pd.DataFrame:
    logging.info(f"loading catalog: {path}")
    df = pd.read_csv(path)

    required = {"ra", "dec", "theta_u", "rt_sign", "phi_h"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"catalog missing required columns: {', '.join(sorted(missing))}"
        )

    return df


def load_axis(path: str):
    """
    json must contain:
        {
            "ra_deg": float,
            "dec_deg": float
        }
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        ra = float(d["ra_deg"])
        dec = float(d["dec_deg"])
        logging.info(f"loaded unified axis: ra={ra:.6f}, dec={dec:.6f}")
        return ra, dec
    except Exception as exc:
        raise RuntimeError(
            f"could not load axis json '{path}' ({exc})"
        )


def bin_variable(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    simple bin indexer for 1D variable.
    result: integer bin index or -1 if outside.
    """
    idx = np.digitize(x, edges, right=False) - 1
    idx[(x < edges[0]) | (x >= edges[-1])] = -1
    return idx


def joint_entropy(joint_counts: np.ndarray) -> float:
    """
    shannon entropy of a 3D joint distribution.
    """
    total = joint_counts.sum()
    if total <= 0:
        raise RuntimeError("joint count array is empty or zero.")

    p = joint_counts.astype(float) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ----------------------------------------------------------------------
# core test logic
# ----------------------------------------------------------------------

def run_test91(df: pd.DataFrame, n_null: int, seed: int):

    rng = np.random.default_rng(seed)

    # extract variables
    theta_u = df["theta_u"].to_numpy(float)
    rt = df["rt_sign"].to_numpy(float)
    phi = df["phi_h"].to_numpy(float)

    # -------------- bin definitions (fixed) -----------------------------
    # theta_u: axis-distance structure is strongest in 20–40° ranges
    theta_edges = np.array([0, 20, 35, 50, 90, 180], dtype=float)

    # remnant-time sign: 2 bins { -1 , +1 }
    rt_edges = np.array([-2, 0, 2], dtype=float)  # bin: [-2,0) = -1, [0,2) = +1

    # phi_h: wrap to [0, 2π)
    phi = np.mod(phi, 2 * np.pi)
    n_phi_bins = 12
    phi_edges = np.linspace(0, 2 * np.pi, n_phi_bins + 1)

    # -------------- assign bins ---------------------------------------
    b_theta = bin_variable(theta_u, theta_edges)
    b_rt = bin_variable(rt, rt_edges)
    b_phi = bin_variable(phi, phi_edges)

    valid = (b_theta >= 0) & (b_rt >= 0) & (b_phi >= 0)
    if not np.any(valid):
        raise RuntimeError("no FRBs fall into the joint bin grid.")

    b_theta = b_theta[valid]
    b_rt = b_rt[valid]
    b_phi = b_phi[valid]

    Nθ = len(theta_edges) - 1
    Nrt = 2
    Nφ = n_phi_bins

    # -------------- real joint histogram ------------------------------
    joint_real = np.zeros((Nθ, Nrt, Nφ), dtype=int)
    for t, r, p in zip(b_theta, b_rt, b_phi):
        joint_real[t, r, p] += 1

    H_real = joint_entropy(joint_real)
    logging.info(f"H_real = {H_real:.6f}")

    # -------------- null ensemble -------------------------------------
    # preserve:
    #   - theta bins    (geometry)
    #   - rt sign count (global)
    #   - phi histogram (global)
    #
    # randomize:
    #   - rt signs permuted independently
    #   - phi phases permuted independently

    rt_vals = rt[valid]
    phi_vals = phi[valid]

    null_H = np.empty(n_null, dtype=float)

    for i in range(n_null):
        rt_shuf = rng.permutation(rt_vals)
        phi_shuf = rng.permutation(phi_vals)

        # assign bins for shuffled rt, phi
        b_rt_s = bin_variable(rt_shuf, rt_edges)
        b_phi_s = bin_variable(phi_shuf, phi_edges)

        joint = np.zeros((Nθ, Nrt, Nφ), dtype=int)
        for t, r, p in zip(b_theta, b_rt_s, b_phi_s):
            joint[t, r, p] += 1

        null_H[i] = joint_entropy(joint)

    null_mean = float(np.mean(null_H))
    null_std = float(np.std(null_H, ddof=1)) if n_null > 1 else 0.0

    # deficit: real is smaller than null distribution
    p_deficit = float(np.mean(null_H <= H_real))

    return H_real, null_mean, null_std, p_deficit, Nθ, Nrt, Nφ


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="pure joint-entropy deficit test (theta_u, rt_sign, phi_h)"
    )
    p.add_argument("catalog", type=str, help="FRB unified catalog CSV")
    p.add_argument("--axis-json", type=str, required=True,
                   help="axis JSON (ra_deg, dec_deg), required for consistency")
    p.add_argument("--n-null", type=int, default=5000,
                   help="number of null shuffles")
    p.add_argument("--seed", type=int, default=1,
                   help="rng seed")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    # load catalog and axis
    df = load_catalog(args.catalog)
    load_axis(args.axis_json)  # only to enforce presence; theta_u must already exist

    H_real, null_mean, null_std, p_deficit, Nθ, Nrt, Nφ = run_test91(
        df=df,
        n_null=args.n_null,
        seed=args.seed
    )

    print("=" * 60)
    print("test 91 — joint entropy deficit (pure version)")
    print("=" * 60)
    print(f"H_real        = {H_real:.6f}")
    print(f"null_mean     = {null_mean:.6f}")
    print(f"null_std      = {null_std:.6f}")
    print(f"p_deficit     = {p_deficit:.6f}")
    print(f"bins: theta={Nθ}, rt={Nrt}, phi={Nφ}")
    print("=" * 60)
    print("interpretation:")
    print("  - p_deficit << 0.05 → joint field unusually ordered")
    print("  - p_deficit ~ 0.5   → consistent with shuffled independent fields")
    print("=" * 60)


if __name__ == "__main__":
    main()

