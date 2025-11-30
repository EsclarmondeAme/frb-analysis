#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test 92 — axis-rotation robustness of the joint entropy deficit (pure version)

this test keeps the canonical field (phi_h, rt_sign) fixed as in test 91
and only randomizes the ORIENTATION of the unified axis relative to
the fixed FRB sky.

for each random rotation of the axis, we:
    - recompute theta_u^rot = angular distance to the rotated axis
    - compute the joint entropy H(theta_u^rot, rt_sign, phi_h)
the distribution of H_rot is then compared qualitatively to H_real from test 91.

pure version:
    - no fallbacks
    - no apply_filter
    - requires frbs_unified_for_test91.csv with columns:
        ra, dec, theta_u, phi_h, rt_sign
"""

import argparse
import json
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
    required = {"ra", "dec", "theta_u", "phi_h", "rt_sign"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"catalog missing required columns: {sorted(missing)}")
    return df


def load_axis(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return float(d["ra_deg"]), float(d["dec_deg"])
    except Exception as exc:
        raise RuntimeError(f"cannot load axis json '{path}' ({exc})")


def radec_to_vec(ra_deg, dec_deg):
    ra  = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cosd = np.cos(dec)
    x = cosd * np.cos(ra)
    y = cosd * np.sin(ra)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=-1)


def random_rotation_matrix(rng):
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1-u1) * np.sin(2*np.pi*u2)
    q2 = np.sqrt(1-u1) * np.cos(2*np.pi*u2)
    q3 = np.sqrt(u1)   * np.sin(2*np.pi*u3)
    q4 = np.sqrt(u1)   * np.cos(2*np.pi*u3)
    q1, q2, q3, q4 = q1, q2, q3, q4
    R = np.array([
        [1-2*(q2*q2+q3*q3),   2*(q1*q2 - q3*q4),   2*(q1*q3 + q2*q4)],
        [2*(q1*q2 + q3*q4),   1-2*(q1*q1+q3*q3),   2*(q2*q3 - q1*q4)],
        [2*(q1*q3 - q2*q4),   2*(q2*q3 + q1*q4),   1-2*(q1*q1+q2*q2)]
    ])
    return R


def bin_var(x, edges):
    idx = np.digitize(x, edges) - 1
    idx[(x < edges[0]) | (x >= edges[-1])] = -1
    return idx


def joint_entropy(theta_u, rt_sign, phi_h):
    theta_edges = np.array([0, 20, 35, 50, 90, 180])
    rt_edges    = np.array([-2, 0, 2])
    phi_edges   = np.linspace(0, 2*np.pi, 12+1)

    # wrap phase exactly as in test 91
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

    P = C / C.sum()
    P = P[P > 0]
    return float(-np.sum(P * np.log(P)))


# -------------------------------------------------------------
# core test
# -------------------------------------------------------------

def run_axis_rotation_test(df, axis_ra, axis_dec, n_rot, seed):

    rng = np.random.default_rng(seed)

    # FRB positions fixed
    ra = df["ra"].to_numpy(float)
    dec = df["dec"].to_numpy(float)
    frb_vec = radec_to_vec(ra, dec)

    # canonical field from test 81C / 91
    phi_h   = df["phi_h"].to_numpy(float)
    rt_sign = df["rt_sign"].to_numpy(float)

    # original axis vector
    axis_vec0 = radec_to_vec(
        np.array([axis_ra], dtype=float),
        np.array([axis_dec], dtype=float)
    )[0]

    H_rot = np.empty(n_rot, dtype=float)

    for i in range(n_rot):
        R = random_rotation_matrix(rng)
        axis_vec_rot = R @ axis_vec0
        # cosine of angle between each FRB and rotated axis
        dots = np.clip(frb_vec @ axis_vec_rot, -1.0, 1.0)
        theta_u_rot = np.rad2deg(np.arccos(dots))

        H_rot[i] = joint_entropy(theta_u_rot, rt_sign, phi_h)

    return H_rot


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Test 92 — axis-rotation robustness of joint entropy"
    )
    p.add_argument("catalog", type=str,
                   help="enhanced catalog (frbs_unified_for_test91.csv)")
    p.add_argument("--axis-json", type=str, required=True,
                   help="unified axis json (ra_deg, dec_deg)")
    p.add_argument("--n-rot", type=int, default=500,
                   help="number of random axis rotations")
    p.add_argument("--seed", type=int, default=1,
                   help="rng seed")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    df = load_catalog(args.catalog)
    axis_ra, axis_dec = load_axis(args.axis_json)

    logging.info("running axis-rotation null test...")
    H_rot = run_axis_rotation_test(df, axis_ra, axis_dec,
                                   n_rot=args.n_rot, seed=args.seed)

    logging.info("axis-rotation results:")
    logging.info(f"mean(H_rot) = {H_rot.mean():.6f}")
    logging.info(f"std(H_rot)  = {H_rot.std(ddof=1):.6f}")

    print("="*60)
    print("Test 92 — axis-rotation robustness of joint entropy")
    print("="*60)
    print(f"mean(H_rot)  = {H_rot.mean():.6f}")
    print(f"std(H_rot)   = {H_rot.std(ddof=1):.6f}")
    print(f"min(H_rot)   = {H_rot.min():.6f}")
    print(f"max(H_rot)   = {H_rot.max():.6f}")
    print("="*60)
    print("interpretation:")
    print("  - if H_real from Test 91 is an extreme low-entropy outlier")
    print("    relative to this H_rot distribution, the deficit is")
    print("    locked to the true axis orientation.")
    print("  - if many rotations reach similarly low entropy, the")
    print("    deficit is not uniquely tied to the unified axis.")
    print("="*60)


if __name__ == "__main__":
    main()
