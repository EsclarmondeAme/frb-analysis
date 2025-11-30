#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test 90 — pure energy–configuration entropy deficit test

this test measures whether FRB fluences occupy a statistically
low–entropy configuration relative to the unified axis geometry.

strict version:
    - no fallbacks
    - no apply_filter
    - no placeholders
    - no alternative modes
    - axis is required
    - fluence is required
    - RA/DEC are required
    - deterministic shell definition
    - simple maximum–entropy null via fluence permutation
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# utilities
# ----------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )


def load_catalog(path: str) -> pd.DataFrame:
    logging.info(f"loading catalog: {path}")
    df = pd.read_csv(path)

    required = {"ra", "dec", "fluence"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"catalog missing required columns: {', '.join(missing)}"
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
            data = json.load(f)
        ra = float(data["ra_deg"])
        dec = float(data["dec_deg"])
        logging.info(
            f"loaded unified axis: ra={ra:.6f} deg, dec={dec:.6f} deg"
        )
        return ra, dec
    except Exception as exc:
        raise RuntimeError(
            f"could not load axis json '{path}' ({exc})"
        )


def radec_to_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cosd = np.cos(dec)
    x = cosd * np.cos(ra)
    y = cosd * np.sin(ra)
    z = np.sin(dec)
    return np.stack((x, y, z), axis=-1)


def angular_distance_to_axis(ra: np.ndarray, dec: np.ndarray,
                             axis_ra: float, axis_dec: float) -> np.ndarray:
    v = radec_to_unit(ra, dec)
    a = radec_to_unit(
        np.array([axis_ra]),
        np.array([axis_dec])
    )[0]
    dots = np.clip(v @ a, -1.0, 1.0)
    theta = np.rad2deg(np.arccos(dots))
    return theta


def assign_shells(theta: np.ndarray) -> np.ndarray:
    """
    fixed shell edges in degrees.
    """
    edges = np.array([0, 20, 35, 50, 90, 180], dtype=float)
    idx = np.digitize(theta, edges, right=False) - 1
    # mark invalid as -1
    idx[(theta < edges[0]) | (theta >= edges[-1])] = -1
    return idx, edges


def shell_entropy(fluence: np.ndarray, shells: np.ndarray, n_shells: int) -> float:
    energy_per_shell = np.bincount(shells, weights=fluence, minlength=n_shells)
    total = energy_per_shell.sum()
    if total <= 0:
        raise RuntimeError("total fluence is zero; cannot compute entropy.")
    p = energy_per_shell / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ----------------------------------------------------------------------
# main test logic
# ----------------------------------------------------------------------

def run_test90(df: pd.DataFrame, axis_ra: float, axis_dec: float,
               n_null: int, seed: int):

    rng = np.random.default_rng(seed)

    # compute theta_u
    theta_u = angular_distance_to_axis(
        df["ra"].to_numpy(float),
        df["dec"].to_numpy(float),
        axis_ra,
        axis_dec
    )

    # assign shells
    shell_idx, shell_edges = assign_shells(theta_u)
    valid = shell_idx >= 0
    if not np.any(valid):
        raise RuntimeError("no FRBs fall into the defined shells.")

    shell_idx = shell_idx[valid]
    flu = df["fluence"].to_numpy(float)[valid]

    n_shells = int(shell_idx.max()) + 1

    # real entropy
    H_real = shell_entropy(flu, shell_idx, n_shells)
    logging.info(f"H_real = {H_real:.6f}")
    logging.info(f"n_shells = {n_shells}")

    # null distribution
    H_null = np.empty(n_null, dtype=float)
    for i in range(n_null):
        shuffled = rng.permutation(flu)
        H_null[i] = shell_entropy(shuffled, shell_idx, n_shells)

    null_mean = float(np.mean(H_null))
    null_std = float(np.std(H_null, ddof=1)) if n_null > 1 else 0.0
    p_deficit = float(np.mean(H_null <= H_real))

    return H_real, null_mean, null_std, p_deficit, n_shells


# ----------------------------------------------------------------------
# cli
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="pure FRB energy–configuration entropy deficit test (test 90)"
    )
    p.add_argument("catalog", type=str, help="path to FRB unified CSV")
    p.add_argument("--axis-json", type=str, required=True,
                   help="json containing unified axis (ra_deg, dec_deg)")
    p.add_argument("--n-null", type=int, default=5000,
                   help="number of null realisations (default=5000)")
    p.add_argument("--seed", type=int, default=1,
                   help="rng seed (default=1)")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    df = load_catalog(args.catalog)
    axis_ra, axis_dec = load_axis(args.axis_json)

    H_real, null_mean, null_std, p_deficit, n_shells = run_test90(
        df=df,
        axis_ra=axis_ra,
        axis_dec=axis_dec,
        n_null=args.n_null,
        seed=args.seed
    )

    print("=" * 60)
    print("test 90 — energy entropy deficit (pure version)")
    print("=" * 60)
    print(f"H_real        = {H_real:.6f}")
    print(f"null_mean     = {null_mean:.6f}")
    print(f"null_std      = {null_std:.6f}")
    print(f"p_deficit     = {p_deficit:.6f}")
    print(f"n_shells_used = {n_shells}")
    print("=" * 60)
    print("interpretation (not printed in paper):")
    print("  - p_deficit << 0.05 → statistically low entropy")
    print("  - p_deficit ~ 0.5   → consistent with randomised energy field")
    print("=" * 60)


if __name__ == "__main__":
    main()
