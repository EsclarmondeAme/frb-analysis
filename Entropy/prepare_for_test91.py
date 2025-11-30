#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
prepare_for_test91.py

this script prepares the unified FRB catalog for test 91 by computing:

    - theta_u  : angular distance to the unified axis
    - phi_h    : harmonic phase from Y_lm (l <= 8)
    - rt_sign  : sign(cos(phi_h))

definitions follow STRICTLY the logic of test 81C.

pure version:
    - no fallbacks
    - no placeholders
    - no apply_filter
    - no tolerance for missing columns
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
from scipy.special import sph_harm


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
    required = {"ra", "dec"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"catalog missing required columns: {', '.join(sorted(missing))}"
        )
    logging.info(f"loaded catalog: {path}, n = {len(df)}")
    return df


def load_axis(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        ra = float(d["ra_deg"])
        dec = float(d["dec_deg"])
        logging.info(f"loaded axis: ra={ra:.6f}, dec={dec:.6f}")
        return ra, dec
    except Exception as exc:
        raise RuntimeError(
            f"could not load axis json '{path}' ({exc})"
        )


def radec_to_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)  # corrected here
    cosd = np.cos(dec)
    return np.stack([cosd*np.cos(ra), cosd*np.sin(ra), np.sin(dec)], axis=-1)



def compute_theta_u(df: pd.DataFrame, axis_ra: float, axis_dec: float) -> np.ndarray:
    v = radec_to_unit(df["ra"].to_numpy(float), df["dec"].to_numpy(float))
    a = radec_to_unit(np.array([axis_ra]), np.array([axis_dec]))[0]
    dots = np.clip(v @ a, -1.0, 1.0)
    return np.rad2deg(np.arccos(dots))


def compute_phi_h(df: pd.DataFrame) -> np.ndarray:
    """
    compute harmonic phase as in test 81C:

        phi = angle( Σ_{l=1..8} Σ_{m=-l..l} Y_lm )
    """
    ra = np.deg2rad(df["ra"].to_numpy(float))
    dec = np.deg2rad(df["dec"].to_numpy(float))

    theta = np.pi/2 - dec    # colatitude
    phi = ra                 # longitude

    Z = np.zeros(len(df), dtype=complex)

    for l in range(1, 9):
        for m in range(-l, l+1):
            Y = sph_harm(m, l, phi, theta)
            Z += Y

    phase = np.angle(Z)
    return phase


def compute_rt_sign(phi_h: np.ndarray) -> np.ndarray:
    return np.sign(np.cos(phi_h))


# -------------------------------------------------------------
# main
# -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="prepare enhanced FRB catalog for test 91 (theta_u, phi_h, rt_sign)"
    )
    p.add_argument("catalog", type=str, help="frbs_unified.csv input")
    p.add_argument("--axis-json", type=str, required=True,
                   help="json containing unified axis (ra_deg, dec_deg)")
    p.add_argument("--output", type=str, default="frbs_unified_for_test91.csv",
                   help="output csv")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    df = load_catalog(args.catalog)
    axis_ra, axis_dec = load_axis(args.axis_json)

    theta_u = compute_theta_u(df, axis_ra, axis_dec)
    phi_h   = compute_phi_h(df)
    rt_sign = compute_rt_sign(phi_h)

    df["theta_u"] = theta_u
    df["phi_h"]   = phi_h
    df["rt_sign"] = rt_sign

    df.to_csv(args.output, index=False)
    logging.info(f"written enhanced catalog: {args.output}")


if __name__ == "__main__":
    main()
