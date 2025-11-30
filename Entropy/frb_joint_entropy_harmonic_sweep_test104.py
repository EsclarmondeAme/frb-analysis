#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Test 104 — Harmonic–Order Sweep Robustness for Joint Entropy Deficit
#
# Uses the same definitions of:
# - theta_u       (axis distance)
# - phi_h         (harmonic phase)
# - rt_sign       (remnant-time sign)
#
# as Test 81C + Test 91.
#
# For ℓ = 1..12:
#   1. Construct φ_h(ℓ) using sph_harm up to given ℓ.
#   2. Compute joint entropy H(θ_u, rt_sign, φ_h(ℓ)).
#   3. Build isotropic Monte-Carlo null as in Test 91.
#   4. Report H_real, null_mean, null_std, p_deficit.
#
# This determines whether the anomaly is tied to a specific harmonic order
# or is scale-invariant.

import numpy as np
import pandas as pd
import argparse
from scipy.special import sph_harm
from math import log
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# -----------------------------
# entropy helpers
# -----------------------------

def joint_entropy_3d(theta_bin, rt_bin, phi_bin):
    H = 0.0
    N = len(theta_bin)
    for t, r, p in zip(theta_bin, rt_bin, phi_bin):
        pass
    # faster: compute full histogram
    hist, _ = np.histogramdd(
        np.vstack([theta_bin, rt_bin, phi_bin]).T,
        bins=(np.unique(theta_bin).size,
              np.unique(rt_bin).size,
              np.unique(phi_bin).size)
    )
    hist = hist / np.sum(hist)
    nz = hist[hist > 0]
    H = -np.sum(nz * np.log(nz))
    return H

def monte_carlo_null(theta, rt, phi, n_null, nb_theta, nb_rt, nb_phi):
    H_null = []
    N = len(theta)
    for _ in range(n_null):
        rt_s = np.random.permutation(rt)
        phi_s = np.random.permutation(phi)
        Hs = joint_entropy_3d(
            np.digitize(theta, np.linspace(0, np.pi, nb_theta+1)) - 1,
            np.digitize(rt_s, [-1, 0, 1]) - 1,
            np.digitize(phi_s, np.linspace(0, 2*np.pi, nb_phi+1)) - 1
        )
        H_null.append(Hs)
    H_null = np.array(H_null)
    return H_null.mean(), H_null.std()

# -----------------------------
# harmonic phase constructor
# -----------------------------

def compute_phi_h_l(df, ell_max):
    ra  = np.deg2rad(df["ra"].to_numpy())
    dec = np.deg2rad(df["dec"].to_numpy())
    theta = np.pi/2 - dec
    phi   = ra

    Z = np.zeros(len(df), dtype=complex)

    for l in range(1, ell_max+1):
        for m in range(-l, l+1):
            Y = sph_harm(m, l, phi, theta)
            Z += Y

    return np.angle(Z)

# -----------------------------
# load catalog
# -----------------------------

def load_catalog(path):
    df = pd.read_csv(path)
    for col in ["theta_u", "rt_sign"]:
        if col not in df.columns:
            raise RuntimeError(f"catalog missing required column: {col}")
    return df

# -----------------------------
# main driver
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Test 104 — Harmonic–Order Sweep Robustness")
    parser.add_argument("catalog")
    parser.add_argument("--n-null", type=int, default=2000)
    parser.add_argument("--l-max", type=int, default=12)
    args = parser.parse_args()

    df = load_catalog(args.catalog)
    print("==============================================================")
    print("Test 104 — Harmonic-Order Sweep Robustness for Joint Entropy")
    print("==============================================================")

    theta = df["theta_u"].to_numpy()
    rt    = df["rt_sign"].to_numpy()

    nb_theta = 5
    nb_rt    = 2
    nb_phi   = 12

    for ell in range(1, args.l_max + 1):
        phi_h_l = compute_phi_h_l(df, ell)

        theta_b = np.digitize(theta, np.linspace(0, np.pi, nb_theta+1)) - 1
        rt_b    = np.digitize(rt, [-1, 0, 1]) - 1
        phi_b   = np.digitize(phi_h_l, np.linspace(0, 2*np.pi, nb_phi+1)) - 1

        H_real = joint_entropy_3d(theta_b, rt_b, phi_b)
        null_mean, null_std = monte_carlo_null(theta, rt, phi_h_l, args.n_null,
                                               nb_theta, nb_rt, nb_phi)
        p = (null_mean - H_real) / null_std if null_std > 0 else 0.0

        print(f"ℓ={ell:2d}  H_real={H_real:.6f}  null_mean={null_mean:.6f}  "
              f"null_std={null_std:.6f}  p_deficit={p:.6f}")

    print("==============================================================")

if __name__ == "__main__":
    main()
