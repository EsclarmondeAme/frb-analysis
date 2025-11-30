#!/usr/bin/env python3
# Test 106 — Spherical Wavelet Band–Scrambling Robustness (pure version)

import numpy as np
import pandas as pd
import argparse
from scipy.special import sph_harm
from math import pi

# ------------------------------------------------------------
# utilities
# ------------------------------------------------------------

def radec_to_unit(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])

def compute_joint_entropy(theta_u, rt, phi, n_theta=5, n_rt=2, n_phi=12):
    tbin = np.digitize(theta_u, np.linspace(0, pi, n_theta+1)) - 1
    rbin = np.digitize(rt,      np.linspace(-1, 1, n_rt+1))   - 1
    pbin = np.digitize(phi,     np.linspace(-pi, pi, n_phi+1)) - 1

    H3, _ = np.histogramdd(
        np.column_stack([tbin, rbin, pbin]),
        bins=(n_theta, n_rt, n_phi),
        density=False
    )
    P = H3 / H3.sum()
    mask = P > 0
    return -np.sum(P[mask] * np.log(P[mask]))

def null_mc(theta_u, rt, phi, N):
    Hnull = []
    for _ in range(N):
        Hnull.append(
            compute_joint_entropy(
                theta_u,
                np.random.permutation(rt),
                np.random.permutation(phi)
            )
        )
    return np.array(Hnull)

# ------------------------------------------------------------
# spherical wavelet frame construction
# ------------------------------------------------------------

def compute_wavelet_bands(theta, phi, Lmax):
    """
    Decompose into simple pseudo-wavelet bands:
    Band 1: l=1-3   (large scales)
    Band 2: l=4-7   (mid scales)
    Band 3: l=8-12  (small scales)
    """
    bands = {1: [], 2: [], 3: []}

    for l in range(1, Lmax+1):
        Z = np.zeros_like(theta, dtype=np.complex128)
        for m in range(-l, l+1):
            Z += sph_harm(m, l, phi, theta)

        if 1 <= l <= 3:
            bands[1].append(np.angle(Z))
        elif 4 <= l <= 7:
            bands[2].append(np.angle(Z))
        else:
            bands[3].append(np.angle(Z))

    return {
        1: np.stack(bands[1], axis=1),
        2: np.stack(bands[2], axis=1),
        3: np.stack(bands[3], axis=1),
    }

def combine_bands(B1, B2, B3):
    Z = np.zeros(len(B1), dtype=np.complex128)
    for B in (B1, B2, B3):
        for k in range(B.shape[1]):
            Z += np.exp(1j * B[:, k])
    return np.angle(Z)

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog")
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--l-max",  type=int, default=12)
    args = ap.parse_args()

    df = pd.read_csv(args.catalog)
    ra, dec = df["ra"].to_numpy(float), df["dec"].to_numpy(float)
    theta_u  = df["theta_u"].to_numpy(float)
    rt       = df["rt_sign"].to_numpy(float)

    v = radec_to_unit(ra, dec)
    theta = np.arccos(v[:, 2])
    phi   = np.arctan2(v[:, 1], v[:, 0])

    # build wavelet bands
    W = compute_wavelet_bands(theta, phi, args.l_max)

    # real combined field
    phi_real = combine_bands(W[1], W[2], W[3])
    H_real   = compute_joint_entropy(theta_u, rt, phi_real)
    H_null   = null_mc(theta_u, rt, phi_real, args.n_null)
    p_real   = np.mean(H_null >= H_real)

    print("=============================================================")
    print("Test 106 — Spherical Wavelet Band–Scrambling Robustness")
    print("=============================================================")
    print(f"Real: H={H_real:.6f}, null_mean={H_null.mean():.6f}, "
          f"null_std={H_null.std():.6f}, p={p_real:.6f}")
    print("-------------------------------------------------------------")

    tests = [
        ("scramble large scales",  [], [2,3]),
        ("scramble mid scales",    [1,3], []),
        ("scramble small scales",  [1,2], []),
        ("only large kept",        [1], []),
        ("only mid kept",          [2], []),
        ("only small kept",        [3], []),
    ]

    for label, keep, _ in tests:
        keep_bands = keep
        B1 = W[1].copy()
        B2 = W[2].copy()
        B3 = W[3].copy()

        if 1 not in keep_bands:
            B1 = np.random.permutation(B1)
        if 2 not in keep_bands:
            B2 = np.random.permutation(B2)
        if 3 not in keep_bands:
            B3 = np.random.permutation(B3)

        phi_scr = combine_bands(B1, B2, B3)
        H_scr   = compute_joint_entropy(theta_u, rt, phi_scr)
        Null    = null_mc(theta_u, rt, phi_scr, args.n_null)
        p_scr   = np.mean(Null >= H_scr)

        print(f"{label:20s}  H={H_scr:.6f}  mean_null={Null.mean():.6f}  "
              f"std_null={Null.std():.6f}  p={p_scr:.6f}")

    print("=============================================================")

if __name__ == "__main__":
    main()
