#!/usr/bin/env python3
# Test 105 — ℓ-band scrambling robustness for joint-entropy deficit (Test 91)

import numpy as np
import pandas as pd
import argparse
from scipy.special import sph_harm
from math import radians, sin, cos, acos, pi

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

def compute_phase_lmax(df, lmax):
    ra = df["ra"].to_numpy(float)
    dec = df["dec"].to_numpy(float)
    v = radec_to_unit(ra, dec)

    # theta = colatitude
    theta = np.arccos(v[:, 2])
    # phi = longitude
    phi = np.arctan2(v[:, 1], v[:, 0])

    # build complex harmonic sum
    Z = np.zeros(len(df), dtype=np.complex128)
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            Z += sph_harm(m, l, phi, theta)

    return np.angle(Z)

def compute_joint_entropy(theta_u, rt, phi, n_theta=5, n_rt=2, n_phi=12):
    # discretize
    tbin = np.digitize(theta_u,  np.linspace(0, pi,     n_theta + 1)) - 1
    rbin = np.digitize(rt,       np.linspace(-1, 1,     n_rt    + 1)) - 1
    pbin = np.digitize(phi,      np.linspace(-pi, pi,   n_phi   + 1)) - 1

    # joint histogram
    H, _ = np.histogramdd(
        np.column_stack([tbin, rbin, pbin]),
        bins=(n_theta, n_rt, n_phi),
        density=False
    )

    P = H / H.sum()
    mask = P > 0
    return -np.sum(P[mask] * np.log(P[mask]))

def null_mc(theta_u, rt, phi, N=2000):
    H_null = []
    for _ in range(N):
        phi_perm = np.random.permutation(phi)
        rt_perm = np.random.permutation(rt)
        H_null.append(compute_joint_entropy(theta_u, rt_perm, phi_perm))
    return np.array(H_null)

# ------------------------------------------------------------
# ℓ-band scrambling
# ------------------------------------------------------------

def scramble_band(phases_dict, keep_bands):
    """
    phases_dict: {l : phi_l}
    keep_bands: list of harmonic orders to keep intact
    Others get independently permuted.
    """
    scrambled = {}
    for l, phi_l in phases_dict.items():
        if l in keep_bands:
            scrambled[l] = phi_l.copy()
        else:
            scrambled[l] = np.random.permutation(phi_l)
    return scrambled

def combine_phases(phases_dict):
    # combine all ℓ-phase vectors into one unified harmonic field
    # by summing exp(i phi_l) over all ℓ
    Z = np.zeros_like(list(phases_dict.values())[0], dtype=np.complex128)
    for phi_l in phases_dict.values():
        Z += np.exp(1j * phi_l)
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
    N = len(df)

    # extract fields
    theta_u = df["theta_u"].to_numpy(float)
    rt      = df["rt_sign"].to_numpy(float)
    phi_h   = df["phi_h"].to_numpy(float)

    # precompute phase fields for each individual ℓ
    phases = {}
    ra  = df["ra"].to_numpy(float)
    dec = df["dec"].to_numpy(float)
    v   = radec_to_unit(ra, dec)
    theta = np.arccos(v[:,2])
    phi   = np.arctan2(v[:,1], v[:,0])

    for l in range(1, args.l_max + 1):
        Z = np.zeros(N, dtype=np.complex128)
        for m in range(-l, l+1):
            Z += sph_harm(m, l, phi, theta)
        phases[l] = np.angle(Z)

    # real combined field
    phi_real = combine_phases(phases)
    H_real   = compute_joint_entropy(theta_u, rt, phi_real)
    H_null   = null_mc(theta_u, rt, phi_real, N=args.n_null)
    p_real   = np.mean(H_null >= H_real)

    print("==============================================================")
    print("Test 105 — ℓ-band scrambling robustness for Test 91")
    print("==============================================================")
    print(f"Real sample: H_real={H_real:.6f}, null_mean={H_null.mean():.6f}, "
          f"null_std={H_null.std():.6f}, p={p_real:.6f}")
    print("--------------------------------------------------------------")

    # define ℓ bands
    low_band  = list(range(1,5))     # 1–4
    mid_band  = list(range(5,9))     # 5–8
    high_band = list(range(9, args.l_max+1))  # 9–12

    bands = [
        ("only low kept", low_band),
        ("only mid kept", mid_band),
        ("only high kept", high_band),
        ("scramble low", mid_band + high_band),
        ("scramble mid", low_band + high_band),
        ("scramble high", low_band + mid_band),
    ]

    for label, keep in bands:
        scr = scramble_band(phases, keep)
        phi_scr = combine_phases(scr)
        H_scr   = compute_joint_entropy(theta_u, rt, phi_scr)
        Hn      = null_mc(theta_u, rt, phi_scr, N=args.n_null)
        p_scr   = np.mean(Hn >= H_scr)

        print(f"{label:15s}  H={H_scr:.6f}  null_mean={Hn.mean():.6f}  "
              f"null_std={Hn.std():.6f}  p={p_scr:.6f}")

    print("==============================================================")

if __name__ == "__main__":
    main()
