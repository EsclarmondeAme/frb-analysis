#!/usr/bin/env python3
# pure version — no fallbacks, no apply_filter, no placeholders

import numpy as np
import pandas as pd
import json
import argparse
from scipy.special import sph_harm

# ------------------------------------------------------------
# coordinate utilities
# ------------------------------------------------------------

def radec_to_unit(ra_deg, dec_deg):
    ra  = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])

def angular_distance_to_axis(ra_deg, dec_deg, axis_ra, axis_dec):
    v = radec_to_unit(ra_deg, dec_deg)
    a = radec_to_unit(axis_ra, axis_dec)[0]
    cosang = np.clip(v @ a, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

# ------------------------------------------------------------
# harmonic phase φ_h
# ------------------------------------------------------------

def compute_phi_h(ra_deg, dec_deg, lmax=8):
    phi   = np.deg2rad(ra_deg)
    theta = np.deg2rad(90.0 - dec_deg)  # polar angle
    Z = np.zeros(len(phi), dtype=complex)
    for l in range(1, lmax + 1):
        for m in range(-l, l+1):
            Y = sph_harm(m, l, phi, theta)
            Z += Y
    out = np.angle(Z)
    out[out < 0] += 2*np.pi
    return out

# ------------------------------------------------------------
# remnant-time sign (Test 81C definition)
# ------------------------------------------------------------

def compute_rt_sign(mjd):
    return np.where(mjd >= np.median(mjd), +1, -1)

# ------------------------------------------------------------
# 3D joint entropy
# ------------------------------------------------------------

def joint_entropy(theta_vals, rt_vals, phi_vals, n_theta, n_rt, n_phi):
    theta_edges = np.linspace(theta_vals.min(), theta_vals.max(), n_theta+1)
    phi_edges   = np.linspace(0, 2*np.pi, n_phi+1)
    rt_edges    = np.array([-1.5, 0.0, 1.5])  # two bins

    idx_theta = np.searchsorted(theta_edges, theta_vals, side="right") - 1
    idx_rt    = np.searchsorted(rt_edges,   rt_vals,    side="right") - 1
    idx_phi   = np.searchsorted(phi_edges,  phi_vals,   side="right") - 1

    good = (
        (idx_theta >= 0) & (idx_theta < n_theta) &
        (idx_rt    >= 0) & (idx_rt    < n_rt) &
        (idx_phi   >= 0) & (idx_phi   < n_phi)
    )

    idx_theta = idx_theta[good]
    idx_rt    = idx_rt[good]
    idx_phi   = idx_phi[good]

    K = n_theta * n_rt * n_phi
    flat = idx_theta * (n_rt*n_phi) + idx_rt * n_phi + idx_phi
    counts = np.bincount(flat, minlength=K)
    P = counts / counts.sum()
    nonzero = P > 0
    H = -np.sum(P[nonzero] * np.log(P[nonzero]))
    return H, counts.sum()

# ------------------------------------------------------------
# Monte-Carlo null (Test 91 method)
# ------------------------------------------------------------

def mc_null(theta_vals, rt_vals, phi_vals, n_theta, n_rt, n_phi, n_null, rng):
    Hs = np.zeros(n_null)
    for i in range(n_null):
        rt_perm  = np.copy(rt_vals)
        phi_perm = np.copy(phi_vals)
        rng.shuffle(rt_perm)
        rng.shuffle(phi_perm)
        H, _ = joint_entropy(theta_vals, rt_perm, phi_perm,
                             n_theta, n_rt, n_phi)
        Hs[i] = H
    return Hs

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog")
    ap.add_argument("--axis-json", required=True)
    ap.add_argument("--sigma-arcsec", type=float, default=3.0,
                    help="1σ RA/Dec perturbation size (arcsec)")
    ap.add_argument("--n-real", type=int, default=200,
                    help="number of coordinate perturbations to test")
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    df = pd.read_csv(args.catalog)

    # -----------------------------------------
    # Correct axis keys for your JSON
    # -----------------------------------------
    with open(args.axis_json, "r") as f:
        J = json.load(f)
    axis_ra  = J["ra_deg"]
    axis_dec = J["dec_deg"]

    ra0  = df["ra"].to_numpy()
    dec0 = df["dec"].to_numpy()
    mjd  = df["mjd"].to_numpy()

    rng = np.random.default_rng(args.seed)
    sigma_deg = args.sigma_arcsec / 3600.0

    n_theta = 5
    n_rt    = 2
    n_phi   = 12

    print("=======================================================================")
    print("Test 101 — Coordinate-Perturbation Robustness for Joint Entropy Deficit")
    print("=======================================================================")

    results = []

    for k in range(args.n_real):
        ra_pert  = ra0  + rng.normal(scale=sigma_deg, size=len(ra0))
        dec_pert = dec0 + rng.normal(scale=sigma_deg, size=len(dec0))

        theta_u = angular_distance_to_axis(ra_pert, dec_pert, axis_ra, axis_dec)
        phi_h   = compute_phi_h(ra_pert, dec_pert)
        rt_sign = compute_rt_sign(mjd)

        H_real, _ = joint_entropy(theta_u, rt_sign, phi_h,
                                  n_theta, n_rt, n_phi)

        H_null = mc_null(theta_u, rt_sign, phi_h,
                         n_theta, n_rt, n_phi,
                         args.n_null, rng)

        null_mean = H_null.mean()
        null_std  = H_null.std()
        p_deficit = np.mean(H_null <= H_real)

        results.append((H_real, null_mean, null_std, p_deficit))

        if (k+1) % 10 == 0:
            print(f"[{k+1}/{args.n_real}]  H_real={H_real:.6f}  p={p_deficit:.6f}")

    print("=======================================================================")
    print("mean(H_real)      =", np.mean([r[0] for r in results]))
    print("mean(null_mean)   =", np.mean([r[1] for r in results]))
    print("mean(null_std)    =", np.mean([r[2] for r in results]))
    print("mean(p_deficit)   =", np.mean([r[3] for r in results]))
    print("=======================================================================")


if __name__ == "__main__":
    main()
