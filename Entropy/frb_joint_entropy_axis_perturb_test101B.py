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

def unit_to_radec(v):
    x, y, z = v[:,0], v[:,1], v[:,2]
    dec = np.arcsin(z)
    ra  = np.arctan2(y, x)
    ra_deg  = np.degrees(ra)
    dec_deg = np.degrees(dec)
    ra_deg[ra_deg < 0] += 360
    return ra_deg, dec_deg

def perturb_axis(ra_deg, dec_deg, sigma_deg, rng):
    """Isotropic perturbation by drawing a random small-angle vector of size sigma_deg."""
    # convert axis -> unit vector
    a = radec_to_unit(ra_deg, dec_deg)[0]

    # random 3D Gaussian then scale to desired RMS angular deviation
    g = rng.normal(size=3)
    g = g / np.linalg.norm(g)

    # small rotation: angle drawn from N(0, sigma_deg)
    delta = rng.normal(scale=np.deg2rad(sigma_deg))

    # Rodrigues rotation
    k = g / np.linalg.norm(g)
    a_rot = (a * np.cos(delta) +
             np.cross(k, a) * np.sin(delta) +
             k * (np.dot(k, a)) * (1 - np.cos(delta)))

    ra_new, dec_new = unit_to_radec(a_rot.reshape(1,3))
    return float(ra_new[0]), float(dec_new[0])

# ------------------------------------------------------------
# harmonic phase φ_h
# ------------------------------------------------------------

def compute_phi_h(ra_deg, dec_deg, lmax=8):
    phi   = np.deg2rad(ra_deg)
    theta = np.deg2rad(90.0 - dec_deg)
    Z = np.zeros(len(phi), dtype=complex)
    for l in range(1, lmax + 1):
        for m in range(-l, l+1):
            Y = sph_harm(m, l, phi, theta)
            Z += Y
    out = np.angle(Z)
    out[out < 0] += 2*np.pi
    return out

# ------------------------------------------------------------
# remnant-time sign (Test 81C)
# ------------------------------------------------------------

def compute_rt_sign(mjd):
    return np.where(mjd >= np.median(mjd), +1, -1)

# ------------------------------------------------------------
# axis-distance
# ------------------------------------------------------------

def angular_distance_to_axis(ra_deg, dec_deg, axis_ra, axis_dec):
    v = radec_to_unit(ra_deg, dec_deg)
    a = radec_to_unit(axis_ra, axis_dec)[0]
    cosang = np.clip(v @ a, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

# ------------------------------------------------------------
# entropy
# ------------------------------------------------------------

def joint_entropy(theta_vals, rt_vals, phi_vals, n_theta, n_rt, n_phi):
    theta_edges = np.linspace(theta_vals.min(), theta_vals.max(), n_theta+1)
    phi_edges   = np.linspace(0, 2*np.pi, n_phi+1)
    rt_edges    = np.array([-1.5, 0.0, 1.5])

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
    nz = (P > 0)
    return -np.sum(P[nz] * np.log(P[nz])), counts.sum()

# ------------------------------------------------------------
# Monte Carlo null (Test 91 style)
# ------------------------------------------------------------

def mc_null(theta_vals, rt_vals, phi_vals, n_theta, n_rt, n_phi, n_null, rng):
    Hs = np.zeros(n_null)
    for i in range(n_null):
        rt_p  = np.copy(rt_vals)
        phi_p = np.copy(phi_vals)
        rng.shuffle(rt_p)
        rng.shuffle(phi_p)
        Hs[i], _ = joint_entropy(theta_vals, rt_p, phi_p,
                                 n_theta, n_rt, n_phi)
    return Hs

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog")
    ap.add_argument("--axis-json", required=True)
    ap.add_argument("--sigma-deg", type=float, default=1.0,
                    help="1σ axis perturbation size (degrees)")
    ap.add_argument("--n-real", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    df = pd.read_csv(args.catalog)
    with open(args.axis_json, "r") as f:
        J = json.load(f)

    axis_ra0  = J["ra_deg"]
    axis_dec0 = J["dec_deg"]

    ra  = df["ra"].to_numpy()
    dec = df["dec"].to_numpy()
    mjd = df["mjd"].to_numpy()

    rng = np.random.default_rng(args.seed)

    n_theta = 5
    n_rt    = 2
    n_phi   = 12

    print("===================================================================")
    print("Test 101B — Axis-Perturbation Robustness for Joint Entropy Deficit")
    print("===================================================================")

    results = []

    for k in range(args.n_real):

        # perturb the axis by sigma_deg degrees
        axis_ra, axis_dec = perturb_axis(axis_ra0, axis_dec0,
                                         args.sigma_deg, rng)

        # recompute geometry / phase / rt
        theta_u = angular_distance_to_axis(ra, dec, axis_ra, axis_dec)
        phi_h   = compute_phi_h(ra, dec)
        rt_sign = compute_rt_sign(mjd)

        H_real, _ = joint_entropy(theta_u, rt_sign, phi_h,
                                  n_theta, n_rt, n_phi)

        H_null = mc_null(theta_u, rt_sign, phi_h,
                         n_theta, n_rt, n_phi,
                         args.n_null, rng)

        null_mean = H_null.mean()
        null_std  = H_null.std()
        p         = np.mean(H_null <= H_real)

        results.append((H_real, null_mean, null_std, p))

        if (k+1) % 10 == 0:
            print(f"[{k+1}/{args.n_real}] H_real={H_real:.6f}  p={p:.6f}")

    print("===================================================================")
    print("mean(H_real)      =", np.mean([r[0] for r in results]))
    print("mean(null_mean)   =", np.mean([r[1] for r in results]))
    print("mean(null_std)    =", np.mean([r[2] for r in results]))
    print("mean(p_deficit)   =", np.mean([r[3] for r in results]))
    print("===================================================================")


if __name__ == "__main__":
    main()
