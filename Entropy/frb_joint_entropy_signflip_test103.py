#!/usr/bin/env python3
# Test 103 — Sign-Flip Invariance for Test 91 Joint Entropy
# Pure-science version: no fallbacks, no apply_filter, no placeholders.

import numpy as np
import pandas as pd
import argparse

# ------------------------------------------------------------
# Joint entropy H(theta, rt, phi) — identical to Test 91
# ------------------------------------------------------------

def joint_entropy(theta_vals, rt_vals, phi_vals,
                  n_theta=5, n_rt=2, n_phi=12):

    theta_edges = np.linspace(theta_vals.min(), theta_vals.max(), n_theta+1)
    phi_edges   = np.linspace(0, 2*np.pi, n_phi+1)
    rt_edges    = np.array([-1.5, 0.0, 1.5])  # two bins: -1, +1

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

    K = n_theta*n_rt*n_phi
    flat = idx_theta*(n_rt*n_phi) + idx_rt*n_phi + idx_phi
    counts = np.bincount(flat, minlength=K)

    P = counts / counts.sum()
    P = P[P > 0]

    return -np.sum(P * np.log(P))

# ------------------------------------------------------------
# permutation null — same structure as Test 91
# ------------------------------------------------------------

def permutation_null(theta_vals, rt_vals, phi_vals,
                     n_theta, n_rt, n_phi, n_null, rng):
    Hs = np.zeros(n_null)
    rt  = rt_vals.copy()
    phi = phi_vals.copy()
    for i in range(n_null):
        rng.shuffle(rt)
        rng.shuffle(phi)
        Hs[i] = joint_entropy(theta_vals, rt, phi,
                              n_theta=n_theta, n_rt=n_rt, n_phi=n_phi)
    return Hs

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Test 103 — Sign-Flip robustness for Test 91"
    )
    ap.add_argument("catalog", type=str,
                    help="frbs_unified_for_test91.csv")
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    df = pd.read_csv(args.catalog)

    # required fields
    for col in ["theta_u", "rt_sign", "phi_h"]:
        if col not in df.columns:
            raise RuntimeError(f"missing required column: {col}")

    theta = df["theta_u"].to_numpy(float)
    rt    = df["rt_sign"].to_numpy(float)
    phi   = df["phi_h"].to_numpy(float)

    rng = np.random.default_rng(args.seed)

    n_theta = 5
    n_rt    = 2
    n_phi   = 12

    # ------------------------------------------------------------
    # real (NO sign flip)
    # ------------------------------------------------------------
    H_real = joint_entropy(theta, rt, phi,
                           n_theta=n_theta, n_rt=n_rt, n_phi=n_phi)
    H_null_real = permutation_null(theta, rt, phi,
                                   n_theta, n_rt, n_phi,
                                   args.n_null, rng)
    null_mean_real = np.mean(H_null_real)
    null_std_real  = np.std(H_null_real)
    p_real = np.mean(H_null_real <= H_real)

    # ------------------------------------------------------------
    # sign-flipped case
    # ------------------------------------------------------------
    rt_flip = -rt  # invert remnant-time polarity

    H_flip = joint_entropy(theta, rt_flip, phi,
                           n_theta=n_theta, n_rt=n_rt, n_phi=n_phi)
    H_null_flip = permutation_null(theta, rt_flip, phi,
                                   n_theta, n_rt, n_phi,
                                   args.n_null, rng)
    null_mean_flip = np.mean(H_null_flip)
    null_std_flip  = np.std(H_null_flip)
    p_flip = np.mean(H_null_flip <= H_flip)

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------

    print("=====================================================================")
    print("Test 103 — Remnant-Time Sign-Flip Robustness for Joint Entropy")
    print("=====================================================================")
    print("Original Test 91 field:")
    print(f"  H_real      = {H_real:.6f}")
    print(f"  null_mean   = {null_mean_real:.6f}")
    print(f"  null_std    = {null_std_real:.6f}")
    print(f"  p_deficit   = {p_real:.6f}")
    print("---------------------------------------------------------------------")
    print("Sign-flipped field (rt_sign → -rt_sign):")
    print(f"  H_flip      = {H_flip:.6f}")
    print(f"  null_mean   = {null_mean_flip:.6f}")
    print(f"  null_std    = {null_std_flip:.6f}")
    print(f"  p_deficit   = {p_flip:.6f}")
    print("=====================================================================")
    print("Interpretation:")
    print("  - If the Test 91 anomaly is physically tied to the direction of the")
    print("    remnant-time field, then flipping its sign should destroy or")
    print("    strongly weaken the entropy deficit.")
    print("  - If p_flip is large (≈ 0.5), the deficit depends on the actual")
    print("    physical polarity of the remnant-time field.")
    print("  - If p_flip remains small, the anomaly is invariant under sign")
    print("    reversal and the correlation is magnitude-only, not directional.")
    print("=====================================================================")


if __name__ == "__main__":
    main()
