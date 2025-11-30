#!/usr/bin/env python3
# pure version — no fallbacks, no apply_filter, no placeholders

import numpy as np
import pandas as pd
import argparse

# ------------------------------------------------------------------
# joint entropy H(theta, rt, phi) — same structure as Test 91
# ------------------------------------------------------------------

def joint_entropy(theta_vals, rt_vals, phi_vals,
                  n_theta=5, n_rt=2, n_phi=12):
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

    K = n_theta*n_rt*n_phi
    flat = idx_theta*(n_rt*n_phi) + idx_rt*n_phi + idx_phi
    counts = np.bincount(flat, minlength=K)
    P = counts / counts.sum()
    nz = P > 0
    return -np.sum(P[nz] * np.log(P[nz]))

# ------------------------------------------------------------------
# permutation null for a given (theta, rt, phi) field
# ------------------------------------------------------------------

def permutation_null(theta_vals, rt_vals, phi_vals,
                     n_theta, n_rt, n_phi,
                     n_null, rng):
    Hs = np.zeros(n_null)
    rt  = rt_vals.copy()
    phi = phi_vals.copy()
    for i in range(n_null):
        rng.shuffle(rt)
        rng.shuffle(phi)
        Hs[i] = joint_entropy(theta_vals, rt, phi,
                              n_theta=n_theta, n_rt=n_rt, n_phi=n_phi)
    return Hs

# ------------------------------------------------------------------
# compute Test 91 p-value for a given field
# ------------------------------------------------------------------

def test91_p(theta_vals, rt_vals, phi_vals,
             n_theta=5, n_rt=2, n_phi=12,
             n_null=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    H_real = joint_entropy(theta_vals, rt_vals, phi_vals,
                           n_theta=n_theta, n_rt=n_rt, n_phi=n_phi)
    H_null = permutation_null(theta_vals, rt_vals, phi_vals,
                              n_theta, n_rt, n_phi,
                              n_null, rng)
    p = np.mean(H_null <= H_real)
    return H_real, H_null.mean(), H_null.std(), p

# ------------------------------------------------------------------
# main: meta-null calibration
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Test 102 — Meta-null calibration for Test 91"
    )
    ap.add_argument("catalog", type=str,
                    help="frbs_unified_for_test91.csv")
    ap.add_argument("--n-null", type=int, default=2000,
                    help="null permutations per run (inner layer)")
    ap.add_argument("--n-meta", type=int, default=200,
                    help="number of surrogate universes (outer layer)")
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    df = pd.read_csv(args.catalog)

    # real fields from Test 91
    theta_real = df["theta_u"].to_numpy(float)
    rt_real    = df["rt_sign"].to_numpy(float)
    phi_real   = df["phi_h"].to_numpy(float)

    rng = np.random.default_rng(args.seed)

    n_theta = 5
    n_rt    = 2
    n_phi   = 12

    # --- compute real Test 91 p-value ---
    H_real, null_mean_real, null_std_real, p_real = test91_p(
        theta_real, rt_real, phi_real,
        n_theta=n_theta, n_rt=n_rt, n_phi=n_phi,
        n_null=args.n_null, rng=rng
    )

    print("===================================================================")
    print("Test 102 — Meta-null calibration for Test 91")
    print("===================================================================")
    print(f"Real sample:")
    print(f"  H_real      = {H_real:.6f}")
    print(f"  null_mean   = {null_mean_real:.6f}")
    print(f"  null_std    = {null_std_real:.6f}")
    print(f"  p_real      = {p_real:.6f}")
    print("-------------------------------------------------------------------")

    # --- build surrogate universe ensemble ---
    p_meta = np.zeros(args.n_meta)
    H_meta = np.zeros(args.n_meta)

    N = len(theta_real)

    for k in range(args.n_meta):
        # surrogate fields: random rt_sign, random phi, same theta_u geometry
        rt_surr  = rng.choice([-1, +1], size=N)
        phi_surr = rng.uniform(0, 2*np.pi, size=N)

        H_s, m_s, s_s, p_s = test91_p(
            theta_real, rt_surr, phi_surr,
            n_theta=n_theta, n_rt=n_rt, n_phi=n_phi,
            n_null=args.n_null, rng=rng
        )
        H_meta[k] = H_s
        p_meta[k] = p_s

        if (k+1) % 10 == 0:
            print(f"[{k+1}/{args.n_meta}]  H_surr={H_s:.6f}  p_surr={p_s:.6f}")

    # summary over meta-null p-values
    print("===================================================================")
    print("Meta-null surrogate summary:")
    print(f"  mean(p_surr)    = {p_meta.mean():.6f}")
    print(f"  min(p_surr)     = {p_meta.min():.6f}")
    print(f"  max(p_surr)     = {p_meta.max():.6f}")
    print(f"  frac(p_surr <= {p_real:.6f}) = {np.mean(p_meta <= p_real):.6f}")
    print("===================================================================")
    print("interpretation:")
    print("  - p_real is the Test 91 p-value for the actual FRB field.")
    print("  - p_surr values are what the Test 91 p would look like")
    print("    if the Universe were random (random rt_sign, random phi_h).")
    print("  - If p_real lies far below typical p_surr values,")
    print("    the anomaly is not an artefact of the permutation scheme.")
    print("===================================================================")


if __name__ == "__main__":
    main()
