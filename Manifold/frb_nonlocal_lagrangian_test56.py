#!/usr/bin/env python3
"""
TEST 56 — NONLOCAL LAGRANGIAN / KERNEL-ACTION RECONSTRUCTION
Does the latent FRB geometry minimize a nonlocal action?

Action:
    S = sum_{i<j} K(d_ij) * (F_i - F_j)^2

Kernels tested:
    - power-law:    K(d) = 1 / (d + eps)^alpha
    - gaussian:     K(d) = exp(-(d/sigma)^2)
    - exponential:  K(d) = exp(-d/lambda)

We compute S_real for each kernel, compare to Monte Carlo null,
and calculate p-values.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

# ----------------------------------------------
# compute latent field F (grain intensity)
# ----------------------------------------------
def grain_intensity(ra, dec):
    pts = np.vstack([ra, dec]).T
    dist = squareform(pdist(pts))
    k = 5
    sortd = np.sort(dist, axis=1)
    F = 1.0 / (np.mean(sortd[:, 1:k+1], axis=1) + 1e-12)
    F = (F - np.mean(F)) / np.std(F)
    return F

# ----------------------------------------------
# compute nonlocal action
# ----------------------------------------------
def nonlocal_action(F, dist, kernel):
    K = kernel(dist)
    np.fill_diagonal(K, 0.0)
    diff2 = (F[:, None] - F[None, :])**2
    return 0.5 * np.sum(K * diff2)

# kernels
def kernel_powerlaw(dist, alpha=1.0):
    return 1.0 / (dist + 1e-5)**alpha

def kernel_gaussian(dist, sigma=3.0):
    return np.exp(-(dist/sigma)**2)

def kernel_exponential(dist, lam=5.0):
    return np.exp(-dist/lam)

# ----------------------------------------------
# main
# ----------------------------------------------
def main():
    import sys
    if len(sys.argv) < 2:
        print("usage: python frb_nonlocal_lagrangian_test56.py frbs_unified.csv")
        return

    df = pd.read_csv(sys.argv[1])
    ra = np.deg2rad(df["ra"].values)
    dec = np.deg2rad(df["dec"].values)

    pts = np.vstack([ra, dec]).T
    dist = squareform(pdist(pts))

    F = grain_intensity(ra, dec)

    kernels = {
        "powerlaw":   lambda d: kernel_powerlaw(d, alpha=1.0),
        "gaussian":   lambda d: kernel_gaussian(d, sigma=3.0),
        "exp":        lambda d: kernel_exponential(d, lam=5.0),
    }

    print("====================================================")
    print(" FRB NONLOCAL LAGRANGIAN / KERNEL-ACTION TEST (TEST 56)")
    print("====================================================")

    N_MC = 2000

    for name, K in kernels.items():
        print(f"\n-- kernel: {name} --")

        S_real = nonlocal_action(F, dist, K)

        S_null = []
        for _ in tqdm(range(N_MC)):
            F_shuf = np.random.permutation(F)
            S_null.append(nonlocal_action(F_shuf, dist, K))

        S_null = np.array(S_null)
        p = np.mean(S_null <= S_real)

        print(f"  S_real = {S_real:.6f}")
        print(f"  null_mean = {np.mean(S_null):.6f}")
        print(f"  null_std  = {np.std(S_null):.6f}")
        print(f"  p-value   = {p:.6f}")

    print("\n====================================================")
    print(" test 56 complete.")
    print("====================================================")

if __name__ == "__main__":
    main()
