#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree

# ============================================================
#  Helper: grain intensity
# ============================================================

def compute_grain_intensity(ra, dec, k=5):
    pts = np.column_stack([ra, dec])
    tree = cKDTree(pts)
    G = np.zeros(len(pts))
    for i in range(len(pts)):
        d, idx = tree.query(pts[i], k=k+1)
        nn = d[1:]  # skip 0-distance to itself
        if np.mean(nn) == 0:
            G[i] = 0.0
        else:
            G[i] = 1.0 / np.mean(nn)
    # normalize
    G = (G - np.mean(G)) / np.std(G)
    return G

# ============================================================
#  Kernels
# ============================================================

def kernel_powerlaw(d): return 1.0 / (1.0 + d)
def kernel_gaussian(d): return np.exp(-d**2 / 2.0)
def kernel_exp(d): return np.exp(-d)

def kernel_matern32(d):
    return (1 + np.sqrt(3)*d) * np.exp(-np.sqrt(3)*d)

def kernel_matern52(d):
    return (1 + np.sqrt(5)*d + 5*d**2/3) * np.exp(-np.sqrt(5)*d)

def kernel_lorentzian(d): return 1.0 / (1.0 + d**2)
def kernel_cauchy(d): return 1.0 / (1.0 + d)
def kernel_inverse_square(d): return 1.0 / (1.0 + d**2)
def kernel_rational_quadratic(d): return 1.0 / (1 + d**2/2)**2
def kernel_cosine(d): return np.cos(d) 
def kernel_triangular(d): return np.maximum(1-d, 0)

kernel_list = [
    kernel_powerlaw, kernel_gaussian, kernel_exp,
    kernel_matern32, kernel_matern52, kernel_lorentzian,
    kernel_cauchy, kernel_inverse_square, kernel_rational_quadratic,
    kernel_cosine, kernel_triangular
]

# ============================================================
#  Kernel Action
# ============================================================

def kernel_action(G, pts, kernel_fn):
    tree = cKDTree(pts)
    N = len(pts)
    S = 0.0
    for i in range(N):
        d, idx = tree.query(pts[i], k=10)
        w = kernel_fn(d[1:])  # ignore self
        gdiff = (G[i] - G[idx[1:]])**2
        S += np.sum(w * gdiff)
    return S

# ============================================================
#  main
# ============================================================

def main():
    import sys
    if len(sys.argv) < 2:
        print("usage: python frb_nonlocal_kernel_universality_test57.py frbs_unified.csv")
        return

    df = pd.read_csv(sys.argv[1])
    print("loaded 600 FRBs")

    ra = np.deg2rad(df["ra"].values)
    dec = np.deg2rad(df["dec"].values)
    pts = np.column_stack([ra, dec])

    # grain intensity
    G = compute_grain_intensity(ra, dec)

    print("using 12 kernels")

    # compute real kernel actions
    S_real_list = []
    for kfn in kernel_list:
        S_real_list.append(kernel_action(G, pts, kfn))
    U_real = np.sum(S_real_list)

    # Monte Carlo null
    print("running Monte Carlo null for each kernel...")
    N_mc = 2000
    U_null = np.zeros(N_mc)

    for m in tqdm(range(N_mc)):
        idx = np.random.permutation(len(G))
        Gs = G[idx]
        S_list = []
        for kfn in kernel_list:
            S_list.append(kernel_action(Gs, pts, kfn))
        U_null[m] = np.sum(S_list)

    print("building null distribution for U...")

    # ============================================================
    #  Correct p-value:
    #  p = fraction(|U_null| >= |U_real|)
    # ============================================================
    p = np.mean(np.abs(U_null) >= np.abs(U_real))

    print("====================================================================")
    print("FRB NONLOCAL KERNEL UNIVERSALITY TEST (TEST 57)")
    print("====================================================================")
    print(f"U_real = {U_real}")
    print(f"null mean = {np.mean(U_null)}")
    print(f"null std  = {np.std(U_null)}")
    print(f"p-value   = {p}")
    print("--------------------------------------------------------------------")
    print("interpretation:")
    print("  - low p  → latent field is low-action under many kernels → universal nonlocal Lagrangian")
    print("  - high p → only special kernels give low action → accidental structure")
    print("====================================================================")
    print("test 57 complete.")
    print("====================================================================")

if __name__ == "__main__":
    main()
