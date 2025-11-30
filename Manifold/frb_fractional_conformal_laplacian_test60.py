#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------
# utilities
# --------------------------------------------------------------

def pairwise_dist(ra, dec):
    ra  = np.deg2rad(np.asarray(ra))
    dec = np.deg2rad(np.asarray(dec))

    cosd = ( np.sin(dec[:,None])*np.sin(dec[None,:]) +
             np.cos(dec[:,None])*np.cos(dec[None,:]) *
             np.cos(ra[:,None]-ra[None,:]) )
    cosd = np.clip(cosd, -1.0, 1.0)
    D = np.arccos(cosd)

    # distance floor: avoid zero-dist singularities
    eps = 1e-6
    return np.maximum(D, eps)


def grain_intensity(theta, phi):
    N = len(theta)
    vec = np.column_stack([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ])
    d = np.sqrt(((vec[:,None,:] - vec[None,:,:])**2).sum(axis=2))
    d = np.maximum(d, 1e-6)

    G = np.zeros(N)
    for i in range(N):
        nn = np.sort(d[i])[1:7]
        G[i] = 1.0 / np.mean(nn)

    G = (G - np.nanmean(G)) / (np.nanstd(G) + 1e-9)
    return np.nan_to_num(G)


def energy_feature(df):
    E = (df["dm"].values
         + 0.1*df["snr"].values
         + 0.01*df["fluence"].values
         + 0.001*df["width"].values
         + 5.0*df["z_est"].values)

    E = (E - np.nanmean(E)) / (np.nanstd(E)+1e-9)
    return np.nan_to_num(E)


def fractional_kernel(dist, s):
    eps = 1e-6
    return 1.0 / (dist**(2*s) + eps)


def conformal_kernel(dist, s, c):
    eps = 1e-6
    return 1.0/(dist**(2*s) + eps) + c*dist


def nonlocal_action(F, K):
    return float(F @ (K @ F))


# --------------------------------------------------------------
# main
# --------------------------------------------------------------

def main():
    import sys
    if len(sys.argv) < 2:
        print("usage: python frb_fractional_conformal_laplacian_test60.py frbs_unified.csv")
        return

    df = pd.read_csv(sys.argv[1])
    N = len(df)
    print(f"loaded {N} FRBs")

    theta = np.deg2rad(df["theta_unified"].values)
    phi   = np.deg2rad(df["phi_unified"].values)

    G   = grain_intensity(theta, phi)
    E   = energy_feature(df)

    F = 0.6*G + 0.4*E
    F = (F - F.mean())/(F.std()+1e-9)
    F = np.nan_to_num(F)

    D = pairwise_dist(df["ra"].values, df["dec"].values)

    # search grid
    s_grid = np.linspace(0.1, 2.0, 40)
    c_grid = np.linspace(-1.0, 1.0, 21)

    best_S = np.inf
    best_s = None
    best_c = None

    print("searching (s,c) for minimal action...")

    for s in tqdm(s_grid):
        for c in c_grid:
            try:
                K = conformal_kernel(D, s, c)
                np.fill_diagonal(K, 0.0)
                S = nonlocal_action(F, K)

                if np.isfinite(S) and S < best_S:
                    best_S = S
                    best_s = s
                    best_c = c
            except:
                continue

    print("\n--- best parameters found ---")
    print(f"s = {best_s}")
    print(f"c = {best_c}")
    print(f"S = {best_S}")

    # monte carlo
    print("\nrunning Monte Carlo null at best parameters...")
    N_MC = 2000
    null_S = []

    Kbest = conformal_kernel(D, best_s, best_c)
    np.fill_diagonal(Kbest, 0.0)

    for _ in tqdm(range(N_MC)):
        Fsh = np.random.permutation(F)
        null_S.append(nonlocal_action(Fsh, Kbest))

    null_S = np.array(null_S)
    p = np.mean(null_S <= best_S)

    print("================================================================")
    print("FRB FRACTIONAL CONFORMAL LAPLACIAN RECONSTRUCTION (TEST 60)")
    print("================================================================")
    print(f"best fractional order s = {best_s:.4f}")
    print(f"best conformal coefficient c = {best_c:.4f}")
    print(f"real action S_real = {best_S:.6e}")
    print(f"null mean = {null_S.mean():.6e}")
    print(f"null std  = {null_S.std():.6e}")
    print(f"p-value   = {p:.6f}")
    print("----------------------------------------------------------------")
    print("interpretation:")
    print("  - low p → latent field matches fractional-conformal Laplacian")
    print("  - high p → no preferred fractional-conformal operator")
    print("================================================================")
    print("test 60 complete.")
    print("================================================================")


if __name__ == "__main__":
    main()
