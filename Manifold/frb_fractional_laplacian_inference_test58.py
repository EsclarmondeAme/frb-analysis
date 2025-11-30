#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------
# utilities
# ---------------------------------------------------------------

def pairwise_dist(ra, dec):
    """great-circle distance matrix in radians"""
    ra  = np.deg2rad(ra)
    dec = np.deg2rad(dec)
    cosd = (
        np.sin(dec[:,None])*np.sin(dec[None,:]) +
        np.cos(dec[:,None])*np.cos(dec[None,:])*np.cos(ra[:,None]-ra[None,:])
    )
    cosd = np.clip(cosd, -1.0, 1.0)
    return np.arccos(cosd)


def grain_intensity(theta, phi):
    """grain intensity ≈ inverse local spacing from nearest neighbors (robust)"""
    eps = 1e-6

    N = len(theta)
    vec = np.column_stack([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ])

    dvals = np.sqrt(((vec[:,None,:] - vec[None,:,:])**2).sum(axis=2))

    G = np.zeros(N)
    for i in range(N):
        nn = np.sort(dvals[i])[1:7]
        m  = max(np.mean(nn), eps)
        G[i] = 1.0 / m

    # robust normalization
    G = (G - np.nanmean(G)) / (np.nanstd(G) + eps)

    return G



def energy_gradient_encoding(dm, snr, flu, width, z):
    """scalar EGR (same as tests 50–57)"""
    E = (dm + 0.1*snr + 0.01*flu + 0.001*width + 5.0*z)
    E = (E - E.mean()) / E.std()
    return E


def fractional_kernel(dist, s):
    """
    fractional Laplacian kernel on sphere:
    K_s(d) ~ d^{-2s}  with d > 0
    """
    K = 1.0 / (dist**(2*s) + 1e-6)
    np.fill_diagonal(K, 0.0)
    return K


def nonlocal_action(F, K):
    """S = F^T K F"""
    return float(F @ (K @ F))

def pairwise_dist(ra, dec):
    """compute small-angle pairwise separations in radians"""
    ra = np.deg2rad(np.asarray(ra))
    dec = np.deg2rad(np.asarray(dec))

    ra2  = ra[:, None]
    dec2 = dec[:, None]

    cosd = (
        np.sin(dec2) * np.sin(dec2.T)
        + np.cos(dec2) * np.cos(dec2.T) * np.cos(ra2 - ra2.T)
    )

    cosd = np.clip(cosd, -1.0, 1.0)
    return np.arccos(cosd)

# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------

def main():
    import sys
    if len(sys.argv) < 2:
        print("usage: python frb_fractional_laplacian_inference_test58.py frbs_unified.csv")
        return

    df = pd.read_csv(sys.argv[1])
    N = len(df)
    print(f"loaded {N} FRBs")

    # unified coords
    theta = np.deg2rad(df["theta_unified"].values)
    phi   = np.deg2rad(df["phi_unified"].values)

    # latent field
    G   = grain_intensity(theta, phi)
    EGR = energy_gradient_encoding(df["dm"], df["snr"], df["fluence"],
                                   df["width"], df["z_est"])
    F = 0.6*G + 0.4*EGR
    F = (F - F.mean()) / F.std()

    # base geometry
    D = pairwise_dist(df["ra"], df["dec"])

    # scan fractional orders s
    S_list = []
    s_values = np.linspace(0.1, 2.5, 25)   # from ultra-nonlocal to moderately local
    print("scanning fractional orders...")

    for s in s_values:
        K = fractional_kernel(D, s)
        S_list.append(nonlocal_action(F, K))

    S_list = np.array(S_list)
    s_best = s_values[np.argmin(S_list)]
    S_best = np.min(S_list)

    # Monte-Carlo null for best s
    N_MC = 2000
    Sn = []
    print("running Monte Carlo null at best s...")

    Kbest = fractional_kernel(D, s_best)
    for _ in tqdm(range(N_MC)):
        Fsh = np.random.permutation(F)
        Sn.append(nonlocal_action(Fsh, Kbest))

    Sn = np.array(Sn)
    mu = Sn.mean()
    sd = Sn.std()

    p = np.mean(Sn <= S_best)

    print("===================================================================")
    print("FRB FRACTIONAL LAPLACIAN INFERENCE (TEST 58)")
    print("===================================================================")
    print(f"best fractional order s_best = {s_best:.4f}")
    print(f"best action S_best          = {S_best:.4e}")
    print(f"null mean                   = {mu:.4e}")
    print(f"null std                    = {sd:.4e}")
    print(f"p-value                     = {p:.6f}")
    print("-------------------------------------------------------------------")
    print("interpretation:")
    print("  - best-fit s identifies the effective fractional Laplacian order")
    print("  - low p → field strongly matches Δ^s structure")
    print("  - high p → no particular fractional order preferred")
    print("===================================================================")
    print("test 58 complete.")
    print("===================================================================")


if __name__ == "__main__":
    main()
