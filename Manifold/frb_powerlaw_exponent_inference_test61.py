#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# utilities
# ============================================================

def pairwise_dist(ra, dec):
    ra  = np.deg2rad(np.asarray(ra))
    dec = np.deg2rad(np.asarray(dec))

    cosd = (np.sin(dec[:,None])*np.sin(dec[None,:]) +
            np.cos(dec[:,None])*np.cos(dec[None,:])*np.cos(ra[:,None]-ra[None,:]))

    cosd = np.clip(cosd, -1.0, 1.0)
    return np.arccos(cosd)

def grain_intensity(theta, phi):
    """robust grain estimator used in tests 56 & 58 (nan-proof)"""
    N = len(theta)
    vec = np.column_stack([np.sin(theta)*np.cos(phi),
                           np.sin(theta)*np.sin(phi),
                           np.cos(theta)])

    d = np.sqrt(((vec[:,None,:] - vec[None,:,:])**2).sum(axis=2))
    d += 1e-6  # avoid exact zeros

    G = np.zeros(N)
    for i in range(N):
        nn = np.sort(d[i])[1:7]
        G[i] = 1.0 / (np.mean(nn) + 1e-6)

    G = np.nan_to_num(G, nan=np.nanmean(G), posinf=np.nanmean(G), neginf=np.nanmean(G))
    return (G - np.mean(G)) / (np.std(G) + 1e-6)

def energy_gradient_encoding(dm, snr, flu, width, z):
    E = (dm + 0.1*snr + 0.01*flu + 0.001*width + 5.0*z)
    return (E - E.mean()) / (E.std() + 1e-6)

def nonlocal_action(F, D, alpha):
    K = 1.0 / (D**alpha + 1e-6)
    np.fill_diagonal(K, 0.0)
    return float(F @ (K @ F))


# ============================================================
# main
# ============================================================

def main():
    import sys
    if len(sys.argv) < 2:
        print("usage: python frb_powerlaw_exponent_inference_test61.py frbs_unified.csv")
        return

    df = pd.read_csv(sys.argv[1])
    print(f"loaded {len(df)} FRBs")

    theta = np.deg2rad(df["theta_unified"].values)
    phi   = np.deg2rad(df["phi_unified"].values)

    # latent scalar field
    G   = grain_intensity(theta, phi)
    EGR = energy_gradient_encoding(df["dm"], df["snr"], df["fluence"], df["width"], df["z_est"])

    F = 0.6*G + 0.4*EGR
    F = (F - np.mean(F)) / (np.std(F) + 1e-6)
    F = np.nan_to_num(F, nan=0.0)

    # pairwise distances
    D = pairwise_dist(df["ra"], df["dec"]) + 1e-6

    # scan power-law exponents
    alphas = np.linspace(0.1, 4.0, 40)
    actions = []

    print("scanning power-law exponents alpha...")

    for a in alphas:
        S = nonlocal_action(F, D, a)
        actions.append(S)

    actions = np.array(actions)
    idx = np.argmin(actions)
    alpha_best = alphas[idx]
    S_best = actions[idx]

    print("\n--- best exponent ---")
    print(f"alpha_best = {alpha_best:.4f}")
    print(f"S_best     = {S_best:.6e}")

    # Monte Carlo null
    N_MC = 2000
    S_null = []

    print("running Monte Carlo null...")

    for _ in tqdm(range(N_MC)):
        Fsh = np.random.permutation(F)
        S_null.append(nonlocal_action(Fsh, D, alpha_best))

    S_null = np.array(S_null)
    mu = np.mean(S_null)
    sd = np.std(S_null)
    p = np.mean(S_null <= S_best)

    print("===============================================================")
    print("FRB POWER-LAW EXPONENT INFERENCE (TEST 61)")
    print("===============================================================")
    print(f"best alpha = {alpha_best:.4f}")
    print(f"S_real     = {S_best:.6e}")
    print(f"null mean  = {mu:.6e}")
    print(f"null std   = {sd:.6e}")
    print(f"p-value    = {p:.6f}")
    print("---------------------------------------------------------------")
    print("interpretation:")
    print(" - low p  → FRB latent field prefers this specific α (sharp law)")
    print(" - high p → α not special under null")
    print("===============================================================")
    print("test 61 complete.")
    print("===============================================================")


if __name__ == "__main__":
    main()
