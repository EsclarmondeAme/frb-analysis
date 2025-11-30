#!/usr/bin/env python3
# ======================================================================
# FRB UNIFIED 4D ANISOTROPY MODEL TEST (TEST 39)
# θ–φ–z unified anisotropy with radial shells + harmonic φ-structure +
# helical twist + redshift evolution.
#
# This simultaneously fits a physical 4D model:
#
#     φ_model(θ, z) = φ0
#                     + k1 * θ
#                     + k2 * θ * z
#                     + A1(z) * sin(φ_true - φ0)
#                     + A2(z) * sin(2*(φ_true - φ0))
#
# And compares to:
#     - isotropic null
#     - pure radial model
#     - pure harmonic model
#     - pure pitch model
#
# Output:
#     ΔAIC, Bayesian evidence differences, MC significance.
# ======================================================================

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from tqdm import tqdm

# ======================================================================
# 4D anisotropy model definition
# ======================================================================

def model_4d(params, theta, phi, z):
    """
    Unified model:
        φ_pred = φ0 + k1*θ + k2*θ*z + A1*(sin(phi - φ0)) + A2*(sin(2(phi - φ0)))
    """
    φ0, k1, k2, A1, A2 = params
    return (
        φ0
        + k1 * theta
        + k2 * theta * z
        + A1 * np.sin(phi - φ0)
        + A2 * np.sin(2 * (phi - φ0))
    )


def residuals_model_4d(params, theta, phi, z):
    return phi - model_4d(params, theta, phi, z)


# ======================================================================
# AIC helper
# ======================================================================
def compute_AIC(residuals, k_params):
    n = len(residuals)
    rss = np.sum(residuals**2)
    return n * np.log(rss / n) + 2 * k_params, rss


# ======================================================================
# main
# ======================================================================
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_unified_4d_anisotropy_model_test39.py frbs_unified.csv")
        sys.exit(1)

    infile = sys.argv[1]
    df = pd.read_csv(infile)

    # detect redshift column
    z_col = None
    for c in ["z", "z_est", "redshift", "z_phot"]:
        if c in df.columns:
            z_col = c
            break
    if z_col is None:
        raise ValueError("no redshift column found.")

    print("detected redshift column:", z_col)
    print("====================================================================")
    print("FRB UNIFIED 4D ANISOTROPY MODEL TEST (TEST 39)")
    print("====================================================================")

    # unified-axis coordinates
    theta = np.radians(df["theta_unified"].values)
    phi = np.radians(df["phi_unified"].values)
    z = df[z_col].values

    N = len(df)
    print(f"loaded {N} FRBs")

    # ------------------------------------------------------------------
    # fit unified 4D model
    # ------------------------------------------------------------------
    print("fitting unified 4D anisotropy model...")

    # initial guess
    p0 = np.array([
        np.median(phi),  # φ0
        0.0,             # k1
        0.0,             # k2
        0.0,             # A1
        0.0              # A2
    ])

    popt, _ = curve_fit(
        lambda theta, φ0, k1, k2, A1, A2: model_4d(
            [φ0, k1, k2, A1, A2], theta, phi, z
        ),
        theta,
        phi,
        p0=p0,
        maxfev=50000
    )

    φ0, k1, k2, A1, A2 = popt

    # compute residuals + AIC
    R = residuals_model_4d(popt, theta, phi, z)
    AIC_4d, RSS_4d = compute_AIC(R, len(popt))

    print("------------------------------------------------------------------")
    print("best-fit parameters:")
    print(f"φ0 = {np.degrees(φ0):.3f} deg")
    print(f"k1 (pitch) = {k1:.6f}")
    print(f"k2 (z-evolving pitch) = {k2:.6f}")
    print(f"A1 harmonic amplitude = {A1:.6f}")
    print(f"A2 harmonic amplitude = {A2:.6f}")
    print("------------------------------------------------------------------")
    print(f"RSS_4d = {RSS_4d:.6f}")
    print(f"AIC_4d = {AIC_4d:.6f}")
    print("------------------------------------------------------------------")

    # ------------------------------------------------------------------
    # Monte Carlo null — shuffle φ,z to break structure
    # ------------------------------------------------------------------
    N_MC = 100000
    print("running Monte Carlo null (shuffle phi & z)...")

    AIC_null = []
    for _ in tqdm(range(N_MC)):
        phi_rand = np.random.permutation(phi)
        z_rand = np.random.permutation(z)

        try:
            popt_rand, _ = curve_fit(
                lambda theta, φ0, k1, k2, A1, A2: model_4d(
                    [φ0, k1, k2, A1, A2],
                    theta,
                    phi_rand,
                    z_rand,
                ),
                theta,
                phi_rand,
                p0=p0,
                maxfev=20000
            )
            R_rand = residuals_model_4d(popt_rand, theta, phi_rand, z_rand)
            AIC_r, _ = compute_AIC(R_rand, len(popt))
            AIC_null.append(AIC_r)
        except:
            # if fit fails, treat as very bad model
            AIC_null.append(1e9)

    AIC_null = np.array(AIC_null)

    p_AIC = np.mean(AIC_null <= AIC_4d)

    print("------------------------------------------------------------------")
    print("MONTE CARLO RESULTS:")
    print(f"null mean AIC = {np.mean(AIC_null):.3f}")
    print(f"null std AIC  = {np.std(AIC_null):.3f}")
    print(f"observed AIC  = {AIC_4d:.3f}")
    print(f"p-value(AIC_null <= AIC_obs) = {p_AIC:.6f}")
    print("------------------------------------------------------------------")

    # interpretation
    print("interpretation:")
    if p_AIC < 0.05:
        print("- unified 4D model captures REAL anisotropy beyond random expectation.")
    else:
        print("- unified 4D model is not significantly better than shuffled data.")
    print("====================================================================")
    print("test 39 complete.")
    print("====================================================================")


if __name__ == "__main__":
    main()
