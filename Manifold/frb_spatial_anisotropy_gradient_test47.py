#!/usr/bin/env python3
"""
TEST 47 — SPATIAL ANISOTROPY GRADIENT (S-AG) TEST
--------------------------------------------------

Goal:
    Detect whether micro-grain intensity varies systematically with
    angular distance from the unified axis (theta_unified).

    If grain strength increases or decreases monotonically with theta,
    this is direct evidence for a Spatial Anisotropy Gradient (S-AG).

Procedure:
    1. Compute local grain intensity G for each FRB:
           G = |Δφ| / Δθ  or angular Allan variance in a local window.

    2. Bin FRBs by theta_unified.
    3. Compute mean grain intensity per bin.
    4. Fit linear and quadratic gradient models:
           G(theta) = a + b*theta
           G(theta) = a + b*theta + c*theta^2

    5. Monte Carlo test:
        shuffle theta_unified among FRBs
        recompute gradient slopes
        estimate p-value for slope significance.

Output:
    - observed gradient slope b_obs
    - null mean & std of slopes
    - p-value for S-AG detection
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm

def compute_grain_intensity(theta, phi, window=10):
    """
    simple local micro-grain intensity estimator:
    for each FRB, measure local angular roughness of nearby points.
    """
    n = len(theta)
    G = np.zeros(n)

    for i in range(n):
        d = np.sqrt((theta - theta[i])**2 + (phi - phi[i])**2)
        neighbors = np.argsort(d)[1:window+1]
        local_theta = theta[neighbors]
        local_phi = phi[neighbors]

        # local roughness = Allan variance of angular steps
        dphi = np.diff(np.sort(local_phi))
        if len(dphi) > 1:
            allan = 0.5 * np.mean((dphi[1:] - dphi[:-1])**2)
        else:
            allan = 0

        G[i] = allan

    return G


def main():
    if len(sys.argv) < 2:
        print("usage: python frb_spatial_anisotropy_gradient_test47.py frbs_unified.csv")
        sys.exit()

    fname = sys.argv[1]
    df = pd.read_csv(fname)
    print(f"loaded {len(df)} FRBs")

    theta = df["theta_unified"].values
    phi   = df["phi_unified"].values

    # compute grain intensity for each FRB
    print("computing grain intensities...")
    G = compute_grain_intensity(theta, phi, window=10)

    # fit observed gradient
    slope_obs, intercept, r, p_lin, stderr = linregress(theta, G)

    # Monte Carlo null test (shuffle theta)
    N_MC = 5000
    slopes_null = np.zeros(N_MC)

    print(f"running Monte Carlo (N={N_MC})...")
    for i in tqdm(range(N_MC)):
        theta_shuf = np.random.permutation(theta)
        slopes_null[i], _, _, _, _ = linregress(theta_shuf, G)

    mean_null = np.mean(slopes_null)
    std_null  = np.std(slopes_null)

    # p-value: probability null slope >= observed |slope|
    p_value = np.mean(np.abs(slopes_null) >= np.abs(slope_obs))

    # results
    print("====================================================================")
    print("FRB SPATIAL ANISOTROPY GRADIENT TEST (TEST 47)")
    print("====================================================================")
    print(f"observed gradient slope         = {slope_obs:.6f}")
    print(f"null mean slope                 = {mean_null:.6f}")
    print(f"null std slope                  = {std_null:.6f}")
    print(f"p-value(|slope_null| >= |obs|)  = {p_value:.6f}")
    print("--------------------------------------------------------------------")
    print("interpretation:")
    if p_value < 0.05:
        print("  strong evidence for anisotropic grain gradient (S-AG).")
    elif p_value < 0.20:
        print("  weak trend: grain strength slightly directional.")
    else:
        print("  no significant anisotropic grain gradient detected.")
    print("====================================================================")
    print("test 47 complete.")
    print("====================================================================")


if __name__ == "__main__":
    main()
