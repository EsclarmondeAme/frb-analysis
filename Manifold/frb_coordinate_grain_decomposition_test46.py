#!/usr/bin/env python3
"""
FRB Coordinate–Source Grain Decomposition Test (Test 46)

Purpose:
--------
Determine whether the micro-quantization / grain structure detected
in Test 45 originates from:

    (A) raw coordinates        → RA, Dec
    (B) unified-angle coords   → theta_unified, phi_unified
    (C) both
    (D) neither (artifact)

Method:
-------
For each coordinate X ∈ {RA, Dec, theta_unified, phi_unified}:

1. Compute pairwise differences ΔX.
2. Compute the Allan variance of ΔX.
3. Compute the minimum non-zero separation.
4. Generate N_MC isotropic null distributions and compute
   the Allan variance and min-sep distribution for each.

Outputs:
--------
- Allan variance anomaly p-value for RA, Dec, θ, φ
- minimum separation anomaly p-value for RA, Dec, θ, φ
"""

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

N_MC = 2000

# ------------------------------------------------------------
# Core functions
# ------------------------------------------------------------

def allan_variance(x):
    x = np.sort(x)
    diffs = np.diff(x)
    if len(diffs) < 2:
        return np.nan
    return np.mean((diffs[1:] - diffs[:-1])**2)

def min_nonzero_sep(x):
    x = np.sort(x)
    diffs = np.diff(x)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    return np.min(diffs)

def mc_null(N, A, B=None):
    """
    Generate isotropic uniform samples depending on coordinate type.
    RA ∈ [0, 360), Dec ∈ [-90, 90]
    theta_unified ∈ [A, B]
    phi_unified   ∈ [-180, 180]
    """
    if B is None:
        # RA or Dec case
        return np.random.uniform(A, A+1, size=N)  # normalized continuous axis
    else:
        # theta or phi case
        return np.random.uniform(A, B, size=N)

# ------------------------------------------------------------
# Main Test 46
# ------------------------------------------------------------

def run_test(name, data, A, B=None):
    N = len(data)
    X = np.asarray(data)
    X = np.sort(X)

    # observed statistics
    obs_allan = allan_variance(X)
    obs_minsep = min_nonzero_sep(X)

    # Monte Carlo null distributions
    allan_null = []
    minsep_null = []

    for _ in tqdm(range(N_MC), desc=f"MC for {name}"):
        if B is None:
            Xn = mc_null(N, A)
        else:
            Xn = mc_null(N, A, B)
        Xn = np.sort(Xn)

        allan_null.append(allan_variance(Xn))
        minsep_null.append(min_nonzero_sep(Xn))

    allan_null = np.array(allan_null)
    minsep_null = np.array(minsep_null)

    p_allan = np.mean(allan_null >= obs_allan)
    p_minsep = np.mean(minsep_null <= obs_minsep)

    return obs_allan, np.mean(allan_null), p_allan, obs_minsep, np.mean(minsep_null), p_minsep

# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("usage: python frb_coordinate_grain_decomposition_test46.py frbs_unified.csv")
        return

    df = pd.read_csv(sys.argv[1])
    print(f"loaded {len(df)} FRBs")

    RA  = df["ra"].values
    DEC = df["dec"].values
    TH  = df["theta_unified"].values
    PHI = df["phi_unified"].values

    print("\n============================================================")
    print("FRB COORDINATE–GRAIN DECOMPOSITION TEST (TEST 46)")
    print("============================================================\n")

    results = {}

    results["RA"] = run_test("RA", RA, 0.0)
    results["Dec"] = run_test("Dec", DEC, -90.0)
    results["theta"] = run_test("theta_unified", TH, np.min(TH), np.max(TH))
    results["phi"] = run_test("phi_unified", PHI, -180.0, 180.0)

    print("\n============================================================")
    print("SUMMARY – COORDINATE GRAIN SIGNATURES")
    print("============================================================")

    for key, (all_obs, all_null, p_allan, sep_obs, sep_null, p_sep) in results.items():
        print(f"\n-- {key} --")
        print(f"Allan variance obs = {all_obs:.4f}")
        print(f"Allan variance null mean = {all_null:.4f}")
        print(f"p(Allan anomaly) = {p_allan:.6f}")
        print(f"minimum separation obs = {sep_obs:.6f}")
        print(f"minimum separation null mean = {sep_null:.6f}")
        print(f"p(min-sep anomaly) = {p_sep:.6f}")

    print("\n============================================================")
    print("test 46 complete.")
    print("============================================================")

if __name__ == "__main__":
    main()
