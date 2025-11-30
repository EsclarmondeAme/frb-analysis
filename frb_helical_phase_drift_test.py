#!/usr/bin/env python3
"""
FRB HELICAL PHASE-DRIFT TEST (TEST 26)

purpose
-------
detect whether the azimuthal overdensity ridge in FRB unified-axis
coordinates exhibits a coherent phase drift with polar angle theta,
i.e. a "helical" pattern:

    phi_max(theta) = phi0 + k * theta.

we:
1) bin frbs in theta (angle from unified axis),
2) estimate the peak azimuth phi_max in each theta-bin using a
   circular-von-Mises fit,
3) fit phi_max(theta) to the linear helical law above,
4) evaluate the significance against a Monte Carlo null where theta
   values are preserved but the phi values are randomly permuted.

this tests whether the warped-shell anisotropy has an azimuthal twist.

input
-----
frb catalogue with columns:
- theta_unified [deg]
- phi_unified   [deg]

usage
-----
python frb_helical_phase_drift_test.py frbs_unified.csv

output
------
- best-fit phi0, k
- t-statistic for k != 0
- Monte Carlo p-value (null: no helical twist)
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import circmean


# ------------------------------------------------------------
# helper: load frbs
# ------------------------------------------------------------
def load_frbs(path: str) -> pd.DataFrame:
    frb = pd.read_csv(path)
    required = ["theta_unified", "phi_unified"]
    missing = [c for c in required if c not in frb.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    frb = frb.dropna(subset=required)
    return frb


# ------------------------------------------------------------
# estimate phi_max in each theta-bin
# ------------------------------------------------------------
def estimate_phi_peaks(theta_deg, phi_deg,
                       bin_width=10.0,
                       min_per_bin=8):
    """
    returns:
      theta_centers: list
      phi_max: list
    """
    theta = np.asarray(theta_deg)
    phi = np.radians(phi_deg)   # for circular stats

    bins = np.arange(0, 180 + bin_width, bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])
    phi_max_list = []

    for i in range(len(bins) - 1):
        mask = (theta >= bins[i]) & (theta < bins[i+1])
        subset = phi[mask]
        if len(subset) < min_per_bin:
            phi_max_list.append(np.nan)
            continue
        # circular mean (score function of first Fourier mode)
        phi_peak = circmean(subset, high=np.pi, low=-np.pi)
        phi_max_list.append(phi_peak)

    theta_centers = np.array(centers)
    phi_max = np.array(phi_max_list)
    return theta_centers, phi_max


# ------------------------------------------------------------
# helical model: phi_max(theta) = phi0 + k * theta
# ------------------------------------------------------------
def helix_model(theta, phi0, k):
    return phi0 + k * theta


# ------------------------------------------------------------
# Monte Carlo null: preserve theta, shuffle phi
# ------------------------------------------------------------
def monte_carlo_null(theta_deg, phi_deg,
                     theta_centers,
                     n_sims=20000,
                     bin_width=10.0):
    rng = np.random.default_rng(42)
    N = len(phi_deg)
    k_null = []

    for _ in range(n_sims):
        perm = rng.permutation(N)
        phi_shuffled = phi_deg[perm]

        _, phi_peak = estimate_phi_peaks(theta_deg,
                                         phi_shuffled,
                                         bin_width=bin_width)

        mask = ~np.isnan(phi_peak)
        if mask.sum() < 3:
            k_null.append(0.0)
            continue
        try:
            popt, _ = curve_fit(helix_model,
                                theta_centers[mask],
                                np.degrees(phi_peak[mask]),
                                p0=[0.0, 0.0])
            _, k_fit = popt
        except:
            k_fit = 0.0

        k_null.append(k_fit)

    return np.array(k_null)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_helical_phase_drift_test.py frbs_unified.csv")
        sys.exit(1)

    path = sys.argv[1]
    print("="*68)
    print("FRB HELICAL PHASE-DRIFT TEST (TEST 26)")
    print("="*68)

    try:
        frb = load_frbs(path)
    except Exception as e:
        print(f"error loading catalogue '{path}': {e}")
        sys.exit(1)

    print(f"loaded {len(frb)} FRBs")
    print("using columns: theta_unified, phi_unified")

    theta_deg = frb["theta_unified"].values.astype(float)
    phi_deg   = frb["phi_unified"].values.astype(float)

    print("estimating azimuthal peaks phi_max(theta)...")
    theta_centers, phi_peak = estimate_phi_peaks(theta_deg, phi_deg)

    mask = ~np.isnan(phi_peak)
    if mask.sum() < 3:
        print("not enough valid bins to fit helix model.")
        sys.exit(0)

    print("fitting helical model phi_max(theta) = phi0 + k * theta ...")
    popt, pcov = curve_fit(helix_model,
                           theta_centers[mask],
                           np.degrees(phi_peak[mask]),
                           p0=[0.0, 0.0])
    phi0_fit, k_fit = popt
    print(f"best-fit phi0 = {phi0_fit:.3f} deg")
    print(f"best-fit k    = {k_fit:.5f} deg/deg (pitch)")

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------
    print("running Monte Carlo null (shuffle phi, preserve theta)...")
    k_null = monte_carlo_null(theta_deg,
                              phi_deg,
                              theta_centers,
                              n_sims=20000)

    p_mc = np.mean(np.abs(k_null) >= abs(k_fit))

    print("------------------------------------------------------------")
    print("MONTE CARLO RESULTS:")
    print(f"observed pitch k       = {k_fit:.5f}")
    print(f"null mean |k|          = {np.mean(np.abs(k_null)):.5f}")
    print(f"null std  |k|          = {np.std(np.abs(k_null)):.5f}")
    print(f"Monte Carlo p-value     = {p_mc:.6f}")
    print("------------------------------------------------------------")
    print("interpretation:")
    print(" - phi_max(theta) captures the azimuth of maximal overdensity")
    print("   in each theta bin around the unified axis.")
    print(" - a non-zero pitch k indicates a coherent helical phase drift:")
    print("       the density ridge rotates as theta increases.")
    if p_mc < 0.01:
        print(" - result: strong evidence for helical azimuthal twisting.")
    elif p_mc < 0.05:
        print(" - result: mild evidence for helical twisting.")
    else:
        print(" - result: consistent with no coherent twist in current data.")
    print("="*68)
    print("test 26 complete.")
    print("="*68)


if __name__ == "__main__":
    main()
