#!/usr/bin/env python3
"""
FRB RADIAL-SEGMENT HELICAL PITCH DRIFT TEST (TEST 27)

purpose
-------
detect whether the helical pitch k(theta) differs across radial
segments (distance from unified axis). this extends Test 26 by
fitting the helical model *separately* in:

    inner shell  :  0–20 deg
    middle shell : 20–40 deg
    outer shell  : 40–90 deg

for each shell we:
1) bin frbs in theta,
2) estimate phi_max(theta) using circular means,
3) fit helical law   phi_max = phi0 + k * theta,
4) assess significance via Monte Carlo (shuffle phi, preserve theta).

this reveals whether the twist is localized or varies across shells.

input
-----
frb catalogue with:
- theta_unified [deg]
- phi_unified   [deg]

usage
-----
python frb_helical_pitch_radial_test.py frbs_unified.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import circmean


# ------------------------------------------------------------
# load frbs
# ------------------------------------------------------------
def load_frbs(path):
    frb = pd.read_csv(path)
    required = ["theta_unified", "phi_unified"]
    missing = [c for c in required if c not in frb.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    frb = frb.dropna(subset=required)
    return frb


# ------------------------------------------------------------
# estimate phi_max(theta) inside a theta-subset
# ------------------------------------------------------------
def estimate_phi_peaks(theta_deg, phi_deg, bin_width=10.0, min_per_bin=6):
    theta = np.asarray(theta_deg)
    phi = np.radians(phi_deg)

    bins = np.arange(theta.min(), theta.max() + bin_width, bin_width)
    if len(bins) < 2:
        return None, None

    centers = 0.5 * (bins[:-1] + bins[1:])
    out_phi = []

    for i in range(len(bins) - 1):
        mask = (theta >= bins[i]) & (theta < bins[i+1])
        subset = phi[mask]
        if len(subset) < min_per_bin:
            out_phi.append(np.nan)
            continue
        out_phi.append(circmean(subset, high=np.pi, low=-np.pi))

    return np.array(centers), np.array(out_phi)


# ------------------------------------------------------------
# helical model
# ------------------------------------------------------------
def helix(theta, phi0, k):
    return phi0 + k * theta


# ------------------------------------------------------------
# Monte Carlo for a single shell
# ------------------------------------------------------------
def monte_carlo_k(theta_deg, phi_deg, theta_centers, n_sims=20000):
    rng = np.random.default_rng(123)
    N = len(phi_deg)
    k_null = []

    for _ in range(n_sims):
        perm = rng.permutation(N)
        phi_shuff = phi_deg[perm]

        centers, peaks = estimate_phi_peaks(theta_deg, phi_shuff)
        if centers is None:
            k_null.append(0.0)
            continue

        mask = ~np.isnan(peaks)
        if mask.sum() < 3:
            k_null.append(0.0)
            continue

        try:
            popt, _ = curve_fit(
                helix,
                centers[mask],
                np.degrees(peaks[mask]),
                p0=[0.0, 0.0]
            )
            k_null.append(popt[1])
        except:
            k_null.append(0.0)

    return np.array(k_null)


# ------------------------------------------------------------
# process a single shell
# ------------------------------------------------------------
def analyse_shell(name, theta, phi, tmin, tmax):
    print(f"\n=== {name.upper()} SHELL: theta ∈ [{tmin},{tmax}] deg ===")
    mask = (theta >= tmin) & (theta < tmax)
    if mask.sum() < 15:
        print("too few FRBs for reliable fit.\n")
        return None

    theta_sub = theta[mask]
    phi_sub = phi[mask]

    print(f"{mask.sum()} FRBs in shell.")

    centers, peaks = estimate_phi_peaks(theta_sub, phi_sub)
    if centers is None:
        print("insufficient data for phi-peak estimation.")
        return None

    mask2 = ~np.isnan(peaks)
    if mask2.sum() < 3:
        print("not enough phi-max points to fit helix.")
        return None

    print("fitting helical model phi_max = phi0 + k * theta ...")
    popt, pcov = curve_fit(
        helix,
        centers[mask2],
        np.degrees(peaks[mask2]),
        p0=[0.0, 0.0]
    )
    phi0_fit, k_fit = popt
    print(f"phi0 = {phi0_fit:.3f} deg")
    print(f"k    = {k_fit:.5f} deg/deg")

    print("running Monte Carlo null ...")
    k_null = monte_carlo_k(theta_sub, phi_sub, centers)

    p_mc = np.mean(np.abs(k_null) >= abs(k_fit))

    print("------------------------------------------------")
    print(f"observed k            = {k_fit:.5f}")
    print(f"null mean |k|         = {np.mean(np.abs(k_null)):.5f}")
    print(f"null std  |k|         = {np.std(np.abs(k_null)):.5f}")
    print(f"Monte Carlo p-value   = {p_mc:.6f}")
    print("interpretation:")
    if p_mc < 0.01:
        print(" - strong evidence for helical twist in this shell.")
    elif p_mc < 0.05:
        print(" - mild evidence for twist.")
    else:
        print(" - consistent with no twist in this shell.")
    print("------------------------------------------------")

    return {
        "shell": name,
        "phi0": phi0_fit,
        "k": k_fit,
        "p": p_mc
    }


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_helical_pitch_radial_test.py frbs_unified.csv")
        sys.exit(1)

    path = sys.argv[1]
    print("="*70)
    print("FRB RADIAL-SEGMENT HELICAL PITCH DRIFT TEST (TEST 27)")
    print("="*70)

    frb = load_frbs(path)
    print(f"loaded {len(frb)} FRBs")

    theta = frb["theta_unified"].values.astype(float)
    phi   = frb["phi_unified"].values.astype(float)

    results = []

    results.append(analyse_shell("inner",  theta, phi, 0, 20))
    results.append(analyse_shell("middle", theta, phi, 20, 40))
    results.append(analyse_shell("outer",  theta, phi, 40, 90))

    print("\n==================== SUMMARY ====================")
    for r in results:
        if r is None:
            continue
        print(f"{r['shell']:>6} shell:  k = {r['k']:+.5f},  p = {r['p']:.6f}")
    print("==================================================")


if __name__ == "__main__":
    main()
