#!/usr/bin/env python3
"""
FRB REDSHIFT-SLICED HELICAL DRIFT TEST (TEST 28) — FIXED FOR z_est

this version correctly detects your catalog's redshift column "z_est",
renames it to "z", and proceeds normally. if "z_est" is missing, it
will also detect plain "z" or "redshift". only if ALL are missing,
it falls back to DM/1000.

workflow:
1. ensure theta_unified, phi_unified exist
2. detect redshift column in this priority:
      (a) z_est
      (b) z
      (c) redshift
      (d) fallback: DM/1000
3. divide FRBs into redshift slices
4. in each slice: estimate phi_max(theta), fit phi_max = phi0 + k*theta
5. monte carlo null: shuffle phi, preserve theta+z structure
6. output k(z) and p-values for each slice
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import circmean


# ------------------------------------------------------------
# load FRBs with correct redshift handling
# ------------------------------------------------------------
def load_frbs(path):
    df = pd.read_csv(path)

    # required angular coords
    if "theta_unified" not in df.columns or "phi_unified" not in df.columns:
        raise ValueError("catalog must contain theta_unified and phi_unified")

    # detect redshift column
    if "z_est" in df.columns:
        print("detected redshift column: z_est")
        df = df.rename(columns={"z_est": "z"})
    elif "z" in df.columns:
        print("detected redshift column: z")
    elif "redshift" in df.columns:
        print("detected redshift column: redshift")
        df = df.rename(columns={"redshift": "z"})
    else:
        # DM fallback
        if "DM" in df.columns:
            print("no redshift found — using DM-based proxy z ≈ DM/1000")
            df["z"] = df["DM"] / 1000.0
        elif "dm" in df.columns:
            print("no redshift found — using dm-based proxy z ≈ dm/1000")
            df["z"] = df["dm"] / 1000.0
        else:
            raise ValueError("no redshift or DM column found — cannot run Test 28")

    df = df.dropna(subset=["theta_unified", "phi_unified", "z"])
    return df


# ------------------------------------------------------------
def estimate_phi_peaks(theta_deg, phi_deg, bin_width=10.0, min_per_bin=6):
    theta = np.asarray(theta_deg)
    phi = np.radians(phi_deg)

    if len(theta) < min_per_bin * 3:
        return None, None

    bins = np.arange(theta.min(), theta.max() + bin_width, bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])
    peaks = []

    for i in range(len(bins) - 1):
        mask = (theta >= bins[i]) & (theta < bins[i+1])
        subset = phi[mask]
        if len(subset) < min_per_bin:
            peaks.append(np.nan)
        else:
            peaks.append(circmean(subset, high=np.pi, low=-np.pi))

    return centers, np.array(peaks)


# ------------------------------------------------------------
def helix(theta, phi0, k):
    return phi0 + k * theta


# ------------------------------------------------------------
def mc_null(theta_deg, phi_deg, theta_centers, n_sims=20000):
    rng = np.random.default_rng(123)
    N = len(phi_deg)
    k_null = []

    for _ in range(n_sims):
        perm = rng.permutation(N)
        phi_sh = phi_deg[perm]

        cen, peak = estimate_phi_peaks(theta_deg, phi_sh)
        if cen is None:
            k_null.append(0.0)
            continue

        mask = ~np.isnan(peak)
        if mask.sum() < 3:
            k_null.append(0.0)
            continue

        try:
            popt, _ = curve_fit(
                helix, cen[mask], np.degrees(peak[mask]), p0=[0, 0]
            )
            k_null.append(popt[1])
        except:
            k_null.append(0.0)

    return np.array(k_null)


# ------------------------------------------------------------
def analyse_slice(name, df_slice):
    print(f"\n=== REDSHIFT SLICE {name} ===")
    print(f"{len(df_slice)} FRBs")

    theta = df_slice["theta_unified"].values.astype(float)
    phi   = df_slice["phi_unified"].values.astype(float)

    cen, peak = estimate_phi_peaks(theta, phi)
    if cen is None:
        print("insufficient FRBs for phi_max estimation.")
        return None

    mask = ~np.isnan(peak)
    if mask.sum() < 3:
        print("not enough phi_max points to fit helix.")
        return None

    print("fitting helical model ...")
    popt, _ = curve_fit(helix, cen[mask], np.degrees(peak[mask]), p0=[0, 0])
    phi0_fit, k_fit = popt
    print(f"phi0 = {phi0_fit:.3f} deg")
    print(f"k    = {k_fit:.5f} deg/deg")

    print("running Monte Carlo null ...")
    k_null = mc_null(theta, phi, cen)
    p_mc = np.mean(np.abs(k_null) >= abs(k_fit))

    print("------------------------------------------------")
    print(f"observed k      = {k_fit:.5f}")
    print(f"null mean |k|   = {np.mean(np.abs(k_null)):.5f}")
    print(f"null sd   |k|   = {np.std(np.abs(k_null)):.5f}")
    print(f"MC p-value      = {p_mc:.6f}")
    print("------------------------------------------------")

    return {"slice": name, "k": k_fit, "p": p_mc}


# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_helical_redshift_sliced_test.py frbs_unified.csv")
        sys.exit(1)

    print("="*70)
    print("FRB REDSHIFT-SLICED HELICAL DRIFT TEST (TEST 28) — FIXED")
    print("="*70)

    df = load_frbs(sys.argv[1])
    print(f"loaded {len(df)} FRBs")

    # same redshift slicing as Test 28 V1
    bins = [0.0, 0.2, 0.35, 0.55, 0.8]
    labels = [
        "z1 (0–0.2)",
        "z2 (0.2–0.35)",
        "z3 (0.35–0.55)",
        "z4 (0.55–0.8)"
    ]

    results = []

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        sl = df[(df["z"] >= lo) & (df["z"] < hi)]

        if len(sl) < 20:
            print(f"\n=== {labels[i]} ===")
            print("too few FRBs — skipping.")
            continue

        r = analyse_slice(labels[i], sl)
        results.append(r)

    print("\n==================== SUMMARY ====================")
    for r in results:
        if r:
            print(f"{r['slice']:>12}:  k = {r['k']:+.5f},  p = {r['p']:.6f}")
    print("==================================================")


if __name__ == "__main__":
    main()
