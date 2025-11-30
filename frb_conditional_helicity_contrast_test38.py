#!/usr/bin/env python3
"""
FRB CONDITIONAL VS UNCONDITIONAL HELICITY CONTRAST TEST (TEST 38)

purpose
-------
quantify how the helical pitch signal depends on *where* in (theta, z)
space we look.

we compare:
    - an "unconditional" global window
    - outer theta shells
    - theta+redshift windows where structure is strongest

for each window, we:
    1) estimate phi_max(theta) via circular means
    2) fit a linear helical model: phi_max(theta) = phi0 + k * theta
    3) run a Monte Carlo null by shuffling phi while preserving theta
    4) compute a p-value for |k|

this test explicitly contrasts:
    - weak, global helicity averaged over all FRBs
    - strong, conditional helicity in the physically relevant shells.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import circmean
from scipy.optimize import curve_fit

# configuration
BIN_WIDTH = 10.0
MIN_PER_BIN = 6
N_MC = 20000  # Monte Carlo iterations


# ---------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------
def load_frbs(path: str) -> pd.DataFrame:
    """
    load FRB catalog and standardise column names.
    requires theta_unified and phi_unified.
    detects z_est / z / redshift and renames to 'z'.
    """
    df = pd.read_csv(path)

    if "theta_unified" not in df.columns or "phi_unified" not in df.columns:
        raise ValueError("catalog must contain theta_unified and phi_unified")

    # detect redshift
    if "z_est" in df.columns:
        print("detected redshift column: z_est")
        df = df.rename(columns={"z_est": "z"})
    elif "z" in df.columns:
        print("detected redshift column: z")
    elif "redshift" in df.columns:
        print("detected redshift column: redshift")
        df = df.rename(columns={"redshift": "z"})
    else:
        print("warning: no redshift column (z_est/z/redshift) found; setting z = NaN.")
        df["z"] = np.nan

    df = df.dropna(subset=["theta_unified", "phi_unified"])
    return df


def estimate_phi_peaks(theta_deg, phi_deg,
                       bin_width: float = BIN_WIDTH,
                       min_per_bin: int = MIN_PER_BIN):
    """
    estimate phi_max(theta) by binning in theta and taking
    the circular mean of phi in each bin.
    """
    theta = np.asarray(theta_deg)
    phi = np.asarray(phi_deg)

    if theta.size < min_per_bin * 3:
        return None, None

    phi_rad = np.radians(phi)

    bins = np.arange(theta.min(), theta.max() + bin_width, bin_width)
    if bins.size < 4:
        return None, None

    centers = 0.5 * (bins[:-1] + bins[1:])
    peaks = []

    for i in range(len(bins) - 1):
        mask = (theta >= bins[i]) & (theta < bins[i + 1])
        vals = phi_rad[mask]
        if vals.size < min_per_bin:
            peaks.append(np.nan)
        else:
            peaks.append(circmean(vals, high=np.pi, low=-np.pi))

    return centers, np.array(peaks)


def helix(theta_deg, phi0, k):
    """linear helical model: phi(theta) = phi0 + k * theta."""
    return phi0 + k * theta_deg


def measure_pitch_with_mc(theta_deg,
                          phi_deg,
                          n_mc: int = N_MC):
    """
    measure helical pitch k and MC significance for a given
    (theta, phi) sample.

    returns:
        k_obs, mean_null, std_null, p_value
    or
        (None, None, None, None) if measurement fails.
    """
    theta = np.asarray(theta_deg)
    phi = np.asarray(phi_deg)

    # step 1: estimate phi_max(theta)
    centers, peaks = estimate_phi_peaks(theta, phi)
    if centers is None:
        return None, None, None, None

    mask = ~np.isnan(peaks)
    if mask.sum() < 3:
        return None, None, None, None

    # step 2: fit helix
    try:
        popt, _ = curve_fit(
            helix,
            centers[mask],
            np.degrees(peaks[mask]),
            p0=[0.0, 0.0],
        )
    except Exception:
        return None, None, None, None

    phi0_fit, k_obs = popt

    # step 3: MC null (shuffle phi)
    rng = np.random.default_rng(12345)
    k_null = []

    for _ in range(n_mc):
        phi_sh = rng.permutation(phi)
        cen_sh, peak_sh = estimate_phi_peaks(theta, phi_sh)
        if cen_sh is None:
            continue
        mask_sh = ~np.isnan(peak_sh)
        if mask_sh.sum() < 3:
            continue
        try:
            popt_sh, _ = curve_fit(
                helix,
                cen_sh[mask_sh],
                np.degrees(peak_sh[mask_sh]),
                p0=[0.0, 0.0],
            )
            k_sh = popt_sh[1]
            k_null.append(abs(k_sh))
        except Exception:
            continue

    if len(k_null) == 0:
        return k_obs, None, None, None

    k_null = np.array(k_null)
    mean_null = float(np.mean(k_null))
    std_null = float(np.std(k_null))
    p_value = float(np.mean(k_null >= abs(k_obs)))

    return k_obs, mean_null, std_null, p_value


# ---------------------------------------------------------------------
# define windows
# ---------------------------------------------------------------------
def select_window(df,
                  theta_min=None,
                  theta_max=None,
                  z_min=None,
                  z_max=None):
    """apply theta/z cuts to define a window."""
    mask = np.ones(len(df), dtype=bool)

    if theta_min is not None:
        mask &= df["theta_unified"].values >= theta_min
    if theta_max is not None:
        mask &= df["theta_unified"].values <= theta_max

    if z_min is not None:
        mask &= df["z"].values >= z_min
    if z_max is not None:
        mask &= df["z"].values <= z_max

    return df[mask].copy()


def main():
    if len(sys.argv) != 2:
        print("usage: python frb_conditional_helicity_contrast_test38.py frbs_unified.csv")
        sys.exit(1)

    path = sys.argv[1]
    df = load_frbs(path)

    print("====================================================================")
    print(" FRB CONDITIONAL VS UNCONDITIONAL HELICITY CONTRAST TEST (TEST 38)")
    print("====================================================================")
    print(f"loaded {len(df)} FRBs total")
    print()

    # define a set of windows to compare
    windows = [
        {
            "name": "global (all θ, all z)",
            "theta_min": None,
            "theta_max": None,
            "z_min": None,
            "z_max": None,
        },
        {
            "name": "outer shell (40–90 deg, all z)",
            "theta_min": 40.0,
            "theta_max": 90.0,
            "z_min": None,
            "z_max": None,
        },
        {
            "name": "mid shell (25–60 deg, 0.2–0.35)",
            "theta_min": 25.0,
            "theta_max": 60.0,
            "z_min": 0.2,
            "z_max": 0.35,
        },
        {
            "name": "mid shell (25–60 deg, 0.35–0.55)",
            "theta_min": 25.0,
            "theta_max": 60.0,
            "z_min": 0.35,
            "z_max": 0.55,
        },
        {
            "name": "mid shell (25–60 deg, 0.55–0.8)",
            "theta_min": 25.0,
            "theta_max": 60.0,
            "z_min": 0.55,
            "z_max": 0.8,
        },
    ]

    results = []

    for w in windows:
        sub = select_window(
            df,
            theta_min=w["theta_min"],
            theta_max=w["theta_max"],
            z_min=w["z_min"],
            z_max=w["z_max"],
        )
        n_sub = len(sub)

        print("--------------------------------------------------------------------")
        print(f"window: {w['name']}")
        print(f"  N = {n_sub}")

        if n_sub < MIN_PER_BIN * 3:
            print("  not enough FRBs for a stable helicity estimate; skipping.")
            results.append({
                "name": w["name"],
                "N": n_sub,
                "k_obs": None,
                "mean_null": None,
                "std_null": None,
                "p_value": None,
            })
            continue

        k_obs, mean_null, std_null, p_value = measure_pitch_with_mc(
            sub["theta_unified"].values,
            sub["phi_unified"].values,
            n_mc=N_MC,
        )

        if k_obs is None:
            print("  helicity fit failed in this window; skipping.")
        else:
            print(f"  k_obs           = {k_obs:.5f} deg/deg")
            if mean_null is not None:
                print(f"  null mean |k|   = {mean_null:.5f}")
                print(f"  null std  |k|   = {std_null:.5f}")
                print(f"  p(|k_null|>=|k_obs|) = {p_value:.6f}")
            else:
                print("  MC null failed (no valid realisations).")

        results.append({
            "name": w["name"],
            "N": n_sub,
            "k_obs": k_obs,
            "mean_null": mean_null,
            "std_null": std_null,
            "p_value": p_value,
        })

    print("====================================================================")
    print(" SUMMARY – CONDITIONAL VS UNCONDITIONAL HELICITY")
    print("====================================================================")
    for r in results:
        name = r["name"]
        N = r["N"]
        k = r["k_obs"]
        pmc = r["p_value"]
        if k is None or pmc is None:
            print(f"{name}: N={N}, k=None, p=None")
        else:
            print(f"{name}: N={N}, k={k:.5f} deg/deg, p={pmc:.6f}")
    print("====================================================================")
    print(" test 38 complete.")
    print("====================================================================")


if __name__ == "__main__":
    main()
