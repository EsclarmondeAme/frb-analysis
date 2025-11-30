#!/usr/bin/env python3
"""
FRB HELICAL M-MODE REDSHIFT-SPLIT TEST (TEST 29)

purpose
-------
quantify the m=1 and m=2 azimuthal harmonic structure within the
main warped shell around the unified axis, and see how it changes
with redshift.

this is a direct "double-helix" diagnostic:
- m = 1  -> single lopsided ridge (one helix),
- m = 2  -> two opposite ridges (double helix).

we:
1) restrict to 25° <= theta_unified <= 60° (main shell),
2) split the catalogue into redshift slices,
3) bin phi_unified into a histogram in each slice,
4) fit three models to counts(phi):

   pure radial:
       N(phi) = A0

   m=1:
       N(phi) = A0 + A1*cos(phi - phi0)

   m=1 + m=2:
       N(phi) = A0 + A1*cos(phi - phi0) + A2*cos(2*(phi - phi0))

5) compare models using AIC, and evaluate the significance of the
   m=1+m=2 model vs pure radial via Monte Carlo, by randomizing
   phi positions under an isotropic null for each slice.

this tells us whether:
- the strong helical slice z ~ 0.2–0.35 has enhanced m=1/m=2 power,
- higher or lower redshift slices show weaker or different structure,
- the m=2 component (double helix) is important.

input
-----
- frbs_unified.csv with columns:
    theta_unified [deg]
    phi_unified   [deg]
    z_est or z or redshift

usage
-----
python frb_helical_mmode_redshift_test.py frbs_unified.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# shell geometry
THETA_MIN = 25.0   # degrees
THETA_MAX = 60.0   # degrees

# number of phi bins
N_PHI_BINS = 24

# Monte Carlo settings
N_MC = 20000


# ------------------------------------------------------------
# helpers: loading and redshift handling
# ------------------------------------------------------------
def load_frbs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # check unified-axis coordinates
    required = ["theta_unified", "phi_unified"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    # detect redshift column: z_est, z, redshift
    if "z_est" in df.columns:
        print("detected redshift column: z_est")
        df = df.rename(columns={"z_est": "z"})
    elif "z" in df.columns:
        print("detected redshift column: z")
    elif "redshift" in df.columns:
        print("detected redshift column: redshift")
        df = df.rename(columns={"redshift": "z"})
    else:
        raise ValueError("no redshift column (z_est/z/redshift) found in catalogue.")

    df = df.dropna(subset=["theta_unified", "phi_unified", "z"])
    return df


# ------------------------------------------------------------
# model definitions
# ------------------------------------------------------------
def pure_radial(phi, A0):
    """phi-independent baseline."""
    return A0 + 0.0 * phi


def m1_model(phi, A0, A1, phi0):
    """m=1: A0 + A1*cos(phi - phi0)."""
    return A0 + A1 * np.cos(phi - phi0)


def m1m2_model(phi, A0, A1, A2, phi0):
    """
    m=1 + m=2 with a common phase phi0:
    A0 + A1*cos(phi - phi0) + A2*cos(2*(phi - phi0)).
    """
    return A0 + A1 * np.cos(phi - phi0) + A2 * np.cos(2.0 * (phi - phi0))


def AIC_from_RSS(n, rss, k):
    """akaike information criterion from rss and k parameters."""
    if rss <= 0:
        rss = 1e-12
    return 2 * k + n * np.log(rss / n)


# ------------------------------------------------------------
# fit counts(phi) with the three models
# ------------------------------------------------------------
def fit_models(phi_centers, counts, label=""):
    """
    phi_centers: radians
    counts: counts per bin
    returns dict with AICs, params, etc.
    """
    n = len(phi_centers)
    x = np.asarray(phi_centers, dtype=float)
    y = np.asarray(counts, dtype=float)

    results = {}

    # pure radial
    p0_rad = [np.mean(y)]
    popt_rad, _ = curve_fit(pure_radial, x, y, p0=p0_rad, maxfev=20000)
    y_rad = pure_radial(x, *popt_rad)
    rss_rad = np.sum((y - y_rad) ** 2)
    aic_rad = AIC_from_RSS(n, rss_rad, k=1)

    results["pure"] = {
        "name": "pure_radial",
        "aic": aic_rad,
        "rss": rss_rad,
        "params": popt_rad,
    }

    # m=1
    try:
        p0_m1 = [np.mean(y), 0.1 * np.max(y), 0.0]
        popt_m1, _ = curve_fit(m1_model, x, y, p0=p0_m1, maxfev=20000)
        y_m1 = m1_model(x, *popt_m1)
        rss_m1 = np.sum((y - y_m1) ** 2)
        aic_m1 = AIC_from_RSS(n, rss_m1, k=3)
    except Exception:
        popt_m1 = None
        rss_m1 = np.inf
        aic_m1 = np.inf

    results["m1"] = {
        "name": "m1",
        "aic": aic_m1,
        "rss": rss_m1,
        "params": popt_m1,
    }

    # m=1 + m=2
    try:
        p0_m12 = [np.mean(y), 0.1 * np.max(y), 0.1 * np.max(y), 0.0]
        popt_m12, _ = curve_fit(m1m2_model, x, y, p0=p0_m12, maxfev=30000)
        y_m12 = m1m2_model(x, *popt_m12)
        rss_m12 = np.sum((y - y_m12) ** 2)
        aic_m12 = AIC_from_RSS(n, rss_m12, k=4)
    except Exception:
        popt_m12 = None
        rss_m12 = np.inf
        aic_m12 = np.inf

    results["m1m2"] = {
        "name": "m1+m2",
        "aic": aic_m12,
        "rss": rss_m12,
        "params": popt_m12,
    }

    return results


# ------------------------------------------------------------
# monte carlo for one slice
# ------------------------------------------------------------
def mc_significance(phi_raw, n_bins, results_real, n_mc=N_MC):
    """
    phi_raw: FRB phi_unified (radians) in this slice/shell
    results_real: fit_models result for real histogram
    returns p-value for (m1+m2 vs pure radial).
    """
    if results_real["m1m2"]["aic"] == np.inf:
        return 1.0

    delta_real = results_real["pure"]["aic"] - results_real["m1m2"]["aic"]

    delta_null = []
    n_events = len(phi_raw)
    rng = np.random.default_rng(123)

    for _ in range(n_mc):
        # isotropic null: uniform phi in [-pi, pi)
        phi_iso = rng.uniform(-np.pi, np.pi, size=n_events)
        counts_iso, bin_edges = np.histogram(phi_iso, bins=n_bins, range=(-np.pi, np.pi))
        centers_iso = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        res_iso = fit_models(centers_iso, counts_iso)
        if res_iso["m1m2"]["aic"] == np.inf:
            delta_null.append(0.0)
            continue

        d = res_iso["pure"]["aic"] - res_iso["m1m2"]["aic"]
        delta_null.append(d)

    delta_null = np.array(delta_null)
    p_mc = np.mean(delta_null >= delta_real)
    return p_mc


# ------------------------------------------------------------
# analyze one redshift slice
# ------------------------------------------------------------
def analyse_slice(name, df_slice):
    print(f"\n=== REDSHIFT SLICE {name} ===")
    print(f"total FRBs in slice: {len(df_slice)}")

    # restrict to shell
    theta = df_slice["theta_unified"].values.astype(float)
    phi_deg = df_slice["phi_unified"].values.astype(float)

    mask_shell = (theta >= THETA_MIN) & (theta <= THETA_MAX)
    phi_shell_deg = phi_deg[mask_shell]

    print(f"FRBs in shell {THETA_MIN:.1f}–{THETA_MAX:.1f} deg: {len(phi_shell_deg)}")

    if len(phi_shell_deg) < 30:
        print("not enough FRBs in shell for reliable harmonic fit.")
        return None

    phi_shell = np.deg2rad(phi_shell_deg)

    # histogram
    counts, bin_edges = np.histogram(phi_shell,
                                     bins=N_PHI_BINS,
                                     range=(-np.pi, np.pi))
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if np.sum(counts > 0) < 4:
        print("phi histogram too sparse for fitting.")
        return None

    # fit models
    results = fit_models(centers, counts, label=name)

    a_pure = results["pure"]["aic"]
    a_m1 = results["m1"]["aic"]
    a_m12 = results["m1m2"]["aic"]

    print("------------------------------------------------")
    print("model fits (AIC, RSS):")
    print(f"pure radial:   AIC={a_pure:8.2f}  RSS={results['pure']['rss']:8.2f}")
    print(f"m=1:           AIC={a_m1:8.2f}  RSS={results['m1']['rss']:8.2f}")
    print(f"m=1 + m=2:     AIC={a_m12:8.2f}  RSS={results['m1m2']['rss']:8.2f}")

    best_aic = min(a_pure, a_m1, a_m12)
    if best_aic == a_m12:
        best_name = "m=1 + m=2"
    elif best_aic == a_m1:
        best_name = "m=1"
    else:
        best_name = "pure radial"

    print(f"best model by AIC: {best_name}")

    # extract harmonic parameters if available
    A1 = None
    A2 = None
    phi0_deg = None
    if results["m1m2"]["params"] is not None:
        A0_fit, A1_fit, A2_fit, phi0_fit = results["m1m2"]["params"]
        A1 = A1_fit
        A2 = A2_fit
        phi0_deg = np.degrees(phi0_fit)

        ratio = np.nan
        if A1 != 0:
            ratio = A2 / A1

        print("------------------------------------------------")
        print("m=1+m=2 parameters (double-helix diagnostics):")
        print(f"A0  = {A0_fit:.3f}")
        print(f"A1  = {A1_fit:.3f}  (m=1 amplitude)")
        print(f"A2  = {A2_fit:.3f}  (m=2 amplitude)")
        print(f"phi0 (deg) = {phi0_deg:.2f}")
        print(f"A2/A1      = {ratio:.3f}")
    else:
        print("could not fit m=1+m=2 model reliably.")

    # monte carlo significance (m1+m2 vs pure radial)
    print("running Monte Carlo null (m=1+m=2 vs pure radial)...")
    p_mc = mc_significance(phi_shell, N_PHI_BINS, results)

    delta_real = results["pure"]["aic"] - results["m1m2"]["aic"]
    print("------------------------------------------------")
    print(f"ΔAIC_real (pure - m1+m2) = {delta_real:.3f}")
    print(f"MC p-value                = {p_mc:.6f}")
    print("scientific interpretation:")
    if p_mc < 0.01:
        print(" - strong evidence for m=1+m=2 azimuthal structure "
              "in this redshift slice (double-helix like pattern).")
    elif p_mc < 0.05:
        print(" - mild evidence that combined m=1+m=2 structure "
              "outperforms a pure radial model.")
    else:
        print(" - azimuthal structure in this shell is consistent "
              "with an isotropic (pure radial) null.")

    return {
        "slice": name,
        "n_shell": len(phi_shell_deg),
        "aic_pure": a_pure,
        "aic_m1": a_m1,
        "aic_m12": a_m12,
        "A1": A1,
        "A2": A2,
        "phi0_deg": phi0_deg,
        "p_mc": p_mc,
        "delta_aic": delta_real,
    }


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_helical_mmode_redshift_test.py frbs_unified.csv")
        sys.exit(1)

    path = sys.argv[1]

    print("=" * 70)
    print("FRB HELICAL M-MODE REDSHIFT-SPLIT TEST (TEST 29)")
    print("=" * 70)

    df = load_frbs(path)
    print(f"loaded {len(df)} FRBs")

    # same redshift binning as test 28 (so results align)
    bins = [0.0, 0.2, 0.35, 0.55, 0.8]
    labels = [
        "z1 (0–0.2)",
        "z2 (0.2–0.35)",
        "z3 (0.35–0.55)",
        "z4 (0.55–0.8)",
    ]

    results = []

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        sl = df[(df["z"] >= lo) & (df["z"] < hi)]
        if len(sl) < 30:
            print(f"\n=== REDSHIFT SLICE {labels[i]} ===")
            print("too few FRBs in this slice for shell analysis.")
            continue
        r = analyse_slice(labels[i], sl)
        results.append(r)

    print("\n==================== SUMMARY ====================")
    for r in results:
        if r is None:
            continue
        print(
            f"{r['slice']:>12}: "
            f"n_shell={r['n_shell']:3d}, "
            f"ΔAIC={r['delta_aic']:+6.2f}, "
            f"p={r['p_mc']:.6f}, "
            f"A1={r['A1'] if r['A1'] is not None else np.nan:+.3f}, "
            f"A2={r['A2'] if r['A2'] is not None else np.nan:+.3f}, "
            f"phi0={r['phi0_deg'] if r['phi0_deg'] is not None else np.nan:.2f} deg"
        )
    print("==================================================")


if __name__ == "__main__":
    main()
