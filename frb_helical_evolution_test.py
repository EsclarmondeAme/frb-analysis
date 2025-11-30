#!/usr/bin/env python3
"""
FRB COSMIC TWIST EVOLUTION TEST (TEST 30)

combines:
- pitch evolution k(z)
- harmonic evolution A1(z), A2(z)
- phase evolution phi0(z)
- ΔAIC(z)
- MC significance

goal: reconstruct cosmic helical anisotropy evolution with redshift.
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import circmean

# parameters
THETA_MIN = 25.0
THETA_MAX = 60.0
N_PHI_BINS = 24
N_MC = 20000  # upgraded MC precision


# ------------------------------------------------------------
# load FRBs
# ------------------------------------------------------------
def load_frbs(path):
    df = pd.read_csv(path)

    if "theta_unified" not in df.columns or "phi_unified" not in df.columns:
        raise ValueError("catalog must contain theta_unified and phi_unified")

    if "z_est" in df.columns:
        print("detected redshift column: z_est")
        df = df.rename(columns={"z_est": "z"})
    elif "z" in df.columns:
        print("detected redshift column: z")
    elif "redshift" in df.columns:
        print("detected redshift column: redshift")
        df = df.rename(columns={"redshift": "z"})
    else:
        raise ValueError("no redshift column (z_est/z/redshift) found.")

    df = df.dropna(subset=["theta_unified", "phi_unified", "z"])
    return df


# ------------------------------------------------------------
# phi_max(theta) extraction
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
        mask = (theta >= bins[i]) & (theta < bins[i + 1])
        vals = phi[mask]
        if len(vals) < min_per_bin:
            peaks.append(np.nan)
        else:
            peaks.append(circmean(vals, high=np.pi, low=-np.pi))

    return centers, np.array(peaks)


def helix(theta, phi0, k):
    return phi0 + k * theta


def mc_null_pitch(theta_deg, phi_deg, theta_centers, n_sims=20000):
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


def measure_pitch(df_slice):
    theta = df_slice["theta_unified"].values
    phi = df_slice["phi_unified"].values

    cen, peak = estimate_phi_peaks(theta, phi)
    if cen is None:
        return None, None, None

    mask = ~np.isnan(peak)
    if mask.sum() < 3:
        return None, None, None

    popt, _ = curve_fit(
        helix, cen[mask], np.degrees(peak[mask]), p0=[0, 0]
    )
    phi0_fit, k_fit = popt

    k_null = mc_null_pitch(theta, phi, cen, n_sims=N_MC)
    p_mc = np.mean(np.abs(k_null) >= abs(k_fit))

    return k_fit, phi0_fit, p_mc


# ------------------------------------------------------------
# harmonic fitting (from Test 29)
# ------------------------------------------------------------
def pure(phi, A0):
    return A0 + 0 * phi


def m1(phi, A0, A1, phi0):
    return A0 + A1 * np.cos(phi - phi0)


def m1m2(phi, A0, A1, A2, phi0):
    return A0 + A1 * np.cos(phi - phi0) + A2 * np.cos(2 * (phi - phi0))


def AIC(n, rss, k):
    if rss <= 0:
        rss = 1e-12
    return 2 * k + n * np.log(rss / n)


def fit_harmonics(phi_deg):
    phi = np.deg2rad(phi_deg)

    counts, edges = np.histogram(phi, bins=N_PHI_BINS, range=(-np.pi, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    x = centers
    y = counts
    n = len(x)

    results = {}

    try:
        popt_p, _ = curve_fit(pure, x, y, p0=[np.mean(y)])
        y_p = pure(x, *popt_p)
        rss_p = np.sum((y - y_p) ** 2)
        aic_p = AIC(n, rss_p, 1)
    except:
        return None
    results["pure"] = {"aic": aic_p, "params": popt_p, "rss": rss_p}

    try:
        popt_m1, _ = curve_fit(
            m1, x, y, p0=[np.mean(y), 0.5 * np.max(y), 0], maxfev=30000
        )
        y_m1 = m1(x, *popt_m1)
        rss_m1 = np.sum((y - y_m1) ** 2)
        aic_m1 = AIC(n, rss_m1, 3)
    except:
        popt_m1 = None
        aic_m1 = np.inf
        rss_m1 = np.inf
    results["m1"] = {"aic": aic_m1, "params": popt_m1, "rss": rss_m1}

    try:
        popt_m12, _ = curve_fit(
            m1m2,
            x,
            y,
            p0=[np.mean(y), 0.5 * np.max(y), 0.5 * np.max(y), 0],
            maxfev=40000,
        )
        y_m12 = m1m2(x, *popt_m12)
        rss_m12 = np.sum((y - y_m12) ** 2)
        aic_m12 = AIC(n, rss_m12, 4)
    except:
        popt_m12 = None
        aic_m12 = np.inf
        rss_m12 = np.inf
    results["m1m2"] = {"aic": aic_m12, "params": popt_m12, "rss": rss_m12}

    return results


def mc_significance(phi_deg, results):
    if results["m1m2"]["aic"] == np.inf:
        return None

    delta_real = results["pure"]["aic"] - results["m1m2"]["aic"]

    phi = np.deg2rad(phi_deg)
    N = len(phi)
    rng = np.random.default_rng(123)
    delta_null = []

    for _ in range(N_MC):
        phi_null = rng.uniform(-np.pi, np.pi, size=N)
        counts_null, edges_null = np.histogram(
            phi_null, bins=N_PHI_BINS, range=(-np.pi, np.pi)
        )
        centers_null = 0.5 * (edges_null[:-1] + edges_null[1:])
        res_null = fit_harmonics(np.rad2deg(centers_null))
        if res_null is None:
            continue
        delta_null.append(
            res_null["pure"]["aic"] - res_null["m1m2"]["aic"]
        )

    if len(delta_null) == 0:
        return None

    delta_null = np.array(delta_null)
    p_mc = np.mean(delta_null >= delta_real)
    return p_mc


# ------------------------------------------------------------
# combine slice analysis
# ------------------------------------------------------------
def analyse_slice(df_slice):
    theta = df_slice["theta_unified"].values
    phi = df_slice["phi_unified"].values

    mask = (theta >= THETA_MIN) & (theta <= THETA_MAX)
    phi_shell = phi[mask]

    if len(phi_shell) < 20:
        return None

    harm = fit_harmonics(phi_shell)
    if harm is None:
        return None

    p_mc_harm = mc_significance(phi_shell, harm)

    if harm["m1m2"]["params"] is not None:
        A0, A1, A2, phi0 = harm["m1m2"]["params"]
    else:
        A0 = A1 = A2 = phi0 = None

    deltaAIC = harm["pure"]["aic"] - harm["m1m2"]["aic"]

    k_fit, k_phi0, p_k = measure_pitch(df_slice)

    return {
        "n_shell": len(phi_shell),
        "A1": A1,
        "A2": A2,
        "A2/A1": A2 / A1 if A1 not in (0, None) else None,
        "phi0": phi0,
        "ΔAIC": deltaAIC,
        "p_harm": p_mc_harm,
        "k": k_fit,
        "k_phi0": k_phi0,
        "p_k": p_k,
    }


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_helical_evolution_test.py frbs_unified.csv")
        return

    path = sys.argv[1]
    print("======================================================================")
    print("FRB COSMIC TWIST EVOLUTION TEST (TEST 30)")
    print("======================================================================")

    df = load_frbs(path)
    print(f"loaded {len(df)} FRBs")
    print("")

    slices = {
        "z1 (0–0.2)":  (0.0, 0.2),
        "z2 (0.2–0.35)": (0.2, 0.35),
        "z3 (0.35–0.55)": (0.35, 0.55),
        "z4 (0.55–0.8)":  (0.55, 0.8),
    }

    results = {}

    for label, (zmin, zmax) in slices.items():
        print(f"=== processing {label} ===")
        df_slice = df[(df["z"] >= zmin) & (df["z"] < zmax)]
        print(f"total FRBs in slice: {len(df_slice)}")

        out = analyse_slice(df_slice)
        results[label] = out
        print("done.\n")

    print("======================================================================")
    print("SUMMARY – COSMIC TWIST EVOLUTION")
    print("======================================================================")

    for label, r in results.items():
        if r is None:
            print(f"{label}: insufficient data")
            continue

        # safe formatting for A2/A1
        if r["A2/A1"] is None:
            ratio = "None"
        else:
            ratio = f"{r['A2/A1']:.3f}"

        # safe formatting for k
        if r["k"] is None:
            k_str = "None"
        else:
            k_str = f"{r['k']:.4f}"

        # safe formatting for p_k
        if r["p_k"] is None:
            pk_str = "None"
        else:
            pk_str = f"{r['p_k']:.6f}"

        # safe formatting for p_harm
        if r["p_harm"] is None:
            ph_str = "None"
        else:
            ph_str = f"{r['p_harm']:.6f}"

        print(
            f"{label}: "
            f"n={r['n_shell']}, "
            f"A1={r['A1']:.3f}  "
            f"A2={r['A2']:.3f}  "
            f"A2/A1={ratio}  "
            f"phi0={r['phi0']:.2f} deg,  "
            f"k={k_str},  p_k={pk_str},  "
            f"ΔAIC={r['ΔAIC']:.3f},  p_harm={ph_str}"
        )

    print("======================================================================")
    print("test 30 complete.")
    print("======================================================================")

if __name__ == "__main__":
    main()
