#!/usr/bin/env python3
# ============================================================
# FRB GALACTIC-PLANE MASK SHELL TEST
# ============================================================
# this script:
#   1. loads frbs.csv (600 events)
#   2. converts RA/Dec -> galactic (l, b)
#   3. computes angular distance θ from the unified axis
#   4. splits events into 4 radial bands:
#        [0–10°], [10–25°], [25–40°], [40–90°]
#   5. for several galactic latitude cuts (no cut, |b|>=10°, |b|>=20°)
#      it computes:
#        - observed counts per band
#        - isotropic expected counts per band
#        - χ² across the 4 bands
#   6. prints a scientific-style verdict about whether the
#      layered structure survives once the galactic plane is removed.
# ============================================================

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import chi2

CATALOG_FILE = r"C:\Users\ratec\Documents\CrossLayerPhysics\frbs.csv"

# unified axis in galactic coords
UNIFIED_L = 159.85  # deg
UNIFIED_B = -0.51   # deg


def load_frb_galactic(path):
    """
    load frb catalog and return galactic coordinates (l, b) in degrees.
    requires columns 'ra', 'dec' in degrees (icrs).
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["ra", "dec"])
    c_icrs = SkyCoord(ra=df["ra"].values * u.deg,
                      dec=df["dec"].values * u.deg,
                      frame="icrs")
    c_gal = c_icrs.galactic
    l = c_gal.l.deg
    b = c_gal.b.deg
    return l, b


def angdist_gal(l1, b1, l2, b2):
    """
    great-circle separation between (l1,b1) and (l2,b2) in galactic coords (deg).
    l1, b1 can be arrays; l2, b2 are scalars.
    """
    c1 = SkyCoord(l=l1 * u.deg, b=b1 * u.deg, frame="galactic")
    c2 = SkyCoord(l=l2 * u.deg, b=b2 * u.deg, frame="galactic")
    return c1.separation(c2).deg


def band_expectations_iso(n_tot, theta_edges, theta_max=90.0):
    """
    isotropic expected counts per band for a total of n_tot events,
    restricted to 0 <= theta <= theta_max.

    the cumulative probability for a spherical cap of radius θ is:
        P(<=θ) = (1 - cos θ) / 2

    so for a band [θ1, θ2], the probability mass is:
        ΔP = P(<=θ2) - P(<=θ1)

    here we renormalise within [0, theta_max].
    """
    rad = np.deg2rad(theta_edges)
    cos_th = np.cos(rad)

    # full-sphere cumulative up to each edge
    P_full = (1.0 - cos_th) / 2.0

    # renormalise within [0, theta_max]
    theta_max_rad = np.deg2rad(theta_max)
    P_max = (1.0 - np.cos(theta_max_rad)) / 2.0
    P_norm = P_full / P_max

    # band probabilities
    band_probs = np.diff(P_norm)
    mu = n_tot * band_probs
    return mu, band_probs


def shell_stats(theta, theta_edges, label):
    """
    given theta (deg) and theta bin edges,
    compute observed counts, isotropic expectations, and χ².
    """
    n_tot = len(theta)
    counts, _ = np.histogram(theta, bins=theta_edges)
    mu, _ = band_expectations_iso(n_tot, theta_edges, theta_max=theta_edges[-1])

    # avoid divide-by-zero in degenerate cases
    mask = mu > 0
    chi2_val = np.sum((counts[mask] - mu[mask])**2 / mu[mask])
    dof = mask.sum() - 1
    p_val = chi2.sf(chi2_val, dof) if dof > 0 else np.nan

    print(f"\nsubset: {label}")
    print(f"  n_FRB = {n_tot}")
    for i in range(len(counts)):
        th1 = theta_edges[i]
        th2 = theta_edges[i+1]
        print(
            f"  band {th1:4.0f}°–{th2:4.0f}° : "
            f"obs = {counts[i]:4d}, exp_iso = {mu[i]:6.2f}, "
            f"ratio = {counts[i]/mu[i] if mu[i]>0 else np.nan:5.2f}"
        )

    print(f"  total χ² (vs isotropy) = {chi2_val:.2f} (dof={dof})")
    print(f"  p-value (isotropic null) = {p_val:.3e}")
    return {
        "label": label,
        "n": n_tot,
        "counts": counts,
        "mu": mu,
        "chi2": chi2_val,
        "dof": dof,
        "p": p_val,
    }


def main():
    print("===================================================")
    print("FRB GALACTIC-PLANE MASK SHELL TEST")
    print("===================================================\n")

    # load catalog and compute galactic coords
    l, b = load_frb_galactic(CATALOG_FILE)
    n_all = len(l)
    print(f"loaded FRBs with valid positions: {n_all}")

    # angle to unified axis
    theta = angdist_gal(l, b, UNIFIED_L, UNIFIED_B)

    # radial bands as in the layered tests
    theta_edges = np.array([0.0, 10.0, 25.0, 40.0, 90.0])

    # masks in galactic latitude
    masks = {
        "all (no |b| cut)": np.ones_like(b, dtype=bool),
        "|b| >= 10°": np.abs(b) >= 10.0,
        "|b| >= 20°": np.abs(b) >= 20.0,
    }

    results = []
    for label, m in masks.items():
        th_sub = theta[m]
        if len(th_sub) < 20:
            print(f"\nsubset: {label}")
            print(f"  n_FRB = {len(th_sub)} (too few for shell χ²)")
            continue
        res = shell_stats(th_sub, theta_edges, label)
        results.append(res)

    print("\n---------------- scientific verdict ----------------")
    if len(results) >= 2:
        chi2_all = results[0]["chi2"]
        p_all = results[0]["p"]
        # find the strongest-masked subset with enough events
        best_mask = results[-1]
        chi2_mask = best_mask["chi2"]
        p_mask = best_mask["p"]
        label_mask = best_mask["label"]

        print(
            f"for the full sample (no |b| cut), the layered shell test "
            f"shows χ² ≈ {chi2_all:.1f} with p ≈ {p_all:.1e} against "
            f"a purely isotropic radial profile."
        )
        print(
            f"after excluding the galactic plane ({label_mask}), the "
            f"test yields χ² ≈ {chi2_mask:.1f} with p ≈ {p_mask:.1e}."
        )

        if p_all < 1e-3 and p_mask < 1e-3:
            print(
                "→ the layered radial excess remains highly significant "
                "even when the galactic plane is removed. this favours "
                "a geometric anisotropy that is not solely driven by "
                "milky-way foregrounds or low-latitude systematics."
            )
        elif p_all < 1e-3 and p_mask > 0.05:
            print(
                "→ once low-|b| regions are excised, the radial layering "
                "becomes statistically compatible with isotropy. this "
                "indicates that the apparent structure is strongly tied "
                "to the galactic plane and may be dominated by milky-way "
                "correlations or plane-focused selection effects."
            )
        else:
            print(
                "→ the response to galactic-plane masking is mixed. some "
                "anisotropy persists, but the significance and band "
                "structure change. further modelling that separately "
                "tracks individual survey masks and foreground cuts is "
                "needed to disentangle intrinsic structure from plane-"
                "related systematics."
            )
    else:
        print(
            "not enough subsets with sufficient events to draw a robust "
            "comparison between masked and unmasked samples."
        )
    print("---------------------------------------------------")
    print("analysis complete.")
    print("===================================================\n")


if __name__ == "__main__":
    main()
