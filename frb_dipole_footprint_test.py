#!/usr/bin/env python3
# ============================================================
# FRB DIPOLE FOOTPRINT-CORRECTED SIGNIFICANCE TEST
# ============================================================
# this script:
#   1. loads the main frb catalog (frbs.csv, 600 events)
#   2. computes dipole axis and amplitude r for:
#        - full sample
#        - low-z half
#        - high-z half
#   3. constructs a footprint-constrained null by resampling
#      RA and Dec from the empirical distributions
#   4. compares r_real to the footprint null for each subset
#   5. prints a scientific verdict about whether the dipole
#      anisotropy exceeds what the combined survey geometry
#      can generate.
# ============================================================

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# ------------------------------------------------------------
# config
# ------------------------------------------------------------

CATALOG_FILE = r"C:\Users\ratec\Documents\CrossLayerPhysics\frbs.csv"
N_MC = 20000  # monte carlo realisations

# unified axis from previous combined analysis (galactic)
UNIFIED_L = 159.85
UNIFIED_B = -0.51


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def load_frb_catalog(path):
    """
    load the frb catalog and keep only rows with valid ra, dec, z_est.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["ra", "dec", "z_est"])
    return df


def dipole_axis_and_r(ra_deg, dec_deg):
    """
    compute dipole amplitude r and axis direction
    for a set of positions in icrs (ra, dec in degrees).

    r = |mean unit vector|, 0 <= r <= 1.

    returns (l, b, r) in galactic coordinates.
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    M = np.vstack([x, y, z]).T
    mean_vec = M.mean(axis=0)
    r = np.linalg.norm(mean_vec)

    if r == 0:
        axis_vec = np.array([1.0, 0.0, 0.0])
    else:
        axis_vec = mean_vec / r

    c = SkyCoord(
        x=axis_vec[0],
        y=axis_vec[1],
        z=axis_vec[2],
        frame="icrs",
        representation_type="cartesian",
    )
    g = c.galactic
    return g.l.deg, g.b.deg, float(r)


def angsep(l1, b1, l2, b2):
    """
    great-circle separation between two galactic directions in degrees.
    """
    c1 = SkyCoord(l=l1 * u.deg, b=b1 * u.deg, frame="galactic")
    c2 = SkyCoord(l=l2 * u.deg, b=b2 * u.deg, frame="galactic")
    return c1.separation(c2).deg


def sample_within_footprint(ra_data, dec_data, n):
    """
    generate n synthetic frbs drawn from the empirical footprint:

    - dec distribution: sampled from the empirical dec values
    - ra distribution: sampled from the empirical ra values

    this preserves the large-scale sky coverage (bands, windows)
    of the combined surveys. a small gaussian jitter is added to
    avoid exact duplicates while keeping the footprint shape.
    """
    ra_mock = np.random.choice(ra_data, size=n, replace=True)
    dec_mock = np.random.choice(dec_data, size=n, replace=True)

    ra_mock = ra_mock + np.random.normal(0.0, 0.1, size=n)
    dec_mock = dec_mock + np.random.normal(0.0, 0.1, size=n)

    return ra_mock, dec_mock


def mc_r_footprint(ra_data, dec_data, n_events, n_mc):
    """
    monte carlo dipole amplitudes under the empirical footprint
    defined by (ra_data, dec_data).
    """
    r_vals = np.empty(n_mc, dtype=float)
    for i in range(n_mc):
        ra_mock, dec_mock = sample_within_footprint(ra_data, dec_data, n_events)
        _, _, r = dipole_axis_and_r(ra_mock, dec_mock)
        r_vals[i] = r
    return r_vals


def analyse_subset(label, ra, dec, ra_all, dec_all):
    """
    compute axis, r, and footprint-based p_dipole for a subset.

    ra_all, dec_all define the global footprint from which we
    resample mock skies. ra, dec are the subset we are testing.
    """
    n = len(ra)
    l_best, b_best, r_real = dipole_axis_and_r(ra, dec)
    sep = angsep(l_best, b_best, UNIFIED_L, UNIFIED_B)

    r_null = mc_r_footprint(ra_all, dec_all, n, N_MC)
    mean_r = r_null.mean()
    std_r = r_null.std()
    p_footprint = np.mean(r_null >= r_real)

    return {
        "label": label,
        "n": n,
        "l_best": l_best,
        "b_best": b_best,
        "r_real": r_real,
        "sep_unified": sep,
        "mean_r_null": mean_r,
        "std_r_null": std_r,
        "p_footprint": p_footprint,
    }


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    print("===================================================")
    print("FRB DIPOLE FOOTPRINT-CORRECTED SIGNIFICANCE TEST")
    print("===================================================\n")

    df = load_frb_catalog(CATALOG_FILE)
    n_total = len(df)
    print(f"loaded frbs with z_est: {n_total}")

    if n_total < 20:
        print("not enough events for this test.")
        return

    # sort by redshift and split into low/high halves
    df_sorted = df.sort_values("z_est").reset_index(drop=True)
    mid = n_total // 2

    low = df_sorted.iloc[:mid]
    high = df_sorted.iloc[mid:]

    ra_all = df_sorted["ra"].values
    dec_all = df_sorted["dec"].values

    # analyse full, low-z, high-z using the global footprint
    res_all = analyse_subset("all", ra_all, dec_all, ra_all, dec_all)
    res_low = analyse_subset("low-z half", low["ra"].values, low["dec"].values,
                             ra_all, dec_all)
    res_high = analyse_subset("high-z half", high["ra"].values, high["dec"].values,
                              ra_all, dec_all)

    # print results
    for res in [res_all, res_low, res_high]:
        print(f"\nsubset: {res['label']}")
        print(f"n = {res['n']}")
        print(f"l_best = {res['l_best']:.3f} deg")
        print(f"b_best = {res['b_best']:.3f} deg")
        print(f"dipole amplitude r_real = {res['r_real']:.4f}")
        print(f"separation from unified axis = {res['sep_unified']:.3f} deg")
        print(f"<r>_null_footprint ≈ {res['mean_r_null']:.4f}")
        print(f"std(r)_null_footprint ≈ {res['std_r_null']:.4f}")
        print(f"p_footprint ≈ {res['p_footprint']:.4f}")

    # --------------------------------------------------------
    # verdict (scientific tone)
    # --------------------------------------------------------
    print("\n---------------------------------------------------")
    print("verdict")
    print("---------------------------------------------------")

    p_all = res_all["p_footprint"]
    p_low = res_low["p_footprint"]
    p_high = res_high["p_footprint"]

    if p_all > 0.1 and p_low > 0.1 and p_high > 0.1:
        print(
            "once the empirical frb footprint is taken into account, the dipole "
            "amplitudes for the full, low-z, and high-z samples all fall within "
            "the typical range produced by the survey geometry itself. in this "
            "case, the large r values measured under a full-sky isotropic null "
            "are best interpreted as a footprint effect rather than a separate "
            "cosmic dipole."
        )
    elif p_all < 0.01 and p_low < 0.05 and p_high < 0.05:
        print(
            "even relative to the empirical footprint, the frb dipole amplitudes "
            "remain unusually large for the full sample and for both redshift "
            "halves. this indicates that the anisotropy cannot be fully absorbed "
            "into survey geometry and suggests a genuinely extended cosmic "
            "component that is stable across the probed redshift range."
        )
    else:
        print(
            "the footprint-corrected p-values show a mixed picture: some subsets "
            "exceed typical footprint-induced dipoles, while others remain "
            "compatible. this points to a partial interplay between survey "
            "geometry and intrinsic anisotropy, and motivates more detailed "
            "modelling that treats individual surveys and selection functions "
            "explicitly."
        )

    print("---------------------------------------------------")
    print("analysis complete.")
    print("===================================================\n")


if __name__ == "__main__":
    main()
