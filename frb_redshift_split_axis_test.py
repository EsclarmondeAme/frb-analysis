#!/usr/bin/env python3
# ============================================================
# FRB REDSHIFT-SPLIT AXIS AND ANISOTROPY TEST
# ============================================================
# this script:
#   1. loads the main frb catalog with z_est
#   2. splits the sample into low-z and high-z halves
#   3. computes dipole axes and amplitudes for:
#        - full sample
#        - low-z
#        - high-z
#   4. compares their alignment with the unified axis
#   5. runs isotropic monte carlo to get p-values for each
#
# columns expected:
#   name, utc, mjd, ra, dec, dm, snr, width, fluence, z_est
#
# output:
#   - printed summary and scientific verdict
#
# ============================================================

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# ------------------------------------------------------------
# config
# ------------------------------------------------------------

CATALOG_FILE = r"C:\Users\ratec\Documents\CrossLayerPhysics\frbs.csv"


# unified axis from previous combined analysis (galactic)
UNIFIED_L = 159.85
UNIFIED_B = -0.51

N_MC = 20000  # monte carlo realisations


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def load_frb_catalog(path):
    """
    load the frb catalog and keep only rows with valid ra, dec, z_est.
    """
    df = pd.read_csv(path)
    # basic sanity filter
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


def mc_dipole_r(n_events, n_mc):
    """
    isotropic null: sample n_events directions on full sky,
    compute dipole amplitude r, repeat n_mc times.
    """
    r_values = np.empty(n_mc, dtype=float)

    for i in range(n_mc):
        u_rand = np.random.uniform(-1.0, 1.0, size=n_events)
        phi = np.random.uniform(0.0, 2.0 * np.pi, size=n_events)

        theta = np.arccos(u_rand)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        M = np.vstack([x, y, z]).T
        mean_vec = M.mean(axis=0)
        r_values[i] = np.linalg.norm(mean_vec)

    return r_values


def analyse_subset(label, ra, dec):
    """
    compute axis, r, and isotropic p_dipole for a subset.
    returns a dict with all metrics.
    """
    n = len(ra)
    l_best, b_best, r_real = dipole_axis_and_r(ra, dec)
    sep = angsep(l_best, b_best, UNIFIED_L, UNIFIED_B)

    r_null = mc_dipole_r(n, N_MC)
    mean_r = r_null.mean()
    std_r = r_null.std()
    p_dipole = np.mean(r_null >= r_real)

    return {
        "label": label,
        "n": n,
        "l_best": l_best,
        "b_best": b_best,
        "r_real": r_real,
        "sep_unified": sep,
        "mean_r_null": mean_r,
        "std_r_null": std_r,
        "p_dipole": p_dipole,
    }


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    print("===========================================================")
    print("FRB REDSHIFT-SPLIT AXIS AND ANISOTROPY TEST")
    print("===========================================================\n")

    df = load_frb_catalog(CATALOG_FILE)
    n_total = len(df)
    print(f"loaded frbs with z_est: {n_total}")

    if n_total < 20:
        print("not enough events with redshift estimates for this test.")
        return

    # sort by redshift and split into low/high halves
    df_sorted = df.sort_values("z_est").reset_index(drop=True)
    mid = n_total // 2

    low = df_sorted.iloc[:mid]
    high = df_sorted.iloc[mid:]

    # analyse full, low-z, high-z
    res_all = analyse_subset("all", df_sorted["ra"].values, df_sorted["dec"].values)
    res_low = analyse_subset("low-z half", low["ra"].values, low["dec"].values)
    res_high = analyse_subset("high-z half", high["ra"].values, high["dec"].values)

    # print results
    for res in [res_all, res_low, res_high]:
        print(f"\nsubset: {res['label']}")
        print(f"n = {res['n']}")
        print(f"l_best = {res['l_best']:.3f} deg")
        print(f"b_best = {res['b_best']:.3f} deg")
        print(f"dipole amplitude r = {res['r_real']:.4f}")
        print(f"separation from unified axis = {res['sep_unified']:.3f} deg")
        print(f"<r>_null ≈ {res['mean_r_null']:.4f}")
        print(f"std(r)_null ≈ {res['std_r_null']:.4f}")
        print(f"p_dipole ≈ {res['p_dipole']:.4f}")

    # --------------------------------------------------------
    # verdict (scientific tone)
    # --------------------------------------------------------
    print("\n-----------------------------------------------------------")
    print("verdict")
    print("-----------------------------------------------------------")

    p_low = res_low["p_dipole"]
    p_high = res_high["p_dipole"]
    sep_low = res_low["sep_unified"]
    sep_high = res_high["sep_unified"]

    if (p_low < 0.01) and (p_high < 0.01):
        print(
            "both the low-z and high-z halves show dipole amplitudes that are "
            "unlikely under an isotropic null, indicating that the large-scale "
            "anisotropy is not confined to a single redshift slice. this favours "
            "a genuinely extended anisotropy rather than a purely local structure."
        )
    elif (p_low < 0.01) and (p_high > 0.1):
        print(
            "the low-z half exhibits a significant dipole while the high-z half is "
            "consistent with isotropy. this pattern suggests that the anisotropy "
            "may be dominated by relatively nearby structure, with the signal "
            "washing out at larger distances."
        )
    elif (p_high < 0.01) and (p_low > 0.1):
        print(
            "the high-z half shows a significant dipole whereas the low-z half is "
            "closer to isotropic. this points toward an anisotropy that strengthens "
            "with distance, potentially linked to large-scale structure at higher "
            "redshift rather than a local foreground."
        )
    else:
        print(
            "neither redshift half shows a very strong preference over the isotropic "
            "null, or both only show marginal excesses. within the current sample, "
            "the evidence for redshift-dependent anisotropy remains limited, and "
            "a larger catalog with more precise distance estimates would be needed "
            "to sharpen this test."
        )

    # axis comparison comment
    print()
    print(
        f"in addition, the low-z axis is offset by ~{sep_low:.1f} degrees from the "
        f"unified direction, while the high-z axis is offset by ~{sep_high:.1f} "
        f"degrees. the relative alignment of these two axes provides a check on "
        f"whether the preferred direction is stable across distance slices."
    )

    print("-----------------------------------------------------------")
    print("analysis complete.")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
