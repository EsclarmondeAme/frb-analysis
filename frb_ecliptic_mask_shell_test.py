#!/usr/bin/env python3
# ============================================================
# frb ecliptic-plane mask shell test
# ============================================================
# this script mirrors frb_galactic_mask_shell_test.py, but
# instead of masking in galactic latitude |b|, it masks in
# ecliptic latitude |beta| to test whether the radial shell
# excess persists once the solar-system plane is removed.
#
# steps:
#   1. load frbs.csv (600 events)
#   2. convert (ra, dec) -> galactic (l, b) and ecliptic lat beta
#   3. compute angle theta between each frb and the unified axis
#   4. for each ecliptic mask:
#        |beta| >= 0°   (no mask)
#        |beta| >= 10°
#        |beta| >= 20°
#        |beta| >= 30°
#      compute radial band counts in theta:
#        0–10°, 10–25°, 25–40°, 40–90°
#      compare to isotropic expectation using
#        p_band = cos(theta1) - cos(theta2)
#      compute chi² and p-value vs isotropy
#   5. print a scientific-style verdict
# ============================================================

import numpy as np
import pandas as pd
import math

from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import chi2

# ------------------------------------------------------------
# config
# ------------------------------------------------------------
CATALOG_FILE = r"C:\Users\ratec\Documents\CrossLayerPhysics\frbs.csv"

# unified axis (galactic) from earlier analysis
UNIFIED_L = 159.85  # deg
UNIFIED_B = -0.51   # deg

# radial bands around the axis (same as previous shell tests)
BANDS_DEG = [(0.0, 10.0),
             (10.0, 25.0),
             (25.0, 40.0),
             (40.0, 90.0)]


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def angdist_gal(l1_deg, b1_deg, l2_deg, b2_deg):
    """
    great-circle separation between two galactic directions in degrees.
    """
    c1 = SkyCoord(l=l1_deg * u.deg, b=b1_deg * u.deg, frame="galactic")
    c2 = SkyCoord(l=l2_deg * u.deg, b=b2_deg * u.deg, frame="galactic")
    return c1.separation(c2).deg


def load_frbs(path):
    """
    load the frb catalog and keep rows with valid ra/dec.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["ra", "dec"])
    return df


def compute_coords(df):
    """
    from ra, dec (deg) compute:
      - galactic l, b in deg
      - ecliptic latitude beta in deg (geocentric true ecliptic)

    returns l, b, beta as numpy arrays.
    """
    c_icrs = SkyCoord(ra=df["ra"].values * u.deg,
                      dec=df["dec"].values * u.deg,
                      frame="icrs")

    gal = c_icrs.galactic
    l = gal.l.deg
    b = gal.b.deg

    # ecliptic latitude (geocentric true ecliptic frame)
    ecl = c_icrs.geocentrictrueecliptic
    beta = ecl.lat.deg

    return l, b, beta


def radial_shell_stats(theta_deg, n_total):
    """
    given theta (deg) from the unified axis and the total number of frbs
    in the subset (n_total), compute observed counts in each band using
    only theta <= 90°, and isotropic expectations using:

        p_band = cos(theta1) - cos(theta2)

    which sums to 1 over 0–90°.

    returns:
        bands_info: list of (t1, t2, n_obs, n_exp, ratio)
        chi2_val: chi-square statistic over the four bands
        p_val: chi2 tail probability with dof = 3
    """
    theta = np.asarray(theta_deg)

    # only count frbs within 0–90° for the bands
    mask_hem = (theta >= 0.0) & (theta <= 90.0)
    theta_use = theta[mask_hem]

    bands_info = []
    chi2_val = 0.0
    dof = 0

    for t1, t2 in BANDS_DEG:
        band_mask = (theta_use >= t1) & (theta_use < t2)
        n_obs = float(np.sum(band_mask))

        # isotropic probability in this band for a full sphere:
        # P = cos(t1) - cos(t2)
        p_band = math.cos(math.radians(t1)) - math.cos(math.radians(t2))
        n_exp = n_total * p_band

        ratio = n_obs / n_exp if n_exp > 0 else np.nan
        bands_info.append((t1, t2, n_obs, n_exp, ratio))

        if n_exp > 0:
            chi2_val += (n_obs - n_exp) ** 2 / n_exp
            dof += 1

    # nominal dof is number of bands minus 1
    dof_eff = max(dof - 1, 1)
    p_val = chi2.sf(chi2_val, df=dof_eff)

    return bands_info, chi2_val, p_val, dof_eff


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    print("===================================================")
    print("FRB ECLIPTIC-PLANE MASK SHELL TEST")
    print("===================================================\n")

    df = load_frbs(CATALOG_FILE)
    n_all = len(df)
    print(f"loaded FRBs with valid positions: {n_all}")

    if n_all < 50:
        print("not enough events for a meaningful shell test.")
        return

    l, b, beta = compute_coords(df)

    # angle to unified axis
    theta = angdist_gal(UNIFIED_L, UNIFIED_B, l, b)

    # ecliptic latitude masks (deg)
    beta_cuts = [0.0, 10.0, 20.0, 30.0]

    results = []

    for cut in beta_cuts:
        if cut == 0.0:
            label = "all (no |beta| cut)"
            mask = np.ones_like(beta, dtype=bool)
        else:
            label = f"|beta| >= {cut:.0f}°"
            mask = np.abs(beta) >= cut

        theta_sub = theta[mask]
        n_sub = len(theta_sub)

        print(f"\nsubset: {label}")
        print(f"  n_FRB = {n_sub}")

        if n_sub < 30:
            print("  too few events; skipping shell statistics.")
            continue

        bands_info, chi2_val, p_val, dof_eff = radial_shell_stats(theta_sub, n_sub)

        for t1, t2, n_obs, n_exp, ratio in bands_info:
            print(f"  band {t1:4.0f}°–{t2:4.0f}° : "
                  f"obs = {n_obs:4.0f}, exp_iso = {n_exp:6.2f}, ratio = {ratio:5.2f}")

        print(f"  total chi² (vs isotropy) = {chi2_val:.2f} (dof={dof_eff})")
        print(f"  p-value (isotropic null) = {p_val:.3e}")

        results.append((label, chi2_val, p_val, n_sub))

    # --------------------------------------------------------
    # scientific-style verdict
    # --------------------------------------------------------
    print("\n---------------- scientific verdict ----------------")
    if not results:
        print("no valid subsets with enough events for shell testing.")
        print("---------------------------------------------------")
        print("analysis complete.")
        print("===================================================\n")
        return

    # look at the no-mask and highest-cut cases if present
    p_all = None
    p_highcut = None
    for label, chi2_val, p_val, n_sub in results:
        if "no |beta| cut" in label:
            p_all = p_val
        if "|beta| >= 30°" in label:
            p_highcut = p_val

    if p_all is not None and p_highcut is not None:
        if p_all < 1e-6 and p_highcut < 1e-4:
            print("for the full sample (no ecliptic mask), the layered shell test")
            print(f"remains strongly inconsistent with an isotropic radial profile")
            print(f"(p_all ≈ {p_all:.1e}). after excluding the ecliptic plane")
            print(f"(|beta| >= 30°), the shell excess is still highly significant")
            print(f"(p_highcut ≈ {p_highcut:.1e}).")
            print("→ this favours a geometric anisotropy that is not driven by")
            print("   solar-system / ecliptic-plane systematics.")
        elif p_all < 1e-4 and p_highcut > 1e-2:
            print("the shell signal is strong in the full sample but weakens once")
            print("a strict ecliptic mask is applied. this would suggest that a")
            print("non-negligible fraction of the radial anisotropy is coupled to")
            print("ecliptic-aligned survey strategy or solar-system foregrounds.")
        else:
            print("the ecliptic masks produce mixed p-values: some cuts retain a")
            print("strong shell signal, others reduce it. this points to a blend")
            print("of intrinsic anisotropy and ecliptic-linked selection effects,")
            print("and motivates more detailed per-survey footprint modelling.")
    else:
        print("the ecliptic-mask subsets could not all be evaluated; nonetheless,")
        print("the available shells provide a direct check of whether the radial")
        print("excess is tied to the solar-system plane or persists off-ecliptic.")

    print("---------------------------------------------------")
    print("analysis complete.")
    print("===================================================\n")


if __name__ == "__main__":
    main()
