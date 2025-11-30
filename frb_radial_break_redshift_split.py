import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# config
CATALOG_FILE = r"frbs.csv"

# unified axis (galactic)
UNIFIED_L = 159.85
UNIFIED_B = -0.51

# geometry for the test
THETA_BREAK_DEG = 25.0   # fixed break radius
THETA_MAX_DEG   = 60.0   # only use events within 60 deg of the axis
N_MC = 20000             # monte carlo realisations


def load_frb_catalog(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["ra", "dec", "z_est"])
    return df


def compute_theta_to_axis(ra_deg, dec_deg, l_axis, b_axis):
    """
    compute angular distance (theta) from each (ra,dec) to the unified axis
    given in galactic coordinates.
    returns theta in degrees.
    """
    coords_icrs = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    axis_gal = SkyCoord(l=l_axis * u.deg, b=b_axis * u.deg, frame="galactic")
    # convert axis to icrs and compute separation
    axis_icrs = axis_gal.icrs
    theta = coords_icrs.separation(axis_icrs).deg
    return theta


def inner_outer_counts(theta_deg, theta_break_deg, theta_max_deg):
    """
    count events in the inner and outer regions:
      inner: theta < theta_break
      outer: theta_break <= theta <= theta_max
    """
    mask_max = theta_deg <= theta_max_deg
    theta_sel = theta_deg[mask_max]

    inner = theta_sel < theta_break_deg
    outer = (theta_sel >= theta_break_deg) & (theta_sel <= theta_max_deg)

    n_in = inner.sum()
    n_out = outer.sum()
    n_tot = n_in + n_out
    return n_in, n_out, n_tot


def isotropic_expected_counts(n_tot, theta_break_deg, theta_max_deg):
    """
    under isotropy on a cone 0 <= theta <= theta_max,
    the probability to fall in [0, theta_break) is:
      p_in = (1 - cos(theta_break)) / (1 - cos(theta_max))
    """
    tb = np.deg2rad(theta_break_deg)
    tm = np.deg2rad(theta_max_deg)

    p_in = (1.0 - np.cos(tb)) / (1.0 - np.cos(tm))
    p_out = 1.0 - p_in

    mu_in = n_tot * p_in
    mu_out = n_tot * p_out
    return mu_in, mu_out, p_in, p_out


def chi2_inner_outer(n_in, n_out, mu_in, mu_out):
    """
    simple 2-bin chi^2.
    """
    if mu_in <= 0 or mu_out <= 0:
        return 0.0
    chi2 = (n_in - mu_in) ** 2 / mu_in + (n_out - mu_out) ** 2 / mu_out
    return chi2


def mc_chi2_null(n_tot, theta_break_deg, theta_max_deg, n_mc):
    """
    monte carlo null for the 2-bin chi^2 statistic under isotropy
    within the cone theta <= theta_max.
    """
    tb = np.deg2rad(theta_break_deg)
    tm = np.deg2rad(theta_max_deg)

    # inverse-cdf sampling for theta in [0, theta_max]
    # cos(theta) uniform from cos(theta_max) to 1
    cos_tm = np.cos(tm)

    chi2_vals = np.empty(n_mc, dtype=float)

    for i in range(n_mc):
        u = np.random.uniform(0.0, 1.0, size=n_tot)
        cos_theta = 1.0 - u * (1.0 - cos_tm)
        theta = np.arccos(cos_theta)  # radians

        # counts
        mask_in = theta < tb
        mask_out = (theta >= tb) & (theta <= tm)

        n_in = mask_in.sum()
        n_out = mask_out.sum()

        mu_in, mu_out, _, _ = isotropic_expected_counts(n_tot, theta_break_deg, theta_max_deg)
        chi2_vals[i] = chi2_inner_outer(n_in, n_out, mu_in, mu_out)

    return chi2_vals


def analyse_subset(label, theta_deg):
    """
    compute inner/outer counts, chi^2, and monte carlo p-value
    for a given subset (all, low-z, high-z).
    """
    n_in, n_out, n_tot = inner_outer_counts(theta_deg, THETA_BREAK_DEG, THETA_MAX_DEG)

    if n_tot < 20:
        print(f"\nsubset: {label}")
        print("not enough events within theta <= theta_max for a meaningful test.")
        return

    mu_in, mu_out, p_in, p_out = isotropic_expected_counts(
        n_tot, THETA_BREAK_DEG, THETA_MAX_DEG
    )
    chi2_real = chi2_inner_outer(n_in, n_out, mu_in, mu_out)

    # monte carlo
    chi2_null = mc_chi2_null(n_tot, THETA_BREAK_DEG, THETA_MAX_DEG, N_MC)
    p_mc = np.mean(chi2_null >= chi2_real)

    print("\n===================================================")
    print(f" FRB RADIAL BREAK TEST — {label}")
    print("===================================================\n")
    print(f"theta_break = {THETA_BREAK_DEG:.1f} deg, theta_max = {THETA_MAX_DEG:.1f} deg")
    print(f"total events in cone: n_tot   = {n_tot}")
    print(f"observed: n_in  (theta < {THETA_BREAK_DEG:.1f})     = {n_in}")
    print(f"          n_out ({THETA_BREAK_DEG:.1f} <= theta <= {THETA_MAX_DEG:.1f}) = {n_out}")
    print(f"expected under isotropy: mu_in  ≈ {mu_in:.2f}")
    print(f"                           mu_out ≈ {mu_out:.2f}")
    print(f"probabilities: p_in ≈ {p_in:.3f}, p_out ≈ {p_out:.3f}")
    print(f"\nchi^2_real = {chi2_real:.3f}")
    print(f"MC p_value = P(chi^2_null >= chi^2_real) ≈ {p_mc:.4f}")

    print("\n--------------- scientific verdict ---------------")
    if p_mc < 0.01:
        print(
            "a significant excess of events is found inside the inner cone relative "
            "to isotropic expectations, even when restricting to theta <= theta_max. "
            "this supports the presence of a radial break / inner enhancement in "
            "this subset."
        )
    elif p_mc < 0.05:
        print(
            "there is moderate evidence for an inner excess relative to isotropy. "
            "the radial break signal is suggestive but not overwhelmingly strong "
            "for this subset."
        )
    else:
        print(
            "the inner vs outer counts are consistent with isotropy within the "
            "cone for this subset; a radial break at the chosen theta_break is "
            "not significantly detected here."
        )
    print("---------------------------------------------------")


def main():
    print("===================================================")
    print(" FRB RADIAL BREAK REDSHIFT-SPLIT TEST")
    print("===================================================\n")

    df = load_frb_catalog(CATALOG_FILE)
    print(f"loaded frbs with z_est: {len(df)}")

    # compute theta for all events
    theta_all = compute_theta_to_axis(df["ra"].values, df["dec"].values, UNIFIED_L, UNIFIED_B)

    # split by redshift median
    z_med = np.median(df["z_est"].values)
    df_low  = df[df["z_est"] <= z_med].copy()
    df_high = df[df["z_est"] >  z_med].copy()

    theta_low  = compute_theta_to_axis(df_low["ra"].values, df_low["dec"].values, UNIFIED_L, UNIFIED_B)
    theta_high = compute_theta_to_axis(df_high["ra"].values, df_high["dec"].values, UNIFIED_L, UNIFIED_B)

    print(f"low-z sample:  {len(df_low)} events")
    print(f"high-z sample: {len(df_high)} events")

    analyse_subset("ALL", theta_all)
    analyse_subset("LOW-Z HALF", theta_low)
    analyse_subset("HIGH-Z HALF", theta_high)

    print("\nanalysis complete.")
    print("===================================================\n")


if __name__ == "__main__":
    main()
