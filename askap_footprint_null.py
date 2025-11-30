import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

# ----------------------------------------
# configuration
# ----------------------------------------
FITS_DIR = r"C:\Users\ratec\Downloads\data\positions"   # your confirmed folder
N_MC = 20000                                            # monte carlo realisations


# ----------------------------------------
# load RA/Dec from fits headers
# ----------------------------------------
def load_fits_positions(folder):
    """
    load all fits files in the folder and extract ra, dec from header
    using CRVAL1 and CRVAL2. we assume askap posterior cutouts or
    localization products.
    """
    records = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".fits"):
            continue

        path = os.path.join(folder, fname)
        try:
            with fits.open(path) as hdul:
                hdr = hdul[0].header
                ra = hdr.get("CRVAL1", None)
                dec = hdr.get("CRVAL2", None)

                if ra is None or dec is None:
                    print(f"[warning] missing crval1/crval2 in {fname}")
                    continue

                records.append([fname, float(ra), float(dec)])

        except Exception as e:
            print(f"[error] reading {fname}: {e}")

    return pd.DataFrame(records, columns=["fname", "ra", "dec"])


# ----------------------------------------
# dipole amplitude and axis
# ----------------------------------------
def dipole_axis_and_r(ra_deg, dec_deg):
    """
    compute dipole amplitude r = |mean unit vector| and its direction.
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
        axis_vec = np.array([1, 0, 0])
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


# ----------------------------------------
# footprint sampling
# ----------------------------------------
def sample_within_footprint(n, ra_data, dec_data):
    """
    generate n synthetic frbs drawn from the askap footprint:

    - dec distribution: sampled from the empirical dec of the real data
    - ra distribution: sampled from the empirical ra of the real data
      (captures askap's ra windows or clusters)
    - random small jitter added to avoid degenerate duplicates

    this preserves the real selection windows in both ra and dec.

    returns (ra_mock, dec_mock)
    """
    ra_mock = np.random.choice(ra_data, size=n, replace=True)
    dec_mock = np.random.choice(dec_data, size=n, replace=True)

    # tiny jitter to avoid exact duplicates
    ra_mock = ra_mock + np.random.normal(0, 0.05, size=n)
    dec_mock = dec_mock + np.random.normal(0, 0.05, size=n)

    return ra_mock, dec_mock


# ----------------------------------------
# compute footprint-corrected null r distribution
# ----------------------------------------
def mc_r_footprint(df, n_mc):
    """
    monte carlo dipole amplitudes under the askap empirical footprint.
    """
    n = len(df)
    ra_data = df["ra"].values
    dec_data = df["dec"].values

    r_vals = np.empty(n_mc, dtype=float)
    for i in range(n_mc):
        ra_mock, dec_mock = sample_within_footprint(n, ra_data, dec_data)
        _, _, r = dipole_axis_and_r(ra_mock, dec_mock)
        r_vals[i] = r

    return r_vals


# ----------------------------------------
# main
# ----------------------------------------
def main():
    print("===================================================")
    print("askap footprint-corrected dipole significance test")
    print("===================================================\n")

    df = load_fits_positions(FITS_DIR)
    n = len(df)
    print(f"loaded askap frbs: {n}")

    if n < 5:
        print("not enough events for meaningful analysis.")
        return

    # compute real dipole
    l_best, b_best, r_real = dipole_axis_and_r(df["ra"].values, df["dec"].values)
    print("\nreal askap dipole:")
    print(f"l = {l_best:.3f} deg")
    print(f"b = {b_best:.3f} deg")
    print(f"dipole amplitude r_real = {r_real:.4f}")

    # run footprint null
    print("\nrunning footprint-constrained monte carlo...")
    r_null = mc_r_footprint(df, N_MC)
    mean_r = r_null.mean()
    std_r = r_null.std()
    p_footprint = np.mean(r_null >= r_real)

    print("\nfootprint null dipole statistics:")
    print(f"<r>_null_footprint ≈ {mean_r:.4f}")
    print(f"std(r)_null_footprint ≈ {std_r:.4f}")
    print(f"p_footprint = P(r_null >= r_real) ≈ {p_footprint:.4f}")

    print("\n---------------------------------------------------")
    print("verdict (scientific interpretation)")
    print("---------------------------------------------------")

    if p_footprint > 0.1:
        print(
            "once the askap sky footprint is accounted for, the observed dipole "
            "amplitude is entirely compatible with typical anisotropy values "
            "induced by the survey geometry itself. in this case, askap does not "
            "provide meaningful tension with the unified axis; it behaves as an "
            "instrument-dominated sample with limited directional information."
        )
    elif p_footprint > 0.01:
        print(
            "after correcting for the askap footprint, the dipole amplitude remains "
            "somewhat larger than most mock realisations, though the significance "
            "is still modest. this suggests that askap may contain a mixture of "
            "geometric selection effects and mild intrinsic anisotropy, but the "
            "evidence for a distinct cosmic axis remains weak."
        )
    else:
        print(
            "even after constraining the null to the actual askap sky footprint, "
            "the observed dipole amplitude remains unusually large. this indicates "
            "that the askap subset carries genuine anisotropic structure not fully "
            "explained by survey geometry. combined with its large angular offset "
            "from the unified axis, this represents real tension between askap and "
            "the unified-axis model, motivating tests for population splitting or "
            "frequency dependence."
        )

    print("---------------------------------------------------")
    print("analysis complete.")
    print("===================================================\n")


if __name__ == "__main__":
    main()
