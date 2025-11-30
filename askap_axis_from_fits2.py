import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

# user path
FITS_DIR = r"C:\Users\ratec\Downloads\data\positions"


# unified axis from previous combined analysis
UNIFIED_L = 159.85
UNIFIED_B = -0.51

# number of monte carlo realisations for the dipole test
N_MC = 20000


# ----------------------------------------
# load RA/Dec from FITS headers
# ----------------------------------------
def load_fits_positions(folder):
    """
    scan a folder for fits files and extract ra/dec from header keywords
    crval1 and crval2.
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

                records.append([fname.replace("_post.fits", ""), float(ra), float(dec)])

        except Exception as e:
            print(f"[error] reading {fname}: {e}")

    return pd.DataFrame(records, columns=["frb", "ra", "dec"])


# ----------------------------------------
# dipole best-fit axis and amplitude
# ----------------------------------------
def dipole_axis_and_r(ra_deg, dec_deg):
    """
    compute the best-fit dipole axis and the dipole amplitude r.

    r is the length of the mean unit vector over all positions:
        r = |<v>|, where v are unit vectors on the sphere.

    for an isotropic sky, r is expected to be small and its distribution
    can be calibrated with monte carlo.
    """
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)

    # matrix of unit vectors
    M = np.vstack([x, y, z]).T

    # dipole amplitude: length of the mean vector
    mean_vec = M.mean(axis=0)
    r = np.linalg.norm(mean_vec)

    if r > 0:
        axis_vec = mean_vec / r
    else:
        # fallback if everything cancels exactly
        axis_vec = np.array([1.0, 0.0, 0.0])

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
# angular separation
# ----------------------------------------
def angsep(l1, b1, l2, b2):
    """
    great-circle separation in degrees between two galactic directions.
    """
    c1 = SkyCoord(l=l1 * u.deg, b=b1 * u.deg, frame="galactic")
    c2 = SkyCoord(l=l2 * u.deg, b=b2 * u.deg, frame="galactic")
    return c1.separation(c2).deg


# ----------------------------------------
# isotropic monte carlo for the dipole amplitude
# ----------------------------------------
def mc_dipole_r(n_events, n_mc):
    """
    generate dipole amplitudes r under an isotropic null for n_events
    positions per realisation.
    """
    r_values = np.empty(n_mc, dtype=float)

    for i in range(n_mc):
        # isotropic directions: cos(theta) uniform in [-1, 1], phi uniform in [0, 2pi)
        u = np.random.uniform(-1.0, 1.0, size=n_events)
        phi = np.random.uniform(0.0, 2.0 * np.pi, size=n_events)

        theta = np.arccos(u)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        M = np.vstack([x, y, z]).T
        mean_vec = M.mean(axis=0)
        r_values[i] = np.linalg.norm(mean_vec)

    return r_values


# ----------------------------------------
# main
# ----------------------------------------
def main():
    print("===================================================")
    print("askap axis reconstruction with dipole significance")
    print("===================================================\n")

    df = load_fits_positions(FITS_DIR)
    print(f"loaded askap frbs: {len(df)}")

    if len(df) < 5:
        print("not enough events for a meaningful dipole test.")
        return

    # compute dipole axis and amplitude
    l_best, b_best, r_real = dipole_axis_and_r(df["ra"].values, df["dec"].values)

    print("\naskap best-fit galactic axis (dipole direction):")
    print(f"l = {l_best:.3f} deg")
    print(f"b = {b_best:.3f} deg")
    print(f"dipole amplitude r = {r_real:.4f}")

    # separation from unified axis
    sep = angsep(l_best, b_best, UNIFIED_L, UNIFIED_B)
    print(f"\nangular separation from unified axis = {sep:.3f} deg")

    # analytic expectation for random axis misalignment
    # for a random direction on the sphere, cos(theta) is uniform in [-1, 1]
    # P(theta >= sep) = (1 + cos(sep)) / 2
    sep_rad = np.deg2rad(sep)
    p_angle_tail = 0.5 * (1.0 + np.cos(sep_rad))
    print(f"for a random axis, probability to be this far or farther from the "
          f"unified axis is p_angle ≈ {p_angle_tail:.3f}")

    # monte carlo dipole significance
    print("\nrunning isotropic monte carlo for the dipole amplitude...")
    r_null = mc_dipole_r(len(df), N_MC)
    mean_r = r_null.mean()
    std_r = r_null.std()
    p_dipole = np.mean(r_null >= r_real)

    print("\nmonte carlo dipole statistics (isotropic null):")
    print(f"<r>_null ≈ {mean_r:.4f}")
    print(f"std(r)_null ≈ {std_r:.4f}")
    print(f"p_dipole = P(r_null >= r_real) ≈ {p_dipole:.4f}")

    # ----------------------------------------
    # verdict block (scientific tone)
    # ----------------------------------------
    print("\n---------------------------------------------------")
    print("verdict")
    print("---------------------------------------------------")

    if p_dipole > 0.1:
        print(
            "the askap sample does not show a statistically significant dipole "
            "relative to an isotropic null. with only a small number of events, "
            "the best-fit dipole direction is expected to wander widely on the "
            "sky. the ~{:.1f} degree misalignment from the unified axis is "
            "therefore fully compatible with a sample that is effectively "
            "agnostic about the true cosmic direction.".format(sep)
        )
    elif p_dipole > 0.01:
        print(
            "the askap dipole amplitude is somewhat larger than typical isotropic "
            "realisations, but the significance is modest. the estimated axis is "
            "offset by ~{:.1f} degrees from the unified direction, which may "
            "reflect a combination of sample variance, footprint effects, and "
            "genuine sky structure. a larger askap sample or detailed exposure "
            "modelling would be required to turn this into a sharp constraint."
            .format(sep)
        )
    else:
        print(
            "the askap sample exhibits a dipole amplitude that is unlikely under "
            "an isotropic null and the best-fit axis is offset by ~{:.1f} degrees "
            "from the unified direction. if the current selection function is "
            "reliable, this points to genuine tension between the askap subset "
            "and the unified-axis model, motivating more detailed tests for "
            "population or frequency dependence.".format(sep)
        )

    print("---------------------------------------------------")
    print("analysis complete.")
    print("===================================================\n")


if __name__ == "__main__":
    main()
