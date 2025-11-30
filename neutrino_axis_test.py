import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

warnings.filterwarnings("ignore")

# unified FRB axis (galactic)
UNIFIED_L = 159.85
UNIFIED_B = -0.51

# monte carlo realizations
N_MC = 20000

CAT_FILE = "neutrinos.csv"


# ============================================================
# utilities
# ============================================================

def radec_to_unitvec(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T


def unitvec_to_galactic_l_b(xyz):
    """
    convert a direction vector to galactic coordinates
    """
    x, y, z = xyz
    vec = SkyCoord(x=x, y=y, z=z, representation_type="cartesian", frame="icrs")
    gal = vec.galactic
    return gal.l.deg, gal.b.deg


def angular_separation(l1, b1, l2, b2):
    c1 = SkyCoord(l=l1*u.deg, b=b1*u.deg, frame="galactic")
    c2 = SkyCoord(l=l2*u.deg, b=b2*u.deg, frame="galactic")
    return c1.separation(c2).deg


# ============================================================
# dipole estimator
# ============================================================

def estimate_dipole_axis(xyz):
    """
    dipole direction = normalized vector sum of all unit vectors  
    """
    v = np.sum(xyz, axis=0)
    if np.linalg.norm(v) == 0:
        return np.array([0,0,1.0]), 0.0
    dipdir = v / np.linalg.norm(v)
    r = np.linalg.norm(v) / len(xyz)
    return dipdir, r


def random_isotropic_unitvec(n):
    """
    sample N isotropic directions on the sphere
    """
    u = np.random.uniform(-1, 1, size=n)
    phi = np.random.uniform(0, 2*np.pi, size=n)
    theta = np.arccos(u)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.vstack([x, y, z]).T


# ============================================================
# main
# ============================================================

def main():

    print("===================================================")
    print(" NEUTRINO AXIS RECONSTRUCTION AND SIGNIFICANCE TEST")
    print("===================================================\n")

    try:
        df = pd.read_csv(CAT_FILE)
    except:
        print(f"could not load {CAT_FILE}")
        return

    # require RA/Dec columns
    if not {"ra","dec"}.issubset(df.columns):
        print("neutrinos.csv must contain 'ra' and 'dec' columns.")
        return

    df = df.dropna(subset=["ra","dec"])
    N = len(df)

    print(f"loaded neutrinos: {N}")

    if N < 5:
        print("not enough events for dipole analysis.")
        return

    # convert to unit vectors
    xyz = radec_to_unitvec(df["ra"].astype(float).values,
                           df["dec"].astype(float).values)

    # actual dipole
    dip_vec, r_real = estimate_dipole_axis(xyz)
    l_dip, b_dip = unitvec_to_galactic_l_b(dip_vec)

    sep = angular_separation(l_dip, b_dip, UNIFIED_L, UNIFIED_B)

    print("\nneutrino best-fit axis (galactic):")
    print(f"   l = {l_dip:.3f} deg")
    print(f"   b = {b_dip:.3f} deg")
    print(f"dipole amplitude r = {r_real:.4f}")
    print(f"angular separation from unified axis = {sep:.3f} deg")

    # monte carlo
    print("\nrunning isotropic monte carlo...")

    r_null = np.zeros(N_MC)
    for i in range(N_MC):
        xyz_rand = random_isotropic_unitvec(N)
        _, r_null[i] = estimate_dipole_axis(xyz_rand)

    p_dip = np.mean(r_null >= r_real)

    print("\nmonte carlo dipole statistics:")
    print(f"<r>_null ≈ {np.mean(r_null):.4f}")
    print(f"std(r)_null ≈ {np.std(r_null):.4f}")
    print(f"p_dipole = P(r_null >= r_real) ≈ {p_dip:.4f}")

    print("\n--------------- scientific verdict ---------------")

    if p_dip < 0.01:
        print(
            "neutrinos exhibit a dipole amplitude that is unlikely under isotropy. "
            "this indicates genuine large-scale anisotropy in the neutrino sample. "
            "the axis alignment with the FRB unified axis may have physical meaning."
        )
    elif p_dip < 0.05:
        print(
            "moderate evidence for neutrino anisotropy. alignment with the unified "
            "axis is suggestive but not definitive."
        )
    else:
        print(
            "the neutrino dipole amplitude is consistent with isotropy. no significant "
            "alignment with the unified axis is detected in this sample."
        )

    print("---------------------------------------------------")
    print("analysis complete.")
    print("===================================================\n")


if __name__ == "__main__":
    main()
