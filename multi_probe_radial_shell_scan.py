import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

# ============================================================
# unified axis (galactic)
# ============================================================
UNIFIED_L = 159.85
UNIFIED_B = -0.51

# ------------------------------------------------------------
# convert unified axis to cartesian unit vector
# ------------------------------------------------------------
def axis_unit_vector(l_deg, b_deg):
    l = np.deg2rad(l_deg)
    b = np.deg2rad(b_deg)
    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)
    return np.array([x, y, z])


AXIS_VEC = axis_unit_vector(UNIFIED_L, UNIFIED_B)


# ============================================================
# STABLE ROTATION: rotate all RA/DEC so that unified axis → north pole
# ============================================================

def rotate_to_axis_from_radec(ra_deg, dec_deg):
    """
    input ra/dec arrays (floats)
    output theta, phi in degrees
    rotation method: cartesian → rotate so unified axis becomes +z
    """

    # convert data to unit vectors
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    xs = np.cos(dec)*np.cos(ra)
    ys = np.cos(dec)*np.sin(ra)
    zs = np.sin(dec)

    V = np.vstack([xs, ys, zs])   # shape: (3, N)

    # normalized axis vector
    k = AXIS_VEC / np.linalg.norm(AXIS_VEC)

    # if k already equals north pole, skip rotation
    north = np.array([0, 0, 1.0])

    if np.allclose(k, north):
        VX = V
    else:
        # rotation axis = cross(k, north)
        v = np.cross(k, north)
        s = np.linalg.norm(v)
        c = np.dot(k, north)

        # rotation matrix via Rodrigues formula
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2 + 1e-15))

        VX = R @ V  # shape (3,N)

    # convert rotated vectors back to theta, phi
    x2, y2, z2 = VX[0], VX[1], VX[2]

    theta = np.rad2deg(np.arccos(z2))     # 0 at pole
    phi = (np.rad2deg(np.arctan2(y2, x2)) + 360) % 360

    return theta, phi


# ============================================================
# load probes
# ============================================================

def load_frbs():
    df = pd.read_csv("frbs.csv")
    return df["ra"].values, df["dec"].values


def load_neutrinos():
    df = pd.read_csv("neutrinos_clean.csv")
    return df["ra"].values, df["dec"].values


def load_quasars():
    df = pd.read_csv("quasars.csv")   # ensure this exists
    return df["ra"].values, df["dec"].values


# ============================================================
# radial shell scan
# ============================================================

def analyze_probe(name, ra, dec, nbins=40):
    print(f"\n=== analyzing {name} ===")

    theta, phi = rotate_to_axis_from_radec(ra, dec)

    # histogram radial profile
    bins = np.linspace(0, 140, nbins+1)
    H, _ = np.histogram(theta, bins=bins)

    centers = 0.5*(bins[:-1] + bins[1:])

    # fit: minimal AIC-like proxy
    # model 1: flat
    c0 = np.mean(H)
    rss_flat = np.sum((H - c0)**2)

    # model 2: broken shell at 25 deg
    brk = 25
    idx_break = np.where(centers < brk)
    idx_after = np.where(centers >= brk)

    A = np.mean(H[idx_break])
    B = np.mean(H[idx_after])

    H_model = np.zeros_like(H)
    H_model[idx_break] = A
    H_model[idx_after] = B

    rss_shell = np.sum((H - H_model)**2)

    print(f"{name}: RSS flat={rss_flat:.2f}  RSS shell={rss_shell:.2f}")

    return {
        "name": name,
        "rss_flat": rss_flat,
        "rss_shell": rss_shell,
        "theta": theta,
        "phi": phi
    }


# ============================================================
# main driver
# ============================================================

if __name__ == "__main__":

    probes = []

    # only FRB + neutrino by default
    probes.append(analyze_probe("FRBs", *load_frbs()))
    probes.append(analyze_probe("Neutrinos", *load_neutrinos()))

    # add this when quasars.csv exists
    try:
        probes.append(analyze_probe("Quasars", *load_quasars()))
    except:
        print("skipping quasars (file missing)")

    print("\n============================")
    print("SUMMARY")
    print("============================")

    for p in probes:
        name = p["name"]
        f = p["rss_flat"]
        s = p["rss_shell"]
        print(f"{name:12s} : ΔRSS = {f - s:.2f}")

    print("============================")
