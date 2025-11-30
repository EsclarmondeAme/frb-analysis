import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.special import sph_harm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# -----------------------------------------
# load unified catalog (CHIME + FRBCAT only)
# -----------------------------------------
def load_catalog():
    df1 = pd.read_csv("chime_frb_catalog1.csv")
    df2 = pd.read_csv("frbs.csv")

    # standardize names
    df1 = df1.rename(columns={"ra_deg": "ra", "dec_deg": "dec"})
    df2 = df2.rename(columns={"ra_deg": "ra", "dec_deg": "dec"})

    # use correct CHIME DM column
    if "dm_exc_ymw16" in df1.columns:
        df1 = df1.rename(columns={"dm_exc_ymw16": "dm"})
    elif "dm_exc_ne2001" in df1.columns:
        df1 = df1.rename(columns={"dm_exc_ne2001": "dm"})
    elif "bonsai_dm" in df1.columns:
        df1 = df1.rename(columns={"bonsai_dm": "dm"})
    else:
        raise ValueError("no usable DM column found in chime_frb_catalog1.csv")


    # dm already renamed above — no extra rename needed


    # standardize FRBCAT DM
    if "dm_excess" in df2.columns:
        df2 = df2.rename(columns={"dm_excess": "dm"})
    elif "dm" not in df2.columns:
        print("warning: frbs.csv has no DM column; filling with NaN")
        df2["dm"] = np.nan


    # keep only the core columns
    keep = ["ra", "dec", "dm"]

    df1 = df1[keep].dropna()
    df2 = df2[keep].dropna()

    # merge
    df = pd.concat([df1, df2], ignore_index=True).dropna()

    return df

# -------------------------------------------------
# convert RA/DEC/DM → spherical coordinates (θ, φ, r)
# -------------------------------------------------
def convert_to_3d(df):
    c = SkyCoord(ra=df["ra"].values*u.deg,
                 dec=df["dec"].values*u.deg,
                 frame="icrs")

    theta = np.pi/2 - c.dec.radian    # polar angle
    phi   = c.ra.radian

    # crude DM→distance mapping (cosmology-independent shell test)
    # r = k * DM, where k rescales to Mpc or arbitrary units
    k = 1.0
    r = k * df["dm"].values

    return r, theta, phi


# -------------------------------------------------
# build a 3d density grid
# -------------------------------------------------
def build_density_cube(r, theta, phi,
                       Nr=40, Ntheta=64, Nphi=128):

    r_edges     = np.linspace(r.min(), r.max(), Nr+1)
    theta_edges = np.linspace(0, np.pi, Ntheta+1)
    phi_edges   = np.linspace(0, 2*np.pi, Nphi+1)

    hist, _ = np.histogramdd(
        sample = np.vstack([r, theta, phi]).T,
        bins   = [r_edges, theta_edges, phi_edges]
    )

    return hist, r_edges, theta_edges, phi_edges


# -------------------------------------------------
# optional smoothing of 3d density cube
# -------------------------------------------------
def smooth_cube(cube, sigma=1.0):
    return gaussian_filter(cube, sigma=sigma)


# -------------------------------------------------
# reconstruct angular dependence via spherical harmonics
# -------------------------------------------------
def angular_harmonic_reconstruction(cube, theta_edges, phi_edges, Lmax=4):
    # collapse radially
    ang_map = cube.sum(axis=0)

    # centers
    theta_grid = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_grid   = 0.5 * (phi_edges[:-1]   + phi_edges[1:])

    TH, PH = np.meshgrid(theta_grid, phi_grid, indexing="ij")

    coeffs = {}

    for l in range(Lmax+1):
        for m in range(-l, l+1):
            # spherical harmonic evaluated on grid
            Ylm = sph_harm(m, l, PH, TH)

            # inner product with the spherical map
            a_lm = np.sum(ang_map * np.conj(Ylm)) * (4*np.pi / ang_map.size)
            coeffs[(l,m)] = a_lm

    return coeffs, ang_map, TH, PH


# -------------------------------------------------
# main
# -------------------------------------------------
def main():
    print("loading unified catalog…")
    df = load_catalog()
    print("N =", len(df))

    print("converting to 3d…")
    r, theta, phi = convert_to_3d(df)

    print("building 3d density cube…")
    cube, r_edges, theta_edges, phi_edges = build_density_cube(r, theta, phi)

    print("smoothing…")
    cube_s = smooth_cube(cube, sigma=1.5)

    print("computing spherical harmonics up to L=4…")
    coeffs, ang_map, TH, PH = angular_harmonic_reconstruction(
        cube_s, theta_edges, phi_edges, Lmax=4
    )

    # store coefficients
    with open("frb_3d_harmonic_coeffs.txt", "w") as f:
        for (l,m), val in coeffs.items():
            f.write(f"l={l}, m={m}, Re={val.real:.6e}, Im={val.imag:.6e}\n")

    # quick visualization of angular map
    plt.figure(figsize=(10,5))
    plt.imshow(ang_map, origin="lower",
               extent=[0,360,0,180],
               aspect="auto", cmap="inferno")
    plt.colorbar(label="angular density")
    plt.xlabel("phi (deg)")
    plt.ylabel("theta (deg)")
    plt.title("FRB 3d density: angular projection")
    plt.savefig("frb_3d_density_ang.png", dpi=200)

    print("analysis complete.")
    print("saved: frb_3d_harmonic_coeffs.txt, frb_3d_density_ang.png")


if __name__ == "__main__":
    main()
