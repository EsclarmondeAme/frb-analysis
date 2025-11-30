#!/usr/bin/env python3
"""
FRB sky dipole + quadrupole spherical harmonic analysis

This script:
1. Loads FRB sky positions (RA, Dec) from frbs.csv
2. Converts to unit vectors in 3D
3. Fits:
   - Sky dipole vector (Y₁)
   - Sky quadrupole tensor (Y₂)
4. Computes amplitudes and principal axes
5. Monte Carlo significance tests
6. Produces:
   - Summary printout
   - Mollweide sky plot with dipole + quadrupole overlay
   - Saved figure: frb_sky_dipole_quadrupole.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from numpy.linalg import eig

# ----------------------------------------------
# utilities
# ----------------------------------------------

def load_frbs(fname="frbs.csv"):
    """Load FRBs with valid RA/Dec."""
    df = pd.read_csv(fname)
    df = df.dropna(subset=["ra", "dec"])
    return df


def sph_to_unit_vectors(ra_deg, dec_deg):
    """Convert spherical RA/Dec to 3D unit vectors."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T


def fit_dipole(vecs):
    """Compute dipole vector D = <n>."""
    return np.mean(vecs, axis=0)


def fit_quadrupole(vecs):
    """
    Compute quadrupole tensor:
        Q_ij = < n_i n_j > - delta_ij/3
    """
    Q = np.einsum('ni,nj->ij', vecs, vecs) / vecs.shape[0]
    Q -= np.eye(3) / 3.0  # remove trace
    return Q


def random_unit_vectors(n):
    """Monte Carlo random isotropic directions."""
    u = np.random.uniform(0, 1, n)
    v = np.random.uniform(0, 1, n)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.vstack([x, y, z]).T


# ----------------------------------------------
# main analysis
# ----------------------------------------------

def main():

    print("=" * 60)
    print("FRB SKY DIPOLE + QUADRUPOLE ANALYSIS")
    print("=" * 60)

    # load FRBs
    frbs = load_frbs()
    print(f"loaded FRBs with RA/Dec: {len(frbs)}")

    # convert to unit vectors
    vecs = sph_to_unit_vectors(frbs["ra"].values, frbs["dec"].values)

    # ------------------------------------------
    # dipole
    # ------------------------------------------
    dipole = fit_dipole(vecs)
    dipole_amp = np.linalg.norm(dipole)
    dipole_dir = dipole / dipole_amp

    # convert dipole direction to RA/Dec
    # dipole axis in RA/Dec
    dip_coord = SkyCoord(
        x=dipole_dir[0], 
        y=dipole_dir[1], 
        z=dipole_dir[2],
        unit='',
        representation_type='cartesian'
    ).spherical

    dip_ra = dip_coord.lon.deg
    dip_dec = dip_coord.lat.deg



    # ------------------------------------------
    # quadrupole
    # ------------------------------------------
    Q = fit_quadrupole(vecs)

    # eigen decomposition
    evals, evecs = eig(Q)

    # sort by absolute magnitude
    idx = np.argsort(-np.abs(evals))
    evals = evals[idx]
    evecs = evecs[:, idx]

    quad_amp = np.abs(evals[0])
    quad_axis = evecs[:, 0]

    quad_coord = SkyCoord(
        x=quad_axis[0],
        y=quad_axis[1],
        z=quad_axis[2],
        unit='',
        representation_type='cartesian'
    ).spherical

    quad_ra = quad_coord.lon.deg
    quad_dec = quad_coord.lat.deg




    # ------------------------------------------
    # monte carlo significance
    # ------------------------------------------
    print("\ncomputing Monte Carlo significance...")

    nMC = 20000
    dip_rand = []
    quad_rand = []

    for _ in range(nMC):
        rv = random_unit_vectors(len(vecs))
        D = fit_dipole(rv)
        Qr = fit_quadrupole(rv)

        dip_rand.append(np.linalg.norm(D))
        qr_evals, _ = eig(Qr)
        quad_rand.append(np.max(np.abs(qr_evals)))

    dip_rand = np.array(dip_rand)
    quad_rand = np.array(quad_rand)

    p_dip = np.mean(dip_rand >= dipole_amp)
    p_quad = np.mean(quad_rand >= quad_amp)

    # ------------------------------------------
    # print results
    # ------------------------------------------
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)

    print(f"dipole amplitude: {dipole_amp:.4f}")
    print(f"dipole direction: RA={dip_ra:.2f}°, Dec={dip_dec:.2f}°")
    print(f"dipole p-value:   {p_dip:.5f}")

    print("\nquadrupole amplitude:", f"{quad_amp:.4f}")
    print(f"quadrupole axis: RA={quad_ra:.2f}°, Dec={quad_dec:.2f}°")
    print(f"quadrupole p-value: {p_quad:.5f}")

    # ------------------------------------------
    # sky plot
    # ------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="mollweide")

    # FRBs
    ra_rad = np.radians(frbs["ra"].values)
    dec_rad = np.radians(frbs["dec"].values)
    ra_rad_plot = np.remainder(ra_rad + 2*np.pi, 2*np.pi)
    ra_rad_plot[ra_rad_plot > np.pi] -= 2*np.pi

    ax.scatter(ra_rad_plot, dec_rad, s=10, alpha=0.4, label="FRBs")

    # dipole axis
    dra = np.radians(dip_ra)
    dra = np.remainder(dra + 2*np.pi, 2*np.pi)
    if dra > np.pi:
        dra -= 2*np.pi
    ddec = np.radians(dip_dec)

    ax.scatter(dra, ddec, c="red", s=200, marker="*", label="Dipole axis")

    # quadrupole axis
    qra = np.radians(quad_ra)
    qra = np.remainder(qra + 2*np.pi, 2*np.pi)
    if qra > np.pi:
        qra -= 2*np.pi
    qdec = np.radians(quad_dec)

    ax.scatter(qra, qdec, c="blue", s=200, marker="^", label="Quadrupole axis")

    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("FRB Sky Dipole + Quadrupole")

    plt.savefig("frb_sky_dipole_quadrupole.png", dpi=150, bbox_inches="tight")
    print("\nsaved → frb_sky_dipole_quadrupole.png")
    print("\ndone.")
    print("=" * 60)


# ----------------------------------------------
# run
# ----------------------------------------------
if __name__ == "__main__":
    main()
