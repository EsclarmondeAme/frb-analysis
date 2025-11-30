#!/usr/bin/env python
"""
frb_cmb_axis_comparison.py

compare frb sky axes (dipole from frbs.csv, octupole from frb_sky_octupole.py)
to cmb low-ℓ axes (ℓ=2,3) that we measured earlier.

it prints:
- frb dipole axis in galactic coords
- frb octupole axis in galactic coords
- cmb quadrupole & octupole axes (galactic)
- angular separations between all pairs
- analytic probabilities P(theta_rand <= theta_obs) for each pair
  for a random axis on the sphere.

requires:
  - frbs.csv with columns ra, dec (deg)
  - astropy, numpy, matplotlib (for a little sky plot)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u


def unit_vector_from_radec(ra_deg, dec_deg):
    """convert ra,dec in deg to cartesian unit vector."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    v = np.vstack([x, y, z])
    return v


def angle_between(v1, v2):
    """angle in degrees between two 3d unit vectors."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos_t = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


def random_alignment_p(theta_deg):
    """
    analytic probability that a random axis lies within theta of a fixed axis.

    for isotropic directions: P(theta <= θ0) = 1 - cos(θ0).
    """
    theta_rad = np.radians(theta_deg)
    return 1.0 - np.cos(theta_rad)


def main():
    print("=" * 60)
    print("frb vs cmb low-ℓ axis comparison")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. load frb catalog and compute sky dipole axis
    # ------------------------------------------------------------
    frb = pd.read_csv("frbs.csv")
    frb = frb.dropna(subset=["ra", "dec"])

    print(f"loaded frbs with ra/dec: {len(frb)}")

    v = unit_vector_from_radec(frb["ra"].values, frb["dec"].values)
    mean_v = np.mean(v, axis=1)
    mean_v /= np.linalg.norm(mean_v)

    # convert mean vector to equatorial & galactic coordinates using astropy
    # convert mean vector to an ICRS sky coordinate
    dip_coord_icrs = SkyCoord(
        x=mean_v[0], y=mean_v[1], z=mean_v[2],
        unit=(u.one, u.one, u.one),
        frame="icrs",
        representation_type="cartesian"
    )


    # convert cartesian → spherical explicitly
    dip_sph = dip_coord_icrs.represent_as("spherical")

    dip_ra = dip_sph.lon.deg     # right ascension
    dip_dec = dip_sph.lat.deg    # declination



    # convert to galactic coordinates
    dip_coord_gal = dip_coord_icrs.transform_to("galactic")
    dip_l = dip_coord_gal.l.deg
    dip_b = dip_coord_gal.b.deg


    print("\nfrb sky dipole (from mean position):")
    print(f"  ra  = {dip_ra:.2f} deg")
    print(f"  dec = {dip_dec:.2f} deg")
    print(f"  l   = {dip_l:.2f} deg")
    print(f"  b   = {dip_b:.2f} deg")

    # cartesian for geometry
    dip_vec = mean_v.copy()

    # ------------------------------------------------------------
    # 2. frb octupole axis (use result from frb_sky_octupole.py)
    # ------------------------------------------------------------
    # these are the values your octupole script printed:
    #   octupole axis: RA≈135.69°, Dec≈-35.35°
    oct_ra = 135.69
    oct_dec = -35.35

    oct_coord_icrs = SkyCoord(ra=oct_ra * u.deg, dec=oct_dec * u.deg, frame="icrs")
    oct_coord_gal = oct_coord_icrs.galactic
    oct_l = oct_coord_gal.l.deg
    oct_b = oct_coord_gal.b.deg

    oct_vec = unit_vector_from_radec(oct_ra, oct_dec).mean(axis=1)

    print("\nfrb sky octupole axis (from frb_sky_octupole.py):")
    print(f"  ra  = {oct_ra:.2f} deg")
    print(f"  dec = {oct_dec:.2f} deg")
    print(f"  l   = {oct_l:.2f} deg")
    print(f"  b   = {oct_b:.2f} deg")

    # ------------------------------------------------------------
    # 3. cmb low-ℓ axes (from earlier cmb_real_axis_alignment.py)
    # ------------------------------------------------------------
    # planck PR3 quadrupole & octupole axes in galactic coords:
    cmb_l2_l = 130.76
    cmb_l2_b = 1.76

    cmb_l3_l = 292.14
    cmb_l3_b = 38.69

    cmb_l2 = SkyCoord(l=cmb_l2_l * u.deg, b=cmb_l2_b * u.deg, frame="galactic")
    cmb_l3 = SkyCoord(l=cmb_l3_l * u.deg, b=cmb_l3_b * u.deg, frame="galactic")

    cmb_l2_vec = unit_vector_from_radec(
        cmb_l2.icrs.ra.deg, cmb_l2.icrs.dec.deg
    ).mean(axis=1)
    cmb_l3_vec = unit_vector_from_radec(
        cmb_l3.icrs.ra.deg, cmb_l3.icrs.dec.deg
    ).mean(axis=1)

    print("\ncmb low-ℓ axes (planck):")
    print(f"  ℓ=2 quadrupole:  l={cmb_l2_l:.2f} deg, b={cmb_l2_b:.2f} deg")
    print(f"  ℓ=3 octupole:    l={cmb_l3_l:.2f} deg, b={cmb_l3_b:.2f} deg")

    # ------------------------------------------------------------
    # 4. angular separations and analytic probabilities
    # ------------------------------------------------------------
    print("\n" + "-" * 60)
    print("angular separations")
    print("-" * 60)

    def report_pair(name1, v1, name2, v2):
        theta = angle_between(v1, v2)
        p = random_alignment_p(theta)
        print(f"{name1} ↔ {name2}:  θ = {theta:6.2f} deg,  P(θ_rand ≤ θ) = {p:7.4f}")

    report_pair("frb dipole", dip_vec, "cmb ℓ=2", cmb_l2_vec)
    report_pair("frb dipole", dip_vec, "cmb ℓ=3", cmb_l3_vec)
    report_pair("frb octupole", oct_vec, "cmb ℓ=2", cmb_l2_vec)
    report_pair("frb octupole", oct_vec, "cmb ℓ=3", cmb_l3_vec)

    print("\nnote: for a truly random axis, typical separations are ~60 deg;")
    print("smaller angles mean stronger alignment, and P gives the chance")
    print("that a random axis would be at least this well aligned.")

    # ------------------------------------------------------------
    # 5. quick sky plot (galactic coords)
    # ------------------------------------------------------------
    print("\ncreating sky plot with all axes...")

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="mollweide")

    # plot frb positions for context
    frb_coords = SkyCoord(
        ra=frb["ra"].values * u.deg,
        dec=frb["dec"].values * u.deg,
        frame="icrs",
    ).galactic
    l_plot = frb_coords.l.wrap_at(180 * u.deg).rad
    b_plot = frb_coords.b.rad
    ax.scatter(l_plot, b_plot, s=8, alpha=0.5, color="tab:blue", label="frbs")

    def add_axis_marker(coord, label, marker, color):
        l = coord.l.wrap_at(180 * u.deg).rad
        b = coord.b.rad
        ax.scatter(l, b, s=120, marker=marker, color=color,
                   edgecolor="black", linewidth=1.0, label=label)

    add_axis_marker(dip_coord_gal, "frb dipole", "o", "gold")
    add_axis_marker(oct_coord_gal, "frb octupole", "^", "red")
    add_axis_marker(cmb_l2, "cmb ℓ=2", "s", "green")
    add_axis_marker(cmb_l3, "cmb ℓ=3", "D", "purple")

    ax.grid(True, alpha=0.4)
    ax.set_title("frb & cmb low-ℓ axes (galactic)")
    ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    plt.savefig("frb_cmb_axis_comparison.png", dpi=150, bbox_inches="tight")
    print("saved → frb_cmb_axis_comparison.png")
    print("\n(done.)")


if __name__ == "__main__":
    main()
