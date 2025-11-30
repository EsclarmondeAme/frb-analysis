#!/usr/bin/env python
"""
frb_sky_axis_vs_cmb.py

compare the frb sky dipole axis (from ra/dec in frbs.csv)
to the cmb low-ℓ modulation axis in galactic coordinates.

- fits a simple dipole axis to the frb sky distribution
- converts that axis to galactic (l, b)
- computes angular separation to the cmb axis
- runs a monte carlo test with isotropic mock skies
  to estimate how unusual that separation would be by chance.
"""

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation

rng = np.random.default_rng(12345)


def fit_sky_dipole(ra_deg: np.ndarray, dec_deg: np.ndarray) -> SkyCoord:
    """
    fit a simple dipole axis to a set of sky positions in icrs coordinates.

    returns:
        SkyCoord of the best-fit dipole axis in icrs.
    """
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    xyz = coords.cartesian.get_xyz().value  # shape (3, n)

    # mean direction vector
    mean_vec = xyz.mean(axis=1)
    norm = np.linalg.norm(mean_vec)
    if norm == 0:
        raise ValueError("mean direction vector has zero norm")

    mean_vec /= norm

    rep = CartesianRepresentation(
        x=mean_vec[0] * u.one,
        y=mean_vec[1] * u.one,
        z=mean_vec[2] * u.one,
    )
    sph = rep.represent_as("spherical")

    ra_axis = sph.lon.to_value(u.deg) % 360.0
    dec_axis = sph.lat.to_value(u.deg)

    return SkyCoord(ra=ra_axis * u.deg, dec=dec_axis * u.deg, frame="icrs")


def random_isotropic_sky(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    draw n random isotropic directions on the sphere.

    returns:
        ra_deg, dec_deg arrays.
    """
    # ra uniform in [0, 2pi)
    ra = rng.uniform(0.0, 2.0 * np.pi, size=n)

    # cos(dec) uniform in [-1, 1]
    z = rng.uniform(-1.0, 1.0, size=n)
    dec = np.arcsin(z)

    return np.degrees(ra), np.degrees(dec)


def main() -> None:
    print("=" * 60)
    print("frb sky dipole axis vs cmb low-ℓ axis")
    print("=" * 60)

    # ------------------------------------------------------------
    # load frb sample
    # ------------------------------------------------------------
    frb = pd.read_csv("frbs.csv")

    if not {"ra", "dec"}.issubset(frb.columns):
        raise SystemExit("frbs.csv must contain 'ra' and 'dec' columns.")

    frb = frb.dropna(subset=["ra", "dec"])
    ra = frb["ra"].to_numpy(dtype=float)
    dec = frb["dec"].to_numpy(dtype=float)

    print(f"frbs with valid ra/dec: {len(frb)}")

    # ------------------------------------------------------------
    # fit frb sky dipole axis
    # ------------------------------------------------------------
    dip_icrs = fit_sky_dipole(ra, dec)
    dip_gal = dip_icrs.galactic

    print("-" * 60)
    print("frb sky dipole axis:")
    print(f"  icrs:    ra = {dip_icrs.ra.deg:7.2f}°, dec = {dip_icrs.dec.deg:7.2f}°")
    print(f"  galactic: l  = {dip_gal.l.deg:7.2f}°, b   = {dip_gal.b.deg:7.2f}°")

    # ------------------------------------------------------------
    # cmb low-ℓ axis (from your earlier cmb_dipole_modulation.py)
    # ------------------------------------------------------------
    cmb_l = 152.62
    cmb_b = 4.03
    cmb_axis = SkyCoord(l=cmb_l * u.deg, b=cmb_b * u.deg, frame="galactic")

    sep_obs = dip_gal.separation(cmb_axis).deg

    print("-" * 60)
    print("cmb low-ℓ modulation axis:")
    print(f"  galactic: l  = {cmb_l:7.2f}°, b   = {cmb_b:7.2f}°")
    print(f"\nangular separation (frb sky dipole ↔ cmb axis): {sep_obs:6.2f}°")

    # ------------------------------------------------------------
    # monte carlo: how often would an isotropic sky
    # produce a dipole axis at least this close to the cmb axis?
    # ------------------------------------------------------------
    print("-" * 60)
    print("running monte carlo for isotropic comparison...")

    n_mc = 5000
    seps = np.empty(n_mc, dtype=float)

    for i in range(n_mc):
        ra_rand, dec_rand = random_isotropic_sky(len(frb))
        dip_rand_icrs = fit_sky_dipole(ra_rand, dec_rand)
        dip_rand_gal = dip_rand_icrs.galactic
        seps[i] = dip_rand_gal.separation(cmb_axis).deg

    p_close = np.mean(seps <= sep_obs)

    print(f"\nmonte carlo trials: {n_mc}")
    print(f"fraction with sep_rand ≤ {sep_obs:5.2f}°: p = {p_close:.4f}")

    print("-" * 60)
    print("interpretation:")
    if sep_obs < 30.0:
        if p_close < 0.05:
            print(
                "  frb sky dipole appears unusually close to the cmb axis "
                "(p < 0.05)."
            )
        else:
            print(
                "  frb sky dipole is geometrically close to the cmb axis, "
                "but similar alignments are not rare for random skies."
            )
    else:
        print(
            "  frb sky dipole is far from the cmb axis; this points to "
            "survey selection (e.g. declination strip) rather than a shared "
            "cosmic axis."
        )

    print("=" * 60)
    print("done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
