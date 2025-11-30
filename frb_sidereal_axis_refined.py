#!/usr/bin/env python
"""
frb_sidereal_axis_refined.py

compute the true sky direction of the frb sidereal-phase dipole using:

- exact lst per detection
- chime latitude / longitude
- mjd → utc → sidereal time via astropy
- weighted fourier fit to the detection times
- convert sidereal dipole phase → absolute RA direction
- transform RA/Dec → galactic (l, b)
- compare to cmb and atomic clock axes
"""

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

# chime location
chime = EarthLocation.from_geodetic(
    lon=-119.62 * u.deg,
    lat=49.32 * u.deg,
    height=0 * u.m,
)

# cmb low-l modulation axis (planck)
cmb_axis = SkyCoord(
    l=152.62 * u.deg,
    b=4.03 * u.deg,
    frame="galactic"
)

# atomic clock sidereal modulation axis (converted already)
clock_axis = SkyCoord(
    l=166.04 * u.deg,
    b=-0.87 * u.deg,
    frame="galactic"
)


def weighted_sidereal_harmonic(theta, w=None):
    """
    fit A*cos(theta) + B*sin(theta) to sidereal phase angles.
    """
    if w is None:
        w = np.ones_like(theta)

    A = np.sum(w * np.cos(theta))
    B = np.sum(w * np.sin(theta))

    amp = np.sqrt(A**2 + B**2)
    phase = np.arctan2(B, A)  # radians

    return A, B, amp, phase


def main():
    print("=" * 70)
    print("frb sidereal dipole → true sky axis (refined)".center(70))
    print("=" * 70)

    # ---------------------------------------------------------
    # load frbs
    # requires columns: mjd (float), ra (deg), dec (deg)
    # ---------------------------------------------------------
    frb = pd.read_csv("frbs.csv")

    if not {"mjd", "ra", "dec"}.issubset(frb.columns):
        raise SystemExit("frbs.csv must contain mjd, ra, dec columns.")

    frb = frb.dropna(subset=["mjd", "ra", "dec"])

    mjd = frb["mjd"].to_numpy()
    ra = frb["ra"].to_numpy()
    dec = frb["dec"].to_numpy()

    n = len(frb)
    print(f"frbs loaded: {n}")

    # ---------------------------------------------------------
    # compute exact sidereal time for each detection
    # ---------------------------------------------------------
    t = Time(mjd, format="mjd", scale="utc")
    lst = t.sidereal_time("mean", longitude=chime.lon)
    lst_rad = lst.to_value(u.rad)

    # ---------------------------------------------------------
    # fit sidereal harmonic dipole
    # ---------------------------------------------------------
    A1, B1, amp, phase = weighted_sidereal_harmonic(lst_rad)

    print("-" * 70)
    print("sidereal harmonic fit (refined):")
    print(f"A1 = {A1:.5f}")
    print(f"B1 = {B1:.5f}")
    print(f"amplitude = {amp:.5f}")
    print(f"phase φ (rad) = {phase:.5f}")
    print(f"phase φ (deg) = {np.degrees(phase):.2f}°")

    # ---------------------------------------------------------
    # convert sidereal phase → RA axis
    # sidereal phase gives where max rate occurs
    # convert: RA_dipole = φ (sidereal) mapped to sky RA
    # ---------------------------------------------------------
    ra_dipole = (np.degrees(phase) % 360.0)

    # dec of sidereal dipole = weighted mean of FRB declinations
    # (approximation until direction-fitting is added)
    dec_dipole = np.average(dec)

    frb_axis_icrs = SkyCoord(
        ra=ra_dipole * u.deg,
        dec=dec_dipole * u.deg,
        frame="icrs"
    )

    frb_axis_gal = frb_axis_icrs.galactic

    print("-" * 70)
    print("frb sidereal dipole axis (refined):")
    print(f"ICRS:    RA = {frb_axis_icrs.ra.deg:.2f}°, Dec = {frb_axis_icrs.dec.deg:.2f}°")
    print(f"Galactic: l = {frb_axis_gal.l.deg:.2f}°, b = {frb_axis_gal.b.deg:.2f}°")

    # ---------------------------------------------------------
    # angular separations
    # ---------------------------------------------------------
    sep_cmb = frb_axis_gal.separation(cmb_axis).deg
    sep_clock = frb_axis_gal.separation(clock_axis).deg
    sep_cmb_clock = cmb_axis.separation(clock_axis).deg

    print("-" * 70)
    print("angular separations:")
    print(f"FRB ↔ CMB:   {sep_cmb:.2f}°")
    print(f"FRB ↔ Clock: {sep_clock:.2f}°")
    print(f"CMB ↔ Clock: {sep_cmb_clock:.2f}°")

    print("=" * 70)
    print("done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
