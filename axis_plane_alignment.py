#!/usr/bin/env python3
# ============================================================
# AXIS–PLANE ALIGNMENT TEST (fixed)
# ============================================================

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

# ------------------------------------------------------------
# axes we test
# ------------------------------------------------------------

CMB_L = 152.62
CMB_B = 4.03

UNIFIED_L = 159.85
UNIFIED_B = -0.51

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def angle_axis_to_plane(axis_l, axis_b, normal_coord):
    axis = SkyCoord(l=axis_l*u.deg, b=axis_b*u.deg, frame="galactic")
    sep = axis.separation(normal_coord)
    return 90.0 - sep.deg, sep.deg   # angle to plane, angle to normal


def plane_normals():
    """ return normals to galactic, ecliptic, supergalactic planes (in galactic coords) """

    # galactic plane normal (north galactic pole)
    n_gal = SkyCoord(l=0*u.deg, b=90*u.deg, frame="galactic")

    # ecliptic north pole: convert from ecliptic to galactic
    n_ecl = SkyCoord(lat=90*u.deg, lon=0*u.deg, frame="barycentrictrueecliptic").galactic

    # supergalactic north pole (THIS was the bug: use sgl/sgb)
    n_sg = SkyCoord(sgl=0*u.deg, sgb=90*u.deg, frame="supergalactic").galactic

    return n_gal, n_ecl, n_sg


def print_axis_report(name, l, b):
    print(f"\naxis: {name}")
    print(f"  galactic (l,b) = ({l:.2f}°, {b:.2f}°)")

    n_gal, n_ecl, n_sg = plane_normals()

    for plane_name, n_coord in [
        ("galactic plane", n_gal),
        ("ecliptic plane", n_ecl),
        ("supergalactic plane", n_sg),
    ]:
        angle_to_plane, sep_to_normal = angle_axis_to_plane(l, b, n_coord)
        print(f"  relative to {plane_name}:")
        print(f"    angle to plane        ≈ {angle_to_plane:.2f}°")
        print(f"    angle to plane normal ≈ {sep_to_normal:.2f}°")


def main():
    print("===================================================")
    print("AXIS–PLANE ALIGNMENT TEST")
    print("===================================================\n")

    print_axis_report("cmb hemispherical asymmetry", CMB_L, CMB_B)
    print_axis_report("unified axis", UNIFIED_L, UNIFIED_B)

    print("\n---------------- scientific verdict ----------------")
    print("axes lying very close (<10–15°) to any major plane")
    print("normal would suggest a coordinate-system coincidence.")
    print("large separations (>30–40°) indicate the axis is not")
    print("a trivial re-expression of known planes.")
    print("---------------------------------------------------")
    print("analysis complete.")
    print("===================================================\n")


if __name__ == "__main__":
    main()
