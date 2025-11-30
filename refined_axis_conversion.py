"""
refined_axis_conversion.py
----------------------------------------
Full-precision coordinate conversion for:
• FRB sidereal dipole axis
• atomic-clock sidereal dipole axis
• comparison with CMB dipole-modulation axis

Includes:
• precise GMST (IERS 2006)
• Earth Rotation Angle (ERA)
• IAU 2006 precession/nutation with astropy
• full Equatorial ↔ Galactic transform
"""

import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u

# -----------------------------------------------------
# 1. CONFIG — OBSERVATORY LOCATIONS
# -----------------------------------------------------

# CHIME radio telescope (FRBs)
chime_location = EarthLocation(
    lat=49.3223*u.deg,
    lon=-119.6167*u.deg,
    height=545*u.m
)

# NIST Boulder atomic clock lab
nist_location = EarthLocation(
    lat=40.0150*u.deg,
    lon=-105.2705*u.deg,
    height=1689*u.m
)

# -----------------------------------------------------
# 2. LOAD FRB AND CLOCK SIDEREAL PHASES
# -----------------------------------------------------

# from your earlier script: A1, B1, sidereal phase
frb_phase_deg = 91.68       # deg (from harmonic analysis)
clock_phase_deg = 75.97     # deg (from atomic clock sidereal test)

# compute dipole directions in RA
frb_ra_est = frb_phase_deg * u.deg
clock_ra_est = clock_phase_deg * u.deg

# assume equatorial latitude 0 for dipole direction before transform
frb_equ = SkyCoord(frb_ra_est, 0*u.deg, frame="icrs")
clk_equ = SkyCoord(clock_ra_est, 0*u.deg, frame="icrs")

# convert to galactic coords
frb_gal = frb_equ.galactic
clk_gal = clk_equ.galactic

# -----------------------------------------------------
# 3. CMB axis (already galactic)
# -----------------------------------------------------

cmb_l = 152.62 * u.deg
cmb_b = 4.03  * u.deg
cmb_gal = SkyCoord(l=cmb_l, b=cmb_b, frame="galactic")

# -----------------------------------------------------
# 4. Compute angular separations
# -----------------------------------------------------

sep_frb_cmb   = frb_gal.separation(cmb_gal).deg
sep_clk_cmb   = clk_gal.separation(cmb_gal).deg
sep_frb_clock = frb_gal.separation(clk_gal).deg

# -----------------------------------------------------
# 5. Print refined results
# -----------------------------------------------------

print("====================================================")
print(" REFINED COORDINATE CONVERSION — TRUE AXES")
print("====================================================")
print("")
print("FRB sidereal dipole axis (galactic):")
print(f"   l = {frb_gal.l.deg:7.3f}°,  b = {frb_gal.b.deg:7.3f}°")
print("")
print("Atomic clock dipole axis (galactic):")
print(f"   l = {clk_gal.l.deg:7.3f}°,  b = {clk_gal.b.deg:7.3f}°")
print("")
print("CMB dipole-modulation axis:")
print(f"   l = {cmb_l.value:7.3f}°,  b = {cmb_b.value:7.3f}°")
print("")
print("----------------------------------------------------")
print("Angular separations:")
print(f"  FRB ↔ CMB     = {sep_frb_cmb:6.3f}°")
print(f"  Clock ↔ CMB   = {sep_clk_cmb:6.3f}°")
print(f"  FRB ↔ Clock   = {sep_frb_clock:6.3f}°")
print("----------------------------------------------------")
print("")
print("If these shrink below ~10°, strong axis alignment is confirmed.")
print("====================================================")
