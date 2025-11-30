#!/usr/bin/env python3
"""
frb_neutrino_cone_axis_comparison.py

Compare the *geometric sky directions* of:
1. the FRB best-fit sidereal cone axis (converted to RA/Dec),
2. the neutrino best-fit dipole direction (RA/Dec),
3. and compute:
    - angular separation on the sky,
    - Monte Carlo significance of alignment.

Inputs:
    frbs.csv    (columns: name, utc, mjd, ra, dec, ...)
    neutrinos.csv (IceCube data; must contain ra, dec, mjd/utc)

Output:
    prints comparison + plots the two axis directions on a sky map
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import logging

# ------------------------------------------------------------
# logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def sidereal_phase_from_mjd(mjd):
    """
    convert MJD → sidereal phase φ ∈ [0,1)
    φ = GMST/24h (mod 1)
    """
    T = (mjd - 51544.5) / 36525.0
    gmst = 6.697374558 + 2400.051336*T + 0.000025862*T*T
    gmst = (gmst % 24.0) / 24.0
    return gmst


def fit_dipole(phases):
    """
    compute dipole harmonic:
        A = mean(cos 2πφ)
        B = mean(sin 2πφ)
        R = sqrt(A^2 + B^2)
    """
    th = 2*np.pi*phases
    A = np.mean(np.cos(th))
    B = np.mean(np.sin(th))
    R = np.sqrt(A*A + B*B)
    return A, B, R


def axis_RA_from_phase(phi):
    """
    convert sidereal phase (fraction of 24h)
    to approximate RA in degrees:
    
        RA ≈ 360° * φ
    
    (This is exact if φ is defined as GMST/24h.)
    """
    return (360.0 * phi) % 360.0


def random_sky_directions(n):
    """
    sample n random directions on the sphere
    returns (ra_deg, dec_deg)
    """
    ra = np.random.uniform(0, 360, n)
    z = np.random.uniform(-1, 1, n)
    dec = np.degrees(np.arcsin(z))
    return ra, dec


# ------------------------------------------------------------
# load data
# ------------------------------------------------------------
logging.info("loading FRBs...")
frb = pd.read_csv("frbs.csv")
if "mjd" not in frb.columns:
    raise RuntimeError("frbs.csv must contain 'mjd' column")

frb = frb.dropna(subset=["mjd", "ra", "dec"])
logging.info(f"FRBs usable: {len(frb)}")

logging.info("loading neutrinos...")
nu = pd.read_csv("neutrinos.csv")
if "mjd" in nu.columns:
    nu["mjd"] = nu["mjd"]
elif "utc" in nu.columns:
    nu["mjd"] = pd.to_datetime(nu["utc"], utc=True).astype("datetime64[ns]").view('int64')/86400000000000.0 + 40587
else:
    raise RuntimeError("neutrinos.csv must contain mjd or utc")

nu = nu.dropna(subset=["mjd", "ra", "dec"])
logging.info(f"neutrinos usable: {len(nu)}")

# ------------------------------------------------------------
# compute sidereal phases
# ------------------------------------------------------------
logging.info("computing sidereal phases...")
frb["phi"] = sidereal_phase_from_mjd(frb["mjd"].values)
nu["phi"] = sidereal_phase_from_mjd(nu["mjd"].values)

# ------------------------------------------------------------
# fit FRB cone axis ≈ dipole axis in phase
# ------------------------------------------------------------
A_f, B_f, R_f = fit_dipole(frb["phi"].values)
phi0_f = (np.arctan2(B_f, A_f) / (2*np.pi)) % 1.0
RA_f = axis_RA_from_phase(phi0_f)

# approximate Dec of FRB axis:
# use weighted average of FRB arrival directions projected along dipole
th = 2*np.pi*frb["phi"].values
w = np.cos(th - 2*np.pi*phi0_f)
Dec_f = np.average(frb["dec"], weights=np.clip(w, 0, None))

# ------------------------------------------------------------
# fit neutrino dipole axis
# ------------------------------------------------------------
A_n, B_n, R_n = fit_dipole(nu["phi"].values)
phi0_n = (np.arctan2(B_n, A_n) / (2*np.pi)) % 1.0
RA_n = axis_RA_from_phase(phi0_n)

th_n = 2*np.pi*nu["phi"].values
w_n = np.cos(th_n - 2*np.pi*phi0_n)
Dec_n = np.average(nu["dec"], weights=np.clip(w_n, 0, None))

# ------------------------------------------------------------
# sky coordinates
# ------------------------------------------------------------
coord_f = SkyCoord(RA_f*u.deg, Dec_f*u.deg, frame="icrs")
coord_n = SkyCoord(RA_n*u.deg, Dec_n*u.deg, frame="icrs")

sep = coord_f.separation(coord_n).deg

logging.info("------------------------------------------------------------")
logging.info("FRB cone-axis direction (sky):")
logging.info(f"  RA = {RA_f:.2f} deg")
logging.info(f"  Dec = {Dec_f:.2f} deg")
logging.info("")
logging.info("neutrino dipole direction (sky):")
logging.info(f"  RA = {RA_n:.2f} deg")
logging.info(f"  Dec = {Dec_n:.2f} deg")
logging.info("")
logging.info(f"angular separation: {sep:.3f} deg")
logging.info("------------------------------------------------------------")

# ------------------------------------------------------------
# Monte Carlo: random neutrino axis directions
# ------------------------------------------------------------
logging.info("running Monte Carlo...")
Nmc = 20000
ra_rand, dec_rand = random_sky_directions(Nmc)

coords_rand = SkyCoord(ra_rand*u.deg, dec_rand*u.deg, frame="icrs")
sep_rand = coord_f.separation(coords_rand).deg

p_mc = np.mean(sep_rand <= sep)

logging.info(f"Monte Carlo P(sep_rand <= sep_obs) = {p_mc:.4f}")
logging.info("------------------------------------------------------------")

# ------------------------------------------------------------
# plot
# ------------------------------------------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="mollweide")

ax.scatter(np.radians(RA_f-180), np.radians(Dec_f),
           s=120, color="red", label="FRB cone axis")

ax.scatter(np.radians(RA_n-180), np.radians(Dec_n),
           s=120, color="blue", label="neutrino dipole axis")

ax.set_title("FRB cone axis vs neutrino dipole axis")
ax.grid(True)
ax.legend()

plt.savefig("frb_neutrino_cone_axis.png", dpi=150)
logging.info("saved → frb_neutrino_cone_axis.png")
logging.info("done.")
