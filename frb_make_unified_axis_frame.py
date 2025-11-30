#!/usr/bin/env python3
import numpy as np
import pandas as pd
import json
from astropy.coordinates import SkyCoord
import astropy.units as u

# -------------------------------------------------------
# load FRBs
# -------------------------------------------------------
frb = pd.read_csv("frbs.csv")

# must have RA/Dec columns
if not {"ra","dec"}.issubset(frb.columns):
    print("ERROR: frbs.csv must contain ra, dec columns")
    exit()

# -------------------------------------------------------
# load unified axis from axes.json
# -------------------------------------------------------
with open("axes.json","r") as f:
    axes = json.load(f)

if "unified_axis" not in axes:
    print("ERROR: axes.json missing 'unified_axis'")
    exit()

ua = axes["unified_axis"]
ua_l = ua["l"]
ua_b = ua["b"]

axis_coord = SkyCoord(l=ua_l*u.deg, b=ua_b*u.deg, frame="galactic")

# -------------------------------------------------------
# convert FRBs to galactic
# -------------------------------------------------------
coords = SkyCoord(ra=frb["ra"].values*u.deg,
                  dec=frb["dec"].values*u.deg,
                  frame="icrs")

gal = coords.galactic

# angular separation θ from unified axis
theta = gal.separation(axis_coord).deg

# azimuth ϕ around axis (in axis-aligned frame)
# convert to axis frame using position angle
phi = gal.position_angle(axis_coord).deg

# store in frb table
frb["theta_unified"] = theta
frb["phi_unified"] = phi

# -------------------------------------------------------
# save output
# -------------------------------------------------------
frb.to_csv("frbs_unified.csv", index=False)
print("saved: frbs_unified.csv  (now includes theta_unified, phi_unified)")
