import sys
import pandas as pd
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

"""
FINAL NO-ERROR VERSION
usage:
    python build_frbs_parkes_unified.py frbs_parkes.csv frbs_unified.csv
"""

parkes_csv = sys.argv[1]
unified_csv = sys.argv[2]

# ----------------------------------------------------------------------
# load catalogs
# ----------------------------------------------------------------------
df_p = pd.read_csv(parkes_csv)
df_u = pd.read_csv(unified_csv)

# ----------------------------------------------------------------------
# detect RA/Dec columns in Parkes file
# ----------------------------------------------------------------------
def detect_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise RuntimeError(f"Could not find any of {names} in columns {df.columns.tolist()}")

ra_col  = detect_col(df_p, ["ra", "ra_j2000", "RAJ2000", "raj2000", "RA"])
dec_col = detect_col(df_p, ["dec", "dec_j2000", "DEJ2000", "decj2000", "DEC"])

df_p = df_p.rename(columns={ra_col:"ra", dec_col:"dec"})

# ----------------------------------------------------------------------
# parse Parkes RA/Dec safely
# ----------------------------------------------------------------------
def parse_coords(df):
    try:
        # attempt sexagesimal parse (hh:mm:ss)
        return SkyCoord(df["ra"].astype(str),
                        df["dec"].astype(str),
                        unit=(u.hourangle, u.deg),
                        frame="icrs")
    except:
        # fallback to degrees
        return SkyCoord(ra=df["ra"].astype(float).values*u.deg,
                        dec=df["dec"].astype(float).values*u.deg,
                        frame="icrs")

coords_p = parse_coords(df_p)

# ----------------------------------------------------------------------
# compute MJD
# ----------------------------------------------------------------------
df_p["mjd"] = df_p["utc"].apply(lambda x: Time(x, format="isot").mjd)

# ----------------------------------------------------------------------
# compute z_est
# ----------------------------------------------------------------------
df_p["z_est"] = df_p["dm"] * 1e-3

# ----------------------------------------------------------------------
# unified axis reconstruction from unified CSV
# ----------------------------------------------------------------------
coords_full = SkyCoord(ra=df_u["ra"].values*u.deg,
                       dec=df_u["dec"].values*u.deg,
                       frame="icrs")

# full catalog unit vectors
xf = np.cos(coords_full.dec.radian)*np.cos(coords_full.ra.radian)
yf = np.cos(coords_full.dec.radian)*np.sin(coords_full.ra.radian)
zf = np.sin(coords_full.dec.radian)

# weights = -cos(theta_unified)
w = -np.cos(np.radians(df_u["theta_unified"].values))

xu = np.sum(xf * w)
yu = np.sum(yf * w)
zu = np.sum(zf * w)

norm = np.sqrt(xu*xu + yu*yu + zu*zu)
xu /= norm
yu /= norm
zu /= norm

# axis spherical coordinates
axis = SkyCoord(x=xu, y=yu, z=zu, representation_type='cartesian')
axis_sph = axis.spherical
axis_ra  = axis_sph.lon.radian
axis_dec = axis_sph.lat.radian

# ----------------------------------------------------------------------
# compute Parkes theta_unified, phi_unified
# ----------------------------------------------------------------------
xp = np.cos(coords_p.dec.radian)*np.cos(coords_p.ra.radian)
yp = np.cos(coords_p.dec.radian)*np.sin(coords_p.ra.radian)
zp = np.sin(coords_p.dec.radian)

dot = xp*xu + yp*yu + zp*zu
df_p["theta_unified"] = np.degrees(np.arccos(np.clip(dot, -1, 1)))

# build orthonormal basis around axis
e1 = np.array([xu, yu, zu])
temp = np.array([1, 0, 0])
if abs(np.dot(temp, e1)) > 0.8:
    temp = np.array([0, 1, 0])
e2 = temp - np.dot(temp, e1)*e1
e2 /= np.linalg.norm(e2)
e3 = np.cross(e1, e2)

v = np.vstack([xp, yp, zp]).T
p2 = v @ e2
p3 = v @ e3

df_p["phi_unified"] = np.degrees(np.arctan2(p3, p2)) % 360

# ----------------------------------------------------------------------
# fill missing columns
# ----------------------------------------------------------------------
df_p["snr"] = df_p.get("snr", np.nan)
df_p["width"] = df_p.get("width", df_p.get("width_ms", np.nan))
df_p["fluence"] = df_p.get("fluence", np.nan)

# ----------------------------------------------------------------------
# reorder
# ----------------------------------------------------------------------
cols = [
    "name","utc","mjd","ra","dec","dm","snr","width","fluence",
    "z_est","theta_unified","phi_unified"
]

df_out = df_p[cols]

df_out.to_csv("frbs_parkes_unified.csv", index=False)
print("saved frbs_parkes_unified.csv with", len(df_out), "FRBs")
