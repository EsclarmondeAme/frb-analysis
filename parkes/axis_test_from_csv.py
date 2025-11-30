import sys
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

print("="*70)
print("FRB AXIS RECONSTRUCTION — UNIVERSAL COMPATIBILITY VERSION")
print("="*70)

if len(sys.argv) < 2:
    print("Usage: python axis_test_csv_final2.py <csv>")
    sys.exit()

path = sys.argv[1]
df = pd.read_csv(path)

# ----------------------------------------------------------
# detect RA/Dec
# ----------------------------------------------------------
def detect(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise RuntimeError(f"Missing RA/Dec columns: {names}")

ra_col = detect(df, ["ra","ra_j2000","RAJ2000","RA","raj2000"])
dec_col = detect(df, ["dec","dec_j2000","DEC","DEJ2000","decj2000"])

df = df.rename(columns={ra_col:"ra", dec_col:"dec"})

# ----------------------------------------------------------
# try sexagesimal, then numeric degrees
# ----------------------------------------------------------
try:
    coords = SkyCoord(df["ra"].astype(str), df["dec"].astype(str),
                      unit=(u.hourangle, u.deg), frame="icrs")
except:
    coords = SkyCoord(ra=df["ra"].astype(float).values*u.deg,
                      dec=df["dec"].astype(float).values*u.deg,
                      frame="icrs")

# ----------------------------------------------------------
# convert to galactic using SkyCoord (safe)
# ----------------------------------------------------------
g = coords.galactic

# ----------------------------------------------------------
# compute dipole vector sum
# ----------------------------------------------------------
l_rad = g.l.radian
b_rad = g.b.radian

x = np.cos(b_rad) * np.cos(l_rad)
y = np.cos(b_rad) * np.sin(l_rad)
z = np.sin(b_rad)

vec = np.array([x.sum(), y.sum(), z.sum()])
vec /= np.linalg.norm(vec)

# ----------------------------------------------------------
# convert the vector back to Galactic l,b manually
# ----------------------------------------------------------
lx = vec[0]
ly = vec[1]
lz = vec[2]

# latitude
b_est = np.degrees(np.arcsin(lz))

# longitude
l_est = np.degrees(np.arctan2(ly, lx)) % 360

print("\nEstimated axis from FRB CSV:")
print(f"   l = {l_est:.2f}°")
print(f"   b = {b_est:.2f}°")

# ----------------------------------------------------------
# compare to unified axis
# ----------------------------------------------------------
unif_l = 160.4
unif_b = 0.1

# convert to radians
l1 = np.radians(l_est)
b1 = np.radians(b_est)
l2 = np.radians(unif_l)
b2 = np.radians(unif_b)

# spherical angle
sep = np.degrees(
    np.arccos(
        np.sin(b1)*np.sin(b2) +
        np.cos(b1)*np.cos(b2)*np.cos(l1-l2)
    )
)

print(f"\nSeparation from unified axis: {sep:.2f}°")

# ----------------------------------------------------------
# null distribution
# ----------------------------------------------------------
N = 200000
rand_b = np.arcsin(np.random.uniform(-1,1,N))
rand_l = np.random.uniform(0,2*np.pi,N)

seps = np.degrees(
    np.arccos(
        np.sin(rand_b)*np.sin(b2) +
        np.cos(rand_b)*np.cos(b2)*np.cos(rand_l-l2)
    )
)

p = np.mean(seps <= sep)
print(f"p-value (isotropic alignment chance): {p:.4f}")

print("="*70)
print("done.")
