import sys
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

print("="*70)
print("PARKES FOOTPRINT-CONSTRAINED NULL TEST")
print("="*70)

if len(sys.argv) < 2:
    print("Usage: python parkes_footprint_null_test.py frbs_parkes_unified.csv")
    sys.exit()

path = sys.argv[1]
df = pd.read_csv(path)

# ---------------------------------------------------------
# Detect RA/Dec columns
# ---------------------------------------------------------
def detect(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise RuntimeError("Missing RA/Dec")

ra_col = detect(df, ["ra","ra_j2000","RA","raj2000"])
dec_col = detect(df, ["dec","dec_j2000","DEC","decj2000"])
df = df.rename(columns={ra_col:"ra", dec_col:"dec"})

# ---------------------------------------------------------
# Parse Parkes RA/Dec
# ---------------------------------------------------------
try:
    coords = SkyCoord(df["ra"].astype(str),
                      df["dec"].astype(str),
                      unit=(u.hourangle, u.deg), frame="icrs")
except:
    coords = SkyCoord(ra=df["ra"].astype(float).values*u.deg,
                      dec=df["dec"].astype(float).values*u.deg,
                      frame="icrs")

g = coords.galactic

l_real = g.l.rad
b_real = g.b.rad
N = len(df)

print(f"Parkes FRBs: N = {N}")

# ---------------------------------------------------------
# Real dipole amplitude
# ---------------------------------------------------------
x = np.cos(b_real)*np.cos(l_real)
y = np.cos(b_real)*np.sin(l_real)
z = np.sin(b_real)
vec = np.array([x.sum(), y.sum(), z.sum()])
R_real = np.linalg.norm(vec) / N

print(f"Real dipole amplitude R_real = {R_real:.4f}")

# ---------------------------------------------------------
# Define RA/Dec footprint windows
# ---------------------------------------------------------
ra_min = np.min(coords.ra.deg)
ra_max = np.max(coords.ra.deg)
dec_min = np.min(coords.dec.deg)
dec_max = np.max(coords.dec.deg)

print(f"Footprint (ICRS):")
print(f"  RA  in [{ra_min:.2f}, {ra_max:.2f}] deg")
print(f"  Dec in [{dec_min:.2f}, {dec_max:.2f}] deg")

# ---------------------------------------------------------
# Monte Carlo footprint-constrained null
# ---------------------------------------------------------
NSIM = 20000
R_null = []

for _ in range(NSIM):
    # sample RA uniformly in Parkes RA window
    ra_rand = np.random.uniform(ra_min, ra_max, N)
    
    # sample Dec uniformly in Parkes Dec window (not exact, but good footprint approx)
    dec_rand = np.random.uniform(dec_min, dec_max, N)
    
    # convert to galactic
    g_rand = SkyCoord(ra=ra_rand*u.deg,
                      dec=dec_rand*u.deg,
                      frame="icrs").galactic
    
    lr = g_rand.l.rad
    br = g_rand.b.rad
    
    xr = np.cos(br)*np.cos(lr)
    yr = np.cos(br)*np.sin(lr)
    zr = np.sin(br)
    
    R = np.linalg.norm([xr.sum(), yr.sum(), zr.sum()]) / N
    R_null.append(R)

R_null = np.array(R_null)

p = np.mean(R_null >= R_real)

print("\n=== RESULTS ===")
print(f"Footprint-null mean dipole = {np.mean(R_null):.4f}")
print(f"Footprint-null std         = {np.std(R_null):.4f}")
print(f"p-value (R_null >= R_real) = {p:.4f}")
print("="*70)
print("done.")
