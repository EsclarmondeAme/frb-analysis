#!/usr/bin/env python3
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("=======================================================")
print("          FRB AXIS–REDSHIFT DRIFT TEST")
print("=======================================================\n")

# ---------------------------------------------------------
# 1. Load FRB catalog
# ---------------------------------------------------------
try:
    frb = pd.read_csv("frbs.csv")
    print("loaded: frbs.csv")
except:
    print("ERROR: could not load frbs.csv")
    exit()

# require RA, DEC, z
frb = frb.dropna(subset=["ra", "dec", "z_est"])
frb = frb.rename(columns={"z_est": "z"})


ra = frb["ra"].to_numpy()
dec = frb["dec"].to_numpy()
z = frb["z"].to_numpy()

coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
gal = coords.galactic
l = gal.l.deg
b = gal.b.deg

# ---------------------------------------------------------
# 2. Fit axis as function of redshift
# ---------------------------------------------------------
N = len(z)
z_grid = np.linspace(z.min(), z.max(), 20)
l_drift = []
b_drift = []

def fit_axis(sub_ra, sub_dec):
    c = SkyCoord(ra=sub_ra*u.deg, dec=sub_dec*u.deg, frame="icrs")
    g = c.galactic
    return np.mean(g.l.deg), np.mean(g.b.deg)

for i in range(len(z_grid)-1):
    lo = z_grid[i]
    hi = z_grid[i+1]
    
    mask = (z >= lo) & (z < hi)
    if mask.sum() < 15:
        l_drift.append(np.nan)
        b_drift.append(np.nan)
        continue

    l_fit, b_fit = fit_axis(ra[mask], dec[mask])
    l_drift.append(l_fit)
    b_drift.append(b_fit)

z_mid = 0.5 * (z_grid[:-1] + z_grid[1:])

# ---------------------------------------------------------
# 3. Linear regression: does l(z), b(z) drift?
# ---------------------------------------------------------
valid = ~np.isnan(l_drift)
zv = z_mid[valid].reshape(-1, 1)

lr_l = LinearRegression().fit(zv, np.array(l_drift)[valid])
lr_b = LinearRegression().fit(zv, np.array(b_drift)[valid])

l_slope = lr_l.coef_[0]
b_slope = lr_b.coef_[0]

print("-------------------------------------------------------")
print("axis–redshift drift:")
print(f"d(l) / dz = {l_slope:.3f} deg per unit z")
print(f"d(b) / dz = {b_slope:.3f} deg per unit z")
print("-------------------------------------------------------")

# ---------------------------------------------------------
# 4. Monte Carlo null: isotropic sky, same z distribution
# ---------------------------------------------------------
print("running 5000 Monte Carlo drift simulations...")

MC = 5000
sl_l = []
sl_b = []

for _ in range(MC):
    # randomize sky, keep z
    rand_l = np.random.uniform(0, 360, size=N)
    rand_b = np.degrees(np.arcsin(np.random.uniform(-1, 1, size=N)))
    rand_ra_dec = SkyCoord(l=rand_l*u.deg, b=rand_b*u.deg, frame="galactic").icrs
    
    rr = rand_ra_dec.ra.deg
    dd = rand_ra_dec.dec.deg
    
    l_drift_mock = []
    b_drift_mock = []

    for i in range(len(z_grid)-1):
        lo = z_grid[i]
        hi = z_grid[i+1]
        mask = (z >= lo) & (z < hi)
        if mask.sum() < 15:
            l_drift_mock.append(np.nan)
            b_drift_mock.append(np.nan)
            continue
        
        ll, bb = fit_axis(rr[mask], dd[mask])
        l_drift_mock.append(ll)
        b_drift_mock.append(bb)

    valid_mock = ~np.isnan(l_drift_mock)
    if valid_mock.sum() < 5:
        continue

    zv_mock = z_mid[valid_mock].reshape(-1, 1)

    sl_l.append(LinearRegression().fit(zv_mock, np.array(l_drift_mock)[valid_mock]).coef_[0])
    sl_b.append(LinearRegression().fit(zv_mock, np.array(b_drift_mock)[valid_mock]).coef_[0])

sl_l = np.array(sl_l)
sl_b = np.array(sl_b)

p_l = (np.abs(sl_l) >= np.abs(l_slope)).mean()
p_b = (np.abs(sl_b) >= np.abs(b_slope)).mean()

print("-------------------------------------------------------")
print(f"Monte Carlo p-value for drift in l: {p_l:.5f}")
print(f"Monte Carlo p-value for drift in b: {p_b:.5f}")
print("-------------------------------------------------------")

print("\nscientific interpretation:")
if p_l > 0.1 and p_b > 0.1:
    print("no detectable axis drift with redshift — axis is cosmologically stable.")
else:
    print("possible evolution of axis with redshift — indicates local structure or evolving anisotropy.")

# ---------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(z_mid, l_drift, "o-", label="l(z)")
plt.plot(z_mid, b_drift, "o-", label="b(z)")
plt.xlabel("redshift z")
plt.ylabel("axis coordinates (deg)")
plt.title("FRB Axis Position vs Redshift")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("frb_axis_redshift_drift.png")

print("\nsaved plot: frb_axis_redshift_drift.png")
print("analysis complete.")
print("=======================================================\n")
