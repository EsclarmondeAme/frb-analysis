import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit

print("====================================================================")
print("RESIDUAL SKY MAP (GRID VERSION — NO HEALPIX)")
print("footprint-corrected FRB sky using simple spherical binning")
print("====================================================================\n")

# -------------------------------------------------------------------
# 1. load dataset
# -------------------------------------------------------------------
frbs = pd.read_csv("frbs.csv")
N_raw = len(frbs)

coords_icrs = SkyCoord(
    ra=frbs["ra"].values * u.deg,
    dec=frbs["dec"].values * u.deg,
    frame="icrs"
)

coords_gal = coords_icrs.galactic
l = coords_gal.l.deg
b = coords_gal.b.deg

print("1 — dataset")
print("------------------------------------------------------------")
print(f"total FRBs loaded: {N_raw}\n")

# -------------------------------------------------------------------
# 2. define axes
# -------------------------------------------------------------------
l_fp, b_fp = 129.00, 23.00
axis_fp = SkyCoord(l=l_fp*u.deg, b=b_fp*u.deg, frame="galactic")

l_unif, b_unif = 159.85, -0.51
axis_unif = SkyCoord(l=l_unif*u.deg, b=b_unif*u.deg, frame="galactic")

theta_instr = coords_gal.separation(axis_fp).deg
theta_unif = coords_gal.separation(axis_unif).deg

print("2 — angles")
print("------------------------------------------------------------")
print(f"min θ_instr = {theta_instr.min():.2f}°, max = {theta_instr.max():.2f}°")
print(f"min θ_unif  = {theta_unif.min():.2f}°, max = {theta_unif.max():.2f}°\n")

# -------------------------------------------------------------------
# 3. restrict to instrument-valid region
# -------------------------------------------------------------------
mask = theta_instr <= 80
l = l[mask]
b = b[mask]
theta_instr = theta_instr[mask]
N_cut = len(l)

print("3 — footprint cut")
print("------------------------------------------------------------")
print(f"θ_instr <= 80° → retained {N_cut} FRBs\n")

# -------------------------------------------------------------------
# 4. footprint model
# -------------------------------------------------------------------
def poly4(x, a,b,c,d,e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4

theta_norm = theta_instr / 80.0
popt, _ = curve_fit(poly4, theta_norm, np.ones_like(theta_norm))
f_model = poly4(theta_norm, *popt)

print("4 — footprint model")
print("------------------------------------------------------------")
print(f"model range: {f_model.min():.3f} – {f_model.max():.3f}\n")

# -------------------------------------------------------------------
# 5. residual weights
# -------------------------------------------------------------------
weights = 1.0 / f_model

print("5 — residual weights")
print("------------------------------------------------------------")
print(f"min weight = {weights.min():.3f}")
print(f"max weight = {weights.max():.3f}")
print(f"mean weight= {weights.mean():.3f}")
print(f"effective sample size = {weights.sum():.2f}\n")

# -------------------------------------------------------------------
# 6. spherical grid (simple binning)
# -------------------------------------------------------------------
# grid resolution: 1° × 1°
lon_bins = np.arange(-180, 181, 1)
lat_bins = np.arange(-90,  91,  1)

# wrap l into [-180, 180]
l_wrapped = ((l + 180) % 360) - 180

hist = np.zeros((len(lat_bins)-1, len(lon_bins)-1))

# fill weighted map
for lon, lat, w in zip(l_wrapped, b, weights):
    # find indices
    ix = np.searchsorted(lon_bins, lon) - 1
    iy = np.searchsorted(lat_bins, lat) - 1
    if 0 <= ix < hist.shape[1] and 0 <= iy < hist.shape[0]:
        hist[iy, ix] += w

print("6 — grid map")
print("------------------------------------------------------------")
print("constructed 360×180 residual density grid\n")

# -------------------------------------------------------------------
# 7. plotting (mollweide)
# -------------------------------------------------------------------
print("7 — generating plot")

# convert grid to radians for mollweide projection
lon_grid = np.radians(lon_bins[:-1] + 0.5)
lat_grid = np.radians(lat_bins[:-1] + 0.5)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

plt.figure(figsize=(12,6))
ax = plt.subplot(111, projection='mollweide')

img = ax.pcolormesh(lon_mesh, lat_mesh, hist, cmap='inferno', shading='auto')
plt.colorbar(img, label='weighted residual intensity')
ax.grid(alpha=0.3)
plt.title("Residual FRB Sky (Footprint-Corrected)", fontsize=14)

plt.savefig("residual_sky_map.png", dpi=200, bbox_inches="tight")
plt.close()

print("✓ saved: residual_sky_map.png\n")

print("====================================================================")
print("ANALYSIS COMPLETE")
print("====================================================================")
