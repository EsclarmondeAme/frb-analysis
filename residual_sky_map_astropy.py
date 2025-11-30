import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy_healpix import HEALPix
from astropy_healpix import healpix_to_skycoord
from scipy.optimize import curve_fit

print("====================================================================")
print("RESIDUAL SKY MAP (ASTROPY HEALPIX VERSION)")
print("footprint-corrected FRB residual sky")
print("====================================================================\n")

# -------------------------------------------------------------------
# 1. load data
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
print(f"total FRBs loaded: {N_raw}")
print("converted to galactic coords\n")

# -------------------------------------------------------------------
# 2. define axes
# -------------------------------------------------------------------
l_fp, b_fp = 129.00, 23.00
axis_fp = SkyCoord(l=l_fp * u.deg, b=b_fp * u.deg, frame="galactic")

l_unif, b_unif = 159.85, -0.51
axis_unif = SkyCoord(l=l_unif * u.deg, b=b_unif * u.deg, frame="galactic")

theta_instr = coords_gal.separation(axis_fp).deg
theta_unif = coords_gal.separation(axis_unif).deg

print("2 — angles")
print("------------------------------------------------------------")
print(f"min θ_instr = {theta_instr.min():.2f}°, max = {theta_instr.max():.2f}°")
print(f"min θ_unif  = {theta_unif.min():.2f}°, max = {theta_unif.max():.2f}°\n")

# -------------------------------------------------------------------
# 3. restrict to footprint-valid region
# -------------------------------------------------------------------
mask = theta_instr <= 80.0
l = l[mask]
b = b[mask]
theta_instr = theta_instr[mask]
N_cut = len(l)

print("3 — footprint restriction")
print("------------------------------------------------------------")
print(f"θ_instr <= 80° cut → retained {N_cut} FRBs\n")

# -------------------------------------------------------------------
# 4. fit footprint model f(θ_instr)
# -------------------------------------------------------------------
def poly4(x, a,b,c,d,e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4

theta_norm = theta_instr / 80.0
popt, _ = curve_fit(poly4, theta_norm, np.ones_like(theta_norm))
f_model = poly4(theta_norm, *popt)

print("4 — footprint model")
print("------------------------------------------------------------")
print(f"model range: min={f_model.min():.3f}, max={f_model.max():.3f}\n")

# -------------------------------------------------------------------
# 5. residual weights
# -------------------------------------------------------------------
weights = 1.0 / f_model
print("5 — residual weighting")
print("------------------------------------------------------------")
print(f"min weight = {weights.min():.3f}")
print(f"max weight = {weights.max():.3f}")
print(f"mean weight= {weights.mean():.3f}")
print(f"effective sample size = {weights.sum():.2f}\n")

# -------------------------------------------------------------------
# 6. build ASTROPY HEALPIX map
# -------------------------------------------------------------------
NSIDE = 32
hp = HEALPix(nside=NSIDE, order='ring', frame='galactic')
npix = hp.npix

hp_map = np.zeros(npix, dtype=float)

# convert l,b to healpix pixel numbers
pix = hp.skycoord_to_healpix(SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic'))

for p, w in zip(pix, weights):
    hp_map[p] += w

print("6 — healpix map (astropy)")
print("------------------------------------------------------------")
print(f"NSIDE={NSIDE}, pixels={npix}")
print("map built successfully\n")

# -------------------------------------------------------------------
# 7. save map
# -------------------------------------------------------------------
np.save("residual_sky_map_hp.npy", hp_map)
print("7 — saved healpix map to: residual_sky_map_hp.npy\n")

# -------------------------------------------------------------------
# 8. plot Mollweide manually (no healpy needed)
# -------------------------------------------------------------------
# convert each pixel center to coords
ipix = np.arange(npix)
sky = healpix_to_skycoord(ipix, NSIDE, order='ring', frame='galactic')

lon = sky.l.wrap_at(180*u.deg).radian
lat = sky.b.radian

plt.figure(figsize=(12,6))
plt.subplot(111, projection="mollweide")
plt.scatter(lon, lat, c=hp_map, s=5, cmap="inferno")
plt.colorbar(label="weighted residual intensity")
plt.grid(True, alpha=0.3)
plt.title("Residual FRB Sky Map (Footprint-Corrected)", fontsize=14)

plt.savefig("residual_sky_map.png", dpi=200, bbox_inches="tight")
plt.close()

print("8 — saved: residual_sky_map.png\n")

print("====================================================================")
print("ANALYSIS COMPLETE")
print("====================================================================")
