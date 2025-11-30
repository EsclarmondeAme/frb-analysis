import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.stats import median_test
from scipy.stats import linregress

print("====================================================================")
print("FRB PARAMETER SKY CORRELATION TEST")
print("testing whether physical FRB parameters correlate with sky position")
print("====================================================================\n")

# ============================================================
# 1. load data
# ============================================================
frbs = pd.read_csv("frbs.csv")
N = len(frbs)

coords = SkyCoord(
    ra = frbs["ra"].values * u.deg,
    dec = frbs["dec"].values * u.deg,
    frame="icrs"
).galactic

l = coords.l.deg
b = coords.b.deg

print("1 — dataset")
print("------------------------------------------------------------")
print(f"loaded {N} FRBs")
print("converted to galactic coords\n")

# ============================================================
# 2. choose parameter to test
# ============================================================
# pick automatically: fluence if present, else dm, else width
if "fluence" in frbs.columns:
    param_name = "fluence"
elif "dm" in frbs.columns:
    param_name = "dm"
elif "width" in frbs.columns:
    param_name = "width"
else:
    raise ValueError("no suitable physical parameter found")

P = frbs[param_name].values

print("2 — parameter")
print("------------------------------------------------------------")
print(f"parameter selected: {param_name}")
print(f"range: {P.min():.3f} → {P.max():.3f}\n")

# ============================================================
# 3. method A: hemispherical asymmetry
# ============================================================
print("3 — method A: hemispherical asymmetry")
print("------------------------------------------------------------")

axis_l, axis_b = 160, 0  # use unified axis direction for test
axis = SkyCoord(l=axis_l*u.deg, b=axis_b*u.deg, frame="galactic")

# project FRBs onto axis
theta = coords.separation(axis).deg
mask_north = theta < 90
mask_south = ~mask_north

P_N = P[mask_north]
P_S = P[mask_south]

# median comparison test
stat, p_med, _, _ = median_test(P_N, P_S)

print(f"median north: {np.median(P_N):.3f}")
print(f"median south: {np.median(P_S):.3f}")
print(f"p-value (median test): {p_med:.4f}\n")

# ============================================================
# 4. method B: dipole fitting
# ============================================================
print("4 — method B: dipole fitting")
print("------------------------------------------------------------")

# convert to unit vectors
x = np.cos(np.radians(b)) * np.cos(np.radians(l))
y = np.cos(np.radians(b)) * np.sin(np.radians(l))
z = np.sin(np.radians(b))

# fit P = a + bx + cy + dz
A = np.vstack([np.ones(N), x, y, z]).T
coeff, _, _, _ = np.linalg.lstsq(A, P, rcond=None)
a, bx, by, bz = coeff

dip_amp = np.sqrt(bx**2 + by**2 + bz**2)
dip_dir = SkyCoord(
    l=np.degrees(np.arctan2(by, bx))*u.deg,
    b=np.degrees(np.arcsin(bz / dip_amp))*u.deg,
    frame='galactic'
)

print(f"dipole amplitude: {dip_amp:.4f}")
print(f"dipole direction: l={dip_dir.l.deg:.1f}°, b={dip_dir.b.deg:.1f}°\n")

# ============================================================
# 5. method C: sky map / local averages
# ============================================================
print("5 — method C: local smoothing map")
print("------------------------------------------------------------")

lon_bins = np.arange(-180, 181, 5)
lat_bins = np.arange(-90, 91, 5)

l_wrap = ((l + 180) % 360) - 180

heat = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
count = np.zeros_like(heat)

for lon, lat, val in zip(l_wrap, b, P):
    ix = np.searchsorted(lon_bins, lon) - 1
    iy = np.searchsorted(lat_bins, lat) - 1
    if 0 <= ix < heat.shape[1] and 0 <= iy < heat.shape[0]:
        heat[iy, ix] += val
        count[iy, ix] += 1

avg = np.divide(heat, count, out=np.zeros_like(heat), where=count>0)

plt.figure(figsize=(12,6))
ax = plt.subplot(111, projection="mollweide")

lon_grid = np.radians(lon_bins[:-1] + 2.5)
lat_grid = np.radians(lat_bins[:-1] + 2.5)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

img = ax.pcolormesh(lon_mesh, lat_mesh, avg, cmap='viridis', shading='auto')
plt.colorbar(img, label=f"{param_name} (avg)")
ax.grid(alpha=0.3)
plt.title(f"Sky Map of {param_name} (Local Averages)", fontsize=14)

plt.savefig("parameter_sky_map.png", dpi=200, bbox_inches="tight")
plt.close()

print("✓ saved map: parameter_sky_map.png\n")

# ============================================================
# 6. significance via permutation (2000 random relabels)
# ============================================================
print("6 — permutation significance test")
print("------------------------------------------------------------")

def dipole_amp_for(P_new):
    coeff, *_ = np.linalg.lstsq(A, P_new, rcond=None)
    _, bx, by, bz = coeff
    return np.sqrt(bx**2 + by**2 + bz**2)

real_amp = dip_amp

amps = []
for _ in range(2000):
    P_shuffled = np.random.permutation(P)
    amps.append(dipole_amp_for(P_shuffled))
amps = np.array(amps)

p_perm = np.mean(amps >= real_amp)

print(f"permutation p-value (dipole amplitude): {p_perm:.4f}\n")

# ============================================================
# 7. verdict
# ============================================================
print("====================================================================")
print("FINAL VERDICT")
print("====================================================================")

if p_perm < 0.05 or p_med < 0.05:
    print("→ a statistically significant sky–parameter correlation is detected")
else:
    print("→ no significant correlation between FRB parameter and sky position")

print("====================================================================")
