import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import median_test

print("====================================================================")
print("FRB DM SKY CORRELATION TEST")
print("testing whether dispersion measure correlates with sky position")
print("====================================================================\n")

# ============================================================
# 1. load data
# ============================================================
try:
    frbs = pd.read_csv("frbs.csv")
except FileNotFoundError:
    print("[fatal] frbs.csv not found — cannot run DM sky test.")
    raise SystemExit

if "dm" not in frbs.columns:
    print("[fatal] 'dm' column not found in frbs.csv — DM sky test not possible.")
    raise SystemExit

N = len(frbs)

coords = SkyCoord(
    ra=frbs["ra"].values * u.deg,
    dec=frbs["dec"].values * u.deg,
    frame="icrs"
).galactic

l = coords.l.deg
b = coords.b.deg

print("1 — dataset")
print("------------------------------------------------------------")
print(f"loaded {N} FRBs")
print("converted to galactic coordinates\n")

# ============================================================
# 2. dispersion measure parameter
# ============================================================
DM = frbs["dm"].values.astype(float)

print("2 — parameter")
print("------------------------------------------------------------")
print("parameter: dispersion measure (DM)")
print(f"range: {DM.min():.2f} → {DM.max():.2f}\n")

# ============================================================
# 3. method A: hemispherical asymmetry around unified axis
# ============================================================
print("3 — method A: hemispherical asymmetry")
print("------------------------------------------------------------")

# unified axis used in previous tests
axis_l, axis_b = 159.85, -0.51
axis = SkyCoord(l=axis_l*u.deg, b=axis_b*u.deg, frame="galactic")

theta = coords.separation(axis).deg
mask_near = theta < 90.0
mask_far  = ~mask_near

DM_near = DM[mask_near]
DM_far  = DM[mask_far]

stat, p_med, _, _ = median_test(DM_near, DM_far)

print(f"median DM (near hemisphere) = {np.median(DM_near):.2f}")
print(f"median DM (far  hemisphere) = {np.median(DM_far):.2f}")
print(f"p-value (median test)       = {p_med:.4f}\n")

# ============================================================
# 4. method B: dipole fitting
# ============================================================
print("4 — method B: dipole fitting")
print("------------------------------------------------------------")

# unit vectors
x = np.cos(np.radians(b)) * np.cos(np.radians(l))
y = np.cos(np.radians(b)) * np.sin(np.radians(l))
z = np.sin(np.radians(b))

A = np.vstack([np.ones(N), x, y, z]).T
coeff, _, _, _ = np.linalg.lstsq(A, DM, rcond=None)
a, bx, by, bz = coeff

dip_amp = np.sqrt(bx**2 + by**2 + bz**2)

if dip_amp > 0:
    l_dip = np.degrees(np.arctan2(by, bx)) % 360
    b_dip = np.degrees(np.arcsin(bz / dip_amp))
    dip_dir = SkyCoord(l=l_dip*u.deg, b=b_dip*u.deg, frame="galactic")
    print(f"dipole amplitude: {dip_amp:.4f}")
    print(f"dipole direction: l={dip_dir.l.deg:.1f}°, b={dip_dir.b.deg:.1f}°\n")
else:
    print("dipole amplitude: 0 (no directional trend)\n")

# ============================================================
# 5. method C: DM sky map
# ============================================================
print("5 — method C: local DM sky map")
print("------------------------------------------------------------")

lon_bins = np.arange(-180, 181, 5)
lat_bins = np.arange(-90, 91, 5)

l_wrap = ((l + 180) % 360) - 180

heat = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
count = np.zeros_like(heat)

for lon, lat, dm_val in zip(l_wrap, b, DM):
    ix = np.searchsorted(lon_bins, lon) - 1
    iy = np.searchsorted(lat_bins, lat) - 1
    if 0 <= ix < heat.shape[1] and 0 <= iy < heat.shape[0]:
        heat[iy, ix] += dm_val
        count[iy, ix] += 1

avg = np.divide(heat, count, out=np.zeros_like(heat), where=count>0)

lon_grid = np.radians(lon_bins[:-1] + 2.5)
lat_grid = np.radians(lat_bins[:-1] + 2.5)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

plt.figure(figsize=(12,6))
ax = plt.subplot(111, projection="mollweide")
img = ax.pcolormesh(lon_mesh, lat_mesh, avg, cmap="viridis", shading="auto")
plt.colorbar(img, label="DM (average)")
ax.grid(alpha=0.3)
plt.title("FRB DM Sky Map (Local Averages)", fontsize=14)
plt.savefig("dm_sky_map.png", dpi=200, bbox_inches="tight")
plt.close()

print("✓ saved map: dm_sky_map.png\n")

# ============================================================
# 6. permutation significance for dipole amplitude
# ============================================================
print("6 — permutation significance (dipole)")
print("------------------------------------------------------------")

def dipole_amplitude_for(values):
    coeff, *_ = np.linalg.lstsq(A, values, rcond=None)
    _, bx_m, by_m, bz_m = coeff
    return np.sqrt(bx_m**2 + by_m**2 + bz_m**2)

real_amp = dip_amp
amps = []

n_perm = 2000
for _ in range(n_perm):
    DM_shuffled = np.random.permutation(DM)
    amps.append(dipole_amplitude_for(DM_shuffled))

amps = np.array(amps)
p_perm = np.mean(amps >= real_amp)

print(f"real dipole amplitude   = {real_amp:.4f}")
print(f"mean amp (permutations) = {amps.mean():.4f}")
print(f"median amp (permutations)= {np.median(amps):.4f}")
print(f"p-value (perm)          = {p_perm:.4f}\n")

# ============================================================
# 7. final verdict
# ============================================================
print("====================================================================")
print("FINAL VERDICT")
print("====================================================================")

if p_perm < 0.05 or p_med < 0.05:
    print("→ evidence for a DM–sky correlation (anisotropy) at ≲5% level.")
    if p_perm < 0.05 and p_med < 0.05:
        print("  both hemisphere and dipole tests indicate a weak but")
        print("  statistically non-random directional pattern in DM.")
else:
    print("→ no statistically significant correlation between DM and sky")
    print("  position. DM distribution is consistent with isotropy.")

print("====================================================================")
