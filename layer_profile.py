"""
layer_profile.py
------------------------------------------------------------
construct a unified layer profile showing angular structure
around the unified axis, including:

  - frb density ratio
  - shell scan boundaries
  - cone-fit radii
  - cmb axis offset
  - frb dipole offset
  - atomic clock offset

produces: layer_profile.png
------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

# ------------------------------------------------------------
# unified axis
# ------------------------------------------------------------
AXIS_L = 159.85
AXIS_B = -0.51
axis = SkyCoord(l=AXIS_L*u.deg, b=AXIS_B*u.deg, frame="galactic")

print("="*70)
print("layer profile construction")
print("assembling combined angular structure")
print("="*70)

# ------------------------------------------------------------
# messenger offsets (from your results)
# ------------------------------------------------------------
cmb_offset     = 8.71     # deg  (cmb → unified axis)
frb_offset     = 0.80     # deg  (frb sidereal → unified axis)
clock_offset   = 5.03     # deg  (atomic clock → unified axis)

# ------------------------------------------------------------
# shell boundaries from frb_axis_shell_scan.py
# ------------------------------------------------------------
shell_edges = np.array([1.5, 5.5, 14.5, 27.5, 31.5])

# ------------------------------------------------------------
# best-fit cone radii from cone_fit.py
# (triple-cone best model)
# ------------------------------------------------------------
cone_r1 = 8.10
cone_r2 = 21.77
cone_r3 = 40.00

# ------------------------------------------------------------
# load frbs
# ------------------------------------------------------------
try:
    frbs = pd.read_csv("frbs.csv")
except FileNotFoundError:
    print("frbs.csv not found")
    exit()

coords = SkyCoord(ra=frbs["ra"].values*u.deg,
                  dec=frbs["dec"].values*u.deg,
                  frame="icrs").galactic

sep = coords.separation(axis).deg

print("\n1. frb catalog")
print("------------------------------------------------------------")
print(f"total frbs loaded: {len(frbs)}")
print("computed angular distances")

# ------------------------------------------------------------
# frb density ratio vs isotropic
# ------------------------------------------------------------
bin_width = 1.0
bins = np.arange(0, 90 + bin_width, bin_width)
centers = bins[:-1] + bin_width/2

counts = []
iso = []

total = len(frbs)
for i in range(len(bins)-1):
    low, high = bins[i], bins[i+1]
    in_shell = frbs[(sep >= low) & (sep < high)]
    counts.append(len(in_shell))

    t1 = np.radians(low)
    t2 = np.radians(high)
    p = (np.cos(t1) - np.cos(t2)) / 2
    iso.append(total * p)

counts = np.array(counts)
iso = np.array(iso)
density = counts / iso

print("\n2. frb density profile")
print("------------------------------------------------------------")
print("computed density ratio relative to isotropic expectation")

# ------------------------------------------------------------
# plot assembly
# ------------------------------------------------------------
plt.figure(figsize=(14,8))

plt.plot(centers, density, color='black', linewidth=1.8,
         label='frb density ratio')

# ------------------------------------------------------------
# overlay shell-scan boundaries
# ------------------------------------------------------------
for s in shell_edges:
    plt.axvline(s, color='red', linestyle='--', alpha=0.5)
plt.text(shell_edges[-1] + 1, 1.8, "shell boundaries", color='red',
         fontsize=9)

# ------------------------------------------------------------
# overlay cone radii
# ------------------------------------------------------------
for R, col in zip([cone_r1, cone_r2, cone_r3],
                  ['blue', 'blue', 'blue']):
    plt.axvline(R, color=col, linestyle='-', alpha=0.6)
plt.text(cone_r2 + 1, 1.65, "cone-fit radii", color='blue', fontsize=9)

# ------------------------------------------------------------
# overlay messenger offsets
# ------------------------------------------------------------
plt.axvline(frb_offset,   color='green', linestyle='-', alpha=0.8)
plt.axvline(clock_offset, color='green', linestyle='-', alpha=0.8)
plt.axvline(cmb_offset,   color='green', linestyle='-', alpha=0.8)

plt.text(cmb_offset + 1,   1.5, "cmb axis", color='green', fontsize=9)
plt.text(clock_offset + 1, 1.4, "clock axis", color='green', fontsize=9)
plt.text(frb_offset + 1,   1.3, "frb sidereal axis", color='green', fontsize=9)

# ------------------------------------------------------------
# plot formatting
# ------------------------------------------------------------
plt.title("layered angular structure around unified axis", fontsize=13)
plt.xlabel("angle from axis (deg)")
plt.ylabel("density ratio (observed / isotropic)")
plt.grid(alpha=0.3)
plt.ylim(0, 2)

plt.tight_layout()
plt.savefig("layer_profile.png", dpi=200, bbox_inches='tight')

print("\n3. figure output")
print("------------------------------------------------------------")
print("saved: layer_profile.png")

print("\n" + "="*70)
print("analysis complete")
print("="*70)
