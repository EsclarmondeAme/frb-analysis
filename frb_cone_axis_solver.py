"""
frb_cone_axis_solver.py
------------------------------------------------------------
full-sky scan to find the axis that maximizes FRB layered
structure (inner/middle/outer band contrasts).

we compute for each candidate axis:
  - distances of all FRBs from that axis
  - observed counts in three bands:
        inner  : 0–10°
        middle : 10–25°
        outer  : 25–40°
  - expected counts under isotropy
  - chi²_layered = sum( (n_i - mu_i)^2 / mu_i )

we scan the full sky on a grid (default 1° spacing).
best axis = the one with maximum chi²_layered.

outputs:
  - best-fit axis (l,b)
  - chi² value
  - top 10 candidate axes
  - heatmap file: cone_axis_heatmap.png
------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# configuration
# ------------------------------------------------------------
GRID_STEP_DEG = 1     # 1° grid across whole sky (360 x 180 = 64,800 axes)

BANDS = [
    ("inner",  0.0, 10.0),
    ("middle", 10.0, 25.0),
    ("outer",  25.0, 40.0),
]

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def lb_to_vec(l_deg, b_deg):
    """convert (l,b) in degrees to cartesian unit vector"""
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)
    return np.array([x, y, z])

def sep_from_axis_deg(axis_vec, frb_vecs):
    """angular separation (deg) of each FRB from axis direction"""
    dots = axis_vec[0]*frb_vecs[0] + axis_vec[1]*frb_vecs[1] + axis_vec[2]*frb_vecs[2]
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))

def isotropic_expectation(N, low_deg, high_deg):
    """expected count in an angular band under isotropy"""
    t1 = np.radians(low_deg)
    t2 = np.radians(high_deg)
    p = (np.cos(t1) - np.cos(t2)) / 2.0
    return N * p

# ------------------------------------------------------------
# load FRB catalog
# ------------------------------------------------------------
print("="*70)
print("frb cone axis solver")
print("full-sky layered-structure maximization scan")
print("="*70)

try:
    frbs = pd.read_csv("frbs.csv")
except:
    print("\n[error] frbs.csv not found")
    raise SystemExit

print("\n1. frb catalog")
print("------------------------------------------------------------")
print(f"total frbs: {len(frbs)}")

coords = SkyCoord(ra=frbs["ra"].values*u.deg,
                  dec=frbs["dec"].values*u.deg,
                  frame="icrs").galactic

l = coords.l.deg
b = coords.b.deg

# convert all frbs to unit-vector array of shape (3, N)
frb_vecs = np.vstack(lb_to_vec(l, b))

N = frb_vecs.shape[1]

# compute isotropic expectations for each band
exp_counts = np.array([isotropic_expectation(N, low, high)
                       for (_, low, high) in BANDS])

print("\nexpected isotropic counts per band:")
for (name, low, high), mu in zip(BANDS, exp_counts):
    print(f"  {name:6s} {low:4.0f}-{high:4.0f}°  ->  {mu:.2f}")
print()

# ------------------------------------------------------------
# sky scan
# ------------------------------------------------------------
print("\n2. full-sky scan")
print("------------------------------------------------------------")
print(f"grid resolution: {GRID_STEP_DEG}°")

L_vals = np.arange(0, 360, GRID_STEP_DEG)
B_vals = np.arange(-90, 90+GRID_STEP_DEG, GRID_STEP_DEG)

chi2_map = np.zeros((len(B_vals), len(L_vals)))
best_chi2 = -1
best_axis = (0,0)
best_counts = None

for i, b0 in enumerate(B_vals):
    for j, l0 in enumerate(L_vals):
        axis_vec = lb_to_vec(l0, b0)

        # separations
        sep = sep_from_axis_deg(axis_vec, frb_vecs)

        # observed counts
        obs_counts = []
        for (_, low, high) in BANDS:
            obs_counts.append(np.sum((sep >= low) & (sep < high)))
        obs_counts = np.array(obs_counts, dtype=float)

        # chi²
        chi2 = np.sum((obs_counts - exp_counts)**2 / exp_counts)
        chi2_map[i, j] = chi2

        if chi2 > best_chi2:
            best_chi2 = chi2
            best_axis = (l0, b0)
            best_counts = obs_counts.copy()

print("\n3. best-fit axis")
print("------------------------------------------------------------")
print(f"best axis (galactic):")
print(f"   l = {best_axis[0]:.2f}°")
print(f"   b = {best_axis[1]:.2f}°")
print(f"max chi² = {best_chi2:.2f}")

print("\nbest-fit band counts vs expected:")
for (name, low, high), n_obs, mu in zip(BANDS, best_counts, exp_counts):
    ratio = n_obs / mu
    print(f"  {name:6s} {low:4.0f}-{high:4.0f}°:")
    print(f"     observed = {n_obs:7.1f}")
    print(f"     expected = {mu:7.2f}")
    print(f"     ratio    = {ratio:7.2f}\n")

# ------------------------------------------------------------
# produce heatmap
# ------------------------------------------------------------
print("\n4. generating chi² heatmap (cone_axis_heatmap.png)")
print("------------------------------------------------------------")

plt.figure(figsize=(12,6))
plt.imshow(chi2_map, origin='lower',
           extent=[0,360,-90,90],
           aspect='auto', cmap='plasma')
plt.colorbar(label="chi² layered")
plt.scatter([best_axis[0]], [best_axis[1]],
            s=120, c='white', edgecolors='black', label='best axis')
plt.xlabel("galactic longitude l (deg)")
plt.ylabel("galactic latitude b (deg)")
plt.title("FRB layered-structure chi² across sky")
plt.legend()
plt.savefig("cone_axis_heatmap.png", dpi=200, bbox_inches='tight')

print("\n✓ saved: cone_axis_heatmap.png")

print("\n" + "="*70)
print("analysis complete")
print("="*70)
