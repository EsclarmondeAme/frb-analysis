"""
frb_axis_randomization_test.py
------------------------------------------------------------
robustness test: is the unified axis special?

we:
  - take the same three bands as in frb_shell_significance.py:
      inner :  0°–10°
      middle: 10°–25°
      outer : 25°–40°
  - compute the band chi² for the real unified axis
  - generate many random axes on the sky
  - compute the same band chi² for each random axis
  - measure how often random axes look as 'layered' as the real one

output:
  - chi² for unified axis
  - distribution of chi² for random axes
  - p-value: fraction of random axes with chi² >= chi²_unified
------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# ------------------------------------------------------------
# helper: convert (l,b) to cartesian unit vector
# ------------------------------------------------------------
def lb_to_vec(l_deg, b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    x = np.cos(b) * np.cos(l)
    y = np.cos(b) * np.sin(l)
    z = np.sin(b)
    return np.array([x, y, z])

# ------------------------------------------------------------
# unified axis (from your best-fit)
# ------------------------------------------------------------
AXIS_L = 159.85
AXIS_B = -0.51
axis_vec = lb_to_vec(AXIS_L, AXIS_B)

print("="*70)
print("frb axis randomization test")
print("is the unified axis special, or typical among random axes?")
print("="*70)

# ------------------------------------------------------------
# load frb catalog
# ------------------------------------------------------------
try:
    frbs = pd.read_csv("frbs.csv")
except FileNotFoundError:
    print("\n[error] frbs.csv not found")
    raise SystemExit

print("\n1. frb catalog")
print("------------------------------------------------------------")
print(f"total frbs: {len(frbs)}")

# convert to galactic and then to unit vectors
coords = SkyCoord(ra=frbs["ra"].values*u.deg,
                  dec=frbs["dec"].values*u.deg,
                  frame="icrs").galactic

l = coords.l.deg
b = coords.b.deg

frb_vecs = lb_to_vec(l, b)    # shape (3, N)

# ------------------------------------------------------------
# band definitions (same as shell significance test)
# ------------------------------------------------------------
bands = [
    ("inner",  0.0, 10.0),
    ("middle", 10.0, 25.0),
    ("outer",  25.0, 40.0),
]

N = frb_vecs.shape[1]

def isotropic_band_prob(theta_low_deg, theta_high_deg):
    t1 = np.radians(theta_low_deg)
    t2 = np.radians(theta_high_deg)
    return (np.cos(t1) - np.cos(t2)) / 2.0

# expected counts under isotropy (independent of axis orientation)
exp_counts = np.array([N * isotropic_band_prob(low, high)
                       for (_, low, high) in bands])

# ------------------------------------------------------------
# chi² computation for a given axis
# ------------------------------------------------------------
def chi2_for_axis(axis_vec, frb_vecs, bands, exp_counts):
    # cos(theta) = v_axis · v_frb
    dots = axis_vec[0]*frb_vecs[0] + axis_vec[1]*frb_vecs[1] + axis_vec[2]*frb_vecs[2]
    # clamp numerical drift
    dots = np.clip(dots, -1.0, 1.0)
    theta = np.degrees(np.arccos(dots))  # shape (N,)

    obs_counts = []
    for (_, low, high) in bands:
        in_band = (theta >= low) & (theta < high)
        obs_counts.append(np.sum(in_band))
    obs_counts = np.array(obs_counts, dtype=float)

    chi2 = np.sum((obs_counts - exp_counts)**2 / exp_counts)
    return chi2, obs_counts

# ------------------------------------------------------------
# chi² for unified axis
# ------------------------------------------------------------
chi2_unified, obs_unified = chi2_for_axis(axis_vec, frb_vecs, bands, exp_counts)

print("\n2. unified axis band counts and chi²")
print("------------------------------------------------------------")
for (name, low, high), n_obs, mu in zip(bands, obs_unified, exp_counts):
    ratio = n_obs / mu if mu > 0 else np.nan
    print(f"{name:6s} band {low:4.1f}°–{high:4.1f}°:")
    print(f"   observed: {n_obs:5.1f}")
    print(f"   expected: {mu:7.2f}")
    print(f"   ratio   : {ratio:7.2f}\n")

print(f"chi² (unified axis) = {chi2_unified:.2f}")

# ------------------------------------------------------------
# random axes simulation
# ------------------------------------------------------------
print("\n3. random axis ensemble")
print("------------------------------------------------------------")

N_AXES = 10000
print(f"generating {N_AXES} random axes and computing chi² for each...")

# random isotropic unit vectors for axes
u = np.random.uniform(-1.0, 1.0, size=N_AXES)
phi = np.random.uniform(0.0, 2*np.pi, size=N_AXES)
# convert to (l,b)-like angles for axis
b_axis = np.degrees(np.arcsin(u))
l_axis = np.degrees(phi)      # 0–360

chi2_random = np.zeros(N_AXES)

for i in range(N_AXES):
    v_axis = lb_to_vec(l_axis[i], b_axis[i])
    chi2_random[i], _ = chi2_for_axis(v_axis, frb_vecs, bands, exp_counts)

p_value = np.mean(chi2_random >= chi2_unified)

print("\n4. comparison")
print("------------------------------------------------------------")
print(f"mean chi²(random axes)   = {chi2_random.mean():.2f}")
print(f"median chi²(random axes) = {np.median(chi2_random):.2f}")
print(f"95th percentile          = {np.percentile(chi2_random, 95):.2f}")
print(f"max chi²(random)         = {chi2_random.max():.2f}")
print(f"\nchi²(unified axis)       = {chi2_unified:.2f}")
print(f"fraction of random axes with chi² >= chi²_unified: {p_value:.4f}")

print("\n5. verdict")
print("------------------------------------------------------------")
if p_value < 0.001:
    print("unified axis is highly special compared to random orientations")
elif p_value < 0.01:
    print("unified axis is significantly more structured than random")
elif p_value < 0.05:
    print("unified axis is mildly more structured than random")
else:
    print("unified axis is not unusual compared to random axes")

print("\n" + "="*70)
print("analysis complete")
print("="*70)
