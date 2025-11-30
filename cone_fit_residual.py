"""
cone_fit_residual.py
------------------------------------------------------------
fit nested cone layers to the residual-corrected FRB angular
profile around the unified cosmic axis.

steps:
  1. reconstruct footprint-corrected weights (same as
     frb_residual_cone_test2.py) within theta_instr <= 80°.
  2. build a 1-degree radial profile in theta_unif (0–40°).
  3. compute density ratio = observed(weighted) / isotropic(weighted).
  4. fit single-, double-, and triple-cone models to the
     density-ratio curve using least squares.
  5. report best-fit cone radii and goodness-of-fit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

# ------------------------------------------------------------
# configuration
# ------------------------------------------------------------
FOOT_L = 129.0
FOOT_B = 23.0          # instrument footprint axis

UNIF_L = 159.85
UNIF_B = -0.51         # unified cosmic axis

THETA_MAX_FOOT = 80.0  # validity range for footprint model
BIN_WIDTH = 1.0        # radial profiling resolution (deg)
PROFILE_MAX = 40.0     # fit cones up to 40 deg from unified axis

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def band_prob(theta_low_deg, theta_high_deg):
    t1 = np.radians(theta_low_deg)
    t2 = np.radians(theta_high_deg)
    return (np.cos(t1) - np.cos(t2)) / 2.0

def single_cone(theta, R):
    """one cone centered at radius R (simple absolute distance model)."""
    return np.abs(theta - R)

def double_cone(theta, R1, R2):
    """two nested cones."""
    return np.minimum(np.abs(theta - R1), np.abs(theta - R2))

def triple_cone(theta, R1, R2, R3):
    """three nested cones."""
    m = np.minimum(np.abs(theta - R1), np.abs(theta - R2))
    m = np.minimum(m, np.abs(theta - R3))
    return m

# ------------------------------------------------------------
# load catalog and set up axes
# ------------------------------------------------------------
print("="*70)
print("residual cone fit around unified axis")
print("fitting nested layers after footprint correction")
print("="*70)

try:
    frbs = pd.read_csv("frbs.csv")
except FileNotFoundError:
    print("\n[fatal] frbs.csv not found. analysis aborted.")
    raise SystemExit

coords_icrs = SkyCoord(ra=frbs["ra"].values*u.deg,
                       dec=frbs["dec"].values*u.deg,
                       frame="icrs").galactic

foot_axis = SkyCoord(l=FOOT_L*u.deg, b=FOOT_B*u.deg, frame="galactic")
unif_axis = SkyCoord(l=UNIF_L*u.deg, b=UNIF_B*u.deg, frame="galactic")

theta_instr = coords_icrs.separation(foot_axis).deg
theta_unif = coords_icrs.separation(unif_axis).deg

N_all = len(frbs)

print("\n1. dataset and angles")
print("------------------------------------------------------------")
print(f"total frbs loaded: {N_all}")
print(f"footprint axis: l={FOOT_L:.2f}°, b={FOOT_B:.2f}°")
print(f"unified axis : l={UNIF_L:.2f}°, b={UNIF_B:.2f}°")
print(f"min theta_instr = {theta_instr.min():.2f}°, max = {theta_instr.max():.2f}°")
print(f"min theta_unif  = {theta_unif.min():.2f}°, max = {theta_unif.max():.2f}°")

# restrict to region with meaningful exposure
mask_valid = theta_instr <= THETA_MAX_FOOT
frbs = frbs.loc[mask_valid].copy()
theta_instr = theta_instr[mask_valid]
theta_unif = theta_unif[mask_valid]
M = len(frbs)

print(f"\nrestricting to theta_instr <= {THETA_MAX_FOOT:.1f}°")
print(f"frbs retained after cut: {M} (out of {N_all})")

# ------------------------------------------------------------
# footprint modelling (same logic as residual_cone_test2)
# ------------------------------------------------------------
print("\n2. footprint model reconstruction")
print("------------------------------------------------------------")

bins = np.arange(0, THETA_MAX_FOOT + 5, 5)
counts, edges = np.histogram(theta_instr, bins=bins)
centers = 0.5*(edges[1:] + edges[:-1])

shape = counts / counts.max()
poly = np.poly1d(np.polyfit(centers, shape, deg=4))
smooth = np.clip(poly(centers), 1e-6, None)

print("fitted 4th-degree polynomial footprint model")
print(f"smooth model range (within {THETA_MAX_FOOT:.0f}°): "
      f"min={smooth.min():.3f}, max={smooth.max():.3f}")

def footprint_value(theta):
    return np.clip(poly(theta), 1e-6, None)

frbs["weight"] = 1.0 / footprint_value(theta_instr)
total_weight = frbs["weight"].sum()

print("\n3. residual weighting")
print("------------------------------------------------------------")
print("weights correct for footprint; effective sample size = total_weight")
print(f"mean weight = {frbs['weight'].mean():.3f}")
print(f"min  weight = {frbs['weight'].min():.3f}")
print(f"max  weight = {frbs['weight'].max():.3f}")
print(f"total_weight (sum w_i) = {total_weight:.2f}")

# ------------------------------------------------------------
# build radial density profile around unified axis
# ------------------------------------------------------------
print("\n4. radial profile around unified axis (residual)")
print("------------------------------------------------------------")

bins_r = np.arange(0, PROFILE_MAX + BIN_WIDTH, BIN_WIDTH)
centers_r = bins_r[:-1] + BIN_WIDTH/2

obs_profile = []
exp_profile = []
density = []

for low, high in zip(bins_r[:-1], bins_r[1:]):
    band_mask = (theta_unif >= low) & (theta_unif < high)
    w_sum = frbs.loc[band_mask, "weight"].sum()

    p_band = band_prob(low, high)
    mu_band = total_weight * p_band

    obs_profile.append(w_sum)
    exp_profile.append(mu_band)

obs_profile = np.array(obs_profile)
exp_profile = np.array(exp_profile)
density = obs_profile / exp_profile

print(f"profile range: 0–{PROFILE_MAX:.0f}° in {BIN_WIDTH:.1f}° bins")
print(f"non-zero bins: {(exp_profile > 0).sum()}")

# ------------------------------------------------------------
# cone model fits
# ------------------------------------------------------------
print("\n5. cone model fitting (residual profile)")
print("------------------------------------------------------------")

theta = centers_r
y = density

# we can weight by 1/variance ~ exp_profile / obs_profile, but
# keep it simple: equal weights, since deviations are large anyway

# single-cone fit
Rs = np.linspace(0, PROFILE_MAX, 401)
chi1 = []
for R in Rs:
    model = single_cone(theta, R)
    # rescale amplitude to best match y in least squares
    A = np.sum(y * model) / np.sum(model**2)
    fit = A * model
    chi1.append(np.sum((y - fit)**2))
chi1 = np.array(chi1)
best_R1 = Rs[np.argmin(chi1)]
best_chi1 = chi1.min()

print(f"single-cone best radius: R1 = {best_R1:.2f}°")
print(f"single-cone residual sum of squares: {best_chi1:.2f}")

# double-cone fit
Rvals = np.linspace(0, PROFILE_MAX, 81)
best_R2 = None
best_chi2 = np.inf

for R1 in Rvals:
    for R2 in Rvals:
        model = double_cone(theta, R1, R2)
        A = np.sum(y * model) / np.sum(model**2)
        fit = A * model
        chi = np.sum((y - fit)**2)
        if chi < best_chi2:
            best_chi2 = chi
            best_R2 = (R1, R2)

print(f"\ndouble-cone best radii: R1 = {best_R2[0]:.2f}°, R2 = {best_R2[1]:.2f}°")
print(f"double-cone residual sum of squares: {best_chi2:.2f}")

# triple-cone fit
Rvals3 = np.linspace(0, PROFILE_MAX, 41)
best_R3 = None
best_chi3 = np.inf

for R1 in Rvals3:
    for R2 in Rvals3:
        for R3 in Rvals3:
            model = triple_cone(theta, R1, R2, R3)
            A = np.sum(y * model) / np.sum(model**2)
            fit = A * model
            chi = np.sum((y - fit)**2)
            if chi < best_chi3:
                best_chi3 = chi
                best_R3 = (R1, R2, R3)

print(f"\ntriple-cone best radii: "
      f"R1 = {best_R3[0]:.2f}°, R2 = {best_R3[1]:.2f}°, R3 = {best_R3[2]:.2f}°")
print(f"triple-cone residual sum of squares: {best_chi3:.2f}")

# simple model comparison
print("\n6. model comparison (smaller is better)")
print("------------------------------------------------------------")
print(f"single  cone rss = {best_chi1:.2f}")
print(f"double  cone rss = {best_chi2:.2f}")
print(f"triple  cone rss = {best_chi3:.2f}")

rss_values = [best_chi1, best_chi2, best_chi3]
labels = ["single", "double", "triple"]
best_idx = int(np.argmin(rss_values))
print(f"\npreferred model (by rss): {labels[best_idx]} cone")

# ------------------------------------------------------------
# plot for visual inspection
# ------------------------------------------------------------
print("\n7. generating profile figure (cone_fit_residual.png)")
print("------------------------------------------------------------")

# build best triple-cone model curve for plotting
model_best = triple_cone(theta, *best_R3)
A_best = np.sum(y * model_best) / np.sum(model_best**2)
fit_best = A_best * model_best

plt.figure(figsize=(10,6))
plt.plot(theta, y, "ko", markersize=3, label="residual density ratio")
plt.plot(theta, fit_best, "r-", linewidth=1.5, label="triple-cone fit")

plt.xlabel("angle from unified axis (deg)")
plt.ylabel("density ratio (residual / isotropic)")
plt.title("residual frb radial profile and triple-cone fit")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("cone_fit_residual.png", dpi=200)

print("saved: cone_fit_residual.png")

print("\n" + "="*70)
print("analysis complete")
print("="*70)
