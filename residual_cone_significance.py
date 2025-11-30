"""
residual_cone_significance.py
------------------------------------------------------------
monte-carlo significance test for the residual triple-cone fit
around the unified axis.

pipeline:
  - rebuild residual weights (same method as residual_cone_test2)
  - construct radial density profile y(theta) for 0–40°
  - compute best triple-cone rss on real data
  - simulate many isotropic residual skies with the SAME
    footprint distribution and weighting pipeline
  - for each mock, fit best triple-cone and record rss
  - estimate p-value: fraction of mocks with rss <= rss_real
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

# configuration
FOOT_L = 129.0
FOOT_B = 23.0          # footprint axis
UNIF_L = 159.85
UNIF_B = -0.51         # unified axis
THETA_MAX_FOOT = 80.0
BIN_WIDTH = 1.0
PROFILE_MAX = 40.0

N_SIM = 2000           # number of monte-carlo realizations

# helpers
def band_prob(theta_low_deg, theta_high_deg):
    t1 = np.radians(theta_low_deg)
    t2 = np.radians(theta_high_deg)
    return (np.cos(t1) - np.cos(t2)) / 2.0

def triple_cone(theta, R1, R2, R3):
    m = np.minimum(np.abs(theta - R1), np.abs(theta - R2))
    m = np.minimum(m, np.abs(theta - R3))
    return m

print("="*70)
print("residual triple-cone significance test")
print("monte-carlo test under isotropic residual null")
print("="*70)

# load catalog
try:
    frbs = pd.read_csv("frbs.csv")
except FileNotFoundError:
    print("\n[fatal] frbs.csv not found. analysis aborted.")
    raise SystemExit

coords = SkyCoord(ra=frbs["ra"].values*u.deg,
                  dec=frbs["dec"].values*u.deg,
                  frame="icrs").galactic

foot_axis = SkyCoord(l=FOOT_L*u.deg, b=FOOT_B*u.deg, frame="galactic")
unif_axis = SkyCoord(l=UNIF_L*u.deg, b=UNIF_B*u.deg, frame="galactic")

theta_instr = coords.separation(foot_axis).deg
theta_unif  = coords.separation(unif_axis).deg

N_all = len(frbs)

print("\n1. dataset and angles")
print("------------------------------------------------------------")
print(f"total frbs: {N_all}")
print(f"min theta_instr = {theta_instr.min():.2f}°, max = {theta_instr.max():.2f}°")
print(f"min theta_unif  = {theta_unif.min():.2f}°, max = {theta_unif.max():.2f}°")

# restrict to valid footprint region
mask_valid = theta_instr <= THETA_MAX_FOOT
theta_instr = theta_instr[mask_valid]
theta_unif  = theta_unif[mask_valid]
M = len(theta_instr)

print(f"after theta_instr <= {THETA_MAX_FOOT:.1f}° cut: {M} frbs retained")

# footprint modelling (same as residual_cone_test2)
print("\n2. footprint model reconstruction")
print("------------------------------------------------------------")

bins_f = np.arange(0, THETA_MAX_FOOT+5, 5)
counts_f, edges_f = np.histogram(theta_instr, bins=bins_f)
centers_f = 0.5*(edges_f[1:] + edges_f[:-1])

shape = counts_f / counts_f.max()
poly = np.poly1d(np.polyfit(centers_f, shape, deg=4))
smooth = np.clip(poly(centers_f), 1e-6, None)

print("fitted 4th-degree polynomial footprint model")
print(f"smooth model range: min={smooth.min():.3f}, max={smooth.max():.3f}")

def footprint_value(theta):
    return np.clip(poly(theta), 1e-6, None)

weights = 1.0 / footprint_value(theta_instr)
total_weight = weights.sum()

print("\n3. residual weighting")
print("------------------------------------------------------------")
print(f"mean weight = {weights.mean():.3f}")
print(f"min  weight = {weights.min():.3f}")
print(f"max  weight = {weights.max():.3f}")
print(f"total_weight = {total_weight:.2f}")

# build radial profile for real data
print("\n4. real residual profile and triple-cone rss")
print("------------------------------------------------------------")

bins_r = np.arange(0, PROFILE_MAX + BIN_WIDTH, BIN_WIDTH)
centers_r = bins_r[:-1] + BIN_WIDTH/2
theta = centers_r

obs_profile = []
exp_profile = []

for low, high in zip(bins_r[:-1], bins_r[1:]):
    mask_band = (theta_unif >= low) & (theta_unif < high)
    w_sum = weights[mask_band].sum()

    p_band = band_prob(low, high)
    mu_band = total_weight * p_band

    obs_profile.append(w_sum)
    exp_profile.append(mu_band)

obs_profile = np.array(obs_profile)
exp_profile = np.array(exp_profile)
density = obs_profile / exp_profile

print(f"profile 0–{PROFILE_MAX:.0f}° with {len(theta)} bins")
print(f"non-zero expected bins: {(exp_profile > 0).sum()}")

# grid search for best triple-cone radii on real data
Rvals = np.linspace(0, PROFILE_MAX, 41)
best_radii = None
best_rss_real = np.inf

for R1 in Rvals:
    for R2 in Rvals:
        for R3 in Rvals:
            model = triple_cone(theta, R1, R2, R3)
            A = np.sum(density * model) / np.sum(model**2)
            fit = A * model
            rss = np.sum((density - fit)**2)
            if rss < best_rss_real:
                best_rss_real = rss
                best_radii = (R1, R2, R3)

print(f"best-fit triple-cone radii (real): R1={best_radii[0]:.2f}°, "
      f"R2={best_radii[1]:.2f}°, R3={best_radii[2]:.2f}°")
print(f"real triple-cone rss = {best_rss_real:.2f}")

# monte-carlo under isotropic residual null
print("\n5. monte-carlo isotropic residual skies")
print("------------------------------------------------------------")
print(f"simulations: {N_SIM}")

rss_sim = np.zeros(N_SIM)

# precompute a distribution for theta_instr by resampling existing values
theta_instr_array = theta_instr.copy()

for k in range(N_SIM):
    # sample instr angles by bootstrap from real distribution
    theta_instr_mock = np.random.choice(theta_instr_array, size=M, replace=True)

    # sample unified angles isotropically: cos(theta) uniform in [-1,1]
    u = np.random.uniform(-1.0, 1.0, size=M)
    theta_unif_mock = np.degrees(np.arccos(u))

    # keep only mock events with same theta_instr cut
    # (theta_instr_mock already respects <= THETA_MAX_FOOT)

    w_mock = 1.0 / footprint_value(theta_instr_mock)
    total_w_mock = w_mock.sum()

    obs_m = []
    exp_m = []

    for low, high in zip(bins_r[:-1], bins_r[1:]):
        band_mask = (theta_unif_mock >= low) & (theta_unif_mock < high)
        w_sum = w_mock[band_mask].sum()

        p_band = band_prob(low, high)
        mu_band = total_w_mock * p_band

        obs_m.append(w_sum)
        exp_m.append(mu_band)

    obs_m = np.array(obs_m)
    exp_m = np.array(exp_m)
    dens_m = obs_m / exp_m

    # fit best triple-cone to this mock profile
    best_rss_mock = np.inf
    for R1 in Rvals:
        for R2 in Rvals:
            for R3 in Rvals:
                model = triple_cone(theta, R1, R2, R3)
                A = np.sum(dens_m * model) / np.sum(model**2)
                fit = A * model
                rss = np.sum((dens_m - fit)**2)
                if rss < best_rss_mock:
                    best_rss_mock = rss

    rss_sim[k] = best_rss_mock

# significance
print("\n6. significance and comparison")
print("------------------------------------------------------------")
p_value = np.mean(rss_sim <= best_rss_real)

print(f"real triple-cone rss = {best_rss_real:.2f}")
print(f"mean rss (isotropic mocks) = {rss_sim.mean():.2f}")
print(f"median rss (isotropic mocks) = {np.median(rss_sim):.2f}")
print(f"5th percentile rss (mocks) = {np.percentile(rss_sim,5):.2f}")
print(f"p-value (rss_mock <= rss_real) = {p_value:.4f}")

if p_value < 0.001:
    print("\ninterpretation:")
    print("  triple-cone structure in the residual profile is much sharper")
    print("  than expected from isotropic residual skies.")
    print("  → strong evidence that the layered cone is not a random artifact.")
elif p_value < 0.01:
    print("\ninterpretation:")
    print("  triple-cone fit is significantly better than typical isotropic mocks.")
    print("  → evidence that the residual layering is real.")
elif p_value < 0.05:
    print("\ninterpretation:")
    print("  triple-cone structure is marginally better than isotropic expectations.")
    print("  → suggestive but not conclusive.")
else:
    print("\ninterpretation:")
    print("  triple-cone fit quality is compatible with what isotropic residual")
    print("  skies can produce by chance.")
    print("  → no strong evidence that the cone structure is special.")

print("\n" + "="*70)
print("analysis complete")
print("="*70)
