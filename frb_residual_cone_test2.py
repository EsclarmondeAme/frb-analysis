"""
frb_residual_cone_test2.py
------------------------------------------------------------
refined residual anisotropy test:

- restricts to frbs within a safe footprint range (theta_instr <= 80°)
- models the footprint profile f(theta_instr)
- computes weights w = 1 / f(theta_instr)
- uses total_weight = sum(w) for isotropic expectations
- evaluates layered structure around unified axis after correction
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist

# axes
FOOT_L = 129.0
FOOT_B = 23.0

UNIF_L = 159.85
UNIF_B = -0.51

foot_axis = SkyCoord(l=FOOT_L*u.deg, b=FOOT_B*u.deg, frame="galactic")
unif_axis = SkyCoord(l=UNIF_L*u.deg, b=UNIF_B*u.deg, frame="galactic")

print("="*72)
print("REFINED RESIDUAL ANISOTROPY ANALYSIS")
print("footprint-corrected structure around unified cosmic axis")
print("="*72)

# load catalog
try:
    frbs = pd.read_csv("frbs.csv")
except FileNotFoundError:
    print("\n[fatal] frbs.csv not found. analysis aborted.")
    raise SystemExit

N_all = len(frbs)

coords = SkyCoord(ra=frbs["ra"].values*u.deg,
                  dec=frbs["dec"].values*u.deg,
                  frame="icrs").galactic

theta_instr = coords.separation(foot_axis).deg
theta_unif  = coords.separation(unif_axis).deg

frbs["theta_instr"] = theta_instr
frbs["theta_unif"]  = theta_unif

print("\nsection 1 — dataset and angles")
print("--------------------------------------------------------------------")
print(f"total frbs loaded: {N_all}")
print(f"footprint axis: l={FOOT_L:.2f}°, b={FOOT_B:.2f}°")
print(f"unified axis : l={UNIF_L:.2f}°, b={UNIF_B:.2f}°")
print(f"min theta_instr = {theta_instr.min():.2f}°, max = {theta_instr.max():.2f}°")
print(f"min theta_unif  = {theta_unif.min():.2f}°, max = {theta_unif.max():.2f}°")

# restrict to region with meaningful exposure
theta_max = 80.0
mask_valid = theta_instr <= theta_max
frbs = frbs.loc[mask_valid].copy()
coords = coords[mask_valid]
theta_instr = theta_instr[mask_valid]
theta_unif  = theta_unif[mask_valid]
M = len(frbs)

print(f"\nrestricting to theta_instr <= {theta_max:.1f}°")
print(f"frbs retained after cut: {M} (out of {N_all})")

# footprint modelling
print("\nsection 2 — footprint modelling")
print("--------------------------------------------------------------------")
bins = np.arange(0, theta_max+5, 5)
counts, edges = np.histogram(theta_instr, bins=bins)
centers = 0.5*(edges[1:]+edges[:-1])

shape = counts / counts.max()
poly = np.poly1d(np.polyfit(centers, shape, deg=4))
smooth = np.clip(poly(centers), 1e-6, None)

print("fitted 4th-degree polynomial footprint model")
print(f"smooth model range (within {theta_max:.0f}°): "
      f"min={smooth.min():.3f}, max={smooth.max():.3f}")

def footprint_value(theta):
    return np.clip(poly(theta), 1e-6, None)

# weights
frbs["weight"] = 1.0 / footprint_value(theta_instr)
total_weight = frbs["weight"].sum()

print("\nsection 3 — residual weighting")
print("--------------------------------------------------------------------")
print("weights correct for footprint; effective sample size is total_weight")
print(f"mean weight = {frbs['weight'].mean():.3f}")
print(f"min  weight = {frbs['weight'].min():.3f}")
print(f"max  weight = {frbs['weight'].max():.3f}")
print(f"total_weight (sum w_i) = {total_weight:.2f}")

# layered test around unified axis
print("\nsection 4 — layered residual structure around unified axis")
print("--------------------------------------------------------------------")

BANDS = [
    ("inner",  0.0, 10.0),
    ("middle", 10.0, 25.0),
    ("outer",  25.0, 40.0),
]

def iso_prob(low, high):
    t1 = np.radians(low)
    t2 = np.radians(high)
    return (np.cos(t1) - np.cos(t2)) / 2.0

obs = []
exp = []

for (name, low, high) in BANDS:
    band_mask = (frbs["theta_unif"] >= low) & (frbs["theta_unif"] < high)
    w_sum = frbs.loc[band_mask, "weight"].sum()
    p_band = iso_prob(low, high)
    mu_band = total_weight * p_band

    obs.append(w_sum)
    exp.append(mu_band)

    print(f"{name:6s} band {low:4.0f}°–{high:4.0f}°:")
    print(f"   residual observed (weighted) = {w_sum:7.2f}")
    print(f"   isotropic expectation        = {mu_band:7.2f}")
    print(f"   ratio                        = {w_sum/mu_band:7.2f}\n")

obs = np.array(obs)
exp = np.array(exp)

chi2_val = np.sum((obs - exp)**2 / exp)
df = len(BANDS)  # three bands

print(f"residual chi² (3 bands) = {chi2_val:.2f}")

# figure
print("\nsection 5 — figures")
print("--------------------------------------------------------------------")

plt.figure(figsize=(12,5))

plt.subplot(121)
plt.plot(centers, shape, "o", label="raw footprint")
plt.plot(centers, smooth, "-", label="smooth model")
plt.xlabel("theta_instr (deg)")
plt.ylabel("relative exposure")
plt.title("instrument footprint model (restricted range)")
plt.legend()

plt.subplot(122)
plt.hist(frbs["theta_unif"], bins=30, weights=frbs["weight"],
         color="gold", alpha=0.7, edgecolor="black")
plt.xlabel("theta_unif (deg)")
plt.ylabel("weighted count")
plt.title("residual distribution vs unified axis")

plt.tight_layout()
plt.savefig("residual_cone_test2.png", dpi=200)

print("saved: residual_cone_test2.png")

# verdict
print("\nsection 6 — verdict")
print("--------------------------------------------------------------------")
p_resid = 1 - chi2_dist.cdf(chi2_val, df=df)
print(f"residual chi² = {chi2_val:.2f} (df={df})")
print(f"p-value (isotropic null) = {p_resid:.4f}")

if chi2_val < 2:
    print("\ninterpretation:")
    print("  residual distribution is fully consistent with isotropy.")
    print("  → no surviving structure around unified axis.")
elif chi2_val < 7.8:
    print("\ninterpretation:")
    print("  residual deviations are mild and within")
    print("  the ~95% range expected from isotropy.")
    print("  → no statistically significant residual layering.")
elif chi2_val < 15:
    print("\ninterpretation:")
    print("  residual deviations exceed isotropic expectations at >95% level.")
    print("  → weak evidence for residual structure around unified axis.")
else:
    print("\ninterpretation:")
    print("  residual deviations are far above isotropic expectations.")
    print("  → strong evidence that structure remains even after")
    print("    correcting for the footprint in the tested range.")

print("\n" + "="*72)
print("analysis complete")
print("="*72)
