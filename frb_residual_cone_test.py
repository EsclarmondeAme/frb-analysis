"""
frb_residual_cone_test.py
------------------------------------------------------------
Scientific diagnostic:
    Tests whether FRB angular structure around the unified axis
    persists after removing the dominant instrument footprint.

Methodology:
    1. Model footprint profile f(θ_instr) where θ_instr is the
       separation from the instrument axis (l=129°, b=23°).
    2. Fit a smooth exposure function via low-order polynomial.
    3. Compute residual weights w = 1 / f(θ_instr).
    4. Re-evaluate FRB distribution with respect to the unified axis
       in three angular bands (0–10°, 10–25°, 25–40°).
    5. Compare weighted counts against isotropic expectations.
    6. Output significance via χ² statistic.

All steps are purely statistical and do not assume any physical model.
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# axes
# ------------------------------------------------------------
FOOT_L = 129.0
FOOT_B = 23.0      # instrument footprint axis

UNIF_L = 159.85
UNIF_B = -0.51     # unified cosmic axis

foot_axis = SkyCoord(l=FOOT_L*u.deg, b=FOOT_B*u.deg, frame="galactic")
unif_axis = SkyCoord(l=UNIF_L*u.deg, b=UNIF_B*u.deg, frame="galactic")

# ------------------------------------------------------------
# loading data
# ------------------------------------------------------------
print("="*72)
print("RESIDUAL ANISOTROPY ANALYSIS")
print("removing instrument footprint → evaluating unified cosmic axis")
print("="*72)

try:
    frbs = pd.read_csv("frbs.csv")
except FileNotFoundError:
    print("\n[FATAL] frbs.csv not found. Analysis aborted.")
    raise SystemExit

N = len(frbs)

print("\nSECTION 1 — DATASET")
print("--------------------------------------------------------------------")
print(f"• total FRBs loaded: {N}")
print(f"• footprint axis (galactic):  l={FOOT_L:.2f}°, b={FOOT_B:.2f}°")
print(f"• unified axis (galactic):    l={UNIF_L:.2f}°, b={UNIF_B:.2f}°")

coords = SkyCoord(ra=frbs["ra"].values*u.deg,
                  dec=frbs["dec"].values*u.deg,
                  frame="icrs").galactic

# ------------------------------------------------------------
# compute θ_instr and θ_unif
# ------------------------------------------------------------
theta_instr = coords.separation(foot_axis).deg
theta_unif  = coords.separation(unif_axis).deg

frbs["theta_instr"] = theta_instr
frbs["theta_unif"]  = theta_unif

print("\nSECTION 2 — ANGULAR SEPARATIONS")
print("--------------------------------------------------------------------")
print("computed angular distances from both footprint and unified axes")
print(f"• min θ_instr = {theta_instr.min():.2f}°   max = {theta_instr.max():.2f}°")
print(f"• min θ_unif  = {theta_unif.min():.2f}°    max = {theta_unif.max():.2f}°")

# ------------------------------------------------------------
# footprint modelling
# ------------------------------------------------------------
print("\nSECTION 3 — FOOTPRINT MODELLING")
print("--------------------------------------------------------------------")
print("constructing smooth exposure model f(θ_instr)")

bins = np.arange(0, 95, 5)
counts, edges = np.histogram(theta_instr, bins=bins)
centers = 0.5*(edges[1:] + edges[:-1])

# relative shape
shape = counts / counts.max()

# polynomial fit
poly = np.poly1d(np.polyfit(centers, shape, deg=4))
smooth = np.clip(poly(centers), 1e-6, None)

print(f"• polynomial degree: 4")
print(f"• smooth model range: min={smooth.min():.3f}, max={smooth.max():.3f}")

# function to evaluate model at arbitrary θ
def footprint_value(theta):
    return np.clip(poly(theta), 1e-6, None)

# ------------------------------------------------------------
# residual weights
# ------------------------------------------------------------
frbs["weight"] = 1.0 / footprint_value(frbs["theta_instr"].values)

print("\nSECTION 4 — RESIDUAL WEIGHTING")
print("--------------------------------------------------------------------")
print("weights represent footprint-corrected contribution per FRB")
print(f"• mean weight = {frbs['weight'].mean():.3f}")
print(f"• max  weight = {frbs['weight'].max():.3f}")

# ------------------------------------------------------------
# evaluate unified axis after correction
# ------------------------------------------------------------
print("\nSECTION 5 — LAYERED STRUCTURE AROUND UNIFIED AXIS (RESIDUAL)")
print("--------------------------------------------------------------------")

BANDS = [
    ("inner",  0.0, 10.0),
    ("middle", 10.0, 25.0),
    ("outer",  25.0, 40.0),
]

# isotropic expectation
def isotropic_exp(N, low, high):
    t1 = np.radians(low)
    t2 = np.radians(high)
    p = (np.cos(t1) - np.cos(t2)) / 2
    return N*p

exp = np.array([isotropic_exp(N, low, high) for (_, low, high) in BANDS])

obs = []
for (_, low, high) in BANDS:
    mask = (frbs["theta_unif"] >= low) & (frbs["theta_unif"] < high)
    obs.append(frbs.loc[mask, "weight"].sum())

obs = np.array(obs)

for (name, low, high), o, e in zip(BANDS, obs, exp):
    print(f"{name:6s} band {low:4.0f}°–{high:4.0f}°:")
    print(f"   • residual observed = {o:7.2f}")
    print(f"   • isotropic expect. = {e:7.2f}")
    print(f"   • ratio            = {o/e:7.2f}\n")

chi2 = np.sum((obs - exp)**2 / exp)
print(f"→ residual χ² around unified axis = {chi2:.2f}")

# ------------------------------------------------------------
# figures
# ------------------------------------------------------------
print("\nSECTION 6 — FIGURE OUTPUT")
print("--------------------------------------------------------------------")

plt.figure(figsize=(12,5))

# footprint model
plt.subplot(121)
plt.plot(centers, shape, "o", label="raw footprint")
plt.plot(centers, smooth, "-", label="smooth model")
plt.xlabel("θ_instr (deg)")
plt.ylabel("relative exposure")
plt.title("Instrument Footprint Model")
plt.legend()

# weighted θ_unif distribution
plt.subplot(122)
plt.hist(frbs["theta_unif"], bins=30, weights=frbs["weight"],
         color="gold", alpha=0.7, edgecolor="black")
plt.xlabel("θ_unif (deg)")
plt.ylabel("weighted count")
plt.title("Residual Distribution vs Unified Axis")

plt.tight_layout()
plt.savefig("residual_cone_test.png", dpi=200)

print("✓ saved: residual_cone_test.png")

# ------------------------------------------------------------
# SECTION 7 — VERDICT
# ------------------------------------------------------------
print("\nSECTION 7 — VERDICT")
print("--------------------------------------------------------------------")

# expected chi² for 3 bands under isotropy ~ degrees of freedom = 3
# mean ≈ 3, 95th percentile ≈ 7.8 (chi-square df=3)
from scipy.stats import chi2 as chi2_dist
p_resid = 1 - chi2_dist.cdf(chi2, df=3)

print(f"• residual χ² = {chi2:.2f}")
print(f"• p-value (isotropic null, df=3) = {p_resid:.4f}")

# interpret
if chi2 < 2:
    print("\ninterpretation:")
    print("  residual distribution is fully consistent with isotropy.")
    print("  → no surviving structure around unified axis.")
elif chi2 < 7.8:
    print("\ninterpretation:")
    print("  residual distribution shows mild deviations but within")
    print("  the 95% range expected from an isotropic sky.")
    print("  → no statistically significant residual layering.")
elif chi2 < 15:
    print("\ninterpretation:")
    print("  residual deviations exceed isotropic expectations at >95% level.")
    print("  → weak evidence for residual structure around unified axis.")
else:
    print("\ninterpretation:")
    print("  residual deviations are far above isotropic expectations.")
    print("  → strong evidence that structure remains even after")
    print("    removing the instrument footprint.")


print("\n" + "="*72)
print("ANALYSIS COMPLETE")
print("="*72)
