import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.coordinates import SkyCoord
import astropy.units as u

# ================================================================
# 2D RADIAL PROFILE (φ-averaged)
# ================================================================
# produces: radial_profile.png + layer fit report
# ================================================================

# unified axis from your best-fit model
AXIS_L = 159.85
AXIS_B = -0.51

# binning
MAX_THETA = 140.0        # degrees
BIN_SIZE  = 1.0          # degrees
BINS = np.arange(0, MAX_THETA + BIN_SIZE, BIN_SIZE)

# load FRB data
df = pd.read_csv("frbs.csv")

# compute galactic coords
coords = SkyCoord(ra=df["ra"].values * u.deg,
                  dec=df["dec"].values * u.deg,
                  frame="icrs").galactic

# compute theta from unified axis
axis = SkyCoord(l=AXIS_L * u.deg, b=AXIS_B * u.deg, frame='galactic')
theta = coords.separation(axis).deg

df["theta"] = theta

# keep only FRBs inside 140°
mask = theta <= MAX_THETA
thetas = theta[mask]

# ================================================================
# RADIAL COUNTS
# ================================================================
hist, edges = np.histogram(thetas, bins=BINS)
centers = 0.5 * (edges[1:] + edges[:-1])

# ================================================================
# ISOTROPIC EXPECTATION
# ================================================================
# isotropic PDF in theta: p(theta) ∝ sin(theta)
iso_pdf = np.sin(np.deg2rad(centers))
iso_pdf /= iso_pdf.sum()

expected = iso_pdf * hist.sum()

# density ratio
ratio = hist / expected
ratio_err = np.sqrt(hist) / expected

# ================================================================
# LAYER-FITTING MODELS
# ================================================================

# --- model A: smooth polynomial -----------------------------
def poly_model(theta, p):
    return p[0] + p[1]*theta + p[2]*theta**2

def poly_rss(p):
    m = poly_model(centers, p)
    return np.sum((ratio - m)**2)

# --- model B: 2-layer piecewise constant ---------------------
def two_layer_model(theta, R1, v1, v2):
    return np.where(theta < R1, v1, v2)

def two_layer_rss(params):
    R1, v1, v2 = params
    m = two_layer_model(centers, R1, v1, v2)
    return np.sum((ratio - m)**2)

# --- model C: 3-layer piecewise constant ---------------------
def three_layer_model(theta, R1, R2, v1, v2, v3):
    return np.where(theta < R1, v1,
           np.where(theta < R2, v2, v3))

def three_layer_rss(params):
    R1, R2, v1, v2, v3 = params
    if R2 <= R1:
        return 1e9
    m = three_layer_model(centers, R1, R2, v1, v2, v3)
    return np.sum((ratio - m)**2)

# ================================================================
# FIT MODELS
# ================================================================
poly_fit = minimize(poly_rss, x0=[1.0, 0.0, 0.0]).x

two_fit = minimize(two_layer_rss,
                   x0=[25.0, 1.0, 0.5],
                   bounds=[(1,80),(0,None),(0,None)]).x

three_fit = minimize(three_layer_rss,
                     x0=[10.0, 25.0, 2.0, 1.0, 0.5],
                     bounds=[(1,80),(1,80),(0,None),(0,None),(0,None)]).x

rss_poly  = poly_rss(poly_fit)
rss_two   = two_layer_rss(two_fit)
rss_three = three_layer_rss(three_fit)

# AIC calculation
def AIC(rss, k):
    return 2*k + len(centers)*np.log(rss/len(centers))

aic_poly  = AIC(rss_poly, 3)
aic_two   = AIC(rss_two, 3)
aic_three = AIC(rss_three, 5)

# ================================================================
# PRINT RESULTS
# ================================================================
print("==============================================================")
print("2D RADIAL PROFILE — φ-averaged layer structure")
print("==============================================================")

print("\nFitted radii and model comparison:")
print("-----------------------------------")
print(f"Polynomial model RSS = {rss_poly:.2f}, AIC = {aic_poly:.2f}")
print(f"Two-layer model: R1={two_fit[0]:.2f}°, v1={two_fit[1]:.3f}, v2={two_fit[2]:.3f}")
print(f"RSS = {rss_two:.2f}, AIC = {aic_two:.2f}")
print(f"Three-layer model: R1={three_fit[0]:.2f}°, R2={three_fit[1]:.2f}°")
print(f"                v1={three_fit[2]:.3f}, v2={three_fit[3]:.3f}, v3={three_fit[4]:.3f}")
print(f"RSS = {rss_three:.2f}, AIC = {aic_three:.2f}")

best = min(aic_poly, aic_two, aic_three)
if best == aic_three:
    print("\n→ Best model: THREE-LAYER (AIC minimum)")
elif best == aic_two:
    print("\n→ Best model: TWO-LAYER (AIC minimum)")
else:
    print("\n→ Best model: POLYNOMIAL (AIC minimum)")

# ================================================================
# PLOT
# ================================================================
plt.figure(figsize=(12,6))

plt.errorbar(centers, ratio, yerr=ratio_err, fmt='.', color='black',
             label='Observed density ratio')

# plot fits
theta_fine = np.linspace(0, MAX_THETA, 500)
plt.plot(theta_fine,
         poly_model(theta_fine, poly_fit),
         label="Polynomial fit", lw=2)

plt.plot(theta_fine,
         two_layer_model(theta_fine, *two_fit),
         label="Two-layer fit", lw=2)

plt.plot(theta_fine,
         three_layer_model(theta_fine, *three_fit),
         label="Three-layer fit", lw=2)

plt.axvline(two_fit[0], color='gray', ls='--', alpha=0.5)
plt.axvline(three_fit[0], color='gray', ls='--', alpha=0.5)
plt.axvline(three_fit[1], color='gray', ls='--', alpha=0.5)

plt.xlabel("θ from unified axis (deg)")
plt.ylabel("Density ratio (observed / isotropic)")
plt.title("FRB Radial Layer Profile (φ-averaged)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("radial_profile.png", dpi=200, bbox_inches='tight')

print("\nPlot saved as: radial_profile.png")
print("==============================================================")
print("analysis complete")
print("==============================================================")
