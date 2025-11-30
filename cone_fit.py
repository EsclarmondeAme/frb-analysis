"""
cone_fit.py
------------------------------------------------------------
fit a multi-layer cone model to frb angular distances from
the unified axis and test whether observed 'shells' match
a nested cone structure.

outputs:
  - best-fit cone opening angles
  - least-squares fit metrics
  - model comparison using AIC
  - 3d visualization of cones
  - angular shell profile figure
------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------
# unified axis (from your analysis)
# ------------------------------------------------------------
AXIS_L = 159.85
AXIS_B = -0.51
axis = SkyCoord(l=AXIS_L*u.deg, b=AXIS_B*u.deg, frame="galactic")

print("="*70)
print("cone model fit")
print("fitting nested cone layers around unified axis")
print("="*70)

# ------------------------------------------------------------
# load FRBs
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
frbs["sep_deg"] = sep

print("\n1. frb catalog")
print("------------------------------------------------------------")
print(f"total frbs: {len(frbs)}")
print("computed angles from unified axis")

# ------------------------------------------------------------
# detected shell boundaries (from previous scan)
# ------------------------------------------------------------
shells = np.array([1.5, 5.5, 14.5, 27.5, 31.5])
print("\n2. input shell boundaries")
print("------------------------------------------------------------")
print("shell boundaries (deg): " + ", ".join(f"{s:.1f}" for s in shells))

# ------------------------------------------------------------
# model definitions
# ------------------------------------------------------------
def single_cone(theta, R):
    """one cone with radius R"""
    return np.abs(theta - R)

def double_cone(theta, R1, R2):
    """two nested cones"""
    model = np.minimum(np.abs(theta - R1), np.abs(theta - R2))
    return model

def triple_cone(theta, R1, R2, R3):
    """three nested cones"""
    m = np.minimum(np.abs(theta - R1), np.abs(theta - R2))
    m = np.minimum(m, np.abs(theta - R3))
    return m

def fit_single(theta, data):
    Rs = np.linspace(0, 40, 400)
    chi = []
    for R in Rs:
        model = single_cone(theta, R)
        chi.append(np.sum((data - model)**2))
    idx = np.argmin(chi)
    return Rs[idx], chi[idx]

def fit_double(theta, data):
    Rvals = np.linspace(0, 40, 200)
    best = None
    best_chi = np.inf
    for R1 in Rvals:
        for R2 in Rvals:
            model = double_cone(theta, R1, R2)
            chi = np.sum((data - model)**2)
            if chi < best_chi:
                best_chi = chi
                best = (R1, R2)
    return best, best_chi

def fit_triple(theta, data):
    Rvals = np.linspace(0, 40, 80)
    best = None
    best_chi = np.inf
    for R1 in Rvals:
        for R2 in Rvals:
            for R3 in Rvals:
                model = triple_cone(theta, R1, R2, R3)
                chi = np.sum((data - model)**2)
                if chi < best_chi:
                    best_chi = chi
                    best = (R1, R2, R3)
    return best, best_chi

# ------------------------------------------------------------
# observed curve (to fit against)
# ------------------------------------------------------------
# density ratio curve from shell scan
bin_width = 1.0
bins = np.arange(0, 90 + bin_width, bin_width)
centers = bins[:-1] + bin_width/2

# compute isotropic expectation
total = len(frbs)
iso = []
counts = []
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

theta = centers
data = density

# ------------------------------------------------------------
# fit models
# ------------------------------------------------------------
print("\n3. fitting cone models")
print("------------------------------------------------------------")

print("fitting single-cone model...")
R1, chi1 = fit_single(theta, data)
print(f"best-fit radius: {R1:.2f}°")

print("\nfitting double-cone model (may take a moment)...")
(R2a, R2b), chi2 = fit_double(theta, data)
print(f"best-fit radii: {R2a:.2f}°, {R2b:.2f}°")

print("\nfitting triple-cone model (slow)...")
(R3a, R3b, R3c), chi3 = fit_triple(theta, data)
print(f"best-fit radii: {R3a:.2f}°, {R3b:.2f}°, {R3c:.2f}°")

# ------------------------------------------------------------
# AIC comparison
# ------------------------------------------------------------
def aic(chi, k):
    return 2*k + chi

a1 = aic(chi1, 1)
a2 = aic(chi2, 2)
a3 = aic(chi3, 3)

print("\n4. model comparison (aic)")
print("------------------------------------------------------------")
print(f"single cone aic:  {a1:.2f}")
print(f"double cone aic:  {a2:.2f}")
print(f"triple cone aic:  {a3:.2f}")

best_model = np.argmin([a1, a2, a3])
labels = ["single", "double", "triple"]
print(f"\nbest model: {labels[best_model]} cone")

# ------------------------------------------------------------
# 3d visualization of best-fit cones
# ------------------------------------------------------------
print("\n5. generating 3d visualization")
print("------------------------------------------------------------")

def plot_cone(R, color, ax):
    # R is in degrees; convert to radians half-angle
    theta0 = np.radians(R)
    h = 1
    r = np.tan(theta0) * h

    z = np.linspace(0, h, 50)
    t = np.linspace(0, 2*np.pi, 50)
    T, Z = np.meshgrid(t, z)

    X = (Z * np.tan(theta0)) * np.cos(T)
    Y = (Z * np.tan(theta0)) * np.sin(T)

    ax.plot_surface(X, Y, 1-Z, color=color, alpha=0.2, linewidth=0)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("cone structure (best-fit nested cones)")

# plot best-fit cones
if best_model == 0:
    plot_cone(R1, 'blue', ax)
elif best_model == 1:
    plot_cone(R2a, 'blue', ax)
    plot_cone(R2b, 'green', ax)
else:
    plot_cone(R3a, 'blue', ax)
    plot_cone(R3b, 'green', ax)
    plot_cone(R3c, 'red', ax)

plt.savefig("cone_fit_3d.png", dpi=200, bbox_inches='tight')

print("saved: cone_fit_3d.png")

print("\n" + "="*70)
print("analysis complete")
print("="*70)
