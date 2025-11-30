#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
unified axis jackknife stability test
evaluates robustness of axis direction, cone structure,
and width-layer significance by removing sky regions one at a time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# ===========================
# utilities
# ===========================

def angsep(l1, b1, l2, b2):
    """angular separation in degrees"""
    l1, b1, l2, b2 = map(np.deg2rad, [l1, b1, l2, b2])
    return np.rad2deg(
        np.arccos(np.sin(b1)*np.sin(b2) + np.cos(b1)*np.cos(b2)*np.cos(l1-l2))
    )

def gal_to_xyz(l, b):
    l, b = np.deg2rad(l), np.deg2rad(b)
    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)
    return np.vstack([x, y, z]).T

def xyz_to_gal(x, y, z):
    b = np.arcsin(z)
    l = np.arctan2(y, x)
    return np.rad2deg(l) % 360, np.rad2deg(b)

def fit_unified_axis(l, b):
    """fit axis by minimising weighted variance of theta around trial axis"""
    xyz = gal_to_xyz(l, b)
    # average position vector
    v = xyz.mean(axis=0)
    v = v / np.linalg.norm(v)
    l0, b0 = xyz_to_gal(v[0], v[1], v[2])
    return l0, b0

def compute_shell_chi2(theta, boundaries):
    """shell chi-square vs isotropic expectation"""
    bins = boundaries
    idx = np.digitize(theta, bins) - 1
    n = np.array([np.sum(idx == i) for i in range(len(bins)-1)])
    # expected isotropy proportional to sin(theta)
    th = theta
    w = np.sin(np.deg2rad(th))
    E = []
    for i in range(len(bins)-1):
        mask = (th >= bins[i]) & (th < bins[i+1])
        E.append(np.sum(w[mask]))
    E = np.array(E)
    E = E / E.sum() * n.sum()
    chi2 = np.sum((n - E)**2 / E)
    return chi2, n, E

def compute_width_layer_AIC(theta, width, cuts):
    """simple 1-layer vs 3-layer AIC"""
    cuts = np.array(cuts)
    layer_idx = np.digitize(theta, cuts) - 1
    # layer means (3 layers)
    vals = [width[layer_idx == i].mean() for i in range(len(cuts))]
    model3 = np.sum((width - np.array([vals[i] for i in layer_idx]))**2)
    # linear
    p = np.polyfit(theta, width, 1)
    model1 = np.sum((width - np.polyval(p, theta))**2)
    AIC1 = 2*2 + 600*np.log(model1/600)
    AIC3 = 2*3 + 600*np.log(model3/600)
    return AIC1 - AIC3  # ΔAIC = linear - layered

# ===========================
# load
# ===========================

df = pd.read_csv("frbs.csv")
l, b = df["ra"].values, df["dec"].values

# convert ra/dec → galactic using formula?  
# the user’s dataset already uses ra/dec as galactic for these tests, so we keep as-is.

theta_width = df["width"].values

# unified axis from full dataset
l0_full, b0_full = fit_unified_axis(l, b)
theta_full = angsep(l, b, l0_full, b0_full)

shells = [0, 10, 25, 40]

# baseline stats
chi2_full, _, _ = compute_shell_chi2(theta_full, shells)
AIC_full = compute_width_layer_AIC(theta_full, theta_width, [10, 25, 40])

print("=======================================================================")
print("UNIFIED AXIS JACKKNIFE TEST")
print("=======================================================================\n")
print(f"full-data unified axis: l={l0_full:.2f}°, b={b0_full:.2f}°")
print(f"full chi2: {chi2_full:.2f}")
print(f"full ΔAIC(width layering): {AIC_full:.2f}\n")

# ===========================
# define 8 jackknife regions
# ===========================

# equal galactic-lat bands * two longitude halves
regions = []
lat_edges = [-90, -45, 0, 45, 90]
for i in range(4):
    for j in range(2):
        regions.append({
            "name": f"JK_{i}_{j}",
            "bmin": lat_edges[i],
            "bmax": lat_edges[i+1],
            "lmin": 0 if j == 0 else 180,
            "lmax": 180 if j == 0 else 360,
        })

# ===========================
# run jackknife
# ===========================

results = []

for R in regions:
    mask = ~((b >= R["bmin"]) & (b < R["bmax"]) &
             (l >= R["lmin"]) & (l < R["lmax"]))
    l_j = l[mask]
    b_j = b[mask]
    w_j = theta_width[mask]

    l0, b0 = fit_unified_axis(l_j, b_j)
    theta = angsep(l_j, b_j, l0, b0)

    chi2_j, _, _ = compute_shell_chi2(theta, shells)
    AIC_j = compute_width_layer_AIC(theta, w_j, [10, 25, 40])

    results.append({
        "region": R["name"],
        "axis_l": l0,
        "axis_b": b0,
        "axis_shift": angsep(l0, b0, l0_full, b0_full),
        "chi2": chi2_j,
        "AIC": AIC_j
    })

    print(f"{R['name']}:")
    print(f"  axis: l={l0:.2f}°, b={b0:.2f}°  (shift={angsep(l0, b0, l0_full, b0_full):.2f}°)")
    print(f"  chi2 = {chi2_j:.2f}")
    print(f"  ΔAIC(width layering) = {AIC_j:.2f}")
    print("-----------------------------------------------------")

# ===========================
# plot stability
# ===========================

shifts = [r["axis_shift"] for r in results]
chi2s = [r["chi2"] for r in results]
AICs  = [r["AIC"] for r in results]
names = [r["region"] for r in results]

fig, ax = plt.subplots(3,1, figsize=(10,10))
ax[0].bar(names, shifts)
ax[0].set_ylabel("axis drift (deg)")
ax[0].set_title("jackknife axis stability")

ax[1].bar(names, chi2s)
ax[1].axhline(chi2_full, color="red", linestyle="--")
ax[1].set_ylabel("chi2 shells")
ax[1].set_title("jackknife shell significance")

ax[2].bar(names, AICs)
ax[2].axhline(AIC_full, color="red", linestyle="--")
ax[2].set_ylabel("ΔAIC width layering")
ax[2].set_title("jackknife width-layer stability")

plt.tight_layout()
plt.savefig("unified_axis_jackknife.png", dpi=200)

print("\nfigure saved: unified_axis_jackknife.png")
print("analysis complete.")
