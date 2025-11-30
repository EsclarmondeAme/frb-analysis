#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# ============================================================
# unified axis (galactic)
# ============================================================
UNIFIED_L = 159.85
UNIFIED_B = -0.51

# ============================================================
# convert RA/DEC → Galactic using our own rotation matrix
# (avoids astropy completely)
# ============================================================
def radec_to_gal(ra_deg, dec_deg):
    """
    stable RA/DEC → Galactic conversion using J2000 matrix.
    """

    ra  = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    # IAU 1958 + J2000 galactic rotation matrix
    R = np.array([
        [-0.05487556, -0.87343709, -0.48383502],
        [ 0.49410943, -0.44482963,  0.74698225],
        [-0.86766615, -0.19807637,  0.45598378]
    ])

    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)

    v = np.vstack([x, y, z])
    g = R @ v

    l = np.arctan2(g[1], g[0])
    b = np.arcsin(g[2])

    return np.rad2deg(l % (2*np.pi)), np.rad2deg(b)

# ============================================================
# rotate (l,b) so unified axis becomes the pole
# ============================================================
def rotate_to_axis(l_deg, b_deg, L0=UNIFIED_L, B0=UNIFIED_B):
    """
    robust rotation using spherical rotation matrix.
    returns (theta, phi) in degrees.
    """

    # convert to radians
    l = np.deg2rad(l_deg)
    b = np.deg2rad(b_deg)
    L0 = np.deg2rad(L0)
    B0 = np.deg2rad(B0)

    # convert to cartesian
    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)
    v = np.vstack([x, y, z])

    # axis direction
    x0 = np.cos(B0)*np.cos(L0)
    y0 = np.cos(B0)*np.sin(L0)
    z0 = np.sin(B0)
    axis = np.array([x0, y0, z0])

    # construct rotation matrix that maps axis → north pole
    # north pole vector
    npole = np.array([0, 0, 1])

    # rotation axis (cross product)
    k = np.cross(axis, npole)
    s = np.linalg.norm(k)
    c = np.dot(axis, npole)

    if s == 0:
        R = np.eye(3)
    else:
        k = k/s
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
        R = np.eye(3) + K + K@K*((1-c)/s**2)

    # rotate
    vr = R @ v

    xr, yr, zr = vr[0], vr[1], vr[2]

    theta = np.rad2deg(np.arccos(zr))     # polar
    phi   = np.rad2deg(np.arctan2(yr, xr)) % 360

    return theta, phi

# ============================================================
# models
# ============================================================
def model_dipole(theta):
    ct = np.cos(np.deg2rad(theta))
    return 1 + 0.5*ct

def model_quadrupole(theta):
    ct = np.cos(np.deg2rad(theta))
    P2 = 0.5*(3*ct*ct - 1)
    return 1 + 0.5*P2

def model_cone(theta, t0=25):
    return np.exp(-0.5*((theta - t0)/15)**2)

def model_smooth(theta, a=40, b=0.03, c=-0.0003):
    return a + b*theta + c*theta**2

# ============================================================
# likelihood
# ============================================================
def gaussian_ll(data, model):
    res = data - model
    return -0.5*np.sum(res*res)

def AIC(ll, k):
    return 2*k - 2*ll

# ============================================================
# load data
# ============================================================
df = pd.read_csv("frbs.csv")

# convert RA/DEC → galactic
l, b = radec_to_gal(df["ra"].values, df["dec"].values)

# rotate to unified axis
theta, phi = rotate_to_axis(l, b)

# radial density
bins = np.linspace(0, 140, 50)
H, _ = np.histogram(theta, bins=bins)
theta_centers = 0.5*(bins[:-1] + bins[1:])
data = H.astype(float) / np.max(H)

# evaluate models
models = {
    "dipole":      (model_dipole(theta_centers), 1),
    "quadrupole":  (model_quadrupole(theta_centers), 1),
    "cone":        (model_cone(theta_centers), 1),
    "smooth":      (model_smooth(theta_centers), 3)
}

print("=====================================================")
print("UNIFIED ANISOTROPY MODEL (stable rotation version)")
print("=====================================================")

best_name = None
best_aic = 1e99

for name, (m, k) in models.items():
    m = m / np.max(m)
    ll = gaussian_ll(data, m)
    aic = AIC(ll, k)
    print(f"{name:12s} AIC = {aic:8.2f}")
    if aic < best_aic:
        best_aic = aic
        best_name = name

print("-----------------------------------------------------")
print(f"BEST MODEL: {best_name}")
print("=====================================================")

# plot
plt.figure(figsize=(8,5))
plt.plot(theta_centers, data, ".", color="k", label="data")
for name,(m,_) in models.items():
    plt.plot(theta_centers, m/np.max(m), label=name)
plt.xlabel("theta (deg)")
plt.ylabel("normalized density")
plt.legend()
plt.title("unified anisotropy models")
plt.savefig("frb_unified_anisotropy_model.png", dpi=150)
plt.close()

print("saved: frb_unified_anisotropy_model.png")
