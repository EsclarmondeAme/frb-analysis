#!/usr/bin/env python3
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.special import sph_harm
import matplotlib.pyplot as plt

# ============================================================
# unified axis (in galactic)
# ============================================================
UNIFIED_L = 159.85
UNIFIED_B = -0.51

# ============================================================
# stable rotation: RA/Dec → galactic → rotate axis to +Z
# ============================================================
def rotate_to_axis_from_radec(ra_deg, dec_deg,
                              L0=UNIFIED_L, B0=UNIFIED_B):
    """
    convert RA/Dec → galactic → rotate so (L0,B0) becomes +Z.
    returns theta (colatitude), phi (azimuth) in radians.
    this version uses pure 3d rotation matrices (no skyoffset).
    """

    # convert sky to galactic
    icrs = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
    gal = icrs.galactic

    # convert to radians
    l = np.deg2rad(gal.l.value)
    b = np.deg2rad(gal.b.value)

    # convert to 3d unit vectors
    x = np.cos(b) * np.cos(l)
    y = np.cos(b) * np.sin(l)
    z = np.sin(b)
    V = np.vstack([x, y, z]).T  # shape (N,3)

    # target axis → +Z rotation
    Lr = np.deg2rad(L0)
    Br = np.deg2rad(B0)
    ax = np.cos(Br) * np.cos(Lr)
    ay = np.cos(Br) * np.sin(Lr)
    az = np.sin(Br)
    A = np.array([ax, ay, az])

    # compute rotation matrix R that maps A → [0,0,1]
    zhat = np.array([0., 0., 1.])
    v = np.cross(A, zhat)
    s = np.linalg.norm(v)
    c = np.dot(A, zhat)

    if s < 1e-8:
        R = np.eye(3)
    else:
        vx, vy, vz = v
        K = np.array([[0, -vz, vy],
                      [vz, 0, -vx],
                      [-vy, vx, 0]])
        R = np.eye(3) + K + K@K * ((1 - c) / (s**2))

    # rotate all vectors
    Vr = V @ R.T

    # convert back to spherical
    X = Vr[:,0]
    Y = Vr[:,1]
    Z = Vr[:,2]

    theta = np.arccos(Z)             # colatitude
    phi   = np.arctan2(Y, X) % (2*np.pi)

    return theta, phi


# ============================================================
# model components
# ============================================================
def smooth_shell(theta, A, sigma):
    return A * np.exp(-0.5 * ((theta - np.deg2rad(25))/sigma)**2)

def layered_shell(theta, A1, A2, t_break_deg):
    t_break = np.deg2rad(t_break_deg)
    return np.where(theta < t_break, A1, A2)

def patchy_shell(theta, phi, A0, A1, m):
    return A0 * (1 + A1 * np.cos(m * phi))

def cone(theta, C, slope):
    return C * np.exp(-slope * theta)

def multipole(theta, phi, a_dip, a_quad):
    Y10 = sph_harm(0,1,phi,theta).real
    Y20 = sph_harm(0,2,phi,theta).real
    return a_dip * Y10 + a_quad * Y20


# ============================================================
# load data
# ============================================================
df = pd.read_csv("frbs.csv")
ra = df["ra"].values
dec = df["dec"].values

theta, phi = rotate_to_axis_from_radec(ra, dec)

# bins
Nb = 40
th_bins = np.linspace(0, np.deg2rad(140), Nb+1)
ph_bins = np.linspace(0, 2*np.pi, Nb+1)

H, _, _ = np.histogram2d(theta, phi, bins=[th_bins, ph_bins])
data = H.flatten()

# grid centers
th_c = 0.5*(th_bins[:-1] + th_bins[1:])
ph_c = 0.5*(ph_bins[:-1] + ph_bins[1:])
TH, PH = np.meshgrid(th_c, ph_c, indexing='ij')

Theta = TH.flatten()
Phi   = PH.flatten()


# ============================================================
# wrappers for curve_fit
# ============================================================
def fit_model(func, p0):
    try:
        popt, _ = curve_fit(func, (Theta, Phi), data, p0=p0, maxfev=20000)
        pred = func((Theta, Phi), *popt)
        rss = np.sum((data - pred)**2)
        k = len(popt)
        AIC = 2*k + len(data) * np.log(rss/len(data))
        return popt, rss, AIC
    except:
        return None, np.inf, np.inf

def smooth_wrapper(X, A, sigma):
    T, P = X
    return smooth_shell(T, A, sigma)

def layered_wrapper(X, A1, A2, tb):
    T, P = X
    return layered_shell(T, A1, A2, tb)

def patchy_wrapper(X, A0, A1, m):
    T, P = X
    return patchy_shell(T, P, A0, A1, int(round(m)))

def cone_wrapper(X, C, slope):
    T, P = X
    return cone(T, C, slope)

def multipole_wrapper(X, a, b):
    T, P = X
    return multipole(T, P, a, b)


# ============================================================
# fits
# ============================================================
results = []

results.append(("smooth_shell", *fit_model(smooth_wrapper, [10, 0.3])))
results.append(("layered_shell", *fit_model(layered_wrapper, [5, 3, 25])))
results.append(("patchy_shell", *fit_model(patchy_wrapper, [5, 0.3, 2])))
results.append(("cone", *fit_model(cone_wrapper, [5, 2])))
results.append(("dipole+quadrupole", *fit_model(multipole_wrapper, [1,1])))

# sort
results_sorted = sorted(results, key=lambda x: x[3])
best = results_sorted[0]

print("=======================================================")
print("FRB PATCHY-SHELL SHAPE MODEL (stable rotation)")
print("=======================================================\n")

for name,popt,rss,AIC in results_sorted:
    print(f"{name:20s}  AIC={AIC:8.2f}   RSS={rss:10.2f}   params={popt}")

print("\n-------------------------------------------------------")
print(f"BEST MODEL: {best[0]}")
print("-------------------------------------------------------")

# figure
plt.figure(figsize=(8,6))
plt.title("Model Comparison (AIC)")
names = [r[0] for r in results_sorted]
AICs  = [r[3] for r in results_sorted]
plt.bar(range(len(AICs)), AICs, tick_label=names)
plt.xticks(rotation=45)
plt.ylabel("AIC (lower is better)")
plt.tight_layout()
plt.savefig("frb_patchy_shell_fit.png")
print("saved: frb_patchy_shell_fit.png")
