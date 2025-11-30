#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize
from scipy.stats import binned_statistic_2d
from scipy.special import sph_harm

"""
frb_shape_inference.py
---------------------------------------
scientific geometric model inference for the frb sky,
in the unified-axis frame.

fits the following geometric families:
1. polar cap
2. single cone
3. double cone (bicone)
4. toroidal / ring profile
5. gaussian sheet (planar ridge)
6. m-faced pyramid (azimuthal faceting)
7. polynomial radial model (control)

each model produces a predicted 2d density f(theta,phi),
then the rss and aic are computed.

outputs:
- ranked models by aic
- best-fit parameters for each model
- a scientific verdict
- plot of model fits

"""

# unified axis from earlier analyses
UNIFIED_L = 159.85
UNIFIED_B = -0.51

# load frb catalog
DF = pd.read_csv("frbs.csv")

def rotate_to_axis(l, b, L0=UNIFIED_L, B0=UNIFIED_B):
    """
    rotate galactic coords (l,b) to frame where the unified axis is at theta=0.
    uses explicit rotation matrix (no astropy).
    """
    l = np.radians(l)
    b = np.radians(b)
    L0 = np.radians(L0)
    B0 = np.radians(B0)

    # convert FRBs to cartesian
    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)

    # axis vector
    ax_x = np.cos(B0)*np.cos(L0)
    ax_y = np.cos(B0)*np.sin(L0)
    ax_z = np.sin(B0)
    z_new = np.array([ax_x, ax_y, ax_z])

    # pick nonparallel vector for cross product
    tmp = np.array([0,0,1])
    if abs(np.dot(z_new, tmp)) > 0.99:
        tmp = np.array([1,0,0])

    x_new = np.cross(tmp, z_new)
    x_new /= np.linalg.norm(x_new)
    y_new = np.cross(z_new, x_new)

    # rotation matrix (old→new)
    R = np.vstack([x_new, y_new, z_new])

    xyz = np.vstack([x, y, z])
    Xn, Yn, Zn = R @ xyz

    # spherical coords in new frame
    theta = np.arccos(np.clip(Zn, -1, 1))
    phi = np.arctan2(Yn, Xn)
    phi = (phi % (2*np.pi))

    return np.degrees(theta), np.degrees(phi)

# convert ra,dec to galactic
coords = SkyCoord(ra=DF["ra"].values * u.deg,
                  dec=DF["dec"].values * u.deg,
                  frame="icrs")
l = coords.galactic.l.deg
b = coords.galactic.b.deg

# rotate to unified axis
theta, phi = rotate_to_axis(l, b, UNIFIED_L, UNIFIED_B)

theta = np.array(theta)
phi   = np.array(phi)




# make a 2d density map (data)
bins_theta = np.linspace(0,140,60)
bins_phi   = np.linspace(0,360,120)

H, _, _, _ = binned_statistic_2d(theta, phi, None,
                                 statistic="count",
                                 bins=[bins_theta, bins_phi])
DATA = H.flatten()

def rss(model, data):
    return np.sum((data - model)**2)

def aic_k(rss, k, n):
    return 2*k + n*np.log(rss/n + 1e-12)

##########################################################
# geometric model families
##########################################################

def model_polar_cap(params, TH, PH):
    R = params[0]
    return (TH < R).astype(float)

def model_cone(params, TH, PH):
    R = params[0]
    return np.exp(-(TH/R)**2)

def model_double_cone(params, TH, PH):
    R1, R2 = params
    return np.maximum(np.exp(-(TH/R1)**2),
                      np.exp(-((140-TH)/R2)**2))

def model_torus(params, TH, PH):
    T0, sigma = params
    return np.exp(-0.5*((TH - T0)/sigma)**2)

def model_sheet(params, TH, PH):
    phi0, width = params
    dp = np.abs((PH - phi0 + 180) % 360 - 180)
    return np.exp(-0.5*(dp/width)**2)

def model_pyramid(params, TH, PH):
    m, contrast = params
    m = int(m)
    return 1 + contrast*np.cos(np.deg2rad(m*PH))

def model_poly(params, TH, PH):
    a,b,c = params
    return a + b*TH + c*(TH**2)

##########################################################
# fit function
##########################################################

def fit_model(name, func, p0, bounds=None):
    TH, PH = np.meshgrid((bins_theta[:-1]+bins_theta[1:])/2,
                         (bins_phi[:-1]  +bins_phi[1:])/2,
                         indexing="ij")
    THf = TH.flatten()
    PHf = PH.flatten()

    def obj(p):
        M = func(p, THf, PHf)
        return rss(M, DATA)

    res = minimize(obj, p0, bounds=bounds, method="L-BFGS-B")
    best_p = res.x
    best_rss = obj(best_p)
    aic = aic_k(best_rss, len(p0), len(DATA))

    return {
        "name": name,
        "params": best_p,
        "rss": best_rss,
        "aic": aic,
        "model": lambda: func(best_p, THf, PHf)
    }

##########################################################
# perform fits
##########################################################

results = []

results.append(fit_model("polar_cap",
                         model_polar_cap,
                         p0=[25.0],
                         bounds=[(1,140)]))

results.append(fit_model("single_cone",
                         model_cone,
                         p0=[30.0],
                         bounds=[(5,140)]))

results.append(fit_model("double_cone",
                         model_double_cone,
                         p0=[20.0, 40.0],
                         bounds=[(5,140),(5,140)]))

results.append(fit_model("toroid",
                         model_torus,
                         p0=[60.0, 20.0],
                         bounds=[(0,140),(1,80)]))

results.append(fit_model("sheet_ridge",
                         model_sheet,
                         p0=[250.0, 40.0],
                         bounds=[(0,360),(5,180)]))

results.append(fit_model("pyramid_mfaceted",
                         model_pyramid,
                         p0=[6,0.3],
                         bounds=[(2,20),(0,2)]))

results.append(fit_model("radial_polynomial",
                         model_poly,
                         p0=[1.0,0.01,-0.0001],
                         bounds=[(-5,5),(-1,1),(-1,1)]))

##########################################################
# ranking
##########################################################

results_sorted = sorted(results, key=lambda r: r["aic"])
best = results_sorted[0]

print("="*70)
print("FRB SHAPE GEOMETRY INFERENCE")
print("="*70)
print("\nmodel ranking by aic:")
for r in results_sorted:
    print(f"{r['name']:20s}  AIC = {r['aic']:.2f}  RSS = {r['rss']:.1f}")

print("\nbest model:", best["name"])
print("best parameters:", best["params"])

##########################################################
# scientific verdict
##########################################################

print("\n" + "="*70)
print("scientific verdict")
print("="*70)

if best["name"] == "pyramid_mfaceted":
    print("the frb sky is best described by an m-faced azimuthally modulated surface.")
    print("this agrees with: phi-structure, lobe test, axisymmetry, multipole spectrum.")
elif best["name"] == "sheet_ridge":
    print("the frb sky is best fit by a planar or ridge-like sheet structure.")
    print("this corresponds to an extended band aligned around the unified axis.")
elif best["name"] == "double_cone":
    print("the best model is a biconical shape — two opposing cones or lobes.")
elif best["name"] == "toroid":
    print("the best model is a toroidal / ring-like shape around the axis.")
elif best["name"] == "single_cone":
    print("a single cone describes the sky adequately.")
elif best["name"] == "polar_cap":
    print("a simple polar cap fit is preferred (least likely in this dataset).")
elif best["name"] == "radial_polynomial":
    print("no simple geometric shape dominates — smooth radial variation fits best.")

print("="*70)

# optional: produce a plot of data vs best model
TH, PH = np.meshgrid((bins_theta[:-1]+bins_theta[1:])/2,
                     (bins_phi[:-1]  +bins_phi[1:])/2,
                     indexing="ij")

Mbest = best["model"]().reshape(TH.shape)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("data (2d density)")
plt.imshow(H, origin="lower", aspect="auto",
           extent=[0,360,0,140], cmap="inferno")
plt.colorbar()

plt.subplot(1,2,2)
plt.title(f"best-fit model: {best['name']}")
plt.imshow(Mbest, origin="lower", aspect="auto",
           extent=[0,360,0,140], cmap="inferno")
plt.colorbar()

plt.tight_layout()
plt.savefig("frb_shape_inference.png", dpi=200)
plt.close()
