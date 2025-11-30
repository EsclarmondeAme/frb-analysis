#!/usr/bin/env python3
# ================================================================
# FRB COSMOLOGY MODEL FIT — MODEL COMPARISON PIPELINE
# ================================================================
# Tests 5 cosmological scenarios for the warped FRB shell:
#   1. off-center void
#   2. anisotropic expansion (Bianchi-like axis)
#   3. primordial gradient (superhorizon potential mode)
#   4. dipole-modulated progenitor density
#   5. triaxial warped shell embedded in LCDM
#
# Outputs:
#   - AIC ranking
#   - best-fit parameters for each model
#   - 4-panel figure: data, model, residuals, likelihood
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import binned_statistic
from astropy.coordinates import SkyCoord
import astropy.units as u

# ================================================================
# LOAD DATA
# ================================================================
fname = "frbs.csv"
try:
    frb = pd.read_csv(fname)
except:
    raise SystemExit(f"ERROR: could not read {fname}")

frb = frb.dropna(subset=["ra", "dec"])
ra  = np.asarray(frb["ra"])
dec = np.asarray(frb["dec"])

# unified axis (fixed)
L0 = np.radians(159.85)
B0 = np.radians(-0.51)

coords = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs").galactic
l = coords.l.radian
b = coords.b.radian

# compute angular separation from unified axis
def angsep(l1,b1,l2,b2):
    return np.arccos(
        np.sin(b1)*np.sin(b2) +
        np.cos(b1)*np.cos(b2)*np.cos(l1-l2)
    )

theta = angsep(l,b,L0,B0)  # radians
phi   = (l - L0 + 2*np.pi) % (2*np.pi)

# ================================================================
# BINNING IN AXIS FRAME
# ================================================================
N_bins = 60
theta_bins = np.linspace(0, np.radians(90), N_bins+1)
phi_bins   = np.linspace(0, 2*np.pi, N_bins+1)

H_data, _, _ = np.histogram2d(
    theta, phi,
    bins=[theta_bins, phi_bins]
)

# ------------------------------------------------
# AIC helper
# ------------------------------------------------
def AIC(logL, k):
    return 2*k - 2*logL

# ================================================================
# MODEL 1 — OFF-CENTER VOID
# density = 1 + A * exp(-(theta - R)^2 / (2σ^2))
# ================================================================
def model_void(params):
    A, R, sigma = params
    Th, Ph = np.meshgrid(
        0.5*(theta_bins[1:]+theta_bins[:-1]),
        0.5*(phi_bins[1:]+phi_bins[:-1]),
        indexing="ij"
    )
    return 1 + A * np.exp(-(Th-R)**2/(2*sigma**2))

def logL(H,M):
    M = M/np.sum(M) * np.sum(H) + 1e-9
    return np.sum(H * np.log(M))

def fit_void():
    def negL(p):
        M = model_void(p)
        return -logL(H_data, M)
    res = minimize(negL, [0.5, np.radians(25), np.radians(10)],
                   bounds=[(0,5),(0,1.5),(0.01,1)])
    p = res.x
    M = model_void(p)
    return p, logL(H_data,M), 3, M

# ================================================================
# MODEL 2 — ANISOTROPIC EXPANSION (BIANCHI)
# density ∝ 1 + A*cos(theta)
# ================================================================
def model_bianchi(params):
    A = params[0]
    Th, Ph = np.meshgrid(
        0.5*(theta_bins[1:]+theta_bins[:-1]),
        0.5*(phi_bins[1:]+phi_bins[:-1]),
        indexing="ij"
    )
    return 1 + A*np.cos(Th)

def fit_bianchi():
    def negL(p):
        M = model_bianchi(p)
        return -logL(H_data,M)
    res = minimize(negL, [0.2], bounds=[(-2,2)])
    p = res.x
    M = model_bianchi(p)
    return p, logL(H_data,M), 1, M

# ================================================================
# MODEL 3 — PRIMORDIAL GRADIENT
# density ∝ 1 + A*cos(theta) + B*cos(2*theta)
# ================================================================
def model_grad(params):
    A, B = params
    Th, Ph = np.meshgrid(
        0.5*(theta_bins[1:]+theta_bins[:-1]),
        0.5*(phi_bins[1:]+phi_bins[:-1]),
        indexing="ij"
    )
    return 1 + A*np.cos(Th) + B*np.cos(2*Th)

def fit_grad():
    def negL(p):
        return -logL(H_data, model_grad(p))
    res = minimize(negL, [0.3,0.1], bounds=[(-2,2),(-2,2)])
    p = res.x
    M = model_grad(p)
    return p, logL(H_data,M), 2, M

# ================================================================
# MODEL 4 — DIPOLE-MODULATED PROGENITOR DENSITY
# density ∝ 1 + A*cos(phi - φ0)
# ================================================================
def model_dipole(params):
    A, phi0 = params
    Th, Ph = np.meshgrid(
        0.5*(theta_bins[1:]+theta_bins[:-1]),
        0.5*(phi_bins[1:]+phi_bins[:-1]),
        indexing="ij"
    )
    return 1 + A * np.cos(Ph - phi0)

def fit_dipole():
    def negL(p):
        return -logL(H_data, model_dipole(p))
    res = minimize(negL, [0.3,1.0], bounds=[(-2,2),(0,2*np.pi)])
    p = res.x
    M = model_dipole(p)
    return p, logL(H_data,M), 2, M

# ================================================================
# MODEL 5 — TRIAXIAL WARPED SHELL
# density ∝ exp(-((theta - R(φ))²)/(2σ²))
# R(φ) = R0*[1 + a sin φ + b cos φ + c sin 2φ + d cos 2φ]
# ================================================================
def model_shell(params):
    R0, sigma, a,b,c,d = params
    Th, Ph = np.meshgrid(
        0.5*(theta_bins[1:]+theta_bins[:-1]),
        0.5*(phi_bins[1:]+phi_bins[:-1]),
        indexing="ij"
    )
    Rphi = R0*(1 + a*np.sin(Ph) + b*np.cos(Ph) +
                  c*np.sin(2*Ph) + d*np.cos(2*Ph))
    return np.exp(-(Th - Rphi)**2/(2*sigma**2))

def fit_shell():
    def negL(p):
        M = model_shell(p)
        return -logL(H_data,M)
    p0 = [np.radians(40), np.radians(12), 0.2, -0.5, 0.1, 0.05]
    bounds = [
        (0,1.5),(0.01,1),
        (-1,1),(-1,1),(-1,1),(-1,1)
    ]
    res = minimize(negL, p0, bounds=bounds)
    p = res.x
    M = model_shell(p)
    return p, logL(H_data,M), 6, M

# ================================================================
# RUN ALL MODELS
# ================================================================
fits = []
names = ["void","bianchi","gradient","dipole","shell"]

for f in [fit_void, fit_bianchi, fit_grad, fit_dipole, fit_shell]:
    p,L,k,M = f()
    fits.append((p,L,k,M))

AICs = [AIC(L,k) for (_,L,k,_) in fits]
ranking = np.argsort(AICs)

print("=======================================================")
print(" COSMOLOGY MODEL COMPARISON")
print("=======================================================")
for rank,i in enumerate(ranking):
    name = names[i]
    p,L,k,M = fits[i]
    print(f"{rank+1}. {name:10s}   AIC = {AICs[i]:.2f}   params = {p}")

best = ranking[0]
print("-------------------------------------------------------")
print(f"best model: {names[best]}")
print("-------------------------------------------------------")

# ================================================================
# FIGURE (data | best model | residual | theta histogram)
# ================================================================
best_M = fits[best][3]

plt.figure(figsize=(12,10))
plt.suptitle(f"FRB cosmology model fit — best model: {names[best]}")

plt.subplot(2,2,1)
plt.imshow(H_data, origin="lower", aspect="auto")
plt.title("data (θ,φ)")
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(best_M, origin="lower", aspect="auto")
plt.title(f"model: {names[best]}")
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(H_data - best_M/np.sum(best_M)*np.sum(H_data),
           origin="lower", aspect="auto")
plt.title("residual")

plt.subplot(2,2,4)
plt.hist(theta, bins=40)
plt.title("θ histogram (data)")

plt.tight_layout()
plt.savefig("frb_cosmology_model_fit.png", dpi=200)
print("saved plot: frb_cosmology_model_fit.png")
print("analysis complete.")
