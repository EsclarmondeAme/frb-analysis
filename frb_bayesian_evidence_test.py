#!/usr/bin/env python3
# ===============================================================
# FRB COSMOLOGY – FULL BAYESIAN EVIDENCE TEST (TEST 6)
# ===============================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import det
import warnings
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings("ignore")

print("="*62)
print("                 FRB BAYESIAN EVIDENCE TEST (TEST 6)         ")
print("="*62)

# ---------------------------------------------------------------
# load catalogue
# ---------------------------------------------------------------
try:
    frb = pd.read_csv("frbs.csv")
except:
    print("ERROR: could not load frbs.csv")
    raise SystemExit

# ===============================================================
# compute unified-axis angles (θ, φ) automatically
# ===============================================================

# unified axis (galactic)
l0 = 159.85 * np.pi/180
b0 = -0.51 * np.pi/180

# convert RA/Dec → Galactic
coords = SkyCoord(ra=frb["ra"].values*u.deg,
                  dec=frb["dec"].values*u.deg,
                  frame='icrs')

gal = coords.galactic
l = gal.l.radian
b = gal.b.radian

# compute angular distance θ from unified axis
cos_theta = (np.sin(b0)*np.sin(b) +
             np.cos(b0)*np.cos(b)*np.cos(l - l0))
cos_theta = np.clip(cos_theta, -1, 1)
theta = np.arccos(cos_theta)

# compute unified-axis azimuth φ (in its tangent basis)
phi = np.arctan2(
    np.cos(b)*np.sin(l - l0),
    np.sin(b)*np.cos(b0) -
    np.cos(b)*np.sin(b0)*np.cos(l - l0)
)

# keep φ in [0, 2π)
phi = (phi + 2*np.pi) % (2*np.pi)

# binning
nbins_theta = 18
nbins_phi   = 36
H, _, _ = np.histogram2d(theta, phi,
                         bins=[nbins_theta, nbins_phi],
                         range=[[0, np.pi], [0, 2*np.pi]])

H = H.astype(float)
Ntot = H.sum()

# ---------------------------------------------------------------
# Poisson log-likelihood
# ---------------------------------------------------------------
def logL(M):
    M = np.clip(M, 1e-12, None)
    return np.sum(H * np.log(M) - M)

# ---------------------------------------------------------------
# model definitions
# ---------------------------------------------------------------

def model_warped_shell(p):
    R0, a, b, c, d, sigma = p
    phigrid = np.linspace(0, 2*np.pi, nbins_phi, endpoint=False)
    Rphi = R0 * (1 + a*np.sin(phigrid) + b*np.cos(phigrid)
                 + c*np.sin(2*phigrid) + d*np.cos(2*phigrid))
    Rphi = np.clip(Rphi, 5*np.pi/180, 175*np.pi/180)

    thgrid = np.linspace(0, np.pi, nbins_theta, endpoint=False)

    M = np.zeros((nbins_theta, nbins_phi))
    for i, th in enumerate(thgrid):
        for j, ph in enumerate(phigrid):
            M[i, j] = np.exp(-0.5 * ((th - Rphi[j]) / sigma)**2)

    M *= Ntot / M.sum()
    return M

def model_void(p):
    A, r0, s = p
    thgrid = np.linspace(0, np.pi, nbins_theta, endpoint=False)
    phgrid = np.linspace(0, 2*np.pi, nbins_phi, endpoint=False)
    M = np.zeros((nbins_theta, nbins_phi))
    for i, th in enumerate(thgrid):
        for j, ph in enumerate(phgrid):
            r = np.abs(th - r0)
            M[i, j] = A * np.exp(-r/s)
    M *= Ntot / M.sum()
    return M

def model_dipole(p):
    A, B = p
    thgrid = np.linspace(0, np.pi, nbins_theta, endpoint=False)
    phgrid = np.linspace(0, 2*np.pi, nbins_phi, endpoint=False)
    M = np.zeros((nbins_theta, nbins_phi))
    for i, th in enumerate(thgrid):
        for j, ph in enumerate(phgrid):
            M[i, j] = A * (1 + B * np.cos(th))
    M *= Ntot / M.sum()
    return M

def model_bianchi(p):
    B = p[0]
    thgrid = np.linspace(0, np.pi, nbins_theta, endpoint=False)
    phgrid = np.linspace(0, 2*np.pi, nbins_phi, endpoint=False)
    M = np.zeros((nbins_theta, nbins_phi))
    for i, th in enumerate(thgrid):
        for j, ph in enumerate(phgrid):
            M[i, j] = np.exp(B * np.cos(2*th))
    M *= Ntot / M.sum()
    return M

def model_gradient(p):
    A, g = p
    thgrid = np.linspace(0, np.pi, nbins_theta, endpoint=False)
    phgrid = np.linspace(0, 2*np.pi, nbins_phi, endpoint=False)
    M = np.zeros((nbins_theta, nbins_phi))
    for i, th in enumerate(thgrid):
        for j, ph in enumerate(phgrid):
            M[i, j] = A * (1 + g * th/np.pi)
    M *= Ntot / M.sum()
    return M

models = {
    "warped_shell": (model_warped_shell, 6,
                     np.array([40*np.pi/180, -0.2, -0.4, 0.0, 0.0, 10*np.pi/180])),
    "void":         (model_void,        3, np.array([1.0, np.pi/2, 0.5])),
    "dipole":       (model_dipole,      2, np.array([1.0, 0.3])),
    "bianchi":      (model_bianchi,     1, np.array([0.2])),
    "gradient":     (model_gradient,    2, np.array([1.0, 0.1]))
}

# ---------------------------------------------------------------
# Bayesian evidence (Laplace approx.)
# ---------------------------------------------------------------
def bayes_evidence(model_func, p0):
    res = minimize(lambda p: -logL(model_func(p)), p0, method="Nelder-Mead")
    p_best = res.x
    Lmax = logL(model_func(p_best))

    # numerical Hessian
    eps = 1e-4
    k = len(p_best)
    H = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            dp_i = np.zeros_like(p_best); dp_i[i] = eps
            dp_j = np.zeros_like(p_best); dp_j[j] = eps
            fpp = logL(model_func(p_best + dp_i + dp_j))
            fpm = logL(model_func(p_best + dp_i - dp_j))
            fmp = logL(model_func(p_best - dp_i + dp_j))
            fmm = logL(model_func(p_best - dp_i - dp_j))
            H[i, j] = (fpp - fpm - fmp + fmm) / (4*eps*eps)

    cov = -H
    try:
        detcov = det(cov)
        if detcov <= 0:
            raise ValueError
        logZ = Lmax + 0.5*k*np.log(2*np.pi) - 0.5*np.log(detcov)
    except:
        logZ = -np.inf

    return logZ

# ---------------------------------------------------------------
# run all models
# ---------------------------------------------------------------
print("computing Bayesian evidences...\n")
results = {}

for name, (func, ndim, p0) in models.items():
    print(f"  → {name} ...", end="")
    logZ = bayes_evidence(func, p0)
    results[name] = logZ
    print(f" logZ = {logZ:.4f}")

Zref = results["warped_shell"]

print("\n------------------------------------------------------------")
print(" BAYES FACTORS (relative to warped-shell model)")
print("------------------------------------------------------------")
for name in results:
    logB = results[name] - Zref
    print(f"{name:15s}  logB = {logB: .3f}")

print("------------------------------------------------------------")
print("analysis complete.")
print("="*62)
