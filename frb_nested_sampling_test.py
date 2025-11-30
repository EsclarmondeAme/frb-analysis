#!/usr/bin/env python3
# ==============================================================
# FRB NESTED-SAMPLING EVIDENCE TEST (TEST 7)
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# try dynesty first, fallback to ultranest
try:
    import dynesty
    from dynesty import NestedSampler
    USING_DYNNESTY = True
except:
    from ultranest import ReactiveNestedSampler
    USING_DYNNESTY = False

# --------------------------------------------------------------
# load data
# --------------------------------------------------------------
print("==============================================================")
print("          FRB NESTED-SAMPLING BAYES EVIDENCE (TEST 7)         ")
print("==============================================================")

try:
    frb = pd.read_csv("frbs_unified.csv")
except:
    print("ERROR: could not load frbs_unified.csv")
    exit()

# theta_unified must be present (in degrees)
if "theta_unified" not in frb.columns:
    print("ERROR: missing column 'theta_unified'.")
    print("run unified-axis script first.")
    exit()

theta = np.radians(frb["theta_unified"].values)
phi   = np.radians(frb["phi_unified"].values)

# --------------------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------------------

def model_isotropic(th):
    return np.full_like(th, 1/(4*np.pi))

def model_dipole(th, params):
    A = params[0]     # dipole amplitude 0–1
    return (1 + A*np.cos(th)) / (4*np.pi)

def model_shell(th, ph, params):
    # R(phi) = R0 [1 + a sinφ + b cosφ + c sin2φ + d cos2φ]
    R0, a, b, c, d, sigma = params
    Rphi = R0 * (1 + a*np.sin(ph) + b*np.cos(ph) +
                 c*np.sin(2*ph) + d*np.cos(2*ph))
    return np.exp(-0.5*((th - Rphi)/sigma)**2) / (np.sqrt(2*np.pi)*sigma)


# --------------------------------------------------------------
# priors (unit cube -> parameters)
# --------------------------------------------------------------

def prior_shell(u):
    R0    = 0 + u[0]*np.pi/2         # 0–90deg
    a     = -1 + 2*u[1]
    b     = -1 + 2*u[2]
    c     = -1 + 2*u[3]
    d     = -1 + 2*u[4]
    sigma = np.radians(3 + 20*u[5])  # 3–23 deg
    return np.array([R0, a, b, c, d, sigma])

def prior_dipole(u):
    A = 0 + 1*u[0]
    return np.array([A])

# --------------------------------------------------------------
# likelihood
# --------------------------------------------------------------

def loglike_shell(params):
    p = model_shell(theta, phi, params)
    # avoid zero
    p = np.clip(p, 1e-300, None)
    return np.sum(np.log(p))

def loglike_dipole(params):
    p = model_dipole(theta, params)
    p = np.clip(p, 1e-300, None)
    return np.sum(np.log(p))

def loglike_iso(dummy):
    p = model_isotropic(theta)
    return np.sum(np.log(p))


# --------------------------------------------------------------
# RUN NESTED SAMPLING
# --------------------------------------------------------------

def run_dynesty(loglike_fn, prior_fn, ndim, label):
    print(f"\n→ running dynesty nested sampler for: {label}")
    sampler = NestedSampler(loglike_fn, prior_fn, ndim, nlive=500)
    sampler.run_nested(dlogz=0.1)
    res = sampler.results
    logZ = res.logz[-1]
    logZerr = res.logzerr[-1]
    print(f"{label}: logZ = {logZ:.4f} ± {logZerr:.4f}")
    return logZ, logZerr

def run_ultranest(loglike_fn, prior_fn, ndim, label):
    print(f"\n→ running ultranest nested sampler for: {label}")
    sampler = ReactiveNestedSampler(
        label,
        lambda params: np.exp(loglike_fn(params)),
        prior_fn
    )
    result = sampler.run()
    logZ = result["logz"]
    logZerr = result["logzerr"]
    print(f"{label}: logZ = {logZ:.4f} ± {logZerr:.4f}")
    sampler.plot()
    return logZ, logZerr


# wrapper
def run_model(loglike_fn, prior_fn, ndim, label):
    if USING_DYNNESTY:
        return run_dynesty(loglike_fn, prior_fn, ndim, label)
    else:
        return run_ultranest(loglike_fn, prior_fn, ndim, label)


# --------------------------------------------------------------
# run evidence for three core models
# --------------------------------------------------------------

logZ_iso,    dZ1 = run_model(loglike_iso,    lambda u: [], 0, "isotropic")
logZ_dip,    dZ2 = run_model(loglike_dipole, prior_dipole, 1, "dipole")
logZ_shell,  dZ3 = run_model(loglike_shell,  prior_shell,  6, "warped_shell")

# --------------------------------------------------------------
# compute bayes factors
# --------------------------------------------------------------

print("\n------------------------------------------------------------")
print("Bayes factors (relative to warped-shell model)")
print("------------------------------------------------------------")

print(f"isotropic     logB = {logZ_iso   - logZ_shell:.3f}")
print(f"dipole        logB = {logZ_dip   - logZ_shell:.3f}")
print(f"warped_shell  logB = 0 (baseline)")

print("------------------------------------------------------------")
print("analysis complete.")
print("==============================================================")
