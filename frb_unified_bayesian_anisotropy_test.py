import numpy as np
import pandas as pd
from numpy import sin, cos
from dynesty import NestedSampler
import sys

print("="*70)
print(" FRB UNIFIED BAYESIAN ANISOTROPY TEST (TEST 8)")
print("="*70)

# ------------------------------------------------------------
# load data
# ------------------------------------------------------------
if len(sys.argv) < 2:
    print("usage: python frb_unified_bayesian_anisotropy_test.py frbs_unified.csv")
    sys.exit(1)

fname = sys.argv[1]
print(f"loading {fname} ...")

frb = pd.read_csv(fname)

if "theta_unified" not in frb.columns or "phi_unified" not in frb.columns:
    print("ERROR: missing theta_unified or phi_unified columns.")
    sys.exit(1)

theta = np.radians(frb["theta_unified"].values)
phi   = np.radians(frb["phi_unified"].values)
N     = len(theta)

# ------------------------------------------------------------
# 7-parameter warped shell model
# params = [R0, sigma, amp, a1, b1, a2, b2]
# ------------------------------------------------------------
def warped_shell(theta, phi, params):
    R0, sigma, amp, a1, b1, a2, b2 = params
    
    # warped radius
    Rphi = R0 * (
        1
        + a1 * np.sin(phi) 
        + b1 * np.cos(phi)
        + a2 * np.sin(2*phi)
        + b2 * np.cos(2*phi)
    )
    
    # gaussian radial shell
    arg = (theta - Rphi)**2 / (2 * sigma**2)
    M = amp * np.exp(-arg)
    
    return M + 1e-50   # avoid log(0)

# ------------------------------------------------------------
# log-likelihood for dynesty
# ------------------------------------------------------------
def loglike(params):
    M = warped_shell(theta, phi, params)
    return np.sum(np.log(M))


# ------------------------------------------------------------
# prior transform
# params = [R0, sigma, amp, a1, b1, a2, b2]
# ------------------------------------------------------------
def prior_transform(u):
    R0    = 10 + 70*u[0]        # [10°, 80°]
    sigma =  3 + 20*u[1]        # [3°, 23°]
    amp   =  0.1 + 2*u[2]       # [0.1, 2.1]
    a1    = -1 + 2*u[3]         # [-1, +1]
    b1    = -1 + 2*u[4]         # [-1, +1]
    a2    = -1 + 2*u[5]         # [-1, +1]
    b2    = -1 + 2*u[6]         # [-1, +1]
    return np.array([R0, sigma, amp, a1, b1, a2, b2])


# ------------------------------------------------------------
# run nested sampling
# ------------------------------------------------------------
ndim = 7

print("\n→ running nested sampler on unified warped-shell model ...")

sampler = NestedSampler(loglike, prior_transform, ndim, nlive=500)
sampler.run_nested(dlogz=0.1)
res = sampler.results

logZ  = res.logz[-1]
errZ  = res.logzerr[-1]

print("------------------------------------------------------------")
print(f"Bayesian evidence logZ = {logZ:.6f} ± {errZ:.6f}")
print("------------------------------------------------------------")

print("analysis complete.")
print("="*70)
