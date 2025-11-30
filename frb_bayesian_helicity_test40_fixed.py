#!/usr/bin/env python3
import numpy as np
import pandas as pd
from dynesty import NestedSampler
from scipy.special import i0
import sys

# ------------------------------------------------------------
# circular von Mises log-likelihood
# ------------------------------------------------------------

def vonmises_logpdf(phi, mu, kappa):
    """
    circular likelihood for angles in radians.
    """
    return kappa * np.cos(phi - mu) - np.log(2*np.pi*i0(kappa))


# ------------------------------------------------------------
# model definitions
# ------------------------------------------------------------

def model_M0(params, theta, z):
    """
    isotropic model: phi ~ uniform
    represented by von Mises with kappa=0 (flat)
    but to keep it comparable, we treat M0 as:
    mu = free constant, kappa = free concentration
    """
    mu = params[0]
    kappa = np.abs(params[1])
    return mu, kappa


def model_M1(params, theta, z):
    """
    helicity model:
    phi_pred = phi0 + k * theta + k_z * z
    """
    phi0 = params[0]
    k = params[1]
    k_z = params[2]
    kappa = np.abs(params[3])

    phi_pred = phi0 + k * theta + k_z * z
    return phi_pred, kappa


# ------------------------------------------------------------
# log-likelihood wrappers
# ------------------------------------------------------------

def loglike_M0(params, phi, theta, z):
    mu, kappa = model_M0(params, theta, z)
    ll = np.sum(vonmises_logpdf(phi, mu, kappa))
    return ll


def loglike_M1(params, phi, theta, z):
    phi_pred, kappa = model_M1(params, theta, z)
    ll = np.sum(vonmises_logpdf(phi, phi_pred, kappa))
    return ll


# ------------------------------------------------------------
# priors
# ------------------------------------------------------------

def prior_transform_M0(u):
    """
    u in [0,1]^2
    params = [mu, kappa]
    mu in [0, 2π]
    kappa in [0, 50]
    """
    mu = u[0] * 2*np.pi
    kappa = u[1] * 50
    return np.array([mu, kappa])


def prior_transform_M1(u):
    """
    u in [0,1]^4
    params = [phi0, k, k_z, kappa]
    phi0 in [0,2π]
    k in [-1,1]
    k_z in [-2,2]
    kappa in [0,50]
    """
    phi0 = u[0] * 2*np.pi
    k = -1 + 2*u[1]
    k_z = -2 + 4*u[2]
    kappa = u[3] * 50
    return np.array([phi0, k, k_z, kappa])


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("usage: python frb_bayesian_helicity_test40_fixed.py frbs_unified.csv")
        sys.exit()

    df = pd.read_csv(sys.argv[1])
    print(f"detected FRBs: {len(df)}")

    # convert to radians
    theta = np.deg2rad(df["theta_unified"].values)
    phi   = np.deg2rad(df["phi_unified"].values)

    z = df["z_est"].values

  


    # restrict to shell 25°–60°
    shell = (df["theta_unified"] >= 25) & (df["theta_unified"] <= 60)

    theta = theta[shell]
    phi = phi[shell]
    z = z[shell]

    print(f"selected {len(theta)} FRBs in shell 25–60 deg")

    # run M0
    print("running nested sampler for M0 (isotropic-with-kappa)...")
    sampler0 = NestedSampler(lambda p: loglike_M0(p, phi, theta, z),
                             prior_transform_M0, 2, nlive=400)
    sampler0.run_nested()
    res0 = sampler0.results
    logZ0 = res0.logz[-1]

    print("running nested sampler for M1 (helicity)...")
    sampler1 = NestedSampler(lambda p: loglike_M1(p, phi, theta, z),
                             prior_transform_M1, 4, nlive=600)
    sampler1.run_nested()
    res1 = sampler1.results
    logZ1 = res1.logz[-1]

    # summary
    print("===================================================================")
    print("SUMMARY – FIXED BAYESIAN HELICITY EVIDENCE (TEST 40)")
    print("===================================================================")
    print(f"logZ(M0) = {logZ0:.6f}")
    print(f"logZ(M1) = {logZ1:.6f}")
    print(f"ΔlogZ (M1 - M0) = {logZ1 - logZ0:.6f}")

    if logZ1 - logZ0 > 0:
        print("interpretation: helicity slightly/moderately favored")
    else:
        print("interpretation: no strong preference for helicity")
    print("===================================================================")


if __name__ == "__main__":
    main()
