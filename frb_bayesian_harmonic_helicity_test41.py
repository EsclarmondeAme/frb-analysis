#!/usr/bin/env python3
# ======================================================================
# FRB BAYESIAN HARMONIC HELICITY EVIDENCE TEST (TEST 41)
# ======================================================================
# This test evaluates azimuthal harmonic structure (m=1 and m=2 modes)
# within the anisotropy shell. It compares:
#   M0 : isotropic (flat counts)
#   M1 : single-helix (m=1)
#   M2 : double-helix (m=1 + m=2)
#
# Likelihood is Poisson on φ-binned counts.
# Evidence is computed via dynesty nested sampling.
#
# Reinforces earlier helicity detections (test 18, test 29).
# ======================================================================

import sys
import numpy as np
import pandas as pd
import dynesty
from dynesty import NestedSampler
from numpy import cos, sin, pi

# ======================================================================
# utilities
# ======================================================================

def bin_phi(phi, nbins=18):
    """bin φ (in radians) into nbins and return (bin_centers, counts)."""
    bins = np.linspace(-np.pi, np.pi, nbins+1)
    counts, _ = np.histogram(phi, bins=bins)
    centers = 0.5 * (bins[1:] + bins[:-1])
    return centers, counts

# Likelihood – Poisson
def loglike_M0(theta, phi_counts):
    A0 = theta[0]
    if A0 <= 0:
        return -np.inf
    model = np.ones_like(phi_counts) * A0
    return np.sum(phi_counts * np.log(model) - model)

def loglike_M1(theta, phi_centers, phi_counts):
    A0, A1, phi0 = theta
    if A0 <= 0:
        return -np.inf
    model = A0 + A1 * np.cos(phi_centers - phi0)
    if np.any(model <= 0):
        return -np.inf
    return np.sum(phi_counts * np.log(model) - model)

def loglike_M2(theta, phi_centers, phi_counts):
    A0, A1, A2, phi0 = theta
    if A0 <= 0:
        return -np.inf
    model = (
        A0
        + A1 * np.cos(phi_centers - phi0)
        + A2 * np.cos(2 * (phi_centers - phi0))
    )
    if np.any(model <= 0):
        return -np.inf
    return np.sum(phi_counts * np.log(model) - model)

# Priors
def prior_transform_M0(u):
    # u in [0,1]
    A0 = 1 + 500 * u[0]  # broad
    return np.array([A0])

def prior_transform_M1(u):
    A0 = 1 + 500 * u[0]
    A1 = -200 + 400 * u[1]
    phi0 = -np.pi + 2*np.pi * u[2]
    return np.array([A0, A1, phi0])

def prior_transform_M2(u):
    A0 = 1 + 500 * u[0]
    A1 = -200 + 400 * u[1]
    A2 = -200 + 400 * u[2]
    phi0 = -np.pi + 2*np.pi * u[3]
    return np.array([A0, A1, A2, phi0])

# ======================================================================
# main
# ======================================================================

def main():
    if len(sys.argv) < 2:
        print("usage: python frb_bayesian_harmonic_helicity_test41.py frbs_unified.csv")
        sys.exit(1)

    infile = sys.argv[1]
    df = pd.read_csv(infile)
    print("detected FRBs:", len(df))

    # ==================================================================
    # apply standard unified-axis selection: θ ∈ [25°, 60°]
    # ==================================================================
    theta = df["theta_unified"].values
    phi = df["phi_unified"].values
    sel = (theta >= 25) & (theta <= 60)
    df_sel = df[sel]

    print(f"selected {len(df_sel)} FRBs in shell 25–60 deg")

    # convert φ to radians
    phi_rad = np.deg2rad(df_sel["phi_unified"].values)

    # bin φ
    centers, counts = bin_phi(phi_rad, nbins=18)

    # ==================================================================
    # run nested sampling for each model
    # ==================================================================
    print("running nested sampler for M0 (isotropic)...")
    sampler0 = NestedSampler(lambda th: loglike_M0(th, counts),
                             prior_transform_M0, ndim=1,
                             nlive=800, bound="multi", sample="rwalk")
    sampler0.run_nested()
    res0 = sampler0.results
    logZ0 = res0.logz[-1]

    print("running nested sampler for M1 (m=1 harmonic)...")
    sampler1 = NestedSampler(lambda th: loglike_M1(th, centers, counts),
                             prior_transform_M1, ndim=3,
                             nlive=800, bound="multi", sample="rwalk")
    sampler1.run_nested()
    res1 = sampler1.results
    logZ1 = res1.logz[-1]

    print("running nested sampler for M2 (m=1+m=2 harmonics)...")
    sampler2 = NestedSampler(lambda th: loglike_M2(th, centers, counts),
                             prior_transform_M2, ndim=4,
                             nlive=800, bound="multi", sample="rwalk")
    sampler2.run_nested()
    res2 = sampler2.results
    logZ2 = res2.logz[-1]

    # ==================================================================
    # summary
    # ==================================================================
    print("===================================================================")
    print(" SUMMARY – BAYESIAN HARMONIC HELICITY EVIDENCE (TEST 41)")
    print("===================================================================")
    print(f"logZ(M0 isotropic)     = {logZ0:.6f}")
    print(f"logZ(M1 m=1)           = {logZ1:.6f}")
    print(f"logZ(M2 m=1+m=2)       = {logZ2:.6f}")
    print("-------------------------------------------------------------------")
    print(f"ΔlogZ (M1 - M0)        = {logZ1 - logZ0:.6f}")
    print(f"ΔlogZ (M2 - M0)        = {logZ2 - logZ0:.6f}")
    print(f"ΔlogZ (M2 - M1)        = {logZ2 - logZ1:.6f}")
    print("-------------------------------------------------------------------")
    print("interpretation:")
    print("  - positive ΔlogZ → preference for model with harmonic structure")
    print("  - if M2 ≫ M0 and M2 > M1 → strong evidence for double-helix pattern")
    print("===================================================================")
    print("test 41 complete.")
    print("===================================================================")

# ======================================================================
if __name__ == "__main__":
    main()
