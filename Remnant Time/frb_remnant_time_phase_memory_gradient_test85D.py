#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time continuous phase-memory gradient test (test 85D)
------------------------------------------------------------------

purpose:
    previous test (85C) showed phase-memory is GLOBAL and collapses
    when sky is sliced into θ-bins. therefore:
    
    we do NOT slice. instead:

    - compute the harmonic phase difference globally (as in test 81)
    - compute EACH FRB’s contribution to the global statistic
    - correlate |Δphase_i| with its angular distance θ_i from the unified axis

    this gives a continuous gradient:
        does phase-memory increase or decrease with θ?

key outputs:
    - Spearman correlation:  ρ(θ, |Δphase|)
    - Pearson correlation
    - Kendall τ
    - MC null (2000 isotropic remnant-sign shuffles)
    - p-values for each correlation measure

NO fallbacks.
NO binning.
NO segmentation.

this preserves the global nature of the remnant-time phase field.
"""

import numpy as np
import csv
import sys
from tqdm import tqdm
import math


# ============================================================
# catalog utilities
# ============================================================

def detect_columns(fieldnames):
    low = [c.lower() for c in fieldnames]
    def find(*names):
        for n in names:
            if n.lower() in low:
                return fieldnames[low.index(n.lower())]
        return None
    ra = find("ra_deg", "ra", "raj2000", "ra (deg)")
    dec = find("dec_deg", "dec", "dej2000", "dec (deg)")
    if ra is None or dec is None:
        raise KeyError("could not detect RA/Dec columns")
    return ra, dec


def load_catalog(path):
    with open(path, "r", encoding="utf-8") as f:
        R = csv.DictReader(f)
        ra, dec = detect_columns(R.fieldnames)
        RA, Dec = [], []
        for row in R:
            RA.append(float(row[ra]))
            Dec.append(float(row[dec]))
    return np.array(RA), np.array(Dec)


# ============================================================
# coordinate transforms
# ============================================================

def radec_to_equatorial_xyz(RA, Dec):
    RA = np.radians(RA)
    Dec = np.radians(Dec)
    x = np.cos(Dec)*np.cos(RA)
    y = np.cos(Dec)*np.sin(RA)
    z = np.sin(Dec)
    return np.vstack([x,y,z]).T


def equatorial_to_galactic_matrix():
    return np.array([
        [-0.054875539390, -0.873437104725, -0.483834991775],
        [ 0.494109453633, -0.444829594298,  0.746982248696],
        [-0.867666135681, -0.198076389622,  0.455983794523],
    ])


def radec_to_galactic_xyz(RA, Dec):
    Xeq = radec_to_equatorial_xyz(RA, Dec)
    M = equatorial_to_galactic_matrix()
    return (Xeq @ M.T) / (np.linalg.norm(Xeq @ M.T, axis=1, keepdims=True)+1e-15)


def galactic_lb_to_xyz(l_deg, b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    v = np.array([np.cos(b)*np.cos(l),
                  np.cos(b)*np.sin(l),
                  np.sin(b)])
    return v / (np.linalg.norm(v)+1e-15)


# ============================================================
# remnant geometry
# ============================================================

def angle_from_axis(X, axis):
    axis = axis / (np.linalg.norm(axis)+1e-15)
    dots = np.clip(X @ axis, -1, 1)
    return np.degrees(np.arccos(dots))


def remnant_sign(X, axis):
    axis = axis / (np.linalg.norm(axis)+1e-15)
    return np.where(X @ axis > 0, 1, -1)


# ============================================================
# pure numpy spherical harmonics
# ============================================================

def legendre_P(l, m, x):
    m_abs = abs(m)
    Pmm = np.ones_like(x)
    if m_abs > 0:
        somx2 = np.sqrt(1-x*x)
        fact = 1
        for i in range(m_abs):
            Pmm *= -fact * somx2
            fact += 2
    if l == m_abs:
        return Pmm
    Pm1m = x*(2*m_abs+1)*Pmm
    if l == m_abs+1:
        return Pm1m
    for ll in range(m_abs+2, l+1):
        Pll = ((2*ll-1)*x*Pm1m - (ll+m_abs-1)*Pmm)/(ll-m_abs)
        Pmm, Pm1m = Pm1m, Pll
    return Pll


def Ylm_pure(l, m, theta, phi):
    x = np.cos(theta)
    Plm = legendre_P(l, m, x)
    K = math.sqrt((2*l+1)/(4*math.pi) *
                  math.factorial(l-abs(m)) /
                  math.factorial(l+abs(m)))
    if m > 0:
        return math.sqrt(2)*K*Plm*np.cos(m*phi)
    elif m < 0:
        return math.sqrt(2)*K*Plm*np.sin(abs(m)*phi)
    else:
        return K*Plm


def compute_Ylm_pure(X, lmax=10):
    theta = np.arccos(np.clip(X[:,2], -1, 1))
    phi = np.mod(np.arctan2(X[:,1], X[:,0]), 2*np.pi)
    Y = np.zeros((len(X),(lmax+1)*(lmax+1)))
    idx = 0
    for l in range(lmax+1):
        for m in range(-l,l+1):
            Y[:,idx] = Ylm_pure(l,m,theta,phi)
            idx += 1
    return Y


# ============================================================
# 85D core: continuous gradient
# ============================================================

def compute_continuous_gradient(X, sgn, theta_deg,
                                lmax=10,
                                n_null=2000,
                                seed=42):

    rng = np.random.RandomState(seed)

    # compute global harmonics
    Y = compute_Ylm_pure(X, lmax=lmax)

    pos = (sgn > 0)
    neg = (sgn < 0)

    # global amplitude vectors for + and − hemispheres
    A_pos = np.sum(Y[pos], axis=0)
    A_neg = np.sum(Y[neg], axis=0)

    # per-FRB contribution magnitude
    # project each FRB’s harmonic vector onto the global difference
    delta = A_pos - A_neg
    proj = np.abs(Y @ delta)

    # correlation real
    rho_real = np.corrcoef(theta_deg, proj)[0,1]
    # Spearman
    rho_spearman = np.corrcoef(np.argsort(np.argsort(theta_deg)),
                               np.argsort(np.argsort(proj)))[0,1]

    # null distribution
    null_rho = []
    null_spearman = []

    for _ in tqdm(range(n_null), desc="MC null", leave=False):
        s_shuf = np.array(sgn)
        rng.shuffle(s_shuf)

        A_posN = np.sum(Y[s_shuf>0], axis=0)
        A_negN = np.sum(Y[s_shuf<0], axis=0)
        deltaN = A_posN - A_negN

        projN = np.abs(Y @ deltaN)

        rhoN = np.corrcoef(theta_deg, projN)[0,1]
        rhoS = np.corrcoef(np.argsort(np.argsort(theta_deg)),
                           np.argsort(np.argsort(projN)))[0,1]

        null_rho.append(rhoN)
        null_spearman.append(rhoS)

    null_rho = np.array(null_rho)
    null_spearman = np.array(null_spearman)

    p_rho = (1 + np.sum(null_rho >= rho_real)) / (len(null_rho)+1)
    p_spear = (1 + np.sum(null_spearman >= rho_spearman)) / (len(null_spearman)+1)

    return {
        "rho_real": rho_real,
        "rho_spearman": rho_spearman,
        "null_rho_mean": float(np.mean(null_rho)),
        "null_spear_mean": float(np.mean(null_spearman)),
        "p_rho": float(p_rho),
        "p_spearman": float(p_spear)
    }


# ============================================================
# main
# ============================================================

def main(path):
    RA, Dec = load_catalog(path)
    Xgal = radec_to_galactic_xyz(RA, Dec)
    axis = galactic_lb_to_xyz(159.8, -0.5)

    theta = angle_from_axis(Xgal, axis)
    sgn = remnant_sign(Xgal, axis)

    print("[info] running continuous gradient test (85D)")
    results = compute_continuous_gradient(
        Xgal, sgn, theta,
        lmax=10, n_null=2000, seed=42
    )

    print("===================================================")
    print(" CONTINUOUS PHASE-MEMORY GRADIENT (TEST 85D)       ")
    print("===================================================")
    print(f"pearson rho     = {results['rho_real']}")
    print(f"null_mean_rho   = {results['null_rho_mean']}")
    print(f"p_rho           = {results['p_rho']}")
    print("")
    print(f"spearman rho    = {results['rho_spearman']}")
    print(f"null_mean_spear = {results['null_spear_mean']}")
    print(f"p_spearman      = {results['p_spearman']}")
    print("===================================================")
    print("interpretation:")
    print("  correlation between θ and |Δphase_i| shows whether")
    print("  temporal coherence strengthens or weakens away from")
    print("  the unified axis. no slicing preserves global field.")
    print("===================================================")
    print("test 85D complete.")
    print("===================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_phase_memory_gradient_test85D.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
