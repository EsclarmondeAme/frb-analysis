#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time phase-memory — isotropic-sky null test (test 85I)
-------------------------------------------------------------------

goal:
    test whether the global remnant-time phase-memory statistic
    you observe in real FRBs could arise *purely from geometry*,
    given the same sample size N.

procedure:
    - load catalog to get N
    - generate N random isotropic sky positions
    - compute remnant-time signs on an isotropic sky
    - compute global phase-memory statistic using pure numpy Y_lm
    - build 2000-run shuffle null
    - compare Z_real_iso to null_mean

interpretation:
    if Z_real_iso ≈ null_mean → geometry alone produces no signal
    if Z_real_iso << null_mean → geometry *suppresses* phase memory
    if Z_real_iso >> null_mean → geometry *artificially boosts* signal

    ONLY if real sky >> isotropic sky can the signal be cosmological.
"""

import numpy as np
import csv
import sys
import math
from tqdm import tqdm


# ------------------------------------------------------------
# load catalog only for N
# ------------------------------------------------------------

def load_N(path):
    with open(path, "r", encoding="utf-8") as f:
        R = csv.reader(f)
        next(R)
        return sum(1 for _ in R)


# ------------------------------------------------------------
# galactic coordinate utilities
# ------------------------------------------------------------

def random_isotropic_xyz(N, seed=42):
    rng = np.random.RandomState(seed)
    u = rng.uniform(-1.0, 1.0, size=N)
    phi = rng.uniform(0.0, 2*np.pi, size=N)
    x = np.sqrt(1 - u*u) * np.cos(phi)
    y = np.sqrt(1 - u*u) * np.sin(phi)
    z = u
    return np.vstack([x, y, z]).T


def galactic_lb_to_xyz(l_deg, b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    v = np.array([
        np.cos(b)*np.cos(l),
        np.cos(b)*np.sin(l),
        np.sin(b)
    ])
    return v / (np.linalg.norm(v) + 1e-15)


def remnant_sign(X, axis_vec):
    axis = axis_vec / (np.linalg.norm(axis_vec) + 1e-15)
    return np.where(X @ axis > 0, 1, -1)


# ------------------------------------------------------------
# spherical harmonics (pure numpy)
# ------------------------------------------------------------

def legendre_P(l, m, x):
    m_abs = abs(m)
    Pmm = np.ones_like(x)
    if m_abs > 0:
        somx2 = np.sqrt(1 - x*x)
        fact = 1.0
        for _ in range(m_abs):
            Pmm *= -fact * somx2
            fact += 2.0
    if l == m_abs:
        return Pmm
    Pm1m = x*(2*m_abs+1)*Pmm
    if l == m_abs+1:
        return Pm1m
    Pll = None
    for ll in range(m_abs+2, l+1):
        Pll = ((2*ll-1)*x*Pm1m - (ll+m_abs-1)*Pmm)/(ll-m_abs)
        Pmm, Pm1m = Pm1m, Pll
    return Pll


def Ylm_pure(l, m, theta, phi):
    x = np.cos(theta)
    Plm = legendre_P(l, m, x)
    K = math.sqrt((2*l+1)/(4*math.pi) *
                  math.factorial(l-abs(m))/math.factorial(l+abs(m)))
    if m > 0:
        return math.sqrt(2)*K*Plm*np.cos(m*phi)
    elif m < 0:
        return math.sqrt(2)*K*Plm*np.sin(abs(m)*phi)
    else:
        return K*Plm


def compute_Ylm_matrix(X, lmax=8):
    theta = np.arccos(np.clip(X[:,2], -1.0, 1.0))
    phi = np.mod(np.arctan2(X[:,1], X[:,0]), 2*np.pi)
    Y = np.zeros((len(X), (lmax+1)*(lmax+1)))
    idx = 0
    for l in range(lmax+1):
        for m in range(-l, l+1):
            Y[:,idx] = Ylm_pure(l,m,theta,phi)
            idx += 1
    return Y


# ------------------------------------------------------------
# phase-memory statistic + null
# ------------------------------------------------------------

def phase_memory(Y, sgn):
    pos = (sgn > 0)
    neg = (sgn < 0)
    if np.sum(pos) < 2 or np.sum(neg) < 2:
        return np.nan
    A_pos = np.sum(Y[pos], axis=0)
    A_neg = np.sum(Y[neg], axis=0)
    dphi = np.abs(np.arctan2(A_pos - A_neg, np.ones_like(A_pos)))
    return np.mean(dphi)


def compute_null(Y, sgn, n_null=2000, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in tqdm(range(n_null), desc="null", leave=False):
        sh = np.array(sgn)
        rng.shuffle(sh)
        vals.append(phase_memory(Y, sh))
    return np.array(vals)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(path):
    print("[info] loading catalog for sample size…")
    N = load_N(path)
    print(f"[info] N = {N}")

    print("[info] generating isotropic sky…")
    Xiso = random_isotropic_xyz(N)

    print("[info] computing spherical harmonics Y_lm (lmax=8)…")
    Y = compute_Ylm_matrix(Xiso, lmax=8)

    print("[info] computing remnant-time signs on isotropic sky…")
    axis = galactic_lb_to_xyz(159.8, -0.5)
    sgn = remnant_sign(Xiso, axis)

    print("[info] computing real isotropic phase-memory…")
    Z_real = phase_memory(Y, sgn)

    print("[info] building null distribution (2000)…")
    null_vals = compute_null(Y, sgn, n_null=2000)
    null_mean = float(np.mean(null_vals))
    p = (1 + np.sum(null_vals >= Z_real)) / (len(null_vals) + 1)

    print("===================================================")
    print("  ISOTROPIC-SKY NULL TEST — REMNANT TIME (85I)     ")
    print("===================================================")
    print(f"Z_real_iso  = {Z_real:.6f}")
    print(f"null_mean   = {null_mean:.6f}")
    print(f"p_iso       = {p:.6f}")
    print("===================================================")
    print("interpretation:")
    print("  if geometry alone cannot reproduce your real Z,")
    print("  then the real-sky signal is cosmological.")
    print("===================================================")
    print("test 85I complete.")
    print("===================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_phase_memory_isotropic_test85I.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
