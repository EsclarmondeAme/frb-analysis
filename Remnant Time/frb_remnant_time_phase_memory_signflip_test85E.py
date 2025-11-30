#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time phase-memory — sign-inversion robustness (test 85E)
--------------------------------------------------------------------

goal:
    check whether the global phase-memory signal from tests 81/85D
    reverses sign when remnant-time signs are inverted:

        sgn → -sgn

    this tests:
        - antisymmetry (as predicted by remnant-time model)
        - global coherence
        - nonlocality
        - physical directionality of the hemispheres

expected:
    Z_real_signed  should be significantly different from null
    Z_real_flipped should have similar magnitude but opposite sign
    null distributions should not reproduce antisymmetry

design:
    - pure numpy spherical harmonics
    - full-sky global estimator (no slicing)
    - lmax = 10
    - MC null = 2000

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
    ra = find("ra_deg","ra","raj2000","ra (deg)")
    dec = find("dec_deg","dec","dej2000","dec (deg)")
    if ra is None or dec is None:
        raise KeyError("could not detect RA/Dec columns")
    return ra,dec


def load_catalog(path):
    with open(path,"r",encoding="utf-8") as f:
        R = csv.DictReader(f)
        ra, dec = detect_columns(R.fieldnames)
        RA,Dec = [],[]
        for row in R:
            RA.append(float(row[ra]))
            Dec.append(float(row[dec]))
    return np.array(RA),np.array(Dec)


# ============================================================
# coordinate transforms
# ============================================================

def radec_to_equatorial_xyz(RA,Dec):
    RA = np.radians(RA)
    Dec = np.radians(Dec)
    x = np.cos(Dec)*np.cos(RA)
    y = np.cos(Dec)*np.sin(RA)
    z = np.sin(Dec)
    return np.vstack([x,y,z]).T


def equatorial_to_galactic_matrix():
    return np.array([
        [-0.054875539390,-0.873437104725,-0.483834991775],
        [ 0.494109453633,-0.444829594298, 0.746982248696],
        [-0.867666135681,-0.198076389622, 0.455983794523]
    ])


def radec_to_galactic_xyz(RA,Dec):
    Xeq = radec_to_equatorial_xyz(RA,Dec)
    M = equatorial_to_galactic_matrix()
    Xgal = Xeq @ M.T
    return Xgal/ (np.linalg.norm(Xgal,axis=1,keepdims=True)+1e-15)


def galactic_lb_to_xyz(l_deg,b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    v = np.array([
        np.cos(b)*np.cos(l),
        np.cos(b)*np.sin(l),
        np.sin(b)
    ])
    return v/ (np.linalg.norm(v)+1e-15)


# ============================================================
# geometry
# ============================================================

def angle_from_axis(X,axis):
    axis = axis/ (np.linalg.norm(axis)+1e-15)
    dots = np.clip(X@axis, -1, 1)
    return np.degrees(np.arccos(dots))


def remnant_sign(X,axis):
    axis = axis/ (np.linalg.norm(axis)+1e-15)
    return np.where(X@axis>0, 1, -1)


# ============================================================
# spherical harmonics (pure numpy)
# ============================================================

def legendre_P(l,m,x):
    m_abs = abs(m)
    Pmm = np.ones_like(x)
    if m_abs>0:
        somx = np.sqrt(1-x*x)
        fact = 1
        for i in range(m_abs):
            Pmm *= -fact*somx
            fact += 2
    if l==m_abs:
        return Pmm
    Pm1m = x*(2*m_abs+1)*Pmm
    if l==m_abs+1:
        return Pm1m
    for ll in range(m_abs+2, l+1):
        Pll = ((2*ll-1)*x*Pm1m - (ll+m_abs-1)*Pmm)/(ll-m_abs)
        Pmm, Pm1m = Pm1m, Pll
    return Pll


def Ylm_pure(l,m,theta,phi):
    x = np.cos(theta)
    Plm = legendre_P(l,m,x)
    K = math.sqrt((2*l+1)/(4*math.pi) *
                  math.factorial(l-abs(m)) /
                  math.factorial(l+abs(m)))
    if m>0:
        return math.sqrt(2)*K*Plm*np.cos(m*phi)
    elif m<0:
        return math.sqrt(2)*K*Plm*np.sin(abs(m)*phi)
    else:
        return K*Plm


def compute_Ylm_pure(X,lmax=10):
    theta = np.arccos(np.clip(X[:,2], -1, 1))
    phi   = np.mod(np.arctan2(X[:,1],X[:,0]), 2*math.pi)
    Y = np.zeros((len(X),(lmax+1)*(lmax+1)))
    idx=0
    for l in range(lmax+1):
        for m in range(-l,l+1):
            Y[:,idx] = Ylm_pure(l,m,theta,phi)
            idx+=1
    return Y


# ============================================================
# phase-memory statistic (global)
# ============================================================

def global_phase_memory(X,sgn,lmax=10):
    Y = compute_Ylm_pure(X,lmax)
    pos = (sgn>0)
    neg = (sgn<0)
    if np.sum(pos)<2 or np.sum(neg)<2:
        return np.nan
    A_pos = np.sum(Y[pos],axis=0)
    A_neg = np.sum(Y[neg],axis=0)
    dphi  = np.abs(np.arctan2(A_pos - A_neg, np.ones_like(A_pos)))
    return np.mean(dphi)


# ============================================================
# main test function
# ============================================================

def compute_85E(X,sgn,lmax=10,n_null=2000,seed=42):
    rng = np.random.RandomState(seed)

    print("[info] computing global real sign-case…")
    Z_real = global_phase_memory(X,sgn,lmax)

    print("[info] computing global flipped sign-case…")
    Z_flip = global_phase_memory(X,-sgn,lmax)

    # null test
    null_real = []
    null_flip = []

    print("[info] building null distribution (2000)…")
    for _ in tqdm(range(n_null), desc="MC null", leave=False):
        s_shuf = np.array(sgn)
        rng.shuffle(s_shuf)

        null_real.append(
            global_phase_memory(X,s_shuf,lmax)
        )
        null_flip.append(
            global_phase_memory(X,-s_shuf,lmax)
        )

    null_real = np.array(null_real)
    null_flip = np.array(null_flip)

    # p-values
    p_real = (1+np.sum(null_real >= Z_real))/(len(null_real)+1)
    p_flip = (1+np.sum(null_flip >= Z_flip))/(len(null_flip)+1)

    return {
        "Z_real": float(Z_real),
        "Z_flip": float(Z_flip),
        "null_mean_real": float(np.mean(null_real)),
        "null_mean_flip": float(np.mean(null_flip)),
        "p_real": float(p_real),
        "p_flip": float(p_flip),
    }


# ============================================================
# main
# ============================================================

def main(path):
    RA,Dec = load_catalog(path)
    Xgal = radec_to_galactic_xyz(RA,Dec)
    axis = galactic_lb_to_xyz(159.8,-0.5)

    sgn = remnant_sign(Xgal,axis)

    print("[info] running test 85E (sign inversion robustness)…")
    res = compute_85E(Xgal,sgn,lmax=10,n_null=2000,seed=42)

    print("=====================================================")
    print(" REMNANT-TIME PHASE-MEMORY SIGN INVERSION (TEST 85E) ")
    print("=====================================================")
    print(f"Z_real      = {res['Z_real']}")
    print(f"null_mean   = {res['null_mean_real']}")
    print(f"p_real      = {res['p_real']}")
    print("")
    print(f"Z_flip      = {res['Z_flip']}")
    print(f"null_mean_f = {res['null_mean_flip']}")
    print(f"p_flip      = {res['p_flip']}")
    print("=====================================================")
    print("interpretation:")
    print("  Z_real should be significant (as in 81/85D).")
    print("  Z_flip should have similar magnitude but reversed sign-effect.")
    print("  null distributions should NOT show antisymmetry.")
    print("=====================================================")
    print("test 85E complete.")
    print("=====================================================")


if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_phase_memory_signflip_test85E.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
