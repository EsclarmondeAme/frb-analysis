#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time phase-memory — axis wobble robustness (test 85F)
--------------------------------------------------------------------

goal:
    test robustness of the global remnant-time phase-memory signal
    to small perturbations ("wobbles") of the unified axis.

    the unified axis used so far:
        l = 159.8°,  b = -0.5°

    we wobble it by:
        ±3°, ±5°, ±10°
    in both longitude and latitude directions.

for each perturbed axis we compute:
    - remnant sign
    - global harmonic phase-memory (pure numpy)
    - 2000-run null distribution
    - p-value

expected:
    - small wobbles (3°, 5°) preserve signal strength
    - large wobble (10°) weakens signal but does not produce a false positive
    - null skies do NOT show structured stability

this mirrors the robustness logic of the unified-axis tests.
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
    return Xgal/(np.linalg.norm(Xgal,axis=1,keepdims=True)+1e-15)


def galactic_lb_to_xyz(l_deg,b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    v = np.array([
        np.cos(b)*np.cos(l),
        np.cos(b)*np.sin(l),
        np.sin(b)
    ])
    return v/(np.linalg.norm(v)+1e-15)


# ============================================================
# geometry
# ============================================================

def remnant_sign(X,axis):
    axis = axis/(np.linalg.norm(axis)+1e-15)
    return np.where(X@axis>0,1,-1)


# ============================================================
# pure numpy spherical harmonics
# ============================================================

def legendre_P(l,m,x):
    m_abs = abs(m)
    Pmm = np.ones_like(x)
    if m_abs>0:
        s = np.sqrt(1-x*x)
        fact = 1
        for i in range(m_abs):
            Pmm *= -fact*s
            fact += 2
    if l==m_abs:
        return Pmm
    Pm1m = x*(2*m_abs+1)*Pmm
    if l==m_abs+1:
        return Pm1m
    for ll in range(m_abs+2,l+1):
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
    theta = np.arccos(np.clip(X[:,2],-1,1))
    phi = np.mod(np.arctan2(X[:,1],X[:,0]),2*math.pi)
    Y = np.zeros((len(X),(lmax+1)*(lmax+1)))
    idx=0
    for l in range(lmax+1):
        for m in range(-l,l+1):
            Y[:,idx] = Ylm_pure(l,m,theta,phi)
            idx+=1
    return Y


# ============================================================
# global phase-memory
# ============================================================

def global_phase_memory(X,sgn,lmax=10):
    Y = compute_Ylm_pure(X,lmax)
    pos = (sgn>0)
    neg = (sgn<0)
    if np.sum(pos)<2 or np.sum(neg)<2:
        return np.nan
    A_pos = np.sum(Y[pos],axis=0)
    A_neg = np.sum(Y[neg],axis=0)
    dphi = np.abs(np.arctan2(A_pos-A_neg, np.ones_like(A_pos)))
    return np.mean(dphi)


# ============================================================
# evaluate axis wobble
# ============================================================

def evaluate_axis_wobble(X,l0,b0,wobbles=[3,5,10],
                         lmax=10,n_null=2000,seed=42):

    rng = np.random.RandomState(seed)
    results = []

    # precompute full Y matrix for speed
    Yfull = compute_Ylm_pure(X,lmax)
    Xtheta = np.arccos(np.clip(X[:,2], -1, 1))  # not used further

    for w in wobbles:
        for dl, db in [(w,0),(-w,0),(0,w),(0,-w)]:

            l_new = l0 + dl
            b_new = b0 + db
            axis_new = galactic_lb_to_xyz(l_new, b_new)

            sgn = remnant_sign(X,axis_new)
            Z_real = global_phase_memory(X,sgn,lmax)

            # null
            null_vals = []
            for _ in tqdm(range(n_null),
                          desc=f"null wobble {w}° ({dl:+},{db:+})",
                          leave=False):
                s_shuf = np.array(sgn)
                rng.shuffle(s_shuf)
                # compute using precomputed Yfull
                pos = (s_shuf>0)
                neg = (s_shuf<0)
                A_pos = np.sum(Yfull[pos],axis=0)
                A_neg = np.sum(Yfull[neg],axis=0)
                dphi = np.abs(np.arctan2(A_pos-A_neg, np.ones_like(A_pos)))
                null_vals.append(np.mean(dphi))

            null_vals = np.array(null_vals)
            p_val = (1 + np.sum(null_vals >= Z_real)) / (len(null_vals)+1)

            results.append({
                "wobble": w,
                "dl": dl, "db": db,
                "l_new": l_new,
                "b_new": b_new,
                "Z_real": float(Z_real),
                "null_mean": float(np.mean(null_vals)),
                "p": float(p_val)
            })

    return results


# ============================================================
# main
# ============================================================

def main(path):
    RA,Dec = load_catalog(path)
    Xgal = radec_to_galactic_xyz(RA,Dec)

    l0, b0 = 159.8, -0.5

    print("[info] running test 85F (axis wobble robustness)…")
    results = evaluate_axis_wobble(
        Xgal, l0,b0,
        wobbles=[3,5,10],
        lmax=10,
        n_null=2000,
        seed=42
    )

    print("====================================================")
    print(" AXIS WOBBLE ROBUSTNESS — REMNANT-TIME (TEST 85F)   ")
    print("====================================================")
    for r in results:
        print(f"w={r['wobble']:2d}°  dl={r['dl']:3d}  db={r['db']:3d}   "
              f"l_new={r['l_new']:.2f}  b_new={r['b_new']:.2f}   "
              f"Z={r['Z_real']:.6f}   "
              f"null_mean={r['null_mean']:.6f}   "
              f"p={r['p']:.6f}")
    print("====================================================")
    print("interpretation:")
    print("  stable Z across small wobbles (3°,5°) indicates")
    print("  robustness to axis perturbation.")
    print("")
    print("  significant degradation at large wobble (~10°)")
    print("  indicates alignment with unified axis is real.")
    print("====================================================")
    print("test 85F complete.")
    print("====================================================")


if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_phase_memory_axiswobble_test85F.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
