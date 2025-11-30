#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time cross-shell harmonic phase alignment test (test 73)
-------------------------------------------------------------------
purpose:
  detect whether spherical-harmonic phase structure in the 25° and 40° shells
  is coherently rotated or shifted depending on the remnant-time sign (R>0 vs R<0)
  relative to the unified cosmic axis.

  phase coherence across shells is exactly what you expect if the underlying
  spacetime metric is being directionally compressed in a higher-dimensional manifold.

metrics:
  - compute real spherical harmonic coefficients for l<=3 inside each shell
  - compute phase differences between R>0 and R<0 hemispheres
  - compute cross-shell coherence C = |Δφ_shell1 - Δφ_shell2|
  - compare with isotropic MC skies

interpretation:
  low p  -> cross-shell phases respond coherently to the remnant-time field
            (expected if higher-dimensional time compression exists)
  high p -> shells behave independently; no evidence for phase coherence
"""

import numpy as np
import csv
import sys
from tqdm import tqdm
import math
from scipy.special import lpmv

# ============================================================
# utilities
# ============================================================

def detect_columns(fieldnames):
    low = [c.lower() for c in fieldnames]
    def find(*c):
        for name in c:
            if name.lower() in low:
                return fieldnames[low.index(name.lower())]
        return None
    ra = find("ra_deg","ra","raj2000","ra (deg)")
    dec = find("dec_deg","dec","dej2000","dec (deg)")
    if ra is None or dec is None:
        raise KeyError("could not detect ra/dec columns")
    return ra, dec


def load_catalog(path):
    with open(path,"r",encoding="utf-8") as f:
        R = csv.DictReader(f)
        fields = R.fieldnames
        ra, dec = detect_columns(fields)
        RA, Dec = [], []
        for row in R:
            RA.append(float(row[ra]))
            Dec.append(float(row[dec]))
    return np.array(RA), np.array(Dec)


def radec_to_equatorial_xyz(RA,Dec):
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


def radec_to_galactic_xyz(RA,Dec):
    Xeq = radec_to_equatorial_xyz(RA,Dec)
    M = equatorial_to_galactic_matrix()
    Xgal = Xeq @ M.T
    n = np.linalg.norm(Xgal,axis=1,keepdims=True)+1e-15
    return Xgal/n


def galactic_lb_to_xyz(l_deg,b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)
    v = np.array([x,y,z])
    return v/(np.linalg.norm(v)+1e-15)


def angle_from_axis(X,axis):
    axis = axis/(np.linalg.norm(axis)+1e-15)
    dots = X @ axis
    np.clip(dots,-1,1,out=dots)
    return np.degrees(np.arccos(dots))


def remnant_sign(X,axis):
    axis = axis/(np.linalg.norm(axis)+1e-15)
    R = X @ axis
    s = np.ones_like(R)
    s[R<0] = -1
    return s

# ============================================================
# spherical harmonic phase extraction
# ============================================================

def real_sph_harm(l,m,theta,phi):
    # real harmonics
    if m>0:
        K = math.sqrt((2*l+1)/(4*np.pi) * math.factorial(l-m)/math.factorial(l+m))
        return math.sqrt(2)*K*np.cos(m*phi)*lpmv(m,l,np.cos(theta))
    elif m<0:
        m2 = -m
        K = math.sqrt((2*l+1)/(4*np.pi) * math.factorial(l-m2)/math.factorial(l+m2))
        return math.sqrt(2)*K*np.sin(m2*phi)*lpmv(m2,l,np.cos(theta))
    else:
        K = math.sqrt((2*l+1)/(4*np.pi))
        return K*lpmv(0,l,np.cos(theta))


def harmonic_phase_set(X,l_max=3):
    # convert xyz → theta/phi
    x,y,z = X[:,0], X[:,1], X[:,2]
    theta = np.arccos(z)
    phi   = np.arctan2(y,x)

    phases = []
    for l in range(1,l_max+1):
        for m in range(-l,l+1):
            Y = real_sph_harm(l,m,theta,phi)
            # treat coefficients as complex signal after projecting
            c = np.sum(Y) + 1e-15
            phases.append(np.angle(c + 0j))
    return np.array(phases)


def phase_difference(X_pos, X_neg):
    p_pos = harmonic_phase_set(X_pos)
    p_neg = harmonic_phase_set(X_neg)
    # take circular mean difference
    diff = np.angle(np.exp(1j*(p_pos - p_neg)))
    return float(np.mean(diff))

# ============================================================
# test statistic
# ============================================================

def compute_phase_stats(Xgal,axis):
    sign = remnant_sign(Xgal,axis)
    X_pos = Xgal[sign>0]
    X_neg = Xgal[sign<0]

    theta = angle_from_axis(Xgal,axis)

    # shell bands
    s1 = (17.5,32.5)
    s2 = (32.5,47.5)

    # extract shells
    mask1 = (theta>=s1[0]) & (theta<s1[1])
    mask2 = (theta>=s2[0]) & (theta<s2[1])

    X1_pos = Xgal[(sign>0)&mask1]
    X1_neg = Xgal[(sign<0)&mask1]

    X2_pos = Xgal[(sign>0)&mask2]
    X2_neg = Xgal[(sign<0)&mask2]

    # phase differences
    d1 = phase_difference(X1_pos, X1_neg)
    d2 = phase_difference(X2_pos, X2_neg)

    # cross-shell coherence
    C = abs(d1 - d2)

    return C, d1, d2


def random_isotropic(N):
    u = np.random.uniform(-1,1,N)
    phi = np.random.uniform(0,2*np.pi,N)
    st = np.sqrt(1-u*u)
    x = st*np.cos(phi)
    y = st*np.sin(phi)
    z = u
    X = np.vstack([x,y,z]).T
    n = np.linalg.norm(X,axis=1,keepdims=True)+1e-15
    return X/n

# ============================================================
# main
# ============================================================

def main(path):
    print("[info] loading frb catalog...")
    RA,Dec = load_catalog(path)
    N = len(RA)
    print(f"[info] N_FRB = {N}")

    print("[info] converting to galactic coords...")
    Xgal = radec_to_galactic_xyz(RA,Dec)

    axis = galactic_lb_to_xyz(159.8,-0.5)

    print("[info] computing real phase statistics...")
    C_real, d1_real, d2_real = compute_phase_stats(Xgal,axis)

    print("[info] building MC null...")
    C_null = []
    for _ in tqdm(range(2000),desc="MC"):
        Xmc = random_isotropic(N)
        Cmc,_,_ = compute_phase_stats(Xmc,axis)
        C_null.append(Cmc)
    C_null = np.array(C_null)

    mu = float(np.mean(C_null))
    sd = float(np.std(C_null))
    p = (1+np.sum(C_null>=C_real))/(len(C_null)+1)

    print("================================================")
    print(" frb remnant-time cross-shell phase test (test 73)")
    print("================================================")
    print(f"Δφ_shell1   = {d1_real:.6f}")
    print(f"Δφ_shell2   = {d2_real:.6f}")
    print(f"C_real      = {C_real:.6f}")
    print("------------------------------------------------")
    print(f"null mean C = {mu:.6f}")
    print(f"null std C  = {sd:.6f}")
    print(f"p-value     = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  - low p  -> harmonic phases in the 25° and 40° shells respond")
    print("             coherently to the remnant-time field, suggesting")
    print("             higher-dimensional time compression effects.")
    print("  - high p -> phases are symmetric; no evidence of coherence.")
    print("================================================")
    print("test 73 complete.")
    print("================================================")


if __name__ == "__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_crossshell_phase_test73.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
