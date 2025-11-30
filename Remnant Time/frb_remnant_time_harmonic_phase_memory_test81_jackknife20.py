#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB remnant-time harmonic phase memory (81C) — 20-region jackknife
------------------------------------------------------------------

Jackknife scheme:
  - convert to galactic xyz
  - compute longitude l
  - define 20 longitude slices (each 18 deg)
  - for each slice r, drop that region and recompute the 81C statistic
"""

import numpy as np
import csv
import sys
from tqdm import tqdm
from scipy.special import sph_harm

# ============================================================
# catalog + coords (from 81C)
# ============================================================

def detect_columns(fields):
    low=[f.lower() for f in fields]
    def find(*names):
        for n in names:
            if n.lower() in low:
                return fields[low.index(n.lower())]
        return None
    ra=find("ra_deg","ra","raj2000","ra (deg)")
    dec=find("dec_deg","dec","dej2000","dec (deg)")
    if ra is None or dec is None:
        raise KeyError("could not detect RA/Dec")
    return ra,dec

def load_catalog(path):
    with open(path,"r",encoding="utf-8") as f:
        R=csv.DictReader(f)
        ra,dec=detect_columns(R.fieldnames)
        RA,Dec=[],[]
        for row in R:
            RA.append(float(row[ra]))
            Dec.append(float(row[dec]))
    return np.array(RA),np.array(Dec)

def radec_xyz(RA,Dec):
    RA=np.radians(RA)
    Dec=np.radians(Dec)
    x=np.cos(Dec)*np.cos(RA)
    y=np.cos(Dec)*np.sin(RA)
    z=np.sin(Dec)
    return np.vstack([x,y,z]).T

def M_eq_to_gal():
    return np.array([
        [-0.054875539390,-0.873437104725,-0.483834991775],
        [ 0.494109453633,-0.444829594298, 0.746982248696],
        [-0.867666135681,-0.198076389622, 0.455983794523],
    ])

def radec_to_galactic_xyz(RA,Dec):
    Xeq=radec_xyz(RA,Dec)
    M=M_eq_to_gal()
    X=Xeq@M.T
    return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-15)

def xyz_to_lb(X):
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    l = np.degrees(np.arctan2(y,x)) % 360.0
    b = np.degrees(np.arcsin(np.clip(z,-1.0,1.0)))
    return l, b

def gal_lb_xyz(l,b):
    l=np.radians(l)
    b=np.radians(b)
    v=np.array([np.cos(b)*np.cos(l),
                np.cos(b)*np.sin(l),
                np.sin(b)])
    return v/np.linalg.norm(v)

# ============================================================
# unified-axis coordinates and phases
# ============================================================

def unified_angles(X, axis):
    dots=X@axis
    dots=np.clip(dots,-1,1)
    theta=np.arccos(dots)

    tmp=np.array([1,0,0])
    if abs(np.dot(tmp,axis))>0.9:
        tmp=np.array([0,1,0])
    e1=tmp - np.dot(tmp,axis)*axis
    e1/=np.linalg.norm(e1)
    e2=np.cross(axis,e1)

    x1=X@e1
    x2=X@e2
    phi=np.arctan2(x2,x1)

    return theta,phi

def per_object_harmonic_phases(theta,phi, LMAX=10):
    N=len(theta)
    phases=[]
    for l in range(1,LMAX+1):
        for m in range(-l,l+1):
            Y = sph_harm(m,l,phi,theta)
            ph = np.angle(Y)
            phases.append(ph)
    phases = np.array(phases).T
    return phases

def rayleigh_Z(angles):
    N=len(angles)
    C=np.sum(np.cos(angles))
    S=np.sum(np.sin(angles))
    R=np.sqrt(C*C + S*S)/N
    return N * R * R

# ============================================================
# core runner on a subset
# ============================================================

def run_81_on_subset(Xsub, axis, LMAX=10, Nmc=2000):
    N = len(Xsub)
    theta, phi = unified_angles(Xsub, axis)

    R = Xsub @ axis
    sgn = np.ones(N)
    sgn[R<0] = -1

    phases = per_object_harmonic_phases(theta, phi, LMAX=LMAX)

    P = phases[sgn>0]
    Nn = phases[sgn<0]
    if len(P)==0 or len(Nn)==0:
        return np.nan, np.nan, np.nan, np.nan

    Zs=[]
    for k in range(phases.shape[1]):
        phP = P[:,k]
        phN = Nn[:,k]
        m = min(len(phP),len(phN))
        phP = phP[:m]
        phN = phN[:m]
        dphi = phP - phN
        dphi = np.mod(dphi + np.pi, 2*np.pi) - np.pi
        Zs.append(rayleigh_Z(dphi))
    Z_real = float(np.mean(Zs))

    null=[]
    for _ in range(Nmc):
        sh = np.copy(sgn)
        np.random.shuffle(sh)
        Pm = phases[sh>0]
        Nm = phases[sh<0]
        if len(Pm)==0 or len(Nm)==0:
            null.append(0.0)
            continue
        Zlist=[]
        for k in range(phases.shape[1]):
            phP = Pm[:,k]
            phN = Nm[:,k]
            m = min(len(phP),len(phN))
            phP=phP[:m]
            phN=phN[:m]
            dphi = phP - phN
            dphi = np.mod(dphi + np.pi, 2*np.pi) - np.pi
            Zlist.append(rayleigh_Z(dphi))
        null.append(np.mean(Zlist))

    null = np.array(null)
    mu = float(np.mean(null))
    sd = float(np.std(null))
    p  = (1 + np.sum(null >= Z_real)) / (len(null)+1)

    return Z_real, mu, sd, p

# ============================================================
# main jackknife driver
# ============================================================

def main(path):

    print("[info] loading catalog...")
    RA, Dec = load_catalog(path)
    X = radec_to_galactic_xyz(RA,Dec)
    N = len(X)
    print(f"[info] N_FRB = {N}")

    l, b = xyz_to_lb(X)
    axis = gal_lb_xyz(159.8,-0.5)

    print("[info] running full-sample 81C...")
    Z_full, mu_full, sd_full, p_full = run_81_on_subset(X, axis, LMAX=10, Nmc=2000)

    print("================================================")
    print(" full-sample 81C result")
    print("================================================")
    print(f"Z_real (full) = {Z_full:.6f}")
    print(f"null mean Z   = {mu_full:.6f}")
    print(f"null std Z    = {sd_full:.6f}")
    print(f"p-value full  = {p_full:.6f}")
    print("================================================")

    print("[info] running 20-region longitude jackknife...")
    width = 360.0 / 20.0
    region_idx = (l / width).astype(int)
    region_idx = np.clip(region_idx, 0, 19)

    results = []

    for r in range(20):
        mask_keep = (region_idx != r)
        X_jk = X[mask_keep]
        Nj = len(X_jk)
        if Nj < 50:
            print(f"[warn] region {r}: too few FRBs after removal (Nj={Nj}), skipping.")
            results.append((r, Nj, np.nan, np.nan))
            continue

        print(f"[info] jackknife region {r}: removing slice {r}, remaining N={Nj}")
        Z_jk, mu_jk, sd_jk, p_jk = run_81_on_subset(X_jk, axis, LMAX=10, Nmc=2000)
        results.append((r, Nj, Z_jk, p_jk))

    print("================================================")
    print(" 20-region jackknife summary for test 81")
    print(" region  N_keep   Z_real_jk   p_jk")
    for r, Nj, Z_jk, p_jk in results:
        print(f"  {r:2d}    {Nj:4d}    {Z_jk:9.3f}   {p_jk:7.5f}")
    print("================================================")
    print("interpretation:")
    print(" if the p-values stay small across jackknife slices,")
    print(" the phase-memory signal is not being driven by a single")
    print(" sky patch.")
    print("================================================")

if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_harmonic_phase_memory_test81_jackknife20.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
