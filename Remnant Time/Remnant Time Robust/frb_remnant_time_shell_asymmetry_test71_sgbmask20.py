#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anti-bias Test 71B — Supergalactic plane mask |SGB| ≥ 20°.

Standalone masked repetition of:
    frb_remnant_time_shell_asymmetry_test71.py

Purpose:
    Verify that the shell asymmetry (Test 71) persists after
    removing FRBs near the supergalactic plane.
"""

import numpy as np
import csv
import sys
from tqdm import tqdm
import math

# ============================================================
# utilities for catalog loading
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
        raise KeyError("cannot detect RA/Dec columns")
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

# ============================================================
# coordinate transforms: RA/Dec → Galactic → Supergalactic
# ============================================================

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
        [-0.867666135681,-0.198076389622, 0.455983794523]
    ])

def radec_to_gal_xyz(RA,Dec):
    Xeq=radec_xyz(RA,Dec)
    M=M_eq_to_gal()
    X=Xeq@M.T
    return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-15)

# supergalactic transform matrix (standard deVaucouleurs)
def M_gal_to_sgal():
    return np.array([
        [-0.73574257,  0.67726104,  0.00000000],
        [-0.07455378, -0.08099147,  0.99392259],
        [ 0.67314561,  0.73127117,  0.11008126]
    ])

def gal_to_sgal(l,b):
    l=np.radians(l)
    b=np.radians(b)
    X=np.array([np.cos(b)*np.cos(l),
                np.cos(b)*np.sin(l),
                np.sin(b)])
    return (M_gal_to_sgal()@X)

def radec_to_sgal_lB(RA,Dec):
    # first to Galactic xyz
    Xgal=radec_to_gal_xyz(RA,Dec)
    # multiply by supergalactic matrix
    M=M_gal_to_sgal()
    Xsg=Xgal@M.T
    x,y,z=Xsg[:,0],Xsg[:,1],Xsg[:,2]
    SGL=np.degrees(np.arctan2(y,x))%360
    SGB=np.degrees(np.arcsin(z))
    return SGL,SGB

# ============================================================
# unified-axis geometry
# ============================================================

def gal_lb_xyz(l,b):
    l=np.radians(l)
    b=np.radians(b)
    v=np.array([np.cos(b)*np.cos(l),
                np.cos(b)*np.sin(l),
                np.sin(b)])
    return v/np.linalg.norm(v)

def unified_angles(X,axis):
    d=X@axis
    d=np.clip(d,-1,1)
    return np.degrees(np.arccos(d))

# ============================================================
# Test 71B logic
# ============================================================

def main(path):

    print("[info] loading FRB catalog...")
    RA,Dec=load_catalog(path)

    print(f"[info] original N = {len(RA)}")

    # convert to galactic xyz (for unified-axis geometry)
    Xgal=radec_to_gal_xyz(RA,Dec)

    # convert to supergalactic coordinates
    print("[info] converting to supergalactic coords...")
    SGL,SGB=radec_to_sgal_lB(RA,Dec)

    # apply mask |SGB| ≥ 20°
    mask=np.abs(SGB)>=20.0
    Xmask=Xgal[mask]
    Nmask=len(Xmask)

    print(f"[info] masked sky: |SGB|>=20°, remaining N = {Nmask}")

    if Nmask<50:
        print("[error] too few FRBs after mask.")
        return

    # unified axis (fixed)
    axis=gal_lb_xyz(159.8,-0.5)

    print("[info] computing shell angles...")
    theta=unified_angles(Xmask,axis)

    # shell definitions
    s1_min, s1_max = 17.5, 32.5
    s2_min, s2_max = 32.5, 47.5

    R = Xmask@axis
    sgn=np.where(R>=0,1,-1)

    def shell_counts(tmin,tmax):
        sel=(theta>=tmin)&(theta<tmax)
        s=sgn[sel]
        Np=np.sum(s>0)
        Nm=np.sum(s<0)
        return Np,Nm,abs(Np-Nm)

    Np1,Nm1,S1=shell_counts(s1_min,s1_max)
    Np2,Nm2,S2=shell_counts(s2_min,s2_max)
    S_total=S1+S2

    # Monte Carlo
    print("[info] building MC null (masked)...")
    null=[]
    for _ in tqdm(range(2000)):
        sh=np.copy(sgn)
        np.random.shuffle(sh)

        def S_for(sh):
            sel1=(theta>=s1_min)&(theta<s1_max)
            sel2=(theta>=s2_min)&(theta<s2_max)
            S1_=abs(np.sum(sh[sel1]>0)-np.sum(sh[sel1]<0))
            S2_=abs(np.sum(sh[sel2]>0)-np.sum(sh[sel2]<0))
            return S1_+S2_

        null.append(S_for(sh))

    null=np.array(null)
    mu=float(np.mean(null))
    sd=float(np.std(null))
    p=(1+np.sum(null>=S_total))/(len(null)+1)

    # output
    print("================================================")
    print(" FRB REMNANT-TIME SHELL ASYMMETRY TEST (71B)")
    print("   Supergalactic plane mask: |SGB| >= 20°")
    print("================================================")
    print(f"N_after_mask         = {Nmask}")
    print("------------------------------------------------")
    print(f"shell 1: N_plus={Np1}, N_minus={Nm1}, S1={S1}")
    print(f"shell 2: N_plus={Np2}, N_minus={Nm2}, S2={S2}")
    print("------------------------------------------------")
    print(f"S_total              = {S_total}")
    print(f"null mean S_total    = {mu:.6f}")
    print(f"null std S_total     = {sd:.6f}")
    print(f"p-value (masked)     = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> shell asymmetry persists without the")
    print("           supergalactic plane; not a local-supercluster")
    print("           footprint or flow artifact.")
    print("  high p -> asymmetry depends on SGB region; could be")
    print("           influenced by local-supercluster geometry.")
    print("================================================")
    print("test 71B complete.")
    print("================================================")

if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_shell_asymmetry_test71_sgbmask20.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
