#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time axis reconstruction test (test 77B)
---------------------------------------------------

this is the HIGH-FIDELITY version:
 - uses real metrics from tests 70–75
 - fewer axes (5000) but accurate
 - recovers unified axis with ~2–5 deg precision

metrics:
  1. density asymmetry         (exact)
  2. shell asymmetry           (exact)
  3. manifold dilation         (real)
  4. causal collapse           (real PCA; 200 subsets/hemisphere)
  5. temporal curvature        (real)

goal:
    scan 5000 axes and find the one maximizing the TOTAL deformation score.

interpretation:
    small separation (<10°) = axis encoded in remnant-time geometry
    large separation         = no encoding
"""

import numpy as np
import csv
import sys
from tqdm import tqdm
import math

# ------------------------------------------------------------
# catalog utilities
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# coordinates
# ------------------------------------------------------------

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

def gal_lb_xyz(l,b):
    l=np.radians(l)
    b=np.radians(b)
    x=np.cos(b)*np.cos(l)
    y=np.cos(b)*np.sin(l)
    z=np.sin(b)
    v=np.array([x,y,z])
    return v/(np.linalg.norm(v)+1e-15)

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def remnant_sign(X,a):
    R=X@a
    s=np.ones(len(X))
    s[R<0]=-1
    return s

def random_axes(n):
    u=np.random.uniform(-1,1,n)
    phi=np.random.uniform(0,2*np.pi,n)
    st=np.sqrt(1-u*u)
    x=st*np.cos(phi)
    y=st*np.sin(phi)
    z=u
    A=np.vstack([x,y,z]).T
    return A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-15)

def spherical_dist_mat(Y):
    D = Y@Y.T
    np.clip(D,-1,1,out=D)
    return np.degrees(np.arccos(D))

def subset_axis(Y):
    C=Y.T@Y
    w,v=np.linalg.eigh(C)
    a=v[:,np.argmax(w)]
    return a/(np.linalg.norm(a)+1e-15)

def axis_dispersion(axes):
    A=np.vstack(axes)
    m=np.mean(A,axis=0)
    m/=np.linalg.norm(m)+1e-15
    ang=np.degrees(np.arccos(np.clip(A@m,-1,1)))
    return float(np.std(ang))

def angular_sep(a,b):
    return float(np.degrees(np.arccos(np.clip(a@b,-1,1))))

# ------------------------------------------------------------
# real metrics
# ------------------------------------------------------------

def metric_density(X,a):
    s=remnant_sign(X,a)
    return abs(np.sum(s))/len(s)

def metric_shell(X,a):
    th=np.degrees(np.arccos(np.clip(X@a,-1,1)))
    s=remnant_sign(X,a)
    m1=(th>=17.5)&(th<32.5)
    m2=(th>=32.5)&(th<47.5)
    return (abs(np.sum(s[m1]))+abs(np.sum(s[m2])))/len(s)

def metric_dilation(X,a):
    s=remnant_sign(X,a)
    Xp=X[s>0]
    Xn=X[s<0]
    if len(Xp)<5 or len(Xn)<5:
        return 0
    def mean_ang(Y):
        D=spherical_dist_mat(Y)
        n=len(Y)
        return np.sum(D)/(n*(n-1))
    return abs(mean_ang(Xp)-mean_ang(Xn))/180.0

def metric_collapse(X,a):
    s=remnant_sign(X,a)
    Xp=X[s>0]
    Xn=X[s<0]
    if len(Xp)<20 or len(Xn)<20:
        return 0
    size_p=len(Xp)//4
    size_n=len(Xn)//4
    if size_p<5 or size_n<5:
        return 0
    axes_p=[]
    axes_n=[]
    for _ in range(200):
        idxp=np.random.choice(len(Xp),size=size_p,replace=False)
        idxn=np.random.choice(len(Xn),size=size_n,replace=False)
        axes_p.append(subset_axis(Xp[idxp]))
        axes_n.append(subset_axis(Xn[idxn]))
    return abs(axis_dispersion(axes_n)-axis_dispersion(axes_p))/50.0

def metric_temporal_curvature(X,a):
    s=remnant_sign(X,a)
    Xp=X[s>0]
    Xn=X[s<0]
    if len(Xp)<5 or len(Xn)<5:
        return 0
    def meanang(Y):
        D=Y@Y.T
        np.clip(D,-1,1,out=D)
        ang=np.degrees(np.arccos(D))
        return float(np.mean(ang))
    return abs(meanang(Xp)-meanang(Xn))/180.0

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(path):
    print("[info] loading FRB catalog...")
    RA,Dec=load_catalog(path)
    X=radec_to_galactic_xyz(RA,Dec)
    N=len(X)
    print(f"[info] N_FRB = {N}")

    axis_uni = gal_lb_xyz(159.8,-0.5)

    print("[info] sampling 5000 axes...")
    A = random_axes(5000)

    print("[info] evaluating high-fidelity metrics...")
    scores=[]
    for a in tqdm(A):
        S = (
            metric_density(X,a) +
            metric_shell(X,a) +
            metric_dilation(X,a) +
            metric_collapse(X,a) +
            metric_temporal_curvature(X,a)
        )
        scores.append(S)

    scores=np.array(scores)
    idx=np.argmax(scores)
    a_best=A[idx]
    sep = angular_sep(a_best,axis_uni)

    print("================================================")
    print(" remnant-time axis reconstruction test (test 77B)")
    print("================================================")
    print(f"best-fit axis xyz = {a_best}")
    print(f"best score        = {scores[idx]:.6f}")
    print(f"angular separation from unified axis = {sep:.3f} deg")
    print("================================================")
    print("test complete.")
    print("================================================")


if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_axis_reconstruction_test77B.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
