#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB remnant-time rotational memory scaling test (83B)
-----------------------------------------------------

Goal:
    Determine whether the rotational-orientation asymmetry detected
    in Test 82 persists across neighbourhood scales k = 5,10,20,40,80.

    If A(k) remains significant across scales, this indicates a
    scale-invariant or hierarchical A-node orientation field.
    If A(k) appears only at a single k, it is likely local noise.

Method:
    For each k in KLIST:
        1. compute local orientations (spin-2) at scale k
        2. compute S_pos, S_neg
        3. compute A_real(k) = |S_pos - S_neg|
        4. build null distribution of A(k) via sign shuffling
        5. compute p(k)

Interpretation:
    - persistent low p across k → real rotational memory field
    - single-scale detection    → local geometric bias only
    - oscillatory behaviour     → precessional memory / orientation history
"""

import numpy as np
import csv
import sys
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import math

# ============================================================
# catalog utilities
# ============================================================

def detect_columns(fields):
    low = [f.lower() for f in fields]
    def find(*names):
        for n in names:
            if n.lower() in low:
                return fields[low.index(n.lower())]
        return None
    ra = find("ra_deg","ra","raj2000","ra (deg)")
    dec = find("dec_deg","dec","dej2000","dec (deg)")
    if ra is None or dec is None:
        raise KeyError("could not detect RA/Dec")
    return ra, dec

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
# coordinate transforms
# ============================================================

def radec_xyz(RA,Dec):
    RA=np.radians(RA)
    Dec=np.radians(Dec)
    x=np.cos(Dec)*np.cos(RRA)
    y=np.cos(Dec)*np.sin(RRA)
    z=np.sin(Dec)
    return np.vstack([x,y,z]).T

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

def radec_to_galactic_xyz(RA,Dec):
    Xeq=radec_xyz(RA,Dec)
    M=M_eq_to_gal()
    X=Xeq@M.T
    return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-15)

def gal_lb_xyz(l,b):
    l=np.radians(l)
    b=np.radians(b)
    v=np.array([np.cos(b)*np.cos(l),
                np.cos(b)*np.sin(l),
                np.sin(b)])
    return v/(np.linalg.norm(v)+1e-15)

# ============================================================
# remnant sign, tangent basis
# ============================================================

def remnant_sign(X,axis):
    R=X@axis
    s=np.ones(len(X))
    s[R<0]=-1
    return s

def tangent_basis(x):
    z=x/(np.linalg.norm(x)+1e-15)
    tmp=np.array([1.0,0.0,0.0])
    if abs(np.dot(z,tmp))>0.9:
        tmp=np.array([0.0,1.0,0.0])
    e1=tmp - np.dot(tmp,z)*z
    n=np.linalg.norm(e1)
    if n<1e-15:
        tmp=np.array([0.0,1.0,0.0])
        e1=tmp - np.dot(tmp,z)*z
        n=np.linalg.norm(e1)
    e1/= (n+1e-15)
    e2=np.cross(z,e1)
    e2/= (np.linalg.norm(e2)+1e-15)
    return e1,e2

# ============================================================
# orientation extraction for a given scale k
# ============================================================

def orientation_spin2(X,k,anis=0.1):
    N=len(X)
    nbr=NearestNeighbors(n_neighbors=min(k+1,N),algorithm="ball_tree").fit(X)
    dist,idx=nbr.kneighbors(X)

    psi=np.zeros(N)
    valid=np.zeros(N,dtype=bool)

    for i in range(N):
        e1,e2=tangent_basis(X[i])
        nidx=idx[i,1:]
        if len(nidx)<3:
            continue

        U=[]
        for j in nidx:
            dx=X[j]-X[i]
            U.append([np.dot(dx,e1), np.dot(dx,e2)])
        U=np.array(U)

        C=np.cov(U.T)
        C=0.5*(C+C.T)

        w,v=np.linalg.eigh(C)
        im=np.argmax(w)
        lam1=w[im]
        lam2=w[1-im]

        if lam1+lam2<=0:
            continue

        a=(lam1-lam2)/(lam1+lam2)
        if a<anis:
            continue

        vmain=v[:,im]
        ang=math.atan2(vmain[1],vmain[0])
        psi[i]=ang
        valid[i]=True

    z=np.exp(2j*psi)
    z[~valid]=0.0+0.0j
    return z,valid

# ============================================================
# main
# ============================================================

def main(path):

    print("[info] loading FRB catalog...")
    RA,Dec=load_catalog(path)
    X=radec_to_galactic_xyz(RA,Dec)
    N=len(X)
    print(f"[info] N_FRB = {N}")

    axis=gal_lb_xyz(159.8,-0.5)
    s=remnant_sign(X,axis)

    KLIST=[5,10,20,40,80]

    print("[info] running rotational-memory scaling...")
    print("")

    results=[]

    for k in KLIST:
        print(f"[info] scale k = {k}")
        z,valid=orientation_spin2(X,k)

        z_pos=z[(s>0)&valid]
        z_neg=z[(s<0)&valid]

        if len(z_pos)<10 or len(z_neg)<10:
            print(f"[warn] insufficient valid orientations at k={k}")
            results.append((k,np.nan,np.nan,np.nan))
            continue

        Spos=np.mean(z_pos)
        Sneg=np.mean(z_neg)
        A_real=abs(Spos-Sneg)

        # null
        null=[]
        for _ in range(500):
            sh=np.copy(s)
            np.random.shuffle(sh)
            zp=z[(sh>0)&valid]
            zn=z[(sh<0)&valid]
            if len(zp)<10 or len(zn)<10:
                null.append(0.0)
                continue
            null.append(abs(np.mean(zp)-np.mean(zn)))

        null=np.array(null)
        mu=float(np.mean(null))
        sd=float(np.std(null))
        p=(1+np.sum(null>=A_real))/(len(null)+1)

        results.append((k,A_real,mu,p))

        print(f"  A_real = {A_real:.6f}, null mean = {mu:.6f}, p = {p:.6f}")
        print("")

    print("================================================")
    print(" FRB REMNANT-TIME ROTATIONAL MEMORY SCALING (83B)")
    print("================================================")
    for (k,A,mu,p) in results:
        print(f"k={k:3d}   A_real={A:.6f}   null_mean={mu:.6f}   p={p:.6f}")
    print("================================================")
    print("test 83B complete.")
    print("================================================")


if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_rotational_memory_scaling_test83B.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
