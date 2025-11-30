#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv, sys, math
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

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
        raise KeyError("could not detect RA/Dec columns")
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
# coordinate transforms
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
    v=np.array([
        np.cos(b)*np.cos(l),
        np.cos(b)*np.sin(l),
        np.sin(b)
    ])
    return v/(np.linalg.norm(v)+1e-15)

# ------------------------------------------------------------
# askap utilities
# ------------------------------------------------------------

def load_askap_centers():
    # 20 ASKAP pointings used in earlier tests
    # (same as 76C, 77C, etc.)
    return np.radians(np.array([
        [302.0, -55.0],
        [320.0, -52.0],
        [300.0, -45.0],
        [310.0, -50.0],
        [330.0, -55.0],
        [340.0, -60.0],
        [290.0, -40.0],
        [305.0, -48.0],
        [315.0, -53.0],
        [325.0, -58.0],
        [335.0, -62.0],
        [345.0, -57.0],
        [355.0, -50.0],
        [265.0, -40.0],
        [275.0, -42.0],
        [285.0, -44.0],
        [295.0, -46.0],
        [305.0, -49.0],
        [315.0, -51.0],
        [325.0, -54.0],
    ]))

def angular_dist(a,b):
    return np.arccos(np.clip(np.dot(a,b),-1,1))

def assign_askap(X):
    centers = load_askap_centers()
    Cxyz = []
    for ra,dec in centers:
        Cxyz.append(radec_to_galactic_xyz(np.degrees([ra]),np.degrees([dec]))[0])
    Cxyz=np.array(Cxyz)

    ask = np.zeros(len(X),dtype=bool)
    for i,x in enumerate(X):
        d = np.min([angular_dist(x,c) for c in Cxyz])
        if d < np.radians(5.0):
            ask[i]=True
    return ask

# ------------------------------------------------------------
# orientation extraction
# ------------------------------------------------------------

def tangent_basis(x):
    z=x/(np.linalg.norm(x)+1e-15)
    tmp=np.array([1.0,0.0,0.0])
    if abs(np.dot(z,tmp))>0.9:
        tmp=np.array([0.0,1.0,0.0])
    e1=tmp-np.dot(tmp,z)*z
    e1/= (np.linalg.norm(e1)+1e-15)
    e2=np.cross(z,e1)
    e2/= (np.linalg.norm(e2)+1e-15)
    return e1,e2

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
            U.append([np.dot(dx,e1),np.dot(dx,e2)])
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

# ------------------------------------------------------------
# compute A_real + null for a subset
# ------------------------------------------------------------

def run_subset(X,s,KLIST):

    out=[]

    for k in KLIST:
        z,valid = orientation_spin2(X,k)

        z_pos=z[(s>0)&valid]
        z_neg=z[(s<0)&valid]

        if len(z_pos)<10 or len(z_neg)<10:
            out.append((k,np.nan,np.nan,np.nan))
            continue

        A_real=abs(np.mean(z_pos)-np.mean(z_neg))

        null=[]
        for _ in range(500):
            sh=np.copy(s)
            np.random.shuffle(sh)
            zp=z[(sh>0)&valid]
            zn=z[(sh<0)&valid]
            if len(zp)<10 or len(zn)<10:
                null.append(0.0)
            else:
                null.append(abs(np.mean(zp)-np.mean(zn)))

        null=np.array(null)
        mu=float(np.mean(null))
        p=(1+np.sum(null>=A_real))/(len(null)+1)
        out.append((k,A_real,mu,p))

    return out

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(path):

    RA,Dec=load_catalog(path)
    X=radec_to_galactic_xyz(RA,Dec)
    axis=gal_lb_xyz(159.8,-0.5)

    R=X@axis
    s=np.ones(len(X)); s[R<0]=-1

    askap_mask=assign_askap(X)

    X_ask   = X[askap_mask]
    s_ask   = s[askap_mask]

    X_non   = X[~askap_mask]
    s_non   = s[~askap_mask]

    KLIST=[5,10,20,40,80]

    print("================================================")
    print(" Test 83C — rotational memory scaling, ASKAP split")
    print("================================================")

    print("[info] askap count    =",len(X_ask))
    print("[info] non-askap count=",len(X_non))

    print("")
    print("[info] running ASKAP subset...")
    if len(X_ask)<20:
        print("ASKAP subset too small, skipping.")
    else:
        resA = run_subset(X_ask,s_ask,KLIST)
        for k,A,mu,p in resA:
            print(f"k={k:3d}   A_real={A}   null_mean={mu}   p={p}")
    print("")

    print("[info] running non-ASKAP subset...")
    resN = run_subset(X_non,s_non,KLIST)
    for k,A,mu,p in resN:
        print(f"k={k:3d}   A_real={A}   null_mean={mu}   p={p}")

    print("================================================")
    print(" test 83C complete")
    print("================================================")

if __name__=="__main__":
    main(sys.argv[1])
