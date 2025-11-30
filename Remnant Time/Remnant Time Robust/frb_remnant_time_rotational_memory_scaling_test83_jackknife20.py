#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv, sys, math
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# ============================================================
# utilities
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
    return ra,dec

def load_catalog(path):
    with open(path,"r",encoding="utf-8") as f:
        R=csv.DictReader(f)
        ra,dec=detect_columns(R.fieldnames)
        RA,Dec=[],[]
        for row in R:
            RA.append(float(row[ra])); Dec.append(float(row[dec]))
    return np.array(RA),np.array(Dec)

def radec_xyz(RA,Dec):
    RA=np.radians(RA); Dec=np.radians(Dec)
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
    l=np.radians(l); b=np.radians(b)
    v=np.array([np.cos(b)*np.cos(l),
                np.cos(b)*np.sin(l),
                np.sin(b)])
    return v/(np.linalg.norm(v)+1e-15)

def remnant_sign(X,axis):
    R=X@axis
    s=np.ones(len(X)); s[R<0]=-1
    return s

def tangent_basis(x):
    z=x/(np.linalg.norm(x)+1e-15)
    tmp=np.array([1,0,0])
    if abs(np.dot(z,tmp))>0.9:
        tmp=np.array([0,1,0])
    e1=tmp - np.dot(tmp,z)*z
    e1/= (np.linalg.norm(e1)+1e-15)
    e2=np.cross(z,e1)
    e2/= (np.linalg.norm(e2)+1e-15)
    return e1,e2

# ============================================================
# spin-2 orientation at scale k
# ============================================================

def orientation_spin2(X,k,anis=0.1):
    N=len(X)
    nbr=NearestNeighbors(n_neighbors=min(k+1,N),
                         algorithm="ball_tree").fit(X)
    _,idx=nbr.kneighbors(X)

    psi=np.zeros(N)
    valid=np.zeros(N,dtype=bool)

    for i in range(N):
        e1,e2=tangent_basis(X[i])
        nidx=idx[i,1:]
        if len(nidx)<3: continue
        U=[]
        for j in nidx:
            dx=X[j]-X[i]
            U.append([np.dot(dx,e1),np.dot(dx,e2)])
        U=np.array(U)
        C=np.cov(U.T); C=0.5*(C+C.T)
        w,v=np.linalg.eigh(C)
        im=np.argmax(w); lam1=w[im]; lam2=w[1-im]
        if lam1+lam2<=0: continue
        a=(lam1-lam2)/(lam1+lam2)
        if a<anis: continue
        vmain=v[:,im]
        ang=math.atan2(vmain[1],vmain[0])
        psi[i]=ang; valid[i]=True
    z=np.exp(2j*psi); z[~valid]=0
    return z,valid

# ============================================================
# compute A_real and null p-value for given subset
# ============================================================

def compute_A_and_p(X,s,k):
    z,valid=orientation_spin2(X,k)
    zp=z[(s>0)&valid]; zn=z[(s<0)&valid]
    if len(zp)<10 or len(zn)<10:
        return np.nan, np.nan
    A_real=abs(np.mean(zp)-np.mean(zn))

    null=[]
    for _ in range(500):
        sh=np.copy(s); np.random.shuffle(sh)
        zp2=z[(sh>0)&valid]; zn2=z[(sh<0)&valid]
        if len(zp2)<10 or len(zn2)<10:
            null.append(0); continue
        null.append(abs(np.mean(zp2)-np.mean(zn2)))
    null=np.array(null)
    p=(1+np.sum(null>=A_real))/(len(null)+1)
    return A_real,p

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

    K=[5,10,20,40,80]

    print("[info] full-sample results for 83...")
    full_results=[]
    for k in K:
        A,p=compute_A_and_p(X,s,k)
        full_results.append((k,A,p))
        print(f"  k={k:3d}   A_full={A:.6f}   p_full={p:.6f}")

    print("")

    print("[info] running 20-region longitude jackknife...")
    RAdeg = np.degrees(np.arctan2(X[:,1],X[:,0])) % 360
    edges=np.linspace(0,360,21)

    jk=[]
    for i in range(20):
        lo,hi=edges[i],edges[i+1]
        mask = ~((RAdeg>=lo)&(RAdeg<hi))
        Xj=X[mask]; sj=s[mask]
        print(f"[info] region {i}: remove RA∈[{lo:.1f},{hi:.1f})  keep={len(Xj)}")
        row=[i,len(Xj)]
        for k in K:
            A,p=compute_A_and_p(Xj,sj,k)
            row.append(A); row.append(p)
        jk.append(row)

    print("===================================================")
    print(" 20-region jackknife summary for Test 83")
    print(" region | N_keep |  k=5(A,p) | k=10(A,p) | k=20(A,p) | k=40(A,p) | k=80(A,p)")
    print("---------------------------------------------------")
    for r in jk:
        i,Nk,*vals=r
        print(f"{i:3d}   {Nk:6d}   ",end="")
        for j in range(5):
            A=vals[2*j]; p=vals[2*j+1]
            print(f"{A:.3f},{p:.3f}   ",end="")
        print("")
    print("===================================================")

if __name__=="__main__":
    main(sys.argv[1])
