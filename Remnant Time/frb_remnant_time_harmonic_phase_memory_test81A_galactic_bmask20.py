#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv
import sys
from tqdm import tqdm
from scipy.special import sph_harm

# ============================================================
# catalog utils
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

# ============================================================
# coordinate transforms
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
    v=np.array([np.cos(b)*np.cos(l),
                np.cos(b)*np.sin(l),
                np.sin(b)])
    return v/np.linalg.norm(v)

# ============================================================
# galactic b mask
# ============================================================

def radec_to_gal_lb(RA,Dec):
    X = radec_to_galactic_xyz(RA,Dec)
    x,y,z = X[:,0],X[:,1],X[:,2]
    b = np.degrees(np.arcsin(z))
    l = np.degrees(np.arctan2(y,x)) % 360.0
    return l,b

# ============================================================
# unified-axis coordinates
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

# ============================================================
# per-object harmonic phases
# ============================================================

def per_object_harmonic_phases(theta,phi, LMAX=10):
    phases=[]
    for l in range(1,LMAX+1):
        for m in range(-l,l+1):
            Y = sph_harm(m,l,phi,theta)
            phases.append(np.angle(Y))
    return np.array(phases).T

# ============================================================
# Rayleigh Z
# ============================================================

def rayleigh_Z(angles):
    N=len(angles)
    C=np.sum(np.cos(angles))
    S=np.sum(np.sin(angles))
    R=np.sqrt(C*C + S*S)/N
    return N * R * R

# ============================================================
# MAIN 81A
# ============================================================

def main(path):

    print("================================================")
    print(" Test 81A — Harmonic Phase Memory under |b|>=20°")
    print("================================================")

    RA,Dec = load_catalog(path)

    # mask
    l,b = radec_to_gal_lb(RA,Dec)
    mask = np.abs(b) >= 20
    RA,Dec = RA[mask],Dec[mask]
    print(f"[info] N_after_mask = {len(RA)}")

    X = radec_to_galactic_xyz(RA,Dec)

    axis = gal_lb_xyz(159.8,-0.5)

    theta,phi = unified_angles(X,axis)
    R = X@axis
    sgn = np.ones(len(R))
    sgn[R<0] = -1

    phases = per_object_harmonic_phases(theta,phi,LMAX=10)

    P = phases[sgn>0]
    Nn = phases[sgn<0]
    if len(P)==0 or len(Nn)==0:
        print("[error] hemisphere empty")
        return

    Zs=[]
    for k in range(phases.shape[1]):
        phP=P[:,k]
        phN=Nn[:,k]
        m=min(len(phP),len(phN))
        dphi = phP[:m]-phN[:m]
        dphi = np.mod(dphi+np.pi,2*np.pi)-np.pi
        Zs.append(rayleigh_Z(dphi))
    Z_real = float(np.mean(Zs))

    # null
    print("[info] null 2000 shuffles...")
    null=[]
    NMC=2000

    for _ in tqdm(range(NMC)):
        sh=np.copy(sgn)
        np.random.shuffle(sh)
        Pm=phases[sh>0]
        Nm=phases[sh<0]

        if len(Pm)==0 or len(Nm)==0:
            null.append(0)
            continue

        Zlist=[]
        for k in range(phases.shape[1]):
            phP=Pm[:,k]
            phN=Nm[:,k]
            m=min(len(phP),len(phN))
            dphi=phP[:m]-phN[:m]
            dphi=np.mod(dphi+np.pi,2*np.pi)-np.pi
            Zlist.append(rayleigh_Z(dphi))

        null.append(np.mean(Zlist))

    null=np.array(null)
    mu=float(np.mean(null))
    sd=float(np.std(null))
    p=(1+np.sum(null>=Z_real))/(len(null)+1)

    print("------------------------------------------------")
    print(f"Z_real  = {Z_real:.6f}")
    print(f"null µ  = {mu:.6f}")
    print(f"null σ  = {sd:.6f}")
    print(f"p-value = {p:.6f}")
    print("------------------------------------------------")
    print(" low p  -> phase memory survives masking")
    print(" high p -> random-phase behaviour")
    print("================================================")

if __name__=="__main__":
    main(sys.argv[1])
