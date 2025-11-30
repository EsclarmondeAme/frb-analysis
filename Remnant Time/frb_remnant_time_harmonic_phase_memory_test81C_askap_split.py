#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, csv, numpy as np
from tqdm import tqdm
from scipy.special import sph_harm
from astropy.io import fits
from math import radians

# ============================================================
# catalog loading
# ============================================================

def detect_columns(fields):
    low=[f.lower() for f in fields]
    def find(*names):
        for n in names:
            if n.lower() in low:
                return fields[low.index(n.lower())]
        return None
    ra=find("ra","ra_deg","raj2000","ra (deg)")
    dec=find("dec","dec_deg","dej2000","dec (deg)")
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
    Xeq = radec_xyz(RA,Dec)
    M   = M_eq_to_gal()
    X   = Xeq @ M.T
    return X / (np.linalg.norm(X,axis=1,keepdims=True)+1e-15)

def gal_lb_xyz(l,b):
    l=np.radians(l); b=np.radians(b)
    v=np.array([np.cos(b)*np.cos(l),
                np.cos(b)*np.sin(l),
                np.sin(b)])
    return v/np.linalg.norm(v)

# ============================================================
# unified-axis coordinates
# ============================================================

def unified_angles(X, axis):
    dots = np.clip(X @ axis, -1,1)
    theta = np.arccos(dots)

    tmp=np.array([1,0,0])
    if abs(np.dot(tmp,axis))>0.9:
        tmp=np.array([0,1,0])
    e1 = tmp - np.dot(tmp,axis)*axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis,e1)

    x1 = X @ e1
    x2 = X @ e2
    phi = np.arctan2(x2,x1)

    return theta,phi

# ============================================================
# per-object harmonic phases
# ============================================================

def per_object_harmonic_phases(theta,phi,LMAX=10):
    phases=[]
    for l in range(1,LMAX+1):
        for m in range(-l,l+1):
            Y  = sph_harm(m,l,phi,theta)
            ph = np.angle(Y)
            phases.append(ph)
    return np.array(phases).T   # shape (N, modes)

# ============================================================
# Rayleigh Z
# ============================================================

def rayleigh_Z(angles):
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    N = len(angles)
    R = np.sqrt(C*C + S*S)/N
    return N * R * R

# ============================================================
# ASKAP matching
# ============================================================

ASKAP_DIR = "data/positions"
TOL = 2.0

def load_askap_pointings():
    ras=[]; decs=[]
    if not os.path.exists(ASKAP_DIR):
        return np.array(ras),np.array(decs)
    for f in os.listdir(ASKAP_DIR):
        if not f.lower().endswith(".fits"):
            continue
        try:
            h = fits.getheader(os.path.join(ASKAP_DIR,f))
            ras.append(float(h.get("CRVAL1")))
            decs.append(float(h.get("CRVAL2")))
        except:
            pass
    return np.array(ras),np.array(decs)

def angsep(ra1,dec1,ra2,dec2):
    ra1=radians(ra1); dec1=radians(dec1)
    ra2=np.radians(ra2); dec2=np.radians(dec2)
    return np.degrees(np.arccos(
        np.clip(np.sin(dec1)*np.sin(dec2)+
                np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2), -1,1)))

def match_askap(RA,Dec,RAa,Deca):
    out=np.zeros(len(RA),dtype=bool)
    for i in range(len(RA)):
        d = angsep(RA[i],Dec[i],RAa,Deca)
        if np.min(d)<=TOL:
            out[i]=True
    return out

# ============================================================
# subset runner
# ============================================================

def run_subset(RA,Dec,X,axis):
    if len(X)<20:
        return np.nan,np.nan,np.nan,1.0

    theta,phi = unified_angles(X,axis)
    R = X @ axis
    sgn = np.ones(len(X)); sgn[R<0] = -1

    phases = per_object_harmonic_phases(theta,phi,LMAX=10)

    P = phases[sgn>0]
    Nn= phases[sgn<0]
    m=min(len(P),len(Nn))
    if m<10:
        return np.nan,np.nan,np.nan,1.0

    Zs=[]
    for k in range(phases.shape[1]):
        dphi = P[:m,k] - Nn[:m,k]
        dphi = np.mod(dphi + np.pi, 2*np.pi) - np.pi
        Zs.append(rayleigh_Z(dphi))
    Z_real=np.mean(Zs)

    null=[]
    for _ in range(2000):
        sh=np.copy(sgn)
        np.random.shuffle(sh)
        Pm=phases[sh>0]
        Nm=phases[sh<0]
        mm=min(len(Pm),len(Nm))
        if mm<10:
            null.append(0); continue

        Zlist=[]
        for k in range(phases.shape[1]):
            dphi = Pm[:mm,k] - Nm[:mm,k]
            dphi = np.mod(dphi + np.pi,2*np.pi) - np.pi
            Zlist.append(rayleigh_Z(dphi))
        null.append(np.mean(Zlist))

    null=np.array(null)
    mu=float(np.mean(null))
    sd=float(np.std(null))
    p=(1 + np.sum(null>=Z_real)) / (len(null)+1)

    return Z_real,mu,sd,p

# ============================================================
# main
# ============================================================

def main(path):
    RA,Dec = load_catalog(path)
    X = radec_to_galactic_xyz(RA,Dec)

    RAa,Deca = load_askap_pointings()
    isA = match_askap(RA,Dec,RAa,Deca)

    Xa = X[isA]
    Xn = X[~isA]
    RAa_sub = RA[isA]; Deca_sub = Dec[isA]
    RAn_sub = RA[~isA]; Decn_sub = Dec[~isA]

    axis = gal_lb_xyz(159.8,-0.5)

    print("[info] askap count     =",len(Xa))
    print("[info] non-askap count =",len(Xn))

    print("[info] running askap subset...")
    Za, mua, sda, pa = run_subset(RAa_sub,Deca_sub,Xa,axis)

    print("[info] running non-askap subset...")
    Zn, mun, sdn, pn = run_subset(RAn_sub,Decn_sub,Xn,axis)

    print("================================================")
    print(" remnant-time harmonic phase memory test 81C")
    print(" askap split")
    print("================================================")
    print(f"askap:     Z={Za}, null_mean={mua}, null_std={sda}, p={pa}")
    print(f"non-askap: Z={Zn}, null_mean={mun}, null_std={sdn}, p={pn}")
    print("================================================")

if __name__=="__main__":
    main(sys.argv[1])
