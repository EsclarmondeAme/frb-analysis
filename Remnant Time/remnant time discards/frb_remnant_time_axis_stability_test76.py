#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# FRB REMNANT-TIME CURVATURE–DRIFT TEST (76)

----------------------------------------------

goal:
    verify whether the unified axis emerges *automatically* from the remnant-time
    deformation structure — without giving the axis to the algorithm.

method:
  1. sample ~200 random trial axes on the sphere
  2. for each axis compute:
       - density asymmetry score (test 70 core)
       - shell asymmetry score (test 71 core)
       - manifold dilation score (test 72 core)
       - causal collapse score (test 74 core)
       - temporal curvature score (test 75 core)
  3. normalize and sum scores
  4. pick axis that maximizes the combined deformation response
  5. compute angular separation between recovered axis and unified axis

interpretation:
  small separation -> axis is encoded deeply in temporal geometry
  large separation -> axis not determined by remnant-time structure
"""

import numpy as np
import csv
import sys
from tqdm import tqdm
import math

# ============================================================
# utilities
# ============================================================

def detect_columns(fieldnames):
    low=[c.lower() for c in fieldnames]
    def find(*names):
        for n in names:
            if n.lower() in low:
                return fieldnames[low.index(n.lower())]
        return None
    ra=find("ra_deg","ra","raj2000","ra (deg)")
    dec=find("dec_deg","dec","dej2000","dec (deg)")
    if ra is None or dec is None:
        raise KeyError("cannot find RA/Dec columns")
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


def radec_to_equatorial_xyz(RA,Dec):
    RA=np.radians(RA)
    Dec=np.radians(Dec)
    x=np.cos(Dec)*np.cos(RA)
    y=np.cos(Dec)*np.sin(RA)
    z=np.sin(Dec)
    return np.vstack([x,y,z]).T


def equatorial_to_galactic_matrix():
    return np.array([
        [-0.054875539390,-0.873437104725,-0.483834991775],
        [ 0.494109453633,-0.444829594298, 0.746982248696],
        [-0.867666135681,-0.198076389622, 0.455983794523],
    ])


def radec_to_galactic_xyz(RA,Dec):
    Xeq=radec_to_equatorial_xyz(RA,Dec)
    M=equatorial_to_galactic_matrix()
    Xgal=Xeq@M.T
    n=np.linalg.norm(Xgal,axis=1,keepdims=True)+1e-15
    return Xgal/n


def galactic_lb_to_xyz(l,b):
    l=np.radians(l)
    b=np.radians(b)
    x=np.cos(b)*np.cos(l)
    y=np.cos(b)*np.sin(l)
    z=np.sin(b)
    v=np.array([x,y,z])
    return v/(np.linalg.norm(v)+1e-15)


def random_unit_vector():
    u=np.random.uniform(-1,1)
    phi=np.random.uniform(0,2*np.pi)
    st=np.sqrt(1-u*u)
    return np.array([st*np.cos(phi),st*np.sin(phi),u])


def angular_sep(a,b):
    c = np.clip(a@b, -1, 1)
    return float(np.degrees(np.arccos(c)))

# ============================================================
# core remnant-time metrics (light versions)
# ============================================================

def remnant_sign(X,axis):
    axis=axis/(np.linalg.norm(axis)+1e-15)
    R=X@axis
    s=np.ones_like(R)
    s[R<0]=-1
    return s


def density_asymmetry(X,axis):
    sign=remnant_sign(X,axis)
    return abs(np.sum(sign))


def angle_from_axis(X,axis):
    axis=axis/(np.linalg.norm(axis)+1e-15)
    dots=X@axis
    return np.degrees(np.arccos(np.clip(dots,-1,1)))


def shell_asymmetry(X,axis):
    theta=angle_from_axis(X,axis)
    s=remnant_sign(X,axis)
    mask1=(theta>=17.5)&(theta<32.5)
    mask2=(theta>=32.5)&(theta<47.5)
    A1=abs(np.sum(s[mask1]))
    A2=abs(np.sum(s[mask2]))
    return A1+A2


def manifold_dilation(X,axis):
    # approximate: difference in mean pairwise distance
    sign=remnant_sign(X,axis)
    Xp=X[sign>0]
    Xn=X[sign<0]

    def mean_dist(Y):
        dots=Y@Y.T
        np.clip(dots,-1,1,out=dots)
        D=np.degrees(np.arccos(dots))
        N=len(Y)
        return np.sum(D)/(N*(N-1))
    if len(Xp)<3 or len(Xn)<3:
        return 0
    return abs(mean_dist(Xp)-mean_dist(Xn))


def causal_collapse_score(X,axis):
    sign=remnant_sign(X,axis)
    Xp=X[sign>0]
    Xn=X[sign<0]

    if len(Xp)<10 or len(Xn)<10:
        return 0

    size_p=max(5,len(Xp)//4)
    size_n=max(5,len(Xn)//4)

    def subset_axis(Y):
        C=Y.T@Y
        w,v=np.linalg.eigh(C)
        a=v[:,np.argmax(w)]
        return a/(np.linalg.norm(a)+1e-15)

    def dispersion(axes):
        A=np.vstack(axes)
        m=np.mean(A,axis=0)
        m/=np.linalg.norm(m)+1e-15
        ang=np.degrees(np.arccos(np.clip(A@m,-1,1)))
        return float(np.std(ang))

    axes_p,axes_n=[],[]
    for _ in range(100):
        idxp=np.random.choice(len(Xp),size=size_p,replace=False)
        idxn=np.random.choice(len(Xn),size=size_n,replace=False)
        axes_p.append(subset_axis(Xp[idxp]))
        axes_n.append(subset_axis(Xn[idxn]))

    return abs(dispersion(axes_n)-dispersion(axes_p))


def temporal_curvature_score(X,axis):
    sign=remnant_sign(X,axis)
    Xp=X[sign>0]
    Xn=X[sign<0]
    if len(Xp)<10 or len(Xn)<10:
        return 0
    def mean_angle(Y):
        dots=Y@Y.T
        np.clip(dots,-1,1,out=dots)
        D=np.degrees(np.arccos(dots))
        return float(np.mean(D))
    return abs(mean_angle(Xp)-mean_angle(Xn))

# ============================================================
# main axis-scanning routine
# ============================================================

def main(path):
    print("[info] loading FRB catalog...")
    RA,Dec=load_catalog(path)
    print(f"[info] N_FRB = {len(RA)}")

    print("[info] converting to galactic coords...")
    X=radec_to_galactic_xyz(RA,Dec)

    print("[info] scanning random axes...")
    n_axes=200
    AX=[]
    SCORES=[]

    for _ in tqdm(range(n_axes)):
        a=random_unit_vector()

        s70 = density_asymmetry(X,a)
        s71 = shell_asymmetry(X,a)
        s72 = manifold_dilation(X,a)
        s74 = causal_collapse_score(X,a)
        s75 = temporal_curvature_score(X,a)

        # normalized
        S = (
            s70/len(X) +
            s71/len(X) +
            s72/180 +
            s74/50 +
            s75/180
        )
        AX.append(a)
        SCORES.append(S)

    AX=np.vstack(AX)
    SCORES=np.array(SCORES)

    idx=np.argmax(SCORES)
    axis_rec=AX[idx]

    # unified axis from paper
    axis_uni=galactic_lb_to_xyz(159.8,-0.5)

    sep=angular_sep(axis_rec,axis_uni)

    print("================================================")
    print(" frb remnant-time axis stability test (test 76)")
    print("================================================")
    print("best recovered axis (galactic xyz):")
    print(axis_rec)
    print(f"score_max = {SCORES[idx]:.6f}")
    print("------------------------------------------------")
    print(f"angular separation from unified axis = {sep:.3f} deg")
    print("------------------------------------------------")
    print("interpretation:")
    print("  small sep -> remnant-time geometry recovers true axis")
    print("  large sep -> no self-consistent directional structure")
    print("================================================")
    print("test 76 complete.")
    print("================================================")


if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_axis_stability_test76.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
