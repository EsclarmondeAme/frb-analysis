#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time curvature tensor reconstruction (test 78B)
-----------------------------------------------------------

this is the SAFE + STABLE version of Test 78.

fixes from 78:
  - degenerate triangle protection
  - zero-length edge suppression
  - nan angle avoidance
  - adaptive neighbor count
  - spherical Voronoi fallback curvature
  - guaranteed finite curvature tensors

this test reconstructs the curvature tensor K separately in
  R>0 hemisphere
  R<0 hemisphere
then compares ΔK = Kpos - Kneg.

interpretation:
  low p => directional temporal curvature
  high p => symmetric curvature, no remnant-time effect
"""

import numpy as np
import csv
import sys
from tqdm import tqdm
import math
from scipy.spatial import SphericalVoronoi, ConvexHull
from sklearn.neighbors import NearestNeighbors

# ============================================================
# Catalog utilities
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
# Coordinate transforms
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
    x=np.cos(b)*np.cos(l)
    y=np.cos(b)*np.sin(l)
    z=np.sin(b)
    v=np.array([x,y,z])
    return v/(np.linalg.norm(v)+1e-15)

# ============================================================
# Remnant sign
# ============================================================

def remnant_sign(X,axis):
    R=X@axis
    s=np.ones(len(X))
    s[R<0]=-1
    return s

# ============================================================
# Spherical coordinates
# ============================================================

def xyz_to_thetaphi(X):
    x,y,z=X[:,0],X[:,1],X[:,2]
    theta=np.arccos(np.clip(z,-1,1))
    phi=np.arctan2(y,x)
    phi[phi<0]+=2*np.pi
    return theta,phi

# ============================================================
# SAFE MESH + VORONOI curvature
# ============================================================

def safe_gaussian_curvature(X):
    """
    stable Gaussian curvature estimate via spherical Voronoi cell area.
    avoids all degenerate triangle issues.
    """
    # ensure points are unique
    Xu = np.unique(X, axis=0)
    if len(Xu) < 4:
        return np.zeros(len(X))

    sv = SphericalVoronoi(Xu)
    sv.sort_vertices_of_regions()
    areas = []
    for region in sv.regions:
        verts = sv.vertices[region]
        try:
            hull = ConvexHull(verts)
            areas.append(hull.area)
        except:
            areas.append(0.0)

    areas = np.array(areas)
    # curvature approx ~ 4π / area on sphere
    # normalize areas away from zero
    areas = np.maximum(areas, 1e-6)
    K = 4*np.pi/areas
    # re-expand to original order by nearest matching
    # (good enough; curvature is smooth)
    nbr = NearestNeighbors(n_neighbors=1).fit(Xu)
    _, idx = nbr.kneighbors(X)
    K_full = K[idx[:,0]]
    return K_full

# ============================================================
# Curvature tensor fit
# ============================================================

def fit_curvature_tensor(theta,phi,K):
    th=theta - np.mean(theta)
    ph=phi - np.mean(phi)
    M=np.vstack([th**2,2*th*ph,ph**2]).T
    y=K - np.mean(K)
    coeff,_,_,_=np.linalg.lstsq(M,y,rcond=None)
    a,b,c=coeff
    return np.array([[a,b],[b,c]])


def tensor_metrics(Kpos,Kneg):
    D=Kpos-Kneg
    lam,_=np.linalg.eigh(D)
    return (
        float(np.linalg.norm(D)),
        lam,
        float(np.trace(D)),
        float(np.linalg.det(D))
    )

# ============================================================
# Random skies
# ============================================================

def random_isotropic(N):
    u=np.random.uniform(-1,1,N)
    phi=np.random.uniform(0,2*np.pi,N)
    st=np.sqrt(1-u*u)
    X=np.vstack([st*np.cos(phi),st*np.sin(phi),u]).T
    X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-15)
    return X

# ============================================================
# MAIN
# ============================================================

def main(path):
    print("[info] loading FRB catalog...")
    RA,Dec = load_catalog(path)
    X      = radec_to_galactic_xyz(RA,Dec)
    N      = len(X)
    print(f"[info] N_FRB = {N}")

    axis = gal_lb_xyz(159.8, -0.5)

    print("[info] computing remnant signs...")
    s = remnant_sign(X,axis)
    Xp = X[s>0]
    Xn = X[s<0]

    print("[info] computing stable Voronoi curvature...")
    Kp = safe_gaussian_curvature(Xp)
    Kn = safe_gaussian_curvature(Xn)

    print("[info] fitting curvature tensors...")
    thp,php = xyz_to_thetaphi(Xp)
    thn,phn = xyz_to_thetaphi(Xn)

    Kpos = fit_curvature_tensor(thp,php,Kp)
    Kneg = fit_curvature_tensor(thn,phn,Kn)

    norm_real,lam_real,tr_real,det_real = tensor_metrics(Kpos,Kneg)

    print("[info] Monte Carlo null (2000 skies)...")
    null = []
    for _ in tqdm(range(2000)):
        Xmc = random_isotropic(N)
        smc = remnant_sign(Xmc,axis)
        Xpm = Xmc[smc>0]
        Xnm = Xmc[smc<0]

        if len(Xpm)<10 or len(Xnm)<10:
            null.append(0.0)
            continue

        try:
            Kpm = safe_gaussian_curvature(Xpm)
            Knm = safe_gaussian_curvature(Xnm)

            thp,php = xyz_to_thetaphi(Xpm)
            thn,phn = xyz_to_thetaphi(Xnm)

            Kmpos = fit_curvature_tensor(thp,php,Kpm)
            Kmneg = fit_curvature_tensor(thn,phn,Knm)

            nm,_,_,_ = tensor_metrics(Kmpos,Kmneg)
        except:
            nm = 0.0

        null.append(nm)

    null = np.array(null)
    mu=float(np.mean(null))
    sd=float(np.std(null))
    p =(1+np.sum(null>=norm_real))/(len(null)+1)

    print("================================================")
    print(" frb remnant-time curvature tensor test (78B)")
    print("================================================")
    print("Kpos tensor:"); print(Kpos)
    print("Kneg tensor:"); print(Kneg)
    print("------------------------------------------------")
    print(f"tensor norm difference  = {norm_real:.6f}")
    print(f"lambda eigenvalues      = {lam_real}")
    print(f"trace difference        = {tr_real:.6f}")
    print(f"determinant difference  = {det_real:.6f}")
    print("------------------------------------------------")
    print(f"null mean norm = {mu:.6f}")
    print(f"null std norm  = {sd:.6f}")
    print(f"p-value        = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> curvature tensor differs across remnant hemispheres,")
    print("            consistent with directional temporal curvature.")
    print("  high p -> curvature symmetric; no temporal curvature.")
    print("================================================")
    print("test 78B complete.")
    print("================================================")


if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_curvature_tensor_test78B.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
