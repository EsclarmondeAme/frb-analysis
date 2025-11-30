#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time ricci-flow divergence test (test 79)
-----------------------------------------------------

goal:
    compare ricci-flow contraction axes and flow velocities
    between the R>0 and R<0 remnant-time hemispheres.

method:
    1. load frb catalog
    2. convert to galactic xyz
    3. split points by remnant-time sign
    4. perform discrete ricci-flow smoothing on each hemisphere
    5. measure:
         - contraction axis (PCA of flow vectors)
         - flow velocity (mean angular displacement per iteration)
    6. compute:
         Δθ = angular separation between contraction axes
         Δv = difference in flow velocities
         S_real = Δθ + αΔv
    7. null: random isotropic skies, same pipeline
    8. p-value

interpretation:
    low p  -> directional temporal curvature / compression
    high p -> symmetric ricci geometry
"""

import numpy as np
import csv
import sys
from tqdm import tqdm
import math

from sklearn.neighbors import NearestNeighbors

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
    x=np.cos(b)*np.cos(l)
    y=np.cos(b)*np.sin(l)
    z=np.sin(b)
    return np.array([x,y,z])/(np.linalg.norm([x,y,z])+1e-15)

# ============================================================
# remnant sign
# ============================================================

def remnant_sign(X,a):
    R=X@a
    s=np.ones(len(X))
    s[R<0]=-1
    return s

# ============================================================
# ricci-flow util
# ============================================================

def ricci_flow_step(X, k=12, step=0.15):
    nbr=NearestNeighbors(n_neighbors=k+1,algorithm='ball_tree').fit(X)
    dist,idx=nbr.kneighbors(X)
    Xnew = np.copy(X)
    for i in range(len(X)):
        neigh = X[idx[i,1:]]
        m = np.mean(neigh, axis=0)
        v = m - X[i]
        Xnew[i] = X[i] + step * v
        # renormalize to sphere
        Xnew[i] /= np.linalg.norm(Xnew[i]) + 1e-15
    flow = Xnew - X
    return Xnew, flow


def ricci_flow_contraction_axis(flow):
    # PCA of flow vectors
    C = flow.T @ flow
    w,v = np.linalg.eigh(C)
    a = v[:, np.argmax(w)]
    return a/(np.linalg.norm(a)+1e-15)


def angular_sep(a,b):
    return float(np.degrees(np.arccos(np.clip(a@b,-1,1))))

# ============================================================
# random sky
# ============================================================

def random_isotropic(N):
    u=np.random.uniform(-1,1,N)
    phi=np.random.uniform(0,2*np.pi,N)
    st=np.sqrt(1-u*u)
    X=np.vstack([st*np.cos(phi),st*np.sin(phi),u]).T
    return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-15)

# ============================================================
# MAIN
# ============================================================

def main(path):

    print("[info] loading FRB catalog...")
    RA,Dec = load_catalog(path)
    X = radec_to_galactic_xyz(RA,Dec)
    N = len(X)
    print(f"[info] N_FRB = {N}")

    axis_uni = gal_lb_xyz(159.8,-0.5)

    print("[info] computing remnant signs...")
    s = remnant_sign(X,axis_uni)
    Xp = X[s>0]
    Xn = X[s<0]

    if len(Xp)<30 or len(Xn)<30:
        print("[error] not enough points in one hemisphere.")
        return

    # --------------------------------------------------------
    # REAL RICCI FLOW
    # --------------------------------------------------------
    print("[info] performing real ricci flow...")
    Xp_f = np.copy(Xp)
    Xn_f = np.copy(Xn)

    # run 10 iterations
    flow_p_total = []
    flow_n_total = []

    for _ in range(10):
        Xp_f, fp = ricci_flow_step(Xp_f)
        Xn_f, fn = ricci_flow_step(Xn_f)
        flow_p_total.append(fp)
        flow_n_total.append(fn)

    flow_p = np.mean(np.stack(flow_p_total),axis=0)
    flow_n = np.mean(np.stack(flow_n_total),axis=0)

    # contraction axes
    a_p = ricci_flow_contraction_axis(flow_p)
    a_n = ricci_flow_contraction_axis(flow_n)

    # contraction difference
    dtheta = angular_sep(a_p, a_n)

    # flow velocities
    vp = float(np.mean(np.linalg.norm(flow_p,axis=1)))
    vn = float(np.mean(np.linalg.norm(flow_n,axis=1)))
    dv = abs(vp - vn)

    # combined score
    alpha = 50.0
    S_real = dtheta + alpha*dv

    # --------------------------------------------------------
    # NULL
    # --------------------------------------------------------
    print("[info] building null (2000 skies)...")
    null=[]
    for _ in tqdm(range(2000)):
        Xmc = random_isotropic(N)
        smc = remnant_sign(Xmc,axis_uni)
        Xpm = Xmc[smc>0]
        Xnm = Xmc[smc<0]
        if len(Xpm)<30 or len(Xnm)<30:
            null.append(0.0)
            continue

        # ricci flow
        Xpm_f = np.copy(Xpm)
        Xnm_f = np.copy(Xnm)
        fl_p_total=[]
        fl_n_total=[]
        for _ in range(10):
            Xpm_f, fp = ricci_flow_step(Xpm_f)
            Xnm_f, fn = ricci_flow_step(Xnm_f)
            fl_p_total.append(fp)
            fl_n_total.append(fn)
        fp = np.mean(np.stack(fl_p_total),axis=0)
        fn = np.mean(np.stack(fl_n_total),axis=0)

        a_p = ricci_flow_contraction_axis(fp)
        a_n = ricci_flow_contraction_axis(fn)
        dt = angular_sep(a_p,a_n)
        vp = float(np.mean(np.linalg.norm(fp,axis=1)))
        vn = float(np.mean(np.linalg.norm(fn,axis=1)))
        dvn = abs(vp-vn)
        S = dt + alpha*dvn
        null.append(S)

    null=np.array(null)
    mu=float(np.mean(null))
    sd=float(np.std(null))
    p = (1+np.sum(null >= S_real))/(len(null)+1)

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------
    print("================================================")
    print(" FRB REMNANT-TIME RICCI-FLOW DIVERGENCE TEST (79)")
    print("================================================")
    print(f"Δθ (axis separation)     = {dtheta:.6f} deg")
    print(f"Δv (flow speed diff)     = {dv:.6f}")
    print(f"real score S_real        = {S_real:.6f}")
    print("------------------------------------------------")
    print(f"null mean S              = {mu:.6f}")
    print(f"null std S               = {sd:.6f}")
    print(f"p-value                  = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> remnant-time hemispheres show different Ricci-flow")
    print("            contraction behavior, consistent with temporal")
    print("            compression / higher-dimensional curvature.")
    print("  high p -> Ricci-flow symmetric; no temporal curvature.")
    print("================================================")
    print("test 79 complete.")
    print("================================================")


if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_ricci_flow_divergence_test79.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
