#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time null-congruence shear & twist test (test 80)
-------------------------------------------------------------

This generalizes Test 75.

We compute optical scalars (θ expansion, σ shear, ω twist)
for the null-like congruence determined by local kNN geodesic directions,
separately for R>0 and R<0 hemispheres.

Then compare:
    Δθ, Δσ, Δω
and build a combined anisotropy score.

Interpretation:
   low p  -> directional temporal compression / twisting
   high p -> symmetric null congruence; no temporal effect
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
    v=np.array([x,y,z])
    return v/(np.linalg.norm(v)+1e-15)

# ============================================================
# remnant sign
# ============================================================

def remnant_sign(X,a):
    R=X@a
    s=np.ones(len(X))
    s[R<0]=-1
    return s

# ============================================================
# null-congruence / optical scalars
# ============================================================

def local_flow_vectors(X, k=20):
    nbr = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    dist, idx = nbr.kneighbors(X)

    flows=[]
    for i in range(len(X)):
        neigh = X[idx[i,1:]]
        m = np.mean(neigh, axis=0)
        v = m - X[i]
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            v = np.zeros(3)
        else:
            v = v / nrm
        flows.append(v)
    return np.array(flows)

def optical_scalars(X, flows, k=20):
    nbr = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    dist, idx = nbr.kneighbors(X)

    thetas=[]
    sigmas=[]
    omegas=[]

    for i in range(len(X)):
        v0 = flows[i]
        neigh = X[idx[i,1:]]
        J = np.zeros((3,3))

        # finite difference Jacobian
        for nn in neigh:
            dv = flows[np.argmin(np.sum((X - nn)**2, axis=1))] - v0
            dx = nn - X[i]
            nrm = np.linalg.norm(dx)
            if nrm < 1e-12:
                continue
            dx = dx / nrm
            J += np.outer(dv, dx)
        J /= max(1, len(neigh))

        # decompose optical scalars
        theta = np.trace(J)
        S = 0.5*(J + J.T)
        A = 0.5*(J - J.T)

        # remove expansion from shear
        S0 = S - (theta/3.0)*np.eye(3)
        sigma = math.sqrt(np.sum(S0*S0))
        omega = math.sqrt(np.sum(A*A))

        thetas.append(theta)
        sigmas.append(sigma)
        omegas.append(omega)

    return (
        float(np.mean(thetas)),
        float(np.mean(sigmas)),
        float(np.mean(omegas))
    )

# ============================================================
# random sky
# ============================================================

def random_isotropic(N):
    u=np.random.uniform(-1,1,N)
    phi=np.random.uniform(0,2*np.pi,N)
    st=np.sqrt(1-u*u)
    X=np.vstack([st*np.cos(phi), st*np.sin(phi), u]).T
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

    # --------------------------------------------
    # REAL MEASUREMENT
    # --------------------------------------------
    print("[info] computing null-congruence flows...")
    fp = local_flow_vectors(Xp)
    fn = local_flow_vectors(Xn)

    print("[info] computing optical scalars...")
    th_p, sh_p, om_p = optical_scalars(Xp, fp)
    th_n, sh_n, om_n = optical_scalars(Xn, fn)

    dth = abs(th_p - th_n)
    dsh = abs(sh_p - sh_n)
    dom = abs(om_p - om_n)

    beta = 50.0
    gamma = 50.0
    S_real = dth + beta*dsh + gamma*dom

    # --------------------------------------------
    # NULL
    # --------------------------------------------
    print("[info] building null (2000 skies)...")
    null=[]

    for _ in tqdm(range(2000)):
        Xmc = random_isotropic(N)
        smc = remnant_sign(Xmc,axis_uni)
        Xpm = Xmc[smc>0]
        Xnm = Xmc[smc<0]

        if len(Xpm)<20 or len(Xnm)<20:
            null.append(0.0)
            continue

        fpm = local_flow_vectors(Xpm)
        fnm = local_flow_vectors(Xnm)

        thp, shp, omp = optical_scalars(Xpm, fpm)
        thn, shn, omn = optical_scalars(Xnm, fnm)

        dTH = abs(thp - thn)
        dSH = abs(shp - shn)
        dOM = abs(omp - omn)

        S = dTH + beta*dSH + gamma*dOM
        null.append(S)

    null=np.array(null)
    mu=float(np.mean(null))
    sd=float(np.std(null))
    p=(1+np.sum(null>=S_real))/(len(null)+1)

    # --------------------------------------------
    # OUTPUT
    # --------------------------------------------
    print("================================================")
    print(" FRB REMNANT-TIME NULL-CONGRUENCE TEST (80)")
    print("================================================")
    print(f"Δθ (expansion diff)     = {dth:.6e}")
    print(f"Δσ (shear diff)         = {dsh:.6e}")
    print(f"Δω (twist diff)         = {dom:.6e}")
    print(f"real score S_real       = {S_real:.6f}")
    print("------------------------------------------------")
    print(f"null mean S             = {mu:.6f}")
    print(f"null std S              = {sd:.6f}")
    print(f"p-value                 = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> directional null-congruence deformation,")
    print("            consistent with temporal compression.")
    print("  high p -> symmetric optical scalars; no effect.")
    print("================================================")
    print("test 80 complete.")
    print("================================================")

if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_null_congruence_test80.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
