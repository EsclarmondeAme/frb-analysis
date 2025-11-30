#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB REMNANT-TIME PHASE-MEMORY — AXIS ANNULUS TEST (85N)
-------------------------------------------------------

goal:
    probe the remnant-time phase-memory signal in local
    small-circle annuli around the unified axis.

    rather than using full-sky or latitude cuts, we slice
    by angular distance θ from the unified axis:

        [0,20], [20,40], [40,60], [60,90] degrees.

    for each annulus, we compare Z_real to an isotropic
    ensemble restricted to the same annulus (same θ-range),
    using the same axis and hemisphere sign.

design:
    - convert FRBs to galactic xyz.
    - axis: (l,b) = (159.8°, -0.5°).
    - compute θ_i = arccos(x_i · axis).
    - define four annuli in θ.
    - for each annulus:
         * select FRBs
         * compute phase-memory Z_real
         * generate isotropic skies restricted to same θ-range
         * compute Z_iso and p_geom

    if Z_real > iso_mean with small p_geom in one or more
    annuli, this shows the signal is genuinely tied to
    axis-distance structure, not just global geometry.
"""

import numpy as np
import csv
import sys
import math
from tqdm import tqdm

# ------------------------------------------------------------
# catalog utilities
# ------------------------------------------------------------

def detect_columns(fieldnames):
    low = [c.lower() for c in fieldnames]

    def find(*names):
        for n in names:
            if n.lower() in low:
                return fieldnames[low.index(n.lower())]
        return None

    ra = find("ra_deg","ra","raj2000","ra (deg)")
    dec= find("dec_deg","dec","dej2000","dec (deg)")
    if ra is None or dec is None:
        raise KeyError("could not detect RA/Dec columns")
    return ra, dec

def load_catalog(path):
    with open(path,"r",encoding="utf-8") as f:
        R = csv.DictReader(f)
        ra_col, dec_col = detect_columns(R.fieldnames)
        RA, Dec = [], []
        for row in R:
            RA.append(float(row[ra_col]))
            Dec.append(float(row[dec_col]))
    return np.array(RA), np.array(Dec)

# ------------------------------------------------------------
# RA/Dec -> galactic xyz
# ------------------------------------------------------------

def radec_to_xyz(RA,Dec):
    RA = np.radians(RA)
    Dec= np.radians(Dec)
    x = np.cos(Dec)*np.cos(RA)
    y = np.cos(Dec)*np.sin(RA)
    z = np.sin(Dec)
    return np.vstack([x,y,z]).T

def equatorial_to_galactic_matrix():
    return np.array([
        [-0.054875539390, -0.873437104725, -0.483834991775],
        [ 0.494109453633, -0.444829594298,  0.746982248696],
        [-0.867666135681, -0.198076389622,  0.455983794523],
    ])

def radec_to_galactic_xyz(RA,Dec):
    Xeq = radec_to_xyz(RA,Dec)
    M   = equatorial_to_galactic_matrix()
    Xgal= Xeq @ M.T
    Xgal/= (np.linalg.norm(Xgal,axis=1,keepdims=True)+1e-15)
    return Xgal

# ------------------------------------------------------------
# axis, theta, remnant sign
# ------------------------------------------------------------

def galactic_lb_to_axis(l_deg,b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    v = np.array([
        np.cos(b)*np.cos(l),
        np.cos(b)*np.sin(l),
        np.sin(b)
    ])
    return v/np.linalg.norm(v)

def axis_angle(Xgal, axis_vec):
    axis = axis_vec/np.linalg.norm(axis_vec)
    cos_th = np.clip(Xgal @ axis, -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))

def remnant_sign(Xgal, axis_vec):
    axis = axis_vec/np.linalg.norm(axis_vec)
    return np.where(Xgal @ axis > 0, 1, -1)

# ------------------------------------------------------------
# Ylm / phase-memory
# ------------------------------------------------------------

def legendre_P(l,m,x):
    m0 = abs(m)
    Pmm = np.ones_like(x)
    if m0>0:
        somx2 = np.sqrt(1 - x*x)
        fact = 1.0
        for _ in range(m0):
            Pmm *= -fact*somx2
            fact += 2.0
    if l==m0:
        return Pmm
    Pm1m = x*(2*m0+1)*Pmm
    if l==m0+1:
        return Pm1m
    for ll in range(m0+2, l+1):
        Pll = ((2*ll-1)*x*Pm1m - (ll+m0-1)*Pmm)/(ll-m0)
        Pmm, Pm1m = Pm1m, Pll
    return Pll

def Ylm(l,m,theta,phi):
    x = np.cos(theta)
    Plm = legendre_P(l,m,x)
    K = math.sqrt((2*l+1)/(4*math.pi) *
                  math.factorial(l-abs(m)) /
                  math.factorial(l+abs(m)))
    if m>0:
        return math.sqrt(2)*K*Plm*np.cos(m*phi)
    elif m<0:
        return math.sqrt(2)*K*Plm*np.sin(abs(m)*phi)
    else:
        return K*Plm

def compute_Ylm_matrix(X,lmax=8):
    theta = np.arccos(np.clip(X[:,2],-1,1))
    phi   = np.mod(np.arctan2(X[:,1],X[:,0]), 2*np.pi)
    Y = np.zeros((len(X),(lmax+1)*(lmax+1)))
    idx=0
    for l in range(lmax+1):
        for m in range(-l,l+1):
            Y[:,idx] = Ylm(l,m,theta,phi)
            idx+=1
    return Y

def phase_memory(Y,sgn):
    pos = (sgn>0)
    neg = (sgn<0)
    if np.sum(pos)<2 or np.sum(neg)<2:
        return np.nan
    Apos = np.sum(Y[pos],axis=0)
    Aneg = np.sum(Y[neg],axis=0)
    dphi = np.abs(np.arctan2(Apos-Aneg, np.ones_like(Apos)))
    return float(np.mean(dphi))

# ------------------------------------------------------------
# isotropic vectors restricted to an annulus around axis
# ------------------------------------------------------------

def random_isotropic_annulus(N, axis_vec, th_min_deg, th_max_deg, rng):
    """
    generate N random directions on the sphere such that
    their angular distance from axis lies in [th_min_deg, th_max_deg].
    """
    axis = axis_vec/np.linalg.norm(axis_vec)

    # build orthonormal basis around axis
    # pick an arbitrary vector not parallel to axis
    if abs(axis[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= (np.linalg.norm(e1) + 1e-15)
    e2 = np.cross(axis, e1)
    e2 /= (np.linalg.norm(e2) + 1e-15)

    th_min = np.radians(th_min_deg)
    th_max = np.radians(th_max_deg)
    cos_min = np.cos(th_max)
    cos_max = np.cos(th_min)

    X = np.zeros((N,3))
    for i in range(N):
        u = rng.uniform(cos_min, cos_max)  # cos(theta)
        theta = np.arccos(u)
        phi   = rng.uniform(0.0, 2*np.pi)
        sin_t = np.sqrt(max(0.0,1.0 - u*u))
        # local frame
        v = (u * axis +
             sin_t * (np.cos(phi)*e1 + np.sin(phi)*e2))
        X[i] = v / (np.linalg.norm(v) + 1e-15)
    return X

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(path, n_iso=500, seed=999):
    print("[info] loading catalog…")
    RA,Dec = load_catalog(path)

    print("[info] converting to galactic xyz…")
    Xgal = radec_to_galactic_xyz(RA,Dec)

    axis = galactic_lb_to_axis(159.8, -0.5)

    print("[info] computing axis distances…")
    theta = axis_angle(Xgal, axis)

    annuli = [
        (0.0, 20.0),
        (20.0,40.0),
        (40.0,60.0),
        (60.0,90.0),
    ]

    rng = np.random.RandomState(seed)
    results = []

    for (thmin, thmax) in annuli:
        print(f"[info] --- annulus θ∈[{thmin:.1f},{thmax:.1f}] deg ---")
        mask = (theta >= thmin) & (theta < thmax)
        Xc   = Xgal[mask]
        N    = len(Xc)
        print(f"[info] N in annulus = {N}")

        if N < 40:
            print("[warn] too few FRBs, skipping this annulus.")
            continue

        sgn_real = remnant_sign(Xc, axis)
        Y_real   = compute_Ylm_matrix(Xc, lmax=8)
        Z_real   = phase_memory(Y_real, sgn_real)
        print(f"[info] Z_real = {Z_real:.6f}")

        Z_iso_list = []
        for _ in tqdm(range(n_iso), desc=f"iso annulus {thmin:.0f}-{thmax:.0f}", leave=False):
            Xiso = random_isotropic_annulus(N, axis, thmin, thmax, rng)
            sgn_iso = remnant_sign(Xiso, axis)
            Y_iso   = compute_Ylm_matrix(Xiso, lmax=8)
            Z_iso   = phase_memory(Y_iso, sgn_iso)
            Z_iso_list.append(Z_iso)

        Z_iso = np.array(Z_iso_list)
        iso_mean = float(np.mean(Z_iso))
        iso_std  = float(np.std(Z_iso))
        p_geom   = (1 + np.sum(Z_iso >= Z_real)) / (len(Z_iso)+1)

        print(f"[info] iso_mean={iso_mean:.6f}, iso_std={iso_std:.6f}, p_geom={p_geom:.6f}")

        results.append({
            "thmin": thmin,
            "thmax": thmax,
            "N": N,
            "Z_real": Z_real,
            "iso_mean": iso_mean,
            "iso_std": iso_std,
            "p_geom": p_geom,
        })

    print("=================================================================")
    print(" REMNANT-TIME PHASE-MEMORY — AXIS ANNULUS TEST (85N)            ")
    print("=================================================================")
    for r in results:
        print(f"θ∈[{r['thmin']:5.1f},{r['thmax']:5.1f}]  N={r['N']:4d}  "
              f"Z_real={r['Z_real']:.6f}  "
              f"iso_mean={r['iso_mean']:.6f}  "
              f"iso_std={r['iso_std']:.6f}  "
              f"p_geom={r['p_geom']:.6f}")
    print("=================================================================")
    print("interpretation:")
    print("  - annuli where Z_real > iso_mean with low p_geom indicate")
    print("    that remnant-time phase-memory is tied to axis distance,")
    print("    not just to global hemisphere structure.")
    print("  - comparing multiple annuli shows whether the signal is")
    print("    concentrated near the axis or spread across shells.")
    print("=================================================================")
    print("test 85N complete.")
    print("=================================================================")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_phase_memory_annuli_real_vs_iso_test85N.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
