#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB REMNANT-TIME PHASE-MEMORY — EQUAL-AREA PATCH TEST (85M2)
------------------------------------------------------------

goal:
    refine the quadrant test (85M) using more, smaller,
    approximately equal-area galactic patches (healpix-like).
    for each patch, we compare real-sky phase-memory against
    isotropic skies restricted to the same patch geometry.

design:
    - convert FRBs to galactic (l,b, xyz).
    - define patches via:
         * mu = sin(b) bands: [-1,-1/3], [-1/3,1/3], [1/3,1]
         * longitude bins l: [0,90), [90,180), [180,270), [270,360)
      → 3 x 4 = 12 approx equal-area patches.

    - for each patch:
         - select FRBs
         - compute Z_real
         - generate isotropic skies within same (l, mu) ranges
         - compute Z_iso and p_geom

    if Z_real > iso_mean with low p_geom in many patches,
    the phase-memory signal is spatially distributed and
    not tied to any single region or footprint.
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
# equatorial -> galactic
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

def radec_to_galactic_xyz_lb(RA,Dec):
    Xeq = radec_to_xyz(RA,Dec)
    M   = equatorial_to_galactic_matrix()
    Xgal= Xeq @ M.T
    Xgal/= (np.linalg.norm(Xgal,axis=1,keepdims=True)+1e-15)
    x,y,z = Xgal[:,0], Xgal[:,1], Xgal[:,2]
    b = np.degrees(np.arcsin(np.clip(z,-1,1)))
    l = (np.degrees(np.arctan2(y,x)) + 360.0) % 360.0
    return Xgal, l, b

# ------------------------------------------------------------
# remnant sign and Ylm / phase-memory
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

def remnant_sign(Xgal, axis_vec):
    axis = axis_vec/np.linalg.norm(axis_vec)
    return np.where(Xgal @ axis > 0, 1, -1)

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
    dphi = np.abs(np.arctan2(Apos - Aneg, np.ones_like(Apos)))
    return float(np.mean(dphi))

# ------------------------------------------------------------
# random isotropic restricted to a (l, mu) patch
# ------------------------------------------------------------

def random_isotropic_patch(N, lmin, lmax, mu_min, mu_max, rng):
    l = rng.uniform(lmin, lmax, size=N)
    mu = rng.uniform(mu_min, mu_max, size=N)  # mu = sin(b)
    b  = np.degrees(np.arcsin(np.clip(mu,-1,1)))
    l_rad = np.radians(l)
    b_rad = np.radians(b)
    Xgal = np.vstack([
        np.cos(b_rad)*np.cos(l_rad),
        np.cos(b_rad)*np.sin(l_rad),
        np.sin(b_rad)
    ]).T
    return Xgal

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(path, n_iso=300, seed=777):
    print("[info] loading catalog…")
    RA,Dec = load_catalog(path)

    print("[info] converting to galactic…")
    Xgal, l, b = radec_to_galactic_xyz_lb(RA,Dec)
    mu = np.sin(np.radians(b))

    axis = galactic_lb_to_axis(159.8, -0.5)

    # define patches: 3 mu-bands x 4 l-bands
    mu_edges = np.array([-1.0, -1.0/3.0, 1.0/3.0, 1.0])
    l_edges  = np.array([0.0, 90.0, 180.0, 270.0, 360.0])

    rng = np.random.RandomState(seed)
    results = []

    patch_id = 0
    for i_mu in range(3):
        for i_l in range(4):
            patch_id += 1
            mu_min, mu_max = mu_edges[i_mu], mu_edges[i_mu+1]
            lmin, lmax     = l_edges[i_l],  l_edges[i_l+1]

            mask = (mu >= mu_min) & (mu < mu_max) & (l >= lmin) & (l < lmax)
            Xc   = Xgal[mask]
            N    = len(Xc)
            print(f"[info] patch {patch_id}: mu∈[{mu_min:.3f},{mu_max:.3f}], l∈[{lmin:.1f},{lmax:.1f}) -> N={N}")

            if N < 25:
                print(f"[warn] patch {patch_id}: too few FRBs, skipping.")
                continue

            sgn_real = remnant_sign(Xc, axis)
            Y_real   = compute_Ylm_matrix(Xc, lmax=8)
            Z_real   = phase_memory(Y_real, sgn_real)
            print(f"[info] patch {patch_id}: Z_real={Z_real:.6f}")

            Z_iso_list = []
            for _ in tqdm(range(n_iso), desc=f"iso patch {patch_id}", leave=False):
                Xiso = random_isotropic_patch(N, lmin, lmax, mu_min, mu_max, rng)
                sgn_iso = remnant_sign(Xiso, axis)
                Y_iso   = compute_Ylm_matrix(Xiso, lmax=8)
                Z_iso   = phase_memory(Y_iso, sgn_iso)
                Z_iso_list.append(Z_iso)

            Z_iso = np.array(Z_iso_list)
            iso_mean = float(np.mean(Z_iso))
            iso_std  = float(np.std(Z_iso))
            p_geom   = (1 + np.sum(Z_iso >= Z_real)) / (len(Z_iso) + 1)

            print(f"[info] patch {patch_id}: iso_mean={iso_mean:.6f}, iso_std={iso_std:.6f}, p_geom={p_geom:.6f}")

            results.append({
                "patch": patch_id,
                "N": N,
                "mu_min": float(mu_min),
                "mu_max": float(mu_max),
                "lmin": float(lmin),
                "lmax": float(lmax),
                "Z_real": Z_real,
                "iso_mean": iso_mean,
                "iso_std": iso_std,
                "p_geom": p_geom,
            })

    print("=================================================================")
    print(" REMNANT-TIME PHASE-MEMORY — PATCH TEST (85M2)                   ")
    print("=================================================================")
    for r in results:
        print(f"patch {r['patch']:2d}: N={r['N']:4d}  "
              f"mu∈[{r['mu_min']:+.3f},{r['mu_max']:+.3f}]  "
              f"l∈[{r['lmin']:6.1f},{r['lmax']:6.1f})  "
              f"Z_real={r['Z_real']:.6f}  "
              f"iso_mean={r['iso_mean']:.6f}  "
              f"iso_std={r['iso_std']:.6f}  "
              f"p_geom={r['p_geom']:.6f}")
    print("=================================================================")
    print("interpretation:")
    print("  - patches with Z_real > iso_mean and low p_geom show that the")
    print("    phase-memory signal is present locally, not just globally.")
    print("  - a mix of strong and weak patches is expected given the")
    print("    limited N per patch; the key question is whether several")
    print("    independent patches show consistent excess over isotropic.")
    print("=================================================================")
    print("test 85M2 complete.")
    print("=================================================================")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_phase_memory_patches_real_vs_iso_test85M2.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
