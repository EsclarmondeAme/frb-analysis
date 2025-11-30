#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time phase-memory — instrument-split robustness (test 85G)
-----------------------------------------------------------------------

goal:
    verify that the global remnant-time phase-memory signal
    is not an artifact of any telescope/instrument footprint.

procedure:
    - detect instrument/telescope column
    - group FRBs by instrument label
    - compute global harmonic phase-memory for each subset
    - compute MC null (2000 runs) for each subset
    - compute Z_real, null_mean, p-value

    ALSO:
    - random 50/50 split
    - random 33/33/33 split
    - random 25/25/25/25 split

expected:
    - physical signal should appear in each instrument subset
    - footprints should not produce consistent Z values
    - random partitions should behave like instrument partitions

this mirrors the robustness logic of 71/81/83.
"""

import numpy as np
import csv
import sys
import math
from tqdm import tqdm


# ============================================================
# catalog utilities
# ============================================================

def detect_columns(fieldnames):
    low = [c.lower() for c in fieldnames]

    def find(*names):
        for n in names:
            if n.lower() in low:
                return fieldnames[low.index(n.lower())]
        return None

    ra  = find("ra_deg","ra","raj2000","ra (deg)")
    dec = find("dec_deg","dec","dej2000","dec (deg)")
    tel = find("telescope","instrument","survey","facility","site","observatory")

    if ra is None or dec is None:
        raise KeyError("could not detect RA/Dec columns")

    return ra, dec, tel


def load_catalog(path):
    with open(path,"r",encoding="utf-8") as f:
        R = csv.DictReader(f)
        ra,dec,tel = detect_columns(R.fieldnames)

        RA,Dec,Tel = [],[],[]
        for row in R:
            RA.append(float(row[ra]))
            Dec.append(float(row[dec]))
            Tel.append(row[tel] if tel is not None else "UNKNOWN")

    return np.array(RA),np.array(Dec),np.array(Tel)


# ============================================================
# coordinate transforms
# ============================================================

def radec_to_equatorial_xyz(RA,Dec):
    RA = np.radians(RA)
    Dec = np.radians(Dec)
    x = np.cos(Dec)*np.cos(RA)
    y = np.cos(Dec)*np.sin(RA)
    z = np.sin(Dec)
    return np.vstack([x,y,z]).T


def equatorial_to_galactic_matrix():
    return np.array([
        [-0.054875539390,-0.873437104725,-0.483834991775],
        [ 0.494109453633,-0.444829594298, 0.746982248696],
        [-0.867666135681,-0.198076389622, 0.455983794523]
    ])


def radec_to_galactic_xyz(RA,Dec):
    Xeq = radec_to_equatorial_xyz(RA,Dec)
    M = equatorial_to_galactic_matrix()
    Xgal = Xeq @ M.T
    return Xgal/(np.linalg.norm(Xgal,axis=1,keepdims=True)+1e-15)


def galactic_lb_to_xyz(l_deg,b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    v = np.array([
        np.cos(b)*np.cos(l),
        np.cos(b)*np.sin(l),
        np.sin(b)
    ])
    return v/(np.linalg.norm(v)+1e-15)


# ============================================================
# remnant geometry
# ============================================================

def remnant_sign(X,axis):
    axis = axis/(np.linalg.norm(axis)+1e-15)
    return np.where(X@axis>0,1,-1)


# ============================================================
# pure numpy spherical harmonics
# ============================================================

def legendre_P(l,m,x):
    m_abs = abs(m)
    Pmm = np.ones_like(x)
    if m_abs>0:
        s = np.sqrt(1-x*x)
        fact = 1
        for i in range(m_abs):
            Pmm *= -fact*s
            fact += 2
    if l==m_abs:
        return Pmm
    Pm1m = x*(2*m_abs+1)*Pmm
    if l==m_abs+1:
        return Pm1m
    for ll in range(m_abs+2,l+1):
        Pll = ((2*ll-1)*x*Pm1m - (ll+m_abs-1)*Pmm)/(ll-m_abs)
        Pmm, Pm1m = Pm1m, Pll
    return Pll


def Ylm_pure(l,m,theta,phi):
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


def compute_Ylm_pure(X,lmax=10):
    theta = np.arccos(np.clip(X[:,2],-1,1))
    phi = np.mod(np.arctan2(X[:,1],X[:,0]),2*math.pi)
    Y = np.zeros((len(X),(lmax+1)*(lmax+1)))
    idx=0
    for l in range(lmax+1):
        for m in range(-l,l+1):
            Y[:,idx] = Ylm_pure(l,m,theta,phi)
            idx+=1
    return Y


# ============================================================
# global phase-memory
# ============================================================

def global_phase_memory_from_Y(Y,sgn):
    pos = (sgn>0)
    neg = (sgn<0)
    if np.sum(pos)<2 or np.sum(neg)<2:
        return np.nan
    A_pos = np.sum(Y[pos],axis=0)
    A_neg = np.sum(Y[neg],axis=0)
    dphi = np.abs(np.arctan2(A_pos-A_neg, np.ones_like(A_pos)))
    return np.mean(dphi)


# ============================================================
# null distribution
# ============================================================

def compute_null(Y,sgn,n_null=2000,seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in tqdm(range(n_null), leave=False, desc="null"):
        s_shuf = np.array(sgn)
        rng.shuffle(s_shuf)
        vals.append(global_phase_memory_from_Y(Y,s_shuf))
    return np.array(vals)


# ============================================================
# main
# ============================================================

def main(path):
    RA,Dec,Tel = load_catalog(path)
    Xgal = radec_to_galactic_xyz(RA,Dec)
    axis = galactic_lb_to_xyz(159.8,-0.5)

    # precompute spherical harmonics
    Y = compute_Ylm_pure(Xgal,lmax=10)

    # prepare instrument groups
    inst_unique = np.unique(Tel)

    print("[info] instruments detected:", inst_unique)
    print("[info] running test 85G (instrument splits)…")
    print("----------------------------------------------------")

    # run per-instrument analysis
    for inst in inst_unique:
        idx = (Tel==inst)
        if np.sum(idx)<20:
            print(f"[warn] instrument {inst} has too few FRBs: {np.sum(idx)}")
            continue

        Xg = Xgal[idx]
        Yg = Y[idx]
        sgn = remnant_sign(Xg,axis)

        print(f"[info] instrument: {inst}   N={len(Xg)}")

        Z_real = global_phase_memory_from_Y(Yg,sgn)
        null_vals = compute_null(Yg,sgn,n_null=2000,seed=42)
        null_mean = np.mean(null_vals)
        p = (1+np.sum(null_vals>=Z_real))/(len(null_vals)+1)

        print(f"    Z_real={Z_real:.6f}   null_mean={null_mean:.6f}   p={p:.6f}")

        print("----------------------------------------------------")

    # random splits 50%, 33%, 25%
    N = len(Xgal)
    rng = np.random.RandomState(42)

    def random_partition(groups):
        idxs = np.arange(N)
        rng.shuffle(idxs)
        splits = np.array_split(idxs, groups)
        return splits

    print("[info] random 50/50 split")
    for split in random_partition(2):
        Xg = Xgal[split]
        Yg = Y[split]
        sgn = remnant_sign(Xg,axis)
        Z_real = global_phase_memory_from_Y(Yg,sgn)
        null_vals = compute_null(Yg,sgn)
        null_mean = np.mean(null_vals)
        p = (1+np.sum(null_vals>=Z_real))/(len(null_vals)+1)
        print(f"    Z={Z_real:.6f}   null_mean={null_mean:.6f}   p={p:.6f}")
    print("----------------------------------------------------")

    print("[info] random 33/33/33 split")
    for split in random_partition(3):
        Xg = Xgal[split]
        Yg = Y[split]
        sgn = remnant_sign(Xg,axis)
        Z_real = global_phase_memory_from_Y(Yg,sgn)
        null_vals = compute_null(Yg,sgn)
        null_mean = np.mean(null_vals)
        p = (1+np.sum(null_vals>=Z_real))/(len(null_vals)+1)
        print(f"    Z={Z_real:.6f}   null_mean={null_mean:.6f}   p={p:.6f}")
    print("----------------------------------------------------")

    print("[info] random 25/25/25/25 split")
    for split in random_partition(4):
        Xg = Xgal[split]
        Yg = Y[split]
        sgn = remnant_sign(Xg,axis)
        Z_real = global_phase_memory_from_Y(Yg,sgn)
        null_vals = compute_null(Yg,sgn)
        null_mean = np.mean(null_vals)
        p = (1+np.sum(null_vals>=Z_real))/(len(null_vals)+1)
        print(f"    Z={Z_real:.6f}   null_mean={null_mean:.6f}   p={p:.6f}")
    print("----------------------------------------------------")

    print("====================================================")
    print(" INSTRUMENT SPLIT ROBUSTNESS — REMNANT-TIME (85G)   ")
    print("====================================================")
    print("interpretation:")
    print("  consistent Z across instruments indicates")
    print("  the global phase-memory signal is physical,")
    print("  not tied to telescope footprint or pipeline.")
    print("====================================================")
    print("test 85G complete.")
    print("====================================================")


if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_phase_memory_instrumentsplit_test85G.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
