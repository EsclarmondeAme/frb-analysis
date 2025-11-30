#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time causal collapse test (test 74)
----------------------------------------------

goal:
    detect whether random FRB subsets produce more tightly-collapsed preferred
    directions ("subset axes") in the forward-time hemisphere (R>0) relative to
    the backward-time hemisphere (R<0).

why this matters:
    if higher-dimensional time compression is real, then temporal ordering
    breaks directionally. independent subsets of the sky will repeatedly point
    toward the same direction on the forward-time side: a causal-collapse
    signature.

statistic:
    for each hemisphere, sample 600 random subsets of size ~N/4.
    compute PCA/dipole axis for each subset.
    measure angular dispersion (std dev) of those axes.

    score:
        C_real = std_axes_Rneg - std_axes_Rpos
        (large positive -> collapse toward the remnant-time direction)

null:
    2000 isotropic skies -> same computation -> p-value.

interpretation:
    low p -> directional causal collapse detected.
    high p -> subset axes behave isotropically, no collapse.

"""

import numpy as np
import csv
import sys
from tqdm import tqdm
import math

# ============================================================
# utilities: catalog + coordinate transforms
# ============================================================

def detect_columns(fieldnames):
    low = [c.lower() for c in fieldnames]
    def find(*c):
        for name in c:
            if name.lower() in low:
                return fieldnames[low.index(name.lower())]
        return None
    ra = find("ra_deg","ra","raj2000","ra (deg)")
    dec = find("dec_deg","dec","dej2000","dec (deg)")
    if ra is None or dec is None:
        raise KeyError("could not detect ra/dec columns")
    return ra, dec


def load_catalog(path):
    with open(path,"r",encoding="utf-8") as f:
        R = csv.DictReader(f)
        fields = R.fieldnames
        ra,dec = detect_columns(fields)
        RA,Dec = [],[]
        for row in R:
            RA.append(float(row[ra]))
            Dec.append(float(row[dec]))
    return np.array(RA), np.array(Dec)


def radec_to_equatorial_xyz(RA,Dec):
    RA  = np.radians(RA)
    Dec = np.radians(Dec)
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
    Xeq = radec_to_equatorial_xyz(RA,Dec)
    M = equatorial_to_galactic_matrix()
    Xgal = Xeq @ M.T
    n = np.linalg.norm(Xgal,axis=1,keepdims=True)+1e-15
    return Xgal/n


def galactic_lb_to_xyz(l_deg,b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)
    v = np.array([x,y,z])
    return v/(np.linalg.norm(v)+1e-15)


# ============================================================
# remnant-time splitting + helper functions
# ============================================================

def remnant_sign(X,axis):
    axis = axis/(np.linalg.norm(axis)+1e-15)
    R = X @ axis
    s = np.ones_like(R)
    s[R<0] = -1
    return s


def subset_axis(Xsubset):
    """
    compute PCA principal axis for a subset of unit vectors.
    """
    C = Xsubset.T @ Xsubset
    w, v = np.linalg.eigh(C)
    axis = v[:, np.argmax(w)]
    return axis / (np.linalg.norm(axis)+1e-15)


def angular_dispersion(axes):
    """
    compute angular std dev of a list of unit axes.
    """
    A = np.vstack(axes)
    mean_vec = np.mean(A,axis=0)
    mean_vec /= (np.linalg.norm(mean_vec)+1e-15)
    dots = A @ mean_vec
    dots = np.clip(dots,-1,1)
    ang = np.degrees(np.arccos(dots))
    return float(np.std(ang))


def random_isotropic(N):
    u = np.random.uniform(-1,1,N)
    phi = np.random.uniform(0,2*np.pi,N)
    st = np.sqrt(1-u*u)
    x = st*np.cos(phi)
    y = st*np.sin(phi)
    z = u
    X = np.vstack([x,y,z]).T
    n = np.linalg.norm(X,axis=1,keepdims=True)+1e-15
    return X/n


# ============================================================
# core computation
# ============================================================

def compute_causal_collapse_score(Xgal,axis, n_subsets=600):
    sign = remnant_sign(Xgal,axis)
    Xpos = Xgal[sign>0]
    Xneg = Xgal[sign<0]

    # subset size ~ quarter of hemisphere population
    size_pos = max(5, len(Xpos)//4)
    size_neg = max(5, len(Xneg)//4)

    axes_pos = []
    axes_neg = []

    for _ in range(n_subsets):
        idxp = np.random.choice(len(Xpos), size=size_pos, replace=False)
        idxn = np.random.choice(len(Xneg), size=size_neg, replace=False)
        axes_pos.append(subset_axis(Xpos[idxp]))
        axes_neg.append(subset_axis(Xneg[idxn]))

    disp_pos = angular_dispersion(axes_pos)
    disp_neg = angular_dispersion(axes_neg)

    C = disp_neg - disp_pos
    return C, disp_pos, disp_neg


# ============================================================
# main
# ============================================================

def main(path):
    print("[info] loading FRB catalog...")
    RA,Dec = load_catalog(path)
    N = len(RA)
    print(f"[info] N_FRB = {N}")

    print("[info] converting to galactic coords...")
    Xgal = radec_to_galactic_xyz(RA,Dec)

    # unified axis
    axis = galactic_lb_to_xyz(159.8,-0.5)

    print("[info] computing real causal-collapse score...")
    C_real, disp_pos, disp_neg = compute_causal_collapse_score(
        Xgal, axis, n_subsets=600
    )

    print("[info] building MC null (isotropic skies)...")
    C_null = []
    for _ in tqdm(range(2000),desc="MC"):
        Xmc = random_isotropic(N)
        Cmc,_,_ = compute_causal_collapse_score(Xmc,axis, n_subsets=600)
        C_null.append(Cmc)
    C_null = np.array(C_null)

    mu = float(np.mean(C_null))
    sd = float(np.std(C_null))
    p = (1 + np.sum(C_null >= C_real))/(len(C_null)+1)

    print("================================================")
    print(" frb remnant-time causal collapse test (test 74)")
    print("================================================")
    print(f"axis dispersion (R>0 hemisphere) = {disp_pos:.6f} deg")
    print(f"axis dispersion (R<0 hemisphere) = {disp_neg:.6f} deg")
    print(f"C_real (collapse score)          = {C_real:.6f}")
    print("------------------------------------------------")
    print(f"null mean C                      = {mu:.6f}")
    print(f"null std C                       = {sd:.6f}")
    print(f"p-value                          = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  - low p  -> subset axes collapse strongly toward the remnant-time")
    print("             direction, consistent with directional causal compression.")
    print("  - high p -> subset axes behave isotropically; no collapse.")
    print("================================================")
    print("test 74 complete.")
    print("================================================")


if __name__ == "__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_causal_collapse_test74.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
