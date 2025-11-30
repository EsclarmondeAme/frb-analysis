#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time manifold dilation test (test 72)
------------------------------------------------
goal:
  test whether the frb angular manifold is stretched or compressed depending
  on the direction of a remnant-time field aligned with the unified cosmic axis.

summary of what this measures:
  1. pairwise dilation:
        mean angular distance within R>0 hemisphere
        minus
        mean angular distance within R<0 hemisphere

  2. kNN manifold thickness:
        mean k-nearest-neighbour distance in each hemisphere
        (smaller -> more compressed, larger -> more dilated)

  3. harmonic moment difference:
        difference in low-l spherical harmonic power between hemispheres
        (l = 1,2,...,l_max)

statistic:
  we combine all three metrics into a single scalar:
      S_real = |Δ_pairwise| + |Δ_knn| + |Δ_harmonic|

null:
  2000 isotropic Monte Carlo skies
  same splitting via remnant-time sign (R>0 vs R<0)
  compute S for each
  p = fraction with S_null >= S_real

interpretation:
  low p  -> FRB manifold geometry differs strongly between time-hemispheres
            in ways not expected from isotropy
  high p -> geometry looks symmetric with respect to the remnant-time field
"""

import numpy as np
import csv
import sys
from tqdm import tqdm
import math
from scipy.special import lpmv as scipy_lpmv

# ============================================================
# utilities: catalog / coordinates
# ============================================================

def detect_columns(fieldnames):
    low = [c.lower() for c in fieldnames]
    def find(*names):
        for n in names:
            if n.lower() in low:
                return fieldnames[low.index(n.lower())]
        return None
    ra = find("ra_deg","ra","raj2000","ra (deg)")
    dec = find("dec_deg","dec","dej2000","dec (deg)")
    if ra is None or dec is None:
        raise KeyError("could not detect RA/Dec columns")
    return ra, dec


def load_catalog(path):
    with open(path,"r",encoding="utf-8") as f:
        R = csv.DictReader(f)
        fields = R.fieldnames
        ra,dec = detect_columns(fields)
        RA = []
        Dec = []
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
    nrm = np.linalg.norm(Xgal,axis=1,keepdims=True)+1e-15
    return Xgal/nrm


def galactic_lb_to_xyz(l_deg,b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)
    v = np.array([x,y,z])
    return v/(np.linalg.norm(v)+1e-15)

# ============================================================
# geometry helpers
# ============================================================

def spherical_dist_matrix(X):
    dots = X @ X.T
    np.clip(dots,-1,1,out=dots)
    return np.degrees(np.arccos(dots))


def remnant_sign(X,axis_vec):
    axis_vec = axis_vec/(np.linalg.norm(axis_vec)+1e-15)
    R = X @ axis_vec
    s = np.ones_like(R)
    s[R<0] = -1
    return s


def mean_pairwise_distance(X):
    D = spherical_dist_matrix(X)
    N = len(D)
    return np.sum(D)/ (N*(N-1))


def knn_thickness(X,k=12):
    D = spherical_dist_matrix(X)
    np.fill_diagonal(D,np.inf)
    idx = np.argpartition(D,kth=k,axis=1)[:, :k]
    neigh = np.take_along_axis(D,idx,axis=1)
    return np.mean(neigh)


def harmonic_moments(X,l_max=3):
    """
    compute low-l spherical harmonic power using real spherical harmonics.
    no coefficients needed; we only compute relative power.

    output: sum over l=1..l_max of total power in that hemisphere sample
    """
    # convert xyz -> (theta,phi)
    x,y,z = X[:,0],X[:,1],X[:,2]
    theta = np.arccos(z)            # polar
    phi   = np.arctan2(y,x)         # azimuth

    # real spherical harmonics Y_lm
    # normalized versions optional — we only compare differences
    powers = 0.0
    for l in range(1,l_max+1):
        for m in range(-l,l+1):
            Y = real_sph_harm(l,m,theta,phi)
            powers += np.sum(Y*Y)
    return powers


def real_sph_harm(l,m,theta,phi):
    """real spherical harmonics."""
    if m > 0:
        K = np.sqrt((2*l+1)/(4*np.pi) * math.factorial(l-m)/math.factorial(l+m))
        return np.sqrt(2) * K * np.cos(m*phi) * scipy_lpmv(m, l, np.cos(theta))
    elif m < 0:
        m2 = -m
        K = np.sqrt((2*l+1)/(4*np.pi) * math.factorial(l-m2)/math.factorial(l+m2))
        return np.sqrt(2) * K * np.sin(m2*phi) * scipy_lpmv(m2, l, np.cos(theta))
    else:
        K = np.sqrt((2*l+1)/(4*np.pi))
        return K * scipy_lpmv(0, l, np.cos(theta))


# ============================================================
# main test statistic
# ============================================================

def compute_dilation_metrics(Xgal,axis_vec):
    sign = remnant_sign(Xgal,axis_vec)

    X_pos = Xgal[sign>0]
    X_neg = Xgal[sign<0]

    # 1. pairwise dilation
    d_pos = mean_pairwise_distance(X_pos)
    d_neg = mean_pairwise_distance(X_neg)
    delta_pair = d_pos - d_neg

    # 2. kNN thickness
    t_pos = knn_thickness(X_pos)
    t_neg = knn_thickness(X_neg)
    delta_knn = t_pos - t_neg

    # 3. harmonic moment difference
    h_pos = harmonic_moments(X_pos)
    h_neg = harmonic_moments(X_neg)
    delta_h = h_pos - h_neg

    # combined scalar statistic
    S = abs(delta_pair) + abs(delta_knn) + abs(delta_h)

    return S, delta_pair, delta_knn, delta_h


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
# main
# ============================================================

def main(path):
    print("[info] loading frb catalog...")
    RA,Dec = load_catalog(path)
    N = len(RA)
    print(f"[info] N_FRB = {N}")

    print("[info] converting to galactic coords...")
    Xgal = radec_to_galactic_xyz(RA,Dec)

    # unified axis
    axis = galactic_lb_to_xyz(159.8,-0.5)

    print("[info] computing real dilation metrics...")
    S_real, d_pair, d_knn, d_h = compute_dilation_metrics(Xgal,axis)

    print("[info] building MC null (isotropic skies)...")
    S_null = []
    for _ in tqdm(range(2000),desc="MC"):
        Xmc = random_isotropic(N)
        Smc,_,_,_ = compute_dilation_metrics(Xmc,axis)
        S_null.append(Smc)
    S_null = np.array(S_null)

    mu = float(np.mean(S_null))
    sd = float(np.std(S_null))
    p = (1 + np.sum(S_null >= S_real)) / (len(S_null) + 1.0)

    print("================================================")
    print(" frb remnant-time manifold dilation test (test 72)")
    print("================================================")
    print(f"delta_pairwise   = {d_pair:.6f}")
    print(f"delta_knn        = {d_knn:.6f}")
    print(f"delta_harmonic   = {d_h:.6f}")
    print("------------------------------------------------")
    print(f"S_real           = {S_real:.6f}")
    print(f"null mean S      = {mu:.6f}")
    print(f"null std S       = {sd:.6f}")
    print(f"p-value          = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  - low p  -> manifold geometry is directionally dilated/compressed")
    print("             with respect to the remnant-time axis.")
    print("  - high p -> manifold geometry is symmetric between time hemispheres.")
    print("================================================")
    print("test 72 complete.")
    print("================================================")


if __name__ == "__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_manifold_dilation_test72.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
