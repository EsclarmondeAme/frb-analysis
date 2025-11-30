#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant time field test (test 70)
------------------------------------
idea:
  treat "remnant time" as a smooth field on the sky that depends on direction.
  along some axis n, the field value at a unit vector x is R(x) = x · n.
  if this field really stretches or squeezes time, regions with similar R(x)
  should see systematically enhanced or suppressed frb event rates.

this test asks:
  does local frb overdensity correlate unusually strongly with the remnant
  field aligned to the unified frb axis, compared to random axes?

interpretation:
  low p  -> frb overdensities prefer one remnant-time sign along the unified axis
  high p -> any coupling between density and the field is consistent with
            random directions on the same sky
"""

import numpy as np
import csv
import sys
from tqdm import tqdm

# ============================================================
# utilities: catalog loading and coordinates
# ============================================================

def detect_columns(fieldnames):
    """detect ra/dec column names automatically."""
    low = [c.lower() for c in fieldnames]

    def find(*candidates):
        for c in candidates:
            if c.lower() in low:
                return fieldnames[low.index(c.lower())]
        return None

    ra_key  = find("ra_deg","ra","raj2000","ra_deg_","ra (deg)")
    dec_key = find("dec_deg","dec","dej2000","dec_deg","dec (deg)")

    if ra_key is None or dec_key is None:
        raise KeyError("could not detect ra/dec column names")

    return ra_key, dec_key


def load_catalog(path):
    """load ra/dec from frb csv."""
    with open(path,"r",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        ra_key, dec_key = detect_columns(fields)

        RA, Dec = [], []
        for row in reader:
            RA.append(float(row[ra_key]))
            Dec.append(float(row[dec_key]))

    return np.array(RA), np.array(Dec)


def radec_to_equatorial_xyz(RA,Dec):
    """
    convert ra/dec to j2000 equatorial unit vectors.
    ra,dec are in degrees; output shape is (N,3).
    """
    RA  = np.radians(RA)
    Dec = np.radians(Dec)

    x = np.cos(Dec)*np.cos(RA)
    y = np.cos(Dec)*np.sin(RA)
    z = np.sin(Dec)
    return np.vstack([x,y,z]).T


def equatorial_to_galactic_matrix():
    """
    j2000 equatorial -> galactic rotation matrix.

    this is the standard 3x3 matrix N_J such that
        G = N_J * A
    where A is an equatorial unit vector and G is the
    corresponding galactic unit vector.
    """
    return np.array([
        [-0.054875539390, -0.873437104725, -0.483834991775],
        [ 0.494109453633, -0.444829594298,  0.746982248696],
        [-0.867666135681, -0.198076389622,  0.455983794523],
    ])


def radec_to_galactic_xyz(RA,Dec):
    """
    convert ra/dec (degrees, j2000) to galactic unit vectors.
    """
    Xeq = radec_to_equatorial_xyz(RA,Dec)
    M = equatorial_to_galactic_matrix()
    Xgal = Xeq @ M.T
    # renormalise numerically
    norms = np.linalg.norm(Xgal, axis=1, keepdims=True) + 1e-15
    return Xgal / norms


def galactic_lb_to_xyz(l_deg,b_deg):
    """convert galactic (l,b) in degrees to unit xyz."""
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)
    v = np.array([x,y,z],dtype=float)
    return v / np.linalg.norm(v)


# ============================================================
# remnant field and density
# ============================================================

def spherical_distance_matrix(X):
    """vectorized NxN spherical distance matrix (degrees)."""
    dots = X @ X.T
    np.clip(dots, -1.0, 1.0, out=dots)
    return np.degrees(np.arccos(dots))


def local_density(X, k=20):
    """
    simple local-density estimator on the sphere.

    for each point, find k nearest neighbours in angular distance and
    define density ~ 1 / (mean neighbour separation).

    parameters
    ----------
    X : ndarray, shape (N,3)
        unit vectors in some common coordinate frame.
    k : int
        number of nearest neighbours to include (default 20).

    returns
    -------
    dens : ndarray, shape (N,)
        local density proxy (arbitrary units).
    """
    N = X.shape[0]
    if k >= N:
        raise ValueError("k must be smaller than the number of points")

    D = spherical_distance_matrix(X)
    np.fill_diagonal(D, np.inf)

    # indices of k nearest neighbours
    idx = np.argpartition(D, kth=k, axis=1)[:, :k]
    neigh_dist = np.take_along_axis(D, idx, axis=1)

    mean_sep = np.mean(neigh_dist, axis=1)
    return 1.0 / (mean_sep + 1e-6)


def remnant_field_values(X, axis_vec):
    """
    R_i = x_i · n, projection of each unit vector onto axis_vec.
    """
    axis_vec = axis_vec / (np.linalg.norm(axis_vec) + 1e-15)
    return X @ axis_vec


def pearson_correlation(x,y):
    """simple pearson correlation coefficient between 1d arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    x = x - x.mean()
    y = y - y.mean()
    num = np.sum(x*y)
    den = np.sqrt(np.sum(x*x)*np.sum(y*y)) + 1e-15
    return num / den


def random_axis():
    """draw a single isotropic random unit vector on the sphere."""
    u = np.random.uniform(-1.0,1.0)
    phi = np.random.uniform(0.0, 2.0*np.pi)
    sin_theta = np.sqrt(1.0 - u*u)
    x = sin_theta*np.cos(phi)
    y = sin_theta*np.sin(phi)
    z = u
    return np.array([x,y,z],dtype=float)


# ============================================================
# core statistic
# ============================================================

def compute_remnant_density_correlation(RA,Dec, l_axis=159.8, b_axis=-0.5, k=20):
    """
    compute correlation between remnant field and local density.

    steps:
      1. convert frbs to galactic unit vectors X.
      2. estimate local densities dens using k nearest neighbours.
      3. construct remnant field R_i = x_i · n for unified axis n.
      4. return pearson correlation r = corr(R, dens).
    """
    # frbs in galactic coordinates
    Xgal = radec_to_galactic_xyz(RA,Dec)

    # local density estimate
    dens = local_density(Xgal, k=k)

    # unified frb axis from paper: (l,b) ≈ (159.8°, -0.5°)
    axis_unified = galactic_lb_to_xyz(l_axis, b_axis)

    # remnant field values and correlation
    R = remnant_field_values(Xgal, axis_unified)
    r = pearson_correlation(R, dens)

    return r, dens, Xgal, axis_unified


def monte_carlo_random_axes(Xgal, dens, n_mc=2000):
    """
    build null distribution of |corr(R, dens)| over random axes.

    returns
    -------
    abs_corr_null : ndarray, shape (n_mc,)
        absolute correlation values for random axes.
    best_axis      : ndarray, shape (3,)
        axis that maximises |corr|.
    best_r         : float
        maximal |corr| found in the scan.
    """
    abs_corr = np.empty(n_mc, dtype=float)
    best_r = -np.inf
    best_axis = None

    for i in tqdm(range(n_mc), desc="MC random axes"):
        a = random_axis()
        R = remnant_field_values(Xgal, a)
        r = pearson_correlation(R, dens)
        ar = abs(r)
        abs_corr[i] = ar
        if ar > best_r:
            best_r = ar
            best_axis = a

    return abs_corr, best_axis, best_r


# ============================================================
# main
# ============================================================

def main(path):
    print("[info] loading frb catalog...")
    RA,Dec = load_catalog(path)
    N = len(RA)
    print(f"[info] N_FRB = {N}")

    print("[info] computing remnant–density correlation for unified axis...")
    r_real, dens, Xgal, axis_unified = compute_remnant_density_correlation(RA,Dec)

    print("[info] building null over random axes on the same sky...")
    abs_corr_null, best_axis, best_r = monte_carlo_random_axes(Xgal, dens, n_mc=2000)

    # p-value: how often random axes produce |r| >= |r_real|
    abs_r_real = abs(r_real)
    p = (1.0 + np.sum(abs_corr_null >= abs_r_real)) / (len(abs_corr_null) + 1.0)

    # angle between unified axis and best-coupling axis
    dot = float(np.clip(np.dot(axis_unified, best_axis), -1.0, 1.0))
    sep_best_deg = np.degrees(np.arccos(dot))

    print("================================================")
    print(" frb remnant time field test (test 70)")
    print("================================================")
    print(f"r_unified          = {r_real:.6f}")
    print(f"|r_unified|        = {abs_r_real:.6f}")
    print(f"null mean |r|      = {np.mean(abs_corr_null):.6f}")
    print(f"null std |r|       = {np.std(abs_corr_null):.6f}")
    print(f"p-value (axes)     = {p:.6f}")
    print("------------------------------------------------")
    print("axis scan diagnostics:")
    print(f"max |r| over axes  = {best_r:.6f}")
    print(f"sep(unified,best)  = {sep_best_deg:.2f} deg")
    print("------------------------------------------------")
    print("interpretation:")
    print("  - low p  -> local frb overdensities prefer one remnant-time sign")
    print("             along the unified axis more than random directions do.")
    print("  - high p -> coupling between density and remnant field is within")
    print("             the range expected for random axes on this sky.")
    print("================================================")
    print("test 70 complete.")
    print("================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_field_test70.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
