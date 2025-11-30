#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB LATENT FRACTAL–DIMENSION GRADIENT TEST (TEST 70)
----------------------------------------------------

Goal:
  Estimate a local fractal dimension around each FRB on the celestial
  sphere, then test whether this "clumpiness field" is correlated with
  angle from the unified axis (theta_unified).

If the FRB manifold is purely isotropic (given the footprint), the local
dimension D_i should not know about theta_unified.

If there is a real anisotropic manifold, we may see a systematic trend:
  - different D_i near the axis vs far away
  - |corr(D_i, cos(theta_unified))| larger than random.

Test statistic:
  D_i  : local correlation-dimension estimate at FRB i
  x_i  : cos(theta_unified_i)
  rho  : Pearson correlation between D and x

Null hypothesis:
  The association between D and theta_unified is random.
  We preserve the geometry and selection function and perform a
  permutation test by shuffling theta_unified across FRBs.

Outputs:
  - N_valid     : number of FRBs with a valid local dimension estimate
  - rho_real    : correlation between D and cos(theta_unified)
  - null mean   : mean(|rho_null|) under permutation
  - null std    : std(|rho_null|)
  - p-value     : fraction of nulls with |rho_null| >= |rho_real|
"""

import sys
import csv
import math
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree


# ============================================================
# utilities
# ============================================================

def load_frb_catalog(path):
    """
    Load RA, Dec, and theta_unified from frbs_unified.csv.

    Expected columns (lowercase):
      - ra
      - dec
      - theta_unified
    """
    RA = []
    Dec = []
    theta_u = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames

        # be robust to small naming changes
        def pick(possible, cols):
            for c in possible:
                if c in cols:
                    return c
            return None

        ra_key = pick(["ra", "RA", "ra_deg", "RA_deg"], cols)
        dec_key = pick(["dec", "Dec", "dec_deg", "Dec_deg"], cols)
        th_key = pick(["theta_unified", "theta_u", "theta"], cols)

        if ra_key is None or dec_key is None or th_key is None:
            raise RuntimeError(
                "missing one of required columns: ra/dec/theta_unified"
            )

        for row in reader:
            try:
                RA.append(float(row[ra_key]))
                Dec.append(float(row[dec_key]))
                theta_u.append(float(row[th_key]))
            except ValueError:
                # skip malformed rows
                continue

    RA = np.array(RA, dtype=float)
    Dec = np.array(Dec, dtype=float)
    theta_u = np.array(theta_u, dtype=float)

    if len(RA) == 0:
        raise RuntimeError("no valid FRBs loaded from catalog.")

    return RA, Dec, theta_u


def radec_to_xyz(ra_deg, dec_deg):
    """
    Convert arrays of RA/Dec in degrees to unit Cartesian vectors.
    """
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])


def local_fractal_dimension(X, k1=5, k2=20):
    """
    Estimate a local correlation dimension D_i at each point using k-NN distances.

    For each point i:
      - r1 = distance to k1-th nearest neighbour
      - r2 = distance to k2-th nearest neighbour
      - D_i ≈ ln(k2/k1) / ln(r2 / r1)

    Returns:
      D : array of length N with NaN for points where estimate is invalid.
    """
    N = X.shape[0]
    tree = cKDTree(X)

    # query up to k2 neighbours (plus self)
    dists, idxs = tree.query(X, k=k2 + 1)
    # dists[:, 0] = 0 (self); we want rank k1 and k2

    D = np.full(N, np.nan, dtype=float)

    for i in range(N):
        r1 = dists[i, k1]
        r2 = dists[i, k2]

        # guard against degenerate neighbourhoods
        if not np.isfinite(r1) or not np.isfinite(r2):
            continue
        if r1 <= 0 or r2 <= r1:
            continue

        num = math.log(float(k2) / float(k1))
        den = math.log(r2 / r1)
        if den == 0:
            continue

        D[i] = num / den

    return D


def pearson_corr(x, y):
    """
    Compute Pearson correlation between two 1D arrays, ignoring NaNs.
    Returns 0.0 if variance degenerates.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 3:
        return 0.0

    x_mean = x.mean()
    y_mean = y.mean()
    dx = x - x_mean
    dy = y - y_mean

    sx = dx.std()
    sy = dy.std()
    if sx == 0 or sy == 0:
        return 0.0

    rho = np.mean(dx * dy) / (sx * sy)
    return float(rho)


# ============================================================
# test statistic
# ============================================================

def fractal_axis_score(RA, Dec, theta_u_deg, k1=5, k2=20):
    """
    Compute the real test statistic:

      rho_real = corr( D_i , cos(theta_unified_i) )

    where D_i is the local fractal dimension and theta_unified is in degrees.
    """
    X = radec_to_xyz(RA, Dec)
    D = local_fractal_dimension(X, k1=k1, k2=k2)

    # convert theta_unified to cos(theta)
    theta_rad = np.radians(theta_u_deg)
    x_axis = np.cos(theta_rad)

    rho = pearson_corr(D, x_axis)

    # also return N_valid for diagnostics
    N_valid = np.isfinite(D).sum()
    return rho, N_valid, D, x_axis


# ============================================================
# permutation null
# ============================================================

def permutation_null(D, x_axis, n_mc=2000, rng=None):
    """
    Permutation test:
      - keep D fixed (geometry, selection, manifold)
      - shuffle x_axis across FRBs
      - recompute |rho| each time

    Returns:
      null_abs_rho : array of length n_mc
    """
    if rng is None:
        rng = np.random.default_rng()

    D = np.asarray(D, float)
    x_axis = np.asarray(x_axis, float)
    mask = np.isfinite(D) & np.isfinite(x_axis)
    D = D[mask]
    x_axis = x_axis[mask]

    N = D.size
    if N < 10:
        raise RuntimeError("too few valid points for permutation null.")

    null_abs_rho = np.zeros(n_mc, dtype=float)

    for i in tqdm(range(n_mc), desc="MC null"):
        perm = rng.permutation(N)
        x_perm = x_axis[perm]
        rho = pearson_corr(D, x_perm)
        null_abs_rho[i] = abs(rho)

    return null_abs_rho


# ============================================================
# main
# ============================================================

def main(path):
    print("================================================")
    print(" FRB FRACTAL–DIMENSION GRADIENT TEST (TEST 70)")
    print("================================================")

    print("[INFO] loading FRB catalog...")
    RA, Dec, theta_u = load_frb_catalog(path)
    N = len(RA)
    print(f"[INFO] N_FRB (raw) = {N}")

    print("[INFO] computing real fractal-dimension gradient score...")
    rho_real, N_valid, D, x_axis = fractal_axis_score(RA, Dec, theta_u,
                                                      k1=5, k2=20)
    print(f"[INFO] N_valid (with dimension estimate) = {N_valid}")
    print(f"[INFO] rho_real (corr(D, cos(theta_u))) = {rho_real:.6f}")
    abs_rho_real = abs(rho_real)

    print("[INFO] building permutation null (2000 realisations)...")
    null_abs_rho = permutation_null(D, x_axis, n_mc=2000)
    mu = float(null_abs_rho.mean())
    sd = float(null_abs_rho.std())
    p = float(np.mean(null_abs_rho >= abs_rho_real))

    print("------------------------------------------------")
    print(f"abs(rho_real) = {abs_rho_real:.6f}")
    print(f"null mean     = {mu:.6f}")
    print(f"null std      = {sd:.6f}")
    print(f"p-value       = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  - low p  → local fractal dimension aligned with unified axis")
    print("             → structured anisotropic manifold")
    print("  - high p → no preferred alignment; consistent with isotropy / footprint")
    print("================================================")
    print("test 70 complete.")
    print("================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_fractal_dimension_axis_test70.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
