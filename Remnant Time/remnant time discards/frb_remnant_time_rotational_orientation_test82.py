#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB remnant-time rotational orientation test (test 82)
-----------------------------------------------------

Goal:
    Test whether small-scale FRB geometry carries a remnant-time–
    dependent orientation signature ("A-node rotational orientation").

    We treat local FRB neighbourhoods as defining a headless
    orientation in the tangent plane, encode this as a spin-2
    order parameter, and compare the orientation order between
    the R>0 and R<0 hemispheres.

Method:
    1. Load FRB catalog and convert to galactic xyz.
    2. For each FRB, build k-nearest-neighbour set in 3D.
    3. Project neighbour offsets into the local tangent plane
       (e1,e2) at that FRB.
    4. Compute 2x2 covariance of tangential offsets; take the
       principal eigenvector as the local orientation axis.
    5. Encode orientation as a spin-2 phase:
           psi  = atan2(v∙e2, v∙e1)
           z    = exp(2i psi)
    6. Compute complex orientation order parameters:
           S_plus  = mean(z) over R>0
           S_minus = mean(z) over R<0
           A       = |S_plus - S_minus|
    7. Null: shuffle remnant signs 2000 times, recompute A.
    8. Report A_real, null mean/std, and p-value.

Interpretation:
    low p  -> remnant-time hemispheres carry different small-scale
              orientation tags, consistent with an A-node rotational
              orientation field.
    high p -> orientation structure symmetric; no evidence for tags.
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
    low = [f.lower() for f in fields]
    def find(*names):
        for n in names:
            if n.lower() in low:
                return fields[low.index(n.lower())]
        return None
    ra = find("ra_deg", "ra", "raj2000", "ra (deg)")
    dec = find("dec_deg", "dec", "dej2000", "dec (deg)")
    if ra is None or dec is None:
        raise KeyError("could not detect RA/Dec columns")
    return ra, dec


def load_catalog(path):
    with open(path, "r", encoding="utf-8") as f:
        R = csv.DictReader(f)
        ra, dec = detect_columns(R.fieldnames)
        RA, Dec = [], []
        for row in R:
            RA.append(float(row[ra]))
            Dec.append(float(row[dec]))
    return np.array(RA), np.array(Dec)

# ============================================================
# coordinate transforms
# ============================================================

def radec_xyz(RA, Dec):
    RA = np.radians(RA)
    Dec = np.radians(Dec)
    x = np.cos(Dec) * np.cos(RA)
    y = np.cos(Dec) * np.sin(RA)
    z = np.sin(Dec)
    return np.vstack([x, y, z]).T


def M_eq_to_gal():
    return np.array([
        [-0.054875539390, -0.873437104725, -0.483834991775],
        [ 0.494109453633, -0.444829594298,  0.746982248696],
        [-0.867666135681, -0.198076389622,  0.455983794523],
    ])


def radec_to_galactic_xyz(RA, Dec):
    Xeq = radec_xyz(RA, Dec)
    M = M_eq_to_gal()
    X = Xeq @ M.T
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-15)


def gal_lb_xyz(l, b):
    l = np.radians(l)
    b = np.radians(b)
    v = np.array([
        np.cos(b) * np.cos(l),
        np.cos(b) * np.sin(l),
        np.sin(b)
    ])
    return v / (np.linalg.norm(v) + 1e-15)

# ============================================================
# remnant sign and tangent basis
# ============================================================

def remnant_sign(X, axis):
    R = X @ axis
    s = np.ones(len(X))
    s[R < 0] = -1
    return s


def tangent_basis_at_point(x):
    """
    Construct an orthonormal tangent basis (e1,e2) at point x on the unit sphere.
    e1 and e2 live in the tangent plane perpendicular to x.
    """
    z = x / (np.linalg.norm(x) + 1e-15)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(z, tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = tmp - np.dot(tmp, z) * z
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-15:
        # fallback
        e1 = np.array([0.0, 1.0, 0.0])
        e1 -= np.dot(e1, z) * z
        e1_norm = np.linalg.norm(e1)
    e1 /= (e1_norm + 1e-15)
    e2 = np.cross(z, e1)
    e2 /= (np.linalg.norm(e2) + 1e-15)
    return e1, e2

# ============================================================
# local orientation extraction
# ============================================================

def local_orientations(X, k=20, anisotropy_thresh=0.1):
    """
    For each FRB i, find k nearest neighbors, project offsets into
    tangent plane, compute 2x2 covariance, and extract principal
    orientation eigenvector.

    Returns:
        psi   : orientation angles in radians (headless, modulo pi)
        valid : boolean mask indicating which FRBs have well-defined
                (sufficiently anisotropic) local orientation.
    """
    N = len(X)
    nbr = NearestNeighbors(n_neighbors=min(k+1, N),
                           algorithm="ball_tree").fit(X)
    dist, idx = nbr.kneighbors(X)

    psi = np.zeros(N)
    valid = np.zeros(N, dtype=bool)

    for i in range(N):
        xi = X[i]
        e1, e2 = tangent_basis_at_point(xi)

        neigh_idx = idx[i, 1:]  # exclude itself
        if len(neigh_idx) < 3:
            continue

        # project neighbour offsets into tangent plane
        U = []
        for j in neigh_idx:
            dx = X[j] - xi
            # project to tangent plane components
            u = np.dot(dx, e1)
            v = np.dot(dx, e2)
            U.append([u, v])
        U = np.array(U)

        # covariance in tangent plane
        C = np.cov(U.T)
        # ensure symmetric
        C = 0.5 * (C + C.T)

        # eigen-decomposition
        w, v = np.linalg.eigh(C)  # w sorted ascending
        # largest eigenvalue and eigenvector
        idx_max = np.argmax(w)
        lam1 = w[idx_max]
        lam2 = w[1 - idx_max]

        if lam1 + lam2 <= 0:
            continue

        # anisotropy measure
        a = (lam1 - lam2) / (lam1 + lam2)
        if a < anisotropy_thresh:
            # nearly isotropic; orientation not well defined
            continue

        v_main = v[:, idx_max]  # 2-component vector in (e1,e2) basis
        # orientation angle in tangent plane
        ang = math.atan2(v_main[1], v_main[0])  # in radians

        # headless: modulo pi; spin-2 will handle it
        psi[i] = ang
        valid[i] = True

    return psi, valid

# ============================================================
# main
# ============================================================

def main(path):

    print("[info] loading FRB catalog...")
    RA, Dec = load_catalog(path)
    X = radec_to_galactic_xyz(RA, Dec)
    N = len(X)
    print(f"[info] N_FRB = {N}")

    axis = gal_lb_xyz(159.8, -0.5)

    print("[info] computing remnant signs...")
    s = remnant_sign(X, axis)

    print("[info] computing local orientations (spin-2)...")
    psi, valid = local_orientations(X, k=20, anisotropy_thresh=0.1)

    # spin-2 encoding
    z = np.exp(2j * psi)
    z[~valid] = 0.0 + 0.0j  # ignore invalid orientations

    # real hemispheres
    z_pos = z[(s > 0) & valid]
    z_neg = z[(s < 0) & valid]

    if len(z_pos) < 10 or len(z_neg) < 10:
        print("[error] not enough valid orientations in one hemisphere.")
        return

    S_pos = np.mean(z_pos)
    S_neg = np.mean(z_neg)
    A_real = abs(S_pos - S_neg)

    # ========================================================
    # null: shuffle signs
    # ========================================================

    print("[info] building null (2000 shuffles)...")
    null = []

    for _ in tqdm(range(2000)):
        sh = np.copy(s)
        np.random.shuffle(sh)

        z_p = z[(sh > 0) & valid]
        z_n = z[(sh < 0) & valid]

        if len(z_p) < 10 or len(z_n) < 10:
            null.append(0.0)
            continue

        Sp = np.mean(z_p)
        Sn = np.mean(z_n)
        A = abs(Sp - Sn)
        null.append(A)

    null = np.array(null)
    mu = float(np.mean(null))
    sd = float(np.std(null))
    p = (1 + np.sum(null >= A_real)) / (len(null) + 1)

    # ========================================================
    # output
    # ========================================================

    print("================================================")
    print(" FRB REMNANT-TIME ROTATIONAL ORIENTATION TEST (82)")
    print("================================================")
    print(f"S_pos (complex)           = {S_pos}")
    print(f"S_neg (complex)           = {S_neg}")
    print(f"A_real (|S_pos-S_neg|)    = {A_real:.6f}")
    print("------------------------------------------------")
    print(f"null mean A               = {mu:.6f}")
    print(f"null std A                = {sd:.6f}")
    print(f"p-value                   = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> remnant-time hemispheres carry different")
    print("            small-scale orientation tags, consistent with")
    print("            an A-node rotational orientation field.")
    print("  high p -> orientation structure is symmetric; no tags.")
    print("================================================")
    print("test 82 complete.")
    print("================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_rotational_orientation_test82.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
