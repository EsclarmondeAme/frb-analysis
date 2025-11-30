#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB REMNANT-TIME LOCAL PCA PHASE TEST (TEST 85Q)
------------------------------------------------

goal:
    test whether local harmonic structure (Y_lm) in different sky patches
    carries information about the remnant-time sign, *without*
    constructing any global hemisphere slab map.

    in each patch:
        - build a local harmonic basis via PCA (using all FRBs in the patch),
          *without* using remnant labels.
        - project FRBs onto the leading principal component.
        - compare the distributions of that coordinate for R>0 vs R<0.
        - build a null by shuffling remnant signs within the patch.

design:
    - load FRB catalog (RA, Dec; optional scalar remnant-time column).
    - convert to galactic xyz.
    - build remnant sign s_i via:
          (i) scalar remnant column if present and sign-changing; or
          (ii) hemisphere sign relative to the unified axis (fallback).
    - compute real-valued Y_lm up to l_max = 8 for each FRB.
    - define a modest number of sky patches in (l,b) to ensure
      enough FRBs per patch:
          mu = sin(b) = z_gal; bins:
              low-lat:   mu ∈ [-1.0, -0.333)
              mid-lat:   mu ∈ [-0.333, 0.333)
              high-lat:  mu ∈ [0.333, 1.0]
          longitude l:  [0°, 180°) and [180°, 360°)
          → up to 6 patches.
    - for each patch with:
          N_patch >= N_min (default 50)
          and at least N_sign_min (default 15) in both sign groups:
        · standardize Y within the patch (zero mean).
        · compute PCA via SVD and take the first principal component v1.
        · compute projection scores t_i = v1 · Y_i.
        · real statistic:
              Δ_real = mean(t | R>0) − mean(t | R<0).
        · null:
              shuffle remnant signs within patch, recompute Δ, repeat
              n_null = 2000 times.
        · record:
              Δ_real, null_mean, null_std, p_patch, N_patch, N_+, N_-.

interpretation:
    - low p in multiple independent patches →
          local harmonic structure in that region knows about remnant-time
          sign, even though the PCA basis was built sign-blind.
    - high p or lack of usable patches →
          local harmonic structure does not independently predict the
          remnant sign beyond the global effects already captured by
          tests 81, 85, 85P.
"""

import sys
import csv
import math
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------------
# column detection and catalog loading
# ------------------------------------------------------------

def detect_columns(fieldnames):
    low = [c.lower() for c in fieldnames]

    def find(*names):
        for n in names:
            if n.lower() in low:
                return fieldnames[low.index(n.lower())]
        return None

    ra  = find("ra_deg", "ra", "raj2000", "ra (deg)")
    dec = find("dec_deg", "dec", "dej2000", "dec (deg)")

    remnant = None
    for name in fieldnames:
        ln = name.lower()
        if ("remnant" in ln or "r_unified" in ln or "rem_time" in ln) and remnant is None:
            remnant = name

    if ra is None or dec is None:
        raise RuntimeError("could not detect RA/Dec columns in catalog")

    return ra, dec, remnant


def load_catalog(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        ra_col, dec_col, remnant_col = detect_columns(reader.fieldnames)

        RA, Dec = [], []
        Rscalar = []

        for row in reader:
            RA.append(float(row[ra_col]))
            Dec.append(float(row[dec_col]))

            if remnant_col is not None:
                try:
                    Rscalar.append(float(row[remnant_col]))
                except ValueError:
                    Rscalar.append(np.nan)
            else:
                Rscalar.append(np.nan)

    return (
        np.array(RA, dtype=float),
        np.array(Dec, dtype=float),
        np.array(Rscalar, dtype=float),
        ra_col,
        dec_col,
        remnant_col,
    )


# ------------------------------------------------------------
# coordinate transforms: equatorial -> galactic
# ------------------------------------------------------------

def radec_to_xyz(RA_deg, Dec_deg):
    RA = np.radians(RA_deg)
    Dec = np.radians(Dec_deg)
    x = np.cos(Dec) * np.cos(RA)
    y = np.cos(Dec) * np.sin(RA)
    z = np.sin(Dec)
    return np.vstack([x, y, z]).T


def equatorial_to_galactic_matrix():
    # standard J2000 -> Galactic rotation matrix
    return np.array([
        [-0.054875539390, -0.873437104725, -0.483834991775],
        [ 0.494109453633, -0.444829594298,  0.746982248696],
        [-0.867666135681, -0.198076389622,  0.455983794523],
    ])


def radec_to_galactic_xyz(RA_deg, Dec_deg):
    Xeq = radec_to_xyz(RA_deg, Dec_deg)
    M = equatorial_to_galactic_matrix()
    Xgal = Xeq @ M.T
    Xgal /= (np.linalg.norm(Xgal, axis=1, keepdims=True) + 1e-15)
    return Xgal


def galactic_lb_to_axis(l_deg, b_deg):
    l = math.radians(l_deg)
    b = math.radians(b_deg)
    v = np.array([
        math.cos(b) * math.cos(l),
        math.cos(b) * math.sin(l),
        math.sin(b),
    ])
    return v / np.linalg.norm(v)


# ------------------------------------------------------------
# spherical harmonics (real basis)
# ------------------------------------------------------------

def legendre_P(l, m, x):
    """
    Associated Legendre function P_l^m(x) with m >= 0,
    extended to negative m via Ylm_real.
    x can be a numpy array.
    """
    m0 = abs(m)
    # P_m^m
    Pmm = np.ones_like(x)
    if m0 > 0:
        somx2 = np.sqrt(np.maximum(0.0, 1.0 - x*x))
        fact = 1.0
        for _ in range(m0):
            Pmm *= (-fact) * somx2
            fact += 2.0
    if l == m0:
        return Pmm
    # P_{m+1}^m
    Pm1m = x * (2*m0 + 1) * Pmm
    if l == m0 + 1:
        return Pm1m
    # upward recursion
    for ll in range(m0 + 2, l + 1):
        Pll = ((2*ll - 1)*x*Pm1m - (ll + m0 - 1)*Pmm) / (ll - m0)
        Pmm, Pm1m = Pm1m, Pll
    return Pll


def Ylm_real(l, m, theta, phi):
    """
    Real-valued spherical harmonics, orthonormal on the sphere.
    l >= 0, -l <= m <= l.
    theta: polar angle [0,pi]
    phi: azimuth [0,2pi)
    """
    x = np.cos(theta)
    Plm = legendre_P(l, m, x)
    K = math.sqrt((2*l + 1) / (4*math.pi) *
                  math.factorial(l - abs(m)) /
                  math.factorial(l + abs(m)))
    if m > 0:
        return math.sqrt(2.0) * K * Plm * np.cos(m * phi)
    elif m < 0:
        return math.sqrt(2.0) * K * Plm * np.sin(-m * phi)
    else:
        return K * Plm


def compute_Y_matrix(Xgal, lmax=8):
    """
    Compute real Y_lm for all FRBs and all modes up to lmax.
    Returns matrix Y of shape (N, (lmax+1)^2).
    """
    x, y, z = Xgal[:, 0], Xgal[:, 1], Xgal[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2.0 * math.pi)

    N = Xgal.shape[0]
    nmodes = (lmax + 1) * (lmax + 1)
    Y = np.zeros((N, nmodes), dtype=float)

    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y[:, idx] = Ylm_real(l, m, theta, phi)
            idx += 1

    return Y


# ------------------------------------------------------------
# remnant sign handling
# ------------------------------------------------------------

def build_remnant_signs(Rscalar, Xgal, have_scalar, axis_vec):
    """
    Build a +-1 sign array for remnant-time labels.

    priority:
        - if a scalar remnant-time column exists and has both
          positive and negative values, use sign(Rscalar).
        - otherwise, fall back to hemisphere sign relative to
          the unified axis in galactic coordinates.
    """
    if have_scalar and np.isfinite(Rscalar).sum() > 0:
        s = np.sign(Rscalar)
        if np.any(s > 0) and np.any(s < 0):
            print("[info] using scalar remnant-time column as sign source.")
            return s
        else:
            print("[warn] remnant scalar column is single-signed; "
                  "falling back to unified-axis hemisphere.")
    axis = axis_vec / np.linalg.norm(axis_vec)
    proj = Xgal @ axis
    s = np.sign(proj)
    s[s == 0] = 1
    print("[info] using unified-axis hemisphere sign (fallback).")
    return s


# ------------------------------------------------------------
# patching scheme
# ------------------------------------------------------------

def assign_patches(Xgal):
    """
    Assign FRBs to up to 6 sky patches in galactic coords.

    define:
        mu = z_gal = sin(b)
        longitude l in [0,360)

        mu bins:
            bin_mu = 0: [-1.0,  -0.333)
            bin_mu = 1: [-0.333, 0.333)
            bin_mu = 2: [ 0.333, 1.0]

        l bins:
            bin_l = 0: [0°, 180°)
            bin_l = 1: [180°, 360°)

    patch index:
        patch_id = 3*bin_l + bin_mu   (0..5)

    returns:
        patch_ids: array of shape (N,)
    """
    x, y, z = Xgal[:, 0], Xgal[:, 1], Xgal[:, 2]
    mu = z
    l = np.mod(np.degrees(np.arctan2(y, x)), 360.0)

    bin_mu = np.full_like(mu, -1, dtype=int)
    bin_mu[(mu >= -1.0) & (mu < -0.333)] = 0
    bin_mu[(mu >= -0.333) & (mu < 0.333)] = 1
    bin_mu[(mu >= 0.333) & (mu <= 1.0)] = 2

    bin_l = np.zeros_like(l, dtype=int)
    bin_l[(l >= 180.0) & (l < 360.0)] = 1

    patch_id = 3 * bin_l + bin_mu  # 0..5, or -1 for undefined
    return patch_id


# ------------------------------------------------------------
# local PCA test per patch
# ------------------------------------------------------------

def local_pca_patch(Y_patch, sign_patch, n_null=2000, seed=42):
    """
    Perform local PCA on Y_patch (N_patch x M), sign-blind,
    then test whether projections along the leading PC differ
    between positive and negative remnant-time signs.

    returns:
        Δ_real, null_mean, null_std, p, N_patch, N_plus, N_minus
    """
    rng = np.random.RandomState(seed)

    Np = Y_patch.shape[0]
    s = np.sign(sign_patch).astype(int)
    s[s == 0] = 1

    N_plus = np.sum(s > 0)
    N_minus = np.sum(s < 0)

    # standardization
    Y0 = Y_patch - np.mean(Y_patch, axis=0, keepdims=True)

    # PCA via SVD
    U, S, Vt = np.linalg.svd(Y0, full_matrices=False)
    v1 = Vt[0, :]  # leading PC

    # projections
    t = Y0 @ v1

    # real statistic
    t_plus = t[s > 0]
    t_minus = t[s < 0]

    if len(t_plus) < 5 or len(t_minus) < 5:
        return np.nan, np.nan, np.nan, np.nan, Np, N_plus, N_minus

    Delta_real = float(np.mean(t_plus) - np.mean(t_minus))

    # null distribution by shuffling signs (within patch)
    null_vals = []
    for _ in tqdm(range(n_null), desc="null (local PCA)", leave=False):
        s_shuf = np.array(s)
        rng.shuffle(s_shuf)
        tp = t[s_shuf > 0]
        tm = t[s_shuf < 0]
        if len(tp) < 5 or len(tm) < 5:
            null_vals.append(np.nan)
            continue
        Delta = float(np.mean(tp) - np.mean(tm))
        null_vals.append(Delta)

    null_vals = np.array(null_vals)
    null_vals = null_vals[np.isfinite(null_vals)]

    if null_vals.size == 0:
        return Delta_real, np.nan, np.nan, np.nan, Np, N_plus, N_minus

    null_mean = float(np.mean(null_vals))
    null_std = float(np.std(null_vals))
    p = (1 + np.sum(np.abs(null_vals) >= abs(Delta_real))) / (len(null_vals) + 1)

    return Delta_real, null_mean, null_std, p, Np, N_plus, N_minus


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(path):
    print("===================================================")
    print("  REMNANT-TIME LOCAL PCA PHASE TEST (TEST 85Q)     ")
    print("===================================================")

    print(f"[info] loading FRB catalog from: {path}")
    RA, Dec, Rscalar, ra_col, dec_col, remnant_col = load_catalog(path)
    N = len(RA)
    print(f"[info] N_FRB = {N}")
    print(f"[info] RA column: {ra_col}")
    print(f"[info] Dec column: {dec_col}")
    if remnant_col is not None:
        print(f"[info] remnant-time column: {remnant_col}")
    else:
        print("[info] no explicit remnant-time column detected.")

    print("[info] converting to galactic xyz...")
    Xgal = radec_to_galactic_xyz(RA, Dec)

    # unified axis from previous tests
    axis_vec = galactic_lb_to_axis(159.8, -0.5)

    print("[info] building remnant-time sign labels...")
    have_scalar = (remnant_col is not None)
    sign = build_remnant_signs(Rscalar, Xgal, have_scalar, axis_vec)

    print("[info] computing real-valued Y_lm basis (l_max=8)...")
    Y = compute_Y_matrix(Xgal, lmax=8)

    print("[info] assigning sky patches...")
    patch_id = assign_patches(Xgal)

    # parameters for usable patches
    N_min = 50
    N_sign_min = 15
    n_null = 2000

    unique_patches = sorted(p for p in np.unique(patch_id) if p >= 0)

    print("---------------------------------------------------")
    print(f"[info] candidate patches: {unique_patches}")
    print(f"[info] minimum N_patch = {N_min}, minimum N_sign = {N_sign_min}")
    print("---------------------------------------------------")

    results = []

    for p in unique_patches:
        mask = (patch_id == p)
        idx = np.where(mask)[0]
        Np = idx.size
        if Np < N_min:
            print(f"[info] patch {p}: N={Np} < {N_min}, skipping.")
            continue

        Y_patch = Y[idx, :]
        sign_patch = sign[idx]

        N_plus = np.sum(sign_patch > 0)
        N_minus = np.sum(sign_patch < 0)

        if N_plus < N_sign_min or N_minus < N_sign_min:
            print(f"[info] patch {p}: N={Np}, N_plus={N_plus}, N_minus={N_minus} "
                  f"(< {N_sign_min}), skipping.")
            continue

        print("---------------------------------------------------")
        print(f"[info] patch {p}: N={Np}, N_plus={N_plus}, N_minus={N_minus}")
        print("[info] running local PCA + null shuffles...")

        Delta_real, null_mean, null_std, pval, Np2, Np_plus, Np_minus = \
            local_pca_patch(Y_patch, sign_patch, n_null=n_null, seed=42 + p)

        results.append((p, Np2, Np_plus, Np_minus, Delta_real, null_mean, null_std, pval))

        print("---------------------------------------------------")
        print(f"patch {p}:")
        print(f"  Delta_real = {Delta_real:.6e}")
        print(f"  null_mean  = {null_mean:.6e}")
        print(f"  null_std   = {null_std:.6e}")
        print(f"  p_patch    = {pval:.6e}")
        print("---------------------------------------------------")

    print("===================================================")
    print("SUMMARY — LOCAL PCA PHASE TEST (85Q)")
    print("===================================================")
    if not results:
        print("[warn] no patches satisfied N_patch and sign-balance thresholds.")
        print("       test 85Q is inconclusive on this catalog.")
        print("===================================================")
        print("test 85Q complete.")
        print("===================================================")
        return

    print("patch   N_patch   N_plus   N_minus   Delta_real      null_mean      null_std      p_patch")
    for (p, Np, Np_plus, Np_minus, D, nm, ns, pv) in results:
        print(f"{p:5d}  {Np:7d}  {Np_plus:7d}  {Np_minus:8d}  "
              f"{D: .6e}  {nm: .6e}  {ns: .6e}  {pv: .6e}")
    print("===================================================")
    print("interpretation:")
    print("  - patches with low p_patch indicate that the leading local")
    print("    harmonic mode (built without remnant labels) is nonetheless")
    print("    aligned with remnant-time sign.")
    print("  - patches with high p_patch, or absence of usable patches,")
    print("    indicate that local harmonic structure does not independently")
    print("    predict remnant signs beyond the global correlations already")
    print("    captured in tests 81, 85, and 85P.")
    print("===================================================")
    print("test 85Q complete.")
    print("===================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_local_pca_phase_test85Q.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
