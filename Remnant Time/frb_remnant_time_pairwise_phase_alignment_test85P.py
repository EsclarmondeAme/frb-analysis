#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB REMNANT-TIME PAIRWISE PHASE-ALIGNMENT TEST (TEST 85P)
---------------------------------------------------------

goal:
    construct a phase-memory estimator that does NOT rely on a
    global hemisphere slab average.

    instead of comparing harmonic phases between R>0 and R<0
    hemispheres as two global aggregates, we measure how strongly
    the *pairwise phase-alignment* of the spherical-harmonic field
    differs between:

        - pairs with the same remnant-time sign
        - pairs with opposite remnant-time sign

    this reduces the built-in geometry trap of a global step
    function and tests whether remnant-time labels organize the
    local harmonic phase field in a non-random way.

design:
    - load FRB catalog
    - detect RA, Dec columns
    - detect remnant-time scalar column (if present) and reduce it
      to a sign; if absent, fall back to the unified-axis hemisphere
      sign (as in previous tests)
    - convert to galactic xyz
    - compute Y_lm(theta,phi) for each FRB using a real-valued
      basis up to l_max (default: 8)
    - for each FRB i, build a feature vector v_i given by the
      Y_lm components, normalized to unit length
    - compute the Gram matrix G_ij = v_i · v_j (pairwise
      phase-alignment score in [-1,1])
    - define:
          S_same = mean(G_ij over i<j with sign_i = sign_j)
          S_opp  = mean(G_ij over i<j with sign_i != sign_j)
          Delta_real = S_same - S_opp
    - build a null by shuffling remnant signs across fixed
      positions (geometry fixed) and recomputing Delta under each
      shuffle:
          Delta_null[k]
      then evaluate a two-sided p-value from |Delta_null|.

interpretation:
    - low p (Delta_real far from null) →
          pairwise phase-alignment knows about remnant-time
          sign, beyond what is expected from geometry alone.
    - high p →
          phase-alignment is insensitive to remnant-time labels
          once the global hemisphere average is removed.

this test is deliberately global-but-not-slabbed: it uses all
pairs on the sky, but does not construct a global hemisphere
contrast field; the only role of remnant sign is as a pairwise
grouping label.
"""

import sys
import csv
import math
import numpy as np
from tqdm import tqdm


# ------------------------------------------------------------
# column detection
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

    # candidate remnant-time scalar column
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
# coordinate transforms
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
    extended to negative m via the usual relation in Ylm().
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
    # convert xyz -> (theta,phi)
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
          positive and negative values, use sign(Rscalar)
        - otherwise, fall back to hemisphere sign relative to the
          unified axis in galactic coordinates.
    """
    if have_scalar and np.isfinite(Rscalar).sum() > 0:
        s = np.sign(Rscalar)
        # ensure we have both + and -; otherwise useless
        if np.any(s > 0) and np.any(s < 0):
            print("[info] using scalar remnant-time column as sign source.")
            return s
        else:
            print("[warn] remnant scalar column is single-signed; "
                  "falling back to unified-axis hemisphere.")
    # fallback: hemisphere sign w.r.t. unified axis
    axis = axis_vec / np.linalg.norm(axis_vec)
    proj = Xgal @ axis
    s = np.sign(proj)
    # guard against zeros
    s[s == 0] = 1
    print("[info] using unified-axis hemisphere sign (fallback).")
    return s


# ------------------------------------------------------------
# pairwise phase-alignment estimator
# ------------------------------------------------------------

def compute_pairwise_alignment(Y, sign):
    """
    Given:
        Y: (N x M) feature matrix (harmonic basis vectors)
        sign: (N,) in {+1,-1} remnant-time sign labels

    returns:
        Delta_real: mean(G_same) - mean(G_opp)
        N_same: number of same-sign pairs
        N_opp:  number of opposite-sign pairs
        G: Gram matrix (N x N) for reuse in null shuffles
    """
    # normalize rows of Y to unit vectors
    norms = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-15
    V = Y / norms

    # Gram matrix: pairwise dot products (alignment scores)
    G = V @ V.T
    N = V.shape[0]

    # upper-triangular mask (i<j)
    iu = np.triu_indices(N, k=1)

    s = np.sign(sign).astype(int)
    s[s == 0] = 1

    same_mask = (s[:, None] * s[None, :] > 0)
    opp_mask = (s[:, None] * s[None, :] < 0)

    same_mask = same_mask[iu]
    opp_mask = opp_mask[iu]

    G_flat = G[iu]

    G_same = G_flat[same_mask]
    G_opp = G_flat[opp_mask]

    N_same = len(G_same)
    N_opp = len(G_opp)

    if N_same < 10 or N_opp < 10:
        print("[warn] too few same/opposite pairs; estimator unstable.")
        return np.nan, N_same, N_opp, G

    Delta_real = float(np.mean(G_same) - np.mean(G_opp))
    return Delta_real, N_same, N_opp, G


def build_null_distribution(G, sign, n_null=2000, seed=42):
    """
    Build null distribution for Delta = mean(G_same) - mean(G_opp)
    under random shuffles of remnant sign labels.
    """
    rng = np.random.RandomState(seed)
    N = len(sign)
    iu = np.triu_indices(N, k=1)
    G_flat = G[iu]

    null_vals = []
    s0 = np.sign(sign).astype(int)
    s0[s0 == 0] = 1

    for _ in tqdm(range(n_null), desc="null (pairwise)", leave=False):
        s = np.array(s0)
        rng.shuffle(s)
        same_mask = (s[:, None] * s[None, :] > 0)
        opp_mask = (s[:, None] * s[None, :] < 0)
        same_mask = same_mask[iu]
        opp_mask = opp_mask[iu]

        G_same = G_flat[same_mask]
        G_opp = G_flat[opp_mask]

        if len(G_same) < 10 or len(G_opp) < 10:
            null_vals.append(np.nan)
            continue

        Delta = float(np.mean(G_same) - np.mean(G_opp))
        null_vals.append(Delta)

    null_vals = np.array(null_vals)
    null_vals = null_vals[np.isfinite(null_vals)]

    return null_vals


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(path):
    print("===================================================")
    print("  REMNANT-TIME PAIRWISE PHASE-ALIGNMENT (TEST 85P) ")
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

    print("[info] computing REAL pairwise phase-alignment score...")
    Delta_real, N_same, N_opp, G = compute_pairwise_alignment(Y, sign)

    print("---------------------------------------------------")
    print(f"[info] N_same pairs     = {N_same}")
    print(f"[info] N_opp pairs      = {N_opp}")
    print(f"[info] Delta_real       = mean(G_same) - mean(G_opp) = {Delta_real:.6e}")
    print("---------------------------------------------------")

    print("[info] building null distribution by shuffling signs...")
    null_vals = build_null_distribution(G, sign, n_null=2000, seed=42)
    if null_vals.size == 0:
        print("[error] null distribution empty or invalid (all NaN).")
        print("        cannot compute p-value.")
        print("===================================================")
        print("test 85P complete (FAILED: no null).")
        print("===================================================")
        return

    null_mean = float(np.mean(null_vals))
    null_std = float(np.std(null_vals))
    # two-sided p-value
    p = (1 + np.sum(np.abs(null_vals) >= abs(Delta_real))) / (len(null_vals) + 1)

    print("---------------------------------------------------")
    print(f"null mean   = {null_mean:.6e}")
    print(f"null std    = {null_std:.6e}")
    print(f"p-value     = {p:.6e}")
    print("---------------------------------------------------")
    print("interpretation:")
    print("  - low p  → pairs with the same remnant-time sign")
    print("             have systematically different phase alignment")
    print("             compared to opposite-sign pairs, beyond what")
    print("             is expected from geometry + random labels.")
    print("  - high p → pairwise phase-alignment is insensitive to")
    print("             remnant-time labels once the global slab")
    print("             averaging is removed.")
    print("===================================================")
    print("test 85P complete.")
    print("===================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_pairwise_phase_alignment_test85P.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
