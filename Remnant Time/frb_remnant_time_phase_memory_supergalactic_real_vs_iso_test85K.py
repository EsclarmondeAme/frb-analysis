#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time phase-memory — real vs isotropic geometry for supergalactic masks (test 85K)
----------------------------------------------------------------------------------------------

goal:
    extend the 85J logic from galactic latitude masks to the
    supergalactic system. we want to know whether the phase-memory
    signal is tied to local supercluster geometry or persists when
    we cut in supergalactic latitude |SGB|.

procedure:
    - convert real FRBs (RA,Dec) -> galactic xyz -> supergalactic (SGL,SGB)
    - for each |SGB| cut:
        - select FRBs with |SGB| >= cut
        - compute Z_real = global remnant-time phase-memory
        - generate n_iso isotropic skies restricted to same |SGB| cut
          (sample positions isotropically in the supergalactic frame,
           then rotate back to galactic)
        - compute Z_iso for each isotropic sky
        - geometric p-value:
              p_geom = (1 + # {Z_iso >= Z_real}) / (n_iso + 1)

    if p_geom is tiny, real sky has more phase-memory than
    isotropic geometry under that supergalactic mask.

notes:
    - unified axis still defined in galactic coords: (l,b)=(159.8,-0.5)
    - spherical harmonics are in galactic frame as before
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


# ------------------------------------------------------------
# equatorial -> galactic
# ------------------------------------------------------------

def radec_to_equatorial_xyz(RA, Dec):
    RA = np.radians(RA)
    Dec = np.radians(Dec)
    x = np.cos(Dec) * np.cos(RA)
    y = np.cos(Dec) * np.sin(RA)
    z = np.sin(Dec)
    return np.vstack([x, y, z]).T


def equatorial_to_galactic_matrix():
    return np.array([
        [-0.054875539390, -0.873437104725, -0.483834991775],
        [ 0.494109453633, -0.444829594298,  0.746982248696],
        [-0.867666135681, -0.198076389622,  0.455983794523],
    ])


def radec_to_galactic_xyz(RA, Dec):
    Xeq = radec_to_equatorial_xyz(RA, Dec)
    M = equatorial_to_galactic_matrix()
    Xgal = Xeq @ M.T
    return Xgal / (np.linalg.norm(Xgal, axis=1, keepdims=True) + 1e-15)


# ------------------------------------------------------------
# galactic <-> supergalactic
#   use de Vaucouleurs definition:
#     north SGP at (l,b) = (47.37°, +6.32°)
#     SGL=0 at (l,b) = (137.37°, 0°)
# ------------------------------------------------------------

def lb_to_xyz(l_deg, b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    v = np.array([
        np.cos(b) * np.cos(l),
        np.cos(b) * np.sin(l),
        np.sin(b)
    ])
    return v / (np.linalg.norm(v) + 1e-15)


def build_gal_to_sg_basis():
    # z_sg: north supergalactic pole in galactic coordinates
    l_N, b_N = 47.37, 6.32
    z_sg = lb_to_xyz(l_N, b_N)

    # x_sg: direction of SGL=0, SGB=0 (in galactic coords)
    l0, b0 = 137.37, 0.0
    x_temp = lb_to_xyz(l0, b0)

    # make x_sg orthogonal to z_sg
    x_sg = x_temp - np.dot(x_temp, z_sg) * z_sg
    x_sg /= np.linalg.norm(x_sg) + 1e-15

    # y_sg = z_sg x x_sg (right-handed)
    y_sg = np.cross(z_sg, x_sg)
    y_sg /= np.linalg.norm(y_sg) + 1e-15

    # basis vectors in galactic coords
    return x_sg, y_sg, z_sg


X_SG_X, X_SG_Y, X_SG_Z = build_gal_to_sg_basis()


def galactic_xyz_to_supergalactic_lb(Xgal):
    """
    transform galactic xyz to supergalactic longitude/latitude (deg)
    using the de Vaucouleurs supergalactic system.
    """
    vx = Xgal @ X_SG_X
    vy = Xgal @ X_SG_Y
    vz = Xgal @ X_SG_Z
    sgb_rad = np.arcsin(np.clip(vz, -1.0, 1.0))
    sgl_rad = np.arctan2(vy, vx)
    sgl_deg = np.degrees(sgl_rad) % 360.0
    sgb_deg = np.degrees(sgb_rad)
    return sgl_deg, sgb_deg


def supergalactic_lb_to_galactic_xyz(sgl_deg, sgb_deg):
    """
    construct a unit vector in galactic xyz given supergalactic
    longitude/latitude (deg), using the same basis.
    """
    L = np.radians(sgl_deg)
    B = np.radians(sgb_deg)
    x_sg = np.cos(B) * np.cos(L)
    y_sg = np.cos(B) * np.sin(L)
    z_sg = np.sin(B)
    # convert from SG basis to galactic xyz
    v_gal = (x_sg[..., None] * X_SG_X +
             y_sg[..., None] * X_SG_Y +
             z_sg[..., None] * X_SG_Z)
    return v_gal / (np.linalg.norm(v_gal, axis=-1, keepdims=True) + 1e-15)


# ------------------------------------------------------------
# remnant sign and random isotropic in SG frame
# ------------------------------------------------------------

def galactic_lb_to_xyz(l_deg, b_deg):
    return lb_to_xyz(l_deg, b_deg)


def remnant_sign(X, axis_vec):
    axis = axis_vec / (np.linalg.norm(axis_vec) + 1e-15)
    return np.where(X @ axis > 0, 1, -1)


def random_isotropic_xyz_sg_masked(N, cut_sgb_deg, rng):
    """
    generate N isotropic points restricted to |SGB|>=cut_sgb_deg
    in the supergalactic frame, then convert to galactic xyz.

    we sample in SG spherical coordinates, then rotate to galactic.
    """
    cut_rad = np.radians(cut_sgb_deg)
    sin_cut = np.sin(cut_rad)

    sgl = np.empty(N)
    sgb = np.empty(N)

    for i in range(N):
        while True:
            # isotropic in SG: z_sg = sin(B) uniform in [-1,1]
            z = rng.uniform(-1.0, 1.0)
            if abs(z) >= sin_cut:
                break
        B = np.arcsin(z)
        L = rng.uniform(0.0, 2*np.pi)
        sgl[i] = np.degrees(L)
        sgb[i] = np.degrees(B)

    Xgal_iso = supergalactic_lb_to_galactic_xyz(sgl, sgb)
    return Xgal_iso


# ------------------------------------------------------------
# spherical harmonics (pure numpy) in galactic frame
# ------------------------------------------------------------

def legendre_P(l, m, x):
    m_abs = abs(m)
    Pmm = np.ones_like(x)
    if m_abs > 0:
        somx2 = np.sqrt(1 - x*x)
        fact = 1.0
        for _ in range(m_abs):
            Pmm *= -fact * somx2
            fact += 2.0
    if l == m_abs:
        return Pmm
    Pm1m = x * (2*m_abs + 1) * Pmm
    if l == m_abs + 1:
        return Pm1m
    Pll = None
    for ll in range(m_abs + 2, l + 1):
        Pll = ((2*ll - 1)*x*Pm1m - (ll + m_abs - 1)*Pmm) / (ll - m_abs)
        Pmm, Pm1m = Pm1m, Pll
    return Pll


def Ylm_pure(l, m, theta, phi):
    x = np.cos(theta)
    Plm = legendre_P(l, m, x)
    K = math.sqrt((2*l + 1) / (4*math.pi) *
                  math.factorial(l - abs(m)) /
                  math.factorial(l + abs(m)))
    if m > 0:
        return math.sqrt(2)*K*Plm*np.cos(m*phi)
    elif m < 0:
        return math.sqrt(2)*K*Plm*np.sin(abs(m)*phi)
    else:
        return K*Plm


def compute_Ylm_matrix(X, lmax=8):
    theta = np.arccos(np.clip(X[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2*np.pi)
    Y = np.zeros((len(X), (lmax+1)*(lmax+1)))
    idx = 0
    for l in range(lmax+1):
        for m in range(-l, l+1):
            Y[:, idx] = Ylm_pure(l, m, theta, phi)
            idx += 1
    return Y


def phase_memory(Y, sgn):
    pos = (sgn > 0)
    neg = (sgn < 0)
    if np.sum(pos) < 2 or np.sum(neg) < 2:
        return np.nan
    A_pos = np.sum(Y[pos], axis=0)
    A_neg = np.sum(Y[neg], axis=0)
    dphi = np.abs(np.arctan2(A_pos - A_neg, np.ones_like(A_pos)))
    return np.mean(dphi)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(path, n_iso=500, seed=321):
    print("[info] loading real catalog…")
    RA, Dec = load_catalog(path)

    print("[info] converting to galactic xyz…")
    Xgal = radec_to_galactic_xyz(RA, Dec)

    print("[info] converting to supergalactic coords…")
    sgl_real, sgb_real = galactic_xyz_to_supergalactic_lb(Xgal)

    axis = galactic_lb_to_xyz(159.8, -0.5)

    cuts = [0.0, 10.0, 20.0, 30.0]

    results = []
    rng = np.random.RandomState(seed)

    for cut in cuts:
        print(f"[info] --- |SGB|>={cut:.1f}° ---")

        mask = np.abs(sgb_real) >= cut
        Xc = Xgal[mask]
        N = len(Xc)
        if N < 10:
            print(f"[warn] too few FRBs after cut: N={N}, skipping")
            continue

        print(f"[info] real-sky N={N}")
        sgn_real = remnant_sign(Xc, axis)
        Y_real = compute_Ylm_matrix(Xc, lmax=8)
        Z_real = phase_memory(Y_real, sgn_real)
        print(f"[info] real-sky Z={Z_real:.6f}")

        # isotropic ensemble in SG frame with same |SGB| cut
        Z_iso_list = []
        for _ in tqdm(range(n_iso), desc=f"iso |SGB|>={cut:.1f}", leave=False):
            X_iso = random_isotropic_xyz_sg_masked(N, cut, rng)
            sgn_iso = remnant_sign(X_iso, axis)
            Y_iso = compute_Ylm_matrix(X_iso, lmax=8)
            Z_iso = phase_memory(Y_iso, sgn_iso)
            Z_iso_list.append(Z_iso)

        Z_iso_arr = np.array(Z_iso_list)
        iso_mean = float(np.mean(Z_iso_arr))
        iso_std  = float(np.std(Z_iso_arr))
        p_geom = (1 + np.sum(Z_iso_arr >= Z_real)) / (len(Z_iso_arr) + 1)

        print(f"[info] iso mean Z={iso_mean:.6f}, std={iso_std:.6f}, "
              f"geom p={p_geom:.6f}")

        results.append({
            "cut": cut,
            "N": N,
            "Z_real": float(Z_real),
            "iso_mean": iso_mean,
            "iso_std": iso_std,
            "p_geom": float(p_geom),
        })

    print("================================================================")
    print(" REAL VS ISOTROPIC GEOMETRY FOR SUPERGALACTIC MASKS (TEST 85K) ")
    print("================================================================")
    for r in results:
        print(f"|SGB|>={r['cut']:4.1f}°  N={r['N']:4d}  "
              f"Z_real={r['Z_real']:.6f}  "
              f"iso_mean={r['iso_mean']:.6f}  "
              f"iso_std={r['iso_std']:.6f}  "
              f"p_geom={r['p_geom']:.6f}")
    print("================================================================")
    print("interpretation:")
    print("  if p_geom is large, real-sky phase-memory under |SGB| cut")
    print("  is consistent with isotropic geometry tied to the local")
    print("  supercluster plane. if p_geom is tiny, there is extra")
    print("  structure in the real FRB sky beyond that geometry.")
    print("================================================================")
    print("test 85K complete.")
    print("================================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_phase_memory_supergalactic_real_vs_iso_test85K.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
