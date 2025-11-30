#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant-time phase-memory — real vs isotropic geometry for ecliptic masks (test 85L)
----------------------------------------------------------------------------------------

goal:
    replicate the logic of 85J (galactic) and 85K (supergalactic) in
    the ecliptic frame. we want to test whether remnant-time phase
    coherence survives after masking by ecliptic latitude |β| and after
    comparing to isotropic skies restricted to the same |β| cuts.

motivation:
    ecliptic-frame tests rule out:
      - solar avoidance geometry
      - seasonal scanning patterns
      - RA/Dec biases tied to earth’s orbit
      - annual survey cycle artifacts

procedure:
    - convert RA,Dec -> ecliptic longitude/latitude λ,β
    - for each |β| cut:
         - select FRBs with |β|>=cut
         - compute Z_real
         - generate isotropic skies restricted to |β|>=cut
         - compute Z_iso for each
         - p_geom = #Z_iso >= Z_real / (N_iso)

    if p_geom is tiny, real-sky remnant-time coherence cannot be
    explained by solar-system geometry or seasonal sky coverage.
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
    dec = find("dec_deg","dec","dej2000","dec (deg)")
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
# equatorial -> cartesian
# ------------------------------------------------------------

def radec_to_xyz(RA, Dec):
    RA = np.radians(RA)
    Dec = np.radians(Dec)
    x = np.cos(Dec)*np.cos(RA)
    y = np.cos(Dec)*np.sin(RA)
    z = np.sin(Dec)
    return np.vstack([x,y,z]).T


# ------------------------------------------------------------
# equatorial xyz -> ecliptic xyz rotation
# obliquity of the ecliptic: ε = 23.439291°
# ------------------------------------------------------------

def equatorial_to_ecliptic_xyz(Xeq):
    eps = np.radians(23.439291)
    R = np.array([
        [1,               0,              0],
        [0,  np.cos(eps), np.sin(eps)],
        [0, -np.sin(eps), np.cos(eps)]
    ])
    Xec = Xeq @ R.T
    return Xec / (np.linalg.norm(Xec, axis=1, keepdims=True) + 1e-15)


def ecliptic_xyz_to_lonlat(Xec):
    x, y, z = Xec[:,0], Xec[:,1], Xec[:,2]
    beta = np.arcsin(np.clip(z, -1.0, 1.0))
    lam = np.mod(np.arctan2(y, x), 2*np.pi)
    return np.degrees(lam), np.degrees(beta)


def lonlat_to_ecliptic_xyz(lam_deg, beta_deg):
    lam = np.radians(lam_deg)
    beta = np.radians(beta_deg)
    x = np.cos(beta)*np.cos(lam)
    y = np.cos(beta)*np.sin(lam)
    z = np.sin(beta)
    return np.vstack([x,y,z]).T


# ------------------------------------------------------------
# ecliptic xyz -> equatorial xyz (inverse rotation)
# ------------------------------------------------------------

def ecliptic_to_equatorial_xyz(Xec):
    eps = np.radians(23.439291)
    R = np.array([
        [1,               0,              0],
        [0,  np.cos(eps),-np.sin(eps)],
        [0,  np.sin(eps), np.cos(eps)]
    ])
    Xeq = Xec @ R.T
    return Xeq / (np.linalg.norm(Xeq, axis=1, keepdims=True) + 1e-15)


# ------------------------------------------------------------
# equatorial xyz -> galactic xyz
# (same as earlier, reused)
# ------------------------------------------------------------

def equatorial_to_galactic_matrix():
    return np.array([
        [-0.054875539390, -0.873437104725, -0.483834991775],
        [ 0.494109453633, -0.444829594298,  0.746982248696],
        [-0.867666135681, -0.198076389622,  0.455983794523],
    ])


def equatorial_xyz_to_galactic_xyz(Xeq):
    M = equatorial_to_galactic_matrix()
    Xgal = Xeq @ M.T
    return Xgal / (np.linalg.norm(Xgal, axis=1, keepdims=True)+1e-15)


def lonlat_to_equatorial_xyz(lam_deg, beta_deg):
    Xec = lonlat_to_ecliptic_xyz(lam_deg, beta_deg)
    return ecliptic_to_equatorial_xyz(Xec)


# ------------------------------------------------------------
# remnant sign
# ------------------------------------------------------------

def remnant_sign(Xgal, axis_vec):
    axis = axis_vec/np.linalg.norm(axis_vec)
    return np.where(Xgal @ axis > 0, 1, -1)


# ------------------------------------------------------------
# isotropic sky restricted to |β|>=cut
# ------------------------------------------------------------

def random_isotropic_ecliptic_masked(N, cut_beta_deg, rng):
    cut = np.radians(cut_beta_deg)
    sin_cut = np.sin(cut)
    lam = np.empty(N)
    beta = np.empty(N)

    for i in range(N):
        while True:
            z = rng.uniform(-1.0, 1.0)          # z = sin(beta)
            if abs(z) >= sin_cut:
                break
        beta[i] = np.degrees(np.arcsin(z))
        lam[i]  = rng.uniform(0.0, 360.0)

    Xec = lonlat_to_ecliptic_xyz(lam, beta)
    Xeq = ecliptic_to_equatorial_xyz(Xec)
    Xgal = equatorial_xyz_to_galactic_xyz(Xeq)
    return Xgal


# ------------------------------------------------------------
# spherical harmonics (pure numpy)
# ------------------------------------------------------------

def legendre_P(l,m,x):
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
    Pm1m = x*(2*m_abs+1)*Pmm
    if l == m_abs+1:
        return Pm1m
    for ll in range(m_abs+2, l+1):
        Pll = ((2*ll-1)*x*Pm1m - (ll+m_abs-1)*Pmm)/(ll-m_abs)
        Pmm, Pm1m = Pm1m, Pll
    return Pll


def Ylm_pure(l,m,theta,phi):
    x = np.cos(theta)
    Plm = legendre_P(l,m,x)
    K = math.sqrt((2*l+1)/(4*math.pi) *
                  math.factorial(l-abs(m)) /
                  math.factorial(l+abs(m)))
    if m > 0:
        return math.sqrt(2)*K*Plm*np.cos(m*phi)
    elif m < 0:
        return math.sqrt(2)*K*Plm*np.sin(abs(m)*phi)
    else:
        return K*Plm


def compute_Ylm_matrix(Xgal, lmax=8):
    theta = np.arccos(np.clip(Xgal[:,2], -1.0, 1.0))
    phi   = np.mod(np.arctan2(Xgal[:,1], Xgal[:,0]), 2*np.pi)
    Y = np.zeros((len(Xgal),(lmax+1)*(lmax+1)))
    idx = 0
    for l in range(lmax+1):
        for m in range(-l,l+1):
            Y[:,idx] = Ylm_pure(l,m,theta,phi)
            idx += 1
    return Y


def phase_memory(Y, sgn):
    pos = (sgn>0)
    neg = (sgn<0)
    if np.sum(pos)<2 or np.sum(neg)<2:
        return np.nan
    Apos = np.sum(Y[pos],axis=0)
    Aneg = np.sum(Y[neg],axis=0)
    dphi = np.abs(np.arctan2(Apos - Aneg, np.ones_like(Apos)))
    return np.mean(dphi)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(path, n_iso=500, seed=123):
    print("[info] loading real catalog…")
    RA, Dec = load_catalog(path)

    print("[info] converting to equatorial xyz…")
    Xeq = radec_to_xyz(RA, Dec)

    print("[info] converting to ecliptic coords…")
    Xec = equatorial_to_ecliptic_xyz(Xeq)
    lam_real, beta_real = ecliptic_xyz_to_lonlat(Xec)

    print("[info] converting real positions to galactic for Ylm…")
    Xgal_real = equatorial_xyz_to_galactic_xyz(Xeq)

    # unified axis (galactic)
    axis = np.array([
        np.cos(np.radians(-0.5))*np.cos(np.radians(159.8)),
        np.cos(np.radians(-0.5))*np.sin(np.radians(159.8)),
        np.sin(np.radians(-0.5))
    ])

    cuts = [0.0, 10.0, 20.0, 30.0]
    results = []

    rng = np.random.RandomState(seed)

    for cut in cuts:
        print(f"[info] --- |β|>={cut:.1f}° ---")

        mask = np.abs(beta_real) >= cut
        Xgal_c = Xgal_real[mask]
        N = len(Xgal_c)
        print(f"[info] real-sky N={N}")

        sgn_real = remnant_sign(Xgal_c, axis)
        Y_real = compute_Ylm_matrix(Xgal_c, lmax=8)
        Z_real = phase_memory(Y_real, sgn_real)
        print(f"[info] real-sky Z={Z_real:.6f}")

        Z_iso_list = []
        for _ in tqdm(range(n_iso), desc=f"iso |β|>={cut:.1f}", leave=False):
            Xgal_iso = random_isotropic_ecliptic_masked(N, cut, rng)
            sgn_iso  = remnant_sign(Xgal_iso, axis)
            Y_iso    = compute_Ylm_matrix(Xgal_iso, lmax=8)
            Z_iso    = phase_memory(Y_iso, sgn_iso)
            Z_iso_list.append(Z_iso)

        Z_iso = np.array(Z_iso_list)
        iso_mean = float(np.mean(Z_iso))
        iso_std  = float(np.std(Z_iso))
        p_geom   = (1 + np.sum(Z_iso >= Z_real)) / (len(Z_iso)+1)

        print(f"[info] iso mean Z={iso_mean:.6f}, iso_std={iso_std:.6f}, p_geom={p_geom:.6f}")

        results.append({
            "cut": cut,
            "N": N,
            "Z_real": float(Z_real),
            "iso_mean": iso_mean,
            "iso_std": iso_std,
            "p_geom": float(p_geom)
        })

    print("===============================================================")
    print(" REAL VS ISOTROPIC GEOMETRY FOR ECLIPTIC MASKS (TEST 85L) ")
    print("===============================================================")
    for r in results:
        print(f"|β|>={r['cut']:4.1f}°  N={r['N']:4d}  "
              f"Z_real={r['Z_real']:.6f}  "
              f"iso_mean={r['iso_mean']:.6f}  "
              f"iso_std={r['iso_std']:.6f}  "
              f"p_geom={r['p_geom']:.6f}")
    print("===============================================================")
    print("interpretation:")
    print("  if p_geom is large, real-sky phase-memory under |β| cut")
    print("  matches isotropic geometry tied to the ecliptic frame.")
    print("  if p_geom is tiny, solar-system geometry cannot explain")
    print("  the remnant-time phase-memory signal.")
    print("===============================================================")
    print("test 85L complete.")
    print("===============================================================")


if __name__ == "__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_phase_memory_ecliptic_real_vs_iso_test85L.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
