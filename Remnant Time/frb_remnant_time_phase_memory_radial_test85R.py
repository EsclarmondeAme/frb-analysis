#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB REMNANT-TIME PHASE-MEMORY — RADIAL SIGNED-PHASE PROFILE (TEST 85R)

goal:
    measure how harmonic phase-coherence varies with angular distance
    from the unified axis, without using remnant-time signs at all.

    for each radial shell in θ (angle to unified axis):
      - compute harmonic phases Y_lm for each FRB (l_max = 8)
      - compute pairwise phase alignment score inside that shell:
            score = < cos(Δφ_ij) > averaged over all modes and pairs
      - build an isotropic-annulus null with the same θ-range and N
      - compare real score to null distribution (2000 realisations)

    this tests whether phase coherence is radially structured around
    the unified axis beyond what isotropic geometry would generate.

notes:
    - no hemisphere sign, no slab map, no remnant labels used.
    - purely geometric + harmonic phase structure.
"""

import sys
import math
import numpy as np
import pandas as pd
from scipy.special import sph_harm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ------------------------------------------------------------
# coordinate transforms
# ------------------------------------------------------------

def radec_to_galactic(ra_deg, dec_deg):
    """RA,Dec (deg, J2000) -> galactic (l,b) in deg."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    ra_gp  = math.radians(192.85948)
    dec_gp = math.radians(27.12825)
    l_omega = math.radians(32.93192)

    sinb = (np.sin(dec)*np.sin(dec_gp) +
            np.cos(dec)*np.cos(dec_gp)*np.cos(ra-ra_gp))
    b = np.arcsin(np.clip(sinb, -1.0, 1.0))

    y = np.sin(ra-ra_gp)*np.cos(dec)
    x = (np.cos(dec)*np.sin(dec_gp) -
         np.sin(dec)*np.cos(dec_gp)*np.cos(ra-ra_gp))
    l = np.arctan2(y, x) + l_omega

    return (np.degrees(l) % 360.0,
            np.degrees(b))


def axis_angle(l_deg, b_deg, l0=159.8, b0=-0.5):
    """angular distance in deg from unified axis (l0,b0)."""
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    l0 = math.radians(l0)
    b0 = math.radians(b0)
    cosd = (np.sin(b)*np.sin(b0) +
            np.cos(b)*np.cos(b0)*np.cos(l-l0))
    cosd = np.clip(cosd, -1.0, 1.0)
    return np.degrees(np.arccos(cosd))


# ------------------------------------------------------------
# harmonic phases
# ------------------------------------------------------------

def compute_real_phases(l_arr_deg, b_arr_deg, lmax=8):
    """
    compute argument (phase) of complex Y_lm for all modes 1..lmax
    and return array of shape (N_frb, N_modes), where each column
    is phase(phi,theta) for one (l,m).
    """
    phi = np.radians(l_arr_deg)                # azimuth
    theta = np.radians(90.0 - b_arr_deg)       # colatitude
    phases = []
    for ell in range(1, lmax+1):
        for m in range(-ell, ell+1):
            Y = sph_harm(m, ell, phi, theta)   # complex
            phases.append(np.angle(Y))         # phase in [-π,π]
    return np.vstack(phases).T                 # (N, nmode)


# ------------------------------------------------------------
# pairwise alignment in a shell
# ------------------------------------------------------------

def alignment_score(ph):
    """
    given phase matrix ph (N x M), compute mean pairwise alignment:
        score = < cos(Δφ_ij) >_{i<j, over all modes}
    where Δφ_ij is taken mode-wise, then averaged over modes.
    """
    N, M = ph.shape
    if N < 4:
        return np.nan

    # center phases per mode to reduce trivial offsets
    ph0 = ph - np.mean(ph, axis=0, keepdims=True)

    # brute-force pairwise; N here is small per shell
    vals = []
    for i in range(N):
        for j in range(i+1, N):
            dphi = ph0[i] - ph0[j]
            vals.append(np.mean(np.cos(dphi)))
    if not vals:
        return np.nan
    return float(np.mean(vals))


# ------------------------------------------------------------
# isotropic annulus generator
# ------------------------------------------------------------

def isotropic_annulus(N, theta_min, theta_max, l0=159.8, b0=-0.5):
    """
    generate N isotropic sky points constrained to lie in an annulus
    theta_min <= θ <= theta_max around axis (l0,b0).
    """
    L = []
    B = []
    # simple rejection sampling; N is modest per shell
    while len(L) < N:
        # random uniform on sphere
        u = np.random.uniform(-1.0, 1.0)
        phi = np.random.uniform(0.0, 2.0*math.pi)
        b = math.degrees(math.asin(u))
        l = math.degrees(phi)
        th = axis_angle(l, b, l0=l0, b0=b0)
        if theta_min <= th < theta_max:
            L.append(l)
            B.append(b)
    return np.array(L, dtype=float), np.array(B, dtype=float)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(catfile):
    print("===================================================")
    print("  RADIAL SIGNED-PHASE PROFILE (TEST 85R)           ")
    print("===================================================")
    print(f"[info] loading FRB catalog: {catfile}")

    df = pd.read_csv(catfile)

    # auto-detect RA/Dec column names
    cols = [c.lower() for c in df.columns]
    if "ra" in cols:
        col_ra = df.columns[cols.index("ra")]
    elif "raj2000" in cols:
        col_ra = df.columns[cols.index("raj2000")]
    else:
        raise RuntimeError("could not detect RA column (ra / raj2000).")

    if "dec" in cols:
        col_dec = df.columns[cols.index("dec")]
    elif "dej2000" in cols:
        col_dec = df.columns[cols.index("dej2000")]
    else:
        raise RuntimeError("could not detect Dec column (dec / dej2000).")

    ra = df[col_ra].values.astype(float)
    dec = df[col_dec].values.astype(float)
    N = len(ra)

    print(f"[info] N_FRB = {N}")
    print(f"[info] RA column:  {col_ra}")
    print(f"[info] Dec column: {col_dec}")

    print("[info] converting to galactic (l,b)...")
    lgal, bgal = radec_to_galactic(ra, dec)

    print("[info] computing axis distances θ...")
    theta = axis_angle(lgal, bgal)  # deg

    print("[info] computing harmonic phases (l_max=8)...")
    PH = compute_real_phases(lgal, bgal, lmax=8)

    # radial shells in θ (deg)
    shells = [(0.0, 20.0),
              (20.0, 40.0),
              (40.0, 60.0),
              (60.0, 80.0),
              (80.0, 180.0)]

    n_null = 2000
    results = []

    print("[info] shells:", shells)
    print("[info] using n_null =", n_null)
    print("---------------------------------------------------")

    for (tmin, tmax) in shells:
        mask = (theta >= tmin) & (theta < tmax)
        idx = np.where(mask)[0]
        Nsh = idx.size

        print(f"[info] shell θ∈[{tmin:.1f},{tmax:.1f}) deg: N={Nsh}")

        if Nsh < 30:
            print("[warn] too few FRBs for stable estimator, skipping shell.")
            results.append((tmin, tmax, Nsh, np.nan, np.nan, np.nan))
            print("---------------------------------------------------")
            continue

        ph_shell = PH[idx, :]
        score_real = alignment_score(ph_shell)
        print(f"    real alignment score = {score_real:.6e}")

        # isotropic annulus null
        null_scores = []
        for k in range(n_null):
            Liso, Biso = isotropic_annulus(Nsh, tmin, tmax)
            PHiso = compute_real_phases(Liso, Biso, lmax=8)
            sc = alignment_score(PHiso)
            null_scores.append(sc)

        null_scores = np.array(null_scores, dtype=float)
        null_scores = null_scores[np.isfinite(null_scores)]
        if null_scores.size == 0:
            print("[warn] null distribution collapsed (all NaN).")
            results.append((tmin, tmax, Nsh, score_real, np.nan, np.nan))
            print("---------------------------------------------------")
            continue

        mean_null = float(np.mean(null_scores))
        # one-sided p: how often null >= real
        p = (1 + np.sum(null_scores >= score_real)) / (len(null_scores) + 1)

        print(f"    null_mean = {mean_null:.6e}")
        print(f"    p-value   = {p:.6e}")
        print("---------------------------------------------------")

        results.append((tmin, tmax, Nsh, score_real, mean_null, p))

    print("===================================================")
    print(" RADIAL SIGNED-PHASE PROFILE RESULTS (TEST 85R)    ")
    print("===================================================")
    print(" shell(θ)        N_shell   score_real      null_mean       p-value")
    for (tmin, tmax, Nsh, sc, nm, p) in results:
        print(f" [{tmin:5.1f},{tmax:5.1f})  {Nsh:7d}   "
              f"{sc: .6e}   {nm: .6e}   {p: .6e}")
    print("===================================================")
    print("interpretation:")
    print("  - a systematic trend of score_real with θ, especially")
    print("    when score_real > null_mean in multiple shells with")
    print("    low p-values, indicates a radial gradient in phase")
    print("    coherence around the unified axis.")
    print("  - consistency with null across all shells implies no")
    print("    detectable radial phase-memory structure beyond")
    print("    isotropic annulus geometry.")
    print("===================================================")
    print("test 85R complete.")
    print("===================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_phase_memory_radial_test85R.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
