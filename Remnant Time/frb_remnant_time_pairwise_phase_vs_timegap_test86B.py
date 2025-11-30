#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB REMNANT-TIME PAIRWISE PHASE VS TIME SEPARATION (TEST 86B)

goal:
    test whether harmonic phase-coherence between FRBs depends
    on their separation in *observation time* (MJD).

    for each pair (i,j):
        - compute phase-alignment score G_ij from Y_lm phases
          (sign-free, same style as 85R/85P).
        - compute Δt_ij = |MJD_i - MJD_j|.

    then:
        - compute the correlation between Δt_ij and G_ij.
        - build a null distribution by shuffling MJD values
          among FRBs (geometry + phases fixed), recomputing
          Δt_ij and the correlation (2000 realisations).

interpretation:
    - low p and strong negative correlation:
          phase coherence decays with linear time separation.
    - low p and near-zero correlation:
          time gaps behave anomalously compared to null model.
    - high p and correlation ~ 0:
          phase-coherence is independent of observation time,
          consistent with remnant structure being separate from
          linear observation time.
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


# ------------------------------------------------------------
# harmonic phases
# ------------------------------------------------------------

def compute_phases(l_arr_deg, b_arr_deg, lmax=8):
    """
    compute phase of complex Y_lm for 1 <= l <= lmax and all m.
    returns:
        phases: (N_frb, n_modes) array, phases in [-π,π].
    """
    phi = np.radians(l_arr_deg)                # azimuth
    theta = np.radians(90.0 - b_arr_deg)       # colatitude
    phases = []
    for ell in range(1, lmax+1):
        for m in range(-ell, ell+1):
            Y = sph_harm(m, ell, phi, theta)
            phases.append(np.angle(Y))
    return np.vstack(phases).T                 # (N, n_modes)


def pairwise_phase_alignment(ph):
    """
    compute pairwise phase-alignment scores G_ij for all i<j.

    given phase matrix ph (N x M), define for each pair:

        G_ij = <cos(φ_i,m - φ_j,m)>_m

    returns:
        G     : array of shape (N_pairs,)
        i_idx : array of i indices
        j_idx : array of j indices
    """
    N, M = ph.shape
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = i_idx.size

    # center phases per mode (remove trivial offset)
    ph0 = ph - np.mean(ph, axis=0, keepdims=True)

    G = np.empty(n_pairs, dtype=float)

    # loop over pairs but use vectorised operations over modes
    for k in range(n_pairs):
        i = i_idx[k]
        j = j_idx[k]
        dphi = ph0[i] - ph0[j]      # (M,)
        G[k] = np.mean(np.cos(dphi))
    return G, i_idx, j_idx


# ------------------------------------------------------------
# correlation helper
# ------------------------------------------------------------

def corr_coef(x, y):
    """
    compute Pearson correlation coefficient between x and y.
    both 1D arrays of same length.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return np.nan
    mx = np.mean(x)
    my = np.mean(y)
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0.0 or sy == 0.0:
        return np.nan
    return float(np.mean((x-mx)*(y-my)) / (sx*sy))


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(catfile):
    print("===================================================")
    print("  PHASE VS TIME-SEPARATION TEST (TEST 86B)         ")
    print("===================================================")
    print(f"[info] loading FRB catalog: {catfile}")

    df = pd.read_csv(catfile)

    # auto-detect RA/Dec
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

    # detect time column (MJD)
    col_time = None
    for c in df.columns:
        cl = c.lower()
        if "mjd" in cl or "time" in cl or "utc" in cl:
            col_time = c
            break

    if col_time is None:
        raise RuntimeError("could not detect observation-time column (e.g. 'mjd').")

    ra = df[col_ra].values.astype(float)
    dec = df[col_dec].values.astype(float)
    # convert arbitrary datetime strings to numeric epoch (seconds)
    print("[info] parsing observation time as datetime...")
    t_dt = pd.to_datetime(df[col_time], errors='coerce')

    if t_dt.isna().any():
        raise RuntimeError("time column contains unparsable datetime strings.")

    # convert to seconds since UNIX epoch
    t_obs = t_dt.astype('int64').values.astype(float) * 1e-9
    print("[info] time converted to numeric seconds.")


    N = len(ra)
    print(f"[info] N_FRB = {N}")
    print(f"[info] RA column:    {col_ra}")
    print(f"[info] Dec column:   {col_dec}")
    print(f"[info] time column:  {col_time}")

    print("[info] converting to galactic (l,b)...")
    lgal, bgal = radec_to_galactic(ra, dec)

    print("[info] computing harmonic phases (l_max=8)...")
    PH = compute_phases(lgal, bgal, lmax=8)

    print("[info] building pairwise phase-alignment scores G_ij...")
    G, i_idx, j_idx = pairwise_phase_alignment(PH)
    n_pairs = G.size
    print(f"[info] total number of pairs: {n_pairs}")

    print("[info] computing real time separations Δt_ij...")
    dt_real = np.abs(t_obs[i_idx] - t_obs[j_idx])

    print("[info] computing real Pearson correlation ρ(Δt, G)...")
    rho_real = corr_coef(dt_real, G)
    print(f"[info] rho_real = {rho_real:.6e}")

    # null: shuffle times among FRBs
    n_null = 2000
    print(f"[info] building null distribution with {n_null} shuffles...")
    rng = np.random.RandomState(12345)
    rho_null = np.empty(n_null, dtype=float)

    for k in range(n_null):
        t_perm = np.array(t_obs)
        rng.shuffle(t_perm)
        dt_null = np.abs(t_perm[i_idx] - t_perm[j_idx])
        rho_null[k] = corr_coef(dt_null, G)

    # remove any nan
    rho_null = rho_null[np.isfinite(rho_null)]
    if rho_null.size == 0:
        print("[warn] null distribution collapsed (all NaNs).")
        null_mean = np.nan
        null_std = np.nan
        p = np.nan
    else:
        null_mean = float(np.mean(rho_null))
        null_std = float(np.std(rho_null))
        # two-sided p: |rho_null| >= |rho_real|
        p = (1 + np.sum(np.abs(rho_null) >= abs(rho_real))) / (len(rho_null) + 1)

    print("---------------------------------------------------")
    print("TEST 86B — RESULTS")
    print("---------------------------------------------------")
    print(f"rho_real   = {rho_real:.6e}")
    print(f"null_mean  = {null_mean:.6e}")
    print(f"null_std   = {null_std:.6e}")
    print(f"p_value    = {p:.6e}")
    print("---------------------------------------------------")
    print("interpretation:")
    print("  - strong negative rho_real with low p:")
    print("        phase coherence decreases with time gap;")
    print("        linear time acts like a decohering axis.")
    print("  - rho_real ~ 0 with high p:")
    print("        phase coherence is insensitive to time")
    print("        separation, consistent with remnant-like")
    print("        structure being decoupled from observation")
    print("        time (no preferred ordering).")
    print("  - strong positive rho_real with low p:")
    print("        counter-intuitive case: large time gaps")
    print("        show more coherence than small ones.")
    print("===================================================")
    print("test 86B complete.")
    print("===================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_pairwise_phase_vs_timegap_test86B.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
