#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB PHASE VS TIME-SEPARATION — BINNED ROBUST TEST (TEST 86C)

extends 86B by:
    - computing phase alignment in Δt bins
    - building per-bin nulls:
         * time-shuffle null
         * isotropic-geometry null
    - ensures the Δt-profile is not an artifact of:
         * geometry
         * uneven sampling in time
         * pair-count weighting
         * local shell structure
    - no remnant-time sign used anywhere

scientific goal:
    detect whether harmonic phase-coherence G(Δt)
    systematically varies with observational time separation,
    which is a literal probe of compressed / projected time.

"""

import sys
import numpy as np
import pandas as pd
import math
from scipy.special import sph_harm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------
# coords
# ------------------------------------------------------------

def radec_to_galactic(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    ra_gp  = math.radians(192.85948)
    dec_gp = math.radians(27.12825)
    l_omega = math.radians(32.93192)

    sinb = (np.sin(dec)*np.sin(dec_gp) +
            np.cos(dec)*np.cos(dec_gp)*np.cos(ra-ra_gp))
    b = np.arcsin(np.clip(sinb,-1,1))

    y = np.sin(ra-ra_gp)*np.cos(dec)
    x = (np.cos(dec)*np.sin(dec_gp) -
         np.sin(dec)*np.cos(dec_gp)*np.cos(ra-ra_gp))
    l = np.arctan2(y,x) + l_omega

    return (np.degrees(l)%360, np.degrees(b))

# ------------------------------------------------------------
# phases
# ------------------------------------------------------------

def compute_phases(l_deg, b_deg, lmax=8):
    phi = np.radians(l_deg)
    theta = np.radians(90.0 - b_deg)
    phases=[]
    for ell in range(1,lmax+1):
        for m in range(-ell,ell+1):
            Y = sph_harm(m,ell,phi,theta)
            phases.append(np.angle(Y))
    return np.vstack(phases).T  # (N, M)

def pairwise_G(ph, i_idx, j_idx):
    """compute G_ij = average cos(dphi_m) for each stored pair index."""
    dphi = ph[i_idx] - ph[j_idx]     # (n_pairs, M)
    return np.mean(np.cos(dphi), axis=1)

# ------------------------------------------------------------
# isotropic sky for geometry null
# ------------------------------------------------------------

def isotropic_sky(N):
    u = np.random.uniform(-1,1,size=N)
    phi = np.random.uniform(0,2*np.pi,size=N)
    b = np.degrees(np.arcsin(u))
    l = np.degrees(phi)
    return l, b

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(catfile):
    print("="*55)
    print("  PHASE VS TIME-SEPARATION — BINNED ROBUST TEST 86C")
    print("="*55)

    df = pd.read_csv(catfile)

    # detect RA/Dec/time columns
    cols = [c.lower() for c in df.columns]
    if "ra" in cols:
        col_ra = df.columns[cols.index("ra")]
    elif "raj2000" in cols:
        col_ra = df.columns[cols.index("raj2000")]
    else:
        raise RuntimeError("no RA column found")

    if "dec" in cols:
        col_dec = df.columns[cols.index("dec")]
    elif "dej2000" in cols:
        col_dec = df.columns[cols.index("dej2000")]
    else:
        raise RuntimeError("no Dec column found")

    col_time=None
    for c in df.columns:
        cl=c.lower()
        if "mjd" in cl or "time" in cl or "utc" in cl or "date" in cl:
            col_time=c
            break
    if col_time is None:
        raise RuntimeError("no time column found (mjd/utc/date)")

    print(f"[info] RA: {col_ra}")
    print(f"[info] Dec: {col_dec}")
    print(f"[info] time: {col_time}")

    # convert time to numeric seconds
    print("[info] parsing time...")
    t_dt=pd.to_datetime(df[col_time],errors="coerce")
    if t_dt.isna().any():
        raise RuntimeError("bad datetime entries in time column")
    t_obs=t_dt.astype("int64").values.astype(float)*1e-9  # seconds

    ra=df[col_ra].values.astype(float)
    dec=df[col_dec].values.astype(float)
    N=len(ra)
    print(f"[info] N_FRB = {N}")

    print("[info] converting to galactic...")
    lgal,bgal=radec_to_galactic(ra,dec)

    print("[info] computing phases...")
    PH=compute_phases(lgal,bgal,lmax=8)
    M=PH.shape[1]

    print("[info] building global pair list...")
    i_idx,j_idx=np.triu_indices(N,k=1)
    n_pairs=i_idx.size
    print(f"[info] total pairs: {n_pairs}")

    print("[info] computing pairwise Δt...")
    dt_real=np.abs(t_obs[i_idx]-t_obs[j_idx])

    # define Δt bins (auto from distribution)
    tmin=np.percentile(dt_real,0)
    t20=np.percentile(dt_real,20)
    t40=np.percentile(dt_real,40)
    t60=np.percentile(dt_real,60)
    t80=np.percentile(dt_real,80)
    tmax=np.percentile(dt_real,100)

    bins=[(tmin,t20),(t20,t40),(t40,t60),(t60,t80),(t80,tmax)]
    print("[info] time-sep bins:")
    for (a,b) in bins:
        print(f"   [{a:.2e}, {b:.2e})")

    # compute G_ij real only once
    print("[info] computing real G_ij...")
    G_all=pairwise_G(PH,i_idx,j_idx)

    # storage
    results=[]
    n_null=2000
    print(f"[info] n_null per bin = {n_null}")

    # loop over bins
    for b_idx,(a,b) in enumerate(bins):
        print("-"*55)
        print(f"[info] bin {b_idx}: Δt∈[{a:.2e}, {b:.2e})")

        mask=(dt_real>=a)&(dt_real<b)
        if np.sum(mask)<100:
            print("[warn] too few pairs, skipping")
            results.append((a,b,np.nan,np.nan,np.nan,np.nan,np.nan))
            continue

        G_bin=G_all[mask]
        real_mean=float(np.mean(G_bin))

        # build nulls
        null_time=[]
        null_geom=[]

        rng=np.random.RandomState(1234)

        # fixed pair structure for nulls
        for k in range(n_null):
            # time-shuffle null
            t_perm=np.array(t_obs)
            rng.shuffle(t_perm)
            dt_null=np.abs(t_perm[i_idx]-t_perm[j_idx])
            mask_null=(dt_null>=a)&(dt_null<b)
            if np.sum(mask_null)>0:
                null_time.append(float(np.mean(G_all[mask_null])))

            # isotropic geometry null
            Liso,Biso=isotropic_sky(N)
            PHiso=compute_phases(Liso,Biso,lmax=8)
            Giso=pairwise_G(PHiso,i_idx,j_idx)
            dt_fake=dt_real  # keep time separations real for geometry null
            mask_fake=(dt_fake>=a)&(dt_fake<b)
            if np.sum(mask_fake)>0:
                null_geom.append(float(np.mean(Giso[mask_fake])))

        null_time=np.array(null_time)
        null_geom=np.array(null_geom)

        mu_t=np.mean(null_time); sd_t=np.std(null_time)
        mu_g=np.mean(null_geom); sd_g=np.std(null_geom)

        # p-values (two-sided)
        p_t=(1+np.sum(np.abs(null_time-mu_t)>=abs(real_mean-mu_t)))/(len(null_time)+1)
        p_g=(1+np.sum(np.abs(null_geom-mu_g)>=abs(real_mean-mu_g)))/(len(null_geom)+1)

        print(f"[real mean G] = {real_mean:.6e}")
        print(f"[time-null]   mean={mu_t:.6e}  std={sd_t:.6e}  p={p_t:.6e}")
        print(f"[geom-null]   mean={mu_g:.6e}  std={sd_g:.6e}  p={p_g:.6e}")

        results.append((a,b,real_mean,mu_t,sd_t,p_t,p_g))

    print("="*55)
    print("TEST 86C COMPLETE — BINNED Δt PROFILE")
    print("="*55)

    print("bin_start   bin_end    realG    mu_time  sd_time   p_time     mu_geom   sd_geom   p_geom")
    for r in results:
        print(f"{r[0]:.3e}  {r[1]:.3e}  {r[2]:.6e}  {r[3]:.6e}  {r[4]:.6e}  {r[5]:.6e}  {r[6]:.6e}")

if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python frb_remnant_time_phase_vs_timegap_binned_test86C.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
