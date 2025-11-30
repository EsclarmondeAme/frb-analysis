#!/usr/bin/env python3
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.special import sph_harm
import random

N_MC = 20000   # number of MC isotropic skies for significance

# which multipoles to compute
L_MAX = 3      # compute ℓ = 1,2,3

CAT = "frbs.csv"

# -------------------------------------------------------------
# compute spherical-harmonic coefficients for a set of positions
# -------------------------------------------------------------
def compute_alm(ra_deg, dec_deg, lmax=3):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    theta = np.pi/2 - dec    # colatitude
    phi = ra                 # longitude

    alm = {}

    for l in range(1, lmax+1):
        for m in range(-l, l+1):
            Y = sph_harm(m, l, phi, theta)
            alm[(l, m)] = np.sum(Y) / len(ra_deg)  # normalized average

    return alm


# -------------------------------------------------------------
# compute total multipole power C_l = sum |a_lm|^2 / (2l+1)
# -------------------------------------------------------------
def compute_Cl(alm, l):
    vals = []
    for m in range(-l, l+1):
        vals.append(np.abs(alm[(l, m)])**2)
    return np.sum(vals) / (2*l+1)


# -------------------------------------------------------------
# isotropic Monte Carlo for multipole significance
# -------------------------------------------------------------
def mc_null(n_events, lmax=3):
    ra_rand = np.random.uniform(0, 360, n_events)
    cos_dec = np.random.uniform(-1, 1, n_events)
    dec_rand = np.degrees(np.arccos(cos_dec)) - 90.0

    alm_rand = compute_alm(ra_rand, dec_rand, lmax)
    Cl_rand = {l: compute_Cl(alm_rand, l) for l in range(1, lmax+1)}

    return Cl_rand


# -------------------------------------------------------------
# helper: compute multipoles + MC p-values
# -------------------------------------------------------------
def analyze_subset(label, df):
    print("="*60)
    print(f"MULTIPOLE ANALYSIS — {label}")
    print("="*60)

    ra = df["ra"].values
    dec = df["dec"].values
    n = len(df)

    print(f"n = {n}")

    # compute real coefficients
    alm_real = compute_alm(ra, dec, L_MAX)
    Cl_real = {l: compute_Cl(alm_real, l) for l in range(1, L_MAX+1)}

    # run Monte Carlo
    Cl_null = {l: [] for l in range(1, L_MAX+1)}

    for _ in range(N_MC):
        mc = mc_null(n, L_MAX)
        for l in range(1, L_MAX+1):
            Cl_null[l].append(mc[l])

    # compute p-values
    for l in range(1, L_MAX+1):
        arr = np.array(Cl_null[l])
        real = Cl_real[l]
        p = np.mean(arr >= real)
        print(f"ℓ={l}:  C_l_real = {real:.4e}   <C_l_null> = {np.mean(arr):.4e}   p = {p:.4f}")

    print()


# -------------------------------------------------------------
# main
# -------------------------------------------------------------
def main():
    df = pd.read_csv(CAT)
    df = df.dropna(subset=["z_est"])

    print(f"Loaded FRBs with redshift: {len(df)}")

    # redshift split
    z_sorted = np.sort(df["z_est"].values)
    z_med = np.median(z_sorted)
    df_low = df[df["z_est"] <= z_med]
    df_high = df[df["z_est"] >  z_med]

    # run analyses
    analyze_subset("ALL", df)
    analyze_subset("LOW-Z", df_low)
    analyze_subset("HIGH-Z", df_high)

    print("analysis complete.")
    print("="*60)


if __name__ == "__main__":
    main()
