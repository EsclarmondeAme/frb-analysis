#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB Remnant-Time Harmonic-Energy Test (77C — ASKAP split)
Uses ASKAP FITS pointing centers to identify ASKAP vs non-ASKAP FRBs.
Computes harmonic-energy contrast separately in each subset.
"""

import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.io import fits

# ---------------------------------------------------------
# configuration
# ---------------------------------------------------------

ASKAP_FITS_DIR = r"data/positions"   # your working directory (already correct)
MATCH_TOL = 0.2                       # degrees, based on nearest-distance diagnostic

# ---------------------------------------------------------
# utility
# ---------------------------------------------------------

def angsep(ra1, dec1, ra2, dec2):
    """angular separation in degrees, vectorized for ra2/dec2 arrays"""
    ra1 = np.radians(ra1)
    dec1 = np.radians(dec1)
    ra2 = np.radians(ra2)
    dec2 = np.radians(dec2)
    return np.degrees(
        np.arccos(
            np.sin(dec1)*np.sin(dec2)
            + np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2)
        )
    )

def ang2xyz(ra, dec):
    ra = np.radians(ra); dec = np.radians(dec)
    return np.column_stack([
        np.cos(dec)*np.cos(ra),
        np.cos(dec)*np.sin(ra),
        np.sin(dec)
    ])

def harmonic_energy(theta):
    return np.sum(np.cos(np.radians(theta))**2)

def project_sign(X, axis_xyz):
    return X @ axis_xyz

def compute_real(theta, Rsign):
    thP = theta[Rsign > 0]
    thN = theta[Rsign < 0]
    return harmonic_energy(thP) - harmonic_energy(thN)

def random_axis():
    phi = np.random.uniform(0, 2*np.pi)
    z = np.random.uniform(-1,1)
    r = np.sqrt(1 - z*z)
    return np.array([r*np.cos(phi), r*np.sin(phi), z])

# ---------------------------------------------------------
# ASKAP FITS loader
# ---------------------------------------------------------

def load_askap_fits(dirpath):
    ras = []
    decs = []
    for f in os.listdir(dirpath):
        if f.lower().endswith(".fits"):
            path = os.path.join(dirpath, f)
            with fits.open(path) as hdul:
                hdr = hdul[0].header
                ras.append(hdr.get("CRVAL1"))
                decs.append(hdr.get("CRVAL2"))
    return np.array(ras), np.array(decs)

# ---------------------------------------------------------
# main
# ---------------------------------------------------------

def main(path):
    print("===============================================")
    print("FRB REMNANT-TIME HARMONIC-ENERGY TEST (77C — ASKAP split)")
    print("===============================================")

    # load unified catalog
    df = pd.read_csv(path)
    RA  = df["ra"].values
    Dec = df["dec"].values
    Th  = df["theta_unified"].values
    Ph  = df["phi_unified"].values

    # FRB xyz
    X = ang2xyz(RA, Dec)

    # unified axis (average of FRB unit vectors in unified coords)
    a = np.radians(Ph)
    b = np.radians(90 - Th)
    A = np.column_stack([
        np.cos(b)*np.cos(a),
        np.cos(b)*np.sin(a),
        np.sin(b)
    ])
    axis_unified = A.mean(axis=0)
    axis_unified /= np.linalg.norm(axis_unified)

    # load ASKAP pointing centers
    ask_ra, ask_dec = load_askap_fits(ASKAP_FITS_DIR)
    print(f"[info] loaded {len(ask_ra)} ASKAP pointing centers")

    # classify FRBs as ASKAP or non-ASKAP
    is_askap = np.zeros(len(RA), dtype=bool)
    for i in range(len(RA)):
        d = angsep(RA[i], Dec[i], ask_ra, ask_dec)
        if np.min(d) < MATCH_TOL:
            is_askap[i] = True

    RaA = RA[is_askap];    DecA = Dec[is_askap]
    RaN = RA[~is_askap];   DecN = Dec[~is_askap]
    ThA = Th[is_askap];    ThN = Th[~is_askap]
    X_A = X[is_askap];     X_N = X[~is_askap]

    print(f"ASKAP count     = {len(RaA)}")
    print(f"non-ASKAP count = {len(RaN)}")
    print("------------------------------------------------")

    # real metric
    R_A = project_sign(X_A, axis_unified) if len(X_A)>0 else np.array([])
    R_N = project_sign(X_N, axis_unified)

    E_A = compute_real(ThA, R_A) if len(R_A)>0 else np.nan
    E_N = compute_real(ThN, R_N)

    # MC null
    n_mc = 2000
    null_A = []
    null_N = []

    for _ in tqdm(range(n_mc), ncols=80):
        v = random_axis()
        if len(R_A)>0:
            RrA = project_sign(X_A, v)
            null_A.append(compute_real(ThA, RrA))
        RrN = project_sign(X_N, v)
        null_N.append(compute_real(ThN, RrN))

    null_A = np.array(null_A) if len(R_A)>0 else np.array([np.nan])
    null_N = np.array(null_N)

    # compute p-values
    def pval(real, arr):
        if np.isnan(real) or np.all(np.isnan(arr)):
            return 1.0
        arr = arr[~np.isnan(arr)]
        if len(arr)==0:
            return 1.0
        mu = np.mean(arr)
        return np.mean(np.abs(arr - mu) >= np.abs(real - mu))

    pA = pval(E_A, null_A)
    pN = pval(E_N, null_N)

    print("ASKAP:    E_real={}, null_mean={}, null_std={}, p={}".format(
        E_A,
        np.nanmean(null_A),
        np.nanstd(null_A),
        pA,
    ))
    print("nonASKAP: E_real={}, null_mean={}, null_std={}, p={}".format(
        E_N,
        null_N.mean(),
        null_N.std(),
        pN,
    ))

    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> harmonic-energy asymmetry in subset")
    print("  high p -> subset consistent with isotropy")
    print("===============================================")
    print("test 77C complete.")
    print("===============================================")


if __name__ == "__main__":
    main(sys.argv[1])
