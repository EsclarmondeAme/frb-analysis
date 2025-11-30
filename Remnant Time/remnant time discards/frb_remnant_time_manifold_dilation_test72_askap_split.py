#!/usr/bin/env python3
import numpy as np
import sys, os
from astropy.io import fits
from tqdm import tqdm
import csv

ASKAP_FITS_DIR = "data/positions"

def load_unified_catalog(path):
    RA, Dec, rem = [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            RA.append(float(row["ra"]))
            Dec.append(float(row["dec"]))
            s = float(row["theta_unified"])
            rem.append(1 if s < 90 else -1)
    return np.array(RA), np.array(Dec), np.array(rem)

def sph_to_xyz(ra, dec):
    ra_rad  = np.radians(ra)
    dec_rad = np.radians(dec)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.column_stack((x, y, z))

def load_askap_fits_positions():
    files = [f for f in os.listdir(ASKAP_FITS_DIR) if f.lower().endswith(".fits")]
    RA, Dec = [], []
    for fn in files:
        hd = fits.open(os.path.join(ASKAP_FITS_DIR, fn))
        hdr = hd[0].header
        RA.append(float(hdr["CRVAL1"]))
        Dec.append(float(hdr["CRVAL2"]))
        hd.close()
    return np.array(RA), np.array(Dec)

def angular_sep(ra1, dec1, ra2_array, dec2_array):
    ra1  = np.radians(ra1)
    dec1 = np.radians(dec1)
    ra2  = np.radians(ra2_array)
    dec2 = np.radians(dec2_array)
    return np.degrees(
        np.arccos(
            np.sin(dec1)*np.sin(dec2) +
            np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2)
        )
    )

def match_fits_to_FRBs(fRA, fDec, RA, Dec, tol=0.2):
    askap_mask = np.zeros(len(RA), dtype=bool)
    for FRA, FDEC in zip(fRA, fDec):
        d = angular_sep(FRA, FDEC, RA, Dec)
        j = np.argmin(d)
        if d[j] <= tol:
            askap_mask[j] = True
    return askap_mask

def split_hemispheres(X, rem_signs):
    pos = np.where(rem_signs == 1)[0]
    neg = np.where(rem_signs == -1)[0]
    return pos, neg

def balanced_subset(pos, neg):
    Nmin = min(len(pos), len(neg))
    return pos[:Nmin], neg[:Nmin], Nmin

def simple_dilation_score(X, pos, neg):
    if len(pos)==0 or len(neg)==0:
        return 0.0
    xp = X[pos]
    xn = X[neg]
    cp = xp.mean(axis=0)
    cn = xn.mean(axis=0)
    dp = np.mean(np.linalg.norm(xp - cp, axis=1))
    dn = np.mean(np.linalg.norm(xn - cn, axis=1))
    return dp - dn

def random_axis():
    u = np.random.uniform(-1,1)
    phi = np.random.uniform(0,2*np.pi)
    s = np.sqrt(1-u*u)
    return np.array([s*np.cos(phi), s*np.sin(phi), u])

def rotate_signs(X, axis):
    dots = X @ axis
    return np.where(dots>=0, 1, -1)

def main(csv_path):
    RA, Dec, rem = load_unified_catalog(csv_path)
    X = sph_to_xyz(RA, Dec)

    fRA, fDec = load_askap_fits_positions()
    askap_mask = match_fits_to_FRBs(fRA, fDec, RA, Dec, tol=0.2)

    idxA = np.where(askap_mask)[0]
    idxN = np.where(~askap_mask)[0]

    XA, remA = X[idxA], rem[idxA]
    XN, remN = X[idxN], rem[idxN]

    posA, negA = split_hemispheres(XA, remA)
    posN, negN = split_hemispheres(XN, remN)

    posA_b, negA_b, _ = balanced_subset(posA, negA)
    posN_b, negN_b, _ = balanced_subset(posN, negN)

    SA_real = simple_dilation_score(XA, posA_b, negA_b)
    SN_real = simple_dilation_score(XN, posN_b, negN_b)

    nmc = 2000
    SA_null, SN_null = [], []

    for _ in tqdm(range(nmc)):
        ax = random_axis()

        rA = rotate_signs(XA, ax)
        pA, nA = split_hemispheres(XA, rA)
        pA_b, nA_b, _ = balanced_subset(pA, nA)
        SA_null.append(simple_dilation_score(XA, pA_b, nA_b))

        rN = rotate_signs(XN, ax)
        pN, nN = split_hemispheres(XN, rN)
        pN_b, nN_b, _ = balanced_subset(pN, nN)
        SN_null.append(simple_dilation_score(XN, pN_b, nN_b))

    SA_null = np.array(SA_null)
    SN_null = np.array(SN_null)

    pA = np.mean(np.abs(SA_null) >= np.abs(SA_real))
    pN = np.mean(np.abs(SN_null) >= np.abs(SN_real))

    print("===============================================================")
    print(" FRB REMNANT-TIME MANIFOLD DILATION TEST (72C — ASKAP split)")
    print("===============================================================")
    print(f"ASKAP count     = {len(idxA)}")
    print(f"non-ASKAP count = {len(idxN)}")
    print("---------------------------------------------------------------")
    print(f"ASKAP:     S_real={SA_real:.6f}, null_mean={SA_null.mean():.3f}, null_std={SA_null.std():.3f}, p={pA:.6f}")
    print(f"nonASKAP:  S_real={SN_real:.6f}, null_mean={SN_null.mean():.3f}, null_std={SN_null.std():.3f}, p={pN:.6f}")
    print("---------------------------------------------------------------")
    print("interpretation:")
    print(" low p  -> dilation asymmetry in the subset")
    print(" high p -> subset consistent with isotropy")
    print("===============================================================")
    print(" test 72C complete.")
    print("===============================================================")

if __name__ == "__main__":
    main(sys.argv[1])
