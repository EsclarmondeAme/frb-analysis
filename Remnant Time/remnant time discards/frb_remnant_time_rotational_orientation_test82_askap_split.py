#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from math import radians


from frb_remnant_time_rotational_orientation_test82 import (
    load_catalog,
    radec_to_galactic_xyz,
    gal_lb_xyz,
    remnant_sign,
    local_orientations,
)

np.random.seed(0)

ASKAP_DIR = "data/positions"
MATCH_TOL = 2.0  # degrees

def load_askap_pointings():
    ras, decs = [], []
    if not os.path.exists(ASKAP_DIR):
        return np.array(ras), np.array(decs)
    for f in os.listdir(ASKAP_DIR):
        if not f.lower().endswith(".fits"):
            continue
        try:
            hdr = fits.getheader(os.path.join(ASKAP_DIR, f))
            ra  = hdr.get("CRVAL1", None)
            dec = hdr.get("CRVAL2", None)
            if ra is not None and dec is not None:
                ras.append(float(ra))
                decs.append(float(dec))
        except Exception:
            pass
    return np.array(ras), np.array(decs)

def angsep(ra1, dec1, ra2, dec2):
    ra1  = radians(ra1)
    dec1 = radians(dec1)
    ra2  = np.radians(ra2)
    dec2 = np.radians(dec2)
    v = (np.sin(dec1)*np.sin(dec2) +
         np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2))
    return np.degrees(np.arccos(np.clip(v, -1.0, 1.0)))

def match_askap(RA, Dec, RAa, Deca, tol):
    out = np.zeros(len(RA), dtype=bool)
    if len(RAa) == 0:
        return out
    for i in range(len(RA)):
        d = angsep(RA[i], Dec[i], RAa, Deca)
        if np.min(d) <= tol:
            out[i] = True
    return out

def run_subset(X, axis, label):
    if len(X) < 40:
        print(f"[info] {label}: too few objects, skipping.")
        return np.nan, np.nan, np.nan, 1.0

    print(f"[info] {label}: computing remnant signs...")
    s = remnant_sign(X, axis)

    print(f"[info] {label}: computing local orientations...")
    psi, valid = local_orientations(X, k=20, anisotropy_thresh=0.1)

    z = np.exp(2j * psi)
    z[~valid] = 0.0 + 0.0j

    z_pos = z[(s > 0) & valid]
    z_neg = z[(s < 0) & valid]

    if len(z_pos) < 10 or len(z_neg) < 10:
        print(f"[info] {label}: not enough valid orientations in one hemisphere.")
        return np.nan, np.nan, np.nan, 1.0

    S_pos = np.mean(z_pos)
    S_neg = np.mean(z_neg)
    A_real = abs(S_pos - S_neg)

    NMC = 2000
    null = []

    print(f"[info] {label}: building null ({NMC} shuffles)...")
    for _ in tqdm(range(NMC)):
        sh = np.copy(s)
        np.random.shuffle(sh)
        z_p = z[(sh > 0) & valid]
        z_n = z[(sh < 0) & valid]
        if len(z_p) < 10 or len(z_n) < 10:
            null.append(0.0)
            continue
        Sp = np.mean(z_p)
        Sn = np.mean(z_n)
        null.append(abs(Sp - Sn))

    null = np.array(null)
    mu = float(np.mean(null))
    sd = float(np.std(null))
    p = (1.0 + np.sum(null >= A_real)) / (NMC + 1.0)

    return A_real, mu, sd, p

def main(path):

    print("===============================================")
    print(" Test 82C — rotational orientation, askap split")
    print("===============================================")

    RA, Dec = load_catalog(path)
    X = radec_to_galactic_xyz(RA, Dec)

    RAa, Deca = load_askap_pointings()
    print(f"[info] askap pointings loaded: {len(RAa)}")

    isA = match_askap(RA, Dec, RAa, Deca, MATCH_TOL)

    Xa = X[isA]
    Xn = X[~isA]

    print(f"[info] askap count     = {len(Xa)}")
    print(f"[info] non-askap count = {len(Xn)}")

    axis = gal_lb_xyz(159.8, -0.5)

    Aa, mua, sda, pa = run_subset(Xa, axis, "ASKAP")
    An, mun, sdn, pn = run_subset(Xn, axis, "non-ASKAP")

    print("===============================================")
    print(" remnant-time rotational orientation test 82C")
    print(" askap split")
    print("===============================================")
    print(f"askap:     A={Aa}, null_mean={mua}, null_std={sda}, p={pa}")
    print(f"non-askap: A={An}, null_mean={mun}, null_std={sdn}, p={pn}")
    print("===============================================")

if __name__ == "__main__":
    main(sys.argv[1])
