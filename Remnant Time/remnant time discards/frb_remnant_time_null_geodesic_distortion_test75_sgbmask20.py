#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB REMNANT-TIME NULL GEODESIC DISTORTION TEST (75B — |SGB| >= 20° SUPERGALACTIC MASK)

Same distortion proxy as 75A:
    - kNN angular distance
    - compare R>0 and R<0 hemispheres
    - build null by shuffling signs
"""

import sys
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------------
# supergalactic latitude
# ------------------------------------------------------------

def supergalactic_latitude(RA, Dec):
    # de Vaucouleurs-like approximation
    ra  = np.radians(RA)
    dec = np.radians(Dec)

    ra_p  = np.radians(283.763)   # SG north pole RA
    dec_p = np.radians(15.705)    # SG north pole Dec

    sinSGB = (np.sin(dec)*np.sin(dec_p) +
              np.cos(dec)*np.cos(dec_p)*np.cos(ra - ra_p))
    return np.degrees(np.arcsin(sinSGB))


def sgbmask(RA, Dec, sgbmin=20.0):
    sgb = supergalactic_latitude(RA, Dec)
    return np.abs(sgb) >= sgbmin

# ------------------------------------------------------------
# angular distance matrix
# ------------------------------------------------------------

def angsep_matrix(RA, Dec):
    ra  = np.radians(RA)
    dec = np.radians(Dec)

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    xyz = np.vstack([x, y, z]).T

    dot = np.clip(xyz @ xyz.T, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def distortion_proxy(RA, Dec, k=5):
    N = len(RA)
    if N <= k:
        return np.zeros(N)

    D = angsep_matrix(RA, Dec)
    np.fill_diagonal(D, 1e9)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    knn = np.take_along_axis(D, idx, axis=1)
    return knn.mean(axis=1)

# ------------------------------------------------------------
# load unified
# ------------------------------------------------------------

def load_catalog(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    RA   = data["ra"]
    Dec  = data["dec"]
    DM   = data["dm"]
    med_dm = np.median(DM)
    signs  = np.where(DM < med_dm, +1, -1)
    return RA, Dec, signs

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(path):
    print("================================================")
    print("FRB REMNANT-TIME NULL GEODESIC DISTORTION TEST (75B)")
    print("Supergalactic mask: |SGB| >= 20°")
    print("================================================")

    RA, Dec, signs = load_catalog(path)
    N0 = len(RA)

    mask = sgbmask(RA, Dec, sgbmin=20.0)
    RA   = RA[mask]
    Dec  = Dec[mask]
    signs = signs[mask]

    N = len(RA)
    print(f"[info] original N = {N0}, after SGB mask N = {N}")

    distort = distortion_proxy(RA, Dec, k=5)

    pos = signs > 0
    neg = signs < 0
    if pos.sum() == 0 or neg.sum() == 0:
        print("[warn] hemisphere empty after mask; cannot compute.")
        return

    mean_pos = distort[pos].mean()
    mean_neg = distort[neg].mean()
    G_real   = mean_pos - mean_neg

    n_mc = 2000
    G_null = np.zeros(n_mc)

    for i in tqdm(range(n_mc), desc="MC"):
        shuf = np.random.permutation(signs)
        pos_s = shuf > 0
        neg_s = shuf < 0
        if pos_s.sum() == 0 or neg_s.sum() == 0:
            G_null[i] = 0.0
        else:
            G_null[i] = distort[pos_s].mean() - distort[neg_s].mean()

    mu = G_null.mean()
    sd = G_null.std()
    p  = np.mean(np.abs(G_null) >= np.abs(G_real))

    print("------------------------------------------------")
    print(f"mean distortion (R>0) = {mean_pos}")
    print(f"mean distortion (R<0) = {mean_neg}")
    print(f"G_real                = {G_real}")
    print("------------------------------------------------")
    print(f"null mean G           = {mu}")
    print(f"null std G            = {sd}")
    print(f"p-value               = {p}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> distortion asymmetry survives SGB mask")
    print("  high p -> symmetric after mask; consistent with isotropy.")
    print("================================================")
    print("test 75B complete.")
    print("================================================")


if __name__ == "__main__":
    main(sys.argv[1])
