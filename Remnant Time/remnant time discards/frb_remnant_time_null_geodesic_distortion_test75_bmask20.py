#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB REMNANT-TIME NULL GEODESIC DISTORTION TEST (75A — |b| >= 20° GALACTIC MASK)

This is a robustness variant of Test 75.
It does NOT modify your original engine; instead it uses a simple,
transparent proxy of "null-geodesic distortion":

  - for each FRB, measure the mean angular separation to its k nearest
    neighbours (on the sky).
  - compute the average distortion in the R>0 and R<0 hemispheres
    (remnant-time signs defined by DM).
  - compare the difference: G_real = mean_distortion_pos - mean_distortion_neg
  - build a Monte Carlo null by shuffling the remnant signs.

If the remnant-time effect is real and not a Galactic-plane artefact,
we expect G_real to be atypical under the shuffled null, even after
masking |b| < 20°.
"""

import sys
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------------
# galactic latitude mask
# ------------------------------------------------------------

def galactic_latitude(RA, Dec):
    """
    approximate J2000 RA/Dec -> Galactic latitude b (degrees)
    using standard transformation.
    """
    ra  = np.radians(RA)
    dec = np.radians(Dec)

    ra_gp  = np.radians(192.859508)
    dec_gp = np.radians(27.128336)

    sinb = (np.sin(dec)*np.sin(dec_gp) +
            np.cos(dec)*np.cos(dec_gp)*np.cos(ra - ra_gp))
    return np.degrees(np.arcsin(sinb))


def bmask(RA, Dec, bmin=20.0):
    b = galactic_latitude(RA, Dec)
    return np.abs(b) >= bmin

# ------------------------------------------------------------
# angular distance on the sky (great-circle)
# ------------------------------------------------------------

def angsep_matrix(RA, Dec):
    """
    compute full NxN matrix of angular separations in degrees.
    """
    ra  = np.radians(RA)
    dec = np.radians(Dec)

    # use vectorized dot products
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    xyz = np.vstack([x, y, z]).T

    dot = np.clip(xyz @ xyz.T, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

# ------------------------------------------------------------
# distortion proxy
# ------------------------------------------------------------

def distortion_proxy(RA, Dec, k=5):
    """
    for each FRB, compute mean angular separation to its k nearest neighbours.
    returns an array of per-object distortion values.
    """
    N = len(RA)
    if N <= k:
        # not enough objects to define neighbours; return zeros
        return np.zeros(N)

    D = angsep_matrix(RA, Dec)
    # ignore self-distance (set diagonal large so it's not chosen)
    np.fill_diagonal(D, 1e9)

    # indices of k nearest neighbours for each row
    idx = np.argpartition(D, k, axis=1)[:, :k]
    # gather distances
    knn_dists = np.take_along_axis(D, idx, axis=1)
    return knn_dists.mean(axis=1)

# ------------------------------------------------------------
# load unified catalog
# ------------------------------------------------------------

def load_catalog(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    RA   = data["ra"]
    Dec  = data["dec"]
    DM   = data["dm"]

    # use the same sign convention as other remnant-time tests:
    # +1 for lower-DM half, -1 for higher-DM half
    med_dm = np.median(DM)
    signs  = np.where(DM < med_dm, +1, -1)
    return RA, Dec, signs

# ------------------------------------------------------------
# main test
# ------------------------------------------------------------

def main(path):

    print("================================================")
    print("FRB REMNANT-TIME NULL GEODESIC DISTORTION TEST (75A)")
    print("Galactic mask: |b| >= 20°")
    print("================================================")

    RA, Dec, signs = load_catalog(path)
    N0 = len(RA)

    # apply galactic mask
    mask = bmask(RA, Dec, bmin=20.0)
    RA   = RA[mask]
    Dec  = Dec[mask]
    signs = signs[mask]

    N = len(RA)
    print(f"[info] original N = {N0}, after mask N = {N}")

    # compute distortion proxy for all frbs after mask
    distort = distortion_proxy(RA, Dec, k=5)

    # split by remnant sign
    pos = signs > 0
    neg = signs < 0

    if pos.sum() == 0 or neg.sum() == 0:
        print("[warn] one hemisphere is empty after mask; cannot compute.")
        return

    mean_pos = distort[pos].mean()
    mean_neg = distort[neg].mean()
    G_real   = mean_pos - mean_neg

    # build null by shuffling signs
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
    # two-sided: extreme absolute difference
    p = np.mean(np.abs(G_null) >= np.abs(G_real))

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
    print("  low p  -> null-geodesic clustering differs between remnant-time hemispheres")
    print("           even after removing the Galactic plane.")
    print("  high p -> clustering is symmetric after masking; consistent with isotropy.")
    print("================================================")
    print("test 75A complete.")
    print("================================================")


if __name__ == "__main__":
    main(sys.argv[1])
