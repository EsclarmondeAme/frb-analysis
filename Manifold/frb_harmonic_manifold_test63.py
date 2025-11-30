#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRB HARMONIC MANIFOLD DECOMPOSITION (TEST 63)
Ultra-Extended Spectral Geometry Version

This test extracts the intrinsic harmonic structure of the FRB latent manifold,
following Test 62 (latent manifold detection).

We compute:
    - intrinsic kNN graph
    - unnormalized Laplacian L = D - W
    - normalized symmetric Laplacian L_sym = I - D^{-1/2} W D^{-1/2}
    - eigenvalues/eigenvectors of both Laplacians
    - spectral gaps Δ_k = λ_{k+1} - λ_k
    - harmonic smoothness scores φ_i^T L φ_i
    - spectral energy decay curves
    - combined harmonic-manifold score H_real
    - null ensemble H_null (2000 isotropic skies)

Interpretation:
    - low p-value → strong intrinsic harmonic structure (real manifold modes)
    - high p-value → manifold has no special harmonic structure
"""

import numpy as np
import csv
import logging
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph
import numpy.linalg as LA


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")


# ---------------------------------------------------------------------
# Utility: safe CSV loader with auto-detect column names
# ---------------------------------------------------------------------
def load_frb_catalog(path):
    """Load RA/Dec with auto-detect of multiple possible column names."""
    RAs, Decs = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = {k.lower(): k for k in row.keys()}

            # RA keys
            ra_key = None
            for cand in ["ra", "ra_deg", "raj2000"]:
                if cand in keys:
                    ra_key = keys[cand]
                    break

            # Dec keys
            dec_key = None
            for cand in ["dec", "dec_deg", "decj2000"]:
                if cand in keys:
                    dec_key = keys[cand]
                    break

            if ra_key is None or dec_key is None:
                raise KeyError("Could not auto-detect RA/Dec columns.")

            RAs.append(float(row[ra_key]))
            Decs.append(float(row[dec_key]))

    return np.array(RAs), np.array(Decs)


# ---------------------------------------------------------------------
# Spherical distance
# ---------------------------------------------------------------------
def spherical_distance(ra1, dec1, ra2, dec2):
    ra1, dec1 = np.radians(ra1), np.radians(dec1)
    ra2, dec2 = np.radians(ra2), np.radians(dec2)
    s = (np.sin((dec2 - dec1) / 2)**2 +
         np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2)**2)
    return np.degrees(2 * np.arcsin(np.minimum(1.0, np.sqrt(s))))


def pairwise_spherical(RA, Dec):
    n = len(RA)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = spherical_distance(RA[i], Dec[i], RA, Dec)
    return D


# ---------------------------------------------------------------------
# Graph Laplacians
# ---------------------------------------------------------------------
def build_knn_graph(D, k=12):
    """Build kNN graph from spherical distance matrix D."""
    # convert degrees to a scaled similarity weight
    # W_ij = exp(-D_ij / sigma), sigma ~ median neighbor distance
    n = len(D)
    sorted_d = np.sort(D, axis=1)
    sigma = np.median(sorted_d[:, 1:k+1])
    W = np.exp(-D / max(sigma, 1e-6))
    np.fill_diagonal(W, 0)

    # keep only k nearest neighbors symmetrically
    idx = np.argsort(D, axis=1)[:, 1:k+1]
    mask = np.zeros_like(W)
    for i in range(n):
        mask[i, idx[i]] = 1
    W = W * ((mask + mask.T) > 0)

    return W


def unnormalized_laplacian(W):
    d = np.sum(W, axis=1)
    return np.diag(d) - W


def normalized_laplacian(W):
    d = np.sum(W, axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt


# ---------------------------------------------------------------------
# Harmonic metrics
# ---------------------------------------------------------------------
def spectral_gap(eigs):
    gaps = np.diff(eigs)
    return np.max(gaps), gaps


def harmonic_smoothness(L, eigvecs):
    scores = []
    for i in range(eigvecs.shape[1]):
        φ = eigvecs[:, i]
        s = φ.T @ (L @ φ)
        scores.append(s)
    return np.array(scores)


def spectral_energy(eigs):
    return np.cumsum(eigs)


def harmonic_manifold_score(eigs, gaps, smooth, energy):
    # Combine indicators:
    gap_score = np.max(gaps)
    smooth_score = 1.0 / max(np.mean(smooth[1:6]), 1e-6)
    energy_score = 1.0 / max(energy[5], 1e-6)

    return gap_score + smooth_score + energy_score


# ---------------------------------------------------------------------
# Null skies
# ---------------------------------------------------------------------
def random_sky(n):
    ra = np.random.uniform(0, 360, n)
    u = np.random.uniform(-1, 1, n)
    dec = np.degrees(np.arcsin(u))
    return ra, dec


def compute_H_score(RA, Dec, k=12, n_modes=20):
    D = pairwise_spherical(RA, Dec)
    W = build_knn_graph(D, k=k)
    L = unnormalized_laplacian(W)
    Lsym = normalized_laplacian(W)

    # eigenvalues/eigenvectors
    eigs, vecs = LA.eigh(L)
    eigs = np.maximum(eigs[:n_modes], 0)
    vecs = vecs[:, :n_modes]

    # metrics
    mx_gap, gaps = spectral_gap(eigs)
    smooth = harmonic_smoothness(L, vecs)
    energy = spectral_energy(eigs)

    return harmonic_manifold_score(eigs, gaps, smooth, energy)


def compute_null(n, N_mc=2000):
    H = []
    for _ in tqdm(range(N_mc), desc="MC null"):
        ra, dec = random_sky(n)
        H.append(compute_H_score(ra, dec))
    return np.array(H)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main(path):
    logging.info("loading FRB catalog...")
    RA, Dec = load_frb_catalog(path)
    n = len(RA)
    logging.info(f"N_FRB = {n}")

    logging.info("computing harmonic-manifold score for real data...")
    H_real = compute_H_score(RA, Dec)

    logging.info("building null distribution...")
    H_null = compute_null(n, N_mc=2000)

    null_mean = H_null.mean()
    null_std  = H_null.std()
    p = np.mean(H_null >= H_real)

    print("================================================================")
    print(" FRB HARMONIC MANIFOLD DECOMPOSITION (TEST 63)")
    print("================================================================")
    print(f"H_real       = {H_real:.6f}")
    print(f"null mean    = {null_mean:.6f}")
    print(f"null std     = {null_std:.6f}")
    print(f"p-value      = {p:.6f}")
    print("----------------------------------------------------------------")
    print("interpretation:")
    print(" - low p  → strong intrinsic harmonic manifold structure")
    print(" - high p → manifold harmonics consistent with isotropy")
    print("================================================================")
    print("test 63 complete.")
    print("================================================================")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python frb_harmonic_manifold_test63.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
