#!/usr/bin/env python3
# ================================================================
# FRB SPECTRAL SYMMETRY-BREAKING TEST (TEST 67)
# ================================================================
# Detects alignment of Laplacian eigenmodes with the unified cosmic axis
# ================================================================
import sys
import csv
import math
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from numpy.linalg import eigh

# ================================================================
def stable_norm(v):
    v = np.asarray(v, float)
    m = np.nanmean(v)
    s = np.nanstd(v)
    if s == 0 or np.isnan(s):
        return np.zeros_like(v)
    return (v - m) / s

# ================================================================
# load RA/Dec
# ================================================================
def load_frb_catalog(path):
    RA, Dec = [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        first = next(iter(reader))
        cols = list(first.keys())

        def pick(names):
            for n in names:
                if n in cols:
                    return n
            return None

        ra_key  = pick(["RA","ra","RA_deg","ra_deg","RAJ2000"])
        dec_key = pick(["Dec","dec","Dec_deg","dec_deg","DecJ2000"])
        if ra_key is None or dec_key is None:
            raise RuntimeError("RA/Dec columns not found.")

        RA.append(float(first[ra_key]))
        Dec.append(float(first[dec_key]))
        for row in reader:
            RA.append(float(row[ra_key]))
            Dec.append(float(row[dec_key]))
    return np.array(RA), np.array(Dec)

# ================================================================
# unit vectors
# ================================================================
def radec_to_unit(ra, dec):
    ra  = np.deg2rad(ra)
    dec = np.deg2rad(dec)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x,y,z])

# ================================================================
# kNN manifold graph
# ================================================================
def build_knn_graph(X, k=12):
    N = len(X)
    tree = cKDTree(X)
    dists, idxs = tree.query(X, k=k+1)
    idxs  = idxs[:,1:]
    dists = dists[:,1:]

    neighbors = [[] for _ in range(N)]
    eps = 0.02
    for i in range(N):
        for j, d in zip(idxs[i], dists[i]):
            w = math.exp(-(d*d)/eps)
            neighbors[i].append((j, w))
            neighbors[j].append((i, w))
    return neighbors

# ================================================================
# build Laplacian L = D - W
# W is sparse-ish but stored as dense for simplicity (600x600 ok)
# ================================================================
def build_laplacian(X, neighbors):
    N = len(X)
    W = np.zeros((N,N))
    for i in range(N):
        for (j,w) in neighbors[i]:
            W[i,j] = w
            W[j,i] = w
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L

# ================================================================
# estimate spatial gradient of eigenvector φ_k
# ================================================================
def grad_field(X, phi, neighbors):
    N = len(X)
    G = np.zeros((N,3))
    for i in range(N):
        g = np.zeros(3)
        for (j,w) in neighbors[i]:
            diff = phi[j] - phi[i]
            dirv = X[j] - X[i]
            nrm = np.linalg.norm(dirv)
            if nrm > 0:
                g += diff * (dirv / nrm)
        G[i] = g
    return G

# ================================================================
# PCA dominant direction
# ================================================================
def pca_direction(V):
    # V: Nx3 gradients
    C = np.cov(V.T)
    vals, vecs = eigh(C)
    # largest eigenvector
    return vecs[:, np.argmax(vals)]

# ================================================================
# unified axis = z-axis in your coordinate system
# ================================================================
def unified_axis():
    return np.array([0.0, 0.0, 1.0])

# ================================================================
def spectral_alignment_score(RA, Dec):
    X = radec_to_unit(RA, Dec)
    neighbors = build_knn_graph(X, k=12)
    L = build_laplacian(X, neighbors)

    # compute first 11 eigenpairs (0 is trivial)
    # eigh sorts ascending λ
    vals, vecs = eigh(L)
    # eigenvectors: columns
    phis = vecs[:,1:11]  # take φ1 ... φ10

    ua = unified_axis()
    scores = []

    for k in range(10):
        phi = phis[:,k]
        G = grad_field(X, phi, neighbors)
        d = pca_direction(G)
        d = d / (np.linalg.norm(d) + 1e-12)

        # alignment with unified axis
        align = abs(np.dot(d, ua))
        scores.append(align)

    A = float(np.mean(scores))
    return A

# ================================================================
# isotropic distribution
# ================================================================
def random_isotropic(N):
    u = np.random.uniform(0,1,N)
    v = np.random.uniform(-1,1,N)
    ra  = 360*u
    dec = np.rad2deg(np.arcsin(v))
    return ra, dec

# ================================================================
# MAIN
# ================================================================
def main(path):
    print("[INFO] loading FRB catalog...")
    RA, Dec = load_frb_catalog(path)
    N = len(RA)
    print(f"[INFO] N_FRB = {N}")

    print("[INFO] computing spectral alignment on real data...")
    A_real = spectral_alignment_score(RA, Dec)

    print("[INFO] building null distribution...")
    A_null = []
    for _ in tqdm(range(2000), desc="MC null"):
        rra, rdec = random_isotropic(N)
        A_null.append(spectral_alignment_score(rra, rdec))
    A_null = np.array(A_null)

    mean_null = float(np.mean(A_null))
    std_null  = float(np.std(A_null))
    p = float(np.mean(A_null >= A_real))

    print("================================================")
    print(" FRB SPECTRAL SYMMETRY-BREAKING TEST (TEST 67)")
    print("================================================")
    print(f"A_real     = {A_real:.6f}")
    print(f"null mean  = {mean_null:.6f}")
    print(f"null std   = {std_null:.6f}")
    print(f"p-value    = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  - low p  → eigenmodes share preferred direction → symmetry breaking")
    print("  - high p → eigenmodes random → no spectral preference")
    print("================================================")
    print("test 67 complete.")
    print("================================================")

# ================================================================
if __name__ == "__main__":
    main(sys.argv[1])
