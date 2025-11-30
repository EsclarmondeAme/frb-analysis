#!/usr/bin/env python3
# ================================================================
# FRB INTRINSIC CURVATURE RECONSTRUCTION (TEST 64)
# Pure-Python version (stable Ricci)
# ================================================================
import sys
import csv
import math
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# ================================================================
# utility
# ================================================================
def stable_normalize(v):
    v = np.asarray(v, float)
    m = np.nanmean(v)
    s = np.nanstd(v)
    if s == 0 or np.isnan(s):
        return np.zeros_like(v)
    return (v - m) / s

# ================================================================
# load catalog
# ================================================================
def load_frb_catalog(path):
    RA, Dec = [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first = next(iter(reader))
        cols = list(first.keys())

        def pick(keys):
            for k in keys:
                if k in cols:
                    return k
            return None

        ra_key  = pick(["RA","ra","RA_deg","ra_deg","RAJ2000"])
        dec_key = pick(["Dec","dec","Dec_deg","dec_deg","DecJ2000"])

        if ra_key is None or dec_key is None:
            raise RuntimeError("Could not detect RA/Dec headers")

        RA.append(float(first[ra_key])); Dec.append(float(first[dec_key]))
        for row in reader:
            RA.append(float(row[ra_key]))
            Dec.append(float(row[dec_key]))

    return np.array(RA), np.array(Dec)

# ================================================================
# spherical utils
# ================================================================
def radec_to_unit(ra, dec):
    ra  = np.deg2rad(ra)
    dec = np.deg2rad(dec)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x,y,z])

def sph_dist(u, v):
    d = np.sum(u * v, axis=-1)
    return np.arccos(np.clip(d, -1, 1))

# ================================================================
# graph
# ================================================================
def build_knn_graph(distmat, k=12, eps=0.01):
    N = distmat.shape[0]
    idxs = np.argsort(distmat, axis=1)[:,1:k+1]
    neighbors = [[] for _ in range(N)]
    W = {}

    for i in range(N):
        for j in idxs[i]:
            d = distmat[i,j]
            w = math.exp(-(d*d)/eps)
            if w < 1e-14:
                w = 1e-14
            neighbors[i].append((j,w))
            neighbors[j].append((i,w))
            W[(i,j)] = w
            W[(j,i)] = w

    return neighbors, W

# ================================================================
# Forman-Ricci curvature (safe version)
# ================================================================
def forman_ricci(neighbors, W, i, j):
    w_ij = W[(i,j)]
    w_i = sum(w for _,w in neighbors[i])
    w_j = sum(w for _,w in neighbors[j])
    if w_i <= 0 or w_j <= 0:
        return 0.0

    s1, s2 = 0.0, 0.0

    for (k, w_ik) in neighbors[i]:
        if k != j and w_ik > 0:
            s1 += math.sqrt(w_ij / w_ik)

    for (k, w_jk) in neighbors[j]:
        if k != i and w_jk > 0:
            s2 += math.sqrt(w_ij / w_jk)

    return w_ij*(1/w_i + 1/w_j) - (s1 + s2)

# ================================================================
# Wasserstein-1 (safe for k small)
# ================================================================
def wasserstein_1(d_i, w_i, d_j, w_j):
    wi = np.array(w_i, float)
    wj = np.array(w_j, float)
    wi /= np.sum(wi)
    wj /= np.sum(wj)

    C = np.abs(d_i[:,None] - d_j[None,:])

    i=j=0
    cost = 0.0
    wi = wi.copy(); wj = wj.copy()

    while i < len(wi) and j < len(wj):
        flow = min(wi[i], wj[j])
        cost += flow * C[i,j]
        wi[i] -= flow
        wj[j] -= flow
        if wi[i] < 1e-15: i += 1
        if wj[j] < 1e-15: j += 1

    return cost

# ================================================================
# Ollivier–Ricci curvature
# ================================================================
def ollivier_ricci(i, j, neighbors, X, distmat):
    d_i = []; w_i = []
    for (u, w) in neighbors[i]:
        d_i.append(distmat[i,u]); w_i.append(w)

    d_j = []; w_j = []
    for (u, w) in neighbors[j]:
        d_j.append(distmat[j,u]); w_j.append(w)

    d_ij = distmat[i,j]
    if d_ij < 1e-15:
        return 0.0

    W1 = wasserstein_1(np.array(d_i), w_i, np.array(d_j), w_j)
    return 1.0 - W1/d_ij

# ================================================================
# curvature score
# ================================================================
def curvature_score(RA, Dec, k=12):
    X = radec_to_unit(RA, Dec)
    N = len(RA)

    distmat = np.zeros((N,N))
    for i in range(N):
        distmat[i] = sph_dist(X[i], X)

    neighbors, W = build_knn_graph(distmat, k=k)

    olliv, forman = [], []
    for i in range(N):
        for (j,w) in neighbors[i]:
            if i < j:
                k1 = ollivier_ricci(i,j,neighbors,X,distmat)
                k2 = forman_ricci(neighbors,W,i,j)
                olliv.append(k1)
                forman.append(k2)

    olliv  = stable_normalize(olliv)
    forman = stable_normalize(forman)

    C = 0.5*(olliv + forman)

    # curvature mapped to nodes
    nodeC = np.zeros(N)
    count = np.zeros(N)
    idx = 0
    for i in range(N):
        for (j,w) in neighbors[i]:
            if i < j:
                nodeC[i] += C[idx]
                nodeC[j] += C[idx]
                count[i] += 1
                count[j] += 1
                idx += 1
    nodeC = np.divide(nodeC, count, out=np.zeros_like(nodeC), where=(count>0))

    # curvature stats
    C_mean = float(np.nanmean(C))
    C_var  = float(np.nanvar(C))

    # axis angle
    theta_u = np.rad2deg(np.arccos(X[:,2]))
    rho = np.corrcoef(nodeC, theta_u)[0,1]

    # Laplacian
    L = np.zeros((N,N))
    for i in range(N):
        s = 0
        for (j,w) in neighbors[i]:
            L[i,j] = -w
            s += w
        L[i,i] = s

    val, vec = np.linalg.eigh(L)
    idx = np.argsort(val)
    val = val[idx]; vec = vec[:,idx]

    E = [abs(np.dot(nodeC, vec[:,m])) for m in range(20)]
    E = np.array(E)

    R = float(np.sum(E[:5])/(np.sum(E)+1e-12))

    K = C_mean + C_var + abs(rho) + R
    return K

# ================================================================
# isotropic sky
# ================================================================
def random_isotropic(N):
    u = np.random.uniform(0,1,N)
    v = np.random.uniform(-1,1,N)
    ra  = 360*u
    dec = np.rad2deg(np.arcsin(v))
    return ra, dec

# ================================================================
# main
# ================================================================
def main(path):
    print("[INFO] loading FRB catalog...")
    RA, Dec = load_frb_catalog(path)
    N = len(RA)
    print(f"[INFO] N_FRB = {N}")

    print("[INFO] computing real curvature score...")
    K_real = curvature_score(RA, Dec)

    print("[INFO] building null distribution...")
    nulls = []
    for _ in tqdm(range(2000), desc="MC null"):
        rra, rdec = random_isotropic(N)
        nulls.append(curvature_score(rra,rdec))

    nulls = np.array(nulls)
    m = float(np.mean(nulls))
    s = float(np.std(nulls))
    p = float(np.mean(nulls >= K_real))

    print("================================================")
    print(" FRB INTRINSIC CURVATURE RECONSTRUCTION (TEST 64)")
    print("================================================")
    print(f"K_real     = {K_real:.6f}")
    print(f"null mean  = {m:.6f}")
    print(f"null std   = {s:.6f}")
    print(f"p-value    = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  - low p  → intrinsic curvature detected")
    print("  - high p → curvature consistent with isotropy")
    print("================================================")
    print("test 64 complete.")
    print("================================================")

if __name__ == "__main__":
    main(sys.argv[1])
