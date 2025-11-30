#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
import sys

# ------------------------------------------------------------
# compute local grain intensity
# ------------------------------------------------------------
def compute_grain_intensity(theta, phi, k=40):
    pts = np.vstack([theta, phi]).T
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k+1)
    local_density = 1.0 / (np.mean(dists[:, 1:], axis=1) + 1e-12)
    return local_density

# ------------------------------------------------------------
# energy gradient
# ------------------------------------------------------------
def compute_energy_gradient(values):
    order = np.argsort(values)
    diffs = np.diff(values[order])
    if len(diffs) == 0:
        return 0.0
    return np.mean(diffs ** 2)

# ------------------------------------------------------------
# propagate field forward using anisotropy kernel
# ------------------------------------------------------------
def propagate_field(theta, phi, field, steps=50, alpha=0.02):
    t = theta.copy()
    p = phi.copy()
    f = field.copy()

    for _ in range(steps):
        # anisotropy kernel drift
        dtheta = alpha * np.cos(p)
        dphi   = alpha * np.sin(t)

        t = (t + dtheta) % (2*np.pi)
        p = (p + dphi) % (2*np.pi)

        # diffusion of field
        f = f + 0.05 * (np.roll(f,1) + np.roll(f,-1) - 2*f)

    return t, p, f

# ------------------------------------------------------------
# compute tensor deviation between original and propagated field
# ------------------------------------------------------------
def tensor_distance(f0, f1):
    return np.sqrt(np.mean((f0 - f1)**2))

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_latent_geometry_causal_propagation_test52.py frbs_unified.csv")
        return

    df = pd.read_csv(sys.argv[1])
    print(f"loaded {len(df)} FRBs")

    theta = np.deg2rad(df["theta_unified"].values)
    phi   = np.deg2rad(df["phi_unified"].values)

    # -------------------------------
    # latent field at t=0
    # -------------------------------
    G = compute_grain_intensity(theta, phi, k=30)
    E = compute_energy_gradient(df["dm"].values)

    field0 = G / np.max(G) + 0.1 * E

    print("propagating latent geometry...")
    theta_p, phi_p, field1 = propagate_field(theta, phi, field0)

    # -------------------------------
    # compute causal deviation
    # -------------------------------
    D_real = tensor_distance(field0, field1)

    # -------------------------------
    # Monte Carlo null: shuffle field before propagation
    # -------------------------------
    print("running Monte Carlo null ...")
    Nmc = 5000
    D_null = []

    for _ in tqdm(range(Nmc)):
        shuffled = np.random.permutation(field0)
        _, _, fnull = propagate_field(theta, phi, shuffled)
        D_null.append(tensor_distance(shuffled, fnull))

    D_null = np.array(D_null)

    mean_null = np.mean(D_null)
    std_null = np.std(D_null)
    p = np.mean(D_null <= D_real)

    # -------------------------------
    # report
    # -------------------------------
    print("===================================================================")
    print("       FRB LATENT GEOMETRY CAUSAL PROPAGATION TEST (TEST 52)")
    print("===================================================================")
    print(f"real causal deviation D_real = {D_real:.6f}")
    print(f"null mean                   = {mean_null:.6f}")
    print(f"null std                    = {std_null:.6f}")
    print(f"p-value                     = {p:.6f}")
    print("-------------------------------------------------------------------")
    print("interpretation:")
    print("  - small p: latent geometry propagates coherently → causal field")
    print("  - large p: latent geometry collapses → accidental pattern")
    print("===================================================================")
    print("test 52 complete.")
    print("===================================================================")

if __name__ == "__main__":
    main()
