#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------
# helper: compute latent geometry field (same as test 50 / 51)
# --------------------------------------------------------------
def compute_grain_intensity(df):
    ra  = np.deg2rad(df["ra"].values)
    dec = np.deg2rad(df["dec"].values)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    coords = np.vstack([x, y, z]).T
    G = coords @ coords.T
    return G.mean(axis=1)

def compute_spatial_gradient(theta):
    theta = np.deg2rad(theta)
    dtheta = np.gradient(theta)
    return dtheta

def compute_energy_gradient(df):
    fields = ["dm", "snr", "fluence", "width", "z_est"]
    E = np.zeros(len(df))
    for f in fields:
        v = df[f].values
        g = np.gradient(v)
        E += np.abs(g)
    return E

def compute_cross_energy(df):
    fields = ["dm", "snr", "fluence", "width", "z_est"]
    CE = np.zeros(len(df))
    for i in range(len(fields)):
        for j in range(i+1, len(fields)):
            v1 = df[fields[i]].values
            v2 = df[fields[j]].values
            CE += np.abs(np.gradient(v1 * v2))
    return CE

def compute_harmonic_features(phi):
    phi = np.deg2rad(phi)
    a1 = np.abs(np.mean(np.sin(phi)))
    a2 = np.abs(np.mean(np.sin(2 * phi)))
    return a1 + a2

# --------------------------------------------------------------
# curvature operators
# --------------------------------------------------------------

def discrete_curvature(F):
    d1 = np.gradient(F)
    d2 = np.gradient(d1)
    return np.abs(d2).mean()

# forward evolution (pseudo-time)
def evolve_forward(F, step=0.01):
    dF = np.gradient(F)
    return F + step * dF

# backward evolution (pseudo-time)
def evolve_backward(F, step=0.01):
    dF = np.gradient(F)
    return F - step * dF

# --------------------------------------------------------------
# temporal curvature closure tests
# --------------------------------------------------------------

def test_54A_curvature_closure(F):
    k0 = discrete_curvature(F)
    Ff = evolve_forward(F)
    Fb = evolve_backward(F)
    k_final = discrete_curvature((Ff + Fb) / 2)
    return abs(k_final - k0)

def test_54B_harmonic_curvature_drift(F):
    k1 = discrete_curvature(F)
    Ff = evolve_forward(F)
    k2 = discrete_curvature(Ff)
    return abs(k2 - k1)

def test_54C_second_order_temporal_closure(F):
    k0 = discrete_curvature(F)
    Ffb = evolve_backward(evolve_forward(F))
    k_final = discrete_curvature(Ffb)
    return abs(k_final - k0)

# Monte Carlo null
def mc_null(F, N=5000):
    Nf = len(F)
    outA, outB, outC = [], [], []
    for _ in range(N):
        Fshuf = np.random.permutation(F)

        outA.append(test_54A_curvature_closure(Fshuf))
        outB.append(test_54B_harmonic_curvature_drift(Fshuf))
        outC.append(test_54C_second_order_temporal_closure(Fshuf))

    return np.array(outA), np.array(outB), np.array(outC)


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main():
    import sys
    if len(sys.argv) != 2:
        print("usage: python frb_latent_geometry_temporal_closure_test54.py frbs_unified.csv")
        return

    df = pd.read_csv(sys.argv[1])
    N = len(df)
    print(f"loaded {N} FRBs")

    # compute latent geometry components
    G   = compute_grain_intensity(df)
    SAG = compute_spatial_gradient(df["theta_unified"].values)
    EGR = compute_energy_gradient(df)
    CE  = compute_cross_energy(df)
    HF  = compute_harmonic_features(df["phi_unified"].values)

    # full latent field
    F = (G + SAG + EGR + CE + HF)
    F = (F - F.mean()) / F.std()

    print("running real curvature metrics...")
    A_real = test_54A_curvature_closure(F)
    B_real = test_54B_harmonic_curvature_drift(F)
    C_real = test_54C_second_order_temporal_closure(F)

    print("running Monte Carlo null...")
    A_null, B_null, C_null = mc_null(F, N=5000)

    # p-values
    pA = np.mean(A_null <= A_real)
    pB = np.mean(B_null <= B_real)
    pC = np.mean(C_null <= C_real)

    print("===================================================================")
    print("   FRB TEMPORAL CURVATURE CLOSURE SUITE (TEST 54)")
    print("===================================================================")
    print(f"54A curvature closure      = {A_real:.6f}")
    print(f"null mean = {A_null.mean():.6f}, std = {A_null.std():.6f}, p = {pA:.6f}")
    print("-------------------------------------------------------------------")
    print(f"54B harmonic curvature drift = {B_real:.6f}")
    print(f"null mean = {B_null.mean():.6f}, std = {B_null.std():.6f}, p = {pB:.6f}")
    print("-------------------------------------------------------------------")
    print(f"54C second-order closure   = {C_real:.6f}")
    print(f"null mean = {C_null.mean():.6f}, std = {C_null.std():.6f}, p = {pC:.6f}")
    print("-------------------------------------------------------------------")
    print("interpretation:")
    print("  - low pA → strong curvature closure (field-like)")
    print("  - low pB → stable harmonic curvature evolution")
    print("  - low pC → second-order closure → real causal geometry")
    print("===================================================================")
    print("test 54 complete.")
    print("===================================================================")


if __name__ == "__main__":
    main()
