import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def grain_intensity(theta, phi, k=50):
    x = np.deg2rad(theta)
    y = np.deg2rad(phi)
    return np.cos(k * x) * np.sin(k * y)

def latent_projection(G, SAG, EGR, CE, HF):
    return (
        1.3 * G +
        2.0 * SAG +
        1.7 * EGR +
        0.9 * CE +
        2.4 * HF
    )

# ------------------------------------------------------------
# Test 53A — backward-inference reversibility
# ------------------------------------------------------------

def backward_reconstruct_scalar(F):
    """
    forward projection is scalar, so backward inversion reconstructs
    a 5D latent vector pointing in the fixed latent-basis direction.
    """
    # fixed unit direction in latent space
    u = np.array([1, 1, 1, 1, 1], dtype=float)
    u = u / np.linalg.norm(u)

    # return matrix of shape (N,5)
    return np.outer(F, u)

def reversibility_score(G, SAG, EGR, CE, HF):
    """
    compares true latent 5-vector to reconstructed one.
    """
    F = latent_projection(G, SAG, EGR, CE, HF)

    original = np.vstack([G, SAG, EGR, CE, HF]).T          # (N,5)
    recon = backward_reconstruct_scalar(F)                 # (N,5)

    err = np.linalg.norm(original - recon, axis=1)
    return np.mean(err)

# ------------------------------------------------------------
# Test 53B — anisotropy-perturbation response
# ------------------------------------------------------------

def apply_anisotropy_perturbation(theta, phi, epsilon=0.01):
    return theta + epsilon * np.cos(np.deg2rad(phi))

def perturbation_response(G, theta, phi):
    theta_p = apply_anisotropy_perturbation(theta, phi)
    G_p = grain_intensity(theta_p, phi)
    delta = G_p - G
    return np.mean(np.abs(delta))

# ------------------------------------------------------------
# Monte Carlo nulls
# ------------------------------------------------------------

def mc_null_reversibility(N, n):
    out = []
    for _ in range(N):
        G = np.random.randn(n)
        SAG = np.random.randn(n)
        EGR = np.random.randn(n)
        CE = np.random.randn(n)
        HF = np.random.randn(n)
        out.append(reversibility_score(G, SAG, EGR, CE, HF))
    return np.array(out)

def mc_null_perturbation(N, theta, phi):
    out = []
    for _ in range(N):
        phi_s = np.random.permutation(phi)
        G = grain_intensity(theta, phi_s)
        out.append(perturbation_response(G, theta, phi_s))
    return np.array(out)

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    import sys
    if len(sys.argv) < 2:
        print("usage: python frb_latent_geometry_causal_suite_test53.py frbs_unified.csv")
        return

    df = pd.read_csv(sys.argv[1])
    print(f"loaded {len(df)} FRBs")

    theta = df["theta_unified"].values
    phi = df["phi_unified"].values

    G = grain_intensity(theta, phi)
    SAG = 0.03 * theta * np.cos(np.deg2rad(phi))
    EGR = df["z_est"].values * 0.2
    CE = df["dm"].values * df["snr"].values * 1e-5
    HF = np.sin(np.deg2rad(phi)) * np.cos(2 * np.deg2rad(theta))

    print("running Test 53A (reversibility)...")
    R_real = reversibility_score(G, SAG, EGR, CE, HF)
    R_null = mc_null_reversibility(2000, len(df))
    pA = np.mean(R_null <= R_real)

    print("running Test 53B (perturbation response)...")
    PR_real = perturbation_response(G, theta, phi)
    PR_null = mc_null_perturbation(2000, theta, phi)
    pB = np.mean(PR_null >= PR_real)

    print("===========================================================")
    print(" FRB LATENT GEOMETRY CAUSAL SUITE (TEST 53)")
    print("===========================================================")
    print(f"53A reversibility_score = {R_real:.6f}")
    print(f"  null mean = {R_null.mean():.6f}, std = {R_null.std():.6f}")
    print(f"  p-value   = {pA:.6f}")
    print("-----------------------------------------------------------")
    print(f"53B perturbation_response = {PR_real:.6f}")
    print(f"  null mean = {PR_null.mean():.6f}, std = {PR_null.std():.6f}")
    print(f"  p-value   = {pB:.6f}")
    print("-----------------------------------------------------------")
    print("interpretation:")
    print("  - low pA → field reversible → stable latent geometry")
    print("  - low pB → field responds coherently → causal field")
    print("===========================================================")
    print("test 53 complete.")
    print("===========================================================")

if __name__ == "__main__":
    main()
