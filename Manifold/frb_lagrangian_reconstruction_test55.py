import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------
# safe normalisation helper
# ------------------------------------------------------------
def safe_zscore(x):
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() == 0:
        return np.zeros_like(x)
    m = np.nanmean(x[mask])
    s = np.nanstd(x[mask])
    if not np.isfinite(s) or s == 0.0:
        z = x - m
        z[~np.isfinite(z)] = 0.0
        return z
    z = (x - m) / s
    z[~np.isfinite(z)] = 0.0
    return z

# ------------------------------------------------------------
# helper: very simple "spherical" gradient / laplacian proxy
# (we keep it discrete and robust rather than exact)
# ------------------------------------------------------------
def spherical_gradient(F, theta, phi):
    # we treat the index ordering as the path and only use theta
    # to rescale the phi component
    F = np.asarray(F, float)
    theta = np.asarray(theta, float)
    phi = np.asarray(phi, float)

    dF_di = np.gradient(F)  # derivative along index
    dF_dtheta = dF_di
    dF_dphi = dF_di  # symmetric toy model

    # avoid sin(theta) = 0 singularities
    s = np.sin(theta)
    s[~np.isfinite(s)] = 1.0
    s[np.abs(s) < 1e-3] = np.sign(s[np.abs(s) < 1e-3]) * 1e-3

    grad_sq = dF_dtheta**2 + (dF_dphi / s)**2
    return dF_dtheta, dF_dphi, grad_sq

def spherical_laplacian(F, theta, phi):
    dF_dtheta, dF_dphi, _ = spherical_gradient(F, theta, phi)

    d2F_dtheta2 = np.gradient(dF_dtheta)
    d2F_dphi2 = np.gradient(dF_dphi)

    s = np.sin(theta)
    s2 = s**2
    s2[~np.isfinite(s2)] = 1.0
    s2[np.abs(s2) < 1e-3] = np.sign(s2[np.abs(s2) < 1e-3]) * 1e-3

    lap = d2F_dtheta2 + d2F_dphi2 / s2
    lap[~np.isfinite(lap)] = 0.0
    return lap

# ------------------------------------------------------------
# euler–lagrange residuals
# ------------------------------------------------------------
def EL_residual_L0(F, theta, phi):
    # L0 = 1/2 |∇F|^2 → EL residual ≈ -ΔF
    lap = spherical_laplacian(F, theta, phi)
    return -lap

def EL_residual_L1(F, theta, phi):
    # L1 = 1/2 (ΔF)^2 → EL residual ≈ -Δ(ΔF)
    lap = spherical_laplacian(F, theta, phi)
    lap2 = spherical_laplacian(lap, theta, phi)
    return -lap2

def EL_residual_L2(F, theta, phi, alpha=0.1):
    # L2 = 1/2 (|∇F|^2 + alpha F^2) → EL residual ≈ alpha F - ΔF
    lap = spherical_laplacian(F, theta, phi)
    dL_dF = alpha * F
    return dL_dF - lap

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    import sys
    if len(sys.argv) < 2:
        print("usage: python frb_lagrangian_reconstruction_test55.py frbs_unified.csv")
        return

    df = pd.read_csv(sys.argv[1])
    print(f"loaded {len(df)} FRBs")

    # basic angles
    th = np.deg2rad(df["theta_unified"].values)
    ph = np.deg2rad(df["phi_unified"].values)

    # ------------------------------------------------------------
    # robust grain intensity (avoid zero distances)
    # ------------------------------------------------------------
    coords = np.column_stack([th, ph])
    N = len(df)
    G = np.zeros(N, dtype=float)

    print("computing grain intensity...")
    for i in range(N):
        d = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
        # exclude self and exact zeros
        d = d[d > 0.0]
        if d.size == 0:
            G[i] = 0.0
        else:
            k = min(6, d.size)
            local = np.sort(d)[:k]
            mean_d = np.mean(local)
            if mean_d <= 0.0 or not np.isfinite(mean_d):
                G[i] = 0.0
            else:
                G[i] = 1.0 / mean_d

    # sag: unified theta trend
    SAG_raw = th

    # egr: dm trend
    EGR_raw = df["dm"].values

    # ce: simple cross coupling
    CE_raw = (df["snr"].values * df["fluence"].values)

    # hf: harmonic phi structure
    HF_raw = np.sin(ph) + 0.5 * np.sin(2 * ph)

    # z-score each component safely
    Gz   = safe_zscore(G)
    SAGz = safe_zscore(SAG_raw)
    EGRz = safe_zscore(EGR_raw)
    CEz  = safe_zscore(CE_raw)
    HFz  = safe_zscore(HF_raw)

    # latent field
    F = Gz + 0.5 * SAGz + 0.25 * EGRz + 0.25 * CEz + 0.25 * HFz

    # final cleanup
    if not np.all(np.isfinite(F)):
        finite = np.isfinite(F)
        if finite.sum() == 0:
            F[:] = 0.0
        else:
            median_F = np.nanmedian(F[finite])
            F[~finite] = median_F

    print("computing euler–lagrange residuals...")
    R0 = EL_residual_L0(F, th, ph)
    R1 = EL_residual_L1(F, th, ph)
    R2 = EL_residual_L2(F, th, ph, alpha=0.1)

    S0_real = float(np.nansum(R0**2))
    S1_real = float(np.nansum(R1**2))
    S2_real = float(np.nansum(R2**2))

    # ------------------------------------------------------------
    # monte carlo null
    # ------------------------------------------------------------
    Nmc = 2000
    S0_null = np.zeros(Nmc, dtype=float)
    S1_null = np.zeros(Nmc, dtype=float)
    S2_null = np.zeros(Nmc, dtype=float)

    print("running monte carlo null...")
    for j in tqdm(range(Nmc)):
        Fsh = np.random.permutation(F)

        R0s = EL_residual_L0(Fsh, th, ph)
        R1s = EL_residual_L1(Fsh, th, ph)
        R2s = EL_residual_L2(Fsh, th, ph, alpha=0.1)

        S0_null[j] = np.nansum(R0s**2)
        S1_null[j] = np.nansum(R1s**2)
        S2_null[j] = np.nansum(R2s**2)

    # guard against any residual nans in nulls
    for arr in (S0_null, S1_null, S2_null):
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.nanmedian(arr[~bad])

    p0 = float(np.mean(S0_null <= S0_real))
    p1 = float(np.mean(S1_null <= S1_real))
    p2 = float(np.mean(S2_null <= S2_real))

    print("\n===================================================")
    print("      FRB LAGRANGIAN RECONSTRUCTION TEST (TEST 55)")
    print("===================================================\n")

    print(f"L0 gradient field:     S_real={S0_real:.6f}, null_mean={np.mean(S0_null):.6f}, p={p0:.6f}")
    print(f"L1 curvature field:    S_real={S1_real:.6f}, null_mean={np.mean(S1_null):.6f}, p={p1:.6f}")
    print(f"L2 harmonic field:     S_real={S2_real:.6f}, null_mean={np.mean(S2_null):.6f}, p={p2:.6f}")

    print("\n---------------------------------------------------")
    print("interpretation:")
    print("- lowest S_real and small p → field best matches that lagrangian")
    print("- if L1 best: curvature-dominated field")
    print("- if L0 best: gradient-driven field")
    print("- if L2 best: harmonic / oscillatory latent field")
    print("===================================================")
    print("test 55 complete.")
    print("===================================================")


if __name__ == "__main__":
    main()
