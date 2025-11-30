#!/usr/bin/env python3
"""
FRB UNIFIED 3D HELICAL EVOLUTION TEST (TEST 32)

Purpose
-------
Test whether the FRB azimuthal pattern around the unified axis shows a
coherent dependence on both polar angle theta and redshift z, i.e. a
3D helical evolution:

    phi ≈ phi0 + k_theta * theta + k_z * z

compared to a simpler purely angular model:

    phi ≈ phi0 + k_theta * theta

If including z significantly improves the fit and yields a nonzero k_z
under isotropic nulls, this indicates a global 3D twist evolution.

Region
------
We restrict to the main anisotropic shell:

    25° <= theta_unified <= 60°

and 0 <= z <= 0.8 (same redshift range as earlier helical tests).

Null hypothesis
---------------
Redshift z is shuffled among FRBs (theta, phi fixed), thereby destroying
any physical z–phi correlation but preserving the theta–phi structure.

We compare:

- |k_z| against its null distribution
- ΔAIC = AIC_model1 - AIC_model2 against null

where AIC = 2k + N ln(RSS/N) and k is the number of parameters.
"""

import sys
import numpy as np
import pandas as pd

# Monte Carlo samples
N_MC = 20000

# theta shell for helical analysis
THETA_MIN = 25.0
THETA_MAX = 60.0

# redshift range (same as earlier tests)
Z_MIN = 0.0
Z_MAX = 0.8

# ------------------------------------------------------------
# load FRB catalog
# ------------------------------------------------------------
def load_frbs(path):
    df = pd.read_csv(path)

    if "theta_unified" not in df.columns or "phi_unified" not in df.columns:
        raise ValueError("catalog must contain theta_unified and phi_unified")

    # detect redshift column
    if "z_est" in df.columns:
        print("detected redshift column: z_est")
        df = df.rename(columns={"z_est": "z"})
    elif "z" in df.columns:
        print("detected redshift column: z")
    elif "redshift" in df.columns:
        print("detected redshift column: redshift")
        df = df.rename(columns={"redshift": "z"})
    else:
        raise ValueError("no redshift column found (z_est/z/redshift)")

    df = df.dropna(subset=["theta_unified", "phi_unified", "z"])
    return df

# ------------------------------------------------------------
# simple linear least squares helper
# ------------------------------------------------------------
def fit_linear_model(y, X):
    """
    y: (N,) array
    X: (N, k) design matrix (already includes ones column if needed)

    returns: params, RSS
    """
    # least-squares fit
    params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    if residuals.size > 0:
        rss = residuals[0]
    else:
        # if design matrix is exactly determined, compute RSS manually
        y_pred = X @ params
        rss = np.sum((y - y_pred)**2)
    return params, rss

def aic(n, rss, k):
    """AIC = 2k + n ln(RSS/n)."""
    if rss <= 0:
        rss = 1e-12
    return 2*k + n*np.log(rss / n)

# ------------------------------------------------------------
# main test
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_helical_3d_evolution_test32.py frbs_unified.csv")
        sys.exit(1)

    path = sys.argv[1]
    df = load_frbs(path)
    print(f"loaded {len(df)} FRBs total")

    # apply theta + z cuts
    mask = (
        (df["theta_unified"] >= THETA_MIN) &
        (df["theta_unified"] <= THETA_MAX) &
        (df["z"] >= Z_MIN) &
        (df["z"] <= Z_MAX)
    )
    df_sel = df[mask].copy()
    print(f"selected {len(df_sel)} FRBs in shell {THETA_MIN}–{THETA_MAX} deg and z={Z_MIN}–{Z_MAX}")

    if len(df_sel) < 30:
        print("not enough FRBs in this region for Test 32.")
        sys.exit(0)

    # data arrays
    theta = df_sel["theta_unified"].values.astype(float)
    phi   = df_sel["phi_unified"].values.astype(float)
    z     = df_sel["z"].values.astype(float)

    # ensure phi is roughly in [-180,180] to avoid wrap issues
    phi = ((phi + 180.0) % 360.0) - 180.0

    # design matrices
    # model 1: phi = b0 + b1 * theta
    X1 = np.column_stack([np.ones_like(theta), theta])
    # model 2: phi = b0 + b1 * theta + b2 * z
    X2 = np.column_stack([np.ones_like(theta), theta, z])

    n = len(phi)

    # --------------------------------------------------------
    # fit on real data
    # --------------------------------------------------------
    params1, rss1 = fit_linear_model(phi, X1)
    params2, rss2 = fit_linear_model(phi, X2)

    aic1 = aic(n, rss1, k=2)
    aic2 = aic(n, rss2, k=3)

    delta_aic_obs = aic1 - aic2   # improvement by adding z
    k_z_obs = params2[2]          # coefficient on z

    print("======================================================================")
    print("FRB UNIFIED 3D HELICAL EVOLUTION TEST (TEST 32)")
    print("======================================================================")
    print(f"theta shell: {THETA_MIN}–{THETA_MAX} deg,  z range: {Z_MIN}–{Z_MAX}")
    print(f"N used: {n}")
    print("------------------------------------------------------------------")
    print("MODEL 1: phi = b0 + b1 * theta")
    print(f"  b0 = {params1[0]:.4f} deg")
    print(f"  b1 = {params1[1]:.6f} deg/deg")
    print(f"  RSS_1 = {rss1:.4f},  AIC_1 = {aic1:.4f}")
    print("------------------------------------------------------------------")
    print("MODEL 2: phi = b0 + b1 * theta + b2 * z")
    print(f"  b0 = {params2[0]:.4f} deg")
    print(f"  b1 = {params2[1]:.6f} deg/deg")
    print(f"  b2 = {params2[2]:.6f} deg per unit z")
    print(f"  RSS_2 = {rss2:.4f},  AIC_2 = {aic2:.4f}")
    print("------------------------------------------------------------------")
    print(f"observed k_z (b2) = {k_z_obs:.6f} deg per unit z")
    print(f"observed ΔAIC (model1 - model2) = {delta_aic_obs:.4f}")
    print("------------------------------------------------------------------")
    print("running Monte Carlo null (shuffle z, preserve theta, phi)...")

    # --------------------------------------------------------
    # Monte Carlo null: shuffle z only
    # --------------------------------------------------------
    rng = np.random.default_rng(123)
    k_z_null = np.zeros(N_MC, dtype=float)
    delta_aic_null = np.zeros(N_MC, dtype=float)

    # note: model 1 doesn't depend on z, so (params1, rss1, aic1) are fixed

    for i in range(N_MC):
        z_sh = np.copy(z)
        rng.shuffle(z_sh)

        X2_null = np.column_stack([np.ones_like(theta), theta, z_sh])
        params2_null, rss2_null = fit_linear_model(phi, X2_null)
        aic2_null = aic(n, rss2_null, k=3)

        k_z_null[i] = params2_null[2]
        delta_aic_null[i] = aic1 - aic2_null

    # p-values
    p_kz = np.mean(np.abs(k_z_null) >= np.abs(k_z_obs))
    p_delta = np.mean(delta_aic_null >= delta_aic_obs)

    print("------------------------------------------------------------------")
    print("MONTE CARLO RESULTS:")
    print(f"  null mean |k_z|   = {np.mean(np.abs(k_z_null)):.6f}")
    print(f"  null std  |k_z|   = {np.std(np.abs(k_z_null)):.6f}")
    print(f"  p-value(|k_z|)    = {p_kz:.6f}")
    print("------------------------------------------------------------------")
    print(f"  null mean ΔAIC    = {np.mean(delta_aic_null):.6f}")
    print(f"  null std  ΔAIC    = {np.std(delta_aic_null):.6f}")
    print(f"  p-value(ΔAIC)     = {p_delta:.6f}")
    print("------------------------------------------------------------------")
    print("interpretation:")
    print("  - |k_z| p-value tests whether adding z gives a nonzero 3D twist.")
    print("  - ΔAIC p-value tests whether the z-term truly improves the model.")
    print("======================================================================")
    print("test 32 complete.")
    print("======================================================================")

if __name__ == "__main__":
    main()
