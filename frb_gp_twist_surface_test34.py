#!/usr/bin/env python3
"""
FRB FULL 2D HELICAL SURFACE RECONSTRUCTION (TEST 34)

Goal:
    reconstruct the function phi(theta, z) using a 2-D Gaussian Process,
    without any assumed linear/quadratic/piecewise twist model.

    this test checks:
      - whether there is statistically significant azimuthal structure
        in the combined (theta, z) plane,
      - whether the twist depends smoothly on z,
      - whether the reconstructed surface differs from isotropic null.

Method:
    1) select FRBs in the shell where azimuthal structure is strongest:
           25° <= theta <= 60°
    2) build a Gaussian Process GP[(theta, z) → phi_mod]
       phi_mod = wrapped phi (mapped to [-180, 180] deg)
    3) compute:
           - GP log-marginal likelihood
           - RMS curvature of reconstructed surface
           - variance explained by GP vs null
    4) Monte Carlo:
           shuffle phi values, keep (theta,z)
           fit GP each time, compute same stats
           → p-values

Dependencies:
    pip install scikit-learn numpy pandas scipy
"""

import sys
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

N_MC = 500   # GP is heavy; 500 is okay. Increase to 2000 if PC can handle it.

THETA_MIN = 25
THETA_MAX = 60

# ------------------------------------------------------------
# Load FRBs
# ------------------------------------------------------------
def load_frbs(path):
    df = pd.read_csv(path)

    if "theta_unified" not in df or "phi_unified" not in df:
        raise ValueError("catalog must contain theta_unified, phi_unified")

    # detect redshift column
    if "z_est" in df.columns:
        df = df.rename(columns={"z_est": "z"})
        print("detected redshift column: z_est")
    elif "z" not in df.columns:
        raise ValueError("no redshift column (z_est or z) found.")

    df = df.dropna(subset=["theta_unified", "phi_unified", "z"])
    return df


# ------------------------------------------------------------
# Wrap phi into [-180, 180]
# ------------------------------------------------------------
def wrap_phi(phi_deg):
    x = ((phi_deg + 180) % 360) - 180
    return x


# ------------------------------------------------------------
# Gaussian Process fitting
# ------------------------------------------------------------
def fit_gp(theta, z, phi):
    # GP input features: [theta, z]
    X = np.vstack([theta, z]).T
    y = phi

    # kernel: amplitude * RBF(length_theta, length_z)
    kernel = C(50.0, (1e-2, 1e4)) * RBF(
        length_scale=[20.0, 0.2],
        length_scale_bounds=(1e-2, 1e3)
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=4.0,              # small observational noise
        normalize_y=True,
        n_restarts_optimizer=2  # keep light so Monte Carlo works
    )

    gp.fit(X, y)
    y_pred, y_std = gp.predict(X, return_std=True)

    rss = np.sum((y - y_pred)**2)
    var_expl = 1 - rss / np.sum((y - np.mean(y))**2)

    # curvature measure: RMS gradient of surface
    # simple finite diff using GP predictions
    dtheta = np.gradient(y_pred, theta)
    dz = np.gradient(y_pred, z)
    curvature = np.sqrt(np.mean(dtheta**2 + dz**2))

    return gp, rss, var_expl, curvature, y_pred


# ------------------------------------------------------------
# Monte Carlo null: shuffle phi
# ------------------------------------------------------------
def mc_null(theta, z, phi, n=N_MC):
    rss_null = []
    var_null = []
    curv_null = []

    rng = np.random.default_rng(123)

    for _ in range(n):
        phi_sh = phi[rng.permutation(len(phi))]
        _, rss, vexp, curv, _ = fit_gp(theta, z, phi_sh)
        rss_null.append(rss)
        var_null.append(vexp)
        curv_null.append(curv)

    return np.array(rss_null), np.array(var_null), np.array(curv_null)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_gp_twist_surface_test34.py frbs_unified.csv")
        return

    df = load_frbs(sys.argv[1])

    # restrict to theta shell
    mask = (df["theta_unified"] >= THETA_MIN) & (df["theta_unified"] <= THETA_MAX)
    df = df[mask]
    print(f"selected {len(df)} FRBs in θ=[{THETA_MIN},{THETA_MAX}] deg")

    theta = df["theta_unified"].values
    z     = df["z"].values
    phi   = wrap_phi(df["phi_unified"].values)

    print("=====================================================================")
    print("FRB FULL 2D HELICAL SURFACE RECONSTRUCTION (TEST 34)")
    print("=====================================================================")

    # real GP fit
    gp, rss_real, var_real, curv_real, y_pred = fit_gp(theta, z, phi)

    print(f"real RSS = {rss_real:.2f}")
    print(f"real variance explained = {var_real:.4f}")
    print(f"real curvature = {curv_real:.4f}")
    print("---------------------------------------------------------------------")
    print("running Monte Carlo null...")
    print("---------------------------------------------------------------------")

    rss_null, var_null, curv_null = mc_null(theta, z, phi)

    p_rss  = np.mean(rss_null <= rss_real)
    p_var  = np.mean(var_null >= var_real)
    p_curv = np.mean(curv_null >= curv_real)

    print("=====================================================================")
    print("SUMMARY – TEST 34: GAUSSIAN-PROCESS HELICAL SURFACE")
    print("=====================================================================")
    print(f"RSS p-value:         {p_rss:.6f}")
    print(f"variance p-value:    {p_var:.6f}")
    print(f"curvature p-value:   {p_curv:.6f}")
    print("---------------------------------------------------------------------")
    print("interpretation:")
    print(" - variance p tests whether GP captures non-random azimuthal structure")
    print(" - curvature p tests whether φ(θ,z) has structured gradients")
    print("=====================================================================")
    print("test 34 complete.")
    print("=====================================================================")


if __name__ == "__main__":
    main()
