#!/usr/bin/env python3
"""
FRB UNIFIED-AXIS ANISOTROPY FLOW TEST (TEST 35)

Purpose:
--------
Measure the vector flow pattern of the FRB anisotropy surface:
    phi = F(theta, z)

We compute:
    v_theta = dF/dtheta
    v_z     = dF/dz

and evaluate:
    - directional coherence of the vector field
    - mean curl
    - mean divergence
    - total flow energy E = <|v|^2>

All compared to a Monte Carlo isotropic null (phi scrambled).
"""

import sys
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from numpy.random import default_rng

N_MC = 20000
THETA_MIN = 25.0
THETA_MAX = 60.0

def load_frbs(path):
    df = pd.read_csv(path)
    if "theta_unified" not in df or "phi_unified" not in df:
        raise ValueError("Catalog must contain theta_unified, phi_unified")
    if "z_est" in df: df = df.rename(columns={"z_est": "z"})
    elif "redshift" in df: df = df.rename(columns={"redshift": "z"})
    return df.dropna(subset=["theta_unified","phi_unified","z"])

def fit_gp(theta, z, phi):
    X = np.column_stack([theta, z])
    y = phi
    kernel = C(1.0, (0.01, 10.0)) * RBF([10.0, 0.1], (1.0, 1000.0))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
    gp.fit(X, y)
    return gp

def compute_gradients(gp, grid_theta, grid_z):
    T, Z = np.meshgrid(grid_theta, grid_z, indexing="ij")
    pts = np.column_stack([T.ravel(), Z.ravel()])
    f = gp.predict(pts).reshape(T.shape)

    dT = grid_theta[1] - grid_theta[0]
    dZ = grid_z[1] - grid_z[0]

    dF_dT = np.gradient(f, dT, axis=0)
    dF_dZ = np.gradient(f, dZ, axis=1)
    return dF_dT, dF_dZ

def flow_statistics(dF_dT, dF_dZ):
    vT = dF_dT
    vZ = dF_dZ
    vmag = np.sqrt(vT**2 + vZ**2)

    C_dir = np.sqrt((vT.mean())**2 + (vZ.mean())**2) / (vmag.mean() + 1e-12)

    curl = np.gradient(vT, axis=1) - np.gradient(vZ, axis=0)
    div  = np.gradient(vT, axis=0) + np.gradient(vZ, axis=1)

    return {
        "C_dir": C_dir,
        "curl_mean": curl.mean(),
        "div_mean": div.mean(),
        "energy": np.mean(vmag**2)
    }

def mc_null(theta, z, phi, grid_theta, grid_z):
    rng = default_rng(123)
    real_stats = None
    mc_vals = {"C_dir":[], "curl":[], "div":[], "energy":[]}

    # compute real first
    gp = fit_gp(theta, z, phi)
    gT, gZ = compute_gradients(gp, grid_theta, grid_z)
    real_stats = flow_statistics(gT, gZ)

    # monte carlo
    for _ in range(N_MC):
        phi_sh = rng.permutation(phi)
        gp_null = fit_gp(theta, z, phi_sh)
        gT_n, gZ_n = compute_gradients(gp_null, grid_theta, grid_z)
        s = flow_statistics(gT_n, gZ_n)
        mc_vals["C_dir"].append(s["C_dir"])
        mc_vals["curl"].append(s["curl_mean"])
        mc_vals["div"].append(s["div_mean"])
        mc_vals["energy"].append(s["energy"])

    # compute p-values (right-tail)
    def pval(real, null):
        null = np.array(null)
        return np.mean(null >= real)

    return real_stats, {
        "p_C_dir": pval(real_stats["C_dir"], mc_vals["C_dir"]),
        "p_curl":  pval(real_stats["curl_mean"], mc_vals["curl"]),
        "p_div":   pval(real_stats["div_mean"], mc_vals["div"]),
        "p_energy":pval(real_stats["energy"], mc_vals["energy"])
    }


def main():
    if len(sys.argv) < 2:
        print("usage: python frb_anisotropy_flow_test35.py frbs_unified.csv")
        return

    df = load_frbs(sys.argv[1])
    sel = df[(df["theta_unified"]>=THETA_MIN)&(df["theta_unified"]<=THETA_MAX)&
             (df["z"]>=0)&(df["z"]<=0.8)]
    print(f"selected {len(sel)} FRBs")

    theta = sel["theta_unified"].values
    z     = sel["z"].values
    phi   = sel["phi_unified"].values

    grid_theta = np.linspace(THETA_MIN, THETA_MAX, 40)
    grid_z     = np.linspace(0.0, 0.8, 40)

    print("\n=====================================================================")
    print("FRB VECTOR FLOW TWIST TEST (TEST 35)")
    print("=====================================================================\n")

    real, pvals = mc_null(theta, z, phi, grid_theta, grid_z)

    print("Observed flow statistics:")
    print(f"Directional coherence C_dir = {real['C_dir']:.4f}")
    print(f"Mean curl                    = {real['curl_mean']:.4f}")
    print(f"Mean divergence              = {real['div_mean']:.4f}")
    print(f"Flow energy <|v|^2>          = {real['energy']:.4f}")
    print("\nMonte Carlo p-values:")
    print(f"p(C_dir)   = {pvals['p_C_dir']:.6f}")
    print(f"p(curl)    = {pvals['p_curl']:.6f}")
    print(f"p(div)     = {pvals['p_div']:.6f}")
    print(f"p(energy)  = {pvals['p_energy']:.6f}")

    print("\n=====================================================================")
    print("test 35 complete.")
    print("=====================================================================")

if __name__ == "__main__":
    main()
