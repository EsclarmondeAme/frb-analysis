#!/usr/bin/env python3
# ================================================================
# FRB NONLINEAR 3D TWIST EVOLUTION TEST (TEST 33)
# ------------------------------------------------
# This test extends Test 32 by allowing *nonlinear* evolution of
# the azimuthal phase ϕ with redshift z in the unified-axis frame.
#
# Models tested:
#   M1: linear     phi = a0 + a1*theta + a2*z
#   M2: quadratic  phi = b0 + b1*theta + b2*z + b3*z**2
#   M3: piecewise  phi = c0 + c1*theta + c2*z_low   (z < 0.35)
#                   phi = d0 + d1*theta + d2*z_high (z ≥ 0.35)
#
# Monte Carlo null:
#   shuffle z among FRBs (preserving theta, phi).
#   fit each model, compute AIC differences.
#
# Outputs:
#   - best-fit parameters for each model
#   - AIC comparisons
#   - Monte Carlo p-values
#
# ================================================================

import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ------------------------------------------------------------
# utility: AIC from RSS and k parameters
# ------------------------------------------------------------
def AIC_from_RSS(RSS, k, n):
    return 2*k + n*np.log(RSS/n)

# ------------------------------------------------------------
# model definitions
# ------------------------------------------------------------
def model_lin(vars, a0, a1, a2):
    theta, z = vars
    return a0 + a1*theta + a2*z

def model_quad(vars, b0, b1, b2, b3):
    theta, z = vars
    return b0 + b1*theta + b2*z + b3*(z**2)

def model_piecewise(vars, c0, c1, c2, d0, d1, d2):
    theta, z = vars
    out = np.zeros_like(z)
    low = z < 0.35
    high = ~low
    out[low]  = c0 + c1*theta[low]  + c2*z[low]
    out[high] = d0 + d1*theta[high] + d2*z[high]
    return out

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_nonlinear_3d_twist_test33.py frbs_unified.csv")
        sys.exit(1)

    path = sys.argv[1]
    df = pd.read_csv(path)

    if "z_est" not in df.columns:
        print("error: no redshift column found")
        sys.exit(1)

    # unified-axis coordinates
    if ("theta_unified" not in df.columns) or ("phi_unified" not in df.columns):
        print("error: this test requires theta_unified and phi_unified")
        sys.exit(1)

    theta = df["theta_unified"].values
    phi   = df["phi_unified"].values
    z     = df["z_est"].values

    # select shell 25–60 deg where anisotropy is strongest
    mask = (theta >= 25.0) & (theta <= 60.0) & (z >= 0.0) & (z <= 0.8)
    theta = theta[mask]
    phi   = phi[mask]
    z     = z[mask]

    N = len(phi)
    print("detected redshift column: z_est")
    print(f"selected {N} FRBs in shell 25–60 deg and z=0–0.8")

    print("="*70)
    print("FRB NONLINEAR 3D TWIST EVOLUTION TEST (TEST 33)")
    print("="*70)

    # ------------------------------------------------------------
    # fit each model to real data
    # ------------------------------------------------------------
    xvars = (theta, z)

    # linear model
    p_lin, _ = curve_fit(model_lin, xvars, phi, p0=[90, 0, 0])
    phi_lin = model_lin(xvars, *p_lin)
    RSS_lin = np.sum((phi - phi_lin)**2)
    AIC_lin = AIC_from_RSS(RSS_lin, 3, N)

    # quadratic model
    p_quad, _ = curve_fit(model_quad, xvars, phi, p0=[90, 0, 0, 0])
    phi_quad = model_quad(xvars, *p_quad)
    RSS_quad = np.sum((phi - phi_quad)**2)
    AIC_quad = AIC_from_RSS(RSS_quad, 4, N)

    # piecewise model
    p_piece, _ = curve_fit(model_piecewise, xvars, phi, p0=[90, 0, 0, 90, 0, 0])
    phi_piece = model_piecewise(xvars, *p_piece)
    RSS_piece = np.sum((phi - phi_piece)**2)
    AIC_piece = AIC_from_RSS(RSS_piece, 6, N)

    print("------------------------------------------------------------")
    print("MODEL 1: linear")
    print(f"a0={p_lin[0]:.3f}, a1={p_lin[1]:.5f}, a2={p_lin[2]:.5f}")
    print(f"RSS={RSS_lin:.4f}, AIC={AIC_lin:.4f}")

    print("------------------------------------------------------------")
    print("MODEL 2: quadratic")
    print(f"b0={p_quad[0]:.3f}, b1={p_quad[1]:.5f}, b2={p_quad[2]:.5f}, b3={p_quad[3]:.5f}")
    print(f"RSS={RSS_quad:.4f}, AIC={AIC_quad:.4f}")

    print("------------------------------------------------------------")
    print("MODEL 3: piecewise (z<0.35, z≥0.35)")
    print(f"c0={p_piece[0]:.3f}, c1={p_piece[1]:.5f}, c2={p_piece[2]:.5f}")
    print(f"d0={p_piece[3]:.3f}, d1={p_piece[4]:.5f}, d2={p_piece[5]:.5f}")
    print(f"RSS={RSS_piece:.4f}, AIC={AIC_piece:.4f}")

    # ------------------------------------------------------------
    # Monte Carlo null
    # ------------------------------------------------------------
    print("------------------------------------------------------------")
    print("running Monte Carlo null (shuffle z)...")
    M = 5000  # number of MC realisations
    dAIC_lin_quad  = []
    dAIC_lin_piece = []

    for _ in range(M):
        z_shuffled = np.random.permutation(z)
        xv = (theta, z_shuffled)

        # fit models to shuffled z
        pl, _ = curve_fit(model_lin, xv, phi, p0=[90,0,0])
        pql, _ = curve_fit(model_quad, xv, phi, p0=[90,0,0,0])
        ppc, _ = curve_fit(model_piecewise, xv, phi, p0=[90,0,0,90,0,0])

        RSS_l = np.sum((phi - model_lin(xv,*pl))**2)
        RSS_q = np.sum((phi - model_quad(xv,*pql))**2)
        RSS_p = np.sum((phi - model_piecewise(xv,*ppc))**2)

        AIC_l = AIC_from_RSS(RSS_l,3,N)
        AIC_q = AIC_from_RSS(RSS_q,4,N)
        AIC_p = AIC_from_RSS(RSS_p,6,N)

        dAIC_lin_quad.append(AIC_lin - AIC_quad)
        dAIC_lin_piece.append(AIC_lin - AIC_piece)

    dAIC_lin_quad  = np.array(dAIC_lin_quad)
    dAIC_lin_piece = np.array(dAIC_lin_piece)

    p_quad  = np.mean(dAIC_lin_quad  <= (AIC_lin - AIC_quad))
    p_piece = np.mean(dAIC_lin_piece <= (AIC_lin - AIC_piece))

    print("------------------------------------------------------------")
    print("MONTE CARLO RESULTS:")
    print(f"ΔAIC(real linear - quadratic) = {AIC_lin - AIC_quad:.4f}")
    print(f"   MC p-value = {p_quad:.6f}")
    print("------------------------------------------------------------")
    print(f"ΔAIC(real linear - piecewise) = {AIC_lin - AIC_piece:.4f}")
    print(f"   MC p-value = {p_piece:.6f}")

    print("="*70)
    print("test 33 complete.")
    print("="*70)


if __name__ == "__main__":
    main()
