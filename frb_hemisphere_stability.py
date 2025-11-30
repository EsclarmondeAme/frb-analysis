#!/usr/bin/env python3
"""
FRB HEMISPHERE STABILITY TEST
Evaluates whether the unified axis is stable when removing entire hemispheres.
Produces: hemisphere axes, chi² values, width-layer ΔAIC, and a summary figure.
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

# ============================================================
# unified axis (from previous analysis)
# ============================================================
UNIFIED_AXIS_L = 159.85
UNIFIED_AXIS_B = -0.51

# ============================================================
# utilities
# ============================================================
def angdist(l1, b1, l2, b2):
    """
    angular distance between a single point (l1,b1) and arrays (l2,b2)
    all in degrees.
    """
    c1 = SkyCoord(l=l1*u.deg, b=b1*u.deg, frame='galactic')
    c2 = SkyCoord(l=np.asarray(l2, float)*u.deg,
                  b=np.asarray(b2, float)*u.deg,
                  frame='galactic')
    return c1.separation(c2).deg


def width_layer_aic(theta, width):
    """
    compute ΔAIC for width layering:
    compare linear model vs 3-layer model with fixed boundaries (15°, 30°)
    """
    # linear model
    X = np.vstack([np.ones_like(theta), theta]).T
    beta = np.linalg.lstsq(X, width, rcond=None)[0]
    pred_lin = X @ beta
    rss_lin = np.sum((width - pred_lin)**2)

    # 3-layer model
    cuts = [15, 30]
    layer_vals = []
    for i in range(3):
        if i == 0:
            mask = theta <= cuts[0]
        elif i == 1:
            mask = (theta > cuts[0]) & (theta <= cuts[1])
        else:
            mask = theta > cuts[1]

        if np.sum(mask) == 0:
            layer_vals.append(np.mean(width))
        else:
            layer_vals.append(np.mean(width[mask]))

    pred_layer = np.zeros_like(width)
    pred_layer[theta <= cuts[0]] = layer_vals[0]
    pred_layer[(theta > cuts[0]) & (theta <= cuts[1])] = layer_vals[1]
    pred_layer[theta > cuts[1]] = layer_vals[2]

    rss_layer = np.sum((width - pred_layer)**2)

    # AIC = n*ln(rss/n) + 2*k
    n = len(width)
    aic_lin = n*np.log(rss_lin/n) + 2*2      # 2 parameters
    aic_layer = n*np.log(rss_layer/n) + 2*5  # 5 parameters (3 means + 2 fixed boundaries)

    return aic_lin, aic_layer, aic_lin - aic_layer, rss_lin, rss_layer


# ============================================================
# main
# ============================================================
def main():

    print("="*70)
    print("FRB HEMISPHERE STABILITY TEST")
    print("="*70)

    df = pd.read_csv("frbs.csv")
    df["l"] = df["ra"]
    df["b"] = df["dec"]

    # compute angle relative to unified axis
    theta_unif = angdist(UNIFIED_AXIS_L, UNIFIED_AXIS_B,
                         df["l"].values, df["b"].values)

    # layer mask for chi²
    band1 = (theta_unif <= 10)
    band2 = (theta_unif > 10) & (theta_unif <= 25)
    band3 = (theta_unif > 25) & (theta_unif <= 40)

    # isotropic expectations
    N = len(df)
    exp1 = N * (1 - np.cos(np.deg2rad(10))) / 2
    exp2 = N * (np.cos(np.deg2rad(10)) - np.cos(np.deg2rad(25))) / 2
    exp3 = N * (np.cos(np.deg2rad(25)) - np.cos(np.deg2rad(40))) / 2

    regions = {
        "north": df[df["b"] >= 0],
        "south": df[df["b"] < 0],
        "east":  df[df["l"] <= 180],
        "west":  df[df["l"] > 180]
    }

    results = []

    for name, sub in regions.items():

        l = sub["l"].values
        b = sub["b"].values
        w = sub["width"].values

        if len(sub) < 10:
           # name, L, B, shift, chi2, dAIC
           results.append((name, np.nan, np.nan, np.nan, np.nan, np.nan))
           continue


        # compute axis through brute-force small grid (±30° around unified axis)
        scan_L = np.linspace(UNIFIED_AXIS_L-30, UNIFIED_AXIS_L+30, 61)
        scan_B = np.linspace(UNIFIED_AXIS_B-30, UNIFIED_AXIS_B+30, 61)

        best_chi2 = 1e99
        best_L = None
        best_B = None

        for L in scan_L:
            for B in scan_B:
                theta = angdist(L, B, l, b)
                n1 = np.sum(theta <= 10)
                n2 = np.sum((theta > 10) & (theta <= 25))
                n3 = np.sum((theta > 25) & (theta <= 40))
                chi2 = ((n1-exp1)**2/exp1 +
                        (n2-exp2)**2/exp2 +
                        (n3-exp3)**2/exp3)
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_L = L
                    best_B = B

        # width-layer ΔAIC
        theta_local = angdist(best_L, best_B, l, b)
        aic_lin, aic_layer, dAIC, rss_lin, rss_layer = width_layer_aic(theta_local, w)

        shift = angdist(UNIFIED_AXIS_L, UNIFIED_AXIS_B, best_L, best_B)

        results.append((name, best_L, best_B, shift, best_chi2, dAIC))

    # print results
    print("\n==========================================")
    print("HEMISPHERE AXIS RESULTS")
    print("==========================================")
    for r in results:
        name, L, B, shift, chi2, dAIC = r
        print(f"{name:>8}: axis=({L:.2f}°, {B:.2f}°), shift={shift:.2f}°, chi2={chi2:.2f}, ΔAIC={dAIC:.2f}")

    # figure
    fig = plt.figure(figsize=(8,5))
    plt.title("Hemisphere Axis Shifts")
    for r in results:
        name, L, B, shift, chi2, dAIC = r
        if not np.isnan(shift):
            plt.scatter(shift, chi2, label=name, s=80)
    plt.xlabel("Axis shift (deg)")
    plt.ylabel("χ² relative to unified axis")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hemisphere_stability.png", dpi=150)
    plt.close()

    print("\nfigure saved: hemisphere_stability.png")
    print("analysis complete.")
    print("="*70)


if __name__ == "__main__":
    main()
