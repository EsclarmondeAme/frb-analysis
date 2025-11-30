#!/usr/bin/env python3
"""
FRB RANDOM–AXIS HELICITY CONTRAST TEST (TEST 37)

purpose
-------
Test whether the observed FRB helicity (pitch k) is specific to the
true unified axis or whether similar helicity could appear for random axes.

If helicity is cosmological and tied to the unified axis:
    |k_true| >> |k_random| for almost all random axes.
    p-value << 0.01

If helicity is spurious, coordinate-aligned, or due to noise:
    |k_true| ~ typical |k_random|
    p-value ~ 0.5

This is a decisive axis-specificity test.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import circmean
from scipy.optimize import curve_fit

# -------------------------------
# helical model
# -------------------------------
def helix(theta, phi0, k):
    return phi0 + k*theta

# -------------------------------------
# extract phi_max(theta) ridge
# -------------------------------------
def estimate_phi_peaks(theta_deg, phi_deg, bin_width=10.0, min_per_bin=6):
    theta = np.asarray(theta_deg)
    phi = np.radians(phi_deg)

    if len(theta) < min_per_bin*3:
        return None, None

    bins = np.arange(theta.min(), theta.max()+bin_width, bin_width)
    centers = 0.5*(bins[:-1] + bins[1:])
    peaks = []

    for i in range(len(bins)-1):
        mask = (theta >= bins[i]) & (theta < bins[i+1])
        vals = phi[mask]
        if len(vals) < min_per_bin:
            peaks.append(np.nan)
        else:
            peaks.append(circmean(vals, high=np.pi, low=-np.pi))

    return centers, np.array(peaks)

# -------------------------------------
# compute helical pitch for a dataset
# -------------------------------------
def compute_pitch(theta_deg, phi_deg):
    cen, peak = estimate_phi_peaks(theta_deg, phi_deg)
    if cen is None:
        return None

    mask = ~np.isnan(peak)
    if mask.sum() < 3:
        return None

    try:
        popt, _ = curve_fit(helix,
                            cen[mask],
                            np.degrees(peak[mask]),
                            p0=[0,0])
        phi0_fit, k_fit = popt
        return k_fit
    except:
        return None

# -------------------------------------
# random axis generator
# -------------------------------------
def random_axis():
    """uniform random direction on sphere"""
    u = np.random.rand()
    v = np.random.rand()
    theta = np.arccos(1 - 2*u)    # polar
    phi = 2*np.pi*v               # azimuth
    return theta, phi

# -------------------------------------
# rotate FRBs into new axis frame
# -------------------------------------
def rotate_to_axis(theta, phi, theta0, phi0):
    """
    Rotate points so that axis (theta0, phi0) becomes the new north pole.
    """
    # convert to Cartesian
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)

    # target axis vector
    x0 = np.sin(theta0)*np.cos(phi0)
    y0 = np.sin(theta0)*np.sin(phi0)
    z0 = np.cos(theta0)

    axis = np.array([x0, y0, z0])
    axis /= np.linalg.norm(axis)

    # rotate each vector by Rodrigues formula to align axis with z
    k = np.cross(axis, [0,0,1])
    s = np.linalg.norm(k)
    c = np.dot(axis, [0,0,1])

    if s < 1e-12:  # already aligned
        return theta, phi

    k /= s
    K = np.array([[    0, -k[2],  k[1]],
                  [ k[2],     0, -k[0]],
                  [-k[1],  k[0],    0]])

    R = np.eye(3)*c + (1-c)*np.outer(k,k) + s*K

    vec = np.vstack([x,y,z])
    vec2 = R @ vec

    x2, y2, z2 = vec2[0], vec2[1], vec2[2]

    theta2 = np.arccos(np.clip(z2, -1, 1))
    phi2 = np.arctan2(y2, x2)

    return theta2, phi2

# ================================================================
# main
# ================================================================
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_random_axis_helicity_contrast_test37.py frbs_unified.csv")
        return

    df = pd.read_csv(sys.argv[1])

    if "theta_unified" not in df.columns or "phi_unified" not in df.columns:
        raise ValueError("FRB catalog must include theta_unified and phi_unified")

    theta = np.radians(df["theta_unified"].values)
    phi   = np.radians(df["phi_unified"].values)

    print("=====================================================================")
    print(" FRB RANDOM–AXIS HELICITY CONTRAST TEST (TEST 37)")
    print("=====================================================================")

    # --------------------------
    # compute true pitch
    # --------------------------
    k_true = compute_pitch(df["theta_unified"], df["phi_unified"])
    print(f"true pitch k_true = {k_true:.5f} deg/deg")

    # --------------------------
    # Monte Carlo: random axes
    # --------------------------
    N_MC = 3000
    k_rand = []

    for _ in range(N_MC):
        th0, ph0 = random_axis()
        th_new, ph_new = rotate_to_axis(theta, phi, th0, ph0)
        k = compute_pitch(np.degrees(th_new), np.degrees(ph_new))
        if k is not None:
            k_rand.append(abs(k))

    k_rand = np.array(k_rand)
    p = np.mean(k_rand >= abs(k_true))

    print("---------------------------------------------------------------------")
    print(f"random-axis |k| mean = {k_rand.mean():.5f}")
    print(f"random-axis |k| std  = {k_rand.std():.5f}")
    print(f"p-value(|k_rand| >= |k_true|) = {p:.6f}")
    print("---------------------------------------------------------------------")

    print("interpretation:")
    if p < 0.01:
        print(" - helicity is strongly tied to the unified axis (cosmological).")
    elif p < 0.1:
        print(" - helicity is moderately axis-specific.")
    else:
        print(" - helicity appears even for random axes -> suspicious (unlikely).")

    print("=====================================================================")
    print(" test 37 complete.")
    print("=====================================================================")


if __name__ == "__main__":
    main()
