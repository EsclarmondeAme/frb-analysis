#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRB Remnant-Time Harmonic-Energy Test (77B — SGB mask 20°)
Supergalactic-masked version of Test 77.
Measures harmonic energy contrast between R>0 and R<0 hemispheres.
"""

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------
# utilities
# ---------------------------------------------------------

def ang2xyz(ra, dec):
    ra = np.radians(ra)
    dec = np.radians(dec)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])

def load_csv(path):
    df = pd.read_csv(path)
    return (df["ra"].values,
            df["dec"].values,
            df["theta_unified"].values,
            df["phi_unified"].values)

# fully vectorized supergalactic transform
def eq_to_sg(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    # supergalactic north pole (standard)
    ra_gp  = np.radians(283.0)
    dec_gp = np.radians(15.0)

    sinSGB = (
        np.sin(dec)*np.sin(dec_gp)
        + np.cos(dec)*np.cos(dec_gp)*np.cos(ra - ra_gp)
    )

    SGB = np.degrees(np.arcsin(sinSGB))
    return SGB

# ---------------------------------------------------------
# test-specific
# ---------------------------------------------------------

def harmonic_energy(theta):
    return np.sum(np.cos(np.radians(theta))**2)

def compute_real(theta, Rsign):
    theta_pos = theta[Rsign > 0]
    theta_neg = theta[Rsign < 0]
    return harmonic_energy(theta_pos) - harmonic_energy(theta_neg)

def random_axis():
    phi = np.random.uniform(0, 2*np.pi)
    z = np.random.uniform(-1,1)
    r = np.sqrt(1 - z*z)
    return np.array([r*np.cos(phi), r*np.sin(phi), z])

def project_sign(X, axis_xyz):
    return X @ axis_xyz

# ---------------------------------------------------------
# main
# ---------------------------------------------------------

def main(path):
    print("================================================")
    print("FRB REMNANT-TIME HARMONIC-ENERGY TEST (77B)")
    print("Supergalactic mask: |SGB| >= 20°")
    print("================================================")

    RA, Dec, Th, Ph = load_csv(path)
    print(f"[info] original N={len(RA)}")

    # SGB mask
    SGB = eq_to_sg(RA, Dec)
    m = np.abs(SGB) >= 20.0
    RA = RA[m]; Dec = Dec[m]; Th = Th[m]; Ph = Ph[m]
    print(f"[info] after SGB mask N={len(RA)}")

    # FRB xyz
    X = ang2xyz(RA, Dec)

    # unified axis (use field values)
    a = np.radians(Ph)
    b = np.radians(90 - Th)
    A = np.column_stack([
        np.cos(b)*np.cos(a),
        np.cos(b)*np.sin(a),
        np.sin(b)
    ])
    axis_unified = A.mean(axis=0)
    axis_unified /= np.linalg.norm(axis_unified)

    # real value
    Rsign = project_sign(X, axis_unified)
    E_real = compute_real(Th, Rsign)

    # MC null
    n_mc = 2000
    E_null = []
    for _ in tqdm(range(n_mc), desc="MC", ncols=80):
        v = random_axis()
        Rr = project_sign(X, v)
        E_null.append(compute_real(Th, Rr))
    E_null = np.array(E_null)

    meanN = E_null.mean()
    stdN  = E_null.std() if E_null.std()>0 else 1e-12
    p = np.mean(np.abs(E_null - meanN) >= np.abs(E_real - meanN))

    print("------------------------------------------------")
    print(f"E_real        = {E_real}")
    print("------------------------------------------------")
    print(f"null mean E   = {meanN}")
    print(f"null std E    = {stdN}")
    print(f"p-value       = {p}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> harmonic energy differs between hemispheres")
    print("            even after removing SGB plane (robust)")
    print("  high p -> symmetric; consistent with isotropy")
    print("================================================")
    print("test 77B complete.")
    print("================================================")


if __name__ == "__main__":
    main(sys.argv[1])
