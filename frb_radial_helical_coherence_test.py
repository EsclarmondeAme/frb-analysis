#!/usr/bin/env python3
"""
FRB RADIAL–HELICAL COHERENCE TEST (TEST 31)

Purpose
-------
Quantify whether the *radial shells* of the FRB sky share a coherent azimuthal
phase structure, i.e. whether the preferred harmonic phases in different shells
line up in a physically meaningful way.

This test bridges:
- radial structure (θ–layers)
- helical / harmonic structure (φ–modes)
- redshift evolution (z-slices)

We measure:
    C1 = coherence of m = 1 phases across shells
    C2 = coherence of m = 2 phases across shells

and compare against Monte Carlo isotropic nulls.
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import circmean

# -----------------------------------------------
# PARAMETERS
# -----------------------------------------------
THETA_SHELLS = [
    (0, 20),
    (20, 40),
    (40, 60),
    (60, 90),
]

N_PHI_BINS = 24
N_MC = 20000

Z_SLICES = [
    (0.00, 0.20),
    (0.20, 0.35),
    (0.35, 0.55),
    (0.55, 0.80),
]

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
        raise ValueError("no redshift column found (z_est/z/redshift).")

    df = df.dropna(subset=["theta_unified", "phi_unified", "z"])
    return df

# ------------------------------------------------------------
# harmonic model
# ------------------------------------------------------------
def m1_model(phi, A0, A1, phi0):
    return A0 + A1 * np.cos(phi - phi0)

def m2_model(phi, A0, A2, phi0):
    return A0 + A2 * np.cos(2 * (phi - phi0))

# ------------------------------------------------------------
# fit harmonic phase for a shell
# ------------------------------------------------------------
def fit_harmonic_phase(phi_deg):
    phi = np.deg2rad(phi_deg)
    counts, edges = np.histogram(phi, bins=N_PHI_BINS, range=(-np.pi, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = counts
    x = centers
    if len(y) < 5:
        return None

    # m=1
    try:
        popt1, _ = curve_fit(
            m1_model,
            x,
            y,
            p0=[np.mean(y), 0.5*np.max(y), 0],
            maxfev=20000
        )
        A1 = popt1[1]
        phi1 = popt1[2]
    except:
        A1 = None
        phi1 = None

    # m=2
    try:
        popt2, _ = curve_fit(
            m2_model,
            x,
            y,
            p0=[np.mean(y), 0.5*np.max(y), 0],
            maxfev=20000
        )
        A2 = popt2[1]
        phi2 = popt2[2]
    except:
        A2 = None
        phi2 = None

    return dict(A1=A1, phi1=phi1, A2=A2, phi2=phi2)

# ------------------------------------------------------------
# coherence metrics
# ------------------------------------------------------------
def coherence(phases):
    """phases: list of phase angles (radians), ignoring None."""
    phases = [p for p in phases if p is not None]
    if len(phases) < 2:
        return None
    R = np.abs(np.mean(np.exp(1j * np.array(phases))))
    return float(R)

# ------------------------------------------------------------
# Monte Carlo null
# ------------------------------------------------------------
def mc_null(df_slice, theta_shells, n_mc=20000):
    rng = np.random.default_rng(123)
    C1_null = []
    C2_null = []

    phi = df_slice["phi_unified"].values
    theta = df_slice["theta_unified"].values

    N = len(phi)
    for _ in range(n_mc):
        perm = rng.permutation(N)
        phi_sh = phi[perm]

        # compute phases for shuffled catalog
        phases_m1 = []
        phases_m2 = []

        for tmin, tmax in theta_shells:
            mask = (theta >= tmin) & (theta < tmax)
            if mask.sum() < 10:
                phases_m1.append(None)
                phases_m2.append(None)
                continue
            fit = fit_harmonic_phase(phi_sh[mask] * 180/np.pi)
            if fit is None:
                phases_m1.append(None)
                phases_m2.append(None)
            else:
                phases_m1.append(fit["phi1"])
                phases_m2.append(fit["phi2"])

        C1_null.append(coherence(phases_m1))
        C2_null.append(coherence(phases_m2))

    C1_null = np.array([c for c in C1_null if c is not None])
    C2_null = np.array([c for c in C2_null if c is not None])

    return C1_null, C2_null

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("usage: python frb_radial_helical_coherence_test.py frbs_unified.csv")
        return

    df = load_frbs(sys.argv[1])
    print(f"loaded {len(df)} FRBs")

    results = []

    for (zmin, zmax) in Z_SLICES:
        df_slice = df[(df["z"] >= zmin) & (df["z"] < zmax)]
        print(f"\nprocessing z-slice {zmin}–{zmax}, n={len(df_slice)}")

        if len(df_slice) < 40:
            results.append(dict(
                zmin=zmin, zmax=zmax,
                C1=None, p1=None,
                C2=None, p2=None
            ))
            continue

        # phases per shell
        phases_m1 = []
        phases_m2 = []

        for tmin, tmax in THETA_SHELLS:
            mask = (df_slice["theta_unified"] >= tmin) & (df_slice["theta_unified"] < tmax)
            if mask.sum() < 10:
                phases_m1.append(None)
                phases_m2.append(None)
                continue

            fit = fit_harmonic_phase(df_slice["phi_unified"][mask])
            if fit is None:
                phases_m1.append(None)
                phases_m2.append(None)
            else:
                phases_m1.append(fit["phi1"])
                phases_m2.append(fit["phi2"])

        C1 = coherence(phases_m1)
        C2 = coherence(phases_m2)

        # Monte Carlo
        C1_null, C2_null = mc_null(df_slice, THETA_SHELLS, n_mc=N_MC)

        if C1 is None:
            p1 = None
        else:
            p1 = np.mean(C1_null >= C1)

        if C2 is None:
            p2 = None
        else:
            p2 = np.mean(C2_null >= C2)

        results.append(dict(
            zmin=zmin, zmax=zmax,
            C1=C1, p1=p1,
            C2=C2, p2=p2
        ))

    # ------------------------------------------------------------
    # PRINT SUMMARY
    # ------------------------------------------------------------
    print("======================================================================")
    print("SUMMARY – RADIAL–HELICAL COHERENCE")
    print("======================================================================")

    for r in results:
        C1s = "None" if r["C1"] is None else f"{r['C1']:.3f}"
        C2s = "None" if r["C2"] is None else f"{r['C2']:.3f}"
        p1s = "None" if r["p1"] is None else f"{r['p1']:.6f}"
        p2s = "None" if r["p2"] is None else f"{r['p2']:.6f}"

        print(f"z={r['zmin']:.2f}–{r['zmax']:.2f}:  "
              f"C1={C1s}  p1={p1s}   "
              f"C2={C2s}  p2={p2s}")

    print("======================================================================")
    print("test 31 complete.")
    print("======================================================================")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
