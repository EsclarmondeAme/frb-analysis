#!/usr/bin/env python
"""
frb_multipole_cross_correlation_test.py

TEST 11: full low-ell multipole cross-correlation between low-z and high-z FRBs
in the unified-axis frame.

This script:
- loads FRB positions (theta_unified, phi_unified) and redshift estimates z_est
- splits the sample into low-z and high-z halves at the median z_est
- computes spherical-harmonic coefficients a_{lm} for each subset up to ell_max
- computes cross-power spectra and correlation coefficients between low-z and high-z
- uses Monte Carlo random partitions to estimate p-values for the observed coherence
- saves a diagnostic plot of r_ell vs ell, with null bands

Run:
    python frb_multipole_cross_correlation_test.py [frbs_unified.csv]

Requirements:
    numpy, pandas, scipy, matplotlib, tqdm
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import sph_harm
from tqdm import tqdm


def load_frb_catalog(path: str) -> pd.DataFrame:
    """Load the FRB unified-axis catalogue and validate required columns."""
    try:
        frb = pd.read_csv(path)
    except Exception as e:
        print("ERROR: could not load FRB catalogue:", e)
        sys.exit(1)

    required = ["theta_unified", "phi_unified", "z_est"]
    missing = [c for c in required if c not in frb.columns]
    if missing:
        print("ERROR: missing required columns:", missing)
        print("       make sure you ran frb_make_unified_axis_frame.py")
        sys.exit(1)

    frb = frb.dropna(subset=required).copy()
    if len(frb) == 0:
        print("ERROR: no FRBs with valid theta_unified, phi_unified, z_est.")
        sys.exit(1)

    return frb


def compute_alm(theta, phi, ell_max=8):
    """
    Compute spherical-harmonic coefficients a_{lm} for a set of points.

    Parameters
    ----------
    theta : array-like
        Colatitudes in radians (0 at north pole, pi at south).
    phi : array-like
        Longitudes in radians.
    ell_max : int
        Maximum multipole order.

    Returns
    -------
    alm : dict[(ell, m)] -> complex
        Spherical harmonic coefficients.
    Cl : dict[ell] -> float
        Power spectrum C_ell = (1/(2ell+1)) sum_m |a_{lm}|^2.
    """
    n = len(theta)
    alm = {}
    Cl = {}

    for ell in range(1, ell_max + 1):
        # accumulate a_lm = sum_i Y_lm*(theta_i, phi_i)
        a_l = []
        for m in range(-ell, ell + 1):
            Y = sph_harm(m, ell, phi, theta)  # note: sph_harm(m, l, phi, theta)
            a_lm = np.sum(np.conjugate(Y))
            alm[(ell, m)] = a_lm
            a_l.append(a_lm)
        a_l = np.array(a_l)
        Cl[ell] = (np.abs(a_l) ** 2).mean()

    return alm, Cl


def compute_cross_correlation(alm_low, alm_high, Cl_low, Cl_high, ell_max=8):
    """
    Compute cross-power and correlation coefficient r_ell between two alm sets.

    Parameters
    ----------
    alm_low, alm_high : dict[(ell,m)] -> complex
        Harmonic coefficients for low-z and high-z samples.
    Cl_low, Cl_high : dict[ell] -> float
        Power spectra for each subset.
    ell_max : int

    Returns
    -------
    Ccross : dict[ell] -> float
        Cross-power spectra.
    r_ell : dict[ell] -> float
        Normalised correlation coefficients.
    """
    Ccross = {}
    r_ell = {}

    for ell in range(1, ell_max + 1):
        a_low = np.array([alm_low[(ell, m)] for m in range(-ell, ell + 1)])
        a_high = np.array([alm_high[(ell, m)] for m in range(-ell, ell + 1)])

        # cross-power: average of Re(a_low * conj(a_high))
        C = np.real(a_low * np.conjugate(a_high)).mean()
        Ccross[ell] = C

        denom = np.sqrt(Cl_low[ell] * Cl_high[ell]) if Cl_low[ell] > 0 and Cl_high[ell] > 0 else 0.0
        r_ell[ell] = C / denom if denom > 0 else 0.0

    return Ccross, r_ell


def monte_carlo_cross_correlation(theta, phi, z, ell_max=8, n_sims=2000, random_state=42):
    """
    Monte Carlo null for multipole cross-correlation by random partition of FRBs.

    At each simulation:
        - randomly permute indices
        - split into low/high groups of same sizes as real low/high
        - compute r_ell between the two random groups

    Returns
    -------
    r_null : np.ndarray, shape (n_sims, ell_max)
        Simulated r_ell values under the null hypothesis of no z-dependence.
    """
    rng = np.random.default_rng(random_state)
    n = len(theta)

    # define real split sizes
    z_sorted = np.sort(z)
    z_med = np.median(z_sorted)
    n_low = np.sum(z <= z_med)
    n_high = n - n_low

    if n_low == 0 or n_high == 0:
        print("ERROR: degenerate low/high-z split in Monte Carlo.")
        sys.exit(1)

    r_null = np.zeros((n_sims, ell_max), dtype=float)

    idx_all = np.arange(n)

    for i in tqdm(range(n_sims), desc="Monte Carlo random partitions"):
        rng.shuffle(idx_all)
        idx_low = idx_all[:n_low]
        idx_high = idx_all[n_low:]

        th_low, ph_low = theta[idx_low], phi[idx_low]
        th_high, ph_high = theta[idx_high], phi[idx_high]

        alm_low, Cl_low = compute_alm(th_low, ph_low, ell_max=ell_max)
        alm_high, Cl_high = compute_alm(th_high, ph_high, ell_max=ell_max)
        _, r_ell = compute_cross_correlation(alm_low, alm_high, Cl_low, Cl_high, ell_max=ell_max)

        for ell in range(1, ell_max + 1):
            r_null[i, ell - 1] = r_ell[ell]

    return r_null


def main():
    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "frbs_unified.csv"

    ell_max = 8
    n_sims = 2000

    print("=" * 70)
    print("FRB MULTIPOLE CROSS-CORRELATION TEST (TEST 11)")
    print("=" * 70)
    print(f"loading unified-axis catalogue: {path}")

    frb = load_frb_catalog(path)
    print(f"loaded {len(frb)} FRBs with theta_unified, phi_unified, z_est")

    # convert to radians; theta_unified is already an angular distance from axis (0..pi)
    theta = np.radians(frb["theta_unified"].values)
    phi = np.radians(frb["phi_unified"].values)
    z = frb["z_est"].values

    # ------------------------------------------------------------------
    # real low-z / high-z split
    # ------------------------------------------------------------------
    z_med = np.median(z)
    mask_low = z <= z_med
    mask_high = ~mask_low

    theta_low, phi_low = theta[mask_low], phi[mask_low]
    theta_high, phi_high = theta[mask_high], phi[mask_high]

    print("------------------------------------------------------------------")
    print("redshift split:")
    print(f"  median z_est = {z_med:.3f}")
    print(f"  low-z  : N = {theta_low.size}")
    print(f"  high-z : N = {theta_high.size}")
    print("------------------------------------------------------------------")
    print(f"computing a_lm up to ell_max = {ell_max} for low-z and high-z subsets...")

    alm_low, Cl_low = compute_alm(theta_low, phi_low, ell_max=ell_max)
    alm_high, Cl_high = compute_alm(theta_high, phi_high, ell_max=ell_max)

    Ccross, r_ell = compute_cross_correlation(alm_low, alm_high, Cl_low, Cl_high, ell_max=ell_max)

    print("------------------------------------------------------------------")
    print("observed multipole cross-correlation:")
    for ell in range(1, ell_max + 1):
        print(
            f"  ell = {ell:2d}  "
            f"C_cross = {Ccross[ell]: .3e}  "
            f"r_ell = {r_ell[ell]: .3f}"
        )

    # combined statistic (mean absolute correlation across ell)
    r_vals = np.array([r_ell[ell] for ell in range(1, ell_max + 1)])
    T_obs = np.mean(np.abs(r_vals))

    print("------------------------------------------------------------------")
    print(f"combined coherence statistic T_obs = mean_ell |r_ell| = {T_obs:.3f}")

    # ------------------------------------------------------------------
    # Monte Carlo null: random partitions
    # ------------------------------------------------------------------
    print("------------------------------------------------------------------")
    print(f"running Monte Carlo null with {n_sims} random partitions...")
    r_null = monte_carlo_cross_correlation(theta, phi, z, ell_max=ell_max, n_sims=n_sims)

    # per-ell p-values
    p_ell = {}
    for ell in range(1, ell_max + 1):
        obs = r_ell[ell]
        sims = r_null[:, ell - 1]
        p = np.mean(np.abs(sims) >= np.abs(obs))
        p_ell[ell] = p

    # combined T statistic null
    T_null = np.mean(np.abs(r_null), axis=1)
    p_T = np.mean(T_null >= T_obs)

    print("------------------------------------------------------------------")
    print("Monte Carlo p-values (null: no redshift dependence of multipoles):")
    for ell in range(1, ell_max + 1):
        print(f"  ell = {ell:2d}  p(|r_ell,null| >= |r_ell,obs|) = {p_ell[ell]:.5f}")
    print("------------------------------------------------------------------")
    print(f"combined coherence p-value (T_null >= T_obs): p_T = {p_T:.5f}")
    print("------------------------------------------------------------------")
    print("scientific interpretation:")
    print("  - r_ell measures how similar the low-z and high-z multipole patterns are.")
    print("  - small per-ell p_ell indicate strong coherence at that multipole.")
    print("  - small p_T indicates a globally coherent low-ell anisotropy field")
    print("    that does not evolve across the redshift split.")
    print("------------------------------------------------------------------")

    # ------------------------------------------------------------------
    # figure: r_ell vs ell with null bands
    # ------------------------------------------------------------------
    ell_vals = np.arange(1, ell_max + 1)
    r_obs = np.array([r_ell[ell] for ell in ell_vals])
    r_med = np.median(r_null, axis=0)
    r_lo = np.percentile(r_null, 16, axis=0)
    r_hi = np.percentile(r_null, 84, axis=0)

    plt.figure(figsize=(8, 5))
    plt.axhline(0.0, linestyle="--")
    plt.fill_between(ell_vals, r_lo, r_hi, alpha=0.3)
    plt.plot(ell_vals, r_med, marker="o", linestyle="--", label="null median")
    plt.plot(ell_vals, r_obs, marker="o", linestyle="-", label="observed")

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$r_\ell$ (low-z vs high-z)")
    plt.title("FRB multipole cross-correlation (Test 11)")
    plt.legend()
    plt.tight_layout()
    outname = "frb_multipole_cross_correlation.png"
    plt.savefig(outname, dpi=200)
    plt.close()

    print(f"saved figure: {outname}")
    print("==================================================================")
    print("Test 11 complete.")
    print("==================================================================")


if __name__ == "__main__":
    main()
