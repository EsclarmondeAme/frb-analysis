#!/usr/bin/env python
"""
FRB LARGE-SCALE / SMALL-SCALE MODE-COUPLING TEST (TEST 20)

purpose
-------
this test asks whether small-scale frb clustering (local overdensities)
is *modulated* by the large-scale anisotropy field constructed from
low-ell spherical harmonics (ell <= 3) in the unified-axis frame.

intuition:
- low-ell field T_low(theta, phi) encodes the warped shell / global anisotropy.
- small-scale overdensity delta_i counts neighbours within a small cap
  (e.g. 10 degrees) around each frb.
- if there is "lensing-like" mode coupling, regions with high T_low
  should systematically host higher local densities delta_i.

we quantify this with a pearson correlation between T_low and delta,
and assess its significance via monte carlo random permutations that
destroy any coupling while preserving both marginal distributions.

input
-----
- frb catalogue with columns:
  - theta_unified [rad or deg? here: *degrees from axis*]
  - phi_unified   [degrees, azimuth around axis]

usage
-----
python frb_large_scale_small_scale_coupling_test.py frbs_unified.csv

output
------
- printed correlation statistics and monte carlo p-value
- optional figure can be added later if desired
"""

import sys
import numpy as np
import pandas as pd
from scipy.special import sph_harm
from scipy.stats import pearsonr


def load_frbs(path: str) -> pd.DataFrame:
    """load frb catalogue and check required columns."""
    frb = pd.read_csv(path)
    required = ["theta_unified", "phi_unified"]
    missing = [c for c in required if c not in frb.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    # drop rows with nan
    frb = frb.dropna(subset=required)
    return frb


def compute_low_ell_coeffs(theta: np.ndarray,
                           phi: np.ndarray,
                           ell_max: int = 3) -> dict:
    """
    compute low-ell spherical harmonic coefficients a_lm
    for ell = 1..ell_max using the frb angular positions.

    we treat each frb as a unit-weight delta function on the sphere:
        a_lm = sum_i Y_lm*(theta_i, phi_i)
    """
    coeffs = {}
    for ell in range(1, ell_max + 1):
        for m in range(-ell, ell + 1):
            Y = sph_harm(m, ell, phi, theta)  # note: sph_harm(m,l,phi,theta)
            a_lm = np.sum(np.conjugate(Y))
            coeffs[(ell, m)] = a_lm
    return coeffs


def evaluate_low_ell_field(theta: np.ndarray,
                           phi: np.ndarray,
                           coeffs: dict) -> np.ndarray:
    """
    evaluate the real part of the low-ell field:
        T_low(theta, phi) = sum_{ell,m} a_lm Y_lm(theta, phi)
    at the given positions.
    """
    T = np.zeros_like(theta, dtype=np.complex128)
    for (ell, m), a_lm in coeffs.items():
        Y = sph_harm(m, ell, phi, theta)
        T += a_lm * Y
    return np.real(T)


def angular_distance_matrix(theta: np.ndarray,
                            phi: np.ndarray) -> np.ndarray:
    """
    compute pairwise angular distances between all frbs in radians
    using unit vectors and dot products.
    """
    # unit vectors
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    vec = np.vstack((x, y, z)).T  # shape (N,3)

    # pairwise cosine of angle
    dot = np.clip(vec @ vec.T, -1.0, 1.0)
    ang = np.arccos(dot)
    return ang


def compute_small_scale_overdensity(theta: np.ndarray,
                                    phi: np.ndarray,
                                    cap_radius_deg: float = 10.0) -> np.ndarray:
    """
    for each frb, count neighbours within a cap of radius cap_radius_deg
    and convert to a fractional overdensity relative to an isotropic
    expectation with N-1 other points.

    delta_i = (n_i - n_exp) / n_exp

    this is a simple proxy for small-scale clustering at each position.
    """
    N = theta.size
    # pairwise angles
    ang = angular_distance_matrix(theta, phi)
    cap_radius = np.radians(cap_radius_deg)

    # neighbour counts including self; subtract 1 to remove self-count
    neighbours = (ang <= cap_radius).sum(axis=1) - 1

    # isotropic expectation for neighbour count in a cap:
    # area of cap = 2π (1 - cos r)
    area_cap = 2.0 * np.pi * (1.0 - np.cos(cap_radius))
    rho = (N - 1) / (4.0 * np.pi)
    n_exp = rho * area_cap

    delta = (neighbours - n_exp) / n_exp
    return delta


def monte_carlo_null(T_low: np.ndarray,
                     delta: np.ndarray,
                     n_sims: int = 1000,
                     random_state: int = 42) -> np.ndarray:
    """
    build a null distribution for |r| by randomly permuting delta
    with respect to T_low, preserving both marginal distributions
    but destroying any intrinsic coupling.
    """
    rng = np.random.default_rng(random_state)
    N = T_low.size
    r_null = np.zeros(n_sims, dtype=float)

    for k in range(n_sims):
        perm = rng.permutation(N)
        r, _ = pearsonr(T_low, delta[perm])
        r_null[k] = abs(r)

    return r_null


def main():
    if len(sys.argv) < 2:
        print("usage: python frb_large_scale_small_scale_coupling_test.py frbs_unified.csv")
        sys.exit(1)

    path = sys.argv[1]

    print("=" * 70)
    print("frb large-scale / small-scale mode-coupling test (test 20)")
    print("=" * 70)

    # load catalogue
    try:
        frb = load_frbs(path)
    except Exception as e:
        print(f"error: could not load catalogue '{path}': {e}")
        sys.exit(1)

    print(f"loaded {len(frb)} frbs from: {path}")
    print("using columns: theta_unified, phi_unified (degrees)\n")

    # convert to radians (theta_unified is angle from unified axis)
    theta = np.radians(frb["theta_unified"].values.astype(float))
    phi = np.radians(frb["phi_unified"].values.astype(float))

    # compute low-ell field (ell <= 3)
    print("computing low-ell (ell <= 3) spherical-harmonic field...")
    coeffs = compute_low_ell_coeffs(theta, phi, ell_max=3)
    T_low = evaluate_low_ell_field(theta, phi, coeffs)

    # small-scale overdensity
    print("computing small-scale overdensities (10-degree caps)...")
    delta = compute_small_scale_overdensity(theta, phi, cap_radius_deg=10.0)

    # correlation between large-scale field and local overdensity
    print("--------------------------------------------------------------")
    print("observed large-scale / small-scale coupling:")
    r_obs, p_obs = pearsonr(T_low, delta)
    print(f"pearson r(T_low, delta) = {r_obs: .4f}")
    print(f"two-sided p-value (analytic) = {p_obs: .4e}")

    # monte carlo null via permutations
    print("--------------------------------------------------------------")
    print("running monte carlo null (random permutations of delta)...")
    n_sims = 1000
    r_null = monte_carlo_null(T_low, delta, n_sims=n_sims, random_state=123)
    T_obs = abs(r_obs)
    p_mc = np.mean(r_null >= T_obs)

    print("--------------------------------------------------------------")
    print("monte carlo results (null: no coupling, fixed marginals):")
    print(f"T_obs = |r_obs| = {T_obs: .4f}")
    print(f"null mean |r|   = {np.mean(r_null): .4f}")
    print(f"null std |r|    = {np.std(r_null): .4f}")
    print(f"monte carlo p-value = {p_mc: .5f}")
    print("--------------------------------------------------------------")
    print("scientific interpretation:")
    print(" - T_low encodes the large-scale warped-shell anisotropy.")
    print(" - delta encodes small-scale clustering in 10-degree caps.")
    print(" - this test asks whether small-scale overdensities are")
    print("   modulated by the large-scale anisotropy field.")
    if p_mc < 0.01:
        print(" - result: strong evidence for large–small mode coupling;")
        print("   local clustering follows the global anisotropy pattern.")
    elif p_mc < 0.05:
        print(" - result: marginal evidence for coupling; hints that")
        print("   small-scale clustering is weakly modulated by the")
        print("   large-scale anisotropy.")
    else:
        print(" - result: the observed coupling strength is fully")
        print("   compatible with random expectations; given current")
        print("   frb statistics and redshift uncertainties, this test")
        print("   is neutral and does not add new tension or support")
        print("   relative to tests 1–19.")
    print("==============================================================")
    print("test 20 complete.")
    print("==============================================================")


if __name__ == "__main__":
    main()
