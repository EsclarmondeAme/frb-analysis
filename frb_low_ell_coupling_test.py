#!/usr/bin/env python
"""
FRB LOW-ℓ MULTIPOLE COUPLING / AXIS-ALIGNMENT TEST (TEST 10)

Purpose
-------
Quantify how strongly the quadrupole (ℓ=2) and octupole (ℓ=3) of the FRB sky
are aligned with each other and with the unified axis.

Inputs
------
- CSV file with unified-axis coordinates:
    theta_unified [deg]  : polar angle from unified axis
    phi_unified   [deg]  : azimuth about unified axis

Default input: frbs_unified.csv (from frb_make_unified_axis_frame.py)

Outputs
-------
- Printed summary:
    * best-fit ℓ=2 and ℓ=3 axes (θ, φ)
    * angular separation between ℓ=2 and ℓ=3 axes
    * angular separation of each from the unified axis (the pole)
    * Monte Carlo p-values for:
        - quadrupole–octupole alignment
        - joint alignment with unified axis
- no plots (this is a numerical cosmology test)
"""

import sys
import argparse
import numpy as np
import pandas as pd

from scipy.special import sph_harm


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def load_frb_unified(csv_path: str) -> pd.DataFrame:
    """Load unified-axis FRB catalogue and check required columns."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("ERROR: could not load CSV:", csv_path)
        print(e)
        sys.exit(1)

    required = ["theta_unified", "phi_unified"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("ERROR: missing required columns in CSV:", ",".join(missing))
        print("hint: run frb_make_unified_axis_frame.py first to create frbs_unified.csv")
        sys.exit(1)

    df = df.dropna(subset=required)
    return df


def compute_alm(theta: np.ndarray, phi: np.ndarray, ell: int):
    """
    Compute a_{ell m} for a discrete set of points on the sphere.

    Parameters
    ----------
    theta, phi : 1D arrays (radians)
        Polar and azimuthal angles of the FRBs.
    ell : int
        Multipole order.

    Returns
    -------
    m_vals : 1D array of ints
    alm    : 1D complex array of length (2ℓ+1)
    """
    m_vals = np.arange(-ell, ell + 1)
    alm = np.zeros_like(m_vals, dtype=complex)

    for i, m in enumerate(m_vals):
        Y = sph_harm(m, ell, phi, theta)  # note: sph_harm(m, l, phi, theta)
        # simple sum over events (no weighting)
        alm[i] = np.sum(np.conj(Y))

    return m_vals, alm


def multipole_axis_from_alm(ell: int,
                            m_vals: np.ndarray,
                            alm: np.ndarray,
                            n_theta: int = 45,
                            n_phi: int = 90):
    """
    Find the preferred axis of a single ℓ multipole by scanning a coarse grid.

    We reconstruct the field:
        f(θ, φ) = sum_m a_{ℓm} Y_{ℓm}(θ, φ)
    and take the direction of maximum |f| as the multipole axis.

    Parameters
    ----------
    ell : int
    m_vals : 1D array
    alm : 1D array (complex)
    n_theta, n_phi : grid resolution for (θ, φ)

    Returns
    -------
    theta_star, phi_star : floats (radians)
    n_hat : 3D unit vector (x, y, z)
    """
    theta_grid = np.linspace(0.0, np.pi, n_theta)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    TH, PH = np.meshgrid(theta_grid, phi_grid, indexing="ij")

    f = np.zeros_like(TH, dtype=complex)

    for m, a in zip(m_vals, alm):
        Y = sph_harm(m, ell, PH, TH)
        f += a * Y

    amp = np.abs(f)
    idx = np.argmax(amp)
    i, j = np.unravel_index(idx, amp.shape)

    theta_star = TH[i, j]
    phi_star = PH[i, j]

    # convert to 3D unit vector
    sin_t = np.sin(theta_star)
    n_hat = np.array([
        sin_t * np.cos(phi_star),
        sin_t * np.sin(phi_star),
        np.cos(theta_star),
    ])

    return theta_star, phi_star, n_hat


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the angle in degrees between two 3D unit vectors."""
    c = float(np.dot(v1, v2))
    c = max(-1.0, min(1.0, c))
    return np.degrees(np.arccos(c))


# ----------------------------------------------------------------------
# main test logic
# ----------------------------------------------------------------------

def run_low_ell_coupling_test(csv_path: str,
                              n_sims: int = 2000,
                              seed: int | None = 12345):
    print("=======================================================================")
    print("       FRB LOW-ℓ MULTIPOLE COUPLING / AXIS-ALIGNMENT TEST (TEST 10)   ")
    print("=======================================================================")
    print(f"loading unified-axis catalogue: {csv_path}")

    frb = load_frb_unified(csv_path)
    n = len(frb)
    print(f"loaded {n} FRBs with theta_unified, phi_unified")
    print("-----------------------------------------------------------------------")

    theta = np.radians(frb["theta_unified"].values)
    phi = np.radians(frb["phi_unified"].values)

    # observed multipoles ℓ = 2, 3
    print("computing observed ℓ=2 and ℓ=3 multipoles...")
    m2, alm2 = compute_alm(theta, phi, ell=2)
    m3, alm3 = compute_alm(theta, phi, ell=3)

    print("finding preferred axes on the sphere (coarse grid scan)...")
    th2, ph2, n2 = multipole_axis_from_alm(2, m2, alm2, n_theta=45, n_phi=90)
    th3, ph3, n3 = multipole_axis_from_alm(3, m3, alm3, n_theta=45, n_phi=90)

    # unified axis is the pole in this frame
    n_unified = np.array([0.0, 0.0, 1.0])

    # angles
    ang_23 = angle_between(n2, n3)
    ang_2u = angle_between(n2, n_unified)
    ang_3u = angle_between(n3, n_unified)
    best_axis_angle = min(ang_2u, ang_3u)

    print("-----------------------------------------------------------------------")
    print("observed axes (in unified-axis coordinates):")
    print(f"ℓ=2 axis: theta = {np.degrees(th2):6.2f} deg, phi = {np.degrees(ph2):6.2f} deg")
    print(f"ℓ=3 axis: theta = {np.degrees(th3):6.2f} deg, phi = {np.degrees(ph3):6.2f} deg")
    print("-----------------------------------------------------------------------")
    print("observed angular separations:")
    print(f"angle(ℓ2, ℓ3)          = {ang_23:6.3f} deg")
    print(f"angle(ℓ2, unified axis) = {ang_2u:6.3f} deg")
    print(f"angle(ℓ3, unified axis) = {ang_3u:6.3f} deg")
    print(f"min(angle(ℓ2,axis), angle(ℓ3,axis)) = {best_axis_angle:6.3f} deg")
    print("-----------------------------------------------------------------------")

    # Monte Carlo isotropic null
    print("running Monte Carlo isotropic null for axis coupling...")
    print(f"number of simulations: {n_sims}")
    if seed is not None:
        np.random.seed(seed)

    ang_23_sims = np.zeros(n_sims)
    best_axis_sims = np.zeros(n_sims)

    for i in range(n_sims):
        # isotropic directions on the sphere
        u = np.random.uniform(-1.0, 1.0, size=n)
        th_sim = np.arccos(u)
        ph_sim = np.random.uniform(0.0, 2.0 * np.pi, size=n)

        m2_s, alm2_s = compute_alm(th_sim, ph_sim, ell=2)
        m3_s, alm3_s = compute_alm(th_sim, ph_sim, ell=3)

        _, _, n2_s = multipole_axis_from_alm(2, m2_s, alm2_s,
                                             n_theta=30, n_phi=60)
        _, _, n3_s = multipole_axis_from_alm(3, m3_s, alm3_s,
                                             n_theta=30, n_phi=60)

        ang_23_sims[i] = angle_between(n2_s, n3_s)
        ang_2u_s = angle_between(n2_s, n_unified)
        ang_3u_s = angle_between(n3_s, n_unified)
        best_axis_sims[i] = min(ang_2u_s, ang_3u_s)

        # light progress ping
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{n_sims} simulations done", end="\r")

    print("\n-----------------------------------------------------------------------")
    # smaller angle = stronger alignment, so p = fraction with angle <= observed
    p_23 = np.mean(ang_23_sims <= ang_23)
    p_axis = np.mean(best_axis_sims <= best_axis_angle)

    print("Monte Carlo results (isotropic null):")
    print(f"mean angle(ℓ2, ℓ3) in null          = {np.mean(ang_23_sims):6.3f} deg")
    print(f"mean min(axis alignment) in null     = {np.mean(best_axis_sims):6.3f} deg")
    print("-----------------------------------------------------------------------")
    print("Monte Carlo p-values (alignment at least as strong as observed):")
    print(f"p_coupling = P[ angle(ℓ2,ℓ3) <= {ang_23:6.3f} deg ] = {p_23:8.5f}")
    print(f"p_axis     = P[ min(axis angles) <= {best_axis_angle:6.3f} deg ] = {p_axis:8.5f}")
    print("-----------------------------------------------------------------------")
    print("scientific interpretation:")
    print("  - p_coupling quantifies how unusually aligned the quadrupole and octupole")
    print("    are under an isotropic sky.")
    print("  - p_axis quantifies how unusually close at least one low-ℓ multipole axis")
    print("    lies to the unified FRB axis.")
    print("  - small p-values would support a coherent low-ℓ anisotropy field.")
    print("=======================================================================")
    print("Test 10 complete.")
    print("=======================================================================")


# ----------------------------------------------------------------------
# cli
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="FRB low-ℓ multipole coupling / axis-alignment test (Test 10)"
    )
    p.add_argument(
        "csv",
        nargs="?",
        default="frbs_unified.csv",
        help="input CSV file with theta_unified, phi_unified (default: frbs_unified.csv)",
    )
    p.add_argument(
        "--nsims",
        type=int,
        default=2000,
        help="number of Monte Carlo isotropic simulations (default: 2000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="random seed for Monte Carlo (default: 12345)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_low_ell_coupling_test(args.csv, n_sims=args.nsims, seed=args.seed)
