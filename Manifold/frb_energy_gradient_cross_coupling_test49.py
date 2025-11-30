#!/usr/bin/env python3
"""
FRB ENERGY-GRADIENT CROSS-COUPLING TEST (TEST 49)

Goal
----
We test the core Energy-Gradient Encoding (E-GR) idea that:
    "information is stored in the gradients between energy states, not in them."

We already saw in Test 48 that the grain intensity G correlates
strongly with individual energy-like quantities (DM, SNR, fluence,
width, redshift). Here we ask a sharper question:

    Do *pairs* of energy variables (X, Y) jointly encode the grain field
    via a genuine interaction term X*Y, beyond simple linear slopes?

If the universe is using energy gradients "between states" as an
encoding layer, we should see:

    - a nontrivial interaction coefficient for X*Y;
    - that interaction term being significantly stronger than expected
      under an isotropic / shuffled-null where spatial grain patterns
      are decoupled from the energy variables.

Method
------
1. Load FRB catalog with:
       theta_unified, phi_unified, and energy fields
       dm, snr, fluence, width, z_est

2. Construct a "grain intensity" G for each FRB from its local
   sky density:
       - embed (theta, phi) on the unit sphere
       - compute distances to k nearest neighbours (k=5)
       - define G = 1 / mean(kNN distance)

   This is a local-granularity field: high G = dense grain,
   low G = sparse grain.

3. For each ordered pair of energy variables (X, Y) in:
       dm, snr, fluence, width, z_est

   we standardize X, Y (zero mean, unit variance) and fit:

       G = b0 + b1 * X + b2 * Y + b3 * (X * Y) + noise

   Our cross-coupling statistic is:
       T_obs = |b3|   (magnitude of true interaction)

4. Monte Carlo isotropic null:
    - we keep (X, Y) fixed
    - we shuffle the grain field G across FRBs (breaking spatial link)
    - for each shuffle we refit the same model and collect |b3_null|
    - number of simulations: N_MC = 5000

   The p-value is:
       p = P(|b3_null| >= |b3_obs|)

   Small p indicates that the cross-energy interaction X*Y encodes
   the grain structure more strongly than expected by chance.

Output
------
For each pair (X, Y) we print:
    - observed interaction coefficient b3_obs
    - null mean |b3|
    - null std |b3|
    - Monte Carlo p-value

This test complements Test 48 by asking whether
"energy between states" (X*Y) carries additional information about
the grain field beyond simple linear gradients.
"""

import sys
import numpy as np
import pandas as pd

N_NEIGH = 5       # k for kNN density -> grain intensity
N_MC = 5000       # Monte Carlo shuffles for cross-coupling
RNG = np.random.default_rng(123)


# ------------------------------------------------------------
# utilities: load catalog, compute grain intensity
# ------------------------------------------------------------
def load_frbs(path: str) -> pd.DataFrame:
    """Load FRB catalog and check required columns."""
    df = pd.read_csv(path)

    required = [
        "theta_unified",
        "phi_unified",
        "dm",
        "snr",
        "fluence",
        "width",
        "z_est",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = df.dropna(subset=required)
    return df


def spherical_to_cartesian(theta_deg: np.ndarray,
                           phi_deg: np.ndarray) -> np.ndarray:
    """
    Convert (theta, phi) in degrees (polar distance from axis, azimuth)
    to 3D Cartesian coordinates on the unit sphere.

    theta: polar angle from unified axis (0..180 deg)
    phi:   azimuth (-180..180 deg or 0..360 deg)
    """
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    # standard physics convention:
    # theta: polar from +z, phi: azimuth
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.vstack([x, y, z]).T  # shape (N, 3)


def compute_grain_intensity(theta_deg: np.ndarray,
                            phi_deg: np.ndarray,
                            k: int = N_NEIGH) -> np.ndarray:
    """
    Compute local grain intensity G for each FRB based on k-nearest
    neighbour distances on the unit sphere.

    Steps:
        - embed positions on unit sphere
        - compute pairwise Euclidean distances
        - for each FRB, take mean distance to k nearest neighbours
          (excluding self)
        - define G = 1 / <d_k>

    Returns:
        G: array of shape (N,)
    """
    coords = spherical_to_cartesian(theta_deg, phi_deg)  # (N, 3)
    N = coords.shape[0]

    # pairwise distance matrix: |r_i - r_j|
    # shape (N, N)
    # (N=600 -> fine)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    # set diagonal to large number so it is not chosen as neighbour
    np.fill_diagonal(dist, np.inf)

    # kNN distances: use partition for efficiency
    k_eff = min(k, N - 1)
    knn_dists = np.partition(dist, k_eff - 1, axis=1)[:, :k_eff]
    mean_knn = np.mean(knn_dists, axis=1)

    # avoid division by zero
    eps = 1e-8
    G = 1.0 / (mean_knn + eps)
    return G


# ------------------------------------------------------------
# regression machinery for cross-coupling
# ------------------------------------------------------------
def fit_cross_coupling(G: np.ndarray,
                       X: np.ndarray,
                       Y: np.ndarray) -> float:
    """
    Fit linear model:

        G = b0 + b1*X + b2*Y + b3*(X*Y) + noise

    after standardizing X, Y.

    Returns:
        b3 (interaction coefficient).
    """
    Xs = (X - X.mean()) / (X.std() + 1e-12)
    Ys = (Y - Y.mean()) / (Y.std() + 1e-12)
    XY = Xs * Ys

    N = len(G)
    design = np.vstack([np.ones(N), Xs, Ys, XY]).T  # shape (N, 4)
    beta, *_ = np.linalg.lstsq(design, G, rcond=None)
    b3 = beta[3]
    return b3


def mc_null_b3(G: np.ndarray,
               X: np.ndarray,
               Y: np.ndarray,
               n_sims: int = N_MC) -> np.ndarray:
    """
    Monte Carlo null for the interaction coefficient b3:

    - Shuffle grain intensities G across FRBs (break spatial link)
    - Keep X, Y fixed
    - Refit model and record |b3|

    Returns:
        array of |b3_null| values, length n_sims.
    """
    N = len(G)
    b3_null = np.zeros(n_sims, dtype=float)

    for i in range(n_sims):
        G_shuff = G[RNG.permutation(N)]
        b3 = fit_cross_coupling(G_shuff, X, Y)
        b3_null[i] = abs(b3)

    return b3_null


# ------------------------------------------------------------
# main driver
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_energy_gradient_cross_coupling_test49.py frbs_unified.csv")
        sys.exit(1)

    path = sys.argv[1]
    df = load_frbs(path)
    N = len(df)

    print("loaded", N, "FRBs")

    theta = df["theta_unified"].values
    phi = df["phi_unified"].values

    print("computing grain intensities from local sky density ...")
    G = compute_grain_intensity(theta, phi, k=N_NEIGH)

    energy_fields = ["dm", "snr", "fluence", "width", "z_est"]
    print("using energy fields:", energy_fields)

    print("=" * 67)
    print("FRB ENERGY-GRADIENT CROSS-COUPLING TEST (TEST 49)")
    print("=" * 67)
    print()

    results = []

    # iterate over unordered pairs X<Y (to avoid duplicates)
    for i in range(len(energy_fields)):
        for j in range(i + 1, len(energy_fields)):
            fx = energy_fields[i]
            fy = energy_fields[j]
            X = df[fx].values
            Y = df[fy].values

            print(f"pair: {fx} × {fy}")
            b3_obs = fit_cross_coupling(G, X, Y)
            T_obs = abs(b3_obs)

            print(f"  observed interaction b3 = {b3_obs:.6f}  (|b3|={T_obs:.6f})")
            print("  running Monte Carlo null ...")

            b3_null = mc_null_b3(G, X, Y, n_sims=N_MC)
            mean_null = np.mean(b3_null)
            std_null = np.std(b3_null)
            p_val = np.mean(b3_null >= T_obs)

            print(f"  null mean |b3| = {mean_null:.6f}")
            print(f"  null std  |b3| = {std_null:.6f}")
            print(f"  p-value(|b3_null| >= |b3_obs|) = {p_val:.6f}")
            print("-" * 60)

            results.append(
                (fx, fy, b3_obs, T_obs, mean_null, std_null, p_val)
            )

    print()
    print("=" * 67)
    print("SUMMARY – ENERGY-GRADIENT CROSS COUPLING (TEST 49)")
    print("=" * 67)
    for fx, fy, b3_obs, T_obs, mean_null, std_null, p_val in results:
        print(
            f"{fx:8s} × {fy:8s}: "
            f"b3={b3_obs: .6f}, |b3|={T_obs: .6f}, "
            f"null_mean={mean_null: .6f}, null_std={std_null: .6f}, "
            f"p={p_val: .6f}"
        )
    print("=" * 67)
    print("test 49 complete.")
    print("=" * 67)


if __name__ == "__main__":
    main()
