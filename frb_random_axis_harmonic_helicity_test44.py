#!/usr/bin/env python3
"""
FRB RANDOM-AXIS HARMONIC HELICITY CONTRAST TEST (TEST 44)

purpose
-------
test whether the strong m=1+m=2 harmonic / "double-helix" structure
is specifically tied to the unified axis, or whether similar harmonic
strength appears for generic random axes on the sky.

the test compares:

- H_true  : harmonic strength (m=1 + m=2) measured in the unified-axis
            frame, using theta_unified, phi_unified.
- H_rand  : harmonic strengths measured for many random trial axes,
            using the full-sky (ra, dec) and reprojecting the FRBs
            into each trial axis frame.

if H_true lies far in the upper tail of the H_rand distribution,
the double-helix structure is strongly axis-specific.
"""

import sys
import numpy as np
import pandas as pd

# theta shell where harmonic structure is strongest (as in earlier tests)
THETA_MIN = 25.0
THETA_MAX = 60.0

# number of random trial axes
N_MC = 5000

# minimum number of FRBs in shell to consider the axis "usable"
MIN_IN_SHELL = 30


# ------------------------------------------------------------
# helpers: loading and coordinate transforms
# ------------------------------------------------------------

def load_frbs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["ra", "dec", "theta_unified", "phi_unified"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    return df


def radec_to_cartesian(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """convert (ra, dec) in degrees to unit vectors in 3d."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    cos_dec = np.cos(dec)
    x = cos_dec * np.cos(ra)
    y = cos_dec * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T  # shape (N, 3)


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """draw a single isotropic random unit vector."""
    # method: normal vector then normalize
    v = rng.normal(size=3)
    norm = np.linalg.norm(v)
    if norm == 0:
        return random_unit_vector(rng)
    return v / norm


def axis_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    construct an orthonormal basis (e1, e2, e3) with:
      e3 = axis (unit)
      e1, e2 spanning the perpendicular plane
    """
    e3 = axis / np.linalg.norm(axis)

    # choose a reference vector not too parallel to e3
    z_hat = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(e3, z_hat)) < 0.9:
        ref = z_hat
    else:
        ref = np.array([1.0, 0.0, 0.0])

    # make e1 perpendicular to e3
    v1 = ref - np.dot(ref, e3) * e3
    n1 = np.linalg.norm(v1)
    if n1 == 0:
        # fallback
        v1 = np.array([1.0, 0.0, 0.0]) - e3[0] * e3
        n1 = np.linalg.norm(v1)
    e1 = v1 / n1

    # e2 = e3 x e1
    e2 = np.cross(e3, e1)
    e2 = e2 / np.linalg.norm(e2)

    return e1, e2, e3


def project_to_axis_shell(
    frb_vecs: np.ndarray,
    axis: np.ndarray,
    theta_min: float = THETA_MIN,
    theta_max: float = THETA_MAX,
) -> np.ndarray:
    """
    project FRB directions onto a coordinate system whose pole is 'axis',
    then select those in the theta shell [theta_min, theta_max].

    returns array of phi angles (in radians) for FRBs in the shell.
    """
    e1, e2, e3 = axis_basis(axis)

    # projections
    v3 = frb_vecs @ e3  # cos(theta)
    v3_clipped = np.clip(v3, -1.0, 1.0)
    theta = np.degrees(np.arccos(v3_clipped))

    # azimuth from e1, e2 components
    v1 = frb_vecs @ e1
    v2 = frb_vecs @ e2
    phi = np.arctan2(v2, v1)  # range [-pi, pi]

    mask = (theta >= theta_min) & (theta <= theta_max)
    return phi[mask]


# ------------------------------------------------------------
# harmonic-helicity strength measure
# ------------------------------------------------------------

def harmonic_strength(phi: np.ndarray, min_n: int = MIN_IN_SHELL) -> tuple[float, float, float]:
    """
    compute m=1 and m=2 complex Fourier modes and a combined strength:

      a1 = (1/N) sum exp(i * phi)
      a2 = (1/N) sum exp(2i * phi)

    returns (H, |a1|, |a2|) where H = |a1|^2 + |a2|^2.

    if insufficient FRBs, returns (0, 0, 0).
    """
    n = phi.size
    if n < min_n:
        return 0.0, 0.0, 0.0

    a1 = np.mean(np.exp(1j * phi))
    a2 = np.mean(np.exp(2j * phi))
    H1 = np.abs(a1)
    H2 = np.abs(a2)
    H = H1**2 + H2**2
    return float(H), float(H1), float(H2)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("usage: python frb_random_axis_harmonic_helicity_test44.py frbs_unified.csv")
        sys.exit(1)

    path = sys.argv[1]
    df = load_frbs(path)

    # unified-axis shell for "true" harmonic strength
    mask_true = (df["theta_unified"] >= THETA_MIN) & (df["theta_unified"] <= THETA_MAX)
    phi_true = np.deg2rad(df.loc[mask_true, "phi_unified"].values)
    n_true = phi_true.size

    # build FRB 3d vectors from ra, dec for random-axis projections
    frb_vecs = radec_to_cartesian(df["ra"].values, df["dec"].values)

    # compute H_true from unified-axis frame
    H_true, H1_true, H2_true = harmonic_strength(phi_true)

    # monte carlo over random axes
    rng = np.random.default_rng(123)
    H_rand = []

    for _ in range(N_MC):
        axis = random_unit_vector(rng)
        phi_shell = project_to_axis_shell(frb_vecs, axis, THETA_MIN, THETA_MAX)
        H, _, _ = harmonic_strength(phi_shell)
        H_rand.append(H)

    H_rand = np.array(H_rand)
    mean_H = float(np.mean(H_rand))
    std_H = float(np.std(H_rand))

    # p-value: probability that random axis has harmonic strength >= true axis
    if H_true <= 0:
        p = 1.0
    else:
        p = float(np.mean(H_rand >= H_true))

    # --------------------------------------------------------
    # print results
    # --------------------------------------------------------
    print("===================================================================")
    print("FRB RANDOM-AXIS HARMONIC HELICITY CONTRAST TEST (TEST 44)")
    print("===================================================================")
    print(f"theta shell used: {THETA_MIN:.1f}–{THETA_MAX:.1f} deg")
    print(f"unified-axis shell count N_true = {n_true}")
    print("-------------------------------------------------------------------")
    print("unified-axis harmonic strength (m=1, m=2):")
    print(f"  |a1_true| = {H1_true:.4f}")
    print(f"  |a2_true| = {H2_true:.4f}")
    print(f"  H_true    = |a1|^2 + |a2|^2 = {H_true:.6f}")
    print("-------------------------------------------------------------------")
    print(f"random-axis harmonic strengths (N_MC = {N_MC}):")
    print(f"  mean(H_rand) = {mean_H:.6f}")
    print(f"  std(H_rand)  = {std_H:.6f}")
    print(f"  fraction with H_rand >= H_true = {p:.6f}")
    print("-------------------------------------------------------------------")
    print("interpretation:")
    print("  - small p (≲ 0.05): unified-axis double-helix is unusually strong")
    print("    compared to random axes → axis-specific harmonic helicity.")
    print("  - large p: similar harmonic strength appears for many orientations,")
    print("    suggesting the double-helix is not tightly locked to this axis.")
    print("===================================================================")
    print("test 44 complete.")
    print("===================================================================")


if __name__ == "__main__":
    main()
