#!/usr/bin/env python3
# ============================================================
# FRB DIPOLE-SUBTRACTED SHELL TEST
# ============================================================
# this script performs the decisive test:
#
#   does the frb radial shell (0–10°, 10–25°, 25–40°, 40–90°)
#   survive after mathematically removing the best-fit dipole?
#
# steps:
#   1. load frbs.csv (ra, dec)
#   2. compute dipole direction & vector
#   3. subtract dipole component from each unit vector:
#        v_sub = v − (v·d_hat)d_hat
#   4. renormalize v_sub
#   5. convert back to galactic (l,b)
#   6. redo layered χ² radial shell test
#
# if the shell survives, the anisotropy is higher-order (quadrupole+).
# ============================================================

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# ------------------------------------------------------------
# config
# ------------------------------------------------------------
CATALOG_FILE = r"C:\Users\ratec\Documents\CrossLayerPhysics\frbs.csv"


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def load_frbs(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["ra", "dec"])
    return df


def to_unit_vectors(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T  # shape (N,3)


def dipole_direction(vecs):
    mean_vec = vecs.mean(axis=0)
    r = np.linalg.norm(mean_vec)
    if r == 0:
        d_hat = np.array([1.0, 0.0, 0.0])
    else:
        d_hat = mean_vec / r
    # convert to galactic
    c = SkyCoord(
        x=d_hat[0], y=d_hat[1], z=d_hat[2],
        frame="icrs", representation_type="cartesian"
    ).galactic
    return d_hat, c.l.deg, c.b.deg, r


def subtract_dipole(vecs, d_hat):
    """
    v_sub = v − (v·d_hat)d_hat
    then normalized.
    """
    dot = np.dot(vecs, d_hat)  # shape (N,)
    proj = dot[:, None] * d_hat[None, :]
    v_sub = vecs - proj
    # renormalize
    norms = np.linalg.norm(v_sub, axis=1)
    # avoid division by zero
    norms[norms == 0] = 1e-12
    v_sub = v_sub / norms[:, None]
    return v_sub


def vectors_to_galactic(vecs):
    xs, ys, zs = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    c = SkyCoord(
        x=xs, y=ys, z=zs,
        frame="icrs",
        representation_type="cartesian"
    ).galactic
    return c.l.deg, c.b.deg


def radial_shell_counts(l, b):
    """
    compute counts in:
       0–10°,  10–25°,  25–40°,  40–90° relative to galactic north pole.
    but the shell test is actually angle from the *unified axis*.
    for dipole-subtracted test, we use angle from galactic north.
    """
    # angle from galactic pole = colatitude = theta = 90° - |b|
    theta = 90 - np.abs(b)

    bands = [(0, 10), (10, 25), (25, 40), (40, 90)]
    counts = []
    for lo, hi in bands:
        mask = (theta >= lo) & (theta < hi)
        counts.append(mask.sum())
    return counts


def isotropic_expectation(n_total):
    """
    isotropic counts in same 4 bands.
    probability = (cos(lo) - cos(hi)).
    """
    edges = [(0, 10), (10, 25), (25, 40), (40, 90)]
    exp = []
    for lo, hi in edges:
        p = (np.cos(np.deg2rad(lo)) - np.cos(np.deg2rad(hi)))
        exp.append(n_total * p)
    return exp


def chi2_stat(obs, exp):
    return np.sum((obs - exp)**2 / exp)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    print("===================================================")
    print("FRB DIPOLE-SUBTRACTED SHELL TEST")
    print("===================================================\n")

    df = load_frbs(CATALOG_FILE)
    n = len(df)
    print(f"loaded FRBs: {n}")

    # original vectors
    vecs = to_unit_vectors(df["ra"].values, df["dec"].values)

    # compute dipole
    d_hat, l_d, b_d, r = dipole_direction(vecs)

    print("\noriginal FRB dipole:")
    print(f"  galactic l = {l_d:.3f} deg")
    print(f"  galactic b = {b_d:.3f} deg")
    print(f"  dipole amplitude r = {r:.4f}\n")

    # subtract dipole
    vecs_sub = subtract_dipole(vecs, d_hat)

    # convert back to galactic
    l_sub, b_sub = vectors_to_galactic(vecs_sub)

    # radial bands
    obs = radial_shell_counts(l_sub, b_sub)
    exp = isotropic_expectation(n)

    chi2 = chi2_stat(np.array(obs), np.array(exp))

    # degrees of freedom = 3 (4 bins - 1)
    from scipy.stats import chi2 as chi2dist
    p_val = 1 - chi2dist.cdf(chi2, df=3)

    print("dipole-subtracted sky shell counts:")
    print(f"  band   0–10° : obs={obs[0]:4d}, exp={exp[0]:7.2f}, ratio={obs[0]/exp[0]:.2f}")
    print(f"  band 10–25° : obs={obs[1]:4d}, exp={exp[1]:7.2f}, ratio={obs[1]/exp[1]:.2f}")
    print(f"  band 25–40° : obs={obs[2]:4d}, exp={exp[2]:7.2f}, ratio={obs[2]/exp[2]:.2f}")
    print(f"  band 40–90° : obs={obs[3]:4d}, exp={exp[3]:7.2f}, ratio={obs[3]/exp[3]:.2f}")

    print(f"\ntotal chi² (vs isotropy) = {chi2:.2f} (dof=3)")
    print(f"p-value = {p_val:.3e}")

    print("\n--------------- scientific verdict ---------------")
    if p_val < 1e-5:
        print("the layered radial anisotropy survives even after removing the full dipole.")
        print("this indicates the frb sky contains higher-order structure (quadrupole+),")
        print("not merely leakage from the dipole component.")
        print("the shell is therefore a genuine geometric feature.")
    elif p_val < 0.05:
        print("the shell remains moderately significant after dipole subtraction.")
        print("this suggests part of the anisotropy is higher-order, but some")
        print("fraction may have been dipole-related.")
    else:
        print("once the dipole is mathematically removed, the shell evaporates.")
        print("this implies the shell was primarily dipole leakage, not a separate")
        print("higher-order geometric structure.")
    print("---------------------------------------------------")
    print("analysis complete.")
    print("===================================================\n")


if __name__ == "__main__":
    main()
