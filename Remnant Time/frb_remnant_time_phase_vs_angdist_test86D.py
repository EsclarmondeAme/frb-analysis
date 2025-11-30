#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
from scipy.special import sph_harm
from math import radians, sin, cos, acos
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------
# helper: angular separation (great circle)
# ---------------------------------------------------------
def angdist(l1, b1, l2, b2):
    # radians
    l1 = np.radians(l1)
    l2 = np.radians(l2)
    b1 = np.radians(b1)
    b2 = np.radians(b2)

    cosang = (
        np.sin(b1)*np.sin(b2) +
        np.cos(b1)*np.cos(b2)*np.cos(l1 - l2)
    )

    # clamp numerically
    cosang = np.clip(cosang, -1.0, 1.0)

    return np.degrees(np.arccos(cosang))


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main(fname):
    print("="*58)
    print(" PHASE VS ANGULAR-SEPARATION TEST (TEST 86D)")
    print("="*58)

    # -----------------------------------------------------
    # load catalog
    # -----------------------------------------------------
    df = pd.read_csv(fname)
    N = len(df)
    print(f"[info] N_FRB = {N}")

    # -----------------------------------------------------
    # extract coordinates (galactic)
    # -----------------------------------------------------
    # expecting columns: l, b  (if not present, convert from ra/dec)
    if 'l' in df.columns and 'b' in df.columns:
        l = df['l'].values.astype(float)
        b = df['b'].values.astype(float)
    else:
        # convert from ra/dec
        import astropy.coordinates as coord
        import astropy.units as u
        ra = df['ra'].values.astype(float)
        dec = df['dec'].values.astype(float)
        sky = coord.SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
        gal = sky.galactic
        l = gal.l.value
        b = gal.b.value

    # -----------------------------------------------------
    # build spherical harmonics Y_lm, l_max=8
    # -----------------------------------------------------
    l_max = 8
    theta = np.radians(90 - b)   # polar angle
    phi   = np.radians(l)        # azimuth

    Y_list = []
    for ell in range(l_max+1):
        for m in range(-ell, ell+1):
            Y_list.append(sph_harm(m, ell, phi, theta))
    Y = np.vstack(Y_list)   # shape (#modes, N)

    # -----------------------------------------------------
    # compute pairwise phase alignment G_ij
    # -----------------------------------------------------
    # G_ij = real( <exp(i(phase_i)) conj(exp(i(phase_j)))> ) averaged over modes
    phases = np.angle(Y)
    n_modes = phases.shape[0]

    # pairwise differences for all unique pairs
    idx_i, idx_j = np.triu_indices(N, k=1)
    dphase = phases[:, idx_i] - phases[:, idx_j]
    G = np.mean(np.cos(dphase), axis=0)

    # -----------------------------------------------------
    # compute angular distances for each pair
    # -----------------------------------------------------
    ang = angdist(l[idx_i], b[idx_i], l[idx_j], b[idx_j])

    # -----------------------------------------------------
    # compute real correlation ρ(angular distance, G)
    # -----------------------------------------------------
    ang_mean = np.mean(ang)
    G_mean   = np.mean(G)
    cov      = np.mean((ang-ang_mean)*(G-G_mean))
    rho_real = cov / (np.std(ang)*np.std(G))

    print(f"[info] rho_real = {rho_real:.6e}")

    # -----------------------------------------------------
    # build null: shuffle sky positions (break geometry)
    # -----------------------------------------------------
    n_null = 2000
    rho_null = np.zeros(n_null)

    for k in range(n_null):
        # shuffle angular positions but keep G fixed
        perm = np.random.permutation(N)
        ang_null = angdist(l[perm[idx_i]], b[perm[idx_i]],
                           l[perm[idx_j]], b[perm[idx_j]])

        angm = np.mean(ang_null)
        cov_null = np.mean((ang_null-angm)*(G-G_mean))
        rho_null[k] = cov_null / (np.std(ang_null)*np.std(G))

    null_mean = np.mean(rho_null)
    null_std  = np.std(rho_null)
    p_value   = np.mean(np.abs(rho_null) >= abs(rho_real))

    print("---------------------------------------------------")
    print(f"null_mean = {null_mean:.6e}")
    print(f"null_std  = {null_std:.6e}")
    print(f"p_value   = {p_value:.6f}")
    print("---------------------------------------------------")
    print("interpretation:")
    print("  - significant |rho_real| >> 0 indicates phase coherence")
    print("    depends strongly on spatial separation.")
    print("  - comparison with 86B/86C shows whether spatial")
    print("    structure dominates over temporal structure.")
    print("====================================================")
    print("test 86D complete.")
    print("====================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_phase_vs_angdist_test86D.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
