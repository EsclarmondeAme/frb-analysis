#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

import frb_remnant_time_manifold_dilation_test72 as T72


######################################################################
# hijack: safe knn
######################################################################
def knn_thickness_safe(X):
    N = len(X)
    if N <= 1:
        return np.zeros(N)
    K = min(12, max(1, N - 1))
    D = np.sqrt(((X[:,None,:] - X[None,:,:])**2).sum(axis=2))
    idx = np.argpartition(D, kth=K, axis=1)[:, :K]
    return D[np.arange(N)[:,None], idx].mean(axis=1)

T72.knn_thickness = knn_thickness_safe


######################################################################
# NEW: safe dilation metric that uses already-balanced hemispheres
######################################################################
def compute_dilation_balanced_engine(X_pos, X_neg, axis_xyz):
    """
    This is the correct engine bypass:
    - no internal hemisphere re-split
    - compute dilation metrics using the true balanced sets
    """

    # ----- pairwise thickness -----
    def mean_pairwise_dist(X):
        if len(X) <= 1:
            return 0.0
        D = np.sqrt(((X[:,None,:] - X[None,:,:])**2).sum(axis=2))
        return D.mean()

    d_pos = mean_pairwise_dist(X_pos)
    d_neg = mean_pairwise_dist(X_neg)
    dp = d_pos - d_neg

    # ----- knn thickness -----
    t_pos = knn_thickness_safe(X_pos)
    t_neg = knn_thickness_safe(X_neg)

    # take hemisphere averages
    dk = t_pos.mean() - t_neg.mean()

    # ----- harmonic thickness -----
    # use the engine's harmonic spread function
    def harmonic_spread(X):
        n = len(X)
        if n <= 1:
            return 0.0
        v = X.mean(axis=0)
        v = v / np.linalg.norm(v)
        ang = np.arccos(np.clip((X @ v), -1, 1))
        return np.mean(1.0 / np.maximum(ang, 1e-6))

    dh = harmonic_spread(X_pos) - harmonic_spread(X_neg)

    S = dp + dk + dh
    return S, dp, dk, dh


######################################################################
def load_catalog(path):
    df = pd.read_csv(path)
    return df, df["ra"].values, df["dec"].values


def galactic_to_supergalactic(ra_deg, dec_deg):
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    c_sg = c.transform_to("supergalactic")
    return c_sg.sgl.value, c_sg.sgb.value


def xyz_from_radec(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T


######################################################################
# balanced hemispheres top-level
######################################################################
def compute_balanced(Xgal, axis_xyz):
    rng = np.random.default_rng()

    # project onto axis
    z = Xgal @ (axis_xyz / np.linalg.norm(axis_xyz))

    X_pos_full = Xgal[z >= 0]
    X_neg_full = Xgal[z < 0]

    Np = len(X_pos_full)
    Nn = len(X_neg_full)
    Nmin = min(Np, Nn)

    idx_pos = rng.choice(Np, Nmin, replace=False)
    idx_neg = rng.choice(Nn, Nmin, replace=False)

    X_pos = X_pos_full[idx_pos]
    X_neg = X_neg_full[idx_neg]

    S, dp, dk, dh = compute_dilation_balanced_engine(X_pos, X_neg, axis_xyz)
    return S, dp, dk, dh, Np, Nn, Nmin


######################################################################
def main(path):
    print("===============================================================")
    print("FRB REMNANT-TIME MANIFOLD DILATION TEST (72B — SGB mask 20°)")
    print("balanced hemispheres, corrected engine bypass")
    print("===============================================================")

    df, RA, DEC = load_catalog(path)
    SGL, SGB = galactic_to_supergalactic(RA, DEC)
    mask = np.abs(SGB) >= 20

    df_m = df[mask]
    RA_m = RA[mask]
    DEC_m = DEC[mask]

    print(f"[info] N_FRB original = {len(df)}")
    print(f"[info] N_after_mask   = {len(df_m)}")

    Xgal = xyz_from_radec(RA_m, DEC_m)

    th = np.deg2rad(df_m["theta_unified"].values)
    ph = np.deg2rad(df_m["phi_unified"].values)
    unified_axis = np.array([
        np.mean(np.sin(th)*np.cos(ph)),
        np.mean(np.sin(th)*np.sin(ph)),
        np.mean(np.cos(th))
    ])
    unified_axis /= np.linalg.norm(unified_axis)

    # ----- real -----
    S_real, dp, dk, dh, Np, Nn, Nmin = compute_balanced(Xgal, unified_axis)
    print(f"[info] hemisphere sizes: N_pos={Np}, N_neg={Nn} → balanced N={Nmin}")
    print("[info] building null (2000 axes)...")

    rng = np.random.default_rng(0)
    scores = []
    for _ in range(2000):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        S_rand, *_ = compute_balanced(Xgal, v)
        scores.append(S_rand)

    scores = np.array(scores)
    mu = scores.mean()
    sd = scores.std()
    p = (np.sum(scores >= S_real) + 1) / 2001

    print("===============================================================")
    print("72B RESULT")
    print("===============================================================")
    print(f"N_pos = {Np}, N_neg = {Nn}, balanced N = {Nmin}")
    print("--------------------------------------------------------------")
    print(f"delta_pairwise = {dp}")
    print(f"delta_knn      = {dk}")
    print(f"delta_harmonic = {dh}")
    print("--------------------------------------------------------------")
    print(f"S_real         = {S_real}")
    print(f"null mean S    = {mu}")
    print(f"null std S     = {sd}")
    print(f"p-value        = {p}")
    print("===============================================================")
    print("test 72B complete.")
    print("===============================================================")


if __name__ == "__main__":
    main(sys.argv[1])
