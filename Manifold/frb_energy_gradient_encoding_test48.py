#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
import sys

# ---------------------------------------------------------
# helper: compute grain intensity using nearest-neighbor deltas
# ---------------------------------------------------------
def compute_grain_intensity(ra, dec, k=3):
    coords = np.vstack([ra, dec]).T
    tree = cKDTree(coords)
    dists, idx = tree.query(coords, k=k+1)  # include itself as index 0
    d_near = dists[:, 1:]
    return np.mean(d_near, axis=1)

# ---------------------------------------------------------
# helper: linear regression slope
# ---------------------------------------------------------
def linear_slope(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m

# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_energy_gradient_encoding_test48.py frbs_unified.csv")
        sys.exit(1)

    fn = sys.argv[1]
    df = pd.read_csv(fn)
    print(f"loaded {len(df)} FRBs")

    # relevant energy-like fields
    fields = ["dm", "snr", "fluence", "width", "z_est"]
    print("using fields:", fields)

    # raw angular coordinates for micro-grain field
    ra = df["ra"].values
    dec = df["dec"].values

    # compute grain intensity map
    print("computing grain intensities...")
    grain = compute_grain_intensity(ra, dec, k=3)

    # store results
    obs_slopes = {}
    null_slopes = {f: [] for f in fields}

    N_MC = 5000
    print(f"running Monte Carlo null (N={N_MC})...")

    # compute observed slopes
    for f in fields:
        x = df[f].values
        obs_slopes[f] = linear_slope(x, grain)

    # Monte Carlo: shuffle grain intensities
    for _ in tqdm(range(N_MC)):
        g_shuf = np.random.permutation(grain)
        for f in fields:
            x = df[f].values
            null_slopes[f].append(linear_slope(x, g_shuf))

    # compute p-values
    print("\n====================================================================")
    print("         FRB ENERGY-GRADIENT ENCODING TEST (TEST 48)")
    print("====================================================================\n")

    for f in fields:
        obs = obs_slopes[f]
        null = np.array(null_slopes[f])
        p = np.mean(np.abs(null) >= np.abs(obs))

        print(f"field: {f}")
        print(f"  observed slope = {obs:.6f}")
        print(f"  null mean      = {np.mean(null):.6f}")
        print(f"  null std       = {np.std(null):.6f}")
        print(f"  p-value(|null| >= |obs|) = {p:.6f}")
        print("------------------------------------------------------------")

    print("====================================================================")
    print("test 48 complete.")
    print("====================================================================")


if __name__ == "__main__":
    main()
