#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PARAMETER SWEEP VALIDATION — TEST 68C
-------------------------------------
This runs large-scale calibration tests of the geodesic-flow
stability statistic under pure isotropic skies.

Goal:
    Ensure G-score null distribution is stable across all
    reasonable parameter choices.

Outputs:
    For each (k, step_deg, steps, seeds):
        - mean(G_null)
        - std(G_null)
        - p-value test using synthetic G_real ~ isotropic
        - flag "OK" if calibrated (p ≈ 0.5)
"""

import sys
import numpy as np
from tqdm import tqdm
from importlib import import_module
from scipy.spatial import cKDTree

# ============================================================
# load the user’s Test 68C file dynamically
# ============================================================

def load_test68(filename):
    modname = filename.replace(".py", "")
    spec = import_module(modname)
    return spec

# ============================================================
# isotropic FRB generator
# ============================================================

def random_catalog(N):
    RA = np.random.uniform(0, 360, N)
    Dec = np.degrees(np.arcsin(np.random.uniform(-1, 1, N)))
    X = []
    for ra, dec in zip(RA, Dec):
        ra_r = np.radians(ra)
        dec_r = np.radians(dec)
        x = np.cos(dec_r)*np.cos(ra_r)
        y = np.cos(dec_r)*np.sin(ra_r)
        z = np.sin(dec_r)
        X.append([x,y,z])
    return np.array(X)

# ============================================================
# MAIN SWEEP
# ============================================================

def main(test68_file):
    print("=====================================================")
    print(" PARAMETER SWEEP — TEST 68C")
    print("=====================================================")

    test68 = load_test68(test68_file)

    # parameters to sweep
    k_list = [8, 10, 12, 15]
    step_deg_list = [0.1, 0.25, 0.5]
    steps_list = [30, 50, 80]
    seeds_list = [20, 40]

    N = 600  # synthetic FRB count to match real catalog
    Nsim = 120  # number of isotropic null realisations per combination

    print(f"[INFO] using Nsim = {Nsim} per parameter combination")
    print()

    # header
    print(f"{'k':>3} {'step':>6} {'steps':>5} {'seeds':>5} {'mean':>12} {'std':>12} {'p≈0.5?':>8}")
    print("-"*70)

    # ------------------------------------------------------------------
    # sweep
    # ------------------------------------------------------------------
    for k in k_list:
        for step_deg in step_deg_list:
            for steps in steps_list:
                for seeds in seeds_list:

                    # patch parameters into test68 module
                    test68.K_GLOBAL = k
                    test68.STEP_GLOBAL = step_deg
                    test68.NSTEP_GLOBAL = steps
                    test68.SEEDS_GLOBAL = seeds

                    # compute null distribution
                    G_null = []
                    for _ in range(Nsim):
                        X = random_catalog(N)
                        nbrs, _ = test68.build_knn_graph(X, k=k)
                        V = test68.local_principal_axis(X, nbrs)
                        G_null.append(test68.geodesic_flow_score(
                            X, nbrs, V,
                            n_rand=seeds//2,
                            n_high=seeds//2
                        ))

                    G_null = np.array(G_null)
                    mu = G_null.mean()
                    sd = G_null.std()

                    # p-test: for isotropy, any “fake G_real" should sit at center
                    fake_real = random_catalog(N)
                    nbrs_fake, _ = test68.build_knn_graph(fake_real, k=k)
                    V_fake = test68.local_principal_axis(fake_real, nbrs_fake)
                    G_fake_real = test68.geodesic_flow_score(
                        fake_real, nbrs_fake, V_fake,
                        n_rand=seeds//2,
                        n_high=seeds//2
                    )
                    p = np.mean(G_null >= G_fake_real)

                    calibration_flag = "OK" if 0.3 <= p <= 0.7 else "BAD"

                    print(f"{k:>3} {step_deg:>6.2f} {steps:>5} {seeds:>5} "
                          f"{mu:>12.4f} {sd:>12.4f} {calibration_flag:>8}")

    print("-----------------------------------------------------")
    print(" parameter sweep complete.")
    print("-----------------------------------------------------")


# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python param_sweep_68C.py frb_geodesic_flow_stability_test68.py")
        sys.exit(1)
    main(sys.argv[1])
