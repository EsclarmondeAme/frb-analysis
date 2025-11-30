#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VALIDATION — TEST 63 (Harmonic Manifold)
This validator matches exactly the API of frb_harmonic_manifold_test63.py
and does NOT invent any extra arguments.
"""

import sys
import numpy as np
import csv
from tqdm import tqdm

# import the real test
import frb_harmonic_manifold_test63 as T63

# -------------------------------------------------------------
# load RA/Dec with auto-detected column names (matches Test 63)
# -------------------------------------------------------------
def load_catalog(path):
    RA, Dec = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = {k.lower(): k for k in row.keys()}

            # RA
            ra_key = None
            for cand in ["ra", "ra_deg", "raj2000"]:
                if cand in keys:
                    ra_key = keys[cand]
                    break

            # Dec
            dec_key = None
            for cand in ["dec", "dec_deg", "decj2000"]:
                if cand in keys:
                    dec_key = keys[cand]
                    break

            if ra_key is None or dec_key is None:
                raise KeyError("Could not auto-detect RA/Dec columns")

            RA.append(float(row[ra_key]))
            Dec.append(float(row[dec_key]))

    return np.array(RA), np.array(Dec)

# -------------------------------------------------------------
# run validator
# -------------------------------------------------------------
def main(csv_path):

    print("===============================================")
    print(" VALIDATION — TEST 63 (Harmonic Manifold)")
    print("===============================================")

    RA, Dec = load_catalog(csv_path)
    n = len(RA)
    print(f"[INFO] loaded {n} FRBs")

    # compute real score
    print("[VAL] computing real harmonic-manifold score...")
    H_real = T63.compute_H_score(RA, Dec)

    # compute null distribution
    print("[VAL] computing isotropic null distribution (200 MC)...")
    H_null = []
    for _ in tqdm(range(200)):
        u = np.random.uniform(-1, 1, n)
        dec_r = np.degrees(np.arcsin(u))
        ra_r = np.random.uniform(0, 360, n)
        H_null.append(T63.compute_H_score(ra_r, dec_r))

    H_null = np.array(H_null)
    mean_null = H_null.mean()
    std_null  = H_null.std()
    p = np.mean(H_null >= H_real)

    print("-----------------------------------------------")
    print(f"real score   = {H_real:.6f}")
    print(f"null mean    = {mean_null:.6f}")
    print(f"null std     = {std_null:.6f}")
    print(f"p-value      = {p:.6f}")
    print("===============================================")
    print(" validation complete.")
    print("===============================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python validate_test63.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
