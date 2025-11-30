#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv
import sys
import logging
from tqdm import tqdm
import frb_latent_manifold_test62 as T62

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")

# ------------------------------------------------------------
# load RA/Dec
# ------------------------------------------------------------
def load_frb_catalog(path):
    RA, Dec = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = {k.lower(): k for k in row.keys()}

            ra_key = None
            for cand in ["ra","ra_deg","raj2000"]:
                if cand in keys: ra_key = keys[cand]; break

            dec_key = None
            for cand in ["dec","dec_deg","decj2000"]:
                if cand in keys: dec_key = keys[cand]; break

            if ra_key is None or dec_key is None:
                raise KeyError("RA/Dec columns not found")

            RA.append(float(row[ra_key]))
            Dec.append(float(row[dec_key]))

    return np.array(RA), np.array(Dec)

# ------------------------------------------------------------
# validator
# ------------------------------------------------------------
def main(path):
    print("===============================================")
    print(" VALIDATION — TEST 62 (Latent Manifold)")
    print("===============================================")

    logging.info("loading FRB catalog...")
    RA, Dec = load_frb_catalog(path)
    N = len(RA)
    logging.info(f"loaded {N} FRBs")

    logging.info("computing spherical distance matrix...")
    D = T62.compute_pairwise_spherical(RA, Dec)

    logging.info("computing real latent-manifold score...")
    real_score = T62.manifold_score(D)

    logging.info("computing null distribution (2000 MC)...")
    null_scores = []
    for _ in tqdm(range(2000), desc="MC null"):
        ra_r = np.random.uniform(0, 360, N)
        dec_r = np.degrees(np.arcsin(np.random.uniform(-1, 1, N)))
        D_r = T62.compute_pairwise_spherical(ra_r, dec_r)
        null_scores.append(T62.manifold_score(D_r))
    null_scores = np.array(null_scores)

    mu = null_scores.mean()
    sd = null_scores.std()
    p = np.mean(null_scores >= real_score)

    print(f"real score   = {real_score:.6f}")
    print(f"null mean    = {mu:.6f}")
    print(f"null std     = {sd:.6f}")
    print(f"p-value      = {p:.6f}")
    print("===============================================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python validate_test62.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
