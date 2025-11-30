#!/usr/bin/env python3
import sys
import csv
import numpy as np
from tqdm import tqdm
import frb_spectral_symmetry_breaking_test67 as T67

def load_catalog(path):
    RA = []
    Dec = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            RA.append(float(row["ra"]))
            Dec.append(float(row["dec"]))
    return np.array(RA), np.array(Dec)

def main(path):
    print("===============================================")
    print(" VALIDATION — TEST 67 (original version)")
    print("===============================================")

    RA, Dec = load_catalog(path)
    N = len(RA)

    print("[VAL] computing real spectral-alignment score...")
    real_score = T67.spectral_alignment_score(RA, Dec)
    print(f"[RESULT] real score = {real_score:.6f}")

    print("[VAL] computing isotropic null distribution (200 MC)...")
    null_scores = []
    for _ in tqdm(range(200)):
        rra, rdec = T67.random_isotropic(N)
        s = T67.spectral_alignment_score(rra, rdec)
        null_scores.append(s)

    null_scores = np.array(null_scores)
    mu = null_scores.mean()
    sd = null_scores.std()
    p = np.mean(null_scores >= real_score)

    print("-----------------------------------------------")
    print(f"null mean = {mu:.6f}")
    print(f"null std  = {sd:.6f}")
    print(f"p-value   = {p:.6f}")
    print("===============================================")

if __name__ == "__main__":
    main(sys.argv[1])
