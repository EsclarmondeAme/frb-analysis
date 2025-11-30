#!/usr/bin/env python3
# ============================================================
# VALIDATION FRAMEWORK FOR TEST 68C
# ============================================================

import sys
import csv
import numpy as np
from tqdm import tqdm
import importlib.util

# ------------------------------------------------------------
# helper: RA/Dec -> Cartesian
# ------------------------------------------------------------
def radec_to_xyz(ra_deg, dec_deg):
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z])

# ------------------------------------------------------------
# load FRB catalog (ra/dec lowercase)
# ------------------------------------------------------------
def load_catalog(path):
    RA, Dec = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            RA.append(float(row["ra"]))
            Dec.append(float(row["dec"]))
    X = np.array([radec_to_xyz(r, d) for r, d in zip(RA, Dec)])
    return X

# ------------------------------------------------------------
# dynamically import the test file (68C)
# ------------------------------------------------------------
def load_test_module(filename, module_name="test68C_mod"):
    spec = importlib.util.spec_from_file_location(module_name, filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ------------------------------------------------------------
# generate isotropic sky
# ------------------------------------------------------------
def random_catalog(N):
    RA = np.random.uniform(0, 360, N)
    Dec = np.degrees(np.arcsin(np.random.uniform(-1, 1, N)))
    return np.array([radec_to_xyz(r, d) for r, d in zip(RA, Dec)])

# ------------------------------------------------------------
# main validation
# ------------------------------------------------------------
def main(test_file, frb_csv):
    print("===============================================")
    print(" VALIDATION FRAMEWORK — TEST 68C")
    print("===============================================")

    # load FRBs
    X = load_catalog(frb_csv)
    N = len(X)
    print(f"[INFO] loaded {N} FRBs")

    # import test 68C file dynamically
    test68 = load_test_module(test_file)

    # build structures ONCE
    print("[VALIDATION] computing real G score...")
    nbrs, _ = test68.build_knn_graph(X, k=15)
    V       = test68.local_principal_axis(X, nbrs)
    G_real  = test68.geodesic_flow_score(X, nbrs, V)
    print(f"[RESULT] G_real = {G_real:.6f}")

    # --------------------------------------------------------
    # isotropic check — should give p ≈ 0.5
    # --------------------------------------------------------
    print("[VALIDATION] computing isotropic baseline...")
    G_iso = []
    for _ in tqdm(range(200)):
        Xi = random_catalog(N)
        nbrsi, _ = test68.build_knn_graph(Xi, k=15)
        Vi = test68.local_principal_axis(Xi, nbrsi)
        G_iso.append(test68.geodesic_flow_score(Xi, nbrsi, Vi))

    G_iso = np.array(G_iso)
    iso_mu = G_iso.mean()
    iso_sd = G_iso.std()
    print(f"[RESULT] isotropic mean  = {iso_mu:.6f}")
    print(f"[RESULT] isotropic std   = {iso_sd:.6f}")

    # real p vs isotropic
    p_iso = np.mean(G_iso >= G_real)
    print(f"[RESULT] p_vs_isotropic  = {p_iso:.4f}")

    print("===============================================")
    print(" validation complete.")
    print("===============================================")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
