#!/usr/bin/env python3
import numpy as np
import csv
from math import radians, sin, cos, acos
from scipy.spatial import cKDTree

# ============================================================
# helpers
# ============================================================

def ang2cart(ra, dec):
    ra = np.radians(ra)
    dec = np.radians(dec)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x,y,z])

def angular_sep(ra1, dec1, ra2, dec2):
    ra1 = radians(ra1); dec1 = radians(dec1)
    ra2 = radians(ra2); dec2 = radians(dec2)
    return acos(
        sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra1-ra2)
    )

def load_catalog(path):
    RA=[]; Dec=[]; Rsign=[]
    with open(path,newline='',encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for row in rd:
            ra = float(row["ra"])
            dec = float(row["dec"])
            th = float(row["theta_unified"])

            # hemisphere sign (same as tests 70–75)
            s = 1 if th < 90 else -1

            RA.append(ra); Dec.append(dec); Rsign.append(s)
    return np.array(RA), np.array(Dec), np.array(Rsign)

# ============================================================
# galactic latitude computation (for mask)
# ============================================================

def equatorial_to_galactic(ra_deg, dec_deg):
    # IAU 1958 / J2000 transform constants
    ra_gp  = radians(192.85948)
    dec_gp = radians(27.12825)
    l_asc_node = radians(32.93192)

    ra = np.radians(ra_deg)
    dec= np.radians(dec_deg)

    b = np.arcsin(
        np.sin(dec)*np.sin(dec_gp)
        + np.cos(dec)*np.cos(dec_gp)*np.cos(ra - ra_gp)
    )
    return np.degrees(b)

# ============================================================
# curvature–drift metric
# ============================================================

def local_curvature_and_drift(X, k=10):
    """
    X: Nx3 unit vectors
    k: number of neighbors
    returns curvature_est, drift_mag (arrays of shape N)
    """
    tree = cKDTree(X)
    N = len(X)
    curvature = np.zeros(N)
    drift = np.zeros(N)

    for i in range(N):
        d, idx = tree.query(X[i], k+1)
        idx = idx[1:]   # remove self
        nbrs = X[idx]

        # curvature estimator: variance of neighbour chord distances
        chord = np.linalg.norm(nbrs - X[i], axis=1)
        curvature[i] = np.var(chord)

        # drift estimator: norm of mean neighbour displacement
        drift[i] = np.linalg.norm(np.mean(nbrs - X[i], axis=0))

    return curvature, drift

# ============================================================
# main test engine (76A)
# ============================================================

def main(path, NMC=2000):

    print("===============================================")
    print("FRB REMNANT-TIME CURVATURE–DRIFT TEST (76A)")
    print("Galactic mask: |b| >= 20°")
    print("===============================================")

    RA, Dec, Rsign = load_catalog(path)

    # apply galactic mask
    b = equatorial_to_galactic(RA, Dec)
    mask = np.abs(b) >= 20
    RA = RA[mask]; Dec = Dec[mask]; Rsign = Rsign[mask]

    print(f"[info] original N=600, after mask N={len(RA)}")

    X = ang2cart(RA, Dec)

    # compute real metrics
    curv, drift = local_curvature_and_drift(X, k=10)

    curv_pos = curv[Rsign > 0]
    curv_neg = curv[Rsign < 0]
    drift_pos = drift[Rsign > 0]
    drift_neg = drift[Rsign < 0]

    D_real = (np.mean(curv_pos) - np.mean(curv_neg)) \
           + (np.mean(drift_pos) - np.mean(drift_neg))

    # Monte Carlo: rotate sign labels
    D_null = []
    N = len(Rsign)
    for _ in range(NMC):
        p = np.random.permutation(Rsign)
        cpos = curv[p > 0]; cneg = curv[p < 0]
        dpos = drift[p > 0]; dneg = drift[p < 0]
        D = (np.mean(cpos)-np.mean(cneg)) + (np.mean(dpos)-np.mean(dneg))
        D_null.append(D)

    D_null = np.array(D_null)
    p = np.mean(np.abs(D_null) >= np.abs(D_real))

    print("------------------------------------------------")
    print(f"mean curvature (R>0) = {np.mean(curv_pos)}")
    print(f"mean curvature (R<0) = {np.mean(curv_neg)}")
    print(f"mean drift     (R>0) = {np.mean(drift_pos)}")
    print(f"mean drift     (R<0) = {np.mean(drift_neg)}")
    print(f"D_real               = {D_real}")
    print("------------------------------------------------")
    print(f"null mean D          = {np.mean(D_null)}")
    print(f"null std D           = {np.std(D_null)}")
    print(f"p-value              = {p}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> curvature–drift differs between hemispheres")
    print("            even after removing Galactic plane.")
    print("  high p -> symmetric; consistent with isotropy.")
    print("===============================================")
    print("test 76A complete.")
    print("===============================================")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
