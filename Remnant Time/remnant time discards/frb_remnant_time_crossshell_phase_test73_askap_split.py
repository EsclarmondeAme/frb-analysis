import numpy as np
import pandas as pd
import os
from math import radians, sin, cos, acos
import sys
from tqdm import tqdm
from astropy.io import fits

# --------------------------------------------------------
# constants
# --------------------------------------------------------
ASKAP_FITS_DIR = r"data/positions"
MATCH_TOL = 0.5   # degrees, based on nearest-distance diagnostic

# --------------------------------------------------------
# utilities
# --------------------------------------------------------
def sph_to_xyz(ra, dec):
    ra = np.radians(ra)
    dec = np.radians(dec)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])

def angsep(ra1, dec1, ra2, dec2):
    ra1 = radians(ra1); dec1 = radians(dec1)
    ra2 = radians(ra2); dec2 = radians(dec2)
    return np.degrees(
        acos(
            sin(dec1)*sin(dec2) +
            cos(dec1)*cos(dec2)*cos(ra1-ra2)
        )
    )

# --------------------------------------------------------
# load unified csv
# --------------------------------------------------------
def load_unified(path):
    df = pd.read_csv(path)
    RA  = df["ra"].values
    Dec = df["dec"].values
    return df, RA, Dec

# --------------------------------------------------------
# load ASKAP FITS position centers
# --------------------------------------------------------
def load_askap_fits_positions():
    RAs = []
    Decs = []
    files = [f for f in os.listdir(ASKAP_FITS_DIR) if f.lower().endswith(".fits")]
    for f in files:
        hd = fits.open(os.path.join(ASKAP_FITS_DIR, f))
        h = hd[0].header
        RAs.append(float(h["CRVAL1"]))
        Decs.append(float(h["CRVAL2"]))
        hd.close()
    return np.array(RAs), np.array(Decs)

# --------------------------------------------------------
# match FITS centers to unified FRBs
# --------------------------------------------------------
def match_fits_to_frbs(fits_RA, fits_Dec, RA, Dec):
    is_askap = np.zeros(len(RA), dtype=bool)
    for fra, fdec in zip(fits_RA, fits_Dec):
        seps = np.sqrt((RA - fra)**2 + (Dec - fdec)**2)  # good enough for < few degrees
        idx = np.argmin(seps)
        if seps[idx] <= MATCH_TOL:
            is_askap[idx] = True
    return is_askap

# --------------------------------------------------------
# shell membership
# --------------------------------------------------------
def compute_shell_angles(theta):
    """ shell1 = theta in [0,30], shell2 = theta in [30,60] """
    s1 = (theta >= 0) & (theta < 30)
    s2 = (theta >= 30) & (theta < 60)
    return s1, s2

# --------------------------------------------------------
# phase-coherence statistic
# --------------------------------------------------------
def compute_phase_coherence(phi1, phi2):
    return abs(np.mean(phi1) - np.mean(phi2))

# --------------------------------------------------------
# main
# --------------------------------------------------------
def main(path):
    df, RA, Dec = load_unified(path)
    theta = df["theta_unified"].values
    phi   = df["phi_unified"].values

    # load askap fits
    fits_RA, fits_Dec = load_askap_fits_positions()
    is_askap = match_fits_to_frbs(fits_RA, fits_Dec, RA, Dec)

    # split
    A = is_askap
    B = ~is_askap

    # shells
    s1, s2 = compute_shell_angles(theta)

    # compute real
    def run_subset(mask):
        idx1 = mask & s1
        idx2 = mask & s2
        if idx1.sum() == 0 or idx2.sum() == 0:
            return np.nan
        return compute_phase_coherence(phi[idx1], phi[idx2])

    C_real_askap    = run_subset(A)
    C_real_nonaskap = run_subset(B)

    # nulls
    n_mc = 2000
    C_null_A = []
    C_null_B = []

    for _ in tqdm(range(n_mc)):
        # random rotation in phi only (phase scramble)
        phi_rand = np.random.uniform(0, 360, size=len(phi))
        C_null_A.append(run_subset(A))
        C_null_B.append(run_subset(B))

    C_null_A = np.array(C_null_A)
    C_null_B = np.array(C_null_B)

    # p-values
    def pval(real, null):
        if np.isnan(real): return 1.0
        return np.mean(null >= real)

    p_A = pval(C_real_askap,    C_null_A)
    p_B = pval(C_real_nonaskap, C_null_B)

    print("===============================================================")
    print("FRB REMNANT-TIME CROSS-SHELL PHASE TEST (73C — ASKAP split)")
    print("===============================================================")
    print(f"ASKAP count     = {A.sum()}")
    print(f"non-ASKAP count = {B.sum()}")
    print("---------------------------------------------------------------")
    print(f"ASKAP:     C_real={C_real_askap}, null_mean={np.nanmean(C_null_A)}, p={p_A}")
    print(f"nonASKAP:  C_real={C_real_nonaskap}, null_mean={np.nanmean(C_null_B)}, p={p_B}")
    print("---------------------------------------------------------------")
    print("interpretation:")
    print("  low p  -> phase coherence in subset")
    print("  high p -> subset consistent with isotropy")
    print("===============================================================")
    print("test 73C complete.")
    print("===============================================================")

if __name__ == "__main__":
    main(sys.argv[1])
