import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from tqdm import tqdm
import sys

"""
TEST 45 — FRB PLANCK-GRAIN MICRO-CLUSTERING TEST
------------------------------------------------
Goal:
    Detect ultra-small angular-scale clustering patterns which would indicate
    granular structure in the underlying spacetime / emission geometry.

Inputs:
    - FRB catalog with 'theta_unified' and 'phi_unified' columns.

Output:
    - micro-scale pair separation spectrum
    - Allan variance over angular windows
    - null distribution from isotropic MC
"""

# ------------------------------------------------------------
# angular separation on the sphere
# ------------------------------------------------------------
def angsep(t1, p1, t2, p2):
    dphi = np.deg2rad(p1 - p2)
    t1 = np.deg2rad(t1)
    t2 = np.deg2rad(t2)
    return np.rad2deg(np.arccos(
        np.sin(t1)*np.sin(t2) + np.cos(t1)*np.cos(t2)*np.cos(dphi)
    ))


def main():
    if len(sys.argv) < 2:
        print("usage: python frb_planck_grain_microclustering_test45.py frbs_unified.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    print("detected FRBs:", len(df))

    theta = df["theta_unified"].values
    phi   = df["phi_unified"].values
    n = len(theta)

    # ------------------------------------------------------------
    # compute all pairwise separations
    # ------------------------------------------------------------
    coords = np.vstack([theta, phi]).T
    seps = []
    print("computing pairwise separations...")
    for i in range(n):
        for j in range(i+1, n):
            seps.append(angsep(theta[i], phi[i], theta[j], phi[j]))
    seps = np.array(seps)

    # ------------------------------------------------------------
    # MICRO-SCALE SPECTRUM
    # ------------------------------------------------------------
    bins = np.linspace(0, 2, 200)  # 0–2 deg resolution
    hist_real, _ = np.histogram(seps, bins=bins)

    # ------------------------------------------------------------
    # ALLAN VARIANCE ON MICRO-SCALE BINS
    # ------------------------------------------------------------
    diffs = np.diff(hist_real)
    allan_var_real = 0.5 * np.mean(diffs**2)

    # ------------------------------------------------------------
    # Null Monte Carlo
    # ------------------------------------------------------------
    print("running Monte Carlo isotropic (N=5000)...")
    N_MC = 5000

    allan_null = []
    minsep_null = []

    for _ in tqdm(range(N_MC)):
        # random sky: keep number of points fixed
        t_rand = np.degrees(np.arccos(1 - 2*np.random.rand(n)))   # random theta
        p_rand = np.random.rand(n)*360                             # random phi

        # random separations
        seps_r = []
        for i in range(n):
            for j in range(i+1, n):
                seps_r.append(angsep(t_rand[i], p_rand[i], t_rand[j], p_rand[j]))
        seps_r = np.array(seps_r)

        # Allan variance for null
        hist_r, _ = np.histogram(seps_r, bins=bins)
        diffs_r = np.diff(hist_r)
        allan_null.append(0.5 * np.mean(diffs_r**2))

        # min-separation test
        minsep_null.append(seps_r.min())

    allan_null = np.array(allan_null)
    minsep_null = np.array(minsep_null)

    # ------------------------------------------------------------
    # Observed values
    # ------------------------------------------------------------
    allan_obs = allan_var_real
    minsep_obs = seps.min()

    # ------------------------------------------------------------
    # p-values
    # ------------------------------------------------------------
    p_allan = np.mean(allan_null <= allan_obs)
    p_minsep = np.mean(minsep_null <= minsep_obs)

    # ------------------------------------------------------------
    # print results
    # ------------------------------------------------------------
    print("="*68)
    print("FRB PLANCK-GRAIN MICRO-CLUSTERING TEST (TEST 45)")
    print("="*68)
    print(f"N_FRB = {n}")
    print("------------------------------------------------------------")
    print(f"observed Allan variance: {allan_obs:.6f}")
    print(f"null mean Allan variance: {allan_null.mean():.6f}")
    print(f"p-value (granularity): {p_allan:.6f}")
    print("------------------------------------------------------------")
    print(f"observed minimum separation: {minsep_obs:.6f} deg")
    print(f"null mean minimum separation: {minsep_null.mean():.6f}")
    print(f"p-value (min-sep anomaly): {p_minsep:.6f}")
    print("------------------------------------------------------------")
    print("interpretation:")
    print(" - low p(Allan): evidence for discrete angular lattice / micro-grain pattern.")
    print(" - low p(min-sep): evidence for forbidden zones / quantized spacing.")
    print("="*68)
    print("test 45 complete.")
    print("="*68)


if __name__ == "__main__":
    main()
