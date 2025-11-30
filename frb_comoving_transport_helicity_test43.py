import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from astropy.cosmology import Planck18 as cosmo
from tqdm import tqdm
import sys

# -------------------------------------------------------------
# helical model
# -------------------------------------------------------------
def helix(theta, phi0, k):
    return phi0 + k * theta

# -------------------------------------------------------------
# main routine
# -------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: python frb_comoving_transport_helicity_test43.py frbs_unified.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    print("============================================================")
    print(" FRB COMOVING-SPACE HELICITY TRANSPORT TEST (TEST 43)")
    print("============================================================")

    # ---------------------------------------------------------
    # 1. read observed unified-axis angles
    # ---------------------------------------------------------
    theta_obs = np.deg2rad(df["theta_unified"].values)
    phi_obs   = np.deg2rad(df["phi_unified"].values)

    # ---------------------------------------------------------
    # 2. compute comoving coordinates from RA/DEC/z
    # ---------------------------------------------------------
    ra  = np.deg2rad(df["ra"].values)
    dec = np.deg2rad(df["dec"].values)
    z   = df["z_est"].values

    # comoving distance [Mpc]
    dist = cosmo.comoving_distance(z).value

    # cartesian coordinates
    X = dist * np.cos(dec) * np.cos(ra)
    Y = dist * np.cos(dec) * np.sin(ra)
    Z = dist * np.sin(dec)

    # convert back to spherical angles in comoving frame
    r = np.sqrt(X*X + Y*Y + Z*Z)
    theta_c = np.arccos(Z / r)               # polar angle
    phi_c   = np.arctan2(Y, X)               # azimuthal

    # convert to degrees
    theta_c_deg = np.rad2deg(theta_c)
    phi_c_deg   = np.rad2deg(phi_c)

    # ---------------------------------------------------------
    # 3. fit helicity in observed frame
    # ---------------------------------------------------------
    popt_obs, _ = curve_fit(helix, theta_obs, phi_obs)
    phi0_obs, k_obs = popt_obs

    # ---------------------------------------------------------
    # 4. fit helicity in comoving frame
    # ---------------------------------------------------------
    popt_com, _ = curve_fit(helix, theta_c, phi_c)
    phi0_com, k_com = popt_com

    # ---------------------------------------------------------
    # 5. Monte Carlo significance test (comoving)
    # ---------------------------------------------------------
    Nmc = 20000
    k_null = np.zeros(Nmc)

    for i in tqdm(range(Nmc)):
        phi_rand = np.random.permutation(phi_c)
        try:
            popt, _ = curve_fit(helix, theta_c, phi_rand, maxfev=10000)
            k_null[i] = abs(popt[1])
        except:
            k_null[i] = 0.0

    p_value = np.mean(k_null >= abs(k_com))

    # ---------------------------------------------------------
    # 6. print results
    # ---------------------------------------------------------
    print("------------------------------------------------------------")
    print(" observed-frame helicity:")
    print(f"   k_obs = {k_obs:.5f} rad/rad")
    print("")
    print(" comoving-frame helicity:")
    print(f"   k_com = {k_com:.5f} rad/rad")
    print(f"   null mean |k| = {np.mean(k_null):.5f}")
    print(f"   null std  |k| = {np.std(k_null):.5f}")
    print(f"   p-value   = {p_value:.6f}")
    print("------------------------------------------------------------")

    # qualitative interpretation
    if p_value < 0.05 and abs(k_com) > abs(k_obs):
        meaning = "helicity becomes *stronger* in comoving space → physical"
    elif p_value < 0.05:
        meaning = "helicity remains significant in comoving space → physical"
    else:
        meaning = "helicity weakens under deprojection → projection/geometry"

    print(" interpretation:")
    print("  - compares observed sky to true comoving structure")
    print(f"  - result: {meaning}")
    print("============================================================")
    print(" test 43 complete.")
    print("============================================================")


if __name__ == "__main__":
    main()
