import sys
import numpy as np
import pandas as pd
from math import atan2
from tqdm import tqdm
from astropy.coordinates import SkyCoord
import astropy.units as u

# ---------------------------------------------------------
# utilities
# ---------------------------------------------------------

def sph_to_xyz(ra, dec):
    ra = np.radians(ra)
    dec = np.radians(dec)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x,y,z]).T

def angle_between(v1, v2):
    dot = np.sum(v1*v2, axis=1)
    return np.degrees(np.arccos(np.clip(dot, -1, 1)))

def circ_mean_deg(a_deg):
    a = np.radians(a_deg)
    return atan2(np.mean(np.sin(a)), np.mean(np.cos(a)))

def compute_phase(theta, phi):
    # identical functional form as 73
    return phi * np.sin(np.radians(theta))

# ---------------------------------------------------------
# unified axis (fixed)
# ---------------------------------------------------------
UNIFIED_AXIS = np.array([-0.90251931, 0.41486337, 0.11553043])
UNIFIED_AXIS = UNIFIED_AXIS / np.linalg.norm(UNIFIED_AXIS)


# ---------------------------------------------------------
# main 73B logic
# ---------------------------------------------------------

def main(path):
    print("================================================")
    print("FRB REMNANT-TIME CROSS-SHELL PHASE TEST (73B)")
    print("Supergalactic mask: |SGB| >= 20°")
    print("================================================")

    df = pd.read_csv(path)
    RA  = df["ra"].values
    DEC = df["dec"].values

    # convert to SGL/SGB
    c = SkyCoord(ra=RA*u.deg, dec=DEC*u.deg, frame="icrs").transform_to("supergalactic")
    SGB = c.sgb.value

    mask = np.abs(SGB) >= 20
    RA  = RA[mask]
    DEC = DEC[mask]

    N = len(RA)
    print(f"[info] original N=600, after SGB mask N={N}")

    xyz = sph_to_xyz(RA, DEC)
    d = angle_between(xyz, UNIFIED_AXIS)

    # shells
    shell1 = (d >= 17.5) & (d < 32.5)
    shell2 = (d >= 32.5) & (d < 47.5)

    if shell1.sum() == 0 or shell2.sum() == 0:
        print("[warn] empty shell after masking – cannot compute.")
        return

    theta = d
    phi = RA
    ph = compute_phase(theta, phi)

    # real
    m1 = circ_mean_deg(ph[shell1])
    m2 = circ_mean_deg(ph[shell2])
    C_real = abs(m1 - m2)

    # MC null
    M = 2000
    C_null = np.zeros(M)

    for i in tqdm(range(M)):
        rnd = np.random.permutation(ph)
        m1r = circ_mean_deg(rnd[shell1])
        m2r = circ_mean_deg(rnd[shell2])
        C_null[i] = abs(m1r - m2r)

    mu = np.mean(C_null)
    sd = np.std(C_null)
    p  = np.mean(C_null >= C_real)

    print("------------------------------------------------")
    print(f"Δφ_shell1     = {m1:.6f}")
    print(f"Δφ_shell2     = {m2:.6f}")
    print(f"C_real        = {C_real:.6f}")
    print("------------------------------------------------")
    print(f"null mean C   = {mu:.6f}")
    print(f"null std C    = {sd:.6f}")
    print(f"p-value       = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> phase coherence survives SGB masking (robust).")
    print("  high p -> coherence disappears; consistent with isotropy.")
    print("================================================")
    print("test 73B complete.")
    print("================================================")


if __name__ == "__main__":
    main(sys.argv[1])
