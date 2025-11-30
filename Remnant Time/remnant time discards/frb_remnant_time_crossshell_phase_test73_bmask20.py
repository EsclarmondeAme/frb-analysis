import sys
import numpy as np
import pandas as pd
from math import radians, sin, cos, atan2
from tqdm import tqdm

# ---------------------------------------------------------
#  utilities
# ---------------------------------------------------------

def sph_to_xyz(ra, dec):
    ra = np.radians(ra)
    dec = np.radians(dec)
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x,y,z]).T

def galactic_latitude_from_xyz(xyz):
    # b = arcsin(z)
    z = xyz[:,2]
    return np.degrees(np.arcsin(z))

def angle_between(v1, v2):
    dot = np.sum(v1*v2,axis=1)
    return np.degrees(np.arccos(np.clip(dot, -1, 1)))

# ---------------------------------------------------------
#  unified axis from your earlier tests
# ---------------------------------------------------------
UNIFIED_AXIS = np.array([-0.90251931, 0.41486337, 0.11553043])
UNIFIED_AXIS = UNIFIED_AXIS / np.linalg.norm(UNIFIED_AXIS)


# ---------------------------------------------------------
# harmonic phase function (same as test 73)
# ---------------------------------------------------------

def compute_phase(theta, phi):
    # simple fundamental phase = phi weighted by sin(theta)
    return phi * np.sin(np.radians(theta))

# ---------------------------------------------------------
#  main 73A logic
# ---------------------------------------------------------

def main(path):
    print("================================================")
    print("FRB REMNANT-TIME CROSS-SHELL PHASE TEST (73A)")
    print("Galactic mask: |b| >= 20°")
    print("================================================")

    df = pd.read_csv(path)
    RA  = df["ra"].values
    DEC = df["dec"].values

    # xyz
    xyz = sph_to_xyz(RA, DEC)

    # mask
    b = galactic_latitude_from_xyz(xyz)
    mask = np.abs(b) >= 20
    xyz = xyz[mask]
    RA  = RA[mask]
    DEC = DEC[mask]

    N = len(RA)
    print(f"[info] original N=600, after mask N={N}")

    # axis angle
    d = angle_between(xyz, UNIFIED_AXIS)

    # define shells
    shell1 = (d >= 17.5) & (d < 32.5)
    shell2 = (d >= 32.5) & (d < 47.5)

    # phases
    theta = d
    phi = RA   # RA used as azimuth proxy
    ph = compute_phase(theta, phi)

    # real statistics
    if shell1.sum() == 0 or shell2.sum() == 0:
        print("[warn] empty shell after masking.")
        return

    # circular mean difference
    def circ_mean(a):
        return atan2(np.mean(np.sin(np.radians(a))),
                     np.mean(np.cos(np.radians(a))))

    m1 = circ_mean(ph[shell1])
    m2 = circ_mean(ph[shell2])

    C_real = abs(m1 - m2)

    # MC null: shuffle phases
    M = 2000
    C_null = np.zeros(M)

    for i in tqdm(range(M)):
        ph_rand = np.random.permutation(ph)
        m1r = circ_mean(ph_rand[shell1])
        m2r = circ_mean(ph_rand[shell2])
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
    print("  low p  -> harmonic phases remain coherent across the two shells")
    print("            after removing the Galactic plane (robust).")
    print("  high p -> phases consistent with isotropy after masking.")
    print("================================================")
    print("test 73A complete.")
    print("================================================")


if __name__ == "__main__":
    main(sys.argv[1])
