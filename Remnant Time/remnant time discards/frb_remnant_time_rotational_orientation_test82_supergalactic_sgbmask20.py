#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from tqdm import tqdm


from frb_remnant_time_rotational_orientation_test82 import (
    load_catalog,
    radec_to_galactic_xyz,
    gal_lb_xyz,
    remnant_sign,
    local_orientations,
)

np.random.seed(0)


# supergalactic conversion (same constants as other tests)
SGL_NP_RA  = np.radians(283.25)
SGL_NP_DEC = np.radians(15.70)
SGL_LON0   = np.radians(47.37)

def eq_to_sgb(ra_deg, dec_deg):
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    sinB = (np.sin(dec)*np.sin(SGL_NP_DEC) +
            np.cos(dec)*np.cos(SGL_NP_DEC)*np.cos(ra - SGL_NP_RA))
    B = np.arcsin(sinB)
    y = np.cos(dec)*np.sin(ra - SGL_NP_RA)
    x = (np.sin(dec)*np.cos(SGL_NP_DEC) -
         np.cos(dec)*np.sin(SGL_NP_DEC)*np.cos(ra - SGL_NP_RA))
    L = np.arctan2(y, x) + SGL_LON0
    return np.degrees(L)%360.0, np.degrees(B)

def main(path):

    print("================================================")
    print(" Test 82B — rotational orientation under |SGB|>=20°")
    print("================================================")

    RA, Dec = load_catalog(path)
    N = len(RA)

    _, SGB = eq_to_sgb(RA, Dec)
    mask = np.abs(SGB) >= 20.0

    RA = RA[mask]
    Dec = Dec[mask]

    print(f"[info] N original = {N}")
    print(f"[info] N after |SGB|>=20 mask = {len(RA)}")

    if len(RA) < 40:
        print("[error] too few objects after mask")
        return

    X = radec_to_galactic_xyz(RA, Dec)
    axis = gal_lb_xyz(159.8, -0.5)

    print("[info] computing remnant signs...")
    s = remnant_sign(X, axis)

    print("[info] computing local orientations...")
    psi, valid = local_orientations(X, k=20, anisotropy_thresh=0.1)

    z = np.exp(2j * psi)
    z[~valid] = 0.0 + 0.0j

    z_pos = z[(s > 0) & valid]
    z_neg = z[(s < 0) & valid]

    if len(z_pos) < 10 or len(z_neg) < 10:
        print("[error] not enough valid orientations in one hemisphere.")
        return

    S_pos = np.mean(z_pos)
    S_neg = np.mean(z_neg)
    A_real = abs(S_pos - S_neg)

    print("[info] building null (2000 shuffles)...")
    null = []
    NMC = 2000

    for _ in tqdm(range(NMC)):
        sh = np.copy(s)
        np.random.shuffle(sh)

        z_p = z[(sh > 0) & valid]
        z_n = z[(sh < 0) & valid]

        if len(z_p) < 10 or len(z_n) < 10:
            null.append(0.0)
            continue

        Sp = np.mean(z_p)
        Sn = np.mean(z_n)
        null.append(abs(Sp - Sn))

    null = np.array(null)
    mu = float(np.mean(null))
    sd = float(np.std(null))
    p = (1.0 + np.sum(null >= A_real)) / (NMC + 1.0)

    print("------------------------------------------------")
    print(f"S_pos (complex)        = {S_pos}")
    print(f"S_neg (complex)        = {S_neg}")
    print(f"A_real                 = {A_real:.6f}")
    print("------------------------------------------------")
    print(f"null mean A            = {mu:.6f}")
    print(f"null std A             = {sd:.6f}")
    print(f"p-value                = {p:.6f}")
    print("------------------------------------------------")
    print("low p  -> orientation tags survive SGB mask")
    print("high p -> symmetric; consistent with isotropy")
    print("================================================")

if __name__ == "__main__":
    main(sys.argv[1])
