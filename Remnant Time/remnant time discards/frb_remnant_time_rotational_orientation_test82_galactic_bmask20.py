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











def main(path):

    print("================================================")
    print(" Test 82A — rotational orientation under |b|>=20°")
    print("================================================")

    # load catalog and convert to galactic xyz
    RA, Dec = load_catalog(path)
    X = radec_to_galactic_xyz(RA, Dec)
    N = len(X)

    # galactic latitude from xyz
    b = np.degrees(np.arcsin(np.clip(X[:, 2], -1.0, 1.0)))
    mask = np.abs(b) >= 20.0
    X = X[mask]

    print(f"[info] N original = {N}")
    print(f"[info] N after |b|>=20 mask = {len(X)}")

    if len(X) < 40:
        print("[error] too few objects after mask")
        return

    # unified axis
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
    print("low p  -> orientation tags survive |b| mask")
    print("high p -> symmetric; consistent with isotropy")
    print("================================================")

if __name__ == "__main__":
    main(sys.argv[1])
