#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frb remnant time shell asymmetry test (test 71)
----------------------------------------------
idea:
  we already see evidence for preferred angular shells around the unified axis
  (roughly at 25 deg and 40 deg from the axis). this test asks whether those
  shells fall preferentially on one side of a remnant-time field aligned with
  the same axis.

  define a remnant field R(x) = x · n, where x is a galactic unit vector and n
  is the unified frb axis. R>0 and R<0 define two time-hemispheres.

  in each shell band around the axis, we count how many frbs lie in the
  forward-time hemisphere (R>0) and how many in the backward-time hemisphere
  (R<0). we then ask whether the total shell asymmetry is stronger than one
  would expect for isotropic skies.

interpretation:
  low p  -> the axis-aligned shells prefer one remnant-time sign more than
            random isotropic skies typically do
  high p -> shell occupancy across time hemispheres is consistent with isotropy
"""

import numpy as np
import csv
import sys
from tqdm import tqdm

# ============================================================
# utilities: catalog and coordinates
# ============================================================

def detect_columns(fieldnames):
    """detect ra/dec column names automatically."""
    low = [c.lower() for c in fieldnames]

    def find(*candidates):
        for c in candidates:
            if c.lower() in low:
                return fieldnames[low.index(c.lower())]
        return None

    ra_key  = find("ra_deg","ra","raj2000","ra_deg_","ra (deg)")
    dec_key = find("dec_deg","dec","dej2000","dec_deg","dec (deg)")

    if ra_key is None or dec_key is None:
        raise KeyError("could not detect ra/dec column names")

    return ra_key, dec_key


def load_catalog(path):
    """load ra/dec from frb csv."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        ra_key, dec_key = detect_columns(fields)

        RA, Dec = [], []
        for row in reader:
            RA.append(float(row[ra_key]))
            Dec.append(float(row[dec_key]))

    return np.array(RA), np.array(Dec)


def radec_to_equatorial_xyz(RA, Dec):
    """convert ra/dec to j2000 equatorial unit vectors."""
    RA  = np.radians(RA)
    Dec = np.radians(Dec)

    x = np.cos(Dec) * np.cos(RA)
    y = np.cos(Dec) * np.sin(RA)
    z = np.sin(Dec)
    return np.vstack([x, y, z]).T


def equatorial_to_galactic_matrix():
    """
    j2000 equatorial -> galactic rotation matrix.
    standard iau 2000 values.
    """
    return np.array([
        [-0.054875539390, -0.873437104725, -0.483834991775],
        [ 0.494109453633, -0.444829594298,  0.746982248696],
        [-0.867666135681, -0.198076389622,  0.455983794523],
    ])


def radec_to_galactic_xyz(RA, Dec):
    """convert ra/dec (degrees) to galactic unit vectors."""
    Xeq = radec_to_equatorial_xyz(RA, Dec)
    M = equatorial_to_galactic_matrix()
    Xgal = Xeq @ M.T
    norms = np.linalg.norm(Xgal, axis=1, keepdims=True) + 1e-15
    return Xgal / norms


def galactic_lb_to_xyz(l_deg, b_deg):
    """convert galactic (l,b) in degrees to unit xyz."""
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    x = np.cos(b) * np.cos(l)
    y = np.cos(b) * np.sin(l)
    z = np.sin(b)
    v = np.array([x, y, z], dtype=float)
    return v / (np.linalg.norm(v) + 1e-15)


# ============================================================
# remnant field, shells, and mc
# ============================================================

def angle_from_axis(X, axis_vec):
    """
    angle in degrees between each unit vector x_i in X and axis_vec.
    """
    axis_vec = axis_vec / (np.linalg.norm(axis_vec) + 1e-15)
    dots = X @ axis_vec
    np.clip(dots, -1.0, 1.0, out=dots)
    return np.degrees(np.arccos(dots))


def remnant_sign(X, axis_vec):
    """
    remnant time sign array: +1 for R>0, -1 for R<0, with R = x · n.
    """
    axis_vec = axis_vec / (np.linalg.norm(axis_vec) + 1e-15)
    R = X @ axis_vec
    sign = np.ones_like(R, dtype=int)
    sign[R < 0] = -1
    return sign


def shell_counts(theta_deg, sign, shell_min, shell_max):
    """
    count frbs in a shell [shell_min, shell_max] degrees from the axis,
    split by remnant-time sign.
    """
    mask_shell = (theta_deg >= shell_min) & (theta_deg < shell_max)
    in_shell = sign[mask_shell]
    n_plus = int(np.sum(in_shell > 0))
    n_minus = int(np.sum(in_shell < 0))
    return n_plus, n_minus


def total_shell_asymmetry(theta_deg, sign,
                          shell1=(17.5, 32.5),
                          shell2=(32.5, 47.5)):
    """
    compute total shell asymmetry statistic for two shell bands.

    returns
    -------
    stats : dict with counts and asymmetries
    S_total : |delta_total| over both shells combined
    """
    s1_min, s1_max = shell1
    s2_min, s2_max = shell2

    n1_plus, n1_minus = shell_counts(theta_deg, sign, s1_min, s1_max)
    n2_plus, n2_minus = shell_counts(theta_deg, sign, s2_min, s2_max)

    delta1 = n1_plus - n1_minus
    delta2 = n2_plus - n2_minus
    delta_total = delta1 + delta2

    S1 = abs(delta1)
    S2 = abs(delta2)
    S_total = abs(delta_total)

    stats = {
        "shell1_range": (s1_min, s1_max),
        "shell2_range": (s2_min, s2_max),
        "shell1_n_plus": n1_plus,
        "shell1_n_minus": n1_minus,
        "shell1_delta": delta1,
        "shell1_S": S1,
        "shell2_n_plus": n2_plus,
        "shell2_n_minus": n2_minus,
        "shell2_delta": delta2,
        "shell2_S": S2,
        "delta_total": delta_total,
        "S_total": S_total,
    }
    return stats, S_total


def random_isotropic(N):
    """draw N isotropic points on the sphere, returned as galactic xyz."""
    u = np.random.uniform(-1.0, 1.0, size=N)
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=N)
    sin_theta = np.sqrt(1.0 - u * u)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = u
    X = np.vstack([x, y, z]).T
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-15
    return X / norms


# ============================================================
# main test logic
# ============================================================

def main(path):
    print("[info] loading frb catalog...")
    RA, Dec = load_catalog(path)
    N = len(RA)
    print(f"[info] N_FRB = {N}")

    print("[info] converting to galactic coordinates...")
    Xgal = radec_to_galactic_xyz(RA, Dec)

    # unified axis from paper, in galactic (l,b) degrees
    l_axis = 159.8
    b_axis = -0.5
    axis_unified = galactic_lb_to_xyz(l_axis, b_axis)

    print("[info] computing shell angles and remnant-time signs...")
    theta_deg = angle_from_axis(Xgal, axis_unified)
    sign = remnant_sign(Xgal, axis_unified)

    print("[info] measuring real shell asymmetry...")
    stats_real, S_real = total_shell_asymmetry(theta_deg, sign)

    print("[info] building monte carlo null (isotropic skies)...")
    S_null = []
    for _ in tqdm(range(2000), desc="MC shells"):
        X_mc = random_isotropic(N)
        theta_mc = angle_from_axis(X_mc, axis_unified)
        sign_mc = remnant_sign(X_mc, axis_unified)
        _, S_mc = total_shell_asymmetry(theta_mc, sign_mc)
        S_null.append(S_mc)
    S_null = np.array(S_null, dtype=float)

    mu = float(np.mean(S_null))
    sd = float(np.std(S_null))
    # p-value: how often null produces S >= S_real
    p = (1.0 + np.sum(S_null >= S_real)) / (len(S_null) + 1.0)

    print("================================================")
    print(" frb remnant time shell asymmetry test (test 71)")
    print("================================================")
    print("shell definitions (axis distance in degrees):")
    print(f"  shell 1: {stats_real['shell1_range'][0]:.1f} – {stats_real['shell1_range'][1]:.1f}")
    print(f"  shell 2: {stats_real['shell2_range'][0]:.1f} – {stats_real['shell2_range'][1]:.1f}")
    print("------------------------------------------------")
    print("real shell counts by remnant-time hemisphere:")
    print(f"  shell 1: N_plus={stats_real['shell1_n_plus']}, "
          f"N_minus={stats_real['shell1_n_minus']}, "
          f"delta={stats_real['shell1_delta']}, "
          f"S1=|delta|={stats_real['shell1_S']}")
    print(f"  shell 2: N_plus={stats_real['shell2_n_plus']}, "
          f"N_minus={stats_real['shell2_n_minus']}, "
          f"delta={stats_real['shell2_delta']}, "
          f"S2=|delta|={stats_real['shell2_S']}")
    print("------------------------------------------------")
    print(f"combined delta_total   = {stats_real['delta_total']}")
    print(f"S_total (|delta_total|)= {S_real}")
    print("------------------------------------------------")
    print(f"null mean S_total      = {mu:.3f}")
    print(f"null std S_total       = {sd:.3f}")
    print(f"p-value (shells)       = {p:.6f}")
    print("------------------------------------------------")
    print("interpretation:")
    print("  - low p  -> the 25°/40° axis shells prefer one remnant-time")
    print("             hemisphere more strongly than isotropic skies usually do.")
    print("  - high p -> the split of shell frbs between forward and backward")
    print("             remnant-time hemispheres is consistent with isotropy.")
    print("================================================")
    print("test 71 complete.")
    print("================================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python frb_remnant_time_shell_asymmetry_test71.py frbs_unified.csv")
        sys.exit(1)
    main(sys.argv[1])
