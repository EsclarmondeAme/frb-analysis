#!/usr/bin/env python3
# frb_axisymmetry_test.py
# -------------------------------------------------------------------
# test whether FRB excess around the unified axis depends only on
# polar angle theta (axisymmetric cone / shells) or also on azimuth phi
# (pyramid-like faces / directional patches).
# -------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

# unified axis (from previous best-fit)
UNIFIED_L = 159.85
UNIFIED_B = -0.51

# cone shells (deg from axis)
SHELL_BOUNDS = np.array([0.0, 10.0, 25.0, 40.0, 140.0])
SHELL_LABELS = ["inner", "mid1", "mid2", "outer"]

# number of azimuth bins around the axis
N_PHI_BINS = 12  # 30° bins


def main():
    print("="*70)
    print("FRB AXISYMMETRY TEST")
    print("Testing azimuthal (phi) structure around unified axis")
    print("="*70)

    # --------------------------------------------------------------
    # 1. load catalog and transform to galactic
    # --------------------------------------------------------------
    df = pd.read_csv("frbs.csv")

    coords_icrs = SkyCoord(
        ra=df["ra"].values * u.deg,
        dec=df["dec"].values * u.deg,
        frame="icrs"
    )
    coords_gal = coords_icrs.galactic

    unif = SkyCoord(l=UNIFIED_L * u.deg,
                    b=UNIFIED_B * u.deg,
                    frame="galactic")

    # polar angle from axis
    theta = coords_gal.separation(unif).deg

    # azimuth around axis: use position_angle
    # phi is angle from FRB towards unified axis, measured east of north
    # we only care about relative azimuth, so this is fine
    phi = coords_gal.position_angle(unif).deg  # 0–360
    phi = phi % 360.0

    print("\nSECTION 1 — DATA")
    print("------------------------------------------------------------")
    print(f"Total FRBs: {len(df)}")
    print(f"Theta range: {theta.min():.2f}° – {theta.max():.2f}°")
    print(f"Unified axis: l={UNIFIED_L:.2f}°, b={UNIFIED_B:.2f}°")

    # --------------------------------------------------------------
    # 2. bin in shells and azimuth
    # --------------------------------------------------------------
    print("\nSECTION 2 — SHELL AND AZIMUTH BINNING")
    print("------------------------------------------------------------")

    shell_idx = np.digitize(theta, SHELL_BOUNDS) - 1
    valid = (shell_idx >= 0) & (shell_idx < len(SHELL_LABELS))
    theta = theta[valid]
    phi = phi[valid]
    shell_idx = shell_idx[valid]

    phi_edges = np.linspace(0, 360, N_PHI_BINS + 1)

    total_counts = []
    shell_chi2 = []
    shell_pseudo = []

    print(f"Using {len(theta)} FRBs within 0–140° of axis")
    print(f"Shell boundaries (deg): {SHELL_BOUNDS.tolist()}")
    print(f"Azimuth bins: {N_PHI_BINS} (each {360//N_PHI_BINS}°)")

    for s in range(len(SHELL_LABELS)):
        mask = (shell_idx == s)
        n_shell = np.sum(mask)
        if n_shell == 0:
            total_counts.append(0)
            shell_chi2.append(0.0)
            shell_pseudo.append(np.zeros(N_PHI_BINS))
            continue

        phi_shell = phi[mask]
        counts, _ = np.histogram(phi_shell, bins=phi_edges)
        total_counts.append(n_shell)

        expected = n_shell / N_PHI_BINS
        chi2 = np.sum((counts - expected) ** 2 / (expected + 1e-8))
        shell_chi2.append(chi2)
        shell_pseudo.append(counts / expected if expected > 0 else np.zeros_like(counts))

        print(f"{SHELL_LABELS[s]:5s} shell {SHELL_BOUNDS[s]:5.1f}°–{SHELL_BOUNDS[s+1]:5.1f}°:")
        print(f"   n = {n_shell:3d}")
        print(f"   chi²(phi-uniform) = {chi2:.2f}")
        print(f"   max(count/expected) = {np.max(counts/expected):.2f}")
        print(f"   min(count/expected) = {np.min(counts/expected):.2f}")

    shell_chi2 = np.array(shell_chi2)
    total_chi2 = np.sum(shell_chi2)
    dof = len(SHELL_LABELS) * (N_PHI_BINS - 1)

    print("\nSECTION 3 — GLOBAL AZIMUTHAL χ²")
    print("------------------------------------------------------------")
    print(f"Total χ² over all shells: {total_chi2:.2f} (dof={dof})")
    print("This is before Monte Carlo calibration.")

    # --------------------------------------------------------------
    # 4. Monte Carlo: shuffle phi values, keep theta and shells fixed
    # --------------------------------------------------------------
    print("\nSECTION 4 — MONTE CARLO NULL (10,000 shuffles)")
    print("------------------------------------------------------------")

    n_sim = 10000
    chi2_null = np.zeros(n_sim, dtype=float)
    n_total = len(theta)

    for i in range(n_sim):
        # shuffle phi among all FRBs
        phi_shuf = np.random.permutation(phi)

        chi2_accum = 0.0
        for s in range(len(SHELL_LABELS)):
            mask = (shell_idx == s)
            n_shell = np.sum(mask)
            if n_shell == 0:
                continue

            phi_shell = phi_shuf[mask]
            counts, _ = np.histogram(phi_shell, bins=phi_edges)
            expected = n_shell / N_PHI_BINS
            chi2_s = np.sum((counts - expected) ** 2 / (expected + 1e-8))
            chi2_accum += chi2_s

        chi2_null[i] = chi2_accum

        if (i+1) % 2000 == 0:
            print(f"   completed {i+1}/{n_sim}")

    p_mc = np.mean(chi2_null >= total_chi2)
    mean_null = np.mean(chi2_null)
    p95 = np.percentile(chi2_null, 95)

    print(f"\nNull mean χ²: {mean_null:.2f}")
    print(f"Null 95% χ²:  {p95:.2f}")
    print(f"Observed χ²:  {total_chi2:.2f}")
    print(f"Monte Carlo p-value: {p_mc:.4f}")

    # --------------------------------------------------------------
    # 5. verdict
    # --------------------------------------------------------------
    print("\nSECTION 5 — VERDICT")
    print("------------------------------------------------------------")

    if p_mc > 0.1:
        print("→ No significant azimuthal (phi) structure.")
        print("   FRB excess around the axis is consistent with being axisymmetric")
        print("   within the cone shells (more 'ring / cone' than 'pyramid faces').")
    elif p_mc > 0.01:
        print("→ Marginal evidence for azimuthal structure.")
        print("   Some shells may have preferred phi sectors, but not strongly.")
    else:
        print("→ Strong evidence for azimuthal structure.")
        print("   FRB excess is not purely axisymmetric — suggests faceted or")
        print("   patchy structure around the axis (pyramid-like faces / blobs).")

    # --------------------------------------------------------------
    # 6. figure: chi² null vs observed
    # --------------------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.hist(chi2_null, bins=40, alpha=0.7, edgecolor="black")
    plt.axvline(total_chi2, linestyle="--", linewidth=2, label=f"observed χ²={total_chi2:.1f}")
    plt.axvline(p95, linestyle=":", linewidth=2, label=f"null 95%={p95:.1f}")
    plt.xlabel("total χ² (azimuthal uniformity)")
    plt.ylabel("frequency")
    plt.title("Azimuthal structure null distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("axisymmetry_phi_test.png", dpi=200)

    print("\nSaved: axisymmetry_phi_test.png")
    print("\nanalysis complete.")
    print("="*70)


if __name__ == "__main__":
    main()
