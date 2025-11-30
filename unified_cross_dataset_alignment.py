#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIFIED CROSS-DATASET AXIS ALIGNMENT TEST
----------------------------------------
Compares FRB unified axis with major known cosmological anisotropy axes:
 - CMB dipole
 - CMB hemispherical asymmetry
 - Quasar dipole
 - Radio/NVSS dipole
 - X-ray background dipole
 - 2MASS dipole
 - Cosmic bulk flow direction
 - SNe Ia dipole
 - Dark flow axis
 - Fine-structure constant dipole
 - Polarization dipole

Produces:
 - table of pairwise separations
 - Monte Carlo isotropic null distribution
 - combined likelihood for joint alignment
 - sky plot of axis directions
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt


# ==========================================================
# 1. define known cosmological anisotropy axes
#    (all in galactic coordinates)
# ==========================================================
AXES = {
    "FRB_unified_axis":            (159.85,  -0.51),
    "CMB_dipole":                  (264.00,  48.00),
    "CMB_hemispherical_asym":      (152.62,   4.03),
    "Quasar_dipole":               (244.00,  44.00),
    "NVSS_radio_dipole":           (253.00,  27.00),
    "XRB_Xray_dipole":             (255.00,  39.00),
    "2MASS_dipole":                (268.00,  22.00),
    "Bulk_flow":                   (282.00,  11.00),
    "SNeIa_dipole":                (309.00,  -15.00),
    "Dark_flow_axis":              (283.00,   12.00),
    "Alpha_variation_dipole":      (330.00,  -15.00),
    "Polarization_dipole":         (243.00,   30.00)
}


# ==========================================================
# function: angular separation in degrees
# ==========================================================
def angsep(l1, b1, l2, b2):
    c1 = SkyCoord(l=l1*u.deg, b=b1*u.deg, frame='galactic')
    c2 = SkyCoord(l=l2*u.deg, b=b2*u.deg, frame='galactic')
    return c1.separation(c2).deg


# ==========================================================
# 2. pairwise distances from FRB axis
# ==========================================================
def compute_separations():
    frb_l, frb_b = AXES["FRB_unified_axis"]
    results = {}
    for name, (l, b) in AXES.items():
        if name == "FRB_unified_axis":
            continue
        d = angsep(frb_l, frb_b, l, b)
        results[name] = d
    return results


# ==========================================================
# 3. Monte Carlo: random axes vs real cluster
# ==========================================================
def monte_carlo_test(real_dists, nsim=20000):
    real_mean = np.mean(real_dists)
    mc_means = []

    for _ in range(nsim):
        # random axes uniformly on sphere
        l_rand = np.random.uniform(0, 360)
        b_rand = np.degrees(np.arcsin(np.random.uniform(-1, 1)))

        dists = []
        for name, (l, b) in AXES.items():
            if name == "FRB_unified_axis":
                continue
            dists.append(angsep(l_rand, b_rand, l, b))

        mc_means.append(np.mean(dists))

    mc_means = np.array(mc_means)
    p = np.mean(mc_means <= real_mean)
    return real_mean, mc_means, p


# ==========================================================
# 4. plotting helper
# ==========================================================
def plot_axes():
    plt.figure(figsize=(6,6))
    for name, (l, b) in AXES.items():
        plt.scatter(l, b, s=50)
        plt.text(l+1, b+1, name.replace("_"," "), fontsize=8)
    plt.xlabel("galactic longitude l (deg)")
    plt.ylabel("galactic latitude b (deg)")
    plt.title("cosmic anisotropy axes")
    plt.grid(True)
    plt.savefig("unified_axis_cross_dataset_map.png", dpi=150)
    plt.close()


# ==========================================================
# 5. main
# ==========================================================
def main():
    print("="*70)
    print("UNIFIED CROSS-DATASET AXIS ALIGNMENT TEST")
    print("="*70)

    print("\nKnown dataset axes loaded:", len(AXES))

    # compute separations
    seps = compute_separations()
    frb_l, frb_b = AXES["FRB_unified_axis"]

    print("\nPairwise angular separations from FRB unified axis:")
    print("------------------------------------------------------")
    for name, d in seps.items():
        print(f"{name:28s} : {d:6.2f}°")

    real_dists = np.array(list(seps.values()))

    print("\nComputing Monte Carlo null (20k sims)...")
    real_mean, mc_means, p = monte_carlo_test(real_dists)

    print("\n======================================================")
    print("SCIENTIFIC VERDICT")
    print("======================================================")
    print(f"mean separation (real)     = {real_mean:.2f}°")
    print(f"mean separation (null)     = {np.mean(mc_means):.2f}°")
    print(f"1% null percentile         = {np.percentile(mc_means,1):.2f}°")
    print(f"p-value                    = {p:.6f}")

    if p < 1e-3:
        print("\n→ strong evidence that FRB axis is aligned with multiple")
        print("  major cosmological anisotropy axes.")
    elif p < 0.05:
        print("\n→ moderate evidence for cross-dataset axis alignment.")
    else:
        print("\n→ no significant cross-dataset axis correlation detected.")

    plot_axes()
    print("\nmap saved: unified_axis_cross_dataset_map.png")
    print("="*70)


if __name__ == "__main__":
    main()
