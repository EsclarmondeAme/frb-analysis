#!/usr/bin/env python3
# ======================================================================
# FRB MASTER ANISOTROPY LIKELIHOOD (FINAL BOSS, INCLUDING TEST 20)
#
# This script does NOT run new sky statistics.
# It aggregates the p-values from all core detection tests, including
#
#   Test 20: large-scale / small-scale mode coupling
#
# and computes a unified L_core = Σ -log10(p_i)
# and an effective joint probability p_eff_core.
#
# Diagnostic neutral tests (which do NOT add to L_core) are reported
# separately.
# ======================================================================

import math
import json

def logL_from_p(p, p_floor=1e-12):
    p_used = max(p, p_floor)
    return -math.log10(p_used), p_used

# ----------------------------------------------------------------------
# core evidence tests (updated to include Test 20)
# ----------------------------------------------------------------------
core_tests = [
    { "name":"axis_alignment_triple",
      "label":"Axis alignment (FRB / CMB hemispherical / sidereal)",
      "p":1.0e-4 },

    { "name":"radial_break",
      "label":"Radial break at θ ≈ 25°",
      "p":1.0e-6 },

    { "name":"width_layering",
      "label":"Width layering and cone alignment",
      "p":9.0e-3 },

    { "name":"azimuthal_warp",
      "label":"Azimuthal m=1 + m=2 warped shell",
      "p":1.0e-6 },

    { "name":"low_ell_excess",
      "label":"Low-ℓ multipole excess (ℓ=1–3)",
      "p":5.0e-5 },

    { "name":"selection_forward_model",
      "label":"Selection-function forward-model failure",
      "p":4.7e-19 },

    { "name":"cosmology_model_comparison",
      "label":"Cosmology model comparison (warped shell vs void/dipole/Bianchi/gradient)",
      "p":1.0e-10 },

    { "name":"bayesian_model_evidence",
      "label":"Bayesian evidence (warped shell vs isotropic/dipole)",
      "p":1.0e-10 },

    { "name":"tomography_drift",
      "label":"3D spherical-harmonic tomography drift",
      "p":1.4e-2 },

    { "name":"bessel_3d_tomography",
      "label":"3D spherical–Bessel tomography (ℓ,k coherence)",
      "p":1.0/501.0 },

    { "name":"vector_helicity_EB",
      "label":"Vector–spherical–harmonic helicity (E/B, ℓ=1–8)",
      "p":1.0/2001.0 },

    # --------------------------------------------------------------
    # ★ NEW: Test 20 — large-scale / small-scale mode coupling ★
    # --------------------------------------------------------------
    # Observed: r = 0.7861, analytic p ≈ 4.46e-127
    # But Monte Carlo null used |r| permutations: p=0 in 1000 sims.
    # So we apply the same conservative logic as Tests 15 & 16:
    # p = 1 / (1000 + 1) ≈ 0.000999
    { "name":"large_small_mode_coupling",
      "label":"Large-scale / small-scale mode coupling (Test 20)",
      "p":1.0/1001.0 },
]

# ----------------------------------------------------------------------
# diagnostic / neutral tests (not added to L_core)
# ----------------------------------------------------------------------
diagnostic_tests = [
    { "name":"axis_fisher_curvature",
      "label":"Unified-axis Fisher curvature (peak-shape diagnostic)",
      "p":0.472 },

    { "name":"harmonic_bessel_coherence",
      "label":"Harmonic–Bessel 3D–2D coherence",
      "p":0.222 },
]

# ----------------------------------------------------------------------
# main aggregation
# ----------------------------------------------------------------------
def main():
    print("=======================================================================")
    print(" FRB MASTER ANISOTROPY LIKELIHOOD (FINAL COMBINED TEST, NOW WITH TEST 20)")
    print("=======================================================================")

    L_core_total = 0.0
    summary_core = []

    print("Core evidence tests (included in combined likelihood):")
    print("-------------------------------------------------------")

    for t in core_tests:
        p_raw = t["p"]
        L_i, p_used = logL_from_p(p_raw)
        L_core_total += L_i

        summary_core.append({
            "name": t["name"],
            "label": t["label"],
            "p_raw": p_raw,
            "p_used": p_used,
            "L_i": L_i,
        })

        print(f"  - {t['label']}")
        print(f"      p = {p_raw:.3e}  →  L_i = {L_i:.3f}")

    # combined effective p
    p_eff_core = 10.0**(-L_core_total)

    print("-------------------------------------------------------")
    print(f" Combined core statistic:  L_core = {L_core_total:.3f}")
    print(f" Effective joint p (core): p_eff_core ≈ {p_eff_core:.3e}")
    print("-------------------------------------------------------")

    print("Diagnostic (neutral) tests:")
    for t in diagnostic_tests:
        p_raw = t["p"]
        L_i, p_used = logL_from_p(p_raw)
        print(f"  - {t['label']}")
        print(f"      p = {p_raw:.3f}  →  L_diag = {L_i:.3f}")
        if p_raw > 0.05:
            print("      (neutral / consistency check)")
        else:
            print("      (mildly informative, not added to L_core)")

    print("-------------------------------------------------------")
    print(" Interpretation:")
    print("  - Adding Test 20 increases the combined evidence modestly,")
    print("    because its Monte Carlo floor p ≈ 0.001 gives L_20 ≈ 3.")
    print("  - The combined p remains exceptionally small, consistent with")
    print("    a multi-faceted anisotropy involving radial shells,")
    print("    azimuthal warping, low-ℓ modes, and scale-dependent structure.")
    print("-------------------------------------------------------")

    out = {
        "L_core_total": L_core_total,
        "p_eff_core": p_eff_core,
        "core_tests": summary_core,
        "diagnostic_tests": diagnostic_tests,
    }

    with open("frb_master_anisotropy_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("saved: frb_master_anisotropy_summary.json")
    print("=======================================================================")

if __name__ == "__main__":
    main()
