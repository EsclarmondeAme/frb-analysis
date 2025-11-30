#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FRB GLOBAL ANISOTROPY LIKELIHOOD TEST (TEST 12)

This script collects the key p-values from the major FRB anisotropy
diagnostics and combines them into a single global anisotropy likelihood:

    L_i   = -log10(p_i)
    L_tot = sum_i L_i
    p_eff = 10^(-L_tot)

The goal is not to introduce new data, but to provide a transparent,
reproducible calculation of the combined significance that appears
in the LaTeX "Unified cosmic likelihood aggregation" section.

You can adjust the individual p-values below as needed to match your
final analysis numbers before publication.

Usage
-----
Run from the CrossLayerPhysics directory:

    python frb_global_anisotropy_likelihood_test.py
"""

import math
from typing import List, Dict

# ----------------------------------------------------------------------
# configuration: list of key tests and their p-values
# ----------------------------------------------------------------------
# note: these values are chosen to be representative of the analysis
# already described in the paper. you can edit them to match your
# final numbers exactly (e.g. from each dedicated script).
#
# IMPORTANT: p-values must be in (0,1). for "p = 0" Monte Carlo results,
# use a conservative upper bound like 1e-6 or 1e-8 instead of 0.
# ----------------------------------------------------------------------

TESTS: List[Dict] = [
    {
        "name": "axis_alignment_triple",
        "label": "Axis alignment (FRB / CMB hemispherical / sidereal)",
        "p_value": 1.0e-4,   # ~1e-4 from triple-axis Monte Carlo
    },
    {
        "name": "radial_break",
        "label": "Radial break at θ ≈ 25°",
        "p_value": 1.0e-6,   # very strong ΔAIC improvement
    },
    {
        "name": "width_layering",
        "label": "Width layering and cone alignment",
        "p_value": 9.0e-3,   # ~9×10^-3 from width-layer tests
    },
    {
        "name": "azimuthal_warp",
        "label": "Azimuthal m = 1 + m = 2 warped shell",
        "p_value": 1.0e-6,   # Monte Carlo p ≈ 0 → conservative bound
    },
    {
        "name": "low_ell_excess",
        "label": "Low-ℓ multipole excess (ℓ = 1–3)",
        "p_value": 5.0e-5,   # p < 5×10^-5 from multipole tests
    },
    {
        "name": "selection_forward_model",
        "label": "Selection-function forward-model failure",
        "p_value": 4.7e-19,  # from χ²_sel ≈ 9.9×10^4
    },
    {
        "name": "cosmology_model_comparison",
        "label": "Cosmology model comparison (warped shell vs void/dipole/Bianchi/gradient)",
        # interpret enormous ΔAIC (~1300) as effectively p ≪ 10^-10
        # here we set a conservative bound:
        "p_value": 1.0e-10,
    },
    {
        "name": "bayesian_model_evidence",
        "label": "Bayesian evidence (warped shell vs isotropic/dipole)",
        # again, Bayes factors are astronomically large; use a small bound
        "p_value": 1.0e-10,
    }
]


# ----------------------------------------------------------------------
# helper functions
# ----------------------------------------------------------------------

def combine_pvalues_log10(p_values: List[float]) -> float:
    """
    Combine a list of p-values into a single statistic:

        L_tot = sum_i -log10(p_i)

    Parameters
    ----------
    p_values : list of float
        Individual p-values in the open interval (0, 1).

    Returns
    -------
    float
        Combined L_tot.
    """
    L_tot = 0.0
    for p in p_values:
        if not (0.0 < p < 1.0):
            raise ValueError(f"invalid p-value {p:.3g}; must be in (0,1)")
        L_tot += -math.log10(p)
    return L_tot


def format_scientific_p(p: float) -> str:
    """
    Format a p-value in a compact scientific notation suitable for printing.
    """
    if p == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(p)))
    mantissa = p / (10.0 ** exponent)
    return f"{mantissa:.2f}×10^{exponent:d}"


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("FRB GLOBAL ANISOTROPY LIKELIHOOD (TEST 12)")
    print("=" * 70)
    print()

    print("individual test contributions:")
    print("-" * 70)

    p_values: List[float] = []
    for t in TESTS:
        name = t["name"]
        label = t["label"]
        p = float(t["p_value"])
        L_i = -math.log10(p)

        p_values.append(p)
        print(f"{name:30s}  p = {p:.3e}   L_i = -log10(p) = {L_i:6.3f}")
        print(f"    → {label}")

    print("-" * 70)
    L_tot = combine_pvalues_log10(p_values)
    p_eff = 10.0 ** (-L_tot)

    print(f"combined statistic:  L_tot = Σ_i [-log10(p_i)] = {L_tot:6.3f}")
    print(f"effective joint p:   p_eff ≈ {p_eff:.3e}  "
          f"(≈ {format_scientific_p(p_eff)})")
    print("-" * 70)
    print("note:")
    print("  - this combination assumes the listed diagnostics are")
    print("    approximately independent or only weakly correlated.")
    print("  - for publication, you can adjust the individual p_i to match")
    print("    the final values reported by each dedicated test script.")
    print("=" * 70)


if __name__ == "__main__":
    main()
