import pandas as pd
import numpy as np
import sys

def main():
    if len(sys.argv) != 3:
        print("usage: python frb_harmonic_mode_coupling_test111B_geom.py energy.csv flux.csv")
        sys.exit(1)

    energy_file = sys.argv[1]
    flux_file   = sys.argv[2]

    dfE = pd.read_csv(energy_file)
    dfF = pd.read_csv(flux_file)

    # merge on l,m
    df = pd.merge(dfE, dfF, on=['l','m'], suffixes=('_E','_F'))

    print("===============================================")
    print(" Test 111B — Geometric Harmonic Coupling Report")
    print(" (no physical units — harmonic geometry only) ")
    print("===============================================")

    print("\nMode-by-mode harmonic coupling strength:\n")
    print(f"{'l':>2} {'m':>3} | {'E_real':>10} {'E_null':>10} {'p_E':>7} | "
          f"{'F_real':>10} {'F_null':>10} {'p_F':>7} | {'CouplingScore':>14}")

    print("-"*95)

    for _, r in df.iterrows():

        # geometric “energy deviation”
        dE = abs(r['E_real'] - r['E_null_mean']) / (1e-6 + abs(r['E_null_mean']))

        # geometric “flux deviation”
        dF = abs(r['Flux_real'] - r['Flux_null_mean']) / (1e-6 + abs(r['Flux_null_mean']))

        # combined geometric harmonic coupling
        coupling = dE + dF

        print(f"{int(r['l']):>2} {int(r['m']):>3} | "
              f"{r['E_real']:>10.1f} {r['E_null_mean']:>10.1f} {r['p_energy']:>7.3f} | "
              f"{r['Flux_real']:>10.3f} {r['Flux_null_mean']:>10.3f} {r['p_flux']:>7.3f} | "
              f"{coupling:>14.3f}")

    print("\n===============================================")
    print(" Interpretation:")
    print(" • high CouplingScore → strong harmonic interaction / resonance with the manifold")
    print(" • low  p_energy      → geometric over-occupation of that harmonic mode")
    print(" • low  p_flux        → non-random directional flow in harmonic space")
    print(" • high scores across ℓ,m families = harmonic fibres (runes, spell-chains)")
    print("===============================================\n")


if __name__ == "__main__":
    main()
