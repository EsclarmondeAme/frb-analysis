#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse

k_B = 1.380649e-23  # Boltzmann constant (J/K)
pi = np.pi

# ----------------------------------------------------------------------
# helper: safe dictionary get for Q and f0
# ----------------------------------------------------------------------
def get_mode_value(table, l, m, key):
    if (l, m) in table:
        return table[(l, m)][key]
    return np.nan

# ----------------------------------------------------------------------
# main loader
# ----------------------------------------------------------------------
def load_Qtable(path):
    raw = np.load(path, allow_pickle=True).item()
    # raw[(l,m)] = {"f0": f0_value, "Q": Q_value}
    return raw

# ----------------------------------------------------------------------
# mode volume model (very approximate)
# allows user to set R and dR
# ----------------------------------------------------------------------
def mode_volume(R=1e26, dR=1e25):
    # cosmic shell volume
    return 4 * pi * (R**2) * dR

# ----------------------------------------------------------------------
# field amplitude model:
#   A ~ sqrt( 2*u / (eps * omega^2) )
# user can supply substrate permittivity eps
# ----------------------------------------------------------------------
def field_amplitude(u, omega, eps=1.0):
    if omega == 0:
        return 0.0
    return np.sqrt(2 * u / (eps * omega * omega))

# ----------------------------------------------------------------------
# extracted power:
#   P_out ~ (g*omega)^2 * E * tau
# detection threshold: k_B T B
# ----------------------------------------------------------------------
def required_coupling(E, tau, omega, P_detect):
    if E <= 0 or tau <= 0 or omega <= 0:
        return np.nan
    # solve P_out = (g*omega)^2 * E * tau = P_detect
    g = np.sqrt(P_detect / (E * tau)) / omega
    return g

# ----------------------------------------------------------------------
# run calculation
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("energy_csv", help="test111A_energy.csv")
    parser.add_argument("flux_csv", help="test111A_flux.csv")
    parser.add_argument("qfile", help="test107A_results.npy")
    parser.add_argument("--R", type=float, default=1e26,
                        help="mode radius (m), default 1e26")
    parser.add_argument("--dR", type=float, default=1e25,
                        help="mode shell thickness (m), default 1e25")
    parser.add_argument("--eps", type=float, default=1.0,
                        help="substrate permittivity-like constant")
    parser.add_argument("--T", type=float, default=300,
                        help="detector temp (K), default 300")
    parser.add_argument("--B", type=float, default=1.0,
                        help="measurement bandwidth (Hz), default 1 Hz")
    args = parser.parse_args()

    print("================================================")
    print("Test 111B — Mode Coupling & Detectability")
    print("================================================")

    # ------------------------------------------------------------------
    # load data
    # ------------------------------------------------------------------
    dfE = pd.read_csv(args.energy_csv)
    dfF = pd.read_csv(args.flux_csv)
    Qtable = load_Qtable(args.qfile)

    # must include columns: l, m, E_real
    # ensure merge
    df = pd.merge(dfE, dfF, on=["l", "m"], how="inner")

    V = mode_volume(args.R, args.dR)

    # detection threshold
    P_detect = k_B * args.T * args.B

    rows = []
    for _, row in df.iterrows():
        l = int(row["l"])
        m = int(row["m"])
        E_real = float(row["E_real"])
        flux = float(row["Flux_real"])

        f0 = get_mode_value(Qtable, l, m, "f0")
        Qval = get_mode_value(Qtable, l, m, "Q")

        if np.isnan(f0) or np.isnan(Qval):
            omega = np.nan
            tau = np.nan
        else:
            omega = 2 * pi * f0
            tau = Qval / omega if omega > 0 else np.nan

        # energy density
        u = E_real / V if V > 0 else np.nan

        # field amplitude
        A = field_amplitude(u, omega, eps=args.eps)

        # required coupling to exceed detection
        g_req = required_coupling(E_real, tau, omega, P_detect)

        # predicted extracted power if g=1 and eps=1 (upper bound)
        P_out_max = (omega**2) * E_real * tau if (not np.isnan(omega) and not np.isnan(tau)) else np.nan

        rows.append({
            "l": l,
            "m": m,
            "E_real": E_real,
            "flux": flux,
            "f0": f0,
            "Q": Qval,
            "omega": omega,
            "tau": tau,
            "energy_density_u": u,
            "field_amplitude_A": A,
            "g_required": g_req,
            "P_out_max": P_out_max
        })

    out = pd.DataFrame(rows)
    out.to_csv("test111B_coupling.csv", index=False)

    print("------------------------------------------------")
    print("Test 111B complete.")
    print("results saved to test111B_coupling.csv")
    print("------------------------------------------------")

if __name__ == "__main__":
    main()
