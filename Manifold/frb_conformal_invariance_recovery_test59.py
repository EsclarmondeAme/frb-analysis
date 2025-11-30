#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------------------------------------------
# utilities
# ----------------------------------------------------------

def grain_intensity(theta, phi):
    """grain intensity with zero-distance protection"""
    N = len(theta)
    vec = np.column_stack([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ])
    dvals = np.sqrt(np.sum((vec[:,None,:] - vec[None,:,:])**2, axis=2))

    G = np.zeros(N)
    for i in range(N):
        nn = np.sort(dvals[i])[1:7]

        # remove zeros / tiny values
        nn = nn[nn > 1e-8]
        if len(nn) == 0:
            m = 1e-3
        else:
            m = np.mean(nn)
            if m < 1e-8 or not np.isfinite(m):
                m = 1e-3

        G[i] = 1.0 / m

    # normalize safely
    G = (G - np.nanmean(G)) / (np.nanstd(G) + 1e-9)
    return G


def energy_gradient_encoding(dm, snr, flu, width, z):
    """scalar energy gradient feature"""
    E = (dm + 0.1*snr + 0.01*flu + 0.001*width + 5.0*z)
    E = (E - np.mean(E)) / (np.std(E) + 1e-9)
    return E


def conformal_boost(theta, eps):
    """
    spherical conformal boost along +z direction:
    cos(theta') = (cos(theta) - eps) / (1 - eps*cos(theta))
    """
    c = np.cos(theta)
    num = c - eps
    den = 1 - eps*c
    cp = num / (den + 1e-9)

    cp = np.clip(cp, -1.0, 1.0)
    return np.arccos(cp)


# ----------------------------------------------------------
# main
# ----------------------------------------------------------

def main():
    import sys

    if len(sys.argv) < 2:
        print("usage: python frb_conformal_invariance_recovery_test59.py frbs_unified.csv")
        return

    # load
    df = pd.read_csv(sys.argv[1])
    print(f"loaded {len(df)} FRBs")

    # unified coordinates
    theta = np.deg2rad(df["theta_unified"].values)
    phi   = np.deg2rad(df["phi_unified"].values)

    # latent field F used in Tests 50–57
    G   = grain_intensity(theta, phi)
    EGR = energy_gradient_encoding(df["dm"].values,
                                   df["snr"].values,
                                   df["fluence"].values,
                                   df["width"].values,
                                   df["z_est"].values)

    F = 0.6*G + 0.4*EGR
    F = (F - np.nanmean(F)) / (np.nanstd(F) + 1e-9)

    # remove any NaNs that might still exist
    mask = np.isfinite(F) & np.isfinite(theta)
    theta = theta[mask]
    F = F[mask]

    # ------------------------------------------------------
    # baseline anisotropy A0
    # simple linear anisotropy measure: slope vs cos(theta)
    # ------------------------------------------------------
    x0 = np.cos(theta)
    A0, _ = np.polyfit(x0, F, 1)

    # ------------------------------------------------------
    # scan conformal boosts
    # ------------------------------------------------------
    eps_grid = np.linspace(-0.3, 0.3, 121)   # symmetric around 0
    best_A = None
    best_eps = 0.0

    for eps in eps_grid:
        t_new = conformal_boost(theta, eps)
        x = np.cos(t_new)
        A, _ = np.polyfit(x, F, 1)
        if (best_A is None) or (abs(A) < abs(best_A)):
            best_A = A
            best_eps = eps

    Aconf = best_A

    # ------------------------------------------------------
    # recovery ratio
    # ------------------------------------------------------
    r = abs(Aconf) / (abs(A0) + 1e-12)

    # ------------------------------------------------------
    # Monte Carlo null for r
    # ------------------------------------------------------
    print("running Monte Carlo null for conformal recovery...")
    N_MC = 2000
    rnull = []

    for _ in tqdm(range(N_MC)):
        Fsh = np.random.permutation(F)

        x0s = np.cos(theta)
        A0s, _ = np.polyfit(x0s, Fsh, 1)

        best_As = None
        for eps in eps_grid:
            t_new_s = conformal_boost(theta, eps)
            x = np.cos(t_new_s)
            As, _ = np.polyfit(x, Fsh, 1)

            if (best_As is None) or (abs(As) < abs(best_As)):
                best_As = As

        rnull.append(abs(best_As) / (abs(A0s) + 1e-12))

    rnull = np.array(rnull)
    p = np.mean(rnull <= r)

    # ------------------------------------------------------
    # report
    # ------------------------------------------------------
    print("====================================================================")
    print("FRB CONFORMAL INVARIANCE RECOVERY TEST (TEST 59)")
    print("====================================================================")
    print(f"N_FRB = {len(theta)}")
    print("--------------------------------------------------------------------")
    print(f"baseline anisotropy A0         = {A0:.6f}")
    print(f"best recovered anisotropy Aconf = {Aconf:.6f}")
    print(f"best conformal parameter eps    = {best_eps:.3f}")
    print(f"recovery ratio r                = {r:.6f}")
    print("--------------------------------------------------------------------")
    print("Monte Carlo null:")
    print(f"null mean r     = {np.mean(rnull):.6f}")
    print(f"null std  r     = {np.std(rnull):.6f}")
    print(f"p-value (r_null <= r_real) = {p:.6f}")
    print("--------------------------------------------------------------------")
    print("interpretation:")
    print("  - low p (<< 0.5): conformal boost recovers isotropy → conformal structure")
    print("  - high p (~0.5–1): similar recovery happens generically → no conformal invariance")
    print("====================================================================")
    print("test 59 complete.")
    print("====================================================================")


if __name__ == "__main__":
    main()
