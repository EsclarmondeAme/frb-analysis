import numpy as np
import pandas as pd

# ------------------------------------------------------------
# frb_sidereal_harmonics.py
#
# analyze harmonic content of FRB arrival times in sidereal phase
#
# uses:
#   - frbs.csv  (from your CHIME Catalog 1 conversion)
#   - 'mjd' column as the time coordinate
#
# we approximate local sidereal phase via:
#   sidereal_days ≈ mjd * (1 + 1/365.2422)
#   phase = 2π * frac(sidereal_days)
#
# then compute Fourier amplitudes for n = 1..4
# and estimate significance by Monte Carlo.
# ------------------------------------------------------------

def fourier_amplitudes(phases, max_n=4):
    """
    compute simple Fourier amplitudes A_n, B_n, R_n for n = 1..max_n
    phases are in radians, assumed uniform under null hypothesis.
    """
    phases = np.asarray(phases)
    N = len(phases)
    if N == 0:
        raise ValueError("no phases provided")

    results = []
    for n in range(1, max_n + 1):
        c = np.cos(n * phases)
        s = np.sin(n * phases)
        # 2/N normalization is convenient for comparison
        A_n = (2.0 / N) * np.sum(c)
        B_n = (2.0 / N) * np.sum(s)
        R_n = np.sqrt(A_n**2 + B_n**2)
        results.append((n, A_n, B_n, R_n))
    return results


def monte_carlo_significance(phases, max_n=4, n_trials=2000, seed=12345):
    """
    for each harmonic n, estimate how often random uniform phases
    give an amplitude >= observed.
    """
    rng = np.random.default_rng(seed)
    phases = np.asarray(phases)
    N = len(phases)

    obs = fourier_amplitudes(phases, max_n=max_n)
    # store observed R_n in a dict for convenience
    obs_R = {n: R_n for (n, _, _, R_n) in obs}

    counts = {n: 0 for n in range(1, max_n + 1)}

    for _ in range(n_trials):
        rand_phases = rng.uniform(0.0, 2.0 * np.pi, size=N)
        rand_res = fourier_amplitudes(rand_phases, max_n=max_n)
        for (n, _, _, R_n) in rand_res:
            if R_n >= obs_R[n]:
                counts[n] += 1

    pvals = {n: counts[n] / n_trials for n in counts}
    return obs, pvals


def main():
    print("============================================================")
    print("  FRB sidereal harmonic analysis (using MJD from frbs.csv)   ")
    print("============================================================")

    frb = pd.read_csv("frbs.csv")

    # require an mjd column
    if "mjd" not in frb.columns:
        raise ValueError("frbs.csv is missing an 'mjd' column")

    # drop rows without mjd
    frb = frb.dropna(subset=["mjd"]).copy()
    mjd = frb["mjd"].astype(float).values

    N = len(mjd)
    print(f"FRBs with valid MJD: {N}")
    if N < 50:
        print("warning: very few FRBs, significance will be weak.")

    # ----------------------------------------------------------------
    # approximate local sidereal phase
    #
    # sidereal day is ~ 0.99726957 solar days
    # equivalently, sidereal frequency ~ 1.0027379 * solar frequency
    #
    # for our purposes, we just need a consistent phase definition,
    # so:
    #    sidereal_days ≈ mjd * 1.0027379
    #    phase = 2π * frac(sidereal_days)
    # ----------------------------------------------------------------
    sidereal_days = mjd * 1.0027379
    frac = sidereal_days - np.floor(sidereal_days)
    phases = 2.0 * np.pi * frac

    max_n = 4
    n_trials = 2000

    print("computing Fourier amplitudes for n = 1 .. 4 ...")
    obs, pvals = monte_carlo_significance(
        phases,
        max_n=max_n,
        n_trials=n_trials,
        seed=12345,
    )

    print("------------------------------------------------------------")
    print("harmonic amplitudes and Monte Carlo p-values")
    print(" (R_n is the amplitude of the n-th harmonic)")
    print("------------------------------------------------------------")
    print("  n      A_n         B_n         R_n        p(R_rand >= R_n)")
    print("------------------------------------------------------------")
    for (n, A_n, B_n, R_n) in obs:
        p = pvals[n]
        print(
            f"  {n:1d}  {A_n: .4e}  {B_n: .4e}  {R_n: .4e}      {p: .4f}"
        )
    print("------------------------------------------------------------")
    print("interpretation:")
    print("  - p << 0.05 for n=1 → strong dipole modulation in sidereal phase.")
    print("  - p << 0.05 for n>1 → higher-harmonic structure (complex beam / cone).")
    print("  - if all p ≳ 0.1, phases are consistent with uniformity.")
    print("============================================================")


if __name__ == "__main__":
    main()
