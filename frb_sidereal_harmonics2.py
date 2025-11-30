import numpy as np
import pandas as pd

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

# ------------------------------------------------------------
# frb_sidereal_harmonics.py
#
# analyze harmonic content of frb arrival times in sidereal phase
#
# uses:
#   - frbs.csv  (from your chime catalog conversion)
#   - 'mjd' column as the time coordinate
#
# now uses exact local sidereal time at chime via astropy:
#   - mjd → utc (Time)
#   - utc → local mean sidereal time (lst)
#   - phase = lst in radians [0, 2π)
# ------------------------------------------------------------

# chime location
CHIME_LOCATION = EarthLocation.from_geodetic(
    lon=-119.62 * u.deg,
    lat=49.32 * u.deg,
    height=0.0 * u.m,
)


def fourier_amplitudes(phases: np.ndarray, max_n: int = 4):
    """
    compute simple fourier amplitudes A_n, B_n, R_n for n = 1..max_n.

    phases
        array of phases in radians, assumed uniform under the null.
    returns
        list of tuples (n, A_n, B_n, R_n) for n = 1..max_n
        with the 2/N normalization preserved (so R_n ~ O(1)).
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


def monte_carlo_significance(
    phases: np.ndarray,
    max_n: int = 4,
    n_trials: int = 2000,
    seed: int = 12345,
):
    """
    for each harmonic n, estimate how often random uniform phases
    give an amplitude >= observed.

    phases
        array of sidereal phases (radians)
    returns
        obs  : list of (n, A_n, B_n, R_n)
        pvals: dict {n: p_value}
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
    print("  frb sidereal harmonic analysis (using mjd from frbs.csv)  ")
    print("  (with exact local sidereal time at chime)                 ")
    print("============================================================")

    frb = pd.read_csv("frbs.csv")

    # require an mjd column
    if "mjd" not in frb.columns:
        raise ValueError("frbs.csv is missing an 'mjd' column")

    # drop rows without mjd
    frb = frb.dropna(subset=["mjd"]).copy()
    mjd = frb["mjd"].astype(float).values

    N = len(mjd)
    print(f"frbs with valid mjd: {N}")
    if N < 50:
        print("warning: very few frbs, significance will be weak.")

    # ----------------------------------------------------------------
    # exact local sidereal phase at chime
    #
    # mjd (utc) → Time
    # Time → local mean sidereal time at chime longitude
    # lst → phase in radians [0, 2π)
    # ----------------------------------------------------------------
    t = Time(mjd, format="mjd", scale="utc")
    lst = t.sidereal_time("mean", longitude=CHIME_LOCATION.lon)
    phases = lst.to_value(u.rad) % (2.0 * np.pi)

    max_n = 4
    n_trials = 2000

    print("computing fourier amplitudes for n = 1 .. 4 ...")
    obs, pvals = monte_carlo_significance(
        phases,
        max_n=max_n,
        n_trials=n_trials,
        seed=12345,
    )

    print("------------------------------------------------------------")
    print("harmonic amplitudes and monte carlo p-values")
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
