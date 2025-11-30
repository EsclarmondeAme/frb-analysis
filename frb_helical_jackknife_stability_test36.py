#!/usr/bin/env python3
"""
FRB HELICAL JACKKNIFE STABILITY TEST (TEST 36)

purpose
-------
test whether the global helical pitch around the unified axis is
dominated by a single sky patch, or remains stable when we remove
different angular regions ("jackknife" resamples).

we use the same helical pitch estimator as in Test 26:

    - bin in theta (unified-axis polar angle)
    - compute phi_max(theta) via circular mean in each bin
    - fit phi_max(theta) = phi0 + k * theta

then:

    1. compute k_full from the full sample in a shell where helicity
       is strongest (40° <= theta_unified <= 90°).

    2. define four azimuthal jackknife regions in phi_unified:
       Q1: [-180,-90), Q2: [-90,0), Q3: [0,90), Q4: [90,180)

       for each region r, remove that region and recompute k_r.

    3. define the jackknife instability statistic

           S_real = sqrt( mean_r (k_r - k_full)^2 )

       large S_real would mean "one or more patches strongly control k".

    4. generate an isotropic null by shuffling phi while preserving
       theta, recomputing S_null for each realisation.

    5. compute p-value p = P(S_null >= S_real).

if p is small, the real data are *more stable* than random → helicity
is globally coherent. if p is large, the helicity is no more stable
than random fluctuations (but not necessarily ruled out).
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import circmean

# theta shell where twist is strongest (outer unified shell)
THETA_MIN = 40.0
THETA_MAX = 90.0

BIN_WIDTH = 10.0      # theta bin width for phi_max(theta)
MIN_PER_BIN = 6       # minimum FRBs per bin to estimate phi_max
N_MC = 5000           # Monte Carlo realisations for null jackknife RMS


# ---------------------------------------------------------------------
# catalog loading
# ---------------------------------------------------------------------
def load_frbs(path: str) -> pd.DataFrame:
    """
    load FRB catalog, require theta_unified / phi_unified and
    detect redshift column (z_est / z / redshift).
    """
    df = pd.read_csv(path)

    if "theta_unified" not in df.columns or "phi_unified" not in df.columns:
        raise ValueError("catalog must contain theta_unified and phi_unified")

    # redshift detection (not actually needed for this test, but consistent
    # with other recent tests)
    if "z_est" in df.columns:
        print("detected redshift column: z_est")
        df = df.rename(columns={"z_est": "z"})
    elif "z" in df.columns:
        print("detected redshift column: z")
    elif "redshift" in df.columns:
        print("detected redshift column: redshift")
        df = df.rename(columns={"redshift": "z"})
    else:
        print("warning: no redshift column found (z_est/z/redshift). proceeding without z.")
        df["z"] = np.nan

    df = df.dropna(subset=["theta_unified", "phi_unified"])
    return df


# ---------------------------------------------------------------------
# helical pitch machinery (same logic as Test 26)
# ---------------------------------------------------------------------
def estimate_phi_peaks(theta_deg, phi_deg, bin_width=BIN_WIDTH, min_per_bin=MIN_PER_BIN):
    """
    in bins of theta, compute the circular-mean azimuth phi_max(theta).

    parameters
    ----------
    theta_deg : array-like
        unified polar angle in degrees.
    phi_deg : array-like
        unified azimuth in degrees ([-180, 180] assumed).
    """
    theta = np.asarray(theta_deg)
    phi = np.asarray(phi_deg)

    if len(theta) < min_per_bin * 3:
        return None, None

    # convert phi to radians for circular mean
    phi_rad = np.radians(phi)

    bins = np.arange(theta.min(), theta.max() + bin_width, bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])
    peaks = []

    for i in range(len(bins) - 1):
        mask = (theta >= bins[i]) & (theta < bins[i + 1])
        vals = phi_rad[mask]
        if len(vals) < min_per_bin:
            peaks.append(np.nan)
        else:
            peaks.append(circmean(vals, high=np.pi, low=-np.pi))

    return centers, np.array(peaks)


def helix(theta_deg, phi0, k):
    """linear helical model: phi(theta) = phi0 + k * theta."""
    return phi0 + k * theta_deg


def measure_pitch(theta_deg, phi_deg):
    """
    measure helical pitch k in the chosen theta shell.

    returns
    -------
    k : float or None
        best-fit pitch (deg per deg). None if not enough data.
    """
    theta = np.asarray(theta_deg)
    phi = np.asarray(phi_deg)

    # restrict to outer shell
    shell = (theta >= THETA_MIN) & (theta <= THETA_MAX)
    theta_shell = theta[shell]
    phi_shell = phi[shell]

    if theta_shell.size < MIN_PER_BIN * 3:
        return None

    centers, peaks = estimate_phi_peaks(theta_shell, phi_shell)
    if centers is None:
        return None

    mask = ~np.isnan(peaks)
    if mask.sum() < 3:
        return None

    try:
        popt, _ = curve_fit(
            helix,
            centers[mask],
            np.degrees(peaks[mask]),
            p0=[0.0, 0.0],
        )
    except Exception:
        return None

    phi0_fit, k_fit = popt
    return k_fit


# ---------------------------------------------------------------------
# jackknife statistic
# ---------------------------------------------------------------------
def jackknife_regions(phi_deg):
    """
    define four azimuthal jackknife regions in phi (degrees):

        Q1: [-180, -90)
        Q2: [-90,   0)
        Q3: [  0,  90)
        Q4: [ 90, 180]

    returns a dict: name -> boolean mask
    """
    phi = np.asarray(phi_deg)

    regions = {}
    regions["Q1"] = (phi >= -180.0) & (phi < -90.0)
    regions["Q2"] = (phi >= -90.0) & (phi < 0.0)
    regions["Q3"] = (phi >= 0.0) & (phi < 90.0)
    regions["Q4"] = (phi >= 90.0) & (phi <= 180.0)

    return regions


def compute_jackknife_rms(theta_deg, phi_deg):
    """
    compute jackknife RMS statistic:

        S = sqrt( mean_r (k_r - k_full)^2 )

    where k_full is the pitch from the full sample and k_r is the pitch
    after removing region r.
    """
    theta = np.asarray(theta_deg)
    phi = np.asarray(phi_deg)

    k_full = measure_pitch(theta, phi)
    if k_full is None:
        return None, None, None

    regions = jackknife_regions(phi)

    k_jack = {}
    for name, mask_rm in regions.items():
        keep = ~mask_rm
        k_r = measure_pitch(theta[keep], phi[keep])
        k_jack[name] = k_r

    # compute RMS over regions with valid k_r
    valid = [k for k in k_jack.values() if k is not None]
    if len(valid) == 0:
        return k_full, k_jack, None

    diffs = np.array(valid) - k_full
    S = float(np.sqrt(np.mean(diffs**2)))
    return k_full, k_jack, S


def mc_null_jackknife(theta_deg, phi_deg, n_sims=N_MC):
    """
    Monte Carlo null: shuffle phi among FRBs, recompute jackknife RMS
    each time. returns an array S_null of RMS values.
    """
    theta = np.asarray(theta_deg)
    phi = np.asarray(phi_deg)
    N = len(phi)

    rng = np.random.default_rng(12345)
    S_null = []

    for _ in range(n_sims):
        # shuffle phi; preserve theta distribution
        phi_sh = rng.permutation(phi)

        k_full, k_jack, S = compute_jackknife_rms(theta, phi_sh)
        if S is None or k_full is None:
            # skip pathological realisations with too few bins
            continue

        S_null.append(S)

    return np.array(S_null)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("usage: python frb_helical_jackknife_test36.py frbs_unified.csv")
        sys.exit(1)

    path = sys.argv[1]
    df = load_frbs(path)

    theta = df["theta_unified"].values
    phi = df["phi_unified"].values

    print("=" * 69)
    print("FRB HELICAL JACKKNIFE STABILITY TEST (TEST 36)")
    print("=" * 69)
    print(f"loaded {len(df)} FRBs")
    print(f"using theta shell {THETA_MIN:.1f}–{THETA_MAX:.1f} deg")
    print()

    # full-sample pitch and jackknife RMS
    k_full, k_jack, S_real = compute_jackknife_rms(theta, phi)

    if k_full is None or S_real is None:
        print("not enough data in shell to compute stable jackknife statistic.")
        print("=" * 69)
        print("test 36 complete.")
        print("=" * 69)
        return

    print("full-sample pitch in shell:")
    print(f"  k_full = {k_full:.5f} deg/deg")
    print()

    print("jackknife pitches by azimuthal region (remove region, refit k):")
    for name, k_r in k_jack.items():
        if k_r is None:
            print(f"  {name}: k = None (insufficient data)")
        else:
            print(f"  {name}: k = {k_r:.5f} deg/deg")
    print()

    print(f"jackknife RMS statistic S_real = {S_real:.5f} deg/deg")
    print()
    print("running Monte Carlo null (shuffle phi, preserve theta)...")

    S_null = mc_null_jackknife(theta, phi, n_sims=N_MC)

    if S_null.size == 0:
        print("Monte Carlo produced no valid realisations (too few FRBs in shell).")
        print("=" * 69)
        print("test 36 complete.")
        print("=" * 69)
        return

    p_S = np.mean(S_null >= S_real)

    print("-" * 66)
    print("MONTE CARLO RESULTS:")
    print(f"  null mean S    = {np.mean(S_null):.5f}")
    print(f"  null std  S    = {np.std(S_null):.5f}")
    print(f"  observed S_real = {S_real:.5f}")
    print(f"  p-value(S_null >= S_real) = {p_S:.6f}")
    print("-" * 66)
    print("interpretation:")
    print("  - S_real measures how much the pitch k changes when different")
    print("    sky quadrants are removed.")
    print("  - small S_real compared to null (p << 0.5) => helicity is more")
    print("    stable than random: no single patch dominates the twist.")
    print("  - large S_real compared to null (p ~ 1) => helicity is more")
    print("    unstable than random: one or more regions strongly control k.")
    print("=" * 69)
    print("test 36 complete.")
    print("=" * 69)


if __name__ == "__main__":
    main()
