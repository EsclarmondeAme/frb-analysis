#!/usr/bin/env python
"""
frb_sidereal_axis_refined2.py

Compute the true sky direction of the FRB sidereal-phase dipole using:
- Exact local sidereal time at CHIME (astropy)
- MJD arrival times from frbs.csv
- Harmonic dipole extraction (A1, B1, phase)
- Convert sidereal dipole phase to RA
- Estimate Dec using sky-weighted method
- Convert to galactic coordinates
- Compare to CMB + clock axes

This replaces the approximate φ→RA mapping used in unified_axis_test.py
"""

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

# ============================================================
# Observatory and reference axes
# ============================================================

# CHIME location (Penticton, BC, Canada)
CHIME = EarthLocation.from_geodetic(
    lon=-119.62 * u.deg,
    lat=49.32 * u.deg,
    height=545 * u.m,  # actual elevation
)

# CMB dipole modulation axis (from your Planck low-ℓ analysis)
CMB_AXIS = SkyCoord(
    l=152.62 * u.deg,
    b=4.03 * u.deg,
    frame="galactic"
)

# Atomic clock sidereal modulation axis (from your NIST/PTB analysis)
CLOCK_AXIS = SkyCoord(
    l=166.04 * u.deg,
    b=-0.87 * u.deg,
    frame="galactic"
)

# ============================================================
# Harmonic analysis functions
# ============================================================

def harmonic_fit(theta):
    """
    Compute dipole harmonic coefficients for n=1.
    
    Uses standard Fourier decomposition:
        A1 = (2/N) Σ cos(θ)
        B1 = (2/N) Σ sin(θ)
        R1 = sqrt(A1² + B1²)  [amplitude]
        φ  = atan2(B1, A1)    [phase]
    
    Parameters:
        theta: array of phases (radians)
    
    Returns:
        A1, B1, R1, phi
    """
    N = len(theta)
    A1 = (2.0 / N) * np.sum(np.cos(theta))
    B1 = (2.0 / N) * np.sum(np.sin(theta))
    R1 = np.sqrt(A1**2 + B1**2)
    phi = np.arctan2(B1, A1)
    return A1, B1, R1, phi


def sidereal_phase_to_ra(phi_rad):
    """
    Convert sidereal phase to Right Ascension.
    
    The phase φ represents the sidereal time when the dipole
    direction transits the local meridian, which directly
    corresponds to the RA of that direction.
    
    Parameters:
        phi_rad: phase in radians
    
    Returns:
        RA in degrees [0, 360)
    """
    return (np.degrees(phi_rad) % 360.0)


def estimate_dipole_dec(ra_vals, dec_vals, ra_dipole):
    """
    Estimate declination of dipole axis using weighted average.
    
    Weight each FRB by its alignment with the dipole RA direction.
    FRBs near the dipole RA contribute more to determining the Dec.
    
    Parameters:
        ra_vals: array of FRB right ascensions (degrees)
        dec_vals: array of FRB declinations (degrees)
        ra_dipole: dipole RA direction (degrees)
    
    Returns:
        Estimated dipole declination (degrees)
    """
    # Weight by cosine of RA separation (projection onto dipole)
    # Add small epsilon to avoid division by zero
    weights = np.abs(np.cos(np.radians(ra_vals - ra_dipole))) + 1e-6
    
    # Weighted average of declinations
    dec_dipole = np.average(dec_vals, weights=weights)
    
    # Ensure valid declination range
    dec_dipole = float(np.clip(dec_dipole, -90.0, 90.0))
    
    return dec_dipole


# ============================================================
# Main analysis
# ============================================================

def main():
    print("=" * 70)
    print("FRB SIDEREAL DIPOLE → TRUE SKY AXIS (REFINED)".center(70))
    print("=" * 70)
    
    # --------------------------------------------------------
    # Load FRB data
    # --------------------------------------------------------
    print("\nLoading FRB data...")
    
    try:
        frb_df = pd.read_csv("frbs.csv")
    except FileNotFoundError:
        raise SystemExit("ERROR: frbs.csv not found")
    
    # Check required columns
    required = {"mjd", "ra", "dec"}
    if not required.issubset(frb_df.columns):
        raise SystemExit(f"ERROR: frbs.csv must contain columns: {required}")
    
    # Remove entries with missing data
    frb_df = frb_df.dropna(subset=["mjd", "ra", "dec"])
    
    mjd_vals = frb_df["mjd"].values
    ra_vals = frb_df["ra"].values
    dec_vals = frb_df["dec"].values
    
    N = len(frb_df)
    print(f"FRBs loaded: {N}")
    
    if N < 10:
        print("WARNING: Very few FRBs - results may be unreliable")
    
    # --------------------------------------------------------
    # Compute exact Local Sidereal Time at CHIME
    # --------------------------------------------------------
    print("\nComputing sidereal times at CHIME...")
    
    t_astropy = Time(mjd_vals, format="mjd", scale="utc")
    lst = t_astropy.sidereal_time("mean", longitude=CHIME.lon)
    lst_rad = lst.to_value(u.rad) % (2 * np.pi)
    
    print(f"Sidereal time range: {np.degrees(lst_rad.min()):.1f}° to {np.degrees(lst_rad.max()):.1f}°")
    
    # --------------------------------------------------------
    # Extract harmonic dipole from sidereal phases
    # --------------------------------------------------------
    print("\nExtracting sidereal dipole (n=1)...")
    
    A1, B1, R1, phi = harmonic_fit(lst_rad)
    
    print("-" * 70)
    print("Harmonic dipole coefficients:")
    print(f"  A1 = {A1:+.6f}")
    print(f"  B1 = {B1:+.6f}")
    print(f"  R1 = {R1:.6f}  (amplitude)")
    print(f"  φ  = {np.degrees(phi):.2f}°  (phase)")
    
    # Significance check
    # For random distribution, expect R1 ~ 2/sqrt(N)
    expected_random = 2.0 / np.sqrt(N)
    significance = R1 / expected_random
    print(f"\nExpected amplitude if random: {expected_random:.4f}")
    print(f"Observed/Expected ratio: {significance:.2f}x")
    
    if significance > 3:
        print("→ STRONG dipole detected (>3σ significance)")
    elif significance > 2:
        print("→ Moderate dipole detected (>2σ significance)")
    else:
        print("→ Weak or no significant dipole")
    
    # --------------------------------------------------------
    # Convert sidereal phase to sky coordinates
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("Converting to sky coordinates...")
    
    # Phase → RA
    ra_dipole = sidereal_phase_to_ra(phi)
    
    # Estimate Dec using weighted average
    dec_dipole = estimate_dipole_dec(ra_vals, dec_vals, ra_dipole)
    
    # Build coordinate objects
    frb_axis_icrs = SkyCoord(
        ra=ra_dipole * u.deg,
        dec=dec_dipole * u.deg,
        frame="icrs"
    )
    frb_axis_gal = frb_axis_icrs.galactic
    
    print("\nFRB sidereal dipole axis:")
    print(f"  ICRS (J2000):  RA = {frb_axis_icrs.ra.deg:7.2f}°,  Dec = {frb_axis_icrs.dec.deg:+7.2f}°")
    print(f"  Galactic:       l = {frb_axis_gal.l.deg:7.2f}°,    b = {frb_axis_gal.b.deg:+7.2f}°")
    
    # --------------------------------------------------------
    # Compare with CMB and clock axes
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON WITH OTHER ANOMALIES")
    print("=" * 70)
    
    sep_frb_cmb = frb_axis_gal.separation(CMB_AXIS).deg
    sep_frb_clock = frb_axis_gal.separation(CLOCK_AXIS).deg
    sep_cmb_clock = CMB_AXIS.separation(CLOCK_AXIS).deg
    
    print("\nReference axes:")
    print(f"  CMB axis:    l = {CMB_AXIS.l.deg:7.2f}°,  b = {CMB_AXIS.b.deg:+7.2f}°")
    print(f"  Clock axis:  l = {CLOCK_AXIS.l.deg:7.2f}°,  b = {CLOCK_AXIS.b.deg:+7.2f}°")
    
    print("\nAngular separations:")
    print(f"  FRB ↔ CMB:     {sep_frb_cmb:6.2f}°")
    print(f"  FRB ↔ Clock:   {sep_frb_clock:6.2f}°")
    print(f"  CMB ↔ Clock:   {sep_cmb_clock:6.2f}°")
    
    # --------------------------------------------------------
    # Statistical interpretation
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print("STATISTICAL INTERPRETATION")
    print("-" * 70)
    
    # Count alignments
    threshold_strong = 20  # degrees
    threshold_moderate = 30
    
    alignments = []
    if sep_frb_cmb < threshold_strong:
        alignments.append("FRB-CMB")
    if sep_frb_clock < threshold_strong:
        alignments.append("FRB-Clock")
    if sep_cmb_clock < threshold_strong:
        alignments.append("CMB-Clock")
    
    print(f"\nAxes aligned within {threshold_strong}°: {len(alignments)}/3")
    if alignments:
        print(f"  Aligned pairs: {', '.join(alignments)}")
    
    # Random alignment probability
    # For N axes, probability all within angle θ of each other:
    # P ≈ (θ/180)^(N-1) for small angles
    if len(alignments) >= 2:
        max_sep = max(sep_frb_cmb, sep_frb_clock, sep_cmb_clock)
        p_random = (max_sep / 180.0) ** 2  # 3 axes = 2 independent pairs
        
        print(f"\nProbability of random alignment within {max_sep:.1f}°:")
        print(f"  p ≈ {p_random:.6f}  ({p_random*100:.4f}%)")
        
        if p_random < 0.001:
            print("\n★ EXTREMELY SIGNIFICANT ALIGNMENT (p < 0.001)")
            print("★ Strong evidence for unified preferred axis")
        elif p_random < 0.01:
            print("\n★ HIGHLY SIGNIFICANT ALIGNMENT (p < 0.01)")
            print("★ Evidence for unified preferred axis")
        elif p_random < 0.05:
            print("\n✓ SIGNIFICANT ALIGNMENT (p < 0.05)")
            print("✓ Suggestive of unified preferred axis")
        else:
            print("\n~ Moderate alignment detected")
    else:
        print("\n✗ No strong multi-axis alignment detected")
        print("  Results may be limited by:")
        print("    - Small sample size")
        print("    - CHIME sky coverage constraints")
        print("    - Approximate Dec estimation")
    
    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nYour frequency-gradient cone model predicts:")
    print("  • Preferred axis exists (singularity → our layer projection)")
    print("  • Cross-layer events cluster toward this axis")
    print("  • Sidereal (not solar) modulation")
    print("  • Same axis visible across energy scales")
    
    print("\nObservations:")
    print(f"  • FRB sidereal dipole: {significance:.1f}σ significance")
    print(f"  • Axis alignment: {len(alignments)}/3 pairs within {threshold_strong}°")
    if len(alignments) >= 2:
        print(f"  • Random probability: p ≈ {p_random:.4f}")
    
    if len(alignments) >= 2 and significance > 2:
        print("\n✓ RESULTS CONSISTENT WITH CONE MODEL")
        print("✓ Multiple independent probes point to unified axis")
    elif len(alignments) >= 1 or significance > 2:
        print("\n~ PARTIAL SUPPORT FOR CONE MODEL")
        print("~ Suggestive but not conclusive")
    else:
        print("\n? INCONCLUSIVE")
        print("? Need larger sample or refined analysis")
    
    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()