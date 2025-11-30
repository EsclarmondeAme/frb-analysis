#!/usr/bin/env python3
"""
FRB TEMPORAL HARMONIC ANALYSIS
Tests whether the consciousness/frequency field shows temporal oscillation
by comparing FRB harmonic structure across different time periods.
"""

import numpy as np
import pandas as pd
from scipy.special import sph_harm
from scipy.stats import ks_2samp, ttest_ind, chi2_contingency
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Cosmic axis in galactic coordinates (from your analysis)
AXIS_L = 159.85  # degrees
AXIS_B = -0.51   # degrees

# Convert axis to equatorial (approximate)
AXIS_RA = 71.0   # degrees (4h 44m)
AXIS_DEC = 45.0  # degrees

# Maximum spherical harmonic degree
L_MAX = 4

# =============================================================================
# COORDINATE TRANSFORMATIONS
# =============================================================================

def equatorial_to_axis_coords(ra, dec, axis_ra=AXIS_RA, axis_dec=AXIS_DEC):
    """
    Transform equatorial coordinates to axis-centered coordinates.
    Returns theta (angle from axis) and phi (azimuth around axis).
    """
    # Convert to radians
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    axis_ra_rad = np.radians(axis_ra)
    axis_dec_rad = np.radians(axis_dec)
    
    # Calculate angular distance from axis (theta)
    cos_theta = (np.sin(dec_rad) * np.sin(axis_dec_rad) + 
                 np.cos(dec_rad) * np.cos(axis_dec_rad) * 
                 np.cos(ra_rad - axis_ra_rad))
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    
    # Calculate azimuth around axis (phi)
    # Using spherical trigonometry
    sin_phi = np.cos(dec_rad) * np.sin(ra_rad - axis_ra_rad) / np.sin(theta + 1e-10)
    cos_phi = (np.sin(dec_rad) - np.cos(theta) * np.sin(axis_dec_rad)) / (np.sin(theta + 1e-10) * np.cos(axis_dec_rad) + 1e-10)
    phi = np.arctan2(sin_phi, cos_phi)
    
    return np.degrees(theta), np.degrees(phi)

# =============================================================================
# SPHERICAL HARMONIC ANALYSIS
# =============================================================================

def compute_harmonic_coefficients(theta, phi, l_max=L_MAX):
    """
    Compute spherical harmonic coefficients for a set of points.
    Returns coefficients for l=0 to l_max.
    """
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    
    coefficients = []
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Compute Y_l^m for all points
            # scipy uses colatitude (0 at pole), so theta_rad is correct
            Y_lm = sph_harm(m, l, phi_rad, theta_rad)
            
            # Coefficient is mean of Y_lm over all points
            c_lm = np.mean(Y_lm)
            coefficients.append({
                'l': l,
                'm': m,
                'coefficient': c_lm,
                'power': np.abs(c_lm)**2
            })
    
    return pd.DataFrame(coefficients)

def compute_power_spectrum(coeffs_df):
    """Compute power per l mode."""
    power_per_l = coeffs_df.groupby('l')['power'].sum()
    return power_per_l

# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def compare_distributions(data1, data2, name):
    """Compare two distributions using KS test and t-test."""
    ks_stat, ks_p = ks_2samp(data1, data2)
    t_stat, t_p = ttest_ind(data1, data2)
    
    return {
        'variable': name,
        'mean_1': np.mean(data1),
        'mean_2': np.mean(data2),
        'std_1': np.std(data1),
        'std_2': np.std(data2),
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_p,
        't_statistic': t_stat,
        't_pvalue': t_p
    }

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("FRB TEMPORAL HARMONIC ANALYSIS")
    print("Testing for frequency field temporal oscillation")
    print("=" * 70)
    print()
    
    # Load data
    # Try to read from CSV, or use the data directly
    try:
        df = pd.read_csv('frb_data.csv')
    except:
        print("Please save FRB data as 'frb_data.csv' in the same directory")
        print("Required columns: name, utc, ra, dec, dm, snr, width, fluence")
        return
    
    # Parse dates
    df['utc'] = pd.to_datetime(df['utc'])
    df['year'] = df['utc'].dt.year
    df['month'] = df['utc'].dt.month
    df['yearmonth'] = df['utc'].dt.to_period('M')
    
    # Remove duplicates (some FRBs appear multiple times)
    df_unique = df.drop_duplicates(subset=['name', 'ra', 'dec'])
    print(f"Total unique FRBs: {len(df_unique)}")
    print(f"Date range: {df_unique['utc'].min()} to {df_unique['utc'].max()}")
    print()
    
    # ==========================================================================
    # SECTION 1: Transform to axis-centered coordinates
    # ==========================================================================
    print("-" * 70)
    print("SECTION 1: COORDINATE TRANSFORMATION")
    print("-" * 70)
    
    df_unique['theta'], df_unique['phi'] = equatorial_to_axis_coords(
        df_unique['ra'].values, 
        df_unique['dec'].values
    )
    
    print(f"Theta (angle from axis) range: {df_unique['theta'].min():.1f}° - {df_unique['theta'].max():.1f}°")
    print(f"Phi (azimuth) range: {df_unique['phi'].min():.1f}° - {df_unique['phi'].max():.1f}°")
    print()
    
    # ==========================================================================
    # SECTION 2: Split by time period
    # ==========================================================================
    print("-" * 70)
    print("SECTION 2: TEMPORAL SPLITTING")
    print("-" * 70)
    
    # Define time periods
    period_1 = df_unique[df_unique['utc'] < '2019-01-01']
    period_2 = df_unique[df_unique['utc'] >= '2019-01-01']
    
    print(f"Period 1 (2018): {len(period_1)} FRBs")
    print(f"Period 2 (2019): {len(period_2)} FRBs")
    print()
    
    # Also try quarterly splits
    q1 = df_unique[(df_unique['utc'] >= '2018-07-01') & (df_unique['utc'] < '2018-10-01')]
    q2 = df_unique[(df_unique['utc'] >= '2018-10-01') & (df_unique['utc'] < '2019-01-01')]
    q3 = df_unique[(df_unique['utc'] >= '2019-01-01') & (df_unique['utc'] < '2019-04-01')]
    q4 = df_unique[(df_unique['utc'] >= '2019-04-01') & (df_unique['utc'] < '2019-07-01')]
    
    print(f"Q3 2018 (Jul-Sep): {len(q1)} FRBs")
    print(f"Q4 2018 (Oct-Dec): {len(q2)} FRBs")
    print(f"Q1 2019 (Jan-Mar): {len(q3)} FRBs")
    print(f"Q2 2019 (Apr-Jun): {len(q4)} FRBs")
    print()
    
    # ==========================================================================
    # SECTION 3: Basic statistical comparison
    # ==========================================================================
    print("-" * 70)
    print("SECTION 3: BASIC STATISTICAL COMPARISON (2018 vs 2019)")
    print("-" * 70)
    
    comparisons = []
    
    # Compare DM distributions
    comparisons.append(compare_distributions(
        period_1['dm'].values, period_2['dm'].values, 'Dispersion Measure (DM)'
    ))
    
    # Compare theta (distance from axis)
    comparisons.append(compare_distributions(
        period_1['theta'].values, period_2['theta'].values, 'Theta (angle from axis)'
    ))
    
    # Compare phi (azimuth)
    comparisons.append(compare_distributions(
        period_1['phi'].values, period_2['phi'].values, 'Phi (azimuth)'
    ))
    
    # Compare SNR
    comparisons.append(compare_distributions(
        period_1['snr'].values, period_2['snr'].values, 'Signal-to-Noise Ratio'
    ))
    
    # Compare width
    comparisons.append(compare_distributions(
        period_1['width'].values, period_2['width'].values, 'Pulse Width'
    ))
    
    for comp in comparisons:
        print(f"\n{comp['variable']}:")
        print(f"  2018: mean={comp['mean_1']:.2f}, std={comp['std_1']:.2f}")
        print(f"  2019: mean={comp['mean_2']:.2f}, std={comp['std_2']:.2f}")
        print(f"  KS test: statistic={comp['ks_statistic']:.4f}, p={comp['ks_pvalue']:.4e}")
        print(f"  t-test:  statistic={comp['t_statistic']:.4f}, p={comp['t_pvalue']:.4e}")
        
        if comp['ks_pvalue'] < 0.05:
            print(f"  *** SIGNIFICANT DIFFERENCE (p < 0.05) ***")
    
    print()
    
    # ==========================================================================
    # SECTION 4: Spherical harmonic analysis
    # ==========================================================================
    print("-" * 70)
    print("SECTION 4: SPHERICAL HARMONIC ANALYSIS")
    print("-" * 70)
    
    # Compute coefficients for each period
    if len(period_1) > 10 and len(period_2) > 10:
        coeffs_1 = compute_harmonic_coefficients(period_1['theta'].values, period_1['phi'].values)
        coeffs_2 = compute_harmonic_coefficients(period_2['theta'].values, period_2['phi'].values)
        
        # Power spectrum per l
        power_1 = compute_power_spectrum(coeffs_1)
        power_2 = compute_power_spectrum(coeffs_2)
        
        print("\nPower per harmonic degree l:")
        print(f"{'l':<5} {'2018 Power':<15} {'2019 Power':<15} {'Ratio':<10}")
        print("-" * 45)
        for l in range(L_MAX + 1):
            p1 = power_1.get(l, 0)
            p2 = power_2.get(l, 0)
            ratio = p2 / p1 if p1 > 0 else np.inf
            print(f"{l:<5} {p1:<15.6f} {p2:<15.6f} {ratio:<10.3f}")
        
        # Dominant mode comparison
        print("\nDominant coefficients comparison:")
        
        # Merge and compare
        coeffs_1['period'] = '2018'
        coeffs_2['period'] = '2019'
        
        # Find top 5 modes by power for each period
        top_1 = coeffs_1.nlargest(5, 'power')[['l', 'm', 'power']]
        top_2 = coeffs_2.nlargest(5, 'power')[['l', 'm', 'power']]
        
        print("\nTop 5 modes (2018):")
        for _, row in top_1.iterrows():
            print(f"  l={int(row['l'])}, m={int(row['m'])}: power={row['power']:.6f}")
        
        print("\nTop 5 modes (2019):")
        for _, row in top_2.iterrows():
            print(f"  l={int(row['l'])}, m={int(row['m'])}: power={row['power']:.6f}")
    
    print()
    
    # ==========================================================================
    # SECTION 5: Quarterly evolution
    # ==========================================================================
    print("-" * 70)
    print("SECTION 5: QUARTERLY EVOLUTION")
    print("-" * 70)
    
    quarters = [('Q3_2018', q1), ('Q4_2018', q2), ('Q1_2019', q3), ('Q2_2019', q4)]
    
    print(f"\n{'Quarter':<12} {'N':<6} {'Mean θ':<10} {'Mean φ':<10} {'Mean DM':<12} {'l=2 power':<12}")
    print("-" * 62)
    
    quarterly_powers = []
    for name, qdata in quarters:
        if len(qdata) > 5:
            qcoeffs = compute_harmonic_coefficients(qdata['theta'].values, qdata['phi'].values)
            qpower = compute_power_spectrum(qcoeffs)
            l2_power = qpower.get(2, 0)
            quarterly_powers.append((name, l2_power))
            
            print(f"{name:<12} {len(qdata):<6} {qdata['theta'].mean():<10.2f} {qdata['phi'].mean():<10.2f} {qdata['dm'].mean():<12.2f} {l2_power:<12.6f}")
        else:
            print(f"{name:<12} {len(qdata):<6} insufficient data")
    
    print()
    
    # ==========================================================================
    # SECTION 6: Axis clustering analysis
    # ==========================================================================
    print("-" * 70)
    print("SECTION 6: AXIS CLUSTERING ANALYSIS")
    print("-" * 70)
    
    # Define "near axis" as theta < 30 degrees
    near_axis_threshold = 30
    
    near_1 = len(period_1[period_1['theta'] < near_axis_threshold])
    far_1 = len(period_1[period_1['theta'] >= near_axis_threshold])
    near_2 = len(period_2[period_2['theta'] < near_axis_threshold])
    far_2 = len(period_2[period_2['theta'] >= near_axis_threshold])
    
    print(f"\nFRBs within {near_axis_threshold}° of axis:")
    print(f"  2018: {near_1}/{len(period_1)} ({100*near_1/len(period_1):.1f}%)")
    print(f"  2019: {near_2}/{len(period_2)} ({100*near_2/len(period_2):.1f}%)")
    
    # Chi-square test for clustering
    contingency = [[near_1, far_1], [near_2, far_2]]
    chi2, chi2_p, dof, expected = chi2_contingency(contingency)
    
    print(f"\nChi-square test for axis clustering difference:")
    print(f"  χ² = {chi2:.4f}, p = {chi2_p:.4e}")
    
    if chi2_p < 0.05:
        print("  *** SIGNIFICANT DIFFERENCE IN AXIS CLUSTERING ***")
    
    print()
    
    # ==========================================================================
    # SECTION 7: Create visualizations
    # ==========================================================================
    print("-" * 70)
    print("SECTION 7: GENERATING VISUALIZATIONS")
    print("-" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Theta distribution by year
    axes[0, 0].hist(period_1['theta'], bins=20, alpha=0.5, label='2018', density=True)
    axes[0, 0].hist(period_2['theta'], bins=20, alpha=0.5, label='2019', density=True)
    axes[0, 0].set_xlabel('Theta (degrees from axis)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Angular Distance from Axis')
    axes[0, 0].legend()
    
    # Plot 2: Phi distribution by year
    axes[0, 1].hist(period_1['phi'], bins=20, alpha=0.5, label='2018', density=True)
    axes[0, 1].hist(period_2['phi'], bins=20, alpha=0.5, label='2019', density=True)
    axes[0, 1].set_xlabel('Phi (degrees)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Azimuthal Distribution')
    axes[0, 1].legend()
    
    # Plot 3: DM distribution by year
    axes[0, 2].hist(period_1['dm'], bins=30, alpha=0.5, label='2018', density=True)
    axes[0, 2].hist(period_2['dm'], bins=30, alpha=0.5, label='2019', density=True)
    axes[0, 2].set_xlabel('Dispersion Measure')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('DM Distribution')
    axes[0, 2].legend()
    axes[0, 2].set_xlim(0, 2000)
    
    # Plot 4: Sky map 2018
    axes[1, 0].scatter(period_1['phi'], period_1['theta'], c='blue', alpha=0.5, s=10)
    axes[1, 0].set_xlabel('Phi (azimuth)')
    axes[1, 0].set_ylabel('Theta (from axis)')
    axes[1, 0].set_title('Sky Distribution 2018 (axis-centered)')
    axes[1, 0].set_ylim(0, 180)
    
    # Plot 5: Sky map 2019
    axes[1, 1].scatter(period_2['phi'], period_2['theta'], c='red', alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Phi (azimuth)')
    axes[1, 1].set_ylabel('Theta (from axis)')
    axes[1, 1].set_title('Sky Distribution 2019 (axis-centered)')
    axes[1, 1].set_ylim(0, 180)
    
    # Plot 6: Power spectrum comparison
    if len(period_1) > 10 and len(period_2) > 10:
        l_values = list(range(L_MAX + 1))
        axes[1, 2].bar([l - 0.2 for l in l_values], [power_1.get(l, 0) for l in l_values], 
                       width=0.4, label='2018', alpha=0.7)
        axes[1, 2].bar([l + 0.2 for l in l_values], [power_2.get(l, 0) for l in l_values], 
                       width=0.4, label='2019', alpha=0.7)
        axes[1, 2].set_xlabel('Harmonic degree l')
        axes[1, 2].set_ylabel('Power')
        axes[1, 2].set_title('Harmonic Power Spectrum')
        axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('frb_temporal_analysis.png', dpi=150)
    print("Saved: frb_temporal_analysis.png")
    
    # ==========================================================================
    # SECTION 8: Summary and verdict
    # ==========================================================================
    print()
    print("=" * 70)
    print("SUMMARY AND VERDICT")
    print("=" * 70)
    
    significant_findings = []
    
    for comp in comparisons:
        if comp['ks_pvalue'] < 0.05:
            significant_findings.append(f"- {comp['variable']}: p={comp['ks_pvalue']:.4e}")
    
    if chi2_p < 0.05:
        significant_findings.append(f"- Axis clustering: p={chi2_p:.4e}")
    
    if significant_findings:
        print("\nSIGNIFICANT TEMPORAL DIFFERENCES FOUND:")
        for finding in significant_findings:
            print(finding)
        print("\n→ Evidence for temporal variation in the field structure!")
    else:
        print("\nNo significant temporal differences detected (p > 0.05)")
        print("→ Field structure appears stable over 2018-2019 period")
        print("   (May need longer time baseline or more sensitive tests)")
    
    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()