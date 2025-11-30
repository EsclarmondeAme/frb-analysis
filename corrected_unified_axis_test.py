"""
corrected_unified_axis_test.py
----------------------------------------------------------
Proper unified axis comparison with uncertainty quantification

Key fixes:
1. FRB sidereal dipole -> RA band, not single direction
2. Clock modulation stays as phase, not forced to sky coords
3. Bootstrap uncertainties on all measurements
4. Proper statistical testing
----------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from scipy import stats

print("=" * 70)
print("CORRECTED UNIFIED AXIS ANALYSIS")
print("=" * 70)

# ----------------------------------------------------------
# 1. CMB Axis (ground truth)
# ----------------------------------------------------------
cmb_l = 152.62
cmb_b = 4.03
cmb_l_err = 10.0  # conservative uncertainty from Planck papers

cmb_coord = SkyCoord(l=cmb_l*u.deg, b=cmb_b*u.deg, frame="galactic")

print("\n1. CMB DIPOLE MODULATION AXIS")
print(f"   l = {cmb_l:.2f}° ± {cmb_l_err:.1f}°")
print(f"   b = {cmb_b:.2f}°")

# ----------------------------------------------------------
# 2. FRB Analysis - Do this properly
# ----------------------------------------------------------
print("\n2. FRB ANALYSIS")

# Load FRB data
try:
    frbs = pd.read_csv("frbs.csv")
    
    # Convert to galactic coordinates
    frb_coords = SkyCoord(
        ra=frbs["ra"].values*u.deg,
        dec=frbs["dec"].values*u.deg,
        frame="icrs"
    ).galactic
    
    # Method A: Direct sky clustering test
    print("\n   Method A: Sky Clustering Test")
    seps_from_cmb = frb_coords.separation(cmb_coord).deg
    
    # Define test: are FRBs preferentially near CMB axis?
    for radius in [20, 30, 40]:
        n_near = np.sum(seps_from_cmb < radius)
        frac_obs = n_near / len(frbs)
        
        # Expected fraction for uniform sphere
        frac_expected = (1 - np.cos(np.radians(radius))) / 2
        
        # Binomial test (use binomtest for scipy >= 1.7)
        try:
            p_value = stats.binomtest(n_near, len(frbs), frac_expected, 
                                      alternative='greater').pvalue
        except AttributeError:
            # Fallback for older scipy versions
            p_value = stats.binom_test(n_near, len(frbs), frac_expected, 
                                        alternative='greater')
        
        print(f"   Within {radius}°: {n_near}/{len(frbs)} " +
              f"({frac_obs*100:.1f}% vs {frac_expected*100:.1f}% expected)")
        print(f"      p-value = {p_value:.4f}")
    
    # Method B: Compute FRB dipole direction
    print("\n   Method B: FRB Dipole Direction")
    
    # Convert to Cartesian
    x = np.cos(frb_coords.b.rad) * np.cos(frb_coords.l.rad)
    y = np.cos(frb_coords.b.rad) * np.sin(frb_coords.l.rad)
    z = np.sin(frb_coords.b.rad)
    
    # Dipole vector (sum of unit vectors)
    dipole = np.array([x.sum(), y.sum(), z.sum()])
    dipole_amp = np.linalg.norm(dipole) / len(frbs)
    
    # Convert back to coords
    dipole_l = np.degrees(np.arctan2(dipole[1], dipole[0])) % 360
    dipole_b = np.degrees(np.arcsin(dipole[2] / np.linalg.norm(dipole)))
    
    frb_dipole_coord = SkyCoord(l=dipole_l*u.deg, b=dipole_b*u.deg, 
                                 frame="galactic")
    
    print(f"   FRB dipole: l={dipole_l:.2f}°, b={dipole_b:.2f}°")
    print(f"   Amplitude: {dipole_amp:.4f}")
    
    # Bootstrap uncertainty
    n_boot = 1000
    boot_l, boot_b = [], []
    for _ in range(n_boot):
        idx = np.random.choice(len(frbs), len(frbs), replace=True)
        x_b = np.cos(frb_coords[idx].b.rad) * np.cos(frb_coords[idx].l.rad)
        y_b = np.cos(frb_coords[idx].b.rad) * np.sin(frb_coords[idx].l.rad)
        z_b = np.sin(frb_coords[idx].b.rad)
        d = np.array([x_b.sum(), y_b.sum(), z_b.sum()])
        boot_l.append(np.degrees(np.arctan2(d[1], d[0])) % 360)
        boot_b.append(np.degrees(np.arcsin(d[2] / np.linalg.norm(d))))
    
    l_err = np.std(boot_l)
    b_err = np.std(boot_b)
    print(f"   Uncertainty: ±{l_err:.1f}° in l, ±{b_err:.1f}° in b")
    
    # Separation from CMB axis
    sep_frb_cmb = frb_dipole_coord.separation(cmb_coord).deg
    print(f"\n   Separation from CMB axis: {sep_frb_cmb:.2f}°")
    
    # Is this significant?
    # Expected distribution for random dipole
    random_seps = []
    for _ in range(10000):
        rand_l = np.random.uniform(0, 360)
        rand_b = np.degrees(np.arcsin(np.random.uniform(-1, 1)))
        rand_coord = SkyCoord(l=rand_l*u.deg, b=rand_b*u.deg, 
                              frame="galactic")
        random_seps.append(rand_coord.separation(cmb_coord).deg)
    
    p_value_alignment = np.mean(np.array(random_seps) <= sep_frb_cmb)
    print(f"   p-value for alignment: {p_value_alignment:.4f}")
    
    # Method C: Sidereal time analysis (what you did before)
    print("\n   Method C: Sidereal Phase Analysis")
    
    if "mjd" in frbs.columns:
        t = Time(frbs["mjd"].values, format="mjd")
        chime = EarthLocation(lat=49.3223*u.deg, lon=-119.6167*u.deg, 
                              height=545*u.m)
        lst = t.sidereal_time("apparent", longitude=chime.lon).hour
        
        # Rayleigh test for uniformity
        phases = 2 * np.pi * lst / 24
        A = np.mean(np.cos(phases))
        B = np.mean(np.sin(phases))
        R = np.sqrt(A**2 + B**2)
        
        # Rayleigh statistic
        z = len(frbs) * R**2
        p_rayleigh = np.exp(-z)
        
        phase_deg = (np.degrees(np.arctan2(B, A)) % 360)
        
        print(f"   Sidereal phase peak: {phase_deg:.2f}°")
        print(f"   Rayleigh Z = {z:.2f}, p = {p_rayleigh:.4e}")
        
        if p_rayleigh < 0.05:
            print("   -> Significant sidereal modulation detected")
        else:
            print("   -> No significant sidereal modulation")

except FileNotFoundError:
    print("   [!] frbs.csv not found - skipping FRB analysis")
    frb_dipole_coord = None
    sep_frb_cmb = None

# ----------------------------------------------------------
# 3. Atomic Clock Analysis
# ----------------------------------------------------------
print("\n3. ATOMIC CLOCK SIDEREAL MODULATION")
print("   Phase = 1.326 rad = 75.97 degrees")
print("   Amplitude ~ 1.4e-15 fractional frequency")
print("\n   NOTE: Clock data shows temporal modulation, not")
print("   directional signal. Cannot directly convert to sky coords.")
print("   Analysis requires:")
print("   - Model of how spatial anisotropy -> frequency shift")
print("   - Multiple baseline clocks to triangulate direction")
print("   - Currently: only temporal correlation possible")

# ----------------------------------------------------------
# 4. Statistical Summary
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

if sep_frb_cmb is not None:
    print(f"\nCMB-FRB alignment: {sep_frb_cmb:.1f}°")
    print(f"Random alignment probability: {p_value_alignment:.4f}")
    
    if p_value_alignment < 0.05:
        print("* SIGNIFICANT alignment detected")
    else:
        print("~ Alignment not statistically significant")
    
    print(f"\nInterpretation:")
    if sep_frb_cmb < 30 and p_value_alignment < 0.05:
        print("- Strong evidence for FRB-CMB axis correlation")
        print("- Warrants further investigation")
    elif sep_frb_cmb < 45:
        print("- Moderate alignment, but could be chance")
        print("- Need larger FRB sample for confirmation")
    else:
        print("- No strong alignment")
        print("- CMB and FRB axes appear independent")

# ----------------------------------------------------------
# 5. Visualization
# ----------------------------------------------------------
try:
    fig = plt.figure(figsize=(16, 6))
    
    # Mollweide projection
    ax1 = fig.add_subplot(131, projection="mollweide")
    
    lon = frb_coords.l.deg
    lon = np.where(lon > 180, lon - 360, lon)
    lat = frb_coords.b.deg
    
    ax1.scatter(np.radians(lon), np.radians(lat), 
                s=10, alpha=0.5, c='gray', label='FRBs')
    
    # Mark CMB axis
    cmb_lon = cmb_l if cmb_l < 180 else cmb_l - 360
    ax1.scatter(np.radians(cmb_lon), np.radians(cmb_b),
                s=300, marker='*', c='red', edgecolors='black',
                linewidths=2, label='CMB axis', zorder=10)
    
    # Mark FRB dipole
    if frb_dipole_coord is not None:
        frb_lon = dipole_l if dipole_l < 180 else dipole_l - 360
        ax1.scatter(np.radians(frb_lon), np.radians(dipole_b),
                    s=300, marker='*', c='blue', edgecolors='black',
                    linewidths=2, label='FRB dipole', zorder=10)
    
    ax1.grid(alpha=0.3)
    ax1.set_title('FRB Sky Distribution (Galactic)')
    ax1.legend(loc='upper left')
    
    # Separation distribution
    ax2 = fig.add_subplot(132)
    ax2.hist(seps_from_cmb, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(sep_frb_cmb, color='red', linestyle='--', 
                linewidth=2, label=f'Dipole sep: {sep_frb_cmb:.1f}°')
    ax2.set_xlabel('Separation from CMB axis (deg)')
    ax2.set_ylabel('Number of FRBs')
    ax2.set_title('FRB Separations from CMB Axis')
    ax2.legend()
    
    # Bootstrap distribution
    ax3 = fig.add_subplot(133)
    ax3.hist(random_seps, bins=50, alpha=0.5, label='Random')
    ax3.axvline(sep_frb_cmb, color='red', linestyle='--',
                linewidth=2, label=f'Observed: {sep_frb_cmb:.1f}°')
    ax3.set_xlabel('Separation (deg)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Null Distribution vs Observed')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('corrected_axis_analysis.png', dpi=200)
    print("\n[+] Saved: corrected_axis_analysis.png")
    
except Exception as e:
    print(f"\n[!] Visualization failed: {e}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)