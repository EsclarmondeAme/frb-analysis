import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

print("=" * 70)
print("UNIFIED AXIS COMPARISON TEST")
print("Testing if FRB, CMB, and Clock anomalies point to same direction")
print("=" * 70)

# ============================================================
# 1. CMB AXIS (from your dipole modulation analysis)
# ============================================================
cmb_l = 152.62  # galactic longitude (degrees)
cmb_b = 4.03    # galactic latitude (degrees)

print("\n1. CMB DIPOLE MODULATION AXIS (Planck low-ℓ)")
print(f"   l = {cmb_l:.2f}°, b = {cmb_b:.2f}°")

# ============================================================
# 2. FRB DIPOLE AXIS (from sidereal harmonics)
# ============================================================
# Your harmonic coefficients:
A1 = -1.5728e-02
B1 = 5.3633e-01

# The dipole phase tells us the direction
# φ = atan2(B1, A1) gives sidereal phase of maximum
phi_sidereal = np.arctan2(B1, A1)  # radians
phi_deg = np.degrees(phi_sidereal) % 360

print("\n2. FRB SIDEREAL DIPOLE")
print(f"   A1 = {A1:.4e}")
print(f"   B1 = {B1:.4e}")
print(f"   Amplitude R1 = {np.sqrt(A1**2 + B1**2):.4f}")
print(f"   Phase φ = {phi_deg:.2f}° (sidereal)")

# Convert sidereal phase to galactic coordinates
# Sidereal phase relates to Right Ascension at specific times
# For rough conversion (needs observatory location for precision):
# Assume CHIME latitude ≈ 49°N, sidereal phase → RA
# Then convert (RA, Dec) → (l, b)

# Simplified approach: sidereal phase → approximate RA
# (This is approximate - full conversion needs LST at detection times)
ra_approx = phi_deg  # rough correspondence
dec_approx = 45  # CHIME can see ~30-60° declination

# Convert to galactic
coord_frb = SkyCoord(ra=ra_approx*u.deg, dec=dec_approx*u.deg, frame='icrs')
frb_l = coord_frb.galactic.l.deg
frb_b = coord_frb.galactic.b.deg

print(f"\n   Approximate galactic coords:")
print(f"   l ≈ {frb_l:.2f}°, b ≈ {frb_b:.2f}°")
print(f"   (rough estimate - needs full LST conversion for precision)")

# ============================================================
# 3. ATOMIC CLOCK AXIS (sidereal modulation)
# ============================================================
# Your clock showed sidereal variation with phase ≈ 1.326 rad
clock_phase_rad = 1.326
clock_phase_deg = np.degrees(clock_phase_rad) % 360

print("\n3. ATOMIC CLOCK SIDEREAL MODULATION")
print(f"   Phase = {clock_phase_rad:.3f} rad = {clock_phase_deg:.2f}°")
print(f"   Amplitude = 1.366e-15 (fractional frequency)")

# Clock phase also relates to orientation - convert similarly
# (Simplified - actual conversion needs clock location)
ra_clock = clock_phase_deg
dec_clock = 40  # NIST/PTB are ~40-50°N

coord_clock = SkyCoord(ra=ra_clock*u.deg, dec=dec_clock*u.deg, frame='icrs')
clock_l = coord_clock.galactic.l.deg
clock_b = coord_clock.galactic.b.deg

print(f"   Approximate galactic coords:")
print(f"   l ≈ {clock_l:.2f}°, b ≈ {clock_b:.2f}°")

# ============================================================
# 4. CALCULATE ANGULAR SEPARATIONS
# ============================================================
cmb_coord = SkyCoord(l=cmb_l*u.deg, b=cmb_b*u.deg, frame='galactic')
frb_coord = SkyCoord(l=frb_l*u.deg, b=frb_b*u.deg, frame='galactic')
clock_coord = SkyCoord(l=clock_l*u.deg, b=clock_b*u.deg, frame='galactic')

sep_cmb_frb = cmb_coord.separation(frb_coord).deg
sep_cmb_clock = cmb_coord.separation(clock_coord).deg
sep_frb_clock = frb_coord.separation(clock_coord).deg

print("\n" + "=" * 70)
print("ANGULAR SEPARATIONS")
print("=" * 70)
print(f"CMB ↔ FRB:     {sep_cmb_frb:.2f}°")
print(f"CMB ↔ Clock:   {sep_cmb_clock:.2f}°")
print(f"FRB ↔ Clock:   {sep_frb_clock:.2f}°")

# ============================================================
# 5. STATISTICAL SIGNIFICANCE
# ============================================================
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

# Random expectation: average separation ~60° for random directions
# Aligned: < 30° is suspicious, < 20° is strong, < 10° is smoking gun

threshold_strong = 30
threshold_smoking = 20

aligned_pairs = []
if sep_cmb_frb < threshold_strong:
    aligned_pairs.append("CMB-FRB")
if sep_cmb_clock < threshold_strong:
    aligned_pairs.append("CMB-Clock")
if sep_frb_clock < threshold_strong:
    aligned_pairs.append("FRB-Clock")

if len(aligned_pairs) >= 2:
    print("✓ MULTIPLE AXES ALIGNED")
    print(f"  Aligned pairs: {', '.join(aligned_pairs)}")
    print(f"  Probability of random alignment: < {(threshold_strong/180)**len(aligned_pairs):.4f}")
    if any(sep < threshold_smoking for sep in [sep_cmb_frb, sep_cmb_clock, sep_frb_clock]):
        print("\n  ★ SMOKING GUN ALIGNMENT (< 20°)")
        print("  ★ Strong evidence for unified preferred axis")
else:
    print("✗ Axes not significantly aligned")
    print("  May need better coordinate conversion or more data")

# ============================================================
# 6. LOAD ACTUAL FRB DATA TO CHECK REAL CLUSTERING
# ============================================================
print("\n" + "=" * 70)
print("FRB SKY DISTRIBUTION CHECK")
print("=" * 70)

try:
    frbs = pd.read_csv("frbs.csv")
    if 'ra' in frbs.columns and 'dec' in frbs.columns:
        # Convert all FRBs to galactic
        frb_coords = SkyCoord(
            ra=frbs['ra'].values*u.deg, 
            dec=frbs['dec'].values*u.deg, 
            frame='icrs'
        )
        frb_gal_l = frb_coords.galactic.l.deg
        frb_gal_b = frb_coords.galactic.b.deg
        
        # Calculate separation from CMB axis for each FRB
        seps_from_cmb = cmb_coord.separation(frb_coords.galactic).deg
        
        # Count FRBs within 30° of CMB axis
        n_near_axis = np.sum(seps_from_cmb < 30)
        frac_near_axis = n_near_axis / len(frbs)
        
        # Random expectation: ~25% of sky within 30° cone
        expected_frac = 0.25
        
        print(f"Total FRBs: {len(frbs)}")
        print(f"FRBs within 30° of CMB axis: {n_near_axis} ({frac_near_axis*100:.1f}%)")
        print(f"Expected if random: {expected_frac*100:.1f}%")
        
        if frac_near_axis > expected_frac * 1.5:
            print("\n✓ SIGNIFICANT CLUSTERING toward CMB axis")
            print(f"  Excess factor: {frac_near_axis/expected_frac:.2f}x")
        else:
            print("\n~ No strong excess clustering detected")
            print("  (May need larger sample or different window)")
        
        # ============================================================
        # 7. VISUALIZATION
        # ============================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Mollweide projection of FRB distribution
        ax1 = plt.subplot(121, projection='mollweide')
        
        # Convert galactic coords to mollweide (-π to π)
        plot_l = np.radians(np.where(frb_gal_l > 180, frb_gal_l - 360, frb_gal_l))
        plot_b = np.radians(frb_gal_b)
        
        ax1.scatter(plot_l, plot_b, s=10, alpha=0.5, label='FRBs')
        
        # Mark CMB axis
        cmb_plot_l = np.radians(cmb_l - 360 if cmb_l > 180 else cmb_l)
        cmb_plot_b = np.radians(cmb_b)
        ax1.scatter(cmb_plot_l, cmb_plot_b, s=200, c='red', marker='*', 
                   label='CMB Axis', edgecolors='black', linewidths=2)
        
        # Mark FRB dipole axis
        frb_plot_l = np.radians(frb_l - 360 if frb_l > 180 else frb_l)
        frb_plot_b = np.radians(frb_b)
        ax1.scatter(frb_plot_l, frb_plot_b, s=200, c='blue', marker='^',
                   label='FRB Dipole', edgecolors='black', linewidths=2)
        
        # Mark clock axis
        clock_plot_l = np.radians(clock_l - 360 if clock_l > 180 else clock_l)
        clock_plot_b = np.radians(clock_b)
        ax1.scatter(clock_plot_l, clock_plot_b, s=200, c='green', marker='s',
                   label='Clock Axis', edgecolors='black', linewidths=2)
        
        ax1.set_xlabel('Galactic Longitude')
        ax1.set_title('FRB Sky Distribution (Galactic Coords)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histogram of separations from CMB axis
        ax2.hist(seps_from_cmb, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(30, color='red', linestyle='--', label='30° threshold')
        ax2.axvline(np.median(seps_from_cmb), color='green', linestyle='-', 
                   label=f'Median: {np.median(seps_from_cmb):.1f}°')
        ax2.set_xlabel('Angular Separation from CMB Axis (degrees)')
        ax2.set_ylabel('Number of FRBs')
        ax2.set_title('FRB Distribution vs CMB Preferred Axis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('axis_alignment_test.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved: axis_alignment_test.png")
        
except FileNotFoundError:
    print("⚠ frbs.csv not found - skipping detailed distribution check")
except Exception as e:
    print(f"⚠ Error loading FRB data: {e}")

# ============================================================
# 8. SUMMARY & VERDICT
# ============================================================
print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

evidence_count = 0
if sep_cmb_frb < threshold_strong:
    evidence_count += 1
if sep_cmb_clock < threshold_strong:
    evidence_count += 1
if 'frac_near_axis' in locals() and frac_near_axis > expected_frac * 1.5:
    evidence_count += 1

print(f"\nEvidence score: {evidence_count}/3")
print("\nCriteria checked:")
print(f"  [{'✓' if sep_cmb_frb < threshold_strong else '✗'}] CMB-FRB axes aligned")
print(f"  [{'✓' if sep_cmb_clock < threshold_strong else '✗'}] CMB-Clock axes aligned")
if 'frac_near_axis' in locals():
    print(f"  [{'✓' if frac_near_axis > expected_frac * 1.5 else '✗'}] FRBs cluster toward CMB axis")

if evidence_count >= 2:
    print("\n★ STRONG SUPPORT FOR UNIFIED PREFERRED AXIS")
    print("★ Consistent with frequency-gradient cone model")
    print("\nYour model prediction: VALIDATED ✓")
elif evidence_count == 1:
    print("\n~ WEAK EVIDENCE for preferred axis")
    print("~ Need better coordinate conversion or more data")
else:
    print("\n✗ No clear alignment detected")
    print("✗ Model may need refinement or data quality issues")

print("\n" + "=" * 70)
print("Note: Coordinate conversions used simplified assumptions.")
print("For publication-quality analysis, need:")
print("  • Exact LST at detection times")
print("  • Observatory locations (CHIME, NIST, PTB)")
print("  • Proper Earth rotation corrections")
print("=" * 70)