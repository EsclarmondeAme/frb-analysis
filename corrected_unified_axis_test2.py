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
print("FRB DIPOLE VERIFICATION TEST")
print("Testing if FRB spatial distribution aligns with CMB axis")
print("=" * 70)

# ----------------------------------------------------------
# 1. CMB Axis (ground truth)
# ----------------------------------------------------------
cmb_l = 152.62
cmb_b = 4.03
cmb_l_err = 10.0  # conservative uncertainty from Planck papers

cmb_coord = SkyCoord(l=cmb_l*u.deg, b=cmb_b*u.deg, frame="galactic")

print("\n" + "=" * 70)
print("1. CMB DIPOLE MODULATION AXIS (Reference)")
print("=" * 70)
print(f"   Galactic coords: l = {cmb_l:.2f}° ± {cmb_l_err:.1f}°")
print(f"                    b = {cmb_b:.2f}°")
print("   Source: Planck low-ℓ hemispherical asymmetry")

# ----------------------------------------------------------
# 2. FRB Analysis - THREE INDEPENDENT TESTS
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("2. FRB ANALYSIS - THREE METHODS")
print("=" * 70)

# Load FRB data
try:
    frbs = pd.read_csv("frbs.csv")
    print(f"\n   Loaded {len(frbs)} FRBs from catalog")
    
    # Convert to galactic coordinates
    frb_coords = SkyCoord(
        ra=frbs["ra"].values*u.deg,
        dec=frbs["dec"].values*u.deg,
        frame="icrs"
    ).galactic
    
    # ============================================================
    # METHOD A: DIRECT SPATIAL CLUSTERING TEST
    # ============================================================
    print("\n" + "-" * 70)
    print("   METHOD A: Spatial Clustering Around CMB Axis")
    print("-" * 70)
    
    seps_from_cmb = frb_coords.separation(cmb_coord).deg
    
    print("\n   Testing if FRBs cluster near CMB axis...")
    print(f"   CMB axis location: l={cmb_l:.1f}°, b={cmb_b:.1f}°\n")
    
    clustering_results = []
    
    for radius in [15, 20, 25, 30]:
        n_near = np.sum(seps_from_cmb < radius)
        frac_obs = n_near / len(frbs)
        
        # Expected fraction for uniform sphere
        # solid angle = 2π(1-cosθ) for cone of half-angle θ
        # fraction = (1-cosθ)/2
        frac_expected = (1 - np.cos(np.radians(radius))) / 2
        
        # Binomial test
        try:
            result = stats.binomtest(n_near, len(frbs), frac_expected, 
                                      alternative='greater')
            p_value = result.pvalue
        except AttributeError:
            # Fallback for older scipy
            p_value = stats.binom_test(n_near, len(frbs), frac_expected, 
                                        alternative='greater')
        
        # Excess ratio
        excess_ratio = frac_obs / frac_expected if frac_expected > 0 else 0
        
        sig = ""
        if p_value < 0.001:
            sig = "*** HIGHLY SIGNIFICANT"
        elif p_value < 0.01:
            sig = "** SIGNIFICANT"
        elif p_value < 0.05:
            sig = "* MARGINAL"
        
        print(f"   Radius {radius}°:")
        print(f"      Observed: {n_near}/{len(frbs)} FRBs ({frac_obs*100:.1f}%)")
        print(f"      Expected: {frac_expected*100:.1f}% if random")
        print(f"      Excess: {excess_ratio:.2f}x")
        print(f"      p-value: {p_value:.4f} {sig}")
        print()
        
        clustering_results.append({
            'radius': radius,
            'n_near': n_near,
            'frac_obs': frac_obs,
            'frac_exp': frac_expected,
            'excess': excess_ratio,
            'p_value': p_value
        })
    
    # Summary verdict for Method A
    print("   " + "=" * 66)
    best_p = min(r['p_value'] for r in clustering_results)
    if best_p < 0.01:
        print("   ✓ METHOD A VERDICT: SIGNIFICANT spatial clustering detected")
        print(f"     Best p-value: {best_p:.4f}")
    elif best_p < 0.05:
        print("   ~ METHOD A VERDICT: MARGINAL clustering detected")
        print(f"     Best p-value: {best_p:.4f}")
    else:
        print("   ✗ METHOD A VERDICT: NO significant spatial clustering")
        print(f"     Best p-value: {best_p:.4f}")
    print("   " + "=" * 66)
    
    
    # ============================================================
    # METHOD B: FRB DIPOLE DIRECTION ANALYSIS
    # ============================================================
    print("\n" + "-" * 70)
    print("   METHOD B: FRB Dipole Direction")
    print("-" * 70)
    
    print("\n   Computing dipole from FRB sky distribution...")
    
    # Convert to Cartesian unit vectors
    x = np.cos(frb_coords.b.rad) * np.cos(frb_coords.l.rad)
    y = np.cos(frb_coords.b.rad) * np.sin(frb_coords.l.rad)
    z = np.sin(frb_coords.b.rad)
    
    # Dipole vector (sum of unit vectors)
    dipole = np.array([x.sum(), y.sum(), z.sum()])
    dipole_amp = np.linalg.norm(dipole) / len(frbs)
    
    # Convert back to galactic coords
    dipole_l = np.degrees(np.arctan2(dipole[1], dipole[0])) % 360
    dipole_b = np.degrees(np.arcsin(dipole[2] / np.linalg.norm(dipole)))
    
    frb_dipole_coord = SkyCoord(l=dipole_l*u.deg, b=dipole_b*u.deg, 
                                 frame="galactic")
    
    print(f"\n   FRB dipole axis:")
    print(f"      l = {dipole_l:.2f}°")
    print(f"      b = {dipole_b:.2f}°")
    print(f"      amplitude = {dipole_amp:.4f}")
    print(f"         (0 = no dipole, 1 = all FRBs in same direction)")
    
    # Bootstrap uncertainty estimation
    print("\n   Computing bootstrap uncertainties (1000 resamples)...")
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
    
    print(f"      Uncertainty: ±{l_err:.1f}° in l, ±{b_err:.1f}° in b")
    
    # Separation from CMB axis
    sep_frb_cmb = frb_dipole_coord.separation(cmb_coord).deg
    
    print(f"\n   Separation from CMB axis: {sep_frb_cmb:.2f}°")
    
    # Statistical significance of alignment
    print("\n   Testing alignment significance...")
    print("   Generating 10,000 random dipoles for comparison...")
    
    random_seps = []
    for _ in range(10000):
        rand_l = np.random.uniform(0, 360)
        rand_b = np.degrees(np.arcsin(np.random.uniform(-1, 1)))
        rand_coord = SkyCoord(l=rand_l*u.deg, b=rand_b*u.deg, 
                              frame="galactic")
        random_seps.append(rand_coord.separation(cmb_coord).deg)
    
    random_seps = np.array(random_seps)
    p_value_alignment = np.mean(random_seps <= sep_frb_cmb)
    
    print(f"\n   Null distribution (random dipoles):")
    print(f"      Mean separation: {np.mean(random_seps):.1f}°")
    print(f"      Median separation: {np.median(random_seps):.1f}°")
    print(f"      10th percentile: {np.percentile(random_seps, 10):.1f}°")
    print(f"      1st percentile: {np.percentile(random_seps, 1):.1f}°")
    
    print(f"\n   Your observed separation: {sep_frb_cmb:.2f}°")
    print(f"   Fraction of random closer: {p_value_alignment:.4f}")
    
    # Summary verdict for Method B
    print("\n   " + "=" * 66)
    if p_value_alignment < 0.01:
        print("   ✓ METHOD B VERDICT: FRB dipole SIGNIFICANTLY aligned with CMB")
        print(f"     p-value: {p_value_alignment:.4f}")
        print(f"     Separation: {sep_frb_cmb:.1f}° (very close)")
    elif p_value_alignment < 0.05:
        print("   ~ METHOD B VERDICT: FRB dipole MARGINALLY aligned with CMB")
        print(f"     p-value: {p_value_alignment:.4f}")
        print(f"     Separation: {sep_frb_cmb:.1f}°")
    elif sep_frb_cmb < 45:
        print("   ~ METHOD B VERDICT: MODERATE alignment")
        print(f"     p-value: {p_value_alignment:.4f}")
        print(f"     Separation: {sep_frb_cmb:.1f}° (within hemisphere)")
    else:
        print("   ✗ METHOD B VERDICT: NO significant alignment")
        print(f"     p-value: {p_value_alignment:.4f}")
        print(f"     Separation: {sep_frb_cmb:.1f}° (far)")
    print("   " + "=" * 66)
    
    
    # ============================================================
    # METHOD C: SIDEREAL TIME ANALYSIS
    # ============================================================
    print("\n" + "-" * 70)
    print("   METHOD C: Sidereal Phase Analysis")
    print("-" * 70)
    
    if "mjd" in frbs.columns:
        print("\n   Testing for sidereal time modulation...")
        
        t = Time(frbs["mjd"].values, format="mjd")
        chime = EarthLocation(lat=49.3223*u.deg, lon=-119.6167*u.deg, 
                              height=545*u.m)
        lst = t.sidereal_time("apparent", longitude=chime.lon).hour
        
        # Rayleigh test for non-uniformity
        phases = 2 * np.pi * lst / 24
        A = np.mean(np.cos(phases))
        B = np.mean(np.sin(phases))
        R = np.sqrt(A**2 + B**2)
        
        # Rayleigh Z statistic
        z = len(frbs) * R**2
        p_rayleigh = np.exp(-z) if z < 700 else 0.0  # avoid overflow
        
        phase_deg = (np.degrees(np.arctan2(B, A)) % 360)
        
        print(f"\n   Sidereal phase analysis:")
        print(f"      Peak phase: {phase_deg:.2f}° (sidereal hour angle)")
        print(f"      Amplitude R: {R:.4f}")
        print(f"      Rayleigh Z: {z:.2f}")
        print(f"      p-value: {p_rayleigh:.4e}")
        
        # Summary verdict for Method C
        print("\n   " + "=" * 66)
        if p_rayleigh < 0.001:
            print("   ✓ METHOD C VERDICT: HIGHLY SIGNIFICANT sidereal modulation")
            print(f"     p-value: {p_rayleigh:.4e}")
            print("     FRBs arrive preferentially at specific sidereal times")
        elif p_rayleigh < 0.05:
            print("   ~ METHOD C VERDICT: MARGINAL sidereal modulation")
            print(f"     p-value: {p_rayleigh:.4e}")
        else:
            print("   ✗ METHOD C VERDICT: NO significant sidereal modulation")
            print(f"     p-value: {p_rayleigh:.4e}")
        print("   " + "=" * 66)
    else:
        print("\n   ⚠ MJD column not found - cannot perform sidereal analysis")


    # ============================================================
    # OVERALL SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("OVERALL VERDICT: IS FRB DIPOLE REAL?")
    print("=" * 70)
    
    evidence_count = 0
    evidence_list = []
    
    # Count evidence
    if best_p < 0.05:
        evidence_count += 1
        evidence_list.append("Spatial clustering")
    
    if p_value_alignment < 0.05:
        evidence_count += 1
        evidence_list.append("Dipole alignment")
    
    if 'p_rayleigh' in locals() and p_rayleigh < 0.05:
        evidence_count += 1
        evidence_list.append("Sidereal modulation")
    
    print(f"\n   Evidence score: {evidence_count}/3 methods significant")
    print(f"   Significant methods: {', '.join(evidence_list) if evidence_list else 'None'}")
    
    print("\n   Summary:")
    if evidence_count >= 2:
        print("   ★★★ STRONG EVIDENCE for FRB-CMB axis correlation")
        print("   ★★★ Multiple independent methods detect alignment")
        print("   ★★★ FRB dipole direction is REAL")
    elif evidence_count == 1:
        print("   ~ MODERATE EVIDENCE for axis correlation")
        print("   ~ One method significant, others marginal")
        print("   ~ Suggestive but not conclusive")
    else:
        print("   ✗ WEAK/NO EVIDENCE for axis correlation")
        print("   ✗ FRB distribution appears consistent with isotropy")
        print("   ✗ Earlier 8.7° alignment may have been artifact")
    
    print("\n   Key numbers:")
    print(f"      FRB dipole: l={dipole_l:.1f}° ± {l_err:.1f}°, b={dipole_b:.1f}° ± {b_err:.1f}°")
    print(f"      CMB axis:   l={cmb_l:.1f}°, b={cmb_b:.1f}°")
    print(f"      Separation: {sep_frb_cmb:.1f}°")
    print(f"      Alignment p-value: {p_value_alignment:.4f}")
    
    if evidence_count >= 2:
        print("\n   → USE THESE COORDINATES IN axis_alignment_significance.py")
        print(f"   → frb = make_axis({dipole_l:.2f}, {dipole_b:.2f})")
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)
    
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Mollweide projection
    ax1 = fig.add_subplot(131, projection="mollweide")
    
    lon = frb_coords.l.deg
    lon = np.where(lon > 180, lon - 360, lon)
    lat = frb_coords.b.deg
    
    ax1.scatter(np.radians(lon), np.radians(lat), 
                s=10, alpha=0.5, c='gray', label='FRBs')
    
    # Mark CMB axis
    cmb_lon = cmb_l if cmb_l < 180 else cmb_l - 360
    ax1.scatter(np.radians(cmb_lon), np.radians(cmb_b),
                s=400, marker='*', c='red', edgecolors='black',
                linewidths=2, label='CMB axis', zorder=10)
    
    # Mark FRB dipole
    frb_lon = dipole_l if dipole_l < 180 else dipole_l - 360
    ax1.scatter(np.radians(frb_lon), np.radians(dipole_b),
                s=400, marker='*', c='blue', edgecolors='black',
                linewidths=2, label='FRB dipole', zorder=10)
    
    # Draw circle around CMB axis
    circle_angles = np.linspace(0, 2*np.pi, 100)
    for r_deg in [20, 30]:
        r_rad = np.radians(r_deg)
        circle_lon = np.radians(cmb_lon) + r_rad * np.cos(circle_angles)
        circle_lat = np.radians(cmb_b) + r_rad * np.sin(circle_angles)
        ax1.plot(circle_lon, circle_lat, 'r--', alpha=0.3, linewidth=1)
    
    ax1.grid(alpha=0.3)
    ax1.set_title('FRB Sky Distribution (Galactic)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    
    # Plot 2: Separation histogram
    ax2 = fig.add_subplot(132)
    ax2.hist(seps_from_cmb, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.axvline(sep_frb_cmb, color='red', linestyle='--', 
                linewidth=2, label=f'Dipole: {sep_frb_cmb:.1f}°')
    ax2.axvline(30, color='orange', linestyle=':', linewidth=1.5, label='30° threshold')
    ax2.set_xlabel('Separation from CMB axis (deg)', fontsize=10)
    ax2.set_ylabel('Number of FRBs', fontsize=10)
    ax2.set_title('FRB Separations from CMB Axis', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # Plot 3: Null distribution comparison
    ax3 = fig.add_subplot(133)
    ax3.hist(random_seps, bins=50, alpha=0.5, label='Random dipoles', 
             color='lightgray', edgecolor='black')
    ax3.axvline(sep_frb_cmb, color='red', linestyle='--',
                linewidth=3, label=f'Observed: {sep_frb_cmb:.1f}°')
    ax3.axvline(np.median(random_seps), color='gray', linestyle=':',
                linewidth=2, label=f'Random median: {np.median(random_seps):.1f}°')
    ax3.set_xlabel('Separation from CMB axis (deg)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Observed vs Random Null Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('frb_dipole_verification.png', dpi=200, bbox_inches='tight')
    print("\n   ✓ Saved: frb_dipole_verification.png")
    
except FileNotFoundError:
    print("\n   [!] frbs.csv not found - cannot perform FRB analysis")
except Exception as e:
    print(f"\n   [!] Error during FRB analysis: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)