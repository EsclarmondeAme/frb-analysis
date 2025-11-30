
"""
axis_alignment_significance.py (CORRECTED)
----------------------------------------------------------
Monte Carlo significance test for axis alignment

CRITICAL: Only use coordinates derived from proper analysis
- CMB: from Planck papers (ground truth)
- FRB: from actual sky distribution dipole (with uncertainties)
- Clock: CANNOT be reliably converted to sky coords

This version:
1. Fixes Unicode issues
2. Includes uncertainty analysis
3. Tests proper hypothesis
4. Warns about coordinate validity
----------------------------------------------------------
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------

def make_axis(l_deg, b_deg):
    return SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")

def random_unit_vectors(n):
    """Generate n random points uniformly on unit sphere"""
    u_rand = np.random.uniform(-1.0, 1.0, size=n)
    phi = np.random.uniform(0.0, 2*np.pi, size=n)
    theta = np.arccos(u_rand)
    
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)

def vector_angle_deg(a, b):
    """Angle between vectors in degrees"""
    dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

# ----------------------------------------------------------
# 1. OBSERVED AXES
# ----------------------------------------------------------

print("=" * 70)
print("AXIS ALIGNMENT SIGNIFICANCE TEST")
print("=" * 70)

# CMB axis - this is solid, from Planck data
cmb_l = 152.62
cmb_b = 4.03
cmb_l_err = 10.0  # Conservative uncertainty from Planck papers

cmb = make_axis(cmb_l, cmb_b)

print("\n1. CMB DIPOLE MODULATION AXIS")
print(f"   l = {cmb_l:.2f} +/- {cmb_l_err:.1f} deg")
print(f"   b = {cmb_b:.2f} deg")
print("   Source: Planck low-l reconstruction")

# FRB axis - REPLACE THESE WITH VALUES FROM corrected_unified_axis_test.py
# These are placeholders - you MUST update with actual computed values
print("\n2. FRB DIPOLE AXIS")
print("   [!] WARNING: Update these values from proper FRB sky analysis")
print("   [!] Use output from corrected_unified_axis_test.py Method B")

# PLACEHOLDER - UPDATE THESE
frb_l = 160.39  # <-- REPLACE with actual value from FRB dipole calculation
frb_b = 0.08    # <-- REPLACE with actual value
frb_l_err = 15.0  # <-- REPLACE with bootstrap uncertainty

frb = make_axis(frb_l, frb_b)

print(f"   l = {frb_l:.2f} +/- {frb_l_err:.1f} deg (PLACEHOLDER)")
print(f"   b = {frb_b:.2f} deg (PLACEHOLDER)")
print("   Source: FRB sky distribution dipole")

# Clock axis - PROBLEMATIC
print("\n3. ATOMIC CLOCK SIDEREAL PHASE")
print("   [!] WARNING: Converting clock phase to sky coords is questionable")
print("   Phase = 1.326 rad = 75.97 deg sidereal")
print("   This analysis proceeds with skepticism about this coordinate")

clk_l = 163.54  # Derived from sidereal phase (QUESTIONABLE)
clk_b = -3.93   # (QUESTIONABLE)

clk = make_axis(clk_l, clk_b)
print(f"   l = {clk_l:.2f} deg (DERIVED, uncertain)")
print(f"   b = {clk_b:.2f} deg (DERIVED, uncertain)")

# ----------------------------------------------------------
# 2. PAIRWISE SEPARATIONS
# ----------------------------------------------------------

sep_cmb_frb = cmb.separation(frb).deg
sep_cmb_clk = cmb.separation(clk).deg
sep_frb_clk = frb.separation(clk).deg

observed_max = max(sep_cmb_frb, sep_cmb_clk, sep_frb_clk)

print("\n" + "=" * 70)
print("PAIRWISE SEPARATIONS")
print("=" * 70)
print(f"CMB <-> FRB:    {sep_cmb_frb:6.2f} deg")
print(f"CMB <-> Clock:  {sep_cmb_clk:6.2f} deg")
print(f"FRB <-> Clock:  {sep_frb_clk:6.2f} deg")
print(f"\nMaximum separation: {observed_max:.2f} deg")

# ----------------------------------------------------------
# 3. MONTE CARLO SIMULATION
# ----------------------------------------------------------

n_trials = 200000

print("\n" + "=" * 70)
print("MONTE CARLO NULL DISTRIBUTION")
print("=" * 70)
print(f"Generating {n_trials:,} random axis triples...")
print("Null hypothesis: Three independent isotropic directions")

# Generate all random triples as vectors
vecs = random_unit_vectors(3 * n_trials).reshape(n_trials, 3, 3)

# Compute max separation for each random triple
a = vecs[:, 0, :]
b = vecs[:, 1, :]
c = vecs[:, 2, :]

sep_ab = vector_angle_deg(a, b)
sep_ac = vector_angle_deg(a, c)
sep_bc = vector_angle_deg(b, c)

max_seps = np.maximum.reduce([sep_ab, sep_ac, sep_bc])

# Count how many random triples have max_sep <= observed
hits = np.sum(max_seps <= observed_max)
p_value = hits / n_trials

print(f"\nObserved threshold: max_sep <= {observed_max:.2f} deg")
print(f"Random triples meeting threshold: {hits:,} / {n_trials:,}")
print(f"\n*** p-value = {p_value:.5f} ***")

# ----------------------------------------------------------
# 4. UNCERTAINTY ANALYSIS
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("UNCERTAINTY ANALYSIS")
print("=" * 70)

# Test if alignment holds within uncertainties
# Resample CMB and FRB axes within their error bars
n_resample = 10000
p_values_resampled = []

for _ in range(n_resample):
    # Resample CMB longitude within uncertainty
    cmb_l_sample = np.random.normal(cmb_l, cmb_l_err)
    cmb_sample = make_axis(cmb_l_sample, cmb_b)
    
    # Resample FRB within uncertainty
    frb_l_sample = np.random.normal(frb_l, frb_l_err)
    frb_sample = make_axis(frb_l_sample, frb_b)
    
    # Keep clock fixed (it's already uncertain)
    sep1 = cmb_sample.separation(frb_sample).deg
    sep2 = cmb_sample.separation(clk).deg
    sep3 = frb_sample.separation(clk).deg
    
    max_sep_sample = max(sep1, sep2, sep3)
    
    # What fraction of random triples have tighter clustering?
    p_sample = np.mean(max_seps <= max_sep_sample)
    p_values_resampled.append(p_sample)

p_values_resampled = np.array(p_values_resampled)
p_median = np.median(p_values_resampled)
p_16 = np.percentile(p_values_resampled, 16)
p_84 = np.percentile(p_values_resampled, 84)

print("Accounting for coordinate uncertainties:")
print(f"  Median p-value: {p_median:.5f}")
print(f"  16th-84th percentile: {p_16:.5f} to {p_84:.5f}")
print(f"\nIf p_median < 0.01: Strong evidence survives uncertainties")
print(f"If p_median > 0.05: Alignment may be artifact of uncertainties")

# ----------------------------------------------------------
# 5. DISTRIBUTION STATISTICS
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("NULL DISTRIBUTION STATISTICS")
print("=" * 70)

mean_max = np.mean(max_seps)
median_max = np.median(max_seps)
p10 = np.percentile(max_seps, 10)
p01 = np.percentile(max_seps, 1)

print(f"Mean max separation:    {mean_max:6.2f} deg")
print(f"Median max separation:  {median_max:6.2f} deg")
print(f"10th percentile:        {p10:6.2f} deg")
print(f"1st percentile:         {p01:6.2f} deg")

print("\nInterpretation:")
print(f"Your observed value ({observed_max:.2f} deg) is at the")
print(f"{p_value*100:.3f}th percentile of the null distribution")

# ----------------------------------------------------------
# 6. FINAL VERDICT
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

# Use the uncertainty-corrected p-value
p_final = p_median

if p_final < 0.001:
    verdict = "*** EXTREMELY SIGNIFICANT ***"
    interp = "Probability of random 3-way alignment is < 0.1%"
    publish = "PUBLICATION-WORTHY (if coordinates are valid)"
elif p_final < 0.01:
    verdict = "** HIGHLY SIGNIFICANT **"
    interp = "Strong evidence of non-random alignment"
    publish = "PUBLICATION-WORTHY (needs careful validation)"
elif p_final < 0.05:
    verdict = "* SIGNIFICANT *"
    interp = "Moderate evidence for alignment"
    publish = "Potentially interesting (needs more data)"
else:
    verdict = "NOT SIGNIFICANT"
    interp = "Alignment consistent with random chance"
    publish = "Not publication-worthy"

print(f"\n{verdict}")
print(f"p-value (uncertainty-corrected): {p_final:.5f}")
print(f"\n{interp}")
print(f"\nPublication assessment: {publish}")

# Critical warnings
print("\n" + "=" * 70)
print("CRITICAL WARNINGS")
print("=" * 70)
print("[!] FRB coordinates: Verify these came from proper sky dipole analysis")
print("[!] Clock coordinates: Sky position is model-dependent and uncertain")
print("[!] Before publication: Run corrected_unified_axis_test.py to validate")
print("[!] Check for selection effects, detector systematics, foregrounds")
print("[!] Compute trials factor if you tested multiple axis candidates")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)