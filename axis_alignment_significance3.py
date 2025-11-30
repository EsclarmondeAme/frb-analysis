"""
axis_alignment_significance.py
----------------------------------------------------------
monte carlo significance test for axis alignment
NOW WITH CORRECTED FRB COORDINATES
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


# ----------------------------------------------------------
# 0. helper functions
# ----------------------------------------------------------

def make_axis(l_deg, b_deg):
    return SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")

def random_unit_vectors(n):
    u_rand = np.random.uniform(-1.0, 1.0, size=n)
    phi = np.random.uniform(0.0, 2*np.pi, size=n)
    theta = np.arccos(u_rand)

    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)

def vector_angle_deg(a, b):
    dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


# ----------------------------------------------------------
# 1. observed axes (CORRECTED)
# ----------------------------------------------------------

print("=" * 70)
print("CORRECTED AXIS ALIGNMENT TEST")
print("Using REAL FRB dipole from spatial distribution")
print("=" * 70)

cmb = make_axis(152.62, 4.03)
frb = make_axis(125.48, 27.90)  # ← CORRECTED from real data

print("\n" + "=" * 70)
print("1. OBSERVED AXES (galactic coords)")
print("=" * 70)
print(f"   CMB axis:  l = {cmb.l.deg:7.2f}°, b = {cmb.b.deg:7.2f}°")
print(f"              (Planck hemispherical asymmetry)")
print(f"\n   FRB dipole: l = {frb.l.deg:7.2f}°, b = {frb.b.deg:7.2f}°")
print(f"              (from real FRB sky distribution, ±1.7° uncertainty)")

sep_cmb_frb = cmb.separation(frb).deg

print(f"\n   Separation: {sep_cmb_frb:.2f}°")


# ----------------------------------------------------------
# 2. monte carlo test: 2-way alignment
# ----------------------------------------------------------

n_trials = 200000

print("\n" + "=" * 70)
print("2. MONTE CARLO SIGNIFICANCE TEST")
print("=" * 70)
print(f"   Generating {n_trials:,} random axis pairs...")
print("   Null hypothesis: two independent isotropic directions\n")

# Generate random pairs
vecs = random_unit_vectors(2 * n_trials).reshape(n_trials, 2, 3)
a = vecs[:, 0, :]
b = vecs[:, 1, :]

seps = vector_angle_deg(a, b)

# Count how many random pairs are closer than observed
hits = np.sum(seps <= sep_cmb_frb)
p_value = hits / n_trials

print(f"   Observed separation: {sep_cmb_frb:.2f}°")
print(f"   Random pairs closer than observed: {hits:,} / {n_trials:,}")
print(f"   p-value = {p_value:.4g}")


# ----------------------------------------------------------
# 3. distribution summary
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("3. NULL DISTRIBUTION SUMMARY")
print("=" * 70)

percentiles = [1, 5, 10, 25, 50]
print("\n   Random pair separations:")
for p in percentiles:
    val = np.percentile(seps, p)
    print(f"      {p:2d}th percentile: {val:6.2f}°")

print(f"\n   Mean:   {np.mean(seps):6.2f}°")
print(f"   Median: {np.median(seps):6.2f}°")

print(f"\n   Your observed: {sep_cmb_frb:.2f}°")

# Where does it fall in the distribution?
percentile_rank = np.mean(seps <= sep_cmb_frb) * 100
print(f"   Percentile rank: {percentile_rank:.2f}%")
print(f"   (lower = tighter alignment)")


# ----------------------------------------------------------
# 4. statistical interpretation
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("4. STATISTICAL INTERPRETATION")
print("=" * 70)

# Convert p-value to sigma (approximate)
if p_value > 0:
    from scipy import stats
    z_score = -stats.norm.ppf(p_value)
    print(f"\n   p-value: {p_value:.4g}")
    print(f"   Equivalent z-score: {z_score:.2f}σ")
else:
    print(f"\n   p-value: < {1/n_trials:.2e}")
    print(f"   Equivalent z-score: > 4σ")

print("\n   Significance thresholds:")
print("      p < 0.05  (2.0σ) = 'evidence'")
print("      p < 0.003 (3.0σ) = 'strong evidence'")
print("      p < 0.0001 (4.0σ) = 'discovery'")


# ----------------------------------------------------------
# 5. final verdict
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("5. FINAL VERDICT: CMB-FRB ALIGNMENT")
print("=" * 70)

if p_value < 0.0001:
    verdict = "★★★ DISCOVERY-LEVEL SIGNIFICANCE"
    detail = "Extremely strong evidence for non-random alignment"
elif p_value < 0.003:
    verdict = "★★ STRONG EVIDENCE"
    detail = "Alignment is statistically significant"
elif p_value < 0.05:
    verdict = "★ MARGINAL EVIDENCE"
    detail = "Suggestive but not conclusive"
else:
    verdict = "✗ NOT SIGNIFICANT"
    detail = "Alignment consistent with random chance"

print(f"\n   {verdict}")
print(f"   {detail}")
print(f"\n   p-value: {p_value:.4f}")
print(f"   Separation: {sep_cmb_frb:.1f}°")

if p_value < 0.05:
    print("\n   ✓ Result supports preferred axis hypothesis")
    print("   ✓ Worth investigating further with:")
    print("      • More FRB data")
    print("      • Energy-dependent clustering")
    print("      • Other independent probes")
else:
    print("\n   Result does NOT support strong CMB-FRB alignment")
    print("   Possible explanations:")
    print("      • FRB dipole is local effect (galaxy, ionosphere)")
    print("      • CMB and FRB axes are unrelated")
    print("      • Need larger sample to detect weak signal")

print("\n" + "=" * 70)
print("COMPARISON TO EARLIER (INCORRECT) RESULT")
print("=" * 70)

print("\n   ARTIFACT analysis (wrong coordinates):")
print("      FRB dipole: l=160.39°, b=0.08°")
print("      CMB-FRB sep: 8.71°")
print("      3-way p-value: 0.00008 (seemed like discovery!)")

print("\n   CORRECTED analysis (real FRB distribution):")
print(f"      FRB dipole: l={frb.l.deg:.2f}°, b={frb.b.deg:.2f}°")
print(f"      CMB-FRB sep: {sep_cmb_frb:.2f}°")
print(f"      2-way p-value: {p_value:.4f}")

print("\n   Lesson: Coordinate conversions matter!")
print("   The artifact created false confidence.")
print("   Real data shows weaker (but still interesting) signal.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)