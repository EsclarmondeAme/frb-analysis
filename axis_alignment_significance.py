"""
axis_alignment_significance.py
----------------------------------------------------------
monte carlo significance test for 3-way axis alignment
cmb axis • frb sidereal axis • atomic clock sidereal axis
refined coordinates included
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
# 1. observed axes (refined)
# ----------------------------------------------------------

print("=" * 70)
print("3-way axis alignment significance test (refined)")
print("=" * 70)

cmb = make_axis(152.62, 4.03)
frb = make_axis(160.39, 0.08)
clk = make_axis(163.54, -3.93)

# pairwise separations
sep_cmb_frb = cmb.separation(frb).deg
sep_cmb_clk = cmb.separation(clk).deg
sep_frb_clk = frb.separation(clk).deg

observed_max = max(sep_cmb_frb, sep_cmb_clk, sep_frb_clk)

print("\n1. observed axes (galactic coords)")
print(f"   cmb   : l = {cmb.l.deg:7.2f}°, b = {cmb.b.deg:7.2f}°")
print(f"   frb   : l = {frb.l.deg:7.2f}°, b = {frb.b.deg:7.2f}°")
print(f"   clock : l = {clk.l.deg:7.2f}°, b = {clk.b.deg:7.2f}°")

print("\n   pairwise separations:")
print(f"   cmb ↔ frb     = {sep_cmb_frb:6.2f}°")
print(f"   cmb ↔ clock   = {sep_cmb_clk:6.2f}°")
print(f"   frb ↔ clock   = {sep_frb_clk:6.2f}°")
print(f"\n   maximum separation = {observed_max:.2f}°")


# ----------------------------------------------------------
# 2. monte carlo simulation
# ----------------------------------------------------------

n_trials = 200000

print("\n" + "=" * 70)
print("2. monte carlo simulation")
print("=" * 70)
print(f"   generating {n_trials:,} random triples of axes...")
print("   null: three independent isotropic directions\n")

# generate all random triples as vectors
vecs = random_unit_vectors(3 * n_trials).reshape(n_trials, 3, 3)

# compute max separation for each random triple
a = vecs[:, 0, :]
b = vecs[:, 1, :]
c = vecs[:, 2, :]

sep_ab = vector_angle_deg(a, b)
sep_ac = vector_angle_deg(a, c)
sep_bc = vector_angle_deg(b, c)

max_seps = np.maximum.reduce([sep_ab, sep_ac, sep_bc])

hits = np.sum(max_seps <= observed_max)
p_value = hits / n_trials

print(f"   observed threshold: max_sep ≤ {observed_max:.2f}°")
print(f"   random triples meeting threshold: {hits:,} / {n_trials:,}")
print(f"   estimated p-value = {p_value:.4g}")


# ----------------------------------------------------------
# 3. distribution summary
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("3. random distribution summary")
print("=" * 70)

mean_max = np.mean(max_seps)
median_max = np.median(max_seps)
p10 = np.percentile(max_seps, 10)
p01 = np.percentile(max_seps, 1)

print(f"   mean max separation:    {mean_max:6.2f}°")
print(f"   median max separation:  {median_max:6.2f}°")
print(f"   10th percentile:        {p10:6.2f}°")
print(f"   1st percentile:         {p01:6.2f}°")

print("\n   note:")
print("   random triples rarely cluster below ~20–25° total max separation")
print("   your observed value (~13.5°) is unusually low")

# ----------------------------------------------------------
# 4. final verdict
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("4. final verdict")
print("=" * 70)

if p_value < 1e-3:
    print("\n★ extremely significant 3-way alignment")
    print("  probability of random occurrence is very small")
elif p_value < 0.01:
    print("\n★ strong evidence of non-random alignment")
elif p_value < 0.05:
    print("\n~ moderate evidence for alignment")
else:
    print("\n✗ alignment consistent with random chance")

print(f"\n  p-value = {p_value:.4g}")
print("  smaller p means tighter-than-random axis clustering\n")

print("=" * 70)
print("done")
print("=" * 70)
