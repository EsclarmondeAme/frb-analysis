"""
cone_layer_profile.py
----------------------------------------------------------
constructs a simple cone-layer model from your measured axes.

uses the unified best-fit axis and residual angles for:
  - cmb
  - frb sidereal
  - atomic clock

prints a conceptual 'layer' structure of the cone.
"""

import numpy as np

# unified best-fit axis (from unified_best_fit_axis.py)
best_l = 159.85
best_b = -0.51

# measured residual angles (degrees) from your run
theta_cmb   = 8.53
theta_frb   = 0.80
theta_clock = 5.03

print("=" * 70)
print("cone layer profile based on measured axes")
print("=" * 70)

print("\n1. central axis (galactic)")
print(f"   l = {best_l:.2f}°")
print(f"   b = {best_b:.2f}°")

print("\n2. messenger layers (angle from axis)")
layers = [
    ("frb sidereal", theta_frb,   "inner filament / core layer"),
    ("atomic clocks", theta_clock, "mid shell"),
    ("cmb modulation", theta_cmb, "outer shell"),
]

for name, theta, label in layers:
    print(f"   {name:14s}: {theta:5.2f}°   → {label}")

# simple summary statistics
angles = np.array([theta_frb, theta_clock, theta_cmb])
print("\n3. basic cone metrics")
print(f"   inner radius (core)      ≈ {theta_frb:.1f}°")
print(f"   mid shell radius         ≈ {theta_clock:.1f}°")
print(f"   outer modulation radius  ≈ {theta_cmb:.1f}°")
print(f"   total span (inner→outer) ≈ {angles.max()-angles.min():.1f}°")

print("\n4. interpretation")
print("   treat these as three 'frequency levels' sampling the same cone:")
print("   - frb timing hugs the axis (inner level)")
print("   - atomic clocks sit slightly off-axis (intermediate coupling)")
print("   - cmb is a broad outer modulation (most smeared)")

print("\n" + "=" * 70)
print("done")
print("=" * 70)
