"""
unified_best_fit_axis.py
----------------------------------------------------------
computes the best-fit preferred axis from:
• cmb dipole-modulation axis
• frb sidereal-dipole axis (refined)
• atomic clock sidereal axis (refined)

outputs:
• best-fit galactic (l, b)
• residual angles for each input axis
• rms uncertainty estimate
• final formatted verdict
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


# ----------------------------------------------------------
# 0. helpers
# ----------------------------------------------------------

def axis_to_vector(l_deg, b_deg):
    """
    convert galactic (l, b) to a 3D unit vector.
    """
    coord = SkyCoord(l=l_deg*u.deg, b=b_deg*u.deg, frame='galactic')
    x = coord.cartesian.x.value
    y = coord.cartesian.y.value
    z = coord.cartesian.z.value
    return np.array([x, y, z])


def vector_to_lb(vec):
    """
    convert a 3D unit vector back to galactic (l, b).
    """
    from astropy.coordinates import SkyCoord, CartesianRepresentation
    x, y, z = vec
    rep = CartesianRepresentation(x, y, z)
    coord = SkyCoord(rep, frame='galactic')
    return coord.l.deg, coord.b.deg


def angular_sep(v1, v2):
    """
    angle between two unit vectors in degrees.
    """
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


# ----------------------------------------------------------
# 1. inputs (refined)
# ----------------------------------------------------------

cmb_l,   cmb_b   = 152.62,  4.03
frb_l,   frb_b   = 160.39,  0.08
clk_l,   clk_b   = 163.54, -3.93

cmb_vec = axis_to_vector(cmb_l, cmb_b)
frb_vec = axis_to_vector(frb_l, frb_b)
clk_vec = axis_to_vector(clk_l, clk_b)

print("=" * 70)
print("unified best-fit axis estimate (refined)")
print("=" * 70)

print("\n1. input axes (galactic coordinates)")
print(f"   cmb   : l = {cmb_l:7.2f}°, b = {cmb_b:7.2f}°")
print(f"   frb   : l = {frb_l:7.2f}°, b = {frb_b:7.2f}°")
print(f"   clock : l = {clk_l:7.2f}°, b = {clk_b:7.2f}°")


# ----------------------------------------------------------
# 2. assign weights
# ----------------------------------------------------------
# weights chosen based on residual tightness:
# frb–clock ~5°, cmb–frb ~9°, cmb–clock ~13°

w_frb  = 3.0   # strongest, extremely tight sidereal signal
w_clk  = 2.5   # nearly as tight
w_cmb  = 1.5   # slightly farther out

weights = np.array([w_cmb, w_frb, w_clk])
vecs    = np.array([cmb_vec, frb_vec, clk_vec])


# ----------------------------------------------------------
# 3. compute weighted best-fit axis
# ----------------------------------------------------------

weighted_sum = np.sum(weights[:, None] * vecs, axis=0)
best_axis = weighted_sum / np.linalg.norm(weighted_sum)

best_l, best_b = vector_to_lb(best_axis)


# ----------------------------------------------------------
# 4. compute residuals
# ----------------------------------------------------------

res_cmb = angular_sep(best_axis, cmb_vec)
res_frb = angular_sep(best_axis, frb_vec)
res_clk = angular_sep(best_axis, clk_vec)

residuals = np.array([res_cmb, res_frb, res_clk])
rms_uncertainty = np.sqrt(np.mean(residuals**2))


# ----------------------------------------------------------
# 5. print results
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("2. best-fit preferred axis (galactic)")
print("=" * 70)
print(f"   l = {best_l:7.2f}°")
print(f"   b = {best_b:7.2f}°")

print("\n   residual angles:")
print(f"   cmb   → best = {res_cmb:6.2f}°")
print(f"   frb   → best = {res_frb:6.2f}°")
print(f"   clock → best = {res_clk:6.2f}°")

print(f"\n   rms uncertainty estimate = {rms_uncertainty:.2f}°")


# ----------------------------------------------------------
# 6. interpretation
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("3. interpretation")
print("=" * 70)

print("   the best-fit axis lies close to all three signals:")
print(f"   • cmb deviation:   {res_cmb:.2f}°")
print(f"   • frb deviation:   {res_frb:.2f}°")
print(f"   • clock deviation: {res_clk:.2f}°")

if rms_uncertainty < 5:
    print("\n   ★ extremely tight consensus (<5° scatter)")
elif rms_uncertainty < 10:
    print("\n   ★ strong consensus (<10° scatter)")
else:
    print("\n   ~ weak consensus")

print("\n" + "=" * 70)
print("done")
print("=" * 70)
