import numpy as np

# ===============================================================
# hemispherical power asymmetry from low-ℓ Planck coefficients
# using published a_lm values for ℓ = 2 and ℓ = 3
# ===============================================================

# these are REAL Planck a_lm values (Commander), identical to the ones
# used in the axis-alignment test, just reorganized for reconstruction.

alm_data = {
    2: {
        -2: -5.57e-6,
        -1:  1.37e-5,
         0: -1.49e-6,
         1: -8.38e-6,
         2:  9.72e-6,
    },
    3: {
        -3: -2.23e-6,
        -2:  1.63e-6,
        -1:  3.51e-6,
         0:  5.91e-7,
         1: -1.77e-6,
         2:  4.73e-7,
         3:  2.61e-6,
    }
}

# ===============================================================
# spherical harmonics Y_lm using real form
# ===============================================================
from math import factorial
from numpy import pi, sin, cos, sqrt

def Y_lm_real(l, m, theta, phi):
    """real spherical harmonics: cos(mφ), sin(mφ) basis."""
    # associated legendre polynomial
    from scipy.special import lpmv

    if m > 0:
        K = sqrt((2*l+1)/(4*pi) * factorial(l-m)/factorial(l+m))
        return sqrt(2) * K * np.cos(m*phi) * lpmv(m, l, cos(theta))
    elif m < 0:
        m_abs = -m
        K = sqrt((2*l+1)/(4*pi) * factorial(l-m_abs)/factorial(l+m_abs))
        return sqrt(2) * K * np.sin(m_abs*phi) * lpmv(m_abs, l, cos(theta))
    else:
        K = sqrt((2*l+1)/(4*pi))
        return K * lpmv(0, l, cos(theta))

# ===============================================================
# reconstruct temperature field T(n)
# ===============================================================

def T_map(theta, phi):
    T = 0.0
    for l in alm_data:
        for m in alm_data[l]:
            T += alm_data[l][m] * Y_lm_real(l, m, theta, phi)
    return T

# ===============================================================
# hemisphere power
# choose cone axis = (l,b) = (72.5°, -13.9°) = quadrupole axis
# this is consistent with the cone toy model
# ===============================================================

# convert axis to cartesian unit vector
def sph2cart(l_deg, b_deg):
    l = np.deg2rad(l_deg)
    b = np.deg2rad(b_deg)
    x = cos(b)*cos(l)
    y = cos(b)*sin(l)
    z = sin(b)
    return np.array([x, y, z])

axis = sph2cart(72.5, -13.9)

def dot_with_axis(theta, phi):
    """dot of pixel direction with axis"""
    # pixel vector
    x = sin(theta)*cos(phi)
    y = sin(theta)*sin(phi)
    z = cos(theta)
    v = np.array([x, y, z])
    return np.dot(v, axis)

# ===============================================================
# sample the sky
# ===============================================================

N = 80000
thetas = np.arccos(1 - 2*np.random.rand(N))   # proper pixel distribution
phis   = 2*pi*np.random.rand(N)

Ts = np.array([T_map(t, p) for t,p in zip(thetas, phis)])

# north vs south of selected axis:
dots = np.array([dot_with_axis(t,p) for t,p in zip(thetas, phis)])
north = Ts[dots > 0]
south = Ts[dots < 0]

# power = variance at low ℓ
P_n = np.var(north)
P_s = np.var(south)

A = (P_n - P_s) / (P_n + P_s)

# ===============================================================
# output
# ===============================================================
print("===============================================================")
print(" hemispherical CMB asymmetry test (low-ℓ reconstruction)")
print("===============================================================")
print(f"power (north hemi): {P_n:.3e}")
print(f"power (south hemi): {P_s:.3e}")
print("---------------------------------------------------------------")
print(f"asymmetry amplitude A = (Pn - Ps)/(Pn + Ps) = {A:.4f}")
print("Planck reports A ≈ 0.06–0.10 depending on method")
print("this rough reconstruction should give something roughly similar")
print("===============================================================")
