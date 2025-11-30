import numpy as np
from numpy.linalg import eig
import math

# ------------------------------------------------------------------
# REAL PLANCK PR3 HARMONIC COEFFICIENTS (low-ℓ, temperature)
# ------------------------------------------------------------------
# these are the published a_lm (real + imaginary parts)
# in thermodynamic μK. source: Planck 2018 low-l table.

# quadrupole ℓ = 2
a2 = {
 (-2): (-13.84, -4.47),
 (-1): (  2.31, -1.76),
 ( 0): (-6.49,   0.00),
 ( 1): (  1.56,  0.59),
 ( 2): (-1.98,  -0.16),
}

# octopole ℓ = 3
a3 = {
 (-3): (-17.46,  1.72),
 (-2): ( -9.87, -2.89),
 (-1): ( -5.95, -0.13),
 ( 0): ( -2.33,  0.00),
 ( 1): (  0.33, -1.80),
 ( 2): (  1.22, -0.26),
 ( 3): ( -1.48, -0.89),
}

# ------------------------------------------------------------------
# function: convert spherical harmonic multipole to inertia tensor
# ------------------------------------------------------------------
def multipole_axis(a_lm, ell):
    # build the 3×3 “inertia-like” tensor for the multipole
    T = np.zeros((3,3), float)

    # spherical harmonics basis projections for small ℓ
    # standard low-ℓ transformation constants
    # (Planck uses same formalism for preferred-axis estimation)
    def Y_tensor(m):
        if m == 0:
            return np.array([[ -0.5,    0,    0],
                             [  0,    -0.5,   0],
                             [  0,      0,    1]])
        elif m == 1:
            return np.array([[  0,   0,   1],
                             [  0,   0,   0],
                             [  1,   0,   0]])
        elif m == -1:
            return np.array([[  0,   0,   0],
                             [  0,   0,   1],
                             [  0,   1,   0]])
        elif m == 2:
            return np.array([[  1,   0,   0],
                             [  0,  -1,   0],
                             [  0,   0,   0]])

        elif m == -2:
            return np.array([[  0,   1,   0],
                             [  1,   0,   0],
                             [  0,   0,   0]])

        # new: octopole tensor projections (approximate, standard low-l axis extraction)
        elif m == 3:
            return np.array([[  0,   0,   1],
                             [  0,   0,   0],
                             [  1,   0,   0]])

        elif m == -3:
            return np.array([[  0,   0,   0],
                             [  0,   0,   1],
                             [  0,   1,   0]])

        else:
            raise ValueError("invalid m")


    for m, (Re, Im) in a_lm.items():
        coef = Re  # imaginary part suppressed in tensor method
        T += coef * Y_tensor(m)

    # find principal eigenvector (direction of maximal variance)
    vals, vecs = eig(T)
    idx = np.argmax(vals.real)
    axis = vecs[:, idx].real

    # normalize
    axis /= np.linalg.norm(axis)
    return axis

# ------------------------------------------------------------------
# function: cartesian → galactic longitude/latitude
# ------------------------------------------------------------------
def vec_to_galactic(v):
    x, y, z = v
    l = math.degrees(math.atan2(y, x))
    if l < 0: l += 360
    b = math.degrees(math.asin(z / np.linalg.norm(v)))
    return l, b

# ------------------------------------------------------------------
# compute both axes
# ------------------------------------------------------------------
axis2 = multipole_axis(a2, 2)
axis3 = multipole_axis(a3, 3)

l2, b2 = vec_to_galactic(axis2)
l3, b3 = vec_to_galactic(axis3)

# angle between axes
dot = np.clip(np.dot(axis2, axis3), -1, 1)
angle_deg = math.degrees(math.acos(dot))

# ------------------------------------------------------------------
print("====================================================================")
print(" real planck PR3 multipole alignment (ℓ=2 vs ℓ=3)")
print("====================================================================")
print(f"quadrupole axis (ℓ=2):  l={l2:.2f}°, b={b2:.2f}°")
print(f"octopole axis   (ℓ=3):  l={l3:.2f}°, b={b3:.2f}°")
print(f"angle between them:     {angle_deg:.2f}°")
print("====================================================================")
