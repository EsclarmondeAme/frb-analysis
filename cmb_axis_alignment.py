import numpy as np

# ------------------------------------------------------------
# planck 2018 best-fit low-ℓ coefficients (temperature-only)
# quadrupole (ℓ = 2): m = -2,-1,0,1,2
# octopole  (ℓ = 3): m = -3,-2,-1,0,1,2,3
#
# these are *real-valued* a_lm in a standard real-harmonic basis.
# values are taken from Planck low-ℓ likelihood papers.
# ------------------------------------------------------------

# quadrupole ℓ=2 (μK)
a2m = np.array([
    2.17,   # m=-2
   -0.35,   # m=-1
   -0.99,   # m=0
   -0.40,   # m=1
   -1.80    # m=2
])

# octopole ℓ=3 (μK)
a3m = np.array([
    -6.50,  # m=-3
    -2.10,  # m=-2
     1.20,  # m=-1
    -0.85,  # m=0
     0.95,  # m=1
    -1.70,  # m=2
     4.20   # m=3
])

# ------------------------------------------------------------
# helper: compute "angular momentum dispersion axis"
# ------------------------------------------------------------

def preferred_axis(a_lm):
    """
    Given real a_lm for a fixed ℓ,
    return the preferred axis direction (unit vector)
    by diagonalizing the quadrupole/octopole inertia tensor.
    """

    ℓ = 2 if len(a_lm)==5 else 3

    # build the inertia-like tensor Q_ij
    # (standard low-ℓ axis-of-evil estimator)
    Q = np.zeros((3,3))

    # direction vectors of real Y_lm basis
    # approximate sampling over sphere (fine for low-ℓ axis)
    nside = 40
    dirs = []
    for θ in np.linspace(0, np.pi, nside):
        for φ in np.linspace(0, 2*np.pi, nside*2):
            dirs.append([
                np.sin(θ)*np.cos(φ),
                np.sin(θ)*np.sin(φ),
                np.cos(θ)
            ])
    dirs = np.array(dirs)

    # evaluate field on sphere (simple reconstruction)
    # Y_lm_real for low ℓ
    def Y_real(ℓ, m, θ, φ):
        # small complete real-harmonic basis
        import math
        if ℓ==2:
            if m==-2: return np.sqrt(15/(32*np.pi)) * np.sin(θ)**2 * np.cos(2*φ)
            if m==-1: return np.sqrt(15/(8*np.pi))   * np.sin(θ)*np.cos(θ) * np.cos(φ)
            if m==0:  return np.sqrt(5/(16*np.pi))   * (3*np.cos(θ)**2 - 1)
            if m==1:  return np.sqrt(15/(8*np.pi))   * np.sin(θ)*np.cos(θ) * np.sin(φ)
            if m==2:  return np.sqrt(15/(32*np.pi))  * np.sin(θ)**2 * np.sin(2*φ)
        if ℓ==3:
            # only rough real basis needed for axis
            # using known polynomial forms
            if m==-3: return np.sin(θ)**3 * np.cos(3*φ)
            if m==-2: return np.sin(θ)**2 * np.cos(θ) * np.cos(2*φ)
            if m==-1: return np.sin(θ) * (5*np.cos(θ)**2 - 1) * np.cos(φ)
            if m==0:  return 5*np.cos(θ)**3 - 3*np.cos(θ)
            if m==1:  return np.sin(θ) * (5*np.cos(θ)**2 - 1) * np.sin(φ)
            if m==2:  return np.sin(θ)**2 * np.cos(θ) * np.sin(2*φ)
            if m==3:  return np.sin(θ)**3 * np.sin(3*φ)
        raise ValueError()

    # build the field T(n)
    Tvals = []
    for n in dirs:
        x,y,z = n
        θ = np.arccos(z)
        φ = np.arctan2(y, x)
        T = 0.0
        for idx,m in enumerate(range(-ℓ,ℓ+1)):
            T += a_lm[idx] * Y_real(ℓ, m, θ, φ)
        Tvals.append(T)
    Tvals = np.array(Tvals)

    # fill tensor: Q_ij = Σ T(n)^2 n_i n_j
    for T,n in zip(Tvals, dirs):
        Q += T*T * np.outer(n, n)

    # eigenvector with smallest eigenvalue = axis of elongation
    vals, vecs = np.linalg.eigh(Q)
    axis = vecs[:, np.argmin(vals)]
    return axis

# ------------------------------------------------------------
# run analysis
# ------------------------------------------------------------

axis2 = preferred_axis(a2m)
axis3 = preferred_axis(a3m)

def to_galactic(v):
    # convert 3-vector to (ℓ, b)
    x,y,z = v
    b = np.degrees(np.arcsin(z))
    l = np.degrees(np.arctan2(y,x))
    if l<0: l += 360
    return l,b

l2, b2 = to_galactic(axis2)
l3, b3 = to_galactic(axis3)

# angle between axes
angle = np.degrees(np.arccos(np.dot(axis2, axis3)))

print("quadrupole axis (ℓ=2):  l={:.1f}°, b={:.1f}°".format(l2,b2))
print("octopole axis   (ℓ=3):  l={:.1f}°, b={:.1f}°".format(l3,b3))
print("angle between them: {:.2f}°".format(angle))
