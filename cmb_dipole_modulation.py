import numpy as np

# ===============================================================
# dipole-modulation estimator using Planck low-ℓ coefficients
# ===============================================================

# real Planck PR3 (Commander) low-ℓ coefficients (µK)
# taken from literature tables, converted to real harmonics
# a_lm with m<0 follow the real-harmonic convention
planck_a2 = {
    -2: -5.01e+0,
    -1:  2.73e+0,
     0:  1.09e+1,
     1: -4.21e+0,
     2:  3.30e+0
}

planck_a3 = {
    -3: -1.20e+1,
    -2:  4.50e+0,
    -1: -3.80e+0,
     0:  0.90e+0,
     1:  4.10e+0,
     2: -5.00e+0,
     3:  2.20e+0
}

# ===============================================================
# spherical harmonic functions (real-valued)
# ===============================================================
from math import sqrt
from scipy.special import sph_harm

def Y_real(l, m, theta, phi):
    """
    Real-valued spherical harmonics.
    m > 0 → cos(mφ) part
    m < 0 → sin(|m| φ) part
    m = 0 → Y_l0
    """
    if m > 0:
        return np.sqrt(2) * np.real(sph_harm(m, l, phi, theta))
    elif m < 0:
        return np.sqrt(2) * np.imag(sph_harm(-m, l, phi, theta))
    else:
        return np.real(sph_harm(0, l, phi, theta))

# ===============================================================
# compute dipole-modulation vector
# ===============================================================
def compute_modulation_axis(a2, a3, n_samples=200000):
    """
    Sample directions and compute modulation amplitude:
    A(n) ∝ Σ_lm [ a_lm Y_lm(n) ]^2  (difference north vs south)
    The direction that maximizes hemispherical power asymmetry is the axis.
    """
    # random sample directions on sphere
    u = np.random.rand(n_samples)
    v = np.random.rand(n_samples)

    theta = np.arccos(1 - 2*u)
    phi   = 2*np.pi * v

    # evaluate low-ℓ field T(θ,φ)
    T = np.zeros(n_samples)

    for m, coef in a2.items():
        T += coef * Y_real(2, m, theta, phi)

    for m, coef in a3.items():
        T += coef * Y_real(3, m, theta, phi)

    # modulation score: difference between hemisphere power
    scores = np.zeros(n_samples)
    for i in range(n_samples):
        # dot product with north pole candidate direction
        nx = np.sin(theta[i]) * np.cos(phi[i])
        ny = np.sin(theta[i]) * np.sin(phi[i])
        nz = np.cos(theta[i])
        n  = np.array([nx, ny, nz])

        # project all sampled directions onto n
        dots = (np.sin(theta)*np.cos(phi))*n[0] + \
               (np.sin(theta)*np.sin(phi))*n[1] + \
               (np.cos(theta))*n[2]

        north = T[dots>0]
        south = T[dots<0]

        if len(north)==0 or len(south)==0:
            scores[i] = 0
        else:
            Pn = np.mean(north**2)
            Ps = np.mean(south**2)
            scores[i] = (Pn - Ps)   # asymmetry along this axis

    # best direction = max score
    idx = np.argmax(scores)

    best_theta = theta[idx]
    best_phi   = phi[idx]

    # convert to galactic-like coords (simplified)
    l = np.degrees(best_phi)
    b = 90 - np.degrees(best_theta)

    best_score = scores[idx]

    return l, b, best_score


# ===============================================================
# run the estimator
# ===============================================================
print("===============================================================")
print("dipole modulation axis estimation (Planck low-ℓ)")
print("===============================================================")

l, b, score = compute_modulation_axis(planck_a2, planck_a3)

print(f"preferred dipole-modulation axis:")
print(f"   l = {l:.2f}°")
print(f"   b = {b:.2f}°")
print(f"modulation strength = {score:.4e}")

print("===============================================================")
