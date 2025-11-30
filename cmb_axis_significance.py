import math
import numpy as np

# your measured quadrupole–octopole angle from cmb_real_axis_alignment.py
theta_real_deg = 136.06

def prob_angle_le(theta_deg: float) -> float:
    """
    for two random directions on the sphere, the cosine of the angle is
    uniform in [-1, 1]. that means:

        P(theta <= theta0) = (1 - cos(theta0)) / 2
    """
    t = math.radians(theta_deg)
    return (1.0 - math.cos(t)) / 2.0

def monte_carlo_prob(theta_target_deg: float, n_sims: int = 200000) -> float:
    """
    monte carlo sanity check:
    draw random directions on the sphere and measure how often
    the separation angle >= theta_target_deg.
    """
    theta_target = math.radians(theta_target_deg)
    count = 0

    # draw random unit vectors and compute angles
    for _ in range(n_sims):
        # random direction 1
        u1 = random_unit_vector()
        # random direction 2
        u2 = random_unit_vector()

        # dot product and angle
        dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
        theta = math.acos(dot)

        if theta >= theta_target:
            count += 1

    return count / n_sims

def random_unit_vector():
    """
    draw a random point uniformly on the sphere.
    """
    z = 2.0 * np.random.rand() - 1.0      # cos(theta) uniform in [-1, 1]
    phi = 2.0 * math.pi * np.random.rand()
    r_xy = math.sqrt(1.0 - z*z)
    x = r_xy * math.cos(phi)
    y = r_xy * math.sin(phi)
    return np.array([x, y, z])

if __name__ == "__main__":
    print("======================================================")
    print("  cmb multipole axis angle significance (toy test)")
    print("======================================================")
    print(f"observed angle between ℓ=2 and ℓ=3 axes: {theta_real_deg:.2f}°")
    print()

    # 1) analytic probability: separation >= theta_real
    p_le = prob_angle_le(theta_real_deg)
    p_ge = 1.0 - p_le

    print("analytic result (two random directions on the sky):")
    print(f"  P(theta <= {theta_real_deg:.2f}°) = {p_le:.4f}")
    print(f"  P(theta >= {theta_real_deg:.2f}°) = {p_ge:.4f}")
    print()

    # 2) monte carlo sanity check (optional but fun)
    print("running monte carlo check... (this may take a few seconds)")
    mc_p_ge = monte_carlo_prob(theta_real_deg, n_sims=50000)
    print(f"  monte carlo P(theta >= {theta_real_deg:.2f}°) ≈ {mc_p_ge:.4f}")
    print()

    print("interpretation:")
    print("  - if P(theta >= observed) is ~0.1–0.2, this angle is NOT unusual.")
    print("  - only if the angle were very small (< ~20°) or")
    print("    very close to 180° would it look suspicious.")
    print("======================================================")
