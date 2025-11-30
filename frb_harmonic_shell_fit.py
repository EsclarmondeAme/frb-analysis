import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import least_squares
from scipy.special import sph_harm

# unified axis (galactic)
UNIFIED_L = 159.85
UNIFIED_B = -0.51

# shell boundaries
THETA_MIN = 20.0
THETA_MAX = 60.0


def compute_axis_angles(ra_deg, dec_deg):
    ra_arr = np.asarray(ra_deg, dtype=float)
    dec_arr = np.asarray(dec_deg, dtype=float)

    coords = SkyCoord(ra=ra_arr * u.deg, dec=dec_arr * u.deg, frame="icrs")

    gal = coords.galactic

    l = gal.l.deg
    b = gal.b.deg

    # convert unified axis to Cartesian
    ux = np.cos(np.radians(UNIFIED_b := UNIFIED_B)) * np.cos(np.radians(UNIFIED_L))
    uy = np.cos(np.radians(UNIFIED_B)) * np.sin(np.radians(UNIFIED_L))
    uz = np.sin(np.radians(UNIFIED_B))
    u_axis = np.array([ux, uy, uz])

    # event Cartesian
    ex = np.cos(np.radians(b)) * np.cos(np.radians(l))
    ey = np.cos(np.radians(b)) * np.sin(np.radians(l))
    ez = np.sin(np.radians(b))
    e = np.vstack([ex, ey, ez]).T

    # angle to axis (theta)
    dot = np.sum(e * u_axis, axis=1)
    dot = np.clip(dot, -1, 1)
    theta = np.degrees(np.arccos(dot))

    # define phi in plane perpendicular to axis:
    # build orthonormal basis {e1,e2,u_axis}
    # pick e1 not parallel to axis
    z = np.array([0, 0, 1])
    if abs(np.dot(z, u_axis)) > 0.9:
        z = np.array([0, 1, 0])

    e1 = z - u_axis * np.dot(z, u_axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u_axis, e1)

    proj = e     # position vector in UV-plane before projection
    x = np.sum(proj * e1, axis=1)
    y = np.sum(proj * e2, axis=1)

    phi = (np.degrees(np.arctan2(y, x)) + 360) % 360

    return theta, phi


def real_sph_harm(l, m, theta, phi):
    """
    real spherical harmonics Y_lm^real(θ,φ)
    """
    phi_rad = np.radians(phi)
    theta_rad = np.radians(theta)

    if m > 0:
        return np.sqrt(2) * np.real(sph_harm(m, l, phi_rad, theta_rad))
    elif m < 0:
        return np.sqrt(2) * np.imag(sph_harm(-m, l, phi_rad, theta_rad))
    else:
        return np.real(sph_harm(0, l, phi_rad, theta_rad))


def build_design_matrix(theta, phi, lmax=4):
    X = [np.ones_like(theta)]  # constant term R0

    for l in range(1, lmax+1):
        for m in range(-l, l+1):
            X.append(real_sph_harm(l, m, theta, phi))

    return np.vstack(X).T


def main():
    df = pd.read_csv("frbs.csv").dropna(subset=["ra", "dec"])

    theta, phi = compute_axis_angles(df["ra"], df["dec"])
    df["theta"] = theta
    df["phi"] = phi

    shell = df[(df.theta >= THETA_MIN) & (df.theta <= THETA_MAX)].copy()
    print(f"using {len(shell)} frbs in shell {THETA_MIN}-{THETA_MAX} deg")

    # our "radius" quantity is just theta
    y = shell["theta"].values
    t = shell["theta"].values
    p = shell["phi"].values

    X = build_design_matrix(t, p, lmax=4)
    coeff, *_ = np.linalg.lstsq(X, y, rcond=None)

    # save coefficients
    with open("frb_harmonic_shell_coeffs.txt", "w") as f:
        f.write("Harmonic-shell coefficients (l<=4):\n")
        for i, c in enumerate(coeff):
            f.write(f"{i:3d}: {c:.6f}\n")

    # reconstruct model on map
    phi_grid = np.linspace(0, 360, 200)
    theta_grid = np.linspace(0, 90, 200)
    PH, TH = np.meshgrid(phi_grid, theta_grid)

    Xmap = build_design_matrix(TH.ravel(), PH.ravel(), lmax=4)
    Rmap = Xmap @ coeff
    Rmap = Rmap.reshape(TH.shape)

    # plot model map
    plt.figure(figsize=(10,6))
    plt.pcolormesh(PH, TH, Rmap, cmap="coolwarm")
    plt.colorbar(label="model radius R(θ,φ)")
    plt.xlabel("phi (deg)")
    plt.ylabel("theta (deg)")
    plt.title("FRB 2d harmonic-shell reconstruction (l≤4)")
    plt.savefig("frb_harmonic_shell_fit.png", dpi=150)

    # compute residuals
    y_model = (X @ coeff)
    residuals = y - y_model

    plt.figure(figsize=(10,6))
    sc = plt.scatter(p, t, c=residuals, cmap="coolwarm", s=20)
    plt.colorbar(sc, label="θ_obs - θ_model")
    plt.xlabel("phi (deg)")
    plt.ylabel("theta (deg)")
    plt.title("FRB shell residuals after 2d harmonic reconstruction")
    plt.savefig("frb_harmonic_shell_residual.png", dpi=150)

    print("analysis complete.")
    print("saved: frb_harmonic_shell_fit.png, frb_harmonic_shell_residual.png, frb_harmonic_shell_coeffs.txt")


if __name__ == "__main__":
    main()
