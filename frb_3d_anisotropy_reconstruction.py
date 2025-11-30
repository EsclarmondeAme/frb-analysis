import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from astropy.coordinates import SkyCoord
from astropy import units as u

# ============================================================
# 3D ANISOTROPY RECONSTRUCTION (spherical harmonics)
# ============================================================

UNIFIED_L = 159.85
UNIFIED_B = -0.51

MAX_ELL = 10            # harmonic depth
N_TH = 180              # theta samples
N_PH = 360              # phi samples


def rotate_to_axis(l, b, ax_l, ax_b):
    """
    rotate galactic (l,b) to a frame where the unified axis becomes theta=0.
    uses explicit rotation matrix (no skyoffset_frame).
    """
    # convert to radians
    l = np.radians(l)
    b = np.radians(b)
    ax_l = np.radians(ax_l)
    ax_b = np.radians(ax_b)

    # convert points to 3d cartesian
    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)

    # axis (target north pole)
    ax_x = np.cos(ax_b)*np.cos(ax_l)
    ax_y = np.cos(ax_b)*np.sin(ax_l)
    ax_z = np.sin(ax_b)

    # unit vectors for new frame:
    # new z-axis = axis
    z_new = np.array([ax_x, ax_y, ax_z])

    # pick any vector not parallel to axis to build x_new
    tmp = np.array([0, 0, 1])
    if np.abs(np.dot(z_new, tmp)) > 0.99:
        tmp = np.array([1, 0, 0])

    x_new = np.cross(tmp, z_new)
    x_new /= np.linalg.norm(x_new)

    # y_new = z_new × x_new
    y_new = np.cross(z_new, x_new)

    # build rotation matrix (old→new)
    R = np.vstack([x_new, y_new, z_new])

    # apply rotation
    xyz = np.vstack([x, y, z])
    xyz_new = R @ xyz

    Xn, Yn, Zn = xyz_new

    # convert back to spherical
    theta = np.arccos(Zn)          # 0 at unified axis
    phi = np.arctan2(Yn, Xn)
    phi = np.mod(phi, 2*np.pi)

    return theta, phi


def estimate_alm(theta, phi, max_ell):
    """
    standard spherical harmonic projection:
      a_lm = sum Y_lm*(theta_i,phi_i)
    """
    alm = {}

    for ell in range(max_ell+1):
        for m in range(-ell, ell+1):
            Y = sph_harm(m, ell, phi, theta)
            alm[(ell,m)] = np.sum(Y.conjugate())

    return alm


def reconstruct_map(alm, max_ell, n_th, n_ph):
    """
    rebuild field F(theta,phi) = sum a_lm Y_lm(theta,phi)
    """
    th = np.linspace(0, np.pi, n_th)
    ph = np.linspace(0, 2*np.pi, n_ph)

    TH, PH = np.meshgrid(th, ph, indexing='ij')
    field = np.zeros_like(TH, dtype=float)

    for ell in range(max_ell+1):
        for m in range(-ell, ell+1):
            Y = sph_harm(m, ell, PH, TH)
            field += np.real(alm[(ell,m)] * Y)

    return TH, PH, field


def main():

    # ============================================================
    # load data
    # ============================================================
    df = pd.read_csv("frbs.csv")

    # convert to galactic
    sky = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg, frame='icrs')
    gal = sky.galactic
    l = gal.l.deg
    b = gal.b.deg

    # ============================================================
    # rotate to unified-axis frame
    # ============================================================
    theta, phi = rotate_to_axis(l, b, UNIFIED_L, UNIFIED_B)

    print("==========================================================")
    print("FRB 3D ANISOTROPY RECONSTRUCTION")
    print("==========================================================")
    print(f"Total FRBs: {len(df)}")
    print(f"Unified axis: l={UNIFIED_L}°, b={UNIFIED_B}°")
    print(f"Theta range: {np.min(theta)*180/np.pi:.2f}° – {np.max(theta)*180/np.pi:.2f}°")
    print("Estimating a_lm coefficients...")

    # ============================================================
    # compute alm
    # ============================================================
    alm = estimate_alm(theta, phi, MAX_ELL)

    # power spectrum
    C_ell = []
    for ell in range(MAX_ELL+1):
        power = 0
        for m in range(-ell, ell+1):
            power += np.abs(alm[(ell,m)])**2
        C_ell.append(power)

    print("Multipole powers:")
    for ell, c in enumerate(C_ell):
        print(f"  ell={ell:2d}  C_ell={c:.4e}")

    # ============================================================
    # map reconstruction
    # ============================================================
    print("Reconstructing 3D anisotropy field...")
    TH, PH, F = reconstruct_map(alm, MAX_ELL, N_TH, N_PH)

    # ============================================================
    # plots
    # ============================================================

    # power spectrum
    plt.figure(figsize=(7,5))
    plt.plot(range(MAX_ELL+1), C_ell, 'o-')
    plt.yscale('log')
    plt.xlabel("ell")
    plt.ylabel("C_ell (log)")
    plt.title("FRB 3D Anisotropy — Multipole Power")
    plt.grid(True, ls='--', alpha=0.4)
    plt.savefig("frb_3d_multipole_power.png", dpi=160)

    # map (theta vs phi)
    plt.figure(figsize=(11,6))
    plt.imshow(F, extent=[0,360,180,0], cmap='inferno', aspect='auto')
    plt.colorbar(label="reconstructed intensity")
    plt.xlabel("phi (deg)")
    plt.ylabel("theta (deg)")
    plt.title("FRB 3D Anisotropy Field (reconstructed)")
    plt.savefig("frb_3d_anisotropy_map.png", dpi=160)

    # ============================================================
    # scientific verdict
    # ============================================================
    print("==========================================================")
    print("SCIENTIFIC VERDICT (3D reconstruction)")
    print("==========================================================")

    dip = C_ell[1]
    quad = C_ell[2]
    m3 = C_ell[3]

    print(f"dipole (ell=1):      {dip:.3e}")
    print(f"quadrupole (ell=2):   {quad:.3e}")
    print(f"octupole (ell=3):     {m3:.3e}")

    print("\nInterpretation:")
    if dip > quad and dip > m3:
        print(" → strong dipolar anisotropy relative to axis.")
    if quad > 0.3*dip:
        print(" → significant quadrupole: extended ridges, not a simple cone.")
    if m3 > 0.2*dip:
        print(" → non-axisymmetric structure: faceting or patch-like regions present.")

    print("Maps saved:")
    print(" - frb_3d_multipole_power.png")
    print(" - frb_3d_anisotropy_map.png")
    print("==========================================================")


if __name__ == "__main__":
    main()
