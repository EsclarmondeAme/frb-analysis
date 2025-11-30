import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize
from scipy.special import sph_harm_y  # modern replacement

"""
frb_2d_anisotropy_field.py
------------------------------------------------------------
Reconstructs a 2D anisotropy field f(theta, phi) around the
unified axis directly from FRB data.

Steps:
1. load FRBs
2. rotate to aligned (theta, phi) frame
3. build 2D grid in (theta, phi)
4. compute weighted FRB density per cell
5. smooth with Gaussian kernel
6. expand field in spherical harmonics Y_lm
7. reconstruct anisotropy map
8. save figures
"""

# unified axis from earlier work
UNIF_L = 159.85
UNIF_B = -0.51

# grid resolution
theta_bins = 72  # 2.5 degree bins
phi_bins = 72    # 5 degree bins

def rotate_to_axis(l_deg, b_deg, axis_l, axis_b):
    c = SkyCoord(l=l_deg*u.deg, b=b_deg*u.deg, frame='galactic')
    a = SkyCoord(l=axis_l*u.deg, b=axis_b*u.deg, frame='galactic')
    sep = c.separation(a).deg
    relpa = c.position_angle(a).deg
    phi = (-relpa) % 360
    return sep, phi

def gaussian_smooth_2d(field, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(field, sigma=sigma)

def spherical_harmonic_decomp(theta, phi, f, lmax=6):
    Ylm = []
    alm = []
    theta_r = np.radians(theta)
    phi_r = np.radians(phi)
    for ell in range(lmax+1):
        for m in range(-ell, ell+1):
            Y = sph_harm_y(m, ell, phi_r, theta_r)
            a = np.sum(f * np.conjugate(Y))
            alm.append(a)
            Ylm.append((ell, m, a))
    return Ylm

def main():
    df = pd.read_csv("frbs.csv")
    coords = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg, frame="icrs").galactic
    l = coords.l.deg
    b = coords.b.deg
    theta, phi = rotate_to_axis(l, b, UNIF_L, UNIF_B)
    T = np.minimum(theta, 140.0)
    H = np.histogram2d(T, phi, bins=[theta_bins, phi_bins],
                       range=[[0,140],[0,360]])[0]
    Hs = gaussian_smooth_2d(H, sigma=1.5)
    tb = 0.5*(np.linspace(0,140,theta_bins+1)[1:] + np.linspace(0,140,theta_bins+1)[:-1])
    pb = 0.5*(np.linspace(0,360,phi_bins+1)[1:] + np.linspace(0,360,phi_bins+1)[:-1])
    TT, PP = np.meshgrid(tb, pb, indexing='ij')
    field = Hs.flatten()
    theta_flat = TT.flatten()
    phi_flat = PP.flatten()
    Ylm = spherical_harmonic_decomp(theta_flat, phi_flat, field, lmax=6)
    plt.figure(figsize=(10,6))
    plt.imshow(Hs, origin='lower', extent=[0,360,0,140], aspect='auto', cmap='inferno')
    plt.colorbar(label='density (smoothed)')
    plt.xlabel('phi (deg)')
    plt.ylabel('theta (deg)')
    plt.title('FRB 2D Anisotropy Field (smoothed density)')
    plt.savefig('frb_2d_anisotropy_field.png', dpi=200, bbox_inches='tight')
    with open('frb_2d_alm.txt','w') as f:
        for ell, m, a in Ylm:
            f.write(f"ell={ell} m={m} Re={a.real:.6e} Im={a.imag:.6e}\n")
    print("saved: frb_2d_anisotropy_field.png")
    print("saved: frb_2d_alm.txt")
    print("analysis complete.")

if __name__ == '__main__':
    main()