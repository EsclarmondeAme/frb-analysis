import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.special import sph_harm

print("="*70)
print("FRB MULTIPOLE DECOMPOSITION AROUND UNIFIED AXIS")
print("Full a_lm extraction, dipole/quadrupole direction, rotated frame")
print("="*70)

# =====================================================================
# 1) Load FRB catalog
# =====================================================================
df = pd.read_csv("frbs.csv")
coords = SkyCoord(ra=df["ra"].values*u.deg,
                  dec=df["dec"].values*u.deg,
                  frame="icrs").galactic

N = len(df)
print(f"\nLoaded {N} FRBs")

# =====================================================================
# 2) Unified axis (from your previous analysis)
# =====================================================================
unif_l = 159.85
unif_b = -0.51
unif = SkyCoord(l=unif_l*u.deg, b=unif_b*u.deg, frame="galactic")
print(f"Unified axis (galactic): l={unif_l:.2f}°, b={unif_b:.2f}°")

# =====================================================================
# 3) Rotate sky so unified axis → new north pole
# =====================================================================
# astropy does this via position_angle method and spherical trig rotation
pa = coords.position_angle(unif)
theta = coords.separation(unif).rad      # polar angle from new axis
phi = pa.rad                             # azimuth around new axis

print("\nRotation complete:")
print(f"  min(theta)={np.min(theta)*180/np.pi:.2f}°  max={np.max(theta)*180/np.pi:.2f}°")

# =====================================================================
# 4) Spherical-harmonic basis evaluation
# =====================================================================
ell_max = 15
print(f"\nComputing multipoles up to ell={ell_max}")

alm = {}
for ell in range(ell_max+1):
    for m in range(-ell, ell+1):
        Y = sph_harm(m, ell, phi, theta)     # Y_lm(phi,theta)
        alm[(ell,m)] = np.sum(Y)/N           # basic unweighted estimator

# =====================================================================
# 5) Power spectrum C_ell
# =====================================================================
Cl = []
for ell in range(ell_max+1):
    s = 0.0
    for m in range(-ell, ell+1):
        s += np.abs(alm[(ell,m)])**2
    Cl.append(s/(2*ell+1))

# =====================================================================
# 6) Dipole and quadrupole interpretation
# =====================================================================
# Dipole vector is proportional to real/imag parts of a_{1m}
a10 = alm[(1,0)]
a11 = alm[(1,1)]
a1m1 = alm[(1,-1)]

# Convert dipole into cartesian direction
Dx = np.real(a11) - np.real(a1m1)
Dy = np.imag(a11) + np.imag(a1m1)
Dz = np.real(a10)

D = np.array([Dx, Dy, Dz])
D_norm = D / np.linalg.norm(D)

dipole_theta = np.arccos(D_norm[2])*180/np.pi
dipole_phi = np.arctan2(D_norm[1], D_norm[0])*180/np.pi

# =====================================================================
# 7) Print results
# =====================================================================
print("\n================ DIPLOE ==================")
print(f"Dipole amplitude: {np.linalg.norm(D):.4e}")
print(f"Dipole direction (in unified-axis frame):")
print(f"   theta={dipole_theta:.2f}°, phi={dipole_phi:.2f}°")

print("\n================ QUADRUPOLE ================")
print("Quadrupole power C2 =", Cl[2])

# =====================================================================
# 8) Plot C_ell spectrum
# =====================================================================
plt.figure(figsize=(9,5))
plt.plot(range(len(Cl)), Cl, marker="o")
plt.yscale("log")
plt.xlabel("ell")
plt.ylabel("C_ell (log scale)")
plt.title("FRB multipole power spectrum (wrt unified axis)")
plt.grid(alpha=0.3)
plt.savefig("frb_axis_multipole_spectrum.png", dpi=200)

# =====================================================================
# 9) Reconstructed low-l sky map (ℓ<=5)
# =====================================================================
ell_plot = 5
phi_grid = np.linspace(0, 2*np.pi, 400)
theta_grid = np.linspace(0, np.pi, 200)
Phi, Theta = np.meshgrid(phi_grid, theta_grid)
map_rec = np.zeros_like(Phi, dtype=float)

for ell in range(ell_plot+1):
    for m in range(-ell, ell+1):
        Y = sph_harm(m, ell, Phi, Theta)
        map_rec += np.real(alm[(ell,m)] * Y)

plt.figure(figsize=(12,5))
plt.imshow(map_rec, extent=[0,360,180,0], cmap="viridis")
plt.colorbar(label="reconstructed intensity")
plt.title("Reconstructed FRB sky (ℓ ≤ 5) in unified-axis frame")
plt.xlabel("phi (deg)")
plt.ylabel("theta (deg)")
plt.savefig("frb_axis_multipole_map.png", dpi=200)

print("\nMaps saved:")
print("  frb_axis_multipole_spectrum.png")
print("  frb_axis_multipole_map.png")
print("="*70)
print("analysis complete.")
print("="*70)
