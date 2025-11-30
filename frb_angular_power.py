#!/usr/bin/env python3
# frb_angular_power.py
# compute spherical harmonic angular power spectrum (Cl) of FRB positions
# using simple pixelization (no healpy). grids 360x180.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

# load
frb = pd.read_csv("frbs.csv")
coords = SkyCoord(ra=frb["ra"].values*u.deg, dec=frb["dec"].values*u.deg, frame="icrs").galactic
l = coords.l.deg
b = coords.b.deg

# grid
nlon, nlat = 360, 180
grid = np.zeros((nlat, nlon))
for lon, lat in zip(l, b):
    i = int((lon % 360))
    j = int((lat + 90))
    if 0 <= i < nlon and 0 <= j < nlat:
        grid[j, i] += 1

# convert grid to spherical harmonic coefficients via discrete sum
# Y_lm evaluated at grid cell centers
from scipy.special import sph_harm

# grid centers
lon_grid = np.deg2rad(np.arange(nlon))
lat_grid = np.deg2rad(np.arange(-90, 90))
phi, theta = np.meshgrid(lon_grid, np.pi/2 - lat_grid)  # theta=colatitude

# flatten
phi_f = phi.ravel()
theta_f = theta.ravel()
vals = grid.ravel()

lmax = 50
Cl = np.zeros(lmax+1)

# compute alm
alm = {}
for ell in range(lmax+1):
    for m in range(-ell, ell+1):
        Y = sph_harm(m, ell, phi_f, theta_f)
        alm[(ell,m)] = np.sum(vals * Y.conjugate())
    Cl[ell] = np.sum([np.abs(alm[(ell,m)])**2 for m in range(-ell, ell+1)])/(2*ell+1)

# plot
plt.figure(figsize=(8,5))
ell = np.arange(lmax+1)
plt.plot(ell, Cl, marker='o')
plt.yscale('log')
plt.xlabel('ell')
plt.ylabel('Cl (log scale)')
plt.title('FRB angular power spectrum')
plt.grid(True)
plt.savefig('frb_Cl_spectrum.png', dpi=200)
plt.close()

print('====================================================')
print('FRB ANGULAR POWER SPECTRUM')
print('ell range: 0 -', lmax)
print('figure saved: frb_Cl_spectrum.png')
print('====================================================')
