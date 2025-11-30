import numpy as np
import pandas as pd
from scipy.special import sph_harm, spherical_jn
import matplotlib.pyplot as plt
from tqdm import tqdm

print("="*70)
print(" FRB 3D SPHERICAL–BESSEL TOMOGRAPHY (TEST 15)")
print("="*70)

import sys
fname = sys.argv[1] if len(sys.argv) > 1 else "frbs_unified.csv"
frb = pd.read_csv(fname)

theta = np.radians(frb["theta_unified"].values)
phi   = np.radians(frb["phi_unified"].values)
z     = frb["z_est"].values

# linear approx comoving distance
r = 4300 * z    # Mpc

# define modes
ells = np.arange(1,6)
ks   = np.linspace(0.001,0.02,30)

A_k = np.zeros((len(ells), len(ks)))

for i,ell in enumerate(ells):
    for j,k in enumerate(ks):

        j_l = spherical_jn(ell, k*r)
        Ylm = 0

        for m in range(-ell, ell+1):
            Ylm += np.sum(sph_harm(m, ell, phi, theta) * j_l)

        A_k[i,j] = np.abs(Ylm)

print("computing Monte Carlo null...")
NMC = 500
A_k_null = np.zeros((NMC, len(ells), len(ks)))

for s in tqdm(range(NMC)):
    idx = np.random.permutation(len(r))
    for i,ell in enumerate(ells):
        for j,k in enumerate(ks):
            j_l = spherical_jn(ell, k*r[idx])
            Ylm = 0
            for m in range(-ell,ell+1):
                Ylm += np.sum(sph_harm(m,ell,phi,theta)*j_l)
            A_k_null[s,i,j] = np.abs(Ylm)

# compute p-values
pvals = np.mean(A_k_null >= A_k[None,:,:], axis=0)

print("\n3D Bessel tomography p-values (per ell,k):")
for i,ell in enumerate(ells):
    print(f"ell={ell}: min p = {pvals[i].min():.4g}")

plt.figure(figsize=(10,6))
plt.imshow(A_k, aspect='auto', origin='lower',
           extent=[ks[0], ks[-1], ells[0], ells[-1]])
plt.colorbar(label="|a_{lm}(k)|")
plt.xlabel("k (1/Mpc)")
plt.ylabel("ell")
plt.title("FRB 3D Spherical–Bessel Power")
plt.savefig("frb_3d_bessel_power.png", dpi=200)

print("\nsaved: frb_3d_bessel_power.png")
print("="*70)
print(" Test 15 complete")
print("="*70)
