import numpy as np
import pandas as pd
from scipy.special import sph_harm
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------------------------------
# TEST 13: FRB 3D SPHERICAL-HARMONIC TOMOGRAPHY
# -----------------------------------------------------

print("="*70)
print(" FRB 3D SPHERICAL-HARMONIC TOMOGRAPHY (TEST 13)")
print("="*70)

# -------------------------
# Load unified-axis dataset
# -------------------------
frb = pd.read_csv("frbs_unified.csv")

theta = np.radians(frb["theta_unified"].values)
phi   = np.radians(frb["phi_unified"].values)
z     = frb["z_est"].values

# -------------------------
# Define redshift bins
# -------------------------
nbins = 4
percentiles = np.linspace(0,100,nbins+1)
zbins = np.percentile(z, percentiles)

print("redshift bin edges:", zbins)

# -------------------------
# compute a_lm per bin
# -------------------------
ell_max = 4
alm_bins = []

for i in range(nbins):
    mask = (z>=zbins[i]) & (z<zbins[i+1])
    th = theta[mask]
    ph = phi[mask]

    print(f"\ncomputing a_lm for bin {i+1}/{nbins} (N={len(th)})")
    
    alm = {}
    for ell in range(1,ell_max+1):
        for m in range(-ell, ell+1):
            Y = sph_harm(m, ell, ph, th)
            alm[(ell,m)] = np.sum(Y.conjugate())

    alm_bins.append(alm)

# ---------------------------------------------------------
# compute warp parameters a(z), b(z), c(z), d(z) per bin
# from low-order m-modes
# ---------------------------------------------------------
def compute_warp(alm):
    # m= +-1 and m=+-2 encode warp
    a = np.real(alm[(1,-1)] + alm[(1,1)])
    b = np.real(alm[(1,-1)] - alm[(1,1)])
    c = np.real(alm[(2,-2)] + alm[(2,2)])
    d = np.real(alm[(2,-2)] - alm[(2,2)])
    return a,b,c,d

warp_params = []
for i in range(nbins):
    warp_params.append(compute_warp(alm_bins[i]))

# ---------------------------------------------------------
# Plot warp parameters vs redshift
# ---------------------------------------------------------
zmid = 0.5*(zbins[:-1]+zbins[1:])
warp_params = np.array(warp_params)

plt.figure(figsize=(10,6))
plt.plot(zmid, warp_params[:,0], label='a(z)')
plt.plot(zmid, warp_params[:,1], label='b(z)')
plt.plot(zmid, warp_params[:,2], label='c(z)')
plt.plot(zmid, warp_params[:,3], label='d(z)')
plt.xlabel("redshift")
plt.ylabel("warp parameter amplitude")
plt.title("FRB warped-shell harmonic coefficients vs redshift")
plt.legend()
plt.grid()
plt.savefig("frb_3d_warp_vs_redshift.png", dpi=200)
print("\nSaved: frb_3d_warp_vs_redshift.png")

# ---------------------------------------------------------
# Monte Carlo test: does warp drift with redshift?
# ---------------------------------------------------------
print("\nRunning Monte Carlo null for redshift-independence...")

T_obs = np.mean(np.abs(np.diff(warp_params, axis=0)))  # observed drift amplitude

nsim = 2000
T_null = []

for s in tqdm(range(nsim)):
    zperm = np.random.permutation(z)
    Tbins = []
    zbins_tmp = zbins.copy()
    for i in range(nbins):
        mask = (zperm>=zbins_tmp[i]) & (zperm<zbins_tmp[i+1])
        th = theta[mask]
        ph = phi[mask]
        alm = {}
        for ell in range(1,ell_max+1):
            for m in range(-ell, ell+1):
                Y = sph_harm(m, ell, ph, th)
                alm[(ell,m)] = np.sum(Y.conjugate())
        Tbins.append(compute_warp(alm))
    Tbins = np.array(Tbins)
    T_null.append(np.mean(np.abs(np.diff(Tbins, axis=0))))

T_null = np.array(T_null)
pval = np.mean(T_null >= T_obs)

print(f"\nObserved drift statistic T_obs = {T_obs:.5f}")
print(f"Monte Carlo null mean          = {np.mean(T_null):.5f}")
print(f"Monte Carlo p-value            = {pval:.5f}")

print("\nSaved 3D tomography results.")
print("="*70)
print(" Test 13 complete.")
print("="*70)
