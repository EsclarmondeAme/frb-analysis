import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from tqdm import tqdm

# ==============================================================
#   FRB LOW-l HARMONIC TEST  (TEST 9)
# ==============================================================

print("="*70)
print(" FRB LOW-ℓ HARMONIC RECONSTRUCTION (TEST 9)")
print("="*70)

# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------
try:
    frb = pd.read_csv("frbs_unified.csv")
except:
    print("ERROR: could not load frbs_unified.csv")
    exit()

if not {"theta_unified", "phi_unified"} <= set(frb.columns):
    print("ERROR: missing theta_unified, phi_unified — run frb_make_unified_axis_frame.py")
    exit()

theta = np.radians(frb["theta_unified"].values)
phi   = np.radians(frb["phi_unified"].values)
N = len(theta)

print(f"loaded {N} FRBs with unified-axis coordinates")

# ----------------------------------------------------------------------
# HARMONIC RECONSTRUCTION
# ----------------------------------------------------------------------
lmax = 8

# compute a_lm
alm = {}
for ell in range(lmax+1):
    for m in range(-ell, ell+1):
        Y = sph_harm(m, ell, phi, theta)
        alm[(ell, m)] = np.sum(Y.conjugate())   # unnormalized discrete sum

# compute power spectrum Cl
Cl = {}
for ell in range(1, lmax+1):
    s = 0
    for m in range(-ell, ell+1):
        s += np.abs(alm[(ell, m)])**2
    Cl[ell] = s.real / (2*ell + 1)

print("--------------------------------------------------------------")
print(" observed multipole powers:")
for ell in range(1, lmax+1):
    print(f"   ℓ={ell:2d}  Cℓ = {Cl[ell]:.3e}")
print("--------------------------------------------------------------")

# ----------------------------------------------------------------------
# MONTE CARLO NULL — 20,000 isotropic realizations
# ----------------------------------------------------------------------
nsim = 20000
Cl_null = {ell: [] for ell in range(1, lmax+1)}

print("running Monte Carlo isotropic null (20,000 sims)...")

for _ in tqdm(range(nsim)):
    # isotropic random directions on the sphere
    u = np.random.uniform(0, 1, N)
    th = np.arccos(2*u - 1)
    ph = np.random.uniform(0, 2*np.pi, N)

    for ell in range(1, lmax+1):
        s = 0
        for m in range(-ell, ell+1):
            Y = sph_harm(m, ell, ph, th)
            s += np.abs(np.sum(Y.conjugate()))**2
        Cl_null[ell].append((s.real/(2*ell+1)))

print("Monte Carlo complete.")

# compute p-values: P(Cℓ_null ≥ Cℓ_real)
pvals = {}
for ell in range(1, lmax+1):
    arr = np.array(Cl_null[ell])
    pvals[ell] = np.mean(arr >= Cl[ell])

print("--------------------------------------------------------------")
print(" Monte Carlo p-values for each multipole")
for ell in range(1, lmax+1):
    print(f"   ℓ={ell:2d}  p = {pvals[ell]:.5e}")
print("--------------------------------------------------------------")

# ----------------------------------------------------------------------
# PLOT POWER SPECTRUM
# ----------------------------------------------------------------------
plt.figure(figsize=(8,5))
ells = np.arange(1, lmax+1)
plt.plot(ells, [Cl[ell] for ell in ells], "o-", label="observed Cℓ")
plt.yscale("log")
plt.xlabel("multipole ℓ")
plt.ylabel("Cℓ (log scale)")
plt.title("FRB unified-axis low-ℓ power spectrum")
plt.grid(True)
plt.tight_layout()
plt.savefig("frb_lowell_power.png", dpi=200)
plt.close()

# ----------------------------------------------------------------------
# RECONSTRUCT LOW-l MAP
# ----------------------------------------------------------------------
# reconstruct full-sky map on grid
npix = 300
thg = np.linspace(0, np.pi, npix)
phg = np.linspace(0, 2*np.pi, npix)
TH, PH = np.meshgrid(thg, phg)

Ysum = np.zeros_like(TH, dtype=complex)
for ell in range(1, lmax+1):
    for m in range(-ell, ell+1):
        Y = sph_harm(m, ell, PH, TH)
        Ysum += alm[(ell, m)] * Y

Ymap = Ysum.real

plt.figure(figsize=(7,5))
plt.imshow(Ymap.T, origin="lower", aspect="auto", cmap="coolwarm",
           extent=[0,360,0,180])
plt.colorbar(label="reconstructed anisotropy")
plt.xlabel("ϕ (deg)")
plt.ylabel("θ (deg)")
plt.title("FRB low-ℓ harmonic reconstruction (ℓ ≤ 8)")
plt.tight_layout()
plt.savefig("frb_lowell_map.png", dpi=200)
plt.close()

print("saved figures:")
print("   frb_lowell_power.png")
print("   frb_lowell_map.png")

print("="*70)
print(" Test 9 complete.")
print("="*70)
