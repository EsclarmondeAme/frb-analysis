import numpy as np
import pandas as pd
from scipy.special import sph_harm
from scipy.stats import poisson
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------
# load unified catalogue
# ---------------------------------------------------------
import sys
if len(sys.argv) < 2:
    print("usage: python frb_3d_unified_likelihood_test.py frbs_unified.csv")
    sys.exit()

path = sys.argv[1]
frb = pd.read_csv(path)

theta = np.radians(frb["theta_unified"].values)
phi   = np.radians(frb["phi_unified"].values)
z     = frb["z_est"].values

N = len(frb)

print("="*70)
print(" FRB 3D UNIFIED-LIKELIHOOD TOMOGRAPHY (TEST 14)")
print("="*70)

# ---------------------------------------------------------
# redshift bins
# ---------------------------------------------------------
nbins = 4
z_edges = np.quantile(z, np.linspace(0,1,nbins+1))
print("redshift-bin edges:", z_edges)

bin_index = np.digitize(z, z_edges) - 1
bin_index[bin_index==nbins] = nbins-1

# ---------------------------------------------------------
# harmonic power per bin
# ---------------------------------------------------------
def compute_harm(theta, phi):
    ellmax = 4
    a_lm = {}
    for ell in range(1, ellmax+1):
        for m in range(-ell, ell+1):
            Y = sph_harm(m, ell, phi, theta)
            a_lm[(ell,m)] = np.sum(Y.conjugate())
    return a_lm

# ---------------------------------------------------------
# shell-likelihood (warped-shell model)
# ---------------------------------------------------------
def warped_shell_likelihood(theta_bin, phi_bin):
    # fit a simple R(phi) = a + b*sin(phi) + c*cos(phi)
    # likelihood: Poisson binned counts vs model
    ph = phi_bin
    th = np.degrees(theta_bin)
    bins = [0,10,25,40,90]
    H, _ = np.histogram(th, bins=bins)
    # toy model: sinusoidal distortion
    A = np.mean(H)
    model = np.array([
        A*0.7,
        A*0.9,
        A*1.5,
        A*1.0
    ])
    return np.sum(poisson.logpmf(H, model))

# ---------------------------------------------------------
# selection-function likelihood
# ---------------------------------------------------------
def selection_likelihood(theta_bin, phi_bin):
    th = np.degrees(theta_bin)
    H, _ = np.histogram(th, [0,10,25,40,90])
    # expected from selection-only model (from Test 12)
    expected = np.array([7,40,72,398]) * (len(th)/600)
    return np.sum(poisson.logpmf(H, expected+1e-9))

# ---------------------------------------------------------
# per-bin total likelihood
# ---------------------------------------------------------
def total_likelihood(idx):
    th = theta[idx]
    ph = phi[idx]
    L_harm = 0
    a = compute_harm(th, ph)
    for ell in range(1,5):
        for m in range(-ell, ell+1):
            L_harm += -0.5 * np.abs(a[(ell,m)])**2 / (N/20)

    L_shell = warped_shell_likelihood(th, ph)
    L_sel = selection_likelihood(th, ph)
    return L_harm + L_shell + L_sel

# ---------------------------------------------------------
# observed bin-by-bin likelihood
# ---------------------------------------------------------
L_obs = []
for k in range(nbins):
    Lk = total_likelihood(bin_index==k)
    L_obs.append(Lk)
    print(f"bin {k+1} likelihood = {Lk:.2f}")

L_total_obs = np.sum(L_obs)
print("-------------------------------------------------")
print("total observed 3D-likelihood =", L_total_obs)
print("-------------------------------------------------")

# ---------------------------------------------------------
# Monte Carlo null: shuffle redshifts
# ---------------------------------------------------------
MC = 1000
L_null = []
print("running Monte Carlo null...")

for _ in tqdm(range(MC)):
    z_shuffled = np.random.permutation(z)
    idx_shuf = np.digitize(z_shuffled, z_edges) - 1
    idx_shuf[idx_shuf==nbins] = nbins-1

    Ls = 0
    for k in range(nbins):
        Ls += total_likelihood(idx_shuf==k)
    L_null.append(Ls)

L_null = np.array(L_null)
pval = np.mean(L_null >= L_total_obs)

print("-------------------------------------------------")
print(f" Monte Carlo p-value = {pval:.5f}")
print("-------------------------------------------------")

# ---------------------------------------------------------
# plot
# ---------------------------------------------------------
plt.figure(figsize=(7,5))
plt.hist(L_null, bins=40, alpha=0.7, label="null")
plt.axvline(L_total_obs, color="red", lw=2, label="observed")
plt.xlabel("3D unified-likelihood")
plt.ylabel("count")
plt.legend()
plt.title("FRB Test 14: 3D Unified Likelihood Tomography")
plt.tight_layout()
plt.savefig("frb_3d_unified_likelihood.png", dpi=150)
print("saved: frb_3d_unified_likelihood.png")
print("="*70)
print("Test 14 complete.")
print("="*70)
