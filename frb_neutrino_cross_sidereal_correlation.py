import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
from scipy.stats import pearsonr

# ------------------------------------------------------------
# helper: compute sidereal phases in [0,1)
# ------------------------------------------------------------
def sidereal_phase(mjd_array):
    t = Time(mjd_array, format="mjd", scale="utc")
    gmst_hours = t.sidereal_time("mean", "greenwich").value
    return (gmst_hours / 24.0) % 1.0

# ------------------------------------------------------------
# load data
# ------------------------------------------------------------
print("[INFO] loading data...")

frb = pd.read_csv("frbs.csv")
nu  = pd.read_csv("neutrinos.csv")

frb = frb.dropna(subset=["mjd"])
nu  = nu.dropna(subset=["mjd"])

print(f"[INFO] FRBs with valid MJD: {len(frb)}")
print(f"[INFO] neutrinos with valid MJD: {len(nu)}")

# ------------------------------------------------------------
# compute sidereal phases
# ------------------------------------------------------------
print("[INFO] computing sidereal phases...")

phi_frb = sidereal_phase(frb["mjd"].values)
phi_nu  = sidereal_phase(nu["mjd"].values)

# ------------------------------------------------------------
# bin both datasets on the same grid
# ------------------------------------------------------------
nbins = 40
edges = np.linspace(0, 1, nbins + 1)
centers = 0.5 * (edges[:-1] + edges[1:])

H_frb, _ = np.histogram(phi_frb, bins=edges)
H_nu,  _ = np.histogram(phi_nu,  bins=edges)

# normalize to probability densities
H_frb = H_frb / H_frb.sum()
H_nu  = H_nu  / H_nu.sum()

# ------------------------------------------------------------
# compute Pearson correlation
# ------------------------------------------------------------
corr, p_corr = pearsonr(H_frb, H_nu)

print("------------------------------------------------------------")
print("FRB–neutrino sidereal phase cross-correlation")
print("------------------------------------------------------------")
print(f"Pearson r = {corr:.4f}")
print(f"p-value   = {p_corr:.4f}")
print("------------------------------------------------------------")

# ------------------------------------------------------------
# Monte Carlo test:
# shuffle neutrino phases (uniform) and recompute correlation
# ------------------------------------------------------------
print("[INFO] running Monte Carlo...")

Nmc = 20000
r_random = np.zeros(Nmc)

for i in range(Nmc):
    phi_nu_rand = np.random.rand(len(phi_nu))
    H_nu_rand, _ = np.histogram(phi_nu_rand, bins=edges)
    H_nu_rand = H_nu_rand / H_nu_rand.sum()
    r_random[i] = pearsonr(H_frb, H_nu_rand)[0]

p_mc = (np.abs(r_random) >= np.abs(corr)).mean()

print(f"[INFO] Monte Carlo P(|r_rand| >= |r_obs|) = {p_mc:.4f}")
print("------------------------------------------------------------")

# ------------------------------------------------------------
# plot
# ------------------------------------------------------------
plt.figure(figsize=(12,6))
plt.step(centers, H_frb, where="mid", label="FRB sidereal distribution", alpha=0.6)
plt.step(centers, H_nu,  where="mid", label="neutrino sidereal distribution", alpha=0.6)
plt.xlabel("sidereal phase")
plt.ylabel("probability density")
plt.title("FRB vs neutrino sidereal phase distributions")
plt.legend()

plt.savefig("frb_neutrino_cross_sidereal.png", dpi=150)
print("[INFO] saved → frb_neutrino_cross_sidereal.png")
print("[done]")
