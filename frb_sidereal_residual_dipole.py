"""
frb_sidereal_residual_dipole.py

goal
----
1. take FRB arrival times from frbs.csv
2. convert to a simple "sidereal phase" (using MJD fractional day)
3. estimate the selection function S_hat(phase) directly from the data
   using a smooth spline fit to the histogram
4. divide out S_hat to get a "residual" distribution that should be
   uniform if there is no extra cosmic anisotropy
5. measure the dipole harmonic of the residual distribution and
   compare it to random expectations via Monte Carlo

interpretation:
- if residual dipole R1_resid is small and has p ~ O(0.1–1),
  then everything we saw before was survey/selection.
- if R1_resid is still large with p << 0.05, that’s evidence for
  a genuine cosmic (cone-like) modulation on top of survey effects.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. load FRBs and compute sidereal phase
# ---------------------------------------------------------------------

frb = pd.read_csv("frbs.csv")

if "mjd" not in frb.columns:
    raise ValueError("frbs.csv must have an 'mjd' column")

mjd = frb["mjd"].values.astype(float)

# simple phase: fractional part of MJD (solar vs sidereal difference
# is tiny compared to what we are probing here)
phase = mjd % 1.0

# keep in [0,1)
phase = np.mod(phase, 1.0)

N = len(phase)
print("FRBs with valid MJD:", N)

# ---------------------------------------------------------------------
# 2. estimate selection function S_hat(phase) from histogram + spline
# ---------------------------------------------------------------------

# choose number of bins (not too many; we want smooth structure)
nbins = 24
counts, edges = np.histogram(phase, bins=nbins, range=(0.0, 1.0))
centers = 0.5 * (edges[:-1] + edges[1:])

# avoid zeros (add a tiny floor before taking logs or fitting)
counts_safe = counts.astype(float)
counts_safe[counts_safe <= 0] = 0.5

# fit a smooth spline to log(counts) to capture broad structure only
# smoothing factor "s" controls smoothness; tweak if needed
log_counts = np.log(counts_safe)
spline = UnivariateSpline(centers, log_counts, s=len(centers) * 0.5)
log_S_hat = spline(centers)
S_hat_bins = np.exp(log_S_hat)

# normalize S_hat so its mean is 1 (so it’s "relative sensitivity")
S_hat_bins /= np.mean(S_hat_bins)

# map S_hat from bins back to each event by interpolation
S_hat_at_phase = np.interp(phase, centers, S_hat_bins)

# ---------------------------------------------------------------------
# 3. build residual weights and residual histogram
# ---------------------------------------------------------------------

# if S_hat is large → we down-weight (many FRBs expected there)
# if S_hat is small → we up-weight
weights = 1.0 / S_hat_at_phase

# residual histogram (weighted)
res_counts, _ = np.histogram(phase, bins=nbins, range=(0.0, 1.0), weights=weights)
res_counts = res_counts.astype(float)
res_counts[res_counts <= 0] = 1e-9

# normalize residual histogram to be a probability distribution
res_probs = res_counts / np.sum(res_counts)

# ---------------------------------------------------------------------
# 4. compute dipole harmonic of residual distribution
# ---------------------------------------------------------------------

phi_centers = centers  # same centers

# complex Fourier coefficient for n=1 from residual probs
exp_i = np.exp(2j * np.pi * phi_centers)
C1 = np.sum(res_probs * exp_i)
R1_resid = np.abs(C1)

print("------------------------------------------------------------")
print("residual dipole harmonic (after dividing selection function)")
print("R1_resid =", R1_resid)

# ---------------------------------------------------------------------
# 5. Monte Carlo: compare to random (uniform) expectations
# ---------------------------------------------------------------------

n_mc = 5000
rng = np.random.default_rng(12345)
R1_rand = np.empty(n_mc, dtype=float)

for i in range(n_mc):
    # draw N uniform phases
    phases_mc = rng.random(N)
    # build histogram with same binning, same pipeline
    c_mc, _ = np.histogram(phases_mc, bins=nbins, range=(0.0, 1.0))
    c_mc = c_mc.astype(float)
    c_mc[c_mc <= 0] = 1e-9
    p_mc = c_mc / np.sum(c_mc)

    C1_mc = np.sum(p_mc * np.exp(2j * np.pi * centers))
    R1_rand[i] = np.abs(C1_mc)

p_ge = np.mean(R1_rand >= R1_resid)

print("------------------------------------------------------------")
print("Monte Carlo comparison (uniform phases, same binning)")
print(f"median R1_rand  = {np.median(R1_rand):.4f}")
print(f"95th pct R1_rand= {np.percentile(R1_rand,95):.4f}")
print(f"P(R1_rand >= R1_resid) = {p_ge:.4f}")
print("------------------------------------------------------------")

# ---------------------------------------------------------------------
# 6. make a diagnostic plot
# ---------------------------------------------------------------------

phi_grid = np.linspace(0.0, 1.0, 400)

# step 1: original histogram (normalized)
orig_counts, _ = np.histogram(phase, bins=nbins, range=(0.0, 1.0))
orig_probs = orig_counts / np.sum(orig_counts)
orig_probs_smooth = np.interp(phi_grid, centers, orig_probs)

# step 2: spline-based selection function rescaled to probability density
S_hat_smooth = np.interp(phi_grid, centers, S_hat_bins)
S_hat_smooth /= np.trapz(S_hat_smooth, phi_grid)

# step 3: residual probability density (res_probs interpolated)
res_probs_smooth = np.interp(phi_grid, centers, res_probs)
res_probs_smooth /= np.trapz(res_probs_smooth, phi_grid)

plt.figure(figsize=(10, 6))

# show original histogram + selection function + residual
plt.step(centers, orig_probs, where="mid", alpha=0.4, label="original sidereal histogram")
plt.plot(phi_grid, S_hat_smooth, label="estimated selection S_hat(φ)")
plt.plot(phi_grid, res_probs_smooth, label="residual distribution (divided by S_hat)")

plt.xlabel("sidereal phase φ")
plt.ylabel("probability density (normalized)")
plt.title("FRB sidereal phase: selection vs residual")
plt.legend()
plt.tight_layout()
plt.savefig("frb_sidereal_residual_dipole.png", dpi=150)
plt.close()

print("[INFO] saved plot → frb_sidereal_residual_dipole.png")
print("[INFO] done.")
