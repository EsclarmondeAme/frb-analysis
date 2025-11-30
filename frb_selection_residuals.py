import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

SIDEREAL_DAY_SEC = 86164.0905  # seconds

def compute_sidereal_phase_from_mjd(mjd_array):
    """
    turn MJD (days) into sidereal phase in [0,1)
    using the same 'days → seconds → sidereal day' trick
    as in the earlier harmonic script.
    """
    mjd = np.asarray(mjd_array, dtype=float)
    mjd0 = np.nanmin(mjd)
    t_sec = (mjd - mjd0) * 86400.0
    phase = (t_sec / SIDEREAL_DAY_SEC) % 1.0
    return phase

def harmonic_coeffs(phases, n_max=4):
    """
    compute A_n, B_n, R_n up to n_max
    for phases in [0,1).
    """
    phi = np.asarray(phases, dtype=float)
    R = []
    A_list = []
    B_list = []
    for n in range(1, n_max + 1):
        ang = 2.0 * np.pi * n * phi
        cos_term = np.cos(ang).mean()
        sin_term = np.sin(ang).mean()
        R_n = np.sqrt(cos_term**2 + sin_term**2)
        A_list.append(cos_term)
        B_list.append(sin_term)
        R.append(R_n)
    return np.array(A_list), np.array(B_list), np.array(R)

def monte_carlo_pvalues(R_obs, n_samples=2000, n_events=600, n_max=4, rng=None):
    """
    monte carlo: random uniform phases, same N,
    compute R_n distribution and p(R_rand >= R_obs).
    """
    if rng is None:
        rng = np.random.default_rng()
    R_obs = np.asarray(R_obs)
    counts = np.zeros_like(R_obs, dtype=float)

    for _ in range(n_samples):
        phi_rand = rng.random(n_events)
        _, _, R_rand = harmonic_coeffs(phi_rand, n_max=n_max)
        counts += (R_rand >= R_obs).astype(float)

    pvals = counts / float(n_samples)
    return pvals

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

print("============================================================")
print("FRB sidereal selection + residual harmonic test")
print("============================================================")

# 1) load frbs.csv
frb = pd.read_csv("frbs.csv")
if "mjd" not in frb.columns:
    raise ValueError("frbs.csv has no 'mjd' column – cannot continue")

frb = frb.dropna(subset=["mjd"]).copy()
if len(frb) == 0:
    raise ValueError("no FRBs with valid MJD")

print(f"[info] FRBs with valid MJD: {len(frb)}")

# 2) compute sidereal phase
phi = compute_sidereal_phase_from_mjd(frb["mjd"].values)
frb["phi_sid"] = phi

# 3) histogram of phases
nbins = 24
bins = np.linspace(0.0, 1.0, nbins + 1)
counts, edges = np.histogram(phi, bins=bins)
centers = 0.5 * (edges[:-1] + edges[1:])
N_total = counts.sum()

# 4) smooth the histogram to estimate selection function S(phi)
# gaussian kernel over bin index
sigma_bins = 1.5
idx = np.arange(-3 * int(np.ceil(sigma_bins)),
                3 * int(np.ceil(sigma_bins)) + 1)
kernel = np.exp(-0.5 * (idx / sigma_bins) ** 2)
kernel /= kernel.sum()

# extend counts periodically to convolve on the circle
c_ext = np.tile(counts, 3)
s_ext = np.convolve(c_ext, kernel, mode="same")
# take the middle block as smoothed counts
start = len(counts)
stop = 2 * len(counts)
smooth_counts = s_ext[start:stop]

# normalize selection so mean is 1.0
selection = smooth_counts / np.mean(smooth_counts)

# avoid zeros
selection_safe = np.where(selection <= 1e-6, 1e-6, selection)

# 5) build "flattened" residual counts:
#    res_counts ∝ counts / selection_safe, then renormalize to same total
res_counts_raw = counts / selection_safe
res_counts = res_counts_raw * (N_total / res_counts_raw.sum())

# 6) approximate residual sample by putting integer counts at bin centers
res_counts_int = np.rint(res_counts).astype(int)
residual_phases = np.repeat(centers, res_counts_int)
print(f"[info] effective residual sample size: {len(residual_phases)}")

# 7) harmonic analysis: original vs residual
A_orig, B_orig, R_orig = harmonic_coeffs(phi, n_max=4)
A_res, B_res, R_res = harmonic_coeffs(residual_phases, n_max=4)

# 8) monte carlo p-values for residual sample
rng = np.random.default_rng(12345)
p_res = monte_carlo_pvalues(
    R_res,
    n_samples=2000,
    n_events=len(residual_phases),
    n_max=4,
    rng=rng,
)

# ------------------------------------------------------------
# text output
# ------------------------------------------------------------
print("------------------------------------------------------------")
print("harmonic amplitudes (original phases)")
print(" n   A_n        B_n        R_n")
for n in range(1, 5):
    print(f" {n}  {A_orig[n-1]: .4f}   {B_orig[n-1]: .4f}   {R_orig[n-1]: .4f}")
print("------------------------------------------------------------")
print("harmonic amplitudes after selection-flattening")
print(" n   A_res      B_res      R_res     p(R_rand>=R_res)")
for n in range(1, 5):
    print(
        f" {n}  {A_res[n-1]: .4f}   {B_res[n-1]: .4f}   {R_res[n-1]: .4f}"
        f"     {p_res[n-1]: .4f}"
    )

print("------------------------------------------------------------")
print("interpretation:")
print(" - if p_res for n=1 is still very small (< 0.05),")
print("   then even after dividing out a smooth selection,")
print("   there remains significant dipole modulation (cone-like signal).")
print(" - if p_res becomes large (~0.1–0.5),")
print("   the strong modulation was mostly due to selection.")
print("============================================================")

# ------------------------------------------------------------
# plots
# ------------------------------------------------------------

# plot 1: original histogram + smooth selection + residual counts
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

width = edges[1] - edges[0]

# top: original and smoothed
ax[0].bar(
    centers,
    counts,
    width=width,
    align="center",
    alpha=0.3,
    label="FRB counts",
)
ax[0].plot(
    centers,
    smooth_counts,
    "-o",
    label="smoothed selection (unnormalized)",
)
ax[0].set_ylabel("counts per bin")
ax[0].legend()
ax[0].set_title("FRB sidereal phases: counts and smooth selection")

# bottom: residual counts
ax[1].bar(
    centers,
    res_counts,
    width=width,
    align="center",
    alpha=0.5,
    label="selection-flattened counts",
)
ax[1].axhline(N_total / nbins, color="k", linestyle="--", label="uniform")
ax[1].set_xlabel("sidereal phase")
ax[1].set_ylabel("effective counts per bin")
ax[1].legend()
ax[1].set_title("residual distribution after dividing out smooth selection")

plt.tight_layout()
plt.savefig("frb_selection_and_residuals.png", dpi=150)
plt.close()

# plot 2: residual histogram alone, normalized to probability density
fig, ax = plt.subplots(figsize=(10, 4))
prob_res = res_counts / res_counts.sum()
ax.bar(
    centers,
    prob_res,
    width=width,
    align="center",
    alpha=0.6,
    label="residual probability",
)
ax.axhline(1.0 / nbins, color="k", linestyle="--", label="uniform (1/nbins)")
ax.set_xlabel("sidereal phase")
ax.set_ylabel("probability per bin")
ax.set_title("FRB sidereal residuals (after selection correction)")
ax.legend()
plt.tight_layout()
plt.savefig("frb_selection_residuals_only.png", dpi=150)
plt.close()
