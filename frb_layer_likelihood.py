"""
frb_layer_likelihood.py
------------------------------------------------------------
likelihood-ratio test of a layered angular model vs isotropy.

bands (from previous analysis):
  inner :  0°–10°
  middle: 10°–25°
  outer : 25°–40°
  far   : 40°–90°   (control region)

for each band:
  - compute observed counts n_i
  - compute isotropic expected counts mu_i
  - compute density ratios n_i / mu_i

then:
  - null model M0: isotropic (p_i from geometry)
  - alt  model M1: layered (p_i = n_i / N, empirical)

compute:
  - log L0, log L1 (Poisson likelihood)
  - lambda = 2 (log L1 - log L0)
  - monte-carlo p-value under isotropic sky
------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# ------------------------------------------------------------
# unified axis
# ------------------------------------------------------------
AXIS_L = 159.85
AXIS_B = -0.51
axis = SkyCoord(l=AXIS_L*u.deg, b=AXIS_B*u.deg, frame="galactic")

print("="*70)
print("frb layered likelihood test")
print("likelihood-ratio comparison: layered vs isotropic")
print("="*70)

# ------------------------------------------------------------
# load frbs
# ------------------------------------------------------------
try:
    frbs = pd.read_csv("frbs.csv")
except FileNotFoundError:
    print("\n[error] frbs.csv not found")
    raise SystemExit

print("\n1. frb catalog")
print("------------------------------------------------------------")
print(f"total frbs: {len(frbs)}")

coords = SkyCoord(ra=frbs["ra"].values*u.deg,
                  dec=frbs["dec"].values*u.deg,
                  frame="icrs").galactic
sep = coords.separation(axis).deg
frbs["sep_deg"] = sep

# ------------------------------------------------------------
# define four bands
# ------------------------------------------------------------
bands = [
    ("inner",  0.0, 10.0),
    ("middle", 10.0, 25.0),
    ("outer",  25.0, 40.0),
    ("far",    40.0, 90.0),
]

total = len(frbs)

def band_prob(theta_low_deg, theta_high_deg):
    t1 = np.radians(theta_low_deg)
    t2 = np.radians(theta_high_deg)
    # probability that an isotropic direction lies in [low, high]
    return (np.cos(t1) - np.cos(t2)) / 2.0

print("\n2. observed vs isotropic per band")
print("------------------------------------------------------------")

obs_counts = []
exp_counts = []
density_ratios = []

for name, low, high in bands:
    mask = (sep >= low) & (sep < high)
    n_obs = np.sum(mask)
    p_iso = band_prob(low, high)
    mu = total * p_iso

    obs_counts.append(n_obs)
    exp_counts.append(mu)

    ratio = n_obs / mu if mu > 0 else np.nan
    density_ratios.append(ratio)

    print(f"{name:6s} band {low:4.1f}°–{high:4.1f}°:")
    print(f"   observed n_i   = {n_obs:5d}")
    print(f"   expected mu_i  = {mu:7.2f}")
    print(f"   density ratio  = {ratio:7.2f}\n")

obs_counts = np.array(obs_counts, dtype=float)
exp_counts = np.array(exp_counts, dtype=float)

# sanity: totals
print("total in bands 0°–90°:")
print(f"   observed sum n_i = {obs_counts.sum():.0f}")
print(f"   expected sum mu_i= {exp_counts.sum():.2f}")

# ------------------------------------------------------------
# likelihoods
# ------------------------------------------------------------
def log_poisson_likelihood(counts, mu):
    """
    log likelihood sum_i [ n_i ln(mu_i) - mu_i - ln(n_i!) ]
    constant ln(n_i!) drops out in likelihood ratio, so we omit it.
    """
    valid = mu > 0
    n = counts[valid]
    m = mu[valid]
    return np.sum(n * np.log(m) - m)

# null model: isotropic
logL0 = log_poisson_likelihood(obs_counts, exp_counts)

# alt model: empirical band probabilities (piecewise constant)
# MLE mu_i under alt model is just n_i (since we require total = N)
mu_alt = obs_counts.copy()
logL1 = log_poisson_likelihood(obs_counts, mu_alt)

lambda_lr = 2.0 * (logL1 - logL0)

print("\n3. likelihood comparison")
print("------------------------------------------------------------")
print(f"log L0 (isotropic)     = {logL0:.2f}")
print(f"log L1 (layered model) = {logL1:.2f}")
print(f"lambda = 2 (logL1 - logL0) = {lambda_lr:.2f}")

# ------------------------------------------------------------
# monte-carlo null for lambda
# ------------------------------------------------------------
print("\n4. monte-carlo null for lambda")
print("------------------------------------------------------------")

N_SIM = 20000
print(f"simulating {N_SIM} isotropic catalogs...")

# generate random cos(theta) uniformly in [-1, 1]
u_rand = np.random.uniform(-1.0, 1.0, size=(N_SIM, total))
theta_deg = np.degrees(np.arccos(u_rand))

lambda_sim = np.zeros(N_SIM)

for k, (name, low, high) in enumerate(bands):
    mask = (theta_deg >= low) & (theta_deg < high)
    n_sim = mask.sum(axis=1)
    mu = exp_counts[k]

    # logL0: isotropic, mu fixed
    # logL1: alt model with mu' = n_sim (MLE under unconstrained band model)
    logL0_sim = n_sim * np.log(mu) - mu
    # avoid log(0) when n_sim = 0: define n ln n - n = 0 at n=0
    with np.errstate(divide='ignore', invalid='ignore'):
        term_alt = np.where(n_sim > 0, n_sim * np.log(n_sim) - n_sim, 0.0)
    # difference 2*(logL1 - logL0) summed over bands
    lambda_sim += 2.0 * (term_alt - logL0_sim)

p_value = np.mean(lambda_sim >= lambda_lr)

print(f"\nnull distribution of lambda:")
print(f"   mean lambda(null)   = {lambda_sim.mean():.2f}")
print(f"   median lambda(null) = {np.median(lambda_sim):.2f}")
print(f"   95th percentile     = {np.percentile(lambda_sim, 95):.2f}")

print("\n5. significance")
print("------------------------------------------------------------")
print(f"observed lambda = {lambda_lr:.2f}")
print(f"monte-carlo p-value (lambda >= observed) = {p_value:.4f}")

if p_value < 0.001:
    print("verdict: strong evidence that a layered band model fits better than isotropy")
elif p_value < 0.01:
    print("verdict: significant improvement over isotropy")
elif p_value < 0.05:
    print("verdict: marginal improvement over isotropy")
else:
    print("verdict: no significant improvement vs isotropy")

print("\n" + "="*70)
print("analysis complete")
print("="*70)
