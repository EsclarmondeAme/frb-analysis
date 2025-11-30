"""
frb_shell_significance.py
------------------------------------------------------------
monte-carlo test of layered structure around the unified axis.

idea:
  - define three angular bands around the unified axis:
      inner  :  0°–10°
      middle : 10°–25°
      outer  : 25°–40°
  - count how many real frbs fall in each band
  - compare to isotropic expectation
  - compute a multi-band chi-square statistic
  - simulate many isotropic catalogs and measure how often
    they produce a chi-square as large as the observed one

output:
  - per-band excess and z-scores
  - overall chi-square
  - monte-carlo p-value for layered structure
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
print("frb shell significance test")
print("monte-carlo test of layered structure around unified axis")
print("="*70)

# ------------------------------------------------------------
# load frb catalog
# ------------------------------------------------------------
try:
    frbs = pd.read_csv("frbs.csv")
except FileNotFoundError:
    print("\n[error] frbs.csv not found")
    raise SystemExit

print("\n1. frb catalog")
print("------------------------------------------------------------")
print(f"total frbs: {len(frbs)}")

# compute angular separation if not already present
if "sep_deg" in frbs.columns:
    sep = frbs["sep_deg"].values
    print("using existing 'sep_deg' column (angle from axis)")
else:
    coords = SkyCoord(ra=frbs["ra"].values*u.deg,
                      dec=frbs["dec"].values*u.deg,
                      frame="icrs").galactic
    sep = coords.separation(axis).deg
    frbs["sep_deg"] = sep
    print("computed angular separation from unified axis")

# ------------------------------------------------------------
# define bands (inner / middle / outer)
# ------------------------------------------------------------
bands = [
    ("inner",  0.0, 10.0),
    ("middle", 10.0, 25.0),
    ("outer",  25.0, 40.0),
]

total = len(frbs)

def band_prob(theta_low_deg, theta_high_deg):
    """isotropic probability that a random direction lies in [low, high]"""
    t1 = np.radians(theta_low_deg)
    t2 = np.radians(theta_high_deg)
    return (np.cos(t1) - np.cos(t2)) / 2.0

print("\n2. observed band counts and expectations")
print("------------------------------------------------------------")

obs_counts = []
exp_counts = []
z_scores = []

for name, low, high in bands:
    in_band = (sep >= low) & (sep < high)
    n_obs = np.sum(in_band)
    p_iso = band_prob(low, high)
    n_exp = total * p_iso

    obs_counts.append(n_obs)
    exp_counts.append(n_exp)

    if n_exp > 0:
        z = (n_obs - n_exp) / np.sqrt(n_exp)
    else:
        z = np.nan
    z_scores.append(z)

    print(f"{name:6s} band {low:4.1f}°–{high:4.1f}°:")
    print(f"   observed: {n_obs:4d}")
    print(f"   expected: {n_exp:7.2f} (isotropic)")
    print(f"   excess  : {n_obs - n_exp:7.2f}")
    print(f"   z-score : {z:7.2f}\n")

obs_counts = np.array(obs_counts)
exp_counts = np.array(exp_counts)

# chi-square statistic across all bands
chi_obs = np.sum((obs_counts - exp_counts)**2 / exp_counts)
print("overall chi-square across bands:")
print(f"   chi²_obs = {chi_obs:.2f}")

# ------------------------------------------------------------
# monte-carlo null test
# ------------------------------------------------------------
print("\n3. monte-carlo null (isotropic catalogs)")
print("------------------------------------------------------------")

N_SIM = 20000  # number of random catalogs
print(f"simulating {N_SIM} isotropic catalogs with {total} frbs each...")

# isotropic angles: cos(theta) uniform in [-1, 1]
u_rand = np.random.uniform(-1.0, 1.0, size=(N_SIM, total))
theta_deg = np.degrees(np.arccos(u_rand))

chi_sim = np.zeros(N_SIM)

for i, (name, low, high) in enumerate(bands):
    mask = (theta_deg >= low) & (theta_deg < high)
    counts_sim = mask.sum(axis=1)  # counts per simulation
    # chi² contribution for this band
    chi_sim += (counts_sim - exp_counts[i])**2 / exp_counts[i]

# p-value = fraction of simulations with chi² >= chi_obs
p_value = np.mean(chi_sim >= chi_obs)

print(f"\nmonte-carlo distribution:")
print(f"   mean chi²(null)  = {chi_sim.mean():.2f}")
print(f"   median chi²(null)= {np.median(chi_sim):.2f}")
print(f"   95th percentile  = {np.percentile(chi_sim,95):.2f}")

print("\n4. significance")
print("------------------------------------------------------------")
print(f"observed chi²: {chi_obs:.2f}")
print(f"monte-carlo p-value (>= chi²_obs): {p_value:.4f}")

if p_value < 0.001:
    print("verdict: strong evidence for non-isotropic layered structure")
elif p_value < 0.01:
    print("verdict: significant deviation from isotropy")
elif p_value < 0.05:
    print("verdict: marginal evidence for structure")
else:
    print("verdict: no significant deviation from isotropy in these bands")

print("\n" + "="*70)
print("analysis complete")
print("="*70)
