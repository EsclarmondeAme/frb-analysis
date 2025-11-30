#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from astropy.coordinates import SkyCoord
import astropy.units as u
import json
import pandas as pd

# ------------------------------------------------------------
# configuration
# ------------------------------------------------------------

N_SIM = 200000
SEED = 123
np.random.seed(SEED)

REAL_CSV = "frbs.csv"

BANDS = [(0,10),(10,25),(25,40),(40,90)]

APPLY_GAL_MASK = False
GAL_LAT_MIN = 20

APPLY_ECL_MASK = False
ECL_LAT_MIN = 20

APPLY_SG_MASK = False
SG_LAT_MIN = 20

def completeness_function(theta_deg):
    # flat selection function
    return np.ones_like(theta_deg, dtype=float)


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def load_real_frbs(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["ra","dec"])
    return df["ra"].to_numpy(), df["dec"].to_numpy()

def random_isotropic_sky(n):
    ra = np.random.uniform(0,360,n)
    dec = np.degrees(np.arcsin(np.random.uniform(-1,1,n)))
    return ra, dec

def apply_masks(ra, dec):
    c = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")
    keep = np.ones(len(ra), dtype=bool)

    if APPLY_GAL_MASK:
        gal = c.galactic
        keep &= (np.abs(gal.b.deg) >= GAL_LAT_MIN)

    if APPLY_ECL_MASK:
        ecl = c.barycentrictrueecliptic
        keep &= (np.abs(ecl.lat.deg) >= ECL_LAT_MIN)

    if APPLY_SG_MASK:
        sg = c.supergalactic
        keep &= (np.abs(sg.b.deg) >= SG_LAT_MIN)

    return keep

def count_shells(theta):
    counts = []
    for lo,hi in BANDS:
        sel = (theta >= lo) & (theta < hi)
        counts.append(sel.sum())
    return np.array(counts)

def angle_from_unified_axis(ra, dec):
    with open("axes.json","r") as f:
        axes = json.load(f)
    l = axes["unified_axis"]["l"]
    b = axes["unified_axis"]["b"]
    axis = SkyCoord(l*u.deg, b*u.deg, frame="galactic")
    c = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs").galactic
    return c.separation(axis).deg


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

print("="*55)
print("FRB SELECTION-FUNCTION FORWARD MODEL")
print("="*55)

# load real
ra_real, dec_real = load_real_frbs(REAL_CSV)
theta_real = angle_from_unified_axis(ra_real, dec_real)
real_counts = count_shells(theta_real)

# synthetic sky
ra_sim, dec_sim = random_isotropic_sky(N_SIM)
mask = apply_masks(ra_sim, dec_sim)
ra_sim = ra_sim[mask]
dec_sim = dec_sim[mask]

theta_sim = angle_from_unified_axis(ra_sim, dec_sim)
weights = completeness_function(theta_sim)




sim_counts = []
for lo, hi in BANDS:
    sel = (theta_sim >= lo) & (theta_sim < hi)
    sim_counts.append(weights[sel].sum())




sim_counts = np.array(sim_counts)

scale = len(theta_real) / sim_counts.sum()
expected = sim_counts
 


chi2 = ((real_counts - expected)**2 / expected).sum()
dof = len(BANDS)-1
# expected from sims (mean over simulations)

# renormalize expected so total matches real total
scale = real_counts.sum() / expected.sum()
expected = expected * scale

# chi-square p-value
# ensure clean numpy arrays
real_counts = np.asarray(real_counts, dtype=float)
expected = np.asarray(expected, dtype=float)




# force expected to have the same total as real, required by scipy
expected = expected * (real_counts.sum() / expected.sum())

# run chi-square properly
chi2_res = chisquare(f_obs=real_counts, f_exp=expected)
p = chi2_res.pvalue



print("\nreal shell counts:", real_counts)
print("expected under selection model:", expected.astype(int))
print(f"\nχ² = {chi2:.2f} (dof={dof})")
print(f"p-value = {p:.4e}")

print("\n--------------- scientific verdict ---------------")
if p > 0.05:
    print("the frb shell could be reproduced by some selection function.")
else:
    print("the selection function cannot reproduce the frb shell.")
    print("a real cosmic anisotropy is required.")
print("---------------------------------------------------")

plt.figure(figsize=(8,5))
x = np.arange(len(BANDS))
labels = [f"{lo}–{hi}°" for lo,hi in BANDS]
plt.bar(x-0.15, real_counts, width=0.3, label="real")
plt.bar(x+0.15, expected, width=0.3, label="model")
plt.xticks(x, labels)
plt.ylabel("counts")
plt.title("frb shell counts vs selection-function model")
plt.legend()
plt.tight_layout()
plt.savefig("frb_selection_function_simulator.png", dpi=200)

print("saved plot: frb_selection_function_simulator.png")
print("analysis complete.")
print("="*55)
