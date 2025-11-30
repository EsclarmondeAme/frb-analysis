# frb_width_cone_alignment.py
# -------------------------------------------------------------------
# testing whether the FRB width(θ) structure matches the cone-layer
# radii inferred from spatial clustering (10°, 21°, 40°)
# -------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import f_oneway

print("="*70)
print("FRB WIDTH–CONE LAYER ALIGNMENT TEST")
print("Testing if width(θ) layers correspond to cone radii")
print("="*70)

# -------------------------------------------------------------------
# SECTION 1 — load FRBs
# -------------------------------------------------------------------
df = pd.read_csv("frbs.csv")
ra = df["ra"].values
dec = df["dec"].values
width = df["width"].values

coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs").galactic

# unified axis (best-fit cosmic axis)
unif_l = 159.85
unif_b = -0.51
axis = SkyCoord(l=unif_l*u.deg, b=unif_b*u.deg, frame="galactic")

theta = coords.separation(axis).deg

print("\nSECTION 1 — DATA")
print("------------------------------------------------------------")
print(f"Total FRBs: {len(df)}")
print(f"Theta range: {theta.min():.2f}° – {theta.max():.2f}°")

# -------------------------------------------------------------------
# SECTION 2 — defining cone radii from spatial fits
# -------------------------------------------------------------------
cone_radii = np.array([10.0, 21.0, 40.0])
print("\nSECTION 2 — CONE RADII (FROM SPATIAL FITS)")
print("------------------------------------------------------------")
print("Cone layer boundaries (deg):", cone_radii)

# -------------------------------------------------------------------
# SECTION 3 — computing width means in cone-based bins
# -------------------------------------------------------------------
bins = np.concatenate(([0.0], cone_radii, [140.0]))
labels = ["inner", "mid1", "mid2", "outer"]

means = []
stds = []
ns = []

print("\nSECTION 3 — WIDTH BY CONE LAYERS")
print("------------------------------------------------------------")

for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (theta >= lo) & (theta < hi)
    vals = width[mask]
    means.append(np.mean(vals))
    stds.append(np.std(vals))
    ns.append(len(vals))

    print(f"{labels[i]:7s} {lo:5.1f}°–{hi:5.1f}°:  n={len(vals):3d},  "
          f"mean={np.mean(vals):.4f},  std={np.std(vals):.4f}")

# convert to arrays
means = np.array(means)
stds = np.array(stds)
ns = np.array(ns)

# -------------------------------------------------------------------
# SECTION 4 — does variance between layers exceed random?
# -------------------------------------------------------------------
F, p_anova = f_oneway(
    width[(theta < bins[1])],
    width[(theta >= bins[1]) & (theta < bins[2])],
    width[(theta >= bins[2]) & (theta < bins[3])],
    width[(theta >= bins[3])]
)

print("\nSECTION 4 — ANOVA TEST ACROSS CONE LAYERS")
print("------------------------------------------------------------")
print(f"F-statistic: {F:.4f}")
print(f"p-value:     {p_anova:.4f}")

# -------------------------------------------------------------------
# SECTION 5 — monte-carlo test: random θ reshuffling
# -------------------------------------------------------------------
n_sim = 10000
F_null = np.zeros(n_sim)

print("\nSECTION 5 — MONTE CARLO NULL (10,000 sims)")
print("------------------------------------------------------------")

for i in range(n_sim):
    shuffled = np.random.permutation(width)
    F_null[i], _ = f_oneway(
        shuffled[(theta < bins[1])],
        shuffled[(theta >= bins[1]) & (theta < bins[2])],
        shuffled[(theta >= bins[2]) & (theta < bins[3])],
        shuffled[(theta >= bins[3])]
    )

p_mc = np.mean(F_null >= F)

print(f"Monte Carlo p-value: {p_mc:.4f}")
print(f"Null mean F: {np.mean(F_null):.3f}")
print(f"Null 95% F:  {np.percentile(F_null,95):.3f}")

# -------------------------------------------------------------------
# SECTION 6 — scientific verdict
# -------------------------------------------------------------------
print("\nSECTION 6 — VERDICT")
print("------------------------------------------------------------")

if p_mc < 0.01:
    print("→ Strong evidence that width layers align with cone radii.")
elif p_mc < 0.05:
    print("→ Moderate evidence for alignment.")
else:
    print("→ No strong evidence for width–cone alignment.")

print("\nAnalysis complete.")
print("="*70)

# -------------------------------------------------------------------
# SECTION 7 — figure
# -------------------------------------------------------------------
plt.figure(figsize=(10,6))
centers = (bins[:-1] + bins[1:]) / 2

plt.errorbar(centers, means, yerr=stds/np.sqrt(ns),
             fmt='o', markersize=10, capsize=4, label="Width mean ± stderr")
plt.vlines(cone_radii, ymin=min(means)*0.8, ymax=max(means)*1.2,
           colors='r', linestyles='--', label='Cone radii')

plt.xlabel("Angular distance from axis (deg)")
plt.ylabel("Mean width")
plt.title("FRB Width vs Cone Layer Structure")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("width_cone_alignment.png", dpi=200)

print("Saved: width_cone_alignment.png")
