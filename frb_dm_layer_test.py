# frb_dm_layer_test.py
# -------------------------------------------------------------------
# testing whether FRB dispersion measure (DM) varies systematically
# with angular distance from the unified axis, and whether any trend
# aligns with the cone radii (10°, 21°, 40°)
# -------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import f_oneway

print("="*70)
print("FRB DM–LAYER STRUCTURE TEST")
print("Testing DM vs cone radii")
print("="*70)

# -------------------------------------------------------------------
# 1 — load dataset
# -------------------------------------------------------------------
df = pd.read_csv("frbs.csv")

ra = df["ra"].values
dec = df["dec"].values
dm = df["dm"].values

coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs").galactic

# unified cosmic axis
unif_l = 159.85
unif_b = -0.51
axis = SkyCoord(l=unif_l*u.deg, b=unif_b*u.deg, frame="galactic")

theta = coords.separation(axis).deg

print("\nSECTION 1 — DATA")
print("------------------------------------------------------------")
print(f"Total FRBs: {len(df)}")
print(f"Theta range: {theta.min():.2f}° – {theta.max():.2f}°")

# -------------------------------------------------------------------
# 2 — cone radii
# -------------------------------------------------------------------
cone_radii = np.array([10.0, 21.0, 40.0])
bins = np.concatenate([[0], cone_radii, [140]])
labels = ["inner", "mid1", "mid2", "outer"]

print("\nSECTION 2 — CONE RADII")
print("------------------------------------------------------------")
print("Cone boundaries (deg):", cone_radii)

# -------------------------------------------------------------------
# 3 — DM by cone layers
# -------------------------------------------------------------------
print("\nSECTION 3 — MEAN DM BY CONE LAYER")
print("------------------------------------------------------------")

dm_means = []
dm_stds = []
dm_ns = []

for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (theta >= lo) & (theta < hi)
    vals = dm[mask]

    dm_means.append(np.mean(vals))
    dm_stds.append(np.std(vals))
    dm_ns.append(len(vals))

    print(f"{labels[i]:7s} {lo:5.1f}°–{hi:5.1f}°: n={len(vals):3d}, "
          f"mean={np.mean(vals):.2f}, std={np.std(vals):.2f}")

dm_means = np.array(dm_means)
dm_stds = np.array(dm_stds)
dm_ns = np.array(dm_ns)

# -------------------------------------------------------------------
# 4 — ANOVA across layers
# -------------------------------------------------------------------
print("\nSECTION 4 — ANOVA ACROSS LAYERS")
print("------------------------------------------------------------")

groups = [
    dm[(theta >= bins[i]) & (theta < bins[i+1])]
    for i in range(len(bins)-1)
]

F, p_anova = f_oneway(*groups)

print(f"F-statistic: {F:.4f}")
print(f"p-value:     {p_anova:.4f}")

# -------------------------------------------------------------------
# 5 — Monte Carlo null: shuffle DM values 10,000 times
# -------------------------------------------------------------------
print("\nSECTION 5 — MONTE CARLO NULL (10,000 sims)")
print("------------------------------------------------------------")

n_sim = 10000
F_null = np.zeros(n_sim)

for i in range(n_sim):
    shuffled_dm = np.random.permutation(dm)
    F_null[i], _ = f_oneway(
        shuffled_dm[(theta < bins[1])],
        shuffled_dm[(theta >= bins[1]) & (theta < bins[2])],
        shuffled_dm[(theta >= bins[2]) & (theta < bins[3])],
        shuffled_dm[(theta >= bins[3])]
    )

p_mc = np.mean(F_null >= F)

print(f"Monte Carlo p-value: {p_mc:.4f}")
print(f"Null mean F: {np.mean(F_null):.3f}")
print(f"Null 95% F:  {np.percentile(F_null,95):.3f}")

# -------------------------------------------------------------------
# 6 — scientific verdict
# -------------------------------------------------------------------
print("\nSECTION 6 — VERDICT")
print("------------------------------------------------------------")

if p_mc < 0.01:
    print("→ Strong evidence for DM layering aligned with cone radii.")
elif p_mc < 0.05:
    print("→ Moderate evidence for DM–cone alignment.")
else:
    print("→ No significant DM layering relative to cone geometry.")

print("\nAnalysis complete.")
print("="*70)

# -------------------------------------------------------------------
# 7 — Plot
# -------------------------------------------------------------------
plt.figure(figsize=(10,6))

centers = (bins[:-1] + bins[1:]) / 2

plt.errorbar(centers, dm_means, yerr=dm_stds/np.sqrt(dm_ns),
             fmt='o', markersize=10, capsize=4, label="DM mean ± stderr")

plt.vlines(cone_radii, ymin=min(dm_means)*0.8, ymax=max(dm_means)*1.2,
           colors='r', linestyles='--', label="Cone radii")

plt.xlabel("Angular distance from axis (deg)")
plt.ylabel("Mean DM")
plt.title("FRB DM vs Cone Layer Structure")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("dm_layer_test.png", dpi=200)

print("Saved: dm_layer_test.png")
