# frb_energy_layer_test.py
# -------------------------------------------------------------------
# Tests whether FRB intrinsic energy proxy (fluence × DM^2)
# shows layered structure relative to the unified axis and the
# cone radii (10°, 21°, 40°).
# -------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import f_oneway

print("="*70)
print("FRB ENERGY–LAYER STRUCTURE TEST")
print("Testing E_iso ∝ fluence × DM^2 vs cone radii")
print("="*70)

# -------------------------------------------------------------------
# 1 — load dataset
# -------------------------------------------------------------------
df = pd.read_csv("frbs.csv")

ra = df["ra"].values
dec = df["dec"].values
dm = df["dm"].values
fluence = df["fluence"].values

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
# 2 — energy proxy
# -------------------------------------------------------------------
# standard FRB isotropic-equivalent energy scaling:
# E_iso ∝ fluence × DM^2
energy = fluence * (dm**2)

print("\nSECTION 2 — ENERGY PROXY")
print("------------------------------------------------------------")
print("Energy proxy: E = fluence × DM^2")
print(f"Energy range: {energy.min():.3e} – {energy.max():.3e}")

# -------------------------------------------------------------------
# 3 — cone radii
# -------------------------------------------------------------------
cone_radii = np.array([10.0, 21.0, 40.0])
bins = np.concatenate([[0], cone_radii, [140]])
labels = ["inner", "mid1", "mid2", "outer"]

print("\nSECTION 3 — CONE RADII")
print("------------------------------------------------------------")
print("Cone boundaries (deg):", cone_radii)

# -------------------------------------------------------------------
# 4 — mean energy by cone layer
# -------------------------------------------------------------------
print("\nSECTION 4 — MEAN ENERGY BY CONE LAYER")
print("------------------------------------------------------------")

E_means = []
E_stds = []
E_ns = []

for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (theta >= lo) & (theta < hi)
    vals = energy[mask]

    E_means.append(np.mean(vals))
    E_stds.append(np.std(vals))
    E_ns.append(len(vals))

    print(f"{labels[i]:7s} {lo:5.1f}°–{hi:5.1f}°:  n={len(vals):3d}, "
          f"mean={np.mean(vals):.3e},  std={np.std(vals):.3e}")

E_means = np.array(E_means)
E_stds = np.array(E_stds)
E_ns = np.array(E_ns)

# -------------------------------------------------------------------
# 5 — ANOVA across layers
# -------------------------------------------------------------------
print("\nSECTION 5 — ANOVA ACROSS LAYERS")
print("------------------------------------------------------------")

groups = [
    energy[(theta >= bins[i]) & (theta < bins[i+1])]
    for i in range(len(bins)-1)
]

F, p_anova = f_oneway(*groups)

print(f"F-statistic: {F:.4f}")
print(f"p-value:     {p_anova:.4f}")

# -------------------------------------------------------------------
# 6 — Monte Carlo null: shuffle energy values
# -------------------------------------------------------------------
print("\nSECTION 6 — MONTE CARLO NULL (10,000 sims)")
print("------------------------------------------------------------")

n_sim = 10000
F_null = np.zeros(n_sim)

for i in range(n_sim):
    shuffled = np.random.permutation(energy)
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
# 7 — scientific verdict
# -------------------------------------------------------------------
print("\nSECTION 7 — VERDICT")
print("------------------------------------------------------------")

if p_mc < 0.01:
    print("→ Strong evidence for layered energy structure aligned with cone radii.")
elif p_mc < 0.05:
    print("→ Moderate evidence for energy–cone alignment.")
else:
    print("→ No significant energy layering relative to cone geometry.")

print("\nAnalysis complete.")
print("="*70)

# -------------------------------------------------------------------
# 8 — plot
# -------------------------------------------------------------------
plt.figure(figsize=(10,6))

centers = (bins[:-1] + bins[1:]) / 2

plt.errorbar(centers, E_means, yerr=E_stds/np.sqrt(E_ns),
             fmt="o", markersize=10, capsize=4, label="Energy mean ± stderr")

plt.vlines(cone_radii, ymin=min(E_means)*0.8, ymax=max(E_means)*1.2,
           colors="r", linestyles="--", label="Cone radii")

plt.xlabel("Angular distance from axis (deg)")
plt.ylabel("Energy proxy (fluence × DM²)")
plt.title("FRB Energy vs Cone Layer Structure")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("energy_layer_test.png", dpi=200)

print("Saved: energy_layer_test.png")
