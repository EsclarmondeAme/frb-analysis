# frb_frequency_layer_test.py
# -------------------------------------------------------------------
# testing whether FRB spectral content (using proxies) varies with
# angular distance from the unified axis, and whether the variation
# aligns with cone radii (10°, 21°, 40°)
# -------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import f_oneway

print("="*70)
print("FRB FREQUENCY–LAYER STRUCTURE TEST")
print("Testing spectral proxies vs cone radii")
print("="*70)

# -------------------------------------------------------------------
# 1 — load dataset
# -------------------------------------------------------------------
df = pd.read_csv("frbs.csv")
ra = df["ra"].values
dec = df["dec"].values

snr = df["snr"].values
width = df["width"].values
fluence = df["fluence"].values

coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs").galactic

# unified axis (cosmic alignment)
unif_l = 159.85
unif_b = -0.51
axis = SkyCoord(l=unif_l*u.deg, b=unif_b*u.deg, frame="galactic")

theta = coords.separation(axis).deg

print("\nSECTION 1 — DATA")
print("------------------------------------------------------------")
print(f"Total FRBs: {len(df)}")
print(f"Theta range: {theta.min():.2f}° – {theta.max():.2f}°")


# -------------------------------------------------------------------
# 2 — cone radii from spatial fits
# -------------------------------------------------------------------
cone_radii = np.array([10.0, 21.0, 40.0])
bins = np.concatenate([[0], cone_radii, [140]])
labels = ["inner", "mid1", "mid2", "outer"]

print("\nSECTION 2 — CONE RADII")
print("------------------------------------------------------------")
print("Cone radii (deg):", cone_radii)


# -------------------------------------------------------------------
# 3 — define spectral proxies
# -------------------------------------------------------------------
# proxy 1: snr / width  (narrow bursts give artificially higher snr → resembles band-limited structure)
spec1 = snr / np.clip(width, 1e-6, None)

# proxy 2: fluence / width  (energy density proxy)
spec2 = fluence / np.clip(width, 1e-6, None)

# proxy 3: normalized snr (independent check)
spec3 = snr / np.median(snr)

spec_list = [
    ("snr/width", spec1),
    ("fluence/width", spec2),
    ("snr_normalized", spec3)
]


# -------------------------------------------------------------------
# 4 — compute mean spectral values per cone layer
# -------------------------------------------------------------------
print("\nSECTION 3 — MEAN SPECTRAL PROXIES BY CONE LAYER")
print("------------------------------------------------------------")

layer_vals = {name: [] for name, _ in spec_list}
layer_counts = []

for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (theta >= lo) & (theta < hi)
    layer_counts.append(np.sum(mask))

    print(f"\nLayer {labels[i]}  {lo:5.1f}°–{hi:5.1f}°  (n={np.sum(mask)})")

    for name, spec in spec_list:
        vals = spec[mask]
        mean = np.mean(vals)
        std = np.std(vals)
        layer_vals[name].append(mean)
        print(f"   {name:14s}  mean={mean:.4f}, std={std:.4f}")


# -------------------------------------------------------------------
# 5 — ANOVA for each spectral proxy across layers
# -------------------------------------------------------------------
print("\nSECTION 4 — ANOVA ACROSS LAYERS")
print("------------------------------------------------------------")

anova_results = {}

for name, spec in spec_list:
    groups = [
        spec[(theta >= bins[i]) & (theta < bins[i+1])]
        for i in range(len(bins)-1)
    ]
    F, p = f_oneway(*groups)
    anova_results[name] = p
    print(f"{name:14s}  F={F:.4f}, p={p:.4f}")


# -------------------------------------------------------------------
# 6 — Monte Carlo null: shuffle spectral values
# -------------------------------------------------------------------
print("\nSECTION 5 — MONTE CARLO NULL (10,000 sims)")
print("------------------------------------------------------------")

n_sim = 10000
mc_p = {}

for name, spec in spec_list:
    F_real, _ = f_oneway(
        spec[(theta < bins[1])],
        spec[(theta >= bins[1]) & (theta < bins[2])],
        spec[(theta >= bins[2]) & (theta < bins[3])],
        spec[(theta >= bins[3])]
    )
    
    F_null = np.zeros(n_sim)
    for i in range(n_sim):
        shuffled = np.random.permutation(spec)
        F_null[i], _ = f_oneway(
            shuffled[(theta < bins[1])],
            shuffled[(theta >= bins[1]) & (theta < bins[2])],
            shuffled[(theta >= bins[2]) & (theta < bins[3])],
            shuffled[(theta >= bins[3])]
        )
    
    p_mc = np.mean(F_null >= F_real)
    mc_p[name] = p_mc
    print(f"{name:14s}  Monte Carlo p = {p_mc:.4f}")


# -------------------------------------------------------------------
# 7 — Scientific verdict
# -------------------------------------------------------------------
print("\nSECTION 6 — VERDICT")
print("------------------------------------------------------------")

significant = []

for name in layer_vals.keys():
    if mc_p[name] < 0.05:
        significant.append(name)

if len(significant) == 0:
    print("→ No spectral proxy shows significant alignment with cone layers.")
elif len(significant) == 1:
    print(f"→ One spectral proxy aligns with cone layers: {significant[0]}")
else:
    print("→ Multiple proxies show significant alignment:")
    for s in significant:
        print(f"   - {s}")

print("\nAnalysis complete.")
print("="*70)


# -------------------------------------------------------------------
# 8 — Plot (spectral proxy vs cone layers)
# -------------------------------------------------------------------
plt.figure(figsize=(10,6))

x = (bins[:-1] + bins[1:]) / 2

for name, vals in layer_vals.items():
    plt.plot(x, vals, marker="o", label=name)

plt.xlabel("Angular distance from axis (deg)")
plt.ylabel("Mean spectral proxy")
plt.title("Spectral proxies vs cone layers")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("frequency_layer_test.png", dpi=200)

print("Saved: frequency_layer_test.png")
