import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import stats

print("=" * 69)
print("FRB SPECTRAL–AXIS CORRELATION TEST")
print("spectral hardness vs angular distance from unified axis")
print("=" * 69)

# ------------------------------------------------------------
# 1. unified axis (from best-fit earlier)
# ------------------------------------------------------------
unified_l = 159.85
unified_b = -0.51
unified_axis = SkyCoord(l=unified_l*u.deg, b=unified_b*u.deg, frame="galactic")

print("\nSECTION 1 — UNIFIED AXIS")
print("------------------------------------------------------------")
print(f"axis: l = {unified_l:.2f}°,  b = {unified_b:.2f}°")

# ------------------------------------------------------------
# 2. load FRB catalog
# must contain: ra, dec, fluence_low, fluence_high
# ------------------------------------------------------------
print("\nSECTION 2 — DATASET")
print("------------------------------------------------------------")

try:
    frb = pd.read_csv("frbs.csv")
except FileNotFoundError:
    print("ERROR: frbs.csv not found")
    raise

print(f"loaded {len(frb)} FRBs")

required = ["ra", "dec", "fluence_low", "fluence_high"]
missing = [c for c in required if c not in frb.columns]

if missing:
    print(f"ERROR: missing required columns: {missing}")
    raise SystemExit

coords = SkyCoord(ra=frb["ra"].values*u.deg,
                  dec=frb["dec"].values*u.deg,
                  frame="icrs").galactic

# angular separation from unified axis
theta = coords.separation(unified_axis).deg
frb["theta"] = theta

print(f"θ range: {theta.min():.2f}° – {theta.max():.2f}°")

# ------------------------------------------------------------
# 3. spectral hardness H
# ------------------------------------------------------------
print("\nSECTION 3 — SPECTRAL HARDNESS")
print("------------------------------------------------------------")

f_low = frb["fluence_low"].values.astype(float)
f_high = frb["fluence_high"].values.astype(float)

# protect against division by zero
mask = (f_low + f_high) > 0
H = np.zeros_like(f_low, dtype=float)
H[mask] = (f_high[mask] - f_low[mask]) / (f_high[mask] + f_low[mask])

frb["H"] = H

print(f"H min = {H.min():.3f}, max = {H.max():.3f}")

# ------------------------------------------------------------
# 4. correlation tests
# ------------------------------------------------------------
print("\nSECTION 4 — CORRELATION TESTS")
print("------------------------------------------------------------")

pear_r, pear_p = stats.pearsonr(theta, H)
spear_r, spear_p = stats.spearmanr(theta, H)

print(f"pearson r = {pear_r:.4f}   p = {pear_p:.4f}")
print(f"spearman r = {spear_r:.4f} p = {spear_p:.4f}")

# ------------------------------------------------------------
# 5. 3-bin comparison (low/mid/high θ)
# ------------------------------------------------------------
print("\nSECTION 5 — BINNED COMPARISON")
print("------------------------------------------------------------")

edges = np.percentile(theta, [0, 33.3, 66.6, 100])
labels = ["low-θ", "mid-θ", "high-θ"]
frb["bin"] = pd.cut(theta, bins=edges, labels=labels, include_lowest=True)

for lab in labels:
    m = frb[frb["bin"] == lab]["H"].mean()
    print(f"{lab:7s} mean H = {m:.4f}")

# anova for difference between bins
groups = [frb[frb["bin"] == lab]["H"].values for lab in labels]
F, ap = stats.f_oneway(*groups)
print(f"\nANOVA F = {F:.3f}, p = {ap:.4f}")

# ------------------------------------------------------------
# 6. permutation significance (10k shuffles)
# ------------------------------------------------------------
print("\nSECTION 6 — PERMUTATION TEST")
print("------------------------------------------------------------")

n_perm = 10000
count = 0
obs = abs(pear_r)

for _ in range(n_perm):
    perm = np.random.permutation(H)
    r, _ = stats.pearsonr(theta, perm)
    if abs(r) >= obs:
        count += 1

p_perm = count / n_perm
print(f"permutation p-value: {p_perm:.4f}")

# ------------------------------------------------------------
# 7. figure
# ------------------------------------------------------------
print("\nSECTION 7 — FIGURE")
print("------------------------------------------------------------")

plt.figure(figsize=(8,6))
plt.scatter(theta, H, s=12, alpha=0.5, edgecolor="none")
plt.xlabel("angular distance from unified axis θ (deg)")
plt.ylabel("spectral hardness H")
plt.title("FRB spectral hardness vs θ (unified axis)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("frb_spectral_axis.png", dpi=200)
print("saved: frb_spectral_axis.png")

# ------------------------------------------------------------
# 8. verdict
# ------------------------------------------------------------
print("\nSECTION 8 — VERDICT")
print("------------------------------------------------------------")

sig = (pear_p < 0.05) or (spear_p < 0.05) or (ap < 0.05) or (p_perm < 0.05)

print(f"pearson p = {pear_p:.4f}")
print(f"spearman p = {spear_p:.4f}")
print(f"anova   p = {ap:.4f}")
print(f"perm    p = {p_perm:.4f}")

if sig:
    print("\n→ evidence for spectral–axis correlation detected")
else:
    print("\n→ no statistically significant spectral–axis correlation detected")

print("\nanalysis complete")
print("="*69)
