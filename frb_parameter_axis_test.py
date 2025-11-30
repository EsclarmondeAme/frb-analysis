import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import stats

print("=" * 69)
print("FRB PARAMETER–AXIS CORRELATION TEST")
print("testing fluence, snr, width, dm vs angular distance from unified axis")
print("=" * 69)

# ------------------------------------------------------------
# 1. unified axis
# ------------------------------------------------------------
unified_l = 159.85
unified_b = -0.51
unified_axis = SkyCoord(l=unified_l*u.deg, b=unified_b*u.deg, frame="galactic")

print("\nSECTION 1 — UNIFIED AXIS")
print("------------------------------------------------------------")
print(f"axis: l = {unified_l:.2f}°,  b = {unified_b:.2f}°")

# ------------------------------------------------------------
# 2. load FRB catalog
# ------------------------------------------------------------
print("\nSECTION 2 — DATASET")
print("------------------------------------------------------------")

try:
    frb = pd.read_csv("frbs.csv")
except FileNotFoundError:
    print("ERROR: frbs.csv not found")
    raise

print(f"loaded {len(frb)} FRBs")

coords = SkyCoord(ra=frb["ra"].values*u.deg,
                  dec=frb["dec"].values*u.deg,
                  frame="icrs").galactic

theta = coords.separation(unified_axis).deg
frb["theta"] = theta

print(f"θ range: {theta.min():.2f}° – {theta.max():.2f}°")

# parameters to test
params = ["fluence", "snr", "width", "dm"]

# ------------------------------------------------------------
# 3. correlation tests per parameter
# ------------------------------------------------------------
print("\nSECTION 3 — PARAMETER CORRELATION")
print("------------------------------------------------------------")

results = {}

for p in params:
    x = theta
    y = frb[p].values.astype(float)

    # Pearson
    pear_r, pear_p = stats.pearsonr(x, y)
    # Spearman
    spear_r, spear_p = stats.spearmanr(x, y)

    # 3-bin comparison
    edges = np.percentile(theta, [0, 33.3, 66.6, 100])
    frb["bin"] = pd.cut(theta, bins=edges, labels=["low", "mid", "high"], include_lowest=True)

    groups = [frb[frb["bin"]==lab][p].values for lab in ["low", "mid", "high"]]
    F, ap = stats.f_oneway(*groups)

    # permutation test
    n_perm = 10000
    obs = abs(pear_r)
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(y)
        r, _ = stats.pearsonr(x, perm)
        if abs(r) >= obs:
            count += 1
    p_perm = count / n_perm

    results[p] = {
        "pear_r": pear_r,
        "pear_p": pear_p,
        "spear_r": spear_r,
        "spear_p": spear_p,
        "anova_p": ap,
        "perm_p": p_perm
    }

    print(f"\n--- {p.upper()} ---")
    print(f"pearson r = {pear_r:.4f}, p = {pear_p:.4f}")
    print(f"spearman r = {spear_r:.4f}, p = {spear_p:.4f}")
    print(f"anova p    = {ap:.4f}")
    print(f"perm p     = {p_perm:.4f}")


# ------------------------------------------------------------
# 4. figure for each parameter
# ------------------------------------------------------------
print("\nSECTION 4 — FIGURES")
print("------------------------------------------------------------")

for p in params:
    plt.figure(figsize=(7,5))
    plt.scatter(theta, frb[p], s=12, alpha=0.5, edgecolor="none")
    plt.xlabel("angular distance from unified axis θ (deg)")
    plt.ylabel(p)
    plt.title(f"{p} vs θ")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"param_{p}_axis.png", dpi=200)
    print(f"saved: param_{p}_axis.png")


# ------------------------------------------------------------
# 5. final verdict
# ------------------------------------------------------------
print("\nSECTION 5 — FINAL VERDICT")
print("------------------------------------------------------------")

significant = []

for p in params:
    r = results[p]
    if (r["pear_p"] < 0.05 or
        r["spear_p"] < 0.05 or
        r["anova_p"] < 0.05 or
        r["perm_p"] < 0.05):
        significant.append(p)

print("\nparameters showing significant axis correlation:")
if significant:
    print("→ " + ", ".join(significant))
else:
    print("→ none detected")

print("\nanalysis complete")
print("="*69)
