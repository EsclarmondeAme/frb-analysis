#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

# ==========================================================
# CONFIG
# ==========================================================
UNIFIED_L = 159.85
UNIFIED_B = -0.51
BREAK_ANGLE = 25.0    # degrees

# ==========================================================
# ANGULAR DISTANCE (stable, vectorized)
# ==========================================================
def angdist(l1, b1, l2, b2):
    l1 = np.deg2rad(l1)
    b1 = np.deg2rad(b1)
    l2 = np.deg2rad(l2)
    b2 = np.deg2rad(b2)
    return np.rad2deg(
        np.arccos(
            np.sin(b1)*np.sin(b2) +
            np.cos(b1)*np.cos(b2)*np.cos(l1 - l2)
        )
    )

# ==========================================================
# LOAD
# ==========================================================
df = pd.read_csv("frbs.csv")

# ==========================================================
# COMPUTE ANGULAR DISTANCE FROM UNIFIED AXIS
# ==========================================================
theta = angdist(df["ra"], df["dec"], UNIFIED_L, UNIFIED_B)
df["theta"] = theta

inner = df[df["theta"] < BREAK_ANGLE]
outer = df[df["theta"] >= BREAK_ANGLE]

# ==========================================================
# PARAMETERS
# ==========================================================
dm_i, dm_o = inner["dm"].values, outer["dm"].values
fl_i, fl_o = inner["fluence"].values, outer["fluence"].values
wd_i, wd_o = inner["width"].values, outer["width"].values
sn_i, sn_o = inner["snr"].values, outer["snr"].values

energy_i = fl_i * dm_i**2
energy_o = fl_o * dm_o**2

spec_i = fl_i / np.maximum(wd_i, 1e-6)
spec_o = fl_o / np.maximum(wd_o, 1e-6)

sn_norm_i = sn_i / np.mean(sn_i) if len(sn_i) else sn_i
sn_norm_o = sn_o / np.mean(sn_o) if len(sn_o) else sn_o

PARAMS = {
    "DM": (dm_i, dm_o),
    "Fluence": (fl_i, fl_o),
    "Width": (wd_i, wd_o),
    "SNR": (sn_i, sn_o),
    "Energy": (energy_i, energy_o),
    "Fluence/Width": (spec_i, spec_o),
    "SNR_normalized": (sn_norm_i, sn_norm_o),
}

# ==========================================================
# PERMUTATION TEST (no tqdm)
# ==========================================================
def perm_test(x, y, N=3000):
    obs = np.abs(np.mean(x) - np.mean(y))
    all_vals = np.concatenate([x, y])
    n_x = len(x)
    count = 0
    for _ in range(N):
        np.random.shuffle(all_vals)
        x_p = all_vals[:n_x]
        y_p = all_vals[n_x:]
        if np.abs(np.mean(x_p) - np.mean(y_p)) >= obs:
            count += 1
    return count / N

# ==========================================================
# PRINT HEADER
# ==========================================================
print("=====================================================================")
print("FRB RADIAL BREAK POPULATION TEST")
print("=====================================================================")
print(f"break angle: {BREAK_ANGLE:.2f}°")
print(f"inner samples: {len(inner)}, outer samples: {len(outer)}")
print("---------------------------------------------------------------------")

# ==========================================================
# ANALYSIS FOR EACH PARAMETER
# ==========================================================
results = []

for name, (x, y) in PARAMS.items():
    if len(x) < 3 or len(y) < 3:
        results.append((name, np.nan, np.nan, np.nan))
        continue

    ks_p = ks_2samp(x, y, alternative="two-sided").pvalue
    mw_p = mannwhitneyu(x, y, alternative="two-sided").pvalue
    perm_p = perm_test(x, y, N=3000)

    results.append((name, ks_p, mw_p, perm_p))

# ==========================================================
# PRINT RESULTS
# ==========================================================
print("PARAMETER COMPARISON (inner<25° vs outer≥25°)")
print("---------------------------------------------------------------------")
print(f"{'parameter':20s} {'KS-p':>12s} {'MWU-p':>12s} {'perm-p':>12s}")
print("---------------------------------------------------------------------")

for name, ks_p, mw_p, perm_p in results:
    print(f"{name:20s} {ks_p:12.4f} {mw_p:12.4f} {perm_p:12.4f}")

print("---------------------------------------------------------------------")

# ==========================================================
# SCIENTIFIC VERDICT
# ==========================================================
print("=====================================================================")
print("SCIENTIFIC VERDICT")
print("=====================================================================")

significant = [name for name, ks, mw, perm in results if perm < 0.05]

if len(significant) == 0:
    print("no parameter shows significant inner–outer change at the 25° break.")
    print("→ break is geometric rather than physical.")
else:
    print("parameters showing significant population changes:")
    for s in significant:
        print(" →", s)

print("=====================================================================")
print("analysis complete.")
print("=====================================================================")
