"""
frb_energy_axis_test.py
----------------------------------------------------------
tests whether higher-energy frbs align more strongly
with the unified best-fit cosmic axis

best-fit axis used:
    l = 159.85°, b = -0.51°
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


# ----------------------------------------------------------
# 0. unified axis
# ----------------------------------------------------------

best_l = 159.85
best_b = -0.51

best_axis = SkyCoord(l=best_l*u.deg, b=best_b*u.deg, frame="galactic")

print("=" * 70)
print("frb energy-dependent axis alignment test")
print("=" * 70)

print("\n1. unified best-fit axis (galactic)")
print(f"   l = {best_l:7.2f}°")
print(f"   b = {best_b:7.2f}°")

# ----------------------------------------------------------
# 1. load data
# ----------------------------------------------------------

try:
    frbs = pd.read_csv("frbs.csv")
except:
    print("\nerror: frbs.csv not found")
    exit()

print("\n2. frb dataset")
print(f"   total frbs loaded: {len(frbs)}")

if not all(col in frbs.columns for col in ["ra", "dec"]):
    print("   missing columns: frbs.csv must contain 'ra' and 'dec'")
    exit()

coords = SkyCoord(
    ra=frbs["ra"].values*u.deg,
    dec=frbs["dec"].values*u.deg,
    frame="icrs"
).galactic

seps = coords.separation(best_axis).deg
frbs["sep"] = seps

print(f"   angular separation computed for all frbs")


# ----------------------------------------------------------
# 2. choose energy proxy
# ----------------------------------------------------------

print("\n3. energy proxy selection")

if "fluence" in frbs.columns:
    proxy = frbs["fluence"].values
    proxy_name = "fluence"
    print("   using proxy: fluence")
else:
    proxy = frbs["snr"].values
    proxy_name = "snr"
    print("   using proxy: snr (fluence not available)")

# percentile splits
p1 = np.percentile(proxy, 33)
p2 = np.percentile(proxy, 66)

def energy_bin(val):
    if val <= p1: return "low"
    if val <= p2: return "mid"
    return "high"

frbs["bin"] = [energy_bin(v) for v in proxy]


# ----------------------------------------------------------
# 3. alignment statistics per bin
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("4. alignment results by energy bin")
print("=" * 70)

bins = ["low", "mid", "high"]

results = {}

for b in bins:
    subset = frbs[frbs["bin"] == b]
    s = subset["sep"].values
    
    mean_sep   = np.mean(s)
    median_sep = np.median(s)
    frac10     = np.mean(s < 10)
    frac20     = np.mean(s < 20)
    frac30     = np.mean(s < 30)
    
    results[b] = (mean_sep, median_sep, frac10, frac20, frac30)

    print(f"\nenergy bin: {b}")
    print(f"   count              = {len(s)}")
    print(f"   mean separation    = {mean_sep:6.2f}°")
    print(f"   median separation  = {median_sep:6.2f}°")
    print(f"   frac within 10°    = {frac10:.3f}")
    print(f"   frac within 20°    = {frac20:.3f}")
    print(f"   frac within 30°    = {frac30:.3f}")


# ----------------------------------------------------------
# 4. interpretation
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("5. interpretation")
print("=" * 70)

low  = results["low"]
mid  = results["mid"]
high = results["high"]

print("\nkey indicators:")
print("   • decreasing separation with higher energy")
print("   • increasing fraction < 10° / 20° toward high-energy bin")

score = 0
if high[0] < mid[0] and mid[0] < low[0]: score += 1   # mean sep
if high[1] < mid[1] and mid[1] < low[1]: score += 1   # median sep
if high[2] > mid[2] and mid[2] < low[2]: score += 1   # frac < 10°
    
print("\nassessment:")

if score == 3:
    print("   ★ strong energy-dependent alignment detected")
    print("   ★ supports frequency-gradient cone model")
elif score == 2:
    print("   ★ partial support for energy-dependent alignment")
elif score == 1:
    print("   ~ weak indication of alignment trend")
else:
    print("   ✗ no clear energy-dependent trend detected")

print("\n" + "=" * 70)
print("done")
print("=" * 70)
