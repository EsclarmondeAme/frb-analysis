"""
frb_axis_shell_scan.py
------------------------------------------------------------
search for physical shell-like structure around the unified axis.

this script:
  - computes angle of each frb from the unified axis
  - bins frbs in thin radial shells (1 degree)
  - compares observed counts against isotropic expectation
  - computes density ratio and its derivative
  - identifies candidate shell boundaries
  - produces a figure: frb_shell_scan.png
------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

# ------------------------------------------------------------
# unified best-fit axis (from your previous analysis)
# ------------------------------------------------------------
AXIS_L = 159.85
AXIS_B = -0.51
axis = SkyCoord(l=AXIS_L*u.deg, b=AXIS_B*u.deg, frame="galactic")

print("="*70)
print("frb axis shell scan")
print("search for layered structure in angular distance from axis")
print("="*70)

# ------------------------------------------------------------
# load frb catalog
# ------------------------------------------------------------
try:
    frbs = pd.read_csv("frbs.csv")
except FileNotFoundError:
    print("\n[error] frbs.csv not found")
    exit()

coords = SkyCoord(ra=frbs["ra"].values*u.deg,
                  dec=frbs["dec"].values*u.deg,
                  frame="icrs").galactic

sep = coords.separation(axis).deg
frbs["sep_deg"] = sep

print("\n1. frb catalog")
print("------------------------------------------------------------")
print(f"total frbs loaded: {len(frbs)}")
print("computed angular separation from unified axis for all events")

# ------------------------------------------------------------
# radial shells
# ------------------------------------------------------------
shell_width = 1.0        # 1 degree shells
max_angle = 90           # hemisphere

bins = np.arange(0, max_angle + shell_width, shell_width)
centers = bins[:-1] + shell_width/2

counts = []
fluence_mean = []
fluence_median = []

for i in range(len(bins)-1):
    low, high = bins[i], bins[i+1]
    in_shell = frbs[(sep >= low) & (sep < high)]
    counts.append(len(in_shell))

    if "fluence" in frbs.columns:
        fluence_mean.append(
            in_shell["fluence"].mean() if len(in_shell)>0 else np.nan
        )
        fluence_median.append(
            in_shell["fluence"].median() if len(in_shell)>0 else np.nan
        )
    else:
        fluence_mean.append(np.nan)
        fluence_median.append(np.nan)

counts = np.array(counts)

print("\n2. shell binning (1° resolution)")
print("------------------------------------------------------------")
print(f"shells scanned: 0° – {max_angle}°")
print("computed observed counts and fluence statistics")

# ------------------------------------------------------------
# isotropic expectation
# ------------------------------------------------------------
total = len(frbs)
iso_expected = []

for i in range(len(bins)-1):
    t1 = np.radians(bins[i])
    t2 = np.radians(bins[i+1])
    p = (np.cos(t1) - np.cos(t2)) / 2
    iso_expected.append(total * p)

iso_expected = np.array(iso_expected)

print("\n3. isotropic reference model")
print("------------------------------------------------------------")
print("computed expected counts for each shell under isotropy")

# ------------------------------------------------------------
# density + derivative
# ------------------------------------------------------------
density = counts / iso_expected
density_deriv = np.gradient(density)

# detect candidate edges (sharp changes)
threshold = np.percentile(np.abs(density_deriv), 95)
edges = np.where(np.abs(density_deriv) > threshold)[0]
candidate_edges = centers[edges]

print("\n4. shell boundary analysis")
print("------------------------------------------------------------")
print(f"derivative threshold: top 5 percent of |d/dθ density|")
if len(candidate_edges) == 0:
    print("no candidate shell boundaries detected")
else:
    print("candidate shell boundaries (degrees from axis):")
    print(", ".join(f"{e:.1f}" for e in candidate_edges))

# ------------------------------------------------------------
# plotting
# ------------------------------------------------------------
plt.figure(figsize=(16,9))

# top: counts vs isotropic
plt.subplot(3,1,1)
plt.plot(centers, counts, label="observed", color="blue")
plt.plot(centers, iso_expected, label="expected (isotropic)", color="gray")
plt.title("frb counts per 1° shell")
plt.xlabel("angle from axis (deg)")
plt.ylabel("counts")
plt.grid(alpha=0.3)
plt.legend()

# middle: density ratio
plt.subplot(3,1,2)
plt.plot(centers, density, color="black")
for e in candidate_edges:
    plt.axvline(e, color="red", linestyle="--", alpha=0.5)
plt.title("density ratio (observed / isotropic)")
plt.xlabel("angle from axis (deg)")
plt.ylabel("density ratio")
plt.grid(alpha=0.3)

# bottom: derivative
plt.subplot(3,1,3)
plt.plot(centers, density_deriv, color="purple")
for e in candidate_edges:
    plt.axvline(e, color="red", linestyle="--", alpha=0.5)
plt.title("density derivative (boundary detector)")
plt.xlabel("angle from axis (deg)")
plt.ylabel("d/dθ density")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("frb_shell_scan.png", dpi=200, bbox_inches="tight")

print("\n5. figure output")
print("------------------------------------------------------------")
print("saved: frb_shell_scan.png")

print("\n" + "="*70)
print("analysis complete")
print("="*70)
