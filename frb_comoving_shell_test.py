#!/usr/bin/env python3
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# unified axis (from your latex document)
AXIS_L = 159.85   # degrees
AXIS_B = -0.51    # degrees
unified_axis = SkyCoord(l=AXIS_L*u.deg, b=AXIS_B*u.deg, frame="galactic")

# -------------------------------------------------------------------
# 1. load FRB catalogue
print("loading FRB catalogue...")
df = pd.read_csv("frbs.csv")      # <=== update if your filename differs
ra = df["ra"].to_numpy()
dec = df["dec"].to_numpy()
z = df["z_est"].fillna(df["z_est"].median()).to_numpy()

# -------------------------------------------------------------------
# 2. convert to galactic coordinates
coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs").galactic

# angular distance from unified axis
theta = coords.separation(unified_axis).deg      # degrees

# -------------------------------------------------------------------
# 3. convert redshift → comoving distance (Mpc)
Dc = cosmo.comoving_distance(z).value   # Mpc

# convert to 3-D comoving radius relative to axis:
# R = sqrt(Dc^2 + Dc_axis^2 − 2*Dc*Dc_axis*cos(theta))
# but Dc_axis is arbitrary — we only care about “distance perpendicular to axis”
# so we use transverse comoving radius:
R_trans = Dc * np.sin(np.radians(theta))

# -------------------------------------------------------------------
# 4. search for comoving-shell peak
bins = np.linspace(0, np.percentile(R_trans, 99), 30)
hist, edges = np.histogram(R_trans, bins=bins)
centers = 0.5*(edges[1:] + edges[:-1])

peak_index = np.argmax(hist)
R_shell = centers[peak_index]

print("----------------------------------------------------")
print("approx comoving-shell radius:")
print(f"R_shell ≈ {R_shell:.2f} Mpc")
print("----------------------------------------------------")

# -------------------------------------------------------------------
# 5. test if shell is cosmological: Monte-Carlo null
N = len(R_trans)
MC = 5000

print(f"running {MC} Monte Carlo realisations...")

peak_MC = []
for _ in range(MC):
    # isotropic angles, same redshifts
    theta_rand = np.degrees(np.arccos(2*np.random.rand(N)-1))
    R_rand = Dc * np.sin(np.radians(theta_rand))

    h, _ = np.histogram(R_rand, bins=bins)
    peak_MC.append(centers[np.argmax(h)])

peak_MC = np.array(peak_MC)

# p-value: probability that isotropic comoving sky produces same R_shell
p = np.mean(peak_MC >= R_shell)

print("----------------------------------------------------")
print(f"Monte Carlo p-value for a comoving shell:  p = {p:.5f}")
print("interpretation:")
if p < 0.01:
    print("  strong evidence that the shell is cosmological (not angular only)")
elif p < 0.05:
    print("  moderate evidence for a cosmological shell")
else:
    print("  shell likely arises from angular geometry, not comoving structure")
print("----------------------------------------------------")

# -------------------------------------------------------------------
# 6. plot result
plt.figure(figsize=(9,6))
plt.hist(R_trans, bins=bins, alpha=0.6, label="FRBs", density=True)
plt.hist(peak_MC, bins=30, alpha=0.4, label="MC shell peaks", density=True)
plt.axvline(R_shell, color="k", linestyle="--", label=f"FRB peak: {R_shell:.1f} Mpc")
plt.xlabel("transverse comoving radius R_trans (Mpc)")
plt.ylabel("normalized counts")
plt.title("FRB Comoving-Shell Test")
plt.legend()
plt.tight_layout()
plt.savefig("frb_comoving_shell_test.png")
print("saved plot: frb_comoving_shell_test.png")
