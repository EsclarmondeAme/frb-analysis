import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time

# ------------------------------------------------------------
# utility: sidereal phase from MJD
# ------------------------------------------------------------
def sidereal_phase_from_mjd(mjd):
    """
    convert MJD → sidereal phase in [0,1)
    using GMST formula (approx; consistent with other scripts)
    """
    t = Time(mjd, format="mjd", scale="utc")
    gmst = t.sidereal_time("mean", "greenwich")  # in hours
    return (gmst.value / 24.0) % 1.0


# ------------------------------------------------------------
# load FRBs
# ------------------------------------------------------------
print("[INFO] loading FRBs...")

frb = pd.read_csv("frbs.csv")
if "mjd" not in frb.columns:
    raise ValueError("frbs.csv missing 'mjd' column")

frb = frb.dropna(subset=["mjd"])
frb_phase = sidereal_phase_from_mjd(frb["mjd"].values)

print(f"[INFO] FRBs with valid MJD: {len(frb_phase)}")


# ------------------------------------------------------------
# load neutrinos
# ------------------------------------------------------------
print("[INFO] loading neutrinos...")

nu = pd.read_csv("neutrinos.csv")
use_col = None

for c in ["mjd", "MJD", "mjd_time", "utc_mjd"]:
    if c in nu.columns:
        use_col = c
        break

if use_col is None:
    raise ValueError("no MJD column found in neutrinos.csv")

nu = nu.dropna(subset=[use_col])
nu_phase = sidereal_phase_from_mjd(nu[use_col].values)

print(f"[INFO] neutrinos with valid MJD: {len(nu_phase)}")


# ------------------------------------------------------------
# harmonic amplitudes A_n, B_n, R_n
# ------------------------------------------------------------
def harmonic_coeffs(phases, n):
    phi = np.asarray(phases)
    A = np.mean(np.cos(2*np.pi*n*phi))
    B = np.mean(np.sin(2*np.pi*n*phi))
    R = np.sqrt(A*A + B*B)
    return A, B, R


# compute dipole only (n=1)
print("[INFO] computing dipole harmonics...")

A1_frb, B1_frb, R1_frb = harmonic_coeffs(frb_phase, 1)
A1_nu,  B1_nu,  R1_nu  = harmonic_coeffs(nu_phase, 1)

print("------------------------------------------------------------")
print("dipole harmonic amplitudes:")
print(f"FRB:      A1={A1_frb:+.4f}  B1={B1_frb:+.4f}  R1={R1_frb:.4f}")
print(f"neutrino: A1={A1_nu:+.4f}  B1={B1_nu:+.4f}  R1={R1_nu:.4f}")
print("------------------------------------------------------------")

# ------------------------------------------------------------
# joint test: do both datasets have the SAME harmonic phase?
# ------------------------------------------------------------
phi_frb = np.arctan2(B1_frb, A1_frb)
phi_nu  = np.arctan2(B1_nu,  A1_nu)

# convert to degrees in [0,360)
deg_frb = (np.degrees(phi_frb) + 360) % 360
deg_nu  = (np.degrees(phi_nu)  + 360) % 360

phase_diff = abs(deg_frb - deg_nu)
if phase_diff > 180:
    phase_diff = 360 - phase_diff

print("[INFO] joint axis comparison:")
print(f"FRB dipole direction:      {deg_frb:.2f} deg")
print(f"neutrino dipole direction: {deg_nu:.2f} deg")
print(f"absolute phase difference: {phase_diff:.2f} deg")
print("------------------------------------------------------------")


# ------------------------------------------------------------
# Monte Carlo: how often does random neutrino data match FRB axis?
# ------------------------------------------------------------
print("[INFO] running Monte Carlo... this is fast.")

Nmc = 20000
count = 0

for _ in range(Nmc):
    phi_rand = np.random.rand(len(nu_phase))
    A, B, R = harmonic_coeffs(phi_rand, 1)
    angle = (np.degrees(np.arctan2(B, A))+360)%360

    diff = abs(angle - deg_frb)
    if diff > 180:
        diff = 360 - diff

    if diff <= phase_diff:
        count += 1

p_value = count / Nmc

print(f"[INFO] Monte Carlo p-value = {p_value:.4f}")
print("------------------------------------------------------------")

# ------------------------------------------------------------
# plot both histograms + dipole directions
# ------------------------------------------------------------
print("[INFO] plotting...")

plt.figure(figsize=(12,5))
plt.hist(frb_phase, bins=30, alpha=0.4, density=True, label="FRB sidereal")
plt.hist(nu_phase,  bins=30, alpha=0.4, density=True, label="neutrino sidereal")

plt.axvline(deg_frb/360, color="blue", linestyle="--", label="FRB dipole dir")
plt.axvline(deg_nu/360, color="orange", linestyle="-.", label="nu dipole dir")

plt.xlabel("sidereal phase")
plt.ylabel("probability density")
plt.legend()
plt.title("FRB vs neutrino sidereal phases + dipole directions")
plt.tight_layout()
plt.savefig("frb_neutrino_joint_sidereal.png")

print("[INFO] saved → frb_neutrino_joint_sidereal.png")
print("[done]")
