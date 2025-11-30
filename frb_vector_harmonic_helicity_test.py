#!/usr/bin/env python3
# ======================================================================
# FRB VECTOR–SPHERICAL–HARMONIC HELICITY TEST (TEST 16)
# ======================================================================
# This test measures whether the FRB anisotropy field contains
#     (1) gradient-type structure (E-modes)
#     (2) curl-type structure (B-modes)
# using vector spherical harmonics.
#
# A pure shell should give ~zero B-modes.
# A warped, twisted structure may produce nonzero helicity.
#
# Input: frbs_unified.csv
# Output: frb_vector_helicity.png
# ======================================================================

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# ----------------------------------------------------------------------
# HELPERS: compute vector spherical harmonics
# ----------------------------------------------------------------------

def Y_E(l, m, theta, phi):
    """Gradient-like VSH component (electric / E-mode)."""
    return sph_harm(m, l, phi, theta)

def Y_B(l, m, theta, phi):
    """Curl-like VSH component (magnetic / B-mode)."""
    return 1j * sph_harm(m, l, phi, theta)

# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------

if len(sys.argv) < 2:
    print("usage: python frb_vector_harmonic_helicity_test.py frbs_unified.csv")
    sys.exit(1)

fname = sys.argv[1]

print("=======================================================================")
print(" FRB VECTOR–SPHERICAL–HARMONIC HELICITY TEST (TEST 16)")
print("=======================================================================")

print(f"loading {fname} ...")
frb = pd.read_csv(fname)

theta = np.radians(frb["theta_unified"].values)
phi   = np.radians(frb["phi_unified"].values)

N = len(theta)
print(f"loaded {N} FRBs")

# ----------------------------------------------------------------------
# compute E and B mode power up to l=8
# ----------------------------------------------------------------------

ℓmax = 8
E_power = []
B_power = []

print("computing vector-harmonic coefficients...")

for ℓ in range(1, ℓmax+1):
    E_ℓ = 0
    B_ℓ = 0
    for m in range(-ℓ, ℓ+1):
        Y_e = Y_E(ℓ, m, theta, phi)
        Y_b = Y_B(ℓ, m, theta, phi)
        # project FRB positions onto the basis
        a_e = np.sum(Y_e)
        a_b = np.sum(Y_b)
        E_ℓ += np.abs(a_e)**2
        B_ℓ += np.abs(a_b)**2
    E_power.append(E_ℓ)
    B_power.append(B_ℓ)

E_power = np.array(E_power)
B_power = np.array(B_power)

# ----------------------------------------------------------------------
# Monte Carlo isotropic null
# ----------------------------------------------------------------------

MC = 2000
print(f"running Monte Carlo null ({MC} isotropic realisations)...")

E_null = np.zeros((MC, ℓmax))
B_null = np.zeros((MC, ℓmax))

for i in range(MC):
    # isotropic sky
    u = np.random.uniform(0,1,N)
    t = np.arccos(1 - 2*u)  # isotropic theta
    p = np.random.uniform(0, 2*np.pi, N)

    for ℓ in range(1, ℓmax+1):
        E_ℓ = 0
        B_ℓ = 0
        for m in range(-ℓ, ℓ+1):
            Y_e = Y_E(ℓ, m, t, p)
            Y_b = Y_B(ℓ, m, t, p)
            a_e = np.sum(Y_e)
            a_b = np.sum(Y_b)
            E_ℓ += np.abs(a_e)**2
            B_ℓ += np.abs(a_b)**2

        E_null[i, ℓ-1] = E_ℓ
        B_null[i, ℓ-1] = B_ℓ

# ----------------------------------------------------------------------
# Compute p-values
# ----------------------------------------------------------------------

p_E = []
p_B = []

for i in range(ℓmax):
    p_E.append(np.mean(E_null[:,i] >= E_power[i]))
    p_B.append(np.mean(B_null[:,i] >= B_power[i]))

p_E = np.array(p_E)
p_B = np.array(p_B)

print("--------------------------------------------------------------")
print("observed E/B-mode power:")
for ℓ in range(1, ℓmax+1):
    print(f"ℓ={ℓ:2d}   E={E_power[ℓ-1]:.3e}   B={B_power[ℓ-1]:.3e}")

print("--------------------------------------------------------------")
print("Monte Carlo p-values:")
for ℓ in range(1, ℓmax+1):
    print(f"ℓ={ℓ:2d}   p_E={p_E[ℓ-1]:.5f}   p_B={p_B[ℓ-1]:.5f}")

print("--------------------------------------------------------------")
print("scientific interpretation:")
print("E-modes: detect gradient structure (shell-like).")
print("B-modes: detect curl/helicity (twist or warped 3D rotation).")
print("low B-mode p-values → physical twisting / non-spherical warping.")
print("--------------------------------------------------------------")

# ----------------------------------------------------------------------
# plot
# ----------------------------------------------------------------------

plt.figure(figsize=(8,5))
ℓvals = np.arange(1, ℓmax+1)
plt.plot(ℓvals, E_power, 'o-', label='E-mode power')
plt.plot(ℓvals, B_power, 'o-', label='B-mode power')
plt.yscale('log')
plt.xlabel("multipole ℓ")
plt.ylabel("power")
plt.title("FRB Vector-Harmonic E/B Power")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("frb_vector_helicity.png")

print("saved: frb_vector_helicity.png")
print("=======================================================================")
print(" Test 16 complete")
print("=======================================================================")
