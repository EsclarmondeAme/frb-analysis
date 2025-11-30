#!/usr/bin/env python3
# ======================================================================
# FRB MAGNETIC-FIELD ALIGNMENT TEST (TEST 21)
# Tests alignment between the FRB unified axis and several independent
# magnetic-field directions:
#   - Galactic magnetic dipole (Oppermann+ 2015 RM map)
#   - Planck polarized dust dipole
#   - Extragalactic RM dipole reconstruction (Taylor+ 2009-type)
#
# Computes angular separations + Monte Carlo isotropic null.
# ======================================================================

import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

# ---------------------------------------------------------------
# helper: convert (l,b) to unit vector
# ---------------------------------------------------------------
def lb_to_vec(l_deg, b_deg):
    l = np.radians(l_deg)
    b = np.radians(b_deg)
    x = np.cos(b) * np.cos(l)
    y = np.cos(b) * np.sin(l)
    z = np.sin(b)
    return np.array([x, y, z])

# ---------------------------------------------------------------
# angular separation
# ---------------------------------------------------------------
def angsep(v1, v2):
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))


# ======================================================================
# Load FRB unified axis from catalogue
# ======================================================================
if len(sys.argv) < 2:
    print("usage: python frb_magnetic_alignment_test.py frbs_unified.csv")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])

# best-fit unified FRB axis (fixed from earlier tests)
# (l,b) ≈ (159.85°, -0.51°)
FRB_axis = lb_to_vec(159.85, -0.51)


# ======================================================================
# Magnetic-field directions to test
# ======================================================================
# 1. Galactic magnetic dipole axis (Oppermann+ 2015 RM map)
RM_l, RM_b = 84.0, 10.0     # approximate literature values
RM_axis = lb_to_vec(RM_l, RM_b)

# 2. Planck polarized dust dipole (magnetic-field proxy)
PLANCK_l, PLANCK_b = 110.0, 15.0
PLANCK_axis = lb_to_vec(PLANCK_l, PLANCK_b)

# 3. Extragalactic RM dipole (cosmic magnetic field estimate)
XRM_l, XRM_b = 130.0, 25.0
XRM_axis = lb_to_vec(XRM_l, XRM_b)


# ======================================================================
# Compute observed angular separations
# ======================================================================
sep_RM      = angsep(FRB_axis, RM_axis)
sep_PLANCK  = angsep(FRB_axis, PLANCK_axis)
sep_XRM     = angsep(FRB_axis, XRM_axis)

seps_obs = np.array([sep_RM, sep_PLANCK, sep_XRM])
T_obs = np.mean(seps_obs)   # combined alignment statistic


# ======================================================================
# Monte Carlo null: FRB axis random on sphere
# ======================================================================
def random_unit():
    z = np.random.uniform(-1,1)
    t = np.arccos(z)
    p = np.random.uniform(0, 2*np.pi)
    return np.array([
        np.sin(t)*np.cos(p),
        np.sin(t)*np.sin(p),
        np.cos(t)
    ])

T_null = []
for _ in tqdm(range(50000), desc="MC"):
    rand = random_unit()
    s1 = angsep(rand, RM_axis)
    s2 = angsep(rand, PLANCK_axis)
    s3 = angsep(rand, XRM_axis)
    T_null.append(np.mean([s1,s2,s3]))

T_null = np.array(T_null)
p_value = np.mean(T_null <= T_obs)   # how often random is *more aligned*

# ======================================================================
# output
# ======================================================================
print("======================================================================")
print(" FRB MAGNETIC-FIELD ALIGNMENT TEST (TEST 21)")
print("======================================================================")
print(f" FRB axis vs Galactic RM dipole:        {sep_RM:.3f} deg")
print(f" FRB axis vs Planck dust dipole:        {sep_PLANCK:.3f} deg")
print(f" FRB axis vs extragalactic RM dipole:   {sep_XRM:.3f} deg")
print("----------------------------------------------------------------------")
print(f" Combined alignment statistic T_obs:    {T_obs:.3f} deg")
print(f" Monte Carlo p-value:                   {p_value:.5f}")
print("----------------------------------------------------------------------")

if p_value < 0.01:
    print(" verdict: FRB anisotropy shows significant alignment with known magnetic-field directions.")
elif p_value < 0.1:
    print(" verdict: mild magnetic-field alignment.")
else:
    print(" verdict: alignment consistent with random expectations (no magnetic correlation required).")

print("======================================================================")
