import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

# ============================================================
# Cosmic bulk-flow dipole (2M++)
# Published direction (Carrick et al. 2015):
#   l = 276°, b =  30°
# ============================================================

flow_l = np.radians(276.0)
flow_b = np.radians(30.0)

flow_vec = np.array([
    np.cos(flow_b) * np.cos(flow_l),
    np.cos(flow_b) * np.sin(flow_l),
    np.sin(flow_b)
])

# ============================================================
# load FRB catalogue
# ============================================================
if len(sys.argv) < 2:
    print("usage: python frb_flow_alignment_test.py frbs_unified.csv")
    sys.exit()

df = pd.read_csv(sys.argv[1])

theta = np.radians(df["theta_unified"].values)
phi   = np.radians(df["phi_unified"].values)

# unified FRB axis from your previous tests:
l_frb = np.radians(159.85)
b_frb = np.radians(-0.51)

frb_vec = np.array([
    np.cos(b_frb) * np.cos(l_frb),
    np.cos(b_frb) * np.sin(l_frb),
    np.sin(b_frb)
])

# ============================================================
# angular separation
# ============================================================
dot = np.clip(np.dot(flow_vec, frb_vec), -1, 1)
sep = np.degrees(np.arccos(dot))

# ============================================================
# Monte Carlo null
# ============================================================
NMC = 50000
count = 0

for _ in tqdm(range(NMC), desc="MC"):
    # random axis on sphere
    z = np.random.uniform(-1, 1)
    th = np.arccos(z)
    ph = np.random.uniform(0, 2*np.pi)
    v = np.array([np.sin(th)*np.cos(ph),
                  np.sin(th)*np.sin(ph),
                  np.cos(th)])

    d = np.degrees(
        np.arccos(np.clip(np.dot(v, flow_vec), -1, 1))
    )

    if d <= sep:
        count += 1

p = count / NMC

# ============================================================
# report
# ============================================================
print("======================================================================")
print(" FRB – Cosmic Bulk-Flow Alignment Test (Test 22)")
print("======================================================================")
print(f" bulk-flow dipole direction: (l={np.degrees(flow_l):.1f}°, b={np.degrees(flow_b):.1f}°)")
print(f" FRB unified axis direction: (l={np.degrees(l_frb):.1f}°, b={np.degrees(b_frb):.1f}°)")
print(f" angular separation:         {sep:.3f} degrees")
print("----------------------------------------------------------------------")
print(f" Monte Carlo p-value:        {p:.5f}")
print("----------------------------------------------------------------------")

if p < 0.001:
    print(" verdict: strong evidence of alignment with cosmic flow (unexpected).")
elif p < 0.01:
    print(" verdict: mild alignment; could be physically meaningful.")
elif p < 0.1:
    print(" verdict: broad alignment range; not contradictory.")
else:
    print(" verdict: alignment consistent with random expectation.")
print("======================================================================")
