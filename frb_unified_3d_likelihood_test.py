#!/usr/bin/env python3
# ==============================================================
#  FRB UNIFIED 3D LIKELIHOOD TEST  (TEST 14)
#  --------------------------------------------------------------
#  Inputs:
#      vertices.npy   – 3D isosurface vertices (Nx3)
#      faces.npy      – triangular faces (Mx3)
#
#  What this test computes:
#      - 3D shell normals
#      - axis alignment likelihood
#      - curvature warp likelihood
#      - 3D spherical-harmonic surface power (ℓ≤6)
#      - symmetry-breaking score
#      - redshift-shell agreement from Test 13 summary
#      - Bayesian model-evidence contribution (Tests 6–8)
#      - selection-function failure term (Test 12)
#
#  Output:
#      unified_3d_axis_normals.png
#      unified_3d_curvature.png
#      unified_3d_shellwarp.png
#      unified_3d_summary.json
#
# ==============================================================
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.special import sph_harm
from numpy.linalg import norm

# --------------------------------------------------------------
# util: log-likelihood converter
# --------------------------------------------------------------
def L(p):
    """Convert p-value to -log10(p), safe for tiny values."""
    if p <= 0:
        return 50.0
    return -np.log10(p)

# --------------------------------------------------------------
# load marching-cubes geometry
# --------------------------------------------------------------
print("=======================================================================")
print(" FRB UNIFIED 3D LIKELIHOOD TEST (TEST 14)")
print("=======================================================================")

try:
    vertices = np.load("vertices.npy")
    faces    = np.load("faces.npy")
except Exception as e:
    print("ERROR: Could not load vertices.npy / faces.npy")
    raise SystemExit

V = vertices.copy()
F = faces.copy()
print(f"Loaded vertices: {V.shape}, faces: {F.shape}")

# --------------------------------------------------------------
# compute triangle normals
# --------------------------------------------------------------
def compute_normals(vertices, faces):
    v0 = vertices[faces[:,0]]
    v1 = vertices[faces[:,1]]
    v2 = vertices[faces[:,2]]
    n  = np.cross(v1 - v0, v2 - v0)
    norms = norm(n, axis=1)
    n = n / norms[:,None]
    return n

normals = compute_normals(V, F)
print("Computed surface normals.")

# --------------------------------------------------------------
# unified cosmic axis (from tests 1–12)
# --------------------------------------------------------------
axis_dir = np.array([ 
    np.cos(np.deg2rad(159.85))*np.cos(np.deg2rad(-0.51)),
    np.sin(np.deg2rad(159.85))*np.cos(np.deg2rad(-0.51)),
    np.sin(np.deg2rad(-0.51))
])
axis_dir = axis_dir / norm(axis_dir)

# --------------------------------------------------------------
# AXIS ALIGNMENT likelihood
# --------------------------------------------------------------
cos_angles = normals @ axis_dir
mean_align = np.mean(np.abs(cos_angles))

# Monte Carlo null (random axes)
MC = []
for i in range(2000):
    # random isotropic axis
    u = np.random.normal(size=3)
    u /= norm(u)
    MC.append(np.mean(np.abs(normals @ u)))
MC = np.array(MC)

p_axis = np.mean(MC >= mean_align)
L_axis = L(p_axis)
print(f"Axis-alignment p = {p_axis:.3e},  L = {L_axis:.3f}")

# --------------------------------------------------------------
# CURVATURE WARP likelihood
# --------------------------------------------------------------
# measure local curvature via triangle area deviation
def triangle_area(a, b, c):
    return 0.5 * norm(np.cross(b-a, c-a))

areas = []
for f in F:
    a = V[f[0]]
    b = V[f[1]]
    c = V[f[2]]
    areas.append(triangle_area(a, b, c))
areas = np.array(areas)
A_obs = np.std(areas)

# null: random shuffle of vertices (destroys structure)
MC_A = []
for i in range(2000):
    idx = np.random.permutation(len(V))
    A = []
    for f in F:
        a, b, c = V[idx[f[0]]], V[idx[f[1]]], V[idx[f[2]]]
        A.append(triangle_area(a,b,c))
    MC_A.append(np.std(A))

MC_A = np.array(MC_A)
p_curv = np.mean(MC_A >= A_obs)
L_curv = L(p_curv)
print(f"Curvature-warp p = {p_curv:.3e},  L = {L_curv:.3f}")

# --------------------------------------------------------------
# 3D SPHERICAL-HARMONIC POWER (ℓ ≤ 6)
# --------------------------------------------------------------
# convert vertices to spherical coords
x, y, z = V[:,0], V[:,1], V[:,2]
r = np.sqrt(x*x + y*y + z*z)
theta = np.arccos(z / r)
phi = np.arctan2(y, x)

maxell = 6
power_obs = []
for ell in range(1, maxell+1):
    a_lm = []
    for m in range(-ell, ell+1):
        Ylm = sph_harm(m, ell, phi, theta)
        a_lm.append(np.sum(Ylm))
    power = np.sum(np.abs(a_lm)**2)
    power_obs.append(power)

power_obs = np.array(power_obs)

# null: shuffle vertices
MC_power = []
for i in range(800):
    idx = np.random.permutation(len(V))
    x2, y2, z2 = x[idx], y[idx], z[idx]
    r2 = np.sqrt(x2*x2 + y2*y2 + z2*z2)
    th2 = np.arccos(z2 / r2)
    ph2 = np.arctan2(y2, x2)

    P = []
    for ell in range(1, maxell+1):
        a_lm = []
        for m in range(-ell, ell+1):
            Y2 = sph_harm(m, ell, ph2, th2)
            a_lm.append(np.sum(Y2))
        P.append(np.sum(np.abs(a_lm)**2))
    MC_power.append(np.sum(P))

MC_power = np.array(MC_power)
p_harm = np.mean(MC_power >= np.sum(power_obs))
L_harm = L(p_harm)
print(f"Harmonic-3D p = {p_harm:.3e},  L = {L_harm:.3f}")

# --------------------------------------------------------------
# SYMMETRY-BREAKING likelihood (azimuthal)
# --------------------------------------------------------------
# compute deviation from rotational symmetry about axis_dir
axis = axis_dir / norm(axis_dir)
def project(v):
    return v - np.dot(v, axis)*axis

proj = np.array([project(n) for n in normals])
az_angles = np.arctan2(proj[:,1], proj[:,0])
R_obs = np.std(az_angles)

MC_R = []
for i in range(2000):
    idx = np.random.permutation(len(normals))
    ang = az_angles[idx]
    MC_R.append(np.std(ang))

p_sym = np.mean(np.array(MC_R) >= R_obs)
L_sym = L(p_sym)
print(f"Symmetry-breaking p = {p_sym:.3e},  L = {L_sym:.3f}")

# --------------------------------------------------------------
# REDSHIFT-TOMOGRAPHY likelihood (from test 13)
# --------------------------------------------------------------
# use your measured p = 0.014 from Test 13 directly
p_tomo = 0.014
L_tomo = L(p_tomo)
print(f"Tomography drift p = {p_tomo:.3e},  L = {L_tomo:.3f}")

# --------------------------------------------------------------
# SELECTION-FUNCTION failure (from test 12)
# --------------------------------------------------------------
p_sel = 4.7e-19
L_sel = L(p_sel)

# --------------------------------------------------------------
# BAYESIAN: warped shell vs isotropic/dipole (test 6–8)
# --------------------------------------------------------------
p_bayes = 1e-10
L_bayes = L(p_bayes)

# --------------------------------------------------------------
# FINAL UNIFIED 3D LIKELIHOOD
# --------------------------------------------------------------
L_tot = L_axis + L_curv + L_harm + L_sym + L_tomo + L_sel + L_bayes
p_eff = 10**(-L_tot)

print("--------------------------------------------------------------")
print(f"Unified 3D log-likelihood sum  L_tot = {L_tot:.3f}")
print(f"Unified effective p-value      p_eff = {p_eff:.3e}")
print("--------------------------------------------------------------")

# --------------------------------------------------------------
# Save summary
# --------------------------------------------------------------
summary = {
    "L_axis": float(L_axis),
    "L_curv": float(L_curv),
    "L_harm": float(L_harm),
    "L_sym":  float(L_sym),
    "L_tomo": float(L_tomo),
    "L_sel":  float(L_sel),
    "L_bayes":float(L_bayes),
    "L_tot":  float(L_tot),
    "p_eff":  float(p_eff)
}
with open("unified_3d_summary.json","w") as f:
    json.dump(summary, f, indent=4)

print("Saved unified_3d_summary.json")

# --------------------------------------------------------------
# Make plots (axis normals)
# --------------------------------------------------------------
plt.figure(figsize=(6,6))
plt.scatter(normals[:,0], normals[:,1], s=3, alpha=0.5)
plt.title("Unified 3D: Surface Normals (XY-plane)")
plt.savefig("unified_3d_axis_normals.png", dpi=180)

plt.figure(figsize=(6,6))
plt.hist(areas, bins=60, alpha=0.7)
plt.title("Unified 3D: Triangle Area Distribution")
plt.savefig("unified_3d_curvature.png", dpi=180)

plt.figure(figsize=(6,6))
plt.hist(az_angles, bins=60, alpha=0.7)
plt.title("Unified 3D: Azimuthal Warp / Symmetry Breaking")
plt.savefig("unified_3d_shellwarp.png", dpi=180)

print("Saved 3D likelihood plots.")
print("=======================================================================")
print(" Test 14 complete.")
print("=======================================================================")
