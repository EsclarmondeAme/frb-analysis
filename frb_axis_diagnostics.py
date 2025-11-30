"""
frb_axis_diagnostics.py
----------------------------------------------------------
diagnostics of frb anisotropy relative to the cmb axis

methods:
  a) spatial clustering around cmb axis      (cosmic diagnostic)
  b) frb sky dipole direction                (detector footprint diagnostic)
  c) sidereal phase modulation               (cosmic diagnostic)

final verdict only uses (a) and (c) to assess cosmic axis correlation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from scipy import stats

print("=" * 70)
print("frb axis diagnostics relative to cmb")
print("=" * 70)

# ----------------------------------------------------------
# 1. cmb axis reference
# ----------------------------------------------------------

cmb_l = 152.62
cmb_b = 4.03
cmb_l_err = 10.0  # conservative uncertainty

cmb_coord = SkyCoord(l=cmb_l*u.deg, b=cmb_b*u.deg, frame="galactic")

print("\n" + "=" * 70)
print("1. cmb dipole modulation axis (reference)")
print("=" * 70)
print(f"   galactic coords: l = {cmb_l:.2f}° ± {cmb_l_err:.1f}°")
print(f"                    b = {cmb_b:.2f}°")
print("   source: planck low-ℓ hemispherical asymmetry")

# ----------------------------------------------------------
# 2. load frb catalog
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("2. frb catalog")
print("=" * 70)

try:
    frbs = pd.read_csv("frbs.csv")
except FileNotFoundError:
    print("   frbs.csv not found – cannot run diagnostics.")
    raise SystemExit

print(f"   loaded {len(frbs)} frbs from catalog")

if not all(col in frbs.columns for col in ["ra", "dec"]):
    print("   frbs.csv must contain 'ra' and 'dec' columns.")
    raise SystemExit

frb_coords = SkyCoord(
    ra=frbs["ra"].values*u.deg,
    dec=frbs["dec"].values*u.deg,
    frame="icrs"
).galactic

# ==========================================================
# 3. method a: spatial clustering around cmb axis (cosmic)
# ==========================================================

print("\n" + "-" * 70)
print("3. method a: spatial clustering around cmb axis")
print("-" * 70)

seps_from_cmb = frb_coords.separation(cmb_coord).deg

print("\n   testing if frbs cluster near cmb axis...")
print(f"   cmb axis location: l={cmb_l:.1f}°, b={cmb_b:.1f}°\n")

clustering_results = []

for radius in [15, 20, 25, 30]:
    n_near = np.sum(seps_from_cmb < radius)
    frac_obs = n_near / len(frbs)

    # cone fraction for random sky
    frac_expected = (1 - np.cos(np.radians(radius))) / 2

    # binomial test
    try:
        result = stats.binomtest(
            n_near, len(frbs), frac_expected, alternative="greater"
        )
        p_value = result.pvalue
    except AttributeError:
        p_value = stats.binom_test(
            n_near, len(frbs), frac_expected, alternative="greater"
        )

    excess_ratio = frac_obs / frac_expected if frac_expected > 0 else 0

    sig = ""
    if p_value < 0.001:
        sig = "*** highly significant"
    elif p_value < 0.01:
        sig = "** significant"
    elif p_value < 0.05:
        sig = "* marginal"

    print(f"   radius {radius:2d}°:")
    print(f"      observed: {n_near}/{len(frbs)} frbs ({frac_obs*100:.1f}%)")
    print(f"      expected: {frac_expected*100:.1f}% if random")
    print(f"      excess:   {excess_ratio:.2f}x")
    print(f"      p-value:  {p_value:.4g} {sig}\n")

    clustering_results.append({
        "radius": radius,
        "n_near": n_near,
        "frac_obs": frac_obs,
        "frac_exp": frac_expected,
        "excess": excess_ratio,
        "p_value": p_value,
    })

best_p_a = min(r["p_value"] for r in clustering_results)

print("   " + "=" * 66)
if best_p_a < 0.01:
    print("   ✓ method a verdict: significant spatial clustering detected")
    print(f"     best p-value: {best_p_a:.4g}")
elif best_p_a < 0.05:
    print("   ~ method a verdict: marginal clustering detected")
    print(f"     best p-value: {best_p_a:.4g}")
else:
    print("   ✗ method a verdict: no significant spatial clustering")
    print(f"     best p-value: {best_p_a:.4g}")
print("   " + "=" * 66)

# ==========================================================
# 4. method b: frb sky dipole (detector diagnostic)
# ==========================================================

print("\n" + "-" * 70)
print("4. method b: frb sky dipole direction (detector footprint)")
print("-" * 70)

print("\n   computing dipole from raw frb sky distribution...")
print("   note: this is dominated by chime beam and visibility,")
print("         and should be interpreted as an instrument diagnostic,")
print("         not as a cosmic preferred axis.\n")

x = np.cos(frb_coords.b.rad) * np.cos(frb_coords.l.rad)
y = np.cos(frb_coords.b.rad) * np.sin(frb_coords.l.rad)
z = np.sin(frb_coords.b.rad)

dipole = np.array([x.sum(), y.sum(), z.sum()])
dipole_amp = np.linalg.norm(dipole) / len(frbs)

dipole_l = np.degrees(np.arctan2(dipole[1], dipole[0])) % 360
dipole_b = np.degrees(np.arcsin(dipole[2] / np.linalg.norm(dipole)))

frb_dipole_coord = SkyCoord(
    l=dipole_l*u.deg, b=dipole_b*u.deg, frame="galactic"
)

print(f"   frb sky dipole (instrument footprint):")
print(f"      l = {dipole_l:.2f}°")
print(f"      b = {dipole_b:.2f}°")
print(f"      amplitude = {dipole_amp:.4f}")
print("      (0 = isotropic, 1 = all frbs in one direction)")

print("\n   bootstrap uncertainty (1000 resamples)...")
n_boot = 1000
boot_l, boot_b = [], []

for _ in range(n_boot):
    idx = np.random.choice(len(frbs), len(frbs), replace=True)
    bcoords = frb_coords[idx]
    xb = np.cos(bcoords.b.rad) * np.cos(bcoords.l.rad)
    yb = np.cos(bcoords.b.rad) * np.sin(bcoords.l.rad)
    zb = np.sin(bcoords.b.rad)
    d = np.array([xb.sum(), yb.sum(), zb.sum()])
    boot_l.append(np.degrees(np.arctan2(d[1], d[0])) % 360)
    boot_b.append(np.degrees(np.arcsin(d[2] / np.linalg.norm(d))))

l_err = np.std(boot_l)
b_err = np.std(boot_b)

print(f"      uncertainty: ±{l_err:.1f}° in l, ±{b_err:.1f}° in b")

sep_frb_cmb = frb_dipole_coord.separation(cmb_coord).deg
print(f"\n   separation from cmb axis: {sep_frb_cmb:.2f}°")

print("\n   interpretation:")
print("      this dipole primarily reflects where the telescope can see.")
print("      it is not used as cosmic evidence, but as a sanity check.")
print("      large offset from cmb axis is expected and not a problem.")

# ==========================================================
# 5. method c: sidereal phase analysis (cosmic)
# ==========================================================

print("\n" + "-" * 70)
print("5. method c: sidereal phase analysis (temporal anisotropy)")
print("-" * 70)

p_rayleigh = None

if "mjd" in frbs.columns:
    print("\n   testing for sidereal time modulation...")

    t = Time(frbs["mjd"].values, format="mjd")
    chime = EarthLocation(
        lat=49.3223*u.deg, lon=-119.6167*u.deg, height=545*u.m
    )
    lst = t.sidereal_time("apparent", longitude=chime.lon).hour

    phases = 2 * np.pi * lst / 24
    A = np.mean(np.cos(phases))
    B = np.mean(np.sin(phases))
    R = np.sqrt(A**2 + B**2)

    z = len(frbs) * R**2
    p_rayleigh = np.exp(-z) if z < 700 else 0.0

    phase_deg = (np.degrees(np.arctan2(B, A)) % 360)

    print(f"\n   sidereal phase analysis:")
    print(f"      peak phase: {phase_deg:.2f}° (sidereal hour angle)")
    print(f"      amplitude R: {R:.4f}")
    print(f"      rayleigh Z: {z:.2f}")
    print(f"      p-value: {p_rayleigh:.4e}")

    print("\n   " + "=" * 66)
    if p_rayleigh < 1e-3:
        print("   ✓ method c verdict: highly significant sidereal modulation")
        print(f"     p-value: {p_rayleigh:.4e}")
        print("     frbs arrive preferentially at specific sidereal times")
    elif p_rayleigh < 0.05:
        print("   ~ method c verdict: marginal sidereal modulation")
        print(f"     p-value: {p_rayleigh:.4e}")
    else:
        print("   ✗ method c verdict: no significant sidereal modulation")
        print(f"     p-value: {p_rayleigh:.4e}")
    print("   " + "=" * 66)
else:
    print("   mjd column not found – cannot perform sidereal analysis")

# ==========================================================
# 6. overall cosmic verdict (uses only A + C)
# ==========================================================

print("\n" + "=" * 70)
print("6. overall cosmic verdict (frb–cmb axis correlation)")
print("=" * 70)

evidence_count = 0
components = []

if best_p_a < 0.05:
    evidence_count += 1
    components.append("spatial clustering around cmb (method a)")

if p_rayleigh is not None and p_rayleigh < 0.05:
    evidence_count += 1
    components.append("sidereal modulation (method c)")

print(f"\n   evidence score (cosmic diagnostics only): {evidence_count}/2")
print(f"   significant components: {', '.join(components) if components else 'none'}")

print("\n   summary:")
if evidence_count == 2:
    print("   ★★★ strong evidence that frbs correlate with the cmb axis")
    print("   ★★★ both spatial clustering and sidereal modulation agree")
elif evidence_count == 1:
    print("   ~ partial evidence for frb–cmb correlation")
    print("   ~ one cosmic diagnostic is significant, the other is not")
else:
    print("   ✗ no robust evidence for frb–cmb axis correlation in these tests")

print("\n   note on method b:")
print("   the frb sky dipole (method b) is dominated by telescope coverage")
print("   and is not used as cosmic evidence in this verdict.")

# ==========================================================
# 7. simple visualization (optional)
# ==========================================================

print("\n" + "=" * 70)
print("7. generating frb–cmb visualization")
print("=" * 70)

try:
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection="mollweide")

    lon = frb_coords.l.deg
    lon = np.where(lon > 180, lon - 360, lon)
    lat = frb_coords.b.deg

    ax.scatter(np.radians(lon), np.radians(lat),
               s=10, alpha=0.5, c="gray", label="frbs")

    cmb_lon = cmb_l if cmb_l < 180 else cmb_l - 360
    ax.scatter(np.radians(cmb_lon), np.radians(cmb_b),
               s=400, marker="*", c="red", edgecolors="black",
               linewidths=2, label="cmb axis", zorder=10)

    frb_lon = dipole_l if dipole_l < 180 else dipole_l - 360
    ax.scatter(np.radians(frb_lon), np.radians(dipole_b),
               s=300, marker="*", c="blue", edgecolors="black",
               linewidths=1.5, label="frb sky dipole", zorder=9)

    ax.grid(alpha=0.3)
    ax.set_title("frb sky distribution and cmb axis (galactic)")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig("frb_axis_diagnostics.png", dpi=200, bbox_inches="tight")
    print("\n   ✓ saved: frb_axis_diagnostics.png")
except Exception as e:
    print("   could not generate visualization:", e)

print("\n" + "=" * 70)
print("analysis complete")
print("=" * 70)
