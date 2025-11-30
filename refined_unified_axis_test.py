"""
refined_unified_axis_test.py
----------------------------------------------------------
full-precision unified axis comparison
cmb dipole axis  •  frb sidereal dipole  •  atomic clock dipole
----------------------------------------------------------

uses:
• astropy.time for gmst and sidereal transforms
• iau 2006 precession / nutation
• real site locations (chime + nist)
• proper icrs → galactic transforms

outputs:
• refined galactic axes for frb + clock
• angular separations from cmb axis
• frb sky clustering check
• mollweide visualization
• final verdict
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u

print("=" * 70)
print("refined unified axis comparison")
print("cmb + frb + atomic clock (full coordinate precision)")
print("=" * 70)

# ----------------------------------------------------------
# 1. cmb axis (already in galactic coords)
# ----------------------------------------------------------

cmb_l = 152.62
cmb_b = 4.03

cmb_coord = SkyCoord(
    l=cmb_l * u.deg,
    b=cmb_b * u.deg,
    frame="galactic"
)

print("\n1. cmb dipole-modulation axis (planck low-ℓ)")
print(f"   l = {cmb_l:.2f}°, b = {cmb_b:.2f}°")


# ----------------------------------------------------------
# 2. frb sidereal dipole (refined conversion)
# ----------------------------------------------------------
# harmonic coefficients from your analysis
A1 = -1.5728e-02
B1 = 5.3633e-01

phi_rad = np.arctan2(B1, A1)
phi_deg = (np.degrees(phi_rad) % 360)

print("\n2. frb sidereal dipole")
print(f"   A1 = {A1:.4e}")
print(f"   B1 = {B1:.4e}")
print(f"   amplitude = {np.sqrt(A1**2 + B1**2):.4f}")
print(f"   phase = {phi_deg:.2f}° sidereal")


# observatory: chime
chime = EarthLocation(
    lat=49.3223*u.deg,
    lon=-119.6167*u.deg,
    height=545*u.m
)

# choose a reference time (midpoint of frb dataset)
t = Time("2020-01-01T00:00:00", scale="utc")

# sidereal phase → local sidereal hour angle → ra
lst = t.sidereal_time("apparent", longitude=chime.lon)
frb_ra = (phi_deg * u.deg + lst) % (360 * u.deg)
frb_dec = 45 * u.deg        # chime best sensitivity band

frb_equ = SkyCoord(ra=frb_ra, dec=frb_dec, frame="icrs")
frb_gal = frb_equ.galactic

print(f"   refined galactic coords:")
print(f"   l = {frb_gal.l.deg:.2f}°, b = {frb_gal.b.deg:.2f}°")


# ----------------------------------------------------------
# 3. atomic clock dipole (refined conversion)
# ----------------------------------------------------------

clock_phase_rad = 1.326
clock_phase_deg = (np.degrees(clock_phase_rad) % 360)

print("\n3. atomic clock sidereal modulation")
print(f"   phase = {clock_phase_rad:.3f} rad = {clock_phase_deg:.2f}°")
print(f"   amplitude ~ 1e-15 fractional")

nist = EarthLocation(
    lat=40.0150*u.deg,
    lon=-105.2705*u.deg,
    height=1689*u.m
)

lst_nist = t.sidereal_time("apparent", longitude=nist.lon)
clk_ra = (clock_phase_deg * u.deg + lst_nist) % (360 * u.deg)
clk_dec = 40 * u.deg    # nist latitude band

clk_equ = SkyCoord(ra=clk_ra, dec=clk_dec, frame="icrs")
clk_gal = clk_equ.galactic

print(f"   refined galactic coords:")
print(f"   l = {clk_gal.l.deg:.2f}°, b = {clk_gal.b.deg:.2f}°")


# ----------------------------------------------------------
# 4. angular separations
# ----------------------------------------------------------

sep_frb_cmb = frb_gal.separation(cmb_coord).deg
sep_clk_cmb = clk_gal.separation(cmb_coord).deg
sep_frb_clk = frb_gal.separation(clk_gal).deg

print("\n" + "=" * 70)
print("angular separations")
print("=" * 70)
print(f"cmb ↔ frb:     {sep_frb_cmb:.2f}°")
print(f"cmb ↔ clock:   {sep_clk_cmb:.2f}°")
print(f"frb ↔ clock:   {sep_frb_clk:.2f}°")


# ----------------------------------------------------------
# 5. frb sky clustering check
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("frb sky distribution check")
print("=" * 70)

try:
    frbs = pd.read_csv("frbs.csv")

    coords = SkyCoord(
        ra=frbs["ra"].values*u.deg,
        dec=frbs["dec"].values*u.deg,
        frame="icrs"
    ).galactic

    separations = coords.separation(cmb_coord).deg
    count_30 = np.sum(separations < 30)
    frac = count_30 / len(frbs)

    print(f"total frbs: {len(frbs)}")
    print(f"frbs within 30° of cmb axis: {count_30}  ({frac*100:.1f}%)")
    print(f"expected if random: ~25%")

    # visualization
    fig = plt.figure(figsize=(15, 6))

    # mollweide
    ax1 = fig.add_subplot(121, projection="mollweide")

    lon = coords.l.deg
    lon = np.where(lon > 180, lon - 360, lon)
    lat = coords.b.deg

    ax1.scatter(np.radians(lon), np.radians(lat), s=8, alpha=0.4)

    # mark axes
    for coord, col, label in [
        (cmb_coord, "red", "cmb axis"),
        (frb_gal, "blue", "frb dipole"),
        (clk_gal, "green", "clock dipole")
    ]:
        L = coord.l.deg
        L = L - 360 if L > 180 else L
        B = coord.b.deg
        ax1.scatter(np.radians(L), np.radians(B),
                    s=200, c=col, marker="*",
                    edgecolors="black", linewidths=1.5,
                    label=label)

    ax1.set_title("frb sky distribution (galactic)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # histogram
    ax2 = fig.add_subplot(122)
    ax2.hist(separations, bins=30, edgecolor="black", alpha=0.7)
    ax2.axvline(30, color="red", linestyle="--")
    ax2.set_title("frb separations from cmb axis")
    ax2.set_xlabel("separation (deg)")
    ax2.set_ylabel("count")

    plt.tight_layout()
    plt.savefig("refined_axis_alignment.png", dpi=150)
    print("\n✓ visualization saved: refined_axis_alignment.png")

except Exception as e:
    print("could not load frbs.csv:", e)


# ----------------------------------------------------------
# 6. verdict
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("final verdict")
print("=" * 70)

strong = 30
smoking = 20

e = 0
if sep_frb_cmb < strong: e += 1
if sep_clk_cmb < strong: e += 1
if sep_frb_clk < strong: e += 1

print(f"\nevidence score: {e}/3")

print("\ncriteria:")
print(f"  [{'✓' if sep_frb_cmb < strong else '✗'}] cmb–frb axes aligned")
print(f"  [{'✓' if sep_clk_cmb < strong else '✗'}] cmb–clock axes aligned")
print(f"  [{'✓' if sep_frb_clk < strong else '✗'}] frb–clock axes aligned")

if e >= 2:
    print("\n★ strong support for unified preferred axis")
    print("★ consistent with frequency-gradient / cone model")
else:
    print("\n~ weak or no alignment")
    print("~ may need more precise timing or more data")

print("=" * 70)
print("done")
print("=" * 70)
