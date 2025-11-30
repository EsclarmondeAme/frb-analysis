#!/usr/bin/env python3
# ============================================================
# ASKAP AXIS RECONSTRUCTION FROM POSTERIOR CUTOUT FITS FILES
# ============================================================
# This script:
#   1. Loads ASKAP FRB posterior cutout FITS images
#   2. Converts pixel grid -> sky coordinates using WCS
#   3. Finds MAP (max posterior) localization for each burst
#   4. Computes the preferred axis from ASKAP alone
#   5. Runs Monte Carlo isotropic null (10k trials)
#   6. Outputs scientific verdict
#
# Output files:
#   - askap_axis_results.txt
#   - askap_axis_plot.png
#   - askap_axis_null_hist.png
#
# ============================================================

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_DIR = "."

POSITIONS_FILE = os.path.join(DATA_DIR, "positions.txt")
N_MC = 10000  # Monte Carlo simulations

# unified-FRB axis you previously found
UNIFIED_L = 159.85
UNIFIED_B = -0.51


# ------------------------------------------------------------
# LOAD TRUE ASKAP SKY POSITIONS
# ------------------------------------------------------------
def load_positions(file):
    """
    loads ASKAP-style catalogue with multiple columns.
    automatically detects RA and Dec columns.
    """
    with open(file, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # find header line
    header_line = None
    for ln in lines:
        if any(k.lower() in ln.lower() for k in ["ra", "dec", "col"]):
            header_line = ln
            break

    if header_line is None:
        raise RuntimeError("could not find header line with RA/Dec columns")

    headers = header_line.split()
    print("detected columns:", headers)

    # locate RA/Dec
    ra_idx = None
    dec_idx = None

    for i, h in enumerate(headers):
        hlow = h.lower()
        if hlow.startswith("ra"):
            ra_idx = i
        if hlow.startswith("dec") or hlow.startswith("de"):
            dec_idx = i

    if ra_idx is None or dec_idx is None:
        raise RuntimeError("could not detect RA/Dec column indices")

    # parse rows after header
    data = []
    for ln in lines[lines.index(header_line)+1:]:
        parts = ln.split()
        if len(parts) <= max(ra_idx, dec_idx):
            continue
        ra = float(parts[ra_idx])
        dec = float(parts[dec_idx])
        data.append([ra, dec])

    df = pd.DataFrame(data, columns=["ra", "dec"])
    print(f"loaded {len(df)} positions")
    return df

# ------------------------------------------------------------
# FIND MAP POSITION IN A FITS CUTOUT
# ------------------------------------------------------------
def get_map_from_fits(fname):
    h = fits.open(fname)
    data = h[0].data
    wcs = WCS(h[0].header)

    # find maximum posterior pixel
    iy, ix = np.unravel_index(np.argmax(data), data.shape)

    # convert pixel -> sky coordinates
    sky = wcs.pixel_to_world(ix, iy)
    ra = sky.ra.deg
    dec = sky.dec.deg
    h.close()
    return ra, dec


# ------------------------------------------------------------
# COMPUTE GREAT-CIRCLE SEPARATION
# ------------------------------------------------------------
def angsep(ra1, dec1, ra2, dec2):
    c1 = SkyCoord(ra1*u.deg, dec1*u.deg)
    c2 = SkyCoord(ra2*u.deg, dec2*u.deg)
    return c1.separation(c2).deg


# ------------------------------------------------------------
# FIT ASKAP-ONLY PREFERRED AXIS
# ------------------------------------------------------------
def fit_axis(ra, dec):
    # convert to unit vectors
    v = []
    for r, d in zip(ra, dec):
        r = np.deg2rad(r)
        d = np.deg2rad(d)
        v.append([
            np.cos(d)*np.cos(r),
            np.cos(d)*np.sin(r),
            np.sin(d)
        ])
    v = np.array(v)

    # eigenvector of covariance (largest)
    C = v.T @ v
    w, eig = np.linalg.eig(C)
    axis = eig[:, np.argmax(w)]

    # convert eigenvector -> RA/Dec
    x, y, z = axis
    ra0 = np.rad2deg(np.arctan2(y, x)) % 360
    dec0 = np.rad2deg(np.arcsin(z))
    return ra0, dec0


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    print("===========================================================")
    print("ASKAP AXIS RECONSTRUCTION")
    print("===========================================================\n")

    # load positions to know which FITS exist
    df = load_positions(POSITIONS_FILE)

    ras = []
    decs = []

    print("Processing FITS files...")
    for name in df["name"]:
        fits_file = os.path.join(DATA_DIR, f"{name}_post.fits")
        if not os.path.exists(fits_file):
            print(f"warning: missing {fits_file}")
            continue

        ra, dec = get_map_from_fits(fits_file)
        ras.append(ra)
        decs.append(dec)

    ras = np.array(ras)
    decs = np.array(decs)

    print(f"\nLoaded {len(ras)} ASKAP FRBs")

    # fit axis
    ra0, dec0 = fit_axis(ras, decs)
    print(f"ASKAP preferred axis: RA={ra0:.2f}°, Dec={dec0:.2f}°")

    # convert unified axis RA/Dec
    unified = SkyCoord(l=UNIFIED_L*u.deg, b=UNIFIED_B*u.deg, frame='galactic').icrs
    u_ra = unified.ra.deg
    u_dec = unified.dec.deg

    # separation from FRB unified axis
    sep = angsep(ra0, dec0, u_ra, u_dec)
    print(f"Separation from global unified axis: {sep:.2f}°")

    # --------------------------------------------------------
    # Monte Carlo isotropic null
    # --------------------------------------------------------
    print("\nRunning isotropic null distribution...")

    null_sep = []
    for _ in range(N_MC):
        # isotropic random direction
        z = np.random.uniform(-1, 1)
        phi = np.random.uniform(0, 2*np.pi)
        x = np.sqrt(1 - z*z) * np.cos(phi)
        y = np.sqrt(1 - z*z) * np.sin(phi)
        z = z

        # convert
        ra_r = np.rad2deg(np.arctan2(y, x)) % 360
        dec_r = np.rad2deg(np.arcsin(z))

        null_sep.append(angsep(ra_r, dec_r, u_ra, u_dec))

    null_sep = np.array(null_sep)

    p_value = np.mean(null_sep <= sep)

    print("\n===========================================================")
    print("SCIENTIFIC VERDICT")
    print("===========================================================")
    print(f"ASKAP axis = RA {ra0:.2f}°, Dec {dec0:.2f}°")
    print(f"Separation from FRB unified axis = {sep:.2f}°")
    print(f"Null mean separation = {null_sep.mean():.2f}°")
    print(f"p-value = {p_value:.5f}")

    # save text summary
    with open("askap_axis_results.txt", "w") as f:
        f.write("ASKAP AXIS RECONSTRUCTION RESULTS\n")
        f.write("---------------------------------\n")
        f.write(f"ASKAP preferred axis: RA={ra0:.2f}, Dec={dec0:.2f}\n")
        f.write(f"Separation from unified axis: {sep:.2f}°\n")
        f.write(f"Null mean: {null_sep.mean():.2f}°\n")
        f.write(f"p-value: {p_value:.6f}\n")

    # plot
    plt.figure(figsize=(7,5))
    plt.hist(null_sep, bins=40, alpha=0.7)
    plt.axvline(sep, color='r', lw=2, label=f"observed = {sep:.2f}°")
    plt.xlabel("max separation from unified axis (MC isotropy)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("askap_axis_null_hist.png")
    plt.close()

    print("\nSaved: askap_axis_results.txt")
    print("Saved: askap_axis_null_hist.png")
    print("\nAnalysis complete.")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
