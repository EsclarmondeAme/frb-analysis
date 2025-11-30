import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

# =====================================================
# unified axis (galactic coordinates)
# =====================================================
UNIFIED_L = 159.85   # deg
UNIFIED_B = -0.51    # deg

axis_coord = SkyCoord(l=UNIFIED_L*u.deg, b=UNIFIED_B*u.deg, frame="galactic")

# =====================================================
# compute (theta, phi) in axis-aligned coordinates
# =====================================================
def compute_axis_angles(ra, dec):
    # convert from pandas series → numpy arrays (required by astropy 6.x)
    ra_deg = np.asarray(ra, dtype=float)
    dec_deg = np.asarray(dec, dtype=float)

    coords = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
    gal = coords.galactic

    # rotate so unified axis becomes the new north pole
    offset_frame = SkyCoord(l=UNIFIED_L*u.deg, b=UNIFIED_B*u.deg,
                            frame="galactic").skyoffset_frame()
    rot = gal.transform_to(offset_frame)

    theta = 90.0 - rot.lat.deg     # polar angle from axis
    phi   = rot.lon.deg % 360.0    # azimuth
    return theta, phi

# =====================================================
# build 2d histogram & residual map
# =====================================================
def residual_map(theta, phi, name):
    nb_th = 36
    nb_ph = 72

    H, th_edges, ph_edges = np.histogram2d(theta, phi, 
                                           bins=[nb_th, nb_ph],
                                           range=[[0,90],[0,360]])

    # isotropic expectation
    # uniform in phi, proportional to sin(theta)
    th_centers = 0.5*(th_edges[:-1] + th_edges[1:])
    iso = np.zeros_like(H)
    for i, th in enumerate(th_centers):
        sinw = np.sin(np.radians(th + (90/nb_th)/2)) - np.sin(np.radians(th - (90/nb_th)/2))
        iso[i, :] = sinw
    iso = iso / iso.sum() * H.sum()

    R = (H - iso) / np.sqrt(iso + 1e-9)

    plt.figure(figsize=(10,6))
    plt.imshow(R, origin='lower', aspect='auto',
               extent=[0,360,0,90],
               cmap='coolwarm', vmin=-5, vmax=5)
    plt.colorbar(label='(observed - iso)/sqrt(iso)')
    plt.xlabel('phi (deg)')
    plt.ylabel('theta (deg)')
    plt.title(f"FRB 2D residual map — {name}")
    plt.savefig(f"frb_residual_map_2d_{name}.png", dpi=200)
    plt.close()

# =====================================================
# main
# =====================================================
def main():
    df = pd.read_csv("frbs.csv")
    df = df.dropna(subset=["ra","dec","z_est"])

    # split
    z_med = np.median(df["z_est"].values)
    df_low = df[df["z_est"] <= z_med]
    df_high = df[df["z_est"] > z_med]

    print("computing axis angles...")
    theta_all, phi_all   = compute_axis_angles(df["ra"], df["dec"])
    theta_low, phi_low   = compute_axis_angles(df_low["ra"], df_low["dec"])
    theta_high, phi_high = compute_axis_angles(df_high["ra"], df_high["dec"])

    print("building residual maps...")
    residual_map(theta_all,  phi_all,  "ALL")
    residual_map(theta_low,  phi_low,  "LOWZ")
    residual_map(theta_high, phi_high, "HIGHZ")

    print("saved maps: frb_residual_map_2d_ALL.png, LOWZ.png, HIGHZ.png")
    print("analysis complete.")

if __name__ == "__main__":
    main()
