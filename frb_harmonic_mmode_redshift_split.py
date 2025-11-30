import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y
from astropy.coordinates import SkyCoord
import astropy.units as u

UNIFIED_L = 159.85
UNIFIED_B = -0.51
LMAX = 4

def compute_axis_frame(ra_deg, dec_deg):
    coords = SkyCoord(ra_deg*u.deg, dec_deg*u.deg, frame="icrs")
    gal = coords.galactic

    l = gal.l.radian
    b = gal.b.radian

    l0 = np.deg2rad(UNIFIED_L)
    b0 = np.deg2rad(UNIFIED_B)

    # angle from axis
    cos_theta = np.sin(b0)*np.sin(b) + np.cos(b0)*np.cos(b)*np.cos(l - l0)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # azimuth around axis (simple rotation in galactic plane)
    x = np.cos(b)*np.cos(l - l0)
    y = np.cos(b)*np.sin(l - l0)
    phi = np.arctan2(y, x)

    return theta, phi

def compute_alm(theta, phi, lmax):
    alm = {}
    for l in range(lmax+1):
        for m in range(-l, l+1):
            Ylm = sph_harm_y(m, l, phi, theta)
            alm[(l,m)] = np.sum(Ylm)
    return alm

def mmode_amplitudes(alm, lmax):
    amps = {}
    for l in range(1, lmax+1):
        arr = []
        for m in range(-l, l+1):
            arr.append((m, np.abs(alm[(l,m)])))
        amps[l] = arr
    return amps

def print_amps(label, amps):
    print(f"================ {label} ================")
    for l, arr in amps.items():
        mags = [x[1] for x in arr]
        ms   = [x[0] for x in arr]
        m_dom = ms[int(np.argmax(mags))]
        print(f"ℓ={l}: dominant m = {m_dom},  amplitudes:")
        for m,a in arr:
            print(f"  m={m:2d}: |a_ℓm| = {a:.4e}")
        print()

def plot_amps(label, amps, suffix):
    for l, arr in amps.items():
        ms = [x[0] for x in arr]
        mags = [x[1] for x in arr]
        plt.figure(figsize=(6,4))
        plt.bar(ms, mags)
        plt.xlabel("m")
        plt.ylabel("|a_ℓm|")
        plt.title(f"{label}: ℓ={l}")
        plt.tight_layout()
        fname = f"frb_mmode_{suffix}_l{l}.png"
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"saved: {fname}")

def main():
    df = pd.read_csv("frbs.csv")
    df = df.dropna(subset=["ra","dec","z_est"])

    # sort by redshift and split
    df_sorted = df.sort_values("z_est")
    mid = len(df_sorted)//2
    df_low  = df_sorted.iloc[:mid]
    df_high = df_sorted.iloc[mid:]

    # full sample
    theta_all, phi_all = compute_axis_frame(df["ra"].values, df["dec"].values)
    alm_all = compute_alm(theta_all, phi_all, LMAX)
    amps_all = mmode_amplitudes(alm_all, LMAX)
    print_amps("FULL SAMPLE", amps_all)
    plot_amps("full sample", amps_all, "all")

    # low-z
    theta_low, phi_low = compute_axis_frame(df_low["ra"].values, df_low["dec"].values)
    alm_low = compute_alm(theta_low, phi_low, LMAX)
    amps_low = mmode_amplitudes(alm_low, LMAX)
    print_amps("LOW-Z HALF", amps_low)
    plot_amps("low-z half", amps_low, "lowz")

    # high-z
    theta_high, phi_high = compute_axis_frame(df_high["ra"].values, df_high["dec"].values)
    alm_high = compute_alm(theta_high, phi_high, LMAX)
    amps_high = mmode_amplitudes(alm_high, LMAX)
    print_amps("HIGH-Z HALF", amps_high)
    plot_amps("high-z half", amps_high, "highz")

    print("analysis complete.")

if __name__ == "__main__":
    main()
