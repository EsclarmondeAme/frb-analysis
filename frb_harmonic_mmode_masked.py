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

    cos_theta = np.sin(b0)*np.sin(b) + np.cos(b0)*np.cos(b)*np.cos(l - l0)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    x = np.cos(b)*np.cos(l - l0)
    y = np.cos(b)*np.sin(l - l0)
    phi = np.arctan2(y, x)

    return theta, phi, coords

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

def masked_sample(df, mask_type="galactic", cut=20.0):
    coords = SkyCoord(df["ra"].values*u.deg, df["dec"].values*u.deg, frame="icrs")

    if mask_type == "galactic":
        lat = coords.galactic.b.deg

    elif mask_type == "ecliptic":
        lat = coords.barycentrictrueecliptic.lat.deg

    elif mask_type == "supergalactic":
        lat = coords.supergalactic.sgb.deg   # <-- FIXED HERE

    else:
        raise ValueError("unknown mask_type")

    keep = np.abs(lat) >= cut
    return df[keep]

def run_mask_case(df, label, suffix):
    theta, phi, _ = compute_axis_frame(df["ra"].values, df["dec"].values)
    alm = compute_alm(theta, phi, LMAX)
    amps = mmode_amplitudes(alm, LMAX)
    print_amps(label, amps)

    # optional: save bar plots
    for l, arr in amps.items():
        ms   = [x[0] for x in arr]
        mags = [x[1] for x in arr]
        plt.figure(figsize=(6,4))
        plt.bar(ms, mags)
        plt.xlabel("m")
        plt.ylabel("|a_ℓm|")
        plt.title(f"{label}, ℓ={l}")
        plt.tight_layout()
        fname = f"frb_mmode_{suffix}_l{l}.png"
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"saved: {fname}")

def main():
    df = pd.read_csv("frbs.csv").dropna(subset=["ra","dec"])

    # full (no mask) just as reference
    theta_all, phi_all, _ = compute_axis_frame(df["ra"].values, df["dec"].values)
    alm_all = compute_alm(theta_all, phi_all, LMAX)
    amps_all = mmode_amplitudes(alm_all, LMAX)
    print_amps("NO MASK (reference)", amps_all)

    # galactic mask
    df_gal = masked_sample(df, "galactic", cut=20.0)
    print(f"galactic mask |b|>20°: kept {len(df_gal)} of {len(df)} frbs")
    run_mask_case(df_gal, "GALACTIC MASK |b|>20°", "galcut20")

    # ecliptic mask
    df_ecl = masked_sample(df, "ecliptic", cut=20.0)
    print(f"ecliptic mask |beta|>20°: kept {len(df_ecl)} of {len(df)} frbs")
    run_mask_case(df_ecl, "ECLIPTIC MASK |β|>20°", "eclcut20")

    # supergalactic mask
    df_sg = masked_sample(df, "supergalactic", cut=20.0)
    print(f"supergalactic mask |SGB|>20°: kept {len(df_sg)} of {len(df)} frbs")
    run_mask_case(df_sg, "SUPERGALACTIC MASK |SGB|>20°", "sgcut20")

    print("analysis complete.")

if __name__ == "__main__":
    main()
