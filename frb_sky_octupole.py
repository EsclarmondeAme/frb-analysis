"""
FRB Sky Octupole (ℓ = 3) Analysis
---------------------------------
Fits spherical-harmonic octupole moments to the FRB sky distribution.

Inputs:
    - frbs.csv   (columns: ra, dec)

Outputs:
    - frb_sky_octupole.png       (Mollweide map showing octupole axis)
    - printed octupole amplitude and Monte Carlo significance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

# -----------------------------
# load data
# -----------------------------
def load_frbs():
    df = pd.read_csv("frbs.csv")
    df = df.dropna(subset=["ra", "dec"])
    return df

# -----------------------------
# spherical harmonics Y_lm (real)
# -----------------------------
def real_Y3m(theta, phi):
    """
    θ = colatitude (0 at north pole)
    φ = longitude

    returns dictionary:
        Y3m[m] for m = -3 .. +3
    """

    # compute basic angles
    ct = np.cos(theta)
    st = np.sin(theta)

    # Legendre pieces
    # real spherical harmonics normalized (not Condon-Shortley sign)
    Y = {}

    # m = 0
    Y[0] = 0.25 * np.sqrt(7/np.pi) * (5*ct**3 - 3*ct)

    # m = 1
    Y[1]  = -0.5 * np.sqrt(21/(2*np.pi)) * st * (5*ct**2 - 1) * np.cos(phi)
    Y[-1] = -0.5 * np.sqrt(21/(2*np.pi)) * st * (5*ct**2 - 1) * np.sin(phi)

    # m = 2
    Y[2]  = 0.25 * np.sqrt(105/np.pi) * st**2 * (5*ct) * np.cos(2*phi)
    Y[-2] = 0.25 * np.sqrt(105/np.pi) * st**2 * (5*ct) * np.sin(2*phi)

    # m = 3
    Y[3]  = -0.125 * np.sqrt(35/(2*np.pi)) * st**3 * np.cos(3*phi)
    Y[-3] = -0.125 * np.sqrt(35/(2*np.pi)) * st**3 * np.sin(3*phi)

    return Y

# -----------------------------
# compute octupole coefficients
# -----------------------------
def fit_octupole(ra_deg, dec_deg):

    # convert to spherical
    phi = np.radians(ra_deg)
    theta = np.radians(90 - dec_deg)

    # compute Y_3m for each FRB
    Y = [ real_Y3m(t, p) for t, p in zip(theta, phi) ]

    # collect coefficients a_3m = sum(Y_3m)
    a3m = {m: np.sum([y[m] for y in Y]) for m in range(-3,4)}

    # octupole amplitude = sqrt(sum(a_3m^2))
    amp = np.sqrt(np.sum([a3m[m]**2 for m in a3m]))

    return a3m, amp

# -----------------------------
# find octupole axis (direction of maximum)
# -----------------------------
def find_octupole_axis(a3m):

    # brute grid search
    lon = np.linspace(0, 360, 720)
    lat = np.linspace(-90, 90, 360)

    best_amp = -1
    best_dir = (0,0)

    for L in lon:
        for B in lat:
            t = np.radians(90 - B)
            p = np.radians(L)
            Y = real_Y3m(t,p)

            val = sum(a3m[m] * Y[m] for m in a3m)
            if val > best_amp:
                best_amp = val
                best_dir = (L, B)

    return best_dir, best_amp

# -----------------------------
# Monte Carlo significance
# -----------------------------
def mc_significance(N, amp_obs, nmc=5000):
    amps = []
    for _ in range(nmc):
        # generate uniform directions
        phi = np.random.uniform(0, 2*np.pi, N)
        z = np.random.uniform(-1, 1, N)
        theta = np.arccos(z)

        a3m = {m:0 for m in range(-3,4)}
        for t,p in zip(theta,phi):
            Y = real_Y3m(t,p)
            for m in a3m:
                a3m[m] += Y[m]

        amps.append(np.sqrt(sum(a3m[m]**2 for m in a3m)))

    amps = np.array(amps)
    p = np.mean(amps >= amp_obs)
    return p

# -----------------------------
# main
# -----------------------------
def main():
    print("="*60)
    print("FRB SKY OCTUPOLE ANALYSIS (ℓ = 3)")
    print("="*60)

    df = load_frbs()
    ra = df["ra"].values
    dec = df["dec"].values

    print(f"Loaded FRBs: {len(df)}")

    # fit ℓ=3
    a3m, amp3 = fit_octupole(ra,dec)
    print("\nOctupole amplitude:", amp3)

    # direction of max
    (L,B), proj_amp = find_octupole_axis(a3m)
    print(f"Octupole axis: RA≈{L:.2f}°, Dec≈{B:.2f}°")

    # MC
    print("\nComputing Monte Carlo significance…")
    p = mc_significance(len(df), amp3, nmc=3000)
    print(f"Monte Carlo p-value: {p:.5f}")

    # ------------------ plot ------------------
    coords = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")
    lon = coords.galactic.l.wrap_at(180*u.deg).radian
    lat = coords.galactic.b.radian

    plt.figure(figsize=(12,6))
    ax = plt.subplot(111, projection="mollweide")
    ax.scatter(lon,lat,s=8,alpha=0.5,color="steelblue")

    # plot octupole axis
    ax.scatter(
        np.radians(L if L<=180 else L-360),
        np.radians(B),
        s=150, marker="^", color="red", label="Octupole axis"
    )

    ax.grid(True)
    ax.legend(loc="lower left")
    plt.title("FRB Sky Octupole (ℓ = 3)")
    plt.savefig("frb_sky_octupole.png", dpi=150)
    print("\nSaved → frb_sky_octupole.png")

    print("\nDone.")
    print("="*60)

if __name__ == "__main__":
    main()
