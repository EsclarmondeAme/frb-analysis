import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft
from astropy.coordinates import SkyCoord
import astropy.units as u

# ============================================================
# PARAMETERS
# ============================================================

UNIFIED_AXIS_L = 159.85
UNIFIED_AXIS_B = -0.51

M_MAX = 12                 # test periodicities m=1..12
AZIMUTH_BINS = 36          # 10° bins
MC_SIMS = 10000            # Monte Carlo significance


# ============================================================
# UTILITY: ROTATE TO AXIS
# ============================================================

def rotate_to_axis(l, b, axis_l, axis_b):
    """
    rotate sky so that the given axis becomes the new north pole.
    returns (theta, phi) in the rotated frame.
    """

    c = SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic')
    a = SkyCoord(l=axis_l*u.deg, b=axis_b*u.deg, frame='galactic')

    # convert to cartesian
    cx, cy, cz = c.cartesian.x.value, c.cartesian.y.value, c.cartesian.z.value
    ax, ay, az = a.cartesian.x.value, a.cartesian.y.value, a.cartesian.z.value

    # rotation vector
    pole = np.array([0.0, 0.0, 1.0])
    v = np.cross([ax, ay, az], pole)
    s = np.linalg.norm(v)
    c = np.dot([ax, ay, az], pole)

    if s == 0:
        R = np.eye(3)
    else:
        vx, vy, vz = v / s
        K = np.array([[0, -vz, vy],
                      [vz, 0, -vx],
                      [-vy, vx, 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))

    XYZ = np.vstack([cx, cy, cz])
    XYZr = R @ XYZ

    Xr, Yr, Zr = XYZr
    theta = np.arccos(Zr)               # polar
    phi = np.arctan2(Yr, Xr)            # azimuth

    return theta, phi


# ============================================================
# FOURIER ANALYSIS
# ============================================================

def compute_m_harmonics(phi, m_max):
    counts, _ = np.histogram(phi, bins=AZIMUTH_BINS, range=(-np.pi, np.pi))
    N = np.sum(counts)

    fft_vals = rfft(counts)
    magnitudes = np.abs(fft_vals)

    m_vals = magnitudes[1:m_max+1] / N
    return m_vals, counts


def monte_carlo_significance(N, m_max, sims=5000):
    P_null = np.zeros((sims, m_max))

    for i in range(sims):
        phi_rand = np.random.uniform(-np.pi, np.pi, N)
        Pm, _ = compute_m_harmonics(phi_rand, m_max)
        P_null[i] = Pm

    mean = np.mean(P_null, axis=0)
    p95 = np.percentile(P_null, 95, axis=0)

    return mean, p95


# ============================================================
# MAIN
# ============================================================

def main():

    print("="*70)
    print("FRB AZIMUTHAL LOBE FINDER — PERIODICITY AROUND UNIFIED AXIS")
    print("="*70)

    df = pd.read_csv("frbs.csv")

    # --------------------------------------------------------
    # convert RA/Dec → Galactic l,b  (this fixes your error)
    # --------------------------------------------------------
    coords = SkyCoord(ra=df['ra'].values*u.deg,
                      dec=df['dec'].values*u.deg,
                      frame='icrs')

    df['l'] = coords.galactic.l.deg
    df['b'] = coords.galactic.b.deg

    # rotate to unified axis
    theta, phi = rotate_to_axis(df['l'].values, df['b'].values,
                                UNIFIED_AXIS_L, UNIFIED_AXIS_B)

    # restrict region
    mask = theta <= np.deg2rad(60)
    phi_sel = phi[mask]

    print(f"total FRBs = {len(df)}")
    print(f"FRBs used (theta ≤ 60°) = {len(phi_sel)}")
    print("")

    # real harmonic powers
    Pm_real, counts = compute_m_harmonics(phi_sel, M_MAX)

    # monte-carlo
    mean_null, p95 = monte_carlo_significance(len(phi_sel), M_MAX, sims=MC_SIMS)

    print("m |   P_m(real)    P_m(null mean)    P_m(95% null)")
    print("----------------------------------------------------")
    for m in range(1, M_MAX+1):
        print(f"{m:2d} | {Pm_real[m-1]:12.6f}   {mean_null[m-1]:12.6f}   {p95[m-1]:12.6f}")

    # find significant m
    sig = np.where(Pm_real > p95)[0] + 1

    print("")
    if len(sig) == 0:
        print("VERDICT: no significant periodicities → structure is smooth.")
    else:
        print(f"Significant m-modes: {sig}")
        print(f"Dominant periodicity: m = {sig[0]}")
        print("")

    # plot
    m_vals = np.arange(1, M_MAX+1)
    plt.figure(figsize=(10,6))
    plt.plot(m_vals, Pm_real, 'o-', label="Real")
    plt.plot(m_vals, mean_null, '--', label="Null mean")
    plt.plot(m_vals, p95, '--', label="Null 95%")
    plt.xlabel("m")
    plt.ylabel("Harmonic power P_m")
    plt.title("FRB Azimuthal Periodicity Around Unified Axis")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("frb_lobe_spectrum.png")

    print("saved: frb_lobe_spectrum.png")
    print("="*70)
    print("analysis complete.")
    print("="*70)


if __name__ == "__main__":
    main()
