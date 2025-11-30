import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.special import sph_harm

# ============================================================
# coordinate tools
# ============================================================

def to_galactic(df):
    """Convert RA/DEC to galactic l,b using the standard rotation."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    c = SkyCoord(ra=df['ra'].values*u.deg, dec=df['dec'].values*u.deg, frame='icrs')
    df['l'] = c.galactic.l.deg
    df['b'] = c.galactic.b.deg
    return df

def rotate_to_axis(l, b, axis_l, axis_b):
    """Rotate (l,b) so axis_l,axis_b becomes new north pole."""
    # convert to cartesian
    def sph2cart(l_deg, b_deg):
        l_rad = np.deg2rad(l_deg)
        b_rad = np.deg2rad(b_deg)
        x = np.cos(b_rad)*np.cos(l_rad)
        y = np.cos(b_rad)*np.sin(l_rad)
        z = np.sin(b_rad)
        return np.vstack([x,y,z]).T

    def cart2sph(x,y,z):
        hxy = np.hypot(x,y)
        theta = np.arctan2(hxy, z)        # colatitude
        phi = np.arctan2(y, x)
        return theta, phi

    v = sph2cart(l,b)
    ax = sph2cart(axis_l, axis_b)[0]

    # rotation matrix: rotate axis -> z
    z = ax / np.linalg.norm(ax)
    # choose arbitrary perpendicular
    tmp = np.array([1,0,0], dtype=float)
    if np.allclose(z, tmp): tmp = np.array([0,1,0], dtype=float)
    x = np.cross(tmp, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)

    R = np.vstack([x,y,z])
    vr = v @ R.T

    theta, phi = cart2sph(vr[:,0], vr[:,1], vr[:,2])
    return np.rad2deg(theta), np.rad2deg(phi)


# ============================================================
# main
# ============================================================

def main():

    # unified axis you used everywhere
    AXIS_L = 159.85
    AXIS_B = -0.51

    # --------------------------------------------------------
    # load data
    # --------------------------------------------------------
    df = pd.read_csv("frbs.csv")
    df = to_galactic(df)

    # rotate to axis
    theta, phi = rotate_to_axis(df['l'].values, df['b'].values, AXIS_L, AXIS_B)

    # normalize phi to 0–360
    phi = (phi + 360) % 360
    df['theta'] = theta
    df['phi'] = phi

    # we only use θ ≤ 60° (inner zone)
    mask = df['theta'] <= 60
    d = df[mask]

    # --------------------------------------------------------
    # fine φ binning
    # --------------------------------------------------------
    nbins = 360 // 5   # 5° bins → 72 bins
    bins = np.linspace(0, 360, nbins+1)
    hist, _ = np.histogram(d['phi'], bins=bins)

    # smooth it
    smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)

    # --------------------------------------------------------
    # peak finding
    # --------------------------------------------------------
    peaks, props = find_peaks(smooth, prominence=np.mean(smooth)*0.25)
    peak_phis = (bins[peaks] + bins[peaks+1]) / 2

    # --------------------------------------------------------
    # Monte Carlo: significance of number of peaks
    # --------------------------------------------------------
    Nsim = 5000
    n_peaks_null = []

    N = len(d)
    for _ in range(Nsim):
        fake_phi = np.random.uniform(0,360,N)
        fake_hist,_ = np.histogram(fake_phi, bins=bins)
        fake_smooth = gaussian_filter1d(fake_hist.astype(float), sigma=2.0)
        fake_peaks,_ = find_peaks(fake_smooth, prominence=np.mean(fake_smooth)*0.25)
        n_peaks_null.append(len(fake_peaks))

    n_peaks_null = np.array(n_peaks_null)
    real_n = len(peaks)

    p_value = np.mean(n_peaks_null >= real_n)

    # --------------------------------------------------------
    # plot
    # --------------------------------------------------------
    plt.figure(figsize=(12,6))
    centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(centers, smooth, '-', lw=2, label='smoothed φ-profile')
    plt.scatter(peak_phis, smooth[peaks], color='red', zorder=5, label='peaks')
    plt.xlabel("φ (deg)")
    plt.ylabel("smoothed density")
    plt.title("FRB azimuthal structure around unified axis")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig("frb_lobe_peaks.png", dpi=150)
    print("saved: frb_lobe_peaks.png")

    # --------------------------------------------------------
    # scientific verdict
    # --------------------------------------------------------
    print("=======================================================================")
    print(" FRB AZIMUTHAL LOBE STRUCTURE — SCIENTIFIC VERDICT")
    print("=======================================================================")
    print(f"FRBs used (θ ≤ 60°): {N}")
    print(f"Detected azimuthal peaks: {real_n}")
    print(f"Peak longitudes (deg): {np.round(peak_phis,1)}")
    print("-----------------------------------------------------------------------")
    print("Monte Carlo isotropic null (5,000 realizations)")
    print(f"  mean peaks(null) = {np.mean(n_peaks_null):.2f}")
    print(f"  95% peaks(null) = {np.percentile(n_peaks_null,95):.1f}")
    print(f"  p-value = {p_value:.5f}")
    print("-----------------------------------------------------------------------")

    if p_value < 0.001:
        verdict = "very strong evidence for non-uniform azimuthal structure"
    elif p_value < 0.01:
        verdict = "strong evidence for structured azimuthal modulation"
    elif p_value < 0.05:
        verdict = "moderate evidence for azimuthal deviation from uniformity"
    else:
        verdict = "no significant deviation from azimuthal isotropy"

    print(f"VERDICT: {verdict}.")
    print("Interpretation:")
    if real_n == 1:
        print(" → a single broad lobe dominates the azimuthal structure.")
    elif real_n == 2:
        print(" → a bilobed pattern is preferred (m = 2 symmetry).")
    elif real_n == 3:
        print(" → a tri-lobed structure appears around the unified axis.")
    elif real_n >= 4:
        print(" → multiple azimuthal lobes detected, suggesting faceted or")
        print("   pyramid-like angular structure rather than pure axisymmetry.")
    print("=======================================================================")
    print("analysis complete.")
    print("=======================================================================")



if __name__ == "__main__":
    main()
