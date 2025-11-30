import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit

# ------------------------------------------------------------
# unified axis parameters (galactic)
# ------------------------------------------------------------
UNIFIED_L = 159.85
UNIFIED_B = -0.51

# ------------------------------------------------------------
# model definitions
# ------------------------------------------------------------

def phi_only_model(phi, A0, A1, phi0):
    """m=1 model: A0 + A1*cos(phi - phi0)."""
    return A0 + A1 * np.cos(phi - phi0)

def phi_m1m2_model(phi, A0, A1, A2, phi0):
    """m=1 + m=2 model: A0 + A1*cos(phi-phi0) + A2*cos(2*(phi-phi0))."""
    return A0 + A1*np.cos(phi - phi0) + A2*np.cos(2*(phi - phi0))

def pure_radial(phi, A0):
    """phi-independent baseline."""
    return A0 + 0*phi

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def compute_axis_frame_angles(ra_deg, dec_deg, l0=UNIFIED_L, b0=UNIFIED_B):
    """robust RA/Dec → (theta, phi) relative to unified axis."""

    # force clean float arrays
    ra_arr = np.asarray(ra_deg, dtype=float)
    dec_arr = np.asarray(dec_deg, dtype=float)

    coords = SkyCoord(ra=ra_arr*u.deg, dec=dec_arr*u.deg, frame="icrs")
    gal = coords.galactic

    l = np.radians(gal.l.value)
    b = np.radians(gal.b.value)

    l0 = np.radians(l0)
    b0 = np.radians(b0)

    # theta = angular distance to unified axis
    cos_theta = np.sin(b0)*np.sin(b) + np.cos(b0)*np.cos(b)*np.cos(l - l0)
    theta = np.arccos(np.clip(cos_theta, -1, 1))

    # φ = azimuth around unified axis
    x = np.cos(b) * np.cos(l - l0)
    y = np.cos(b) * np.sin(l - l0)
    phi = np.arctan2(y, x)

    return theta, phi



def AIC_from_RSS(n, rss, k):
    """Akaike information criterion."""
    return 2*k + n*np.log(rss/n)

def mc_p_value(delta_real, delta_null):
    return np.mean(delta_null >= delta_real)

# ------------------------------------------------------------
# fitting routine
# ------------------------------------------------------------
def fit_models(phi, counts, name="ALL"):
    n = len(phi)

    # ensure float
    phi = np.asarray(phi, float)
    counts = np.asarray(counts, float)

    # PURE RADIAL
    pop_r, _ = curve_fit(pure_radial, phi, counts,
                         p0=[np.mean(counts)], maxfev=20000)
    rss_r = np.sum((counts - pure_radial(phi, *pop_r))**2)
    aic_r = AIC_from_RSS(n, rss_r, k=1)

    # M = 1
    try:
        pop_m1, _ = curve_fit(phi_only_model, phi, counts,
                              p0=[np.mean(counts), 0.1, 0.0],
                              maxfev=20000)
        rss_m1 = np.sum((counts - phi_only_model(phi, *pop_m1))**2)
        aic_m1 = AIC_from_RSS(n, rss_m1, k=3)
    except:
        pop_m1 = None
        rss_m1 = np.inf
        aic_m1 = np.inf

    # M = 1 + M = 2
    try:
        pop_m12, _ = curve_fit(phi_m1m2_model, phi, counts,
                               p0=[np.mean(counts), 0.1, 0.1, 0.0],
                               maxfev=20000)
        rss_m12 = np.sum((counts - phi_m1m2_model(phi, *pop_m12))**2)
        aic_m12 = AIC_from_RSS(n, rss_m12, k=4)
    except:
        pop_m12 = None
        rss_m12 = np.inf
        aic_m12 = np.inf

    # Monte Carlo significance (robust)
    delta_real = aic_r - aic_m12
    delta_null = []

    for _ in range(2000):
        shuf = np.random.permutation(counts)
        try:
            pop_s, _ = curve_fit(phi_m1m2_model, phi, shuf,
                                 p0=[np.mean(shuf), 0.1, 0.1, 0.0],
                                 maxfev=20000)
            rss_s = np.sum((shuf - phi_m1m2_model(phi, *pop_s))**2)
            aic_s = AIC_from_RSS(n, rss_s, k=4)
        except:
            aic_s = aic_r  # fallback
        delta_null.append(aic_r - aic_s)

    p_mc = mc_p_value(delta_real, np.array(delta_null))

    print("=======================================================")
    print(f"  LOPSIDED SHELL SHAPE FIT — {name}")
    print("=======================================================")
    print(f"pure radial:   AIC={aic_r:.2f}   RSS={rss_r:.2f}")
    print(f"m=1 model:     AIC={aic_m1:.2f}   RSS={rss_m1:.2f}")
    print(f"m=1+m=2 model: AIC={aic_m12:.2f}  RSS={rss_m12:.2f}")
    print("-------------------------------------------------------")

    best = min(aic_r, aic_m1, aic_m12)
    if best == aic_m12:
        print("BEST MODEL: m=1 + m=2 (strong lopsidedness)")
    elif best == aic_m1:
        print("BEST MODEL: m=1 (single lobe)")
    else:
        print("BEST MODEL: pure radial (no φ structure)")

    print("-------------------------------------------------------")
    print(f"MC significance (m1+m2 vs pure radial): p = {p_mc:.4g}")
    print()



# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    df = pd.read_csv("frbs.csv")
    df = df.dropna(subset=["z_est"])

    # unified-axis angles
    theta, phi = compute_axis_frame_angles(df["ra"], df["dec"])

    # bin by theta to get radial shell (25–60 deg)
    mask_shell = (theta >= np.radians(25)) & (theta <= np.radians(60))

    phi_shell = phi[mask_shell]
    z_shell = df["z_est"].values[mask_shell]

    # bin φ into 36 bins
    bins = np.linspace(-np.pi, np.pi, 37)
    centers = 0.5*(bins[:-1] + bins[1:])
    counts_all, _ = np.histogram(phi_shell, bins=bins)

    # redshift split
    z_med = np.median(z_shell)
    mask_low = z_shell <= z_med
    mask_high = z_shell > z_med

    counts_low, _ = np.histogram(phi_shell[mask_low], bins=bins)
    counts_high, _ = np.histogram(phi_shell[mask_high], bins=bins)

    # run fits
    fit_models(centers, counts_all,  name="ALL")
    fit_models(centers, counts_low,  name="LOW-Z")
    fit_models(centers, counts_high, name="HIGH-Z")


if __name__ == "__main__":
    main()
