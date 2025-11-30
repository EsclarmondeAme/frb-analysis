import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

UNIFIED_L = 159.85
UNIFIED_B = -0.51

# =====================================================
# utilities
# =====================================================

def angdist(l1, b1, l2, b2):
    """great-circle separation, all angles in deg."""
    l1 = np.deg2rad(l1)
    b1 = np.deg2rad(b1)
    l2 = np.deg2rad(l2)
    b2 = np.deg2rad(b2)
    cosang = np.sin(b1)*np.sin(b2) + np.cos(b1)*np.cos(b2)*np.cos(l1 - l2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosang))


def broken_power(theta, tbreak, A1, alpha1, A2, alpha2):
    """
    two-branch broken power law:
      f(theta) = A1 * theta^-alpha1   for theta < tbreak
               = A2 * theta^-alpha2   for theta >= tbreak
    """
    theta = np.maximum(theta, 1e-3)
    out = np.zeros_like(theta)
    mask = theta < tbreak
    out[mask]  = A1 * theta[mask] ** (-alpha1)
    out[~mask] = A2 * theta[~mask] ** (-alpha2)
    return out


def fit_broken_power(theta_centers, y):
    """robust fit with curve_fit + bounded fallback."""
    p0 = [25.0, 0.5, 0.3, 5.0, 0.1]  # tbreak, A1, alpha1, A2, alpha2

    # try curve_fit first
    try:
        popt, _ = curve_fit(
            broken_power,
            theta_centers,
            y,
            p0=p0,
            maxfev=20000,
            bounds=(
                [  5.0, 0.0, 0.0, 0.0, 0.0],   # lower bounds
                [ 60.0, 1e3, 4.0, 1e3, 4.0],  # upper bounds
            ),
        )
        return popt
    except Exception:
        pass

    # fallback: minimize RSS
    def rss(params):
        return np.sum((broken_power(theta_centers, *params) - y)**2)

    bounds = [(5.0,60.0),(0.0,1e3),(0.0,4.0),(0.0,1e3),(0.0,4.0)]
    res = minimize(rss, p0, bounds=bounds)
    return res.x


def aic(rss, k, n):
    return n*np.log(rss/n) + 2*k


# =====================================================
# main analysis
# =====================================================

def compute_theta():
    """compute angular distance from unified axis using RA/Dec → galactic."""
    df = pd.read_csv("frbs.csv")
    coords = SkyCoord(
        ra=df["ra"].values * u.deg,
        dec=df["dec"].values * u.deg,
        frame="icrs"
    )
    l = coords.galactic.l.deg
    b = coords.galactic.b.deg
    theta = angdist(l, b, UNIFIED_L, UNIFIED_B)
    return theta


def main():
    theta = compute_theta()

    # restrict to 0–140 deg, as in previous analyses
    mask = (theta >= 0.0) & (theta <= 140.0)
    theta = theta[mask]
    n_frb = len(theta)

    # radial bins (1 deg)
    bins = np.linspace(0, 140, 141)
    centers = 0.5*(bins[:-1] + bins[1:])
    H, _ = np.histogram(theta, bins=bins)
    y = H.astype(float)
    y[y == 0] = 1e-3

    # --- real data: constant vs broken power ---

    # constant "no-break" model
    C = np.mean(y)
    y_const = np.full_like(y, C)
    rss_const = np.sum((y - C)**2)
    aic_const = aic(rss_const, k=1, n=len(y))

    # broken-power model
    bp_params = fit_broken_power(centers, y)
    y_bp = broken_power(centers, *bp_params)
    rss_bp = np.sum((y - y_bp)**2)
    aic_bp = aic(rss_bp, k=len(bp_params), n=len(y))

    delta_aic_real = aic_const - aic_bp

    # --- monte carlo under isotropic null ---

    n_sims = 5000
    delta_aic_null = []

    for _ in range(n_sims):
        # isotropic directions → theta distribution relative to axis:
        # cos(theta) uniform in [-1,1]
        u = np.random.uniform(-1.0, 1.0, size=n_frb)
        theta_iso = np.rad2deg(np.arccos(u))

        # same 0–140 cut
        m_iso = theta_iso <= 140.0
        theta_iso = theta_iso[m_iso]

        # if cut changes count, re-sample to keep stats similar
        if len(theta_iso) == 0:
            continue
        if len(theta_iso) != n_frb:
            # rescale by ratio
            target = n_frb
            idx = np.random.choice(len(theta_iso), size=target, replace=True)
            theta_iso = theta_iso[idx]

        H_iso, _ = np.histogram(theta_iso, bins=bins)
        y_iso = H_iso.astype(float)
        y_iso[y_iso == 0] = 1e-3

        # constant fit
        C_iso = np.mean(y_iso)
        rss_c = np.sum((y_iso - C_iso)**2)
        aic_c = aic(rss_c, k=1, n=len(y_iso))

        # broken-power fit
        bp_iso = fit_broken_power(centers, y_iso)
        y_bp_iso = broken_power(centers, *bp_iso)
        rss_bp_iso = np.sum((y_iso - y_bp_iso)**2)
        aic_bp_iso = aic(rss_bp_iso, k=len(bp_iso), n=len(y_iso))

        delta_aic_null.append(aic_c - aic_bp_iso)

    delta_aic_null = np.array(delta_aic_null)
    p_val = np.mean(delta_aic_null >= delta_aic_real)
    mean_null = np.mean(delta_aic_null)
    p95_null = np.percentile(delta_aic_null, 95)

    # --- print results ---

    print("=====================================================================")
    print("FRB RADIAL BREAK SIGNIFICANCE TEST")
    print("=====================================================================")
    print(f"FRBs used (theta ≤ 140°): {n_frb}")
    print("---------------------------------------------------------------------")
    print("REAL DATA:")
    print(f"  constant model AIC      = {aic_const:.2f}")
    print(f"  broken-power model AIC  = {aic_bp:.2f}")
    print(f"  ΔAIC (const - broken)   = {delta_aic_real:.2f}")
    print("---------------------------------------------------------------------")
    print("ISOTROPIC NULL (Monte Carlo):")
    print(f"  simulations             = {len(delta_aic_null)}")
    print(f"  mean ΔAIC(null)         = {mean_null:.2f}")
    print(f"  95% ΔAIC(null)          = {p95_null:.2f}")
    print(f"  p-value (ΔAIC_null ≥ ΔAIC_real) = {p_val:.4f}")
    print("---------------------------------------------------------------------")
    print("SCIENTIFIC VERDICT:")
    if p_val < 0.01:
        print("  → strong evidence that the radial break at ~25° is not")
        print("    a generic feature of isotropic profiles.")
    elif p_val < 0.05:
        print("  → moderate evidence that the radial break is real.")
    else:
        print("  → the observed radial break could plausibly arise from")
        print("    isotropic-like fluctuations in this dataset.")
    print("=====================================================================")

    # --- figure ---

    plt.figure(figsize=(8,5))
    plt.hist(delta_aic_null, bins=40, alpha=0.7, edgecolor="k",
             label="ΔAIC(null, const - broken)")
    plt.axvline(delta_aic_real, color="r", linestyle="--", linewidth=2,
                label=f"real ΔAIC = {delta_aic_real:.2f}")
    plt.xlabel("ΔAIC (const - broken)")
    plt.ylabel("count")
    plt.title("Radial break significance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("radial_break_significance.png", dpi=200)
    plt.close()
    print("saved: radial_break_significance.png")


if __name__ == "__main__":
    main()
