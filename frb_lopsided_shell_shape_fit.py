import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ============================================================
# unified axis (from your previous tests)
# ============================================================
UNIFIED_L = 159.85   # deg
UNIFIED_B = -0.51    # deg

# ============================================================
#  broken-power radial profile (from earlier best-fit results)
# ============================================================
def radial_profile(theta_deg, A=1.0, alpha1=0.0, alpha2=-1.2, theta_break=25.0):
    """
    smooth broken-power radial profile
    """
    t = np.radians(theta_deg)
    tb = np.radians(theta_break)
    if isinstance(t, np.ndarray):
        out = np.where(t < tb,
                       A * (t / tb)**alpha1,
                       A * (t / tb)**alpha2)
        return out
    else:
        return A * (t/tb)**(alpha1 if t < tb else alpha2)

# ============================================================
# convert to unified-axis frame
# ============================================================
def compute_axis_frame(ra, dec, L0=UNIFIED_L, B0=UNIFIED_B):
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    axis = SkyCoord(l=L0*u.deg, b=B0*u.deg, frame='galactic').icrs

    # great-circle angle from axis
    theta = coords.separation(axis).deg

    # set phi = angle around axis
    # basis: take axis as north pole
    pole = axis
    pole_ra = pole.ra.deg
    pole_dec = pole.dec.deg

    # rotate coordinate system so pole is new north pole
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    pra = np.radians(pole_ra)
    pdec = np.radians(pole_dec)

    # compute sin(phi) and cos(phi)
    sin_phi = np.sin(ra_rad - pra) * np.cos(dec_rad)
    y = sin_phi
    x = (np.sin(dec_rad) * np.sin(pdec) +
         np.cos(dec_rad) * np.cos(pdec) * np.cos(ra_rad - pra))
    phi = np.degrees(np.arctan2(y, x))
    phi = (phi + 360) % 360
    return theta, phi

# ============================================================
# full model: radial * (1 + A1 cos(phi-phi0) + A2 cos2(phi-phi0))
# ============================================================
def model(theta, phi, A1, phi0, A2=0.0):
    phi_rad = np.radians(phi - phi0)
    return radial_profile(theta) * (1 + A1*np.cos(phi_rad) + A2*np.cos(2*phi_rad))

# ============================================================
# negative log-likelihood for Poisson spatial model
# ============================================================
def nll(params, theta, phi):
    A1, phi0, A2 = params
    lam = model(theta, phi, A1, phi0, A2)
    lam = np.clip(lam, 1e-12, None)
    return -np.sum(np.log(lam))

# constrained fit: enforce |A1|<=1 , |A2|<=1
bounds = [(-1,1), (0,360), (-1,1)]

def fit_model(theta, phi, use_A2=True):
    if use_A2:
        p0 = [0.1, 45.0, 0.05]
    else:
        p0 = [0.1, 45.0, 0.0]

    def nll_reduced(p):
        if use_A2:
            return nll(p, theta, phi)
        else:
            return nll([p[0], p[1], 0.0], theta, phi)

    b = bounds if use_A2 else bounds[:2]
    res = minimize(nll_reduced, p0[:len(b)], bounds=b)
    if use_A2:
        A1, phi0, A2 = res.x
    else:
        A1, phi0 = res.x
        A2 = 0.0

    return res.fun, A1, phi0 % 360, A2

# ============================================================
# AIC
# ============================================================
def AIC(nll, k):
    return 2*k + 2*nll

# ============================================================
# Monte Carlo
# ============================================================
def mc_compare(theta, phi, AIC_real, nsim=2000, use_A2=True):
    count = 0
    N = len(theta)
    for _ in range(nsim):
        phi_scr = np.random.uniform(0,360,N)
        nll_mc, A1, phi0, A2 = fit_model(theta, phi_scr, use_A2=use_A2)
        AIC_mc = AIC(nll_mc, 3 if use_A2 else 2)
        if AIC_mc <= AIC_real:
            count += 1
    return count / nsim

# ============================================================
# main
# ============================================================
def main():
    df = pd.read_csv("frbs.csv")
    df = df.dropna(subset=["ra","dec"])

    theta, phi = compute_axis_frame(df["ra"].values, df["dec"].values)

    # model 1: pure radial (no azimuth)
    nll_rad = -np.sum(np.log(radial_profile(theta)))
    AIC_rad = AIC(nll_rad, k=1)

    # model 2: m=1 lopsided
    nll_m1, A1, phi0, _ = fit_model(theta, phi, use_A2=False)
    AIC_m1 = AIC(nll_m1, k=2)

    # model 3: m=1 + m=2
    nll_m2, A1b, phi0b, A2b = fit_model(theta, phi, use_A2=True)
    AIC_m2 = AIC(nll_m2, k=3)

    print("=====================================")
    print("LOPSIDED SHELL SHAPE FIT RESULTS")
    print("=====================================")
    print(f"pure radial:   AIC={AIC_rad:.2f}")
    print(f"m=1 model:     AIC={AIC_m1:.2f}   A1={A1:.3f}   phi0={phi0:.1f}°")
    print(f"m=1+m=2 model: AIC={AIC_m2:.2f}   A1={A1b:.3f}   A2={A2b:.3f}  phi0={phi0b:.1f}°")

    best = min(AIC_rad, AIC_m1, AIC_m2)
    if best == AIC_rad:
        print("\nbest model: pure radial (no azimuth)\n")
    elif best == AIC_m1:
        print("\nbest model: m=1 lopsided shell\n")
        p = mc_compare(theta, phi, AIC_m1, use_A2=False)
        print(f"MC p-value (m=1 significance) = {p:.4f}")
    else:
        print("\nbest model: m=1+m=2 multipatch shell\n")
        p = mc_compare(theta, phi, AIC_m2, use_A2=True)
        print(f"MC p-value (m=1+m=2 significance) = {p:.4f}")

    # save a quick scatter plot in axis frame
    plt.figure(figsize=(6,4))
    plt.scatter(phi, theta, s=4, alpha=0.4)
    plt.xlabel("phi (deg)")
    plt.ylabel("theta (deg)")
    plt.title("FRBs in Unified-Axis Frame")
    plt.gca().invert_yaxis()
    plt.savefig("frb_lopsided_shell_scatter.png", dpi=200)

if __name__ == "__main__":
    main()
