import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---------------------------------------------
# unified axis (galactic)
# ---------------------------------------------
UNIFIED_L = 159.85
UNIFIED_B = -0.51

# shell radial range (deg) for fitting
THETA_MIN = 20.0
THETA_MAX = 60.0

# ---------------------------------------------
# coordinate transform: RA/Dec -> (theta, phi)
# ---------------------------------------------
def compute_axis_frame(ra_deg, dec_deg, l0=UNIFIED_L, b0=UNIFIED_B):
    ra_arr = np.asarray(ra_deg, dtype=float)
    dec_arr = np.asarray(dec_deg, dtype=float)

    coords = SkyCoord(ra=ra_arr * u.deg, dec=dec_arr * u.deg, frame="icrs")
    gal = coords.galactic

    l = np.radians(gal.l.value)
    b = np.radians(gal.b.value)
    l0 = np.radians(l0)
    b0 = np.radians(b0)

    # angle from axis
    cos_theta = np.sin(b0) * np.sin(b) + np.cos(b0) * np.cos(b) * np.cos(l - l0)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # azimuth phi around axis
    x = np.cos(b) * np.cos(l - l0)
    y = np.cos(b) * np.sin(l - l0)
    phi = np.arctan2(y, x)

    return np.degrees(theta), phi  # theta in deg, phi in rad

# ---------------------------------------------
# warped shell model: R(phi)
# ---------------------------------------------
def R_phi(phi, R0, a, b, c, d):
    """
    R(phi) in degrees.
    phi in radians.
    R0 > 0, a,b,c,d describe warp.
    """
    return R0 * (1.0 + a * np.sin(phi) + b * np.cos(phi)
                      + c * np.sin(2.0 * phi) + d * np.cos(2.0 * phi))

def neg_loglike(params, theta_deg, phi, use_m1=True, use_m2=True):
    """
    gaussian shell likelihood:
    p(theta | phi) ~ exp(-(theta - R(phi))^2 / (2 sigma^2))
    """
    R0, sigma = params[0], params[1]
    a = b = c = d = 0.0

    idx = 2
    if use_m1:
        a = params[idx]; b = params[idx+1]; idx += 2
    if use_m2:
        c = params[idx]; d = params[idx+1]; idx += 2

    R = R_phi(phi, R0, a, b, c, d)
    sigma = max(sigma, 1e-3)

    # gaussian log-likelihood up to constant terms
    resid = theta_deg - R
    nll = 0.5 * np.sum((resid / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))
    return nll

def fit_model(theta_deg, phi, use_m1, use_m2):
    # initial guesses
    R0_init = np.median(theta_deg)
    sigma_init = 10.0

    params = [R0_init, sigma_init]
    bounds = [(5.0, 80.0), (1.0, 40.0)]

    if use_m1:
        params += [0.0, 0.0]
        bounds += [(-1.0, 1.0), (-1.0, 1.0)]
    if use_m2:
        params += [0.0, 0.0]
        bounds += [(-1.0, 1.0), (-1.0, 1.0)]

    res = minimize(
        neg_loglike,
        x0=np.array(params),
        args=(theta_deg, phi, use_m1, use_m2),
        bounds=bounds,
    )

    return res

def AIC(nll, k):
    return 2 * k + 2 * nll

# ---------------------------------------------
# monte carlo test
# ---------------------------------------------
def mc_significance(theta_deg, phi, nll_null, k_null, nll_full, k_full,
                    use_m1_full, use_m2_full, nsim=2000):
    """
    shuffle phi to destroy phi-theta correlation, keep theta distribution.
    refit full model each time.
    """
    N = len(theta_deg)
    delta_real = (AIC(nll_null, k_null) - AIC(nll_full, k_full))
    deltas = []

    for _ in range(nsim):
        phi_shuf = np.random.permutation(phi)
        res_mc = fit_model(theta_deg, phi_shuf, use_m1_full, use_m2_full)
        if not res_mc.success:
            continue
        nll_mc = res_mc.fun
        delta_mc = AIC(nll_null, k_null) - AIC(nll_mc, k_full)
        deltas.append(delta_mc)

    deltas = np.array(deltas)
    if len(deltas) == 0:
        return 0.0, delta_real

    p = np.mean(deltas >= delta_real)
    return p, delta_real

# ---------------------------------------------
# main
# ---------------------------------------------
def main():
    print("===================================================")
    print("FRB WARPED-SHELL SHAPE FIT AROUND UNIFIED AXIS")
    print("===================================================")

    df = pd.read_csv("frbs.csv")
    df = df.dropna(subset=["ra", "dec", "z_est"])

    theta, phi = compute_axis_frame(df["ra"], df["dec"])

    # select shell-like region
    mask_shell = (theta >= THETA_MIN) & (theta <= THETA_MAX)
    theta_shell = theta[mask_shell]
    phi_shell = phi[mask_shell]

    print(f"using {len(theta_shell)} frbs in shell {THETA_MIN}–{THETA_MAX} deg")

    # null model: axisymmetric shell (no warp)
    res_null = fit_model(theta_shell, phi_shell, use_m1=False, use_m2=False)
    nll_null = res_null.fun
    k_null = 2  # R0, sigma

    # m=1 only (one-sided warp)
    res_m1 = fit_model(theta_shell, phi_shell, use_m1=True, use_m2=False)
    nll_m1 = res_m1.fun
    k_m1 = 4  # R0, sigma, a, b

    # m=1 + m=2 (full warp)
    res_full = fit_model(theta_shell, phi_shell, use_m1=True, use_m2=True)
    nll_full = res_full.fun
    k_full = 6  # R0, sigma, a, b, c, d

    print("---------------------------------------------------")
    print("fit results (theta ~ shell around unified axis)")
    print("---------------------------------------------------")

    def unpack(res, use_m1, use_m2):
        p = res.x
        R0, sigma = p[0], p[1]
        a = b = c = d = 0.0
        idx = 2
        if use_m1:
            a, b = p[idx], p[idx+1]
            idx += 2
        if use_m2:
            c, d = p[idx], p[idx+1]
        return R0, sigma, a, b, c, d

    R0_null, sig_null, a0, b0, c0, d0 = unpack(res_null, False, False)
    R0_m1, sig_m1, a1, b1, c1, d1 = unpack(res_m1, True, False)
    R0_f, sig_f, af, bf, cf, dfc = unpack(res_full, True, True)

    print(f"axisymmetric shell:")
    print(f"  R0 = {R0_null:.2f} deg, sigma = {sig_null:.2f} deg")
    print(f"  AIC = {AIC(nll_null, k_null):.2f}")
    print()
    print(f"m=1 warped shell:")
    print(f"  R0 = {R0_m1:.2f} deg, sigma = {sig_m1:.2f} deg")
    print(f"  a (sinφ) = {a1:.3f}, b (cosφ) = {b1:.3f}")
    print(f"  AIC = {AIC(nll_m1, k_m1):.2f}")
    print()
    print(f"m=1 + m=2 warped shell:")
    print(f"  R0 = {R0_f:.2f} deg, sigma = {sig_f:.2f} deg")
    print(f"  a (sinφ)  = {af:.3f}, b (cosφ)  = {bf:.3f}")
    print(f"  c (sin2φ) = {cf:.3f}, d (cos2φ) = {dfc:.3f}")
    print(f"  AIC = {AIC(nll_full, k_full):.2f}")
    print()

    # monte carlo: does full warp beat axisymmetric shell?
    print("running monte carlo for warp significance (vs axisymmetric shell)...")
    p_warp, delta_real = mc_significance(
        theta_shell, phi_shell,
        nll_null, k_null,
        nll_full, k_full,
        use_m1_full=True,
        use_m2_full=True,
        nsim=2000,
    )
    print("---------------------------------------------------")
    print(f"warp vs axisymmetric shell: ΔAIC_real = {delta_real:.2f}")
    print(f"MC p-value = {p_warp:.4g}")
    print("---------------------------------------------------")

    # quick visualization
    phi_plot = np.linspace(-np.pi, np.pi, 360)
    R_axis = R_phi(phi_plot, R0_null, 0, 0, 0, 0)
    R_warp = R_phi(phi_plot, R0_f, af, bf, cf, dfc)

    plt.figure(figsize=(7,4))
    plt.scatter(phi_shell, theta_shell, s=5, alpha=0.3, label="frbs (shell)")
    plt.plot(phi_plot, R_axis, linestyle="--", label="axisymmetric R0")
    plt.plot(phi_plot, R_warp, label="warped shell R(φ)")
    plt.xlabel("phi (rad, unified-axis frame)")
    plt.ylabel("theta (deg)")
    plt.title("warped shell fit around unified axis")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig("frb_warped_shell_fit.png", dpi=200)

    print("saved plot: frb_warped_shell_fit.png")
    print("analysis complete.")
    print("===================================================")

if __name__ == "__main__":
    main()
