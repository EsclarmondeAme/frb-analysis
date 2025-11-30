import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt

# ============================================================
# unified axis (galactic) — same as your main pipeline
# ============================================================
UNIFIED_L = 159.85
UNIFIED_B = -0.51


# ============================================================
# spherical distance (deg)
# ============================================================
def angdist(l1, b1, l2, b2):
    l1 = np.deg2rad(l1)
    b1 = np.deg2rad(b1)
    l2 = np.deg2rad(l2)
    b2 = np.deg2rad(b2)
    return np.rad2deg(
        np.arccos(
            np.sin(b1)*np.sin(b2) + np.cos(b1)*np.cos(b2)*np.cos(l1 - l2)
        )
    )


# ============================================================
# torus model
#
# A torus projected onto a sphere becomes a band with a
# preferred radius R0 and thickness W. This gives a
# Gaussian-in-the-radial-direction profile.
#
# f(theta) = A * exp[- (theta - R0)^2 / (2 W^2) ] + C
# ============================================================
def torus_profile(theta, R0, W, A, C):
    return A * np.exp(-0.5 * ((theta - R0)/W)**2) + C


# ============================================================
# AIC / BIC
# ============================================================
def compute_aic(rss, k, n):
    return n*np.log(rss/n) + 2*k

def compute_bic(rss, k, n):
    return n*np.log(rss/n) + k*np.log(n)


# ============================================================
# fit torus model
# ============================================================
def fit_torus(theta_centers, y):
    def rss(params):
        R0, W, A, C = params
        if W <= 0:
            return 1e30
        y_pred = torus_profile(theta_centers, R0, W, A, C)
        return np.sum((y - y_pred)**2)

    p0 = [30.0, 10.0, 1.0, 1.0]
    bounds = [(0, 140), (1e-6, 50), (0, None), (0, None)]

    res = minimize(rss, p0, bounds=bounds)
    return res.x, res.fun



# ============================================================
# simple polynomial comparator
# ============================================================
def fit_poly(theta_centers, y):
    X = np.vstack([np.ones_like(theta_centers),
                   theta_centers,
                   theta_centers**2]).T
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    rss = np.sum((y - y_pred)**2)
    return coef, rss



# ============================================================
# main
# ============================================================
def main():
    print("====================================================")
    print(" FRB TORUS PROJECTION FIT")
    print("====================================================")

    # ----------------------------------------------------
    # data
    # ----------------------------------------------------
    df = pd.read_csv("frbs.csv")
    theta = angdist(df["ra"], df["dec"], UNIFIED_L, UNIFIED_B)

    # radial profile (1° bins)
    bins = np.linspace(0, 140, 141)
    H, _, _ = binned_statistic(theta, theta, statistic="count", bins=bins)
    theta_centers = 0.5*(bins[:-1] + bins[1:])
    y = H

    # remove zeros for log-like stability
    mask = y > 0
    theta_centers = theta_centers[mask]
    y = y[mask]

    n = len(y)

    # ----------------------------------------------------
    # torus fit
    # ----------------------------------------------------
    params_torus, rss_torus = fit_torus(theta_centers, y)
    R0, W, A, C = params_torus

    aic_torus = compute_aic(rss_torus, 4, n)
    bic_torus = compute_bic(rss_torus, 4, n)

    # ----------------------------------------------------
    # polynomial comparator
    # ----------------------------------------------------
    coef_poly, rss_poly = fit_poly(theta_centers, y)
    aic_poly = compute_aic(rss_poly, 3, n)
    bic_poly = compute_bic(rss_poly, 3, n)

    # ----------------------------------------------------
    # monte-carlo isotropic null
    # ----------------------------------------------------
    sims = 2000
    null_aic_diff = []
    for _ in range(sims):
        # isotropic: random theta drawn from sin(theta)
        # invert CDF: theta = arccos(1 - u)
        u = np.random.rand(len(theta))
        th_sim = np.rad2deg(np.arccos(1 - u))

        Hs, _, _ = binned_statistic(th_sim, th_sim, statistic="count", bins=bins)
        yc = Hs[bins[:-1] < 140]
        yc = yc[yc > 0]

        # fit poly
        _, rss_p = fit_poly(theta_centers[:len(yc)], yc)
        aic_p = compute_aic(rss_p, 3, len(yc))

        # fit torus
        _, rss_t = fit_torus(theta_centers[:len(yc)], yc)
        aic_t = compute_aic(rss_t, 4, len(yc))

        null_aic_diff.append(aic_poly - aic_t)

    null_aic_diff = np.array(null_aic_diff)
    p_value = np.mean(null_aic_diff >= (aic_poly - aic_torus))

    # ----------------------------------------------------
    # print results
    # ----------------------------------------------------
    print("\n==================== RESULTS ====================")
    print("Torus params:")
    print(f"  R0     = {R0:.3f} deg")
    print(f"  W      = {W:.3f} deg")
    print(f"  A      = {A:.3f}")
    print(f"  C      = {C:.3f}")
    print("-------------------------------------------------")
    print(f"RSS_torus = {rss_torus:.2f}")
    print(f"AIC_torus = {aic_torus:.2f}")
    print(f"BIC_torus = {bic_torus:.2f}")
    print("-------------------------------------------------")
    print("Polynomial model:")
    print(f"RSS_poly = {rss_poly:.2f}")
    print(f"AIC_poly = {aic_poly:.2f}")
    print(f"BIC_poly = {bic_poly:.2f}")
    print("-------------------------------------------------")
    print(f"ΔAIC(real) = AIC_poly - AIC_torus = {aic_poly - aic_torus:.2f}")
    print(f"Monte Carlo p-value = {p_value:.4f}")
    print("=================================================\n")

    # ----------------------------------------------------
    # scientific verdict
    # ----------------------------------------------------
    print("====================================================")
    print(" SCIENTIFIC VERDICT")
    print("====================================================")

    if aic_torus < aic_poly:
        print("→ torus model preferred over polynomial.")
        if p_value < 0.01:
            print("→ torus preference is statistically significant (p < 0.01).")
        elif p_value < 0.05:
            print("→ torus preference is marginally significant (p < 0.05).")
        else:
            print("→ torus preference is not significant under isotropic null.")
    else:
        print("→ radial polynomial fits better. no toroidal signature detected.")

    print("====================================================")

    # ----------------------------------------------------
    # figure
    # ----------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(theta_centers, y, 'k.', label="data")
    th_fine = np.linspace(0,140,500)
    plt.plot(th_fine, torus_profile(th_fine, *params_torus), 'r-', label="torus fit")
    plt.xlabel("θ from unified axis (deg)")
    plt.ylabel("FRB density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("frb_torus_fit.png")
    print("saved: frb_torus_fit.png")


if __name__ == "__main__":
    main()
