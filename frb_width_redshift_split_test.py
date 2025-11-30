import numpy as np
import pandas as pd
import warnings
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

CATALOG_FILE = "frbs.csv"

# unified axis (galactic)
UNIFIED_L = 159.85
UNIFIED_B = -0.51

# cone boundaries used in earlier width-layering work
BREAKS_DEG = [10.0, 21.0, 40.0]

# monte carlo for ΔAIC significance
N_MC = 20000


# ===============================================================
# angular distance to unified axis
# ===============================================================

def compute_theta_to_axis(ra_deg, dec_deg, l_axis, b_axis):
    """
    compute angular distances (degrees) from RA,Dec to unified axis (galactic).
    """
    # force numpy float arrays
    ra_arr = np.asarray(ra_deg, dtype=float)
    dec_arr = np.asarray(dec_deg, dtype=float)

    coords_icrs = SkyCoord(
        ra=ra_arr * u.deg,
        dec=dec_arr * u.deg,
        frame="icrs"
    )

    axis_gal = SkyCoord(l=l_axis * u.deg, b=b_axis * u.deg, frame="galactic")
    axis_icrs = axis_gal.icrs

    theta = coords_icrs.separation(axis_icrs).deg
    return theta



# ===============================================================
# width models
# ===============================================================

def model_constant(theta, w0):
    return w0 * np.ones_like(theta)

def model_linear(theta, a, b):
    return a + b * theta

def model_layered(theta, w1, w2, w3):
    """
    three layers:
    0–10°, 10–21°, 21–40° ; ignore >40° same as last layer
    """
    theta = np.array(theta)
    out = np.zeros_like(theta)

    b1, b2, b3 = BREAKS_DEG

    out[theta < b1] = w1
    out[(theta >= b1) & (theta < b2)] = w2
    out[(theta >= b2) & (theta < b3)] = w3
    out[theta >= b3] = w3  # extend last layer
    return out


# ===============================================================
# fit helpers
# ===============================================================

def fit_model(func, theta, width, p0):
    try:
        popt, _ = curve_fit(func, theta, width, p0=p0, maxfev=20000)
        pred = func(theta, *popt)
        rss = np.sum((width - pred) ** 2)
        k = len(popt)
        n = len(width)
        aic = 2*k + n * np.log(rss/n + 1e-12)
        return aic, rss, popt
    except:
        return np.inf, np.inf, None


# ===============================================================
# main analysis for a subset
# ===============================================================

def analyse_subset(label, theta, width):
    print("\n===================================================")
    print(f" WIDTH–AXIS TEST — {label}")
    print("===================================================\n")

    # remove extremely small/zero widths (just safety)
    mask = (width > 0)
    theta = theta[mask]
    width = width[mask]

    if len(width) < 20:
        print("not enough FRBs in this subset for meaningful fitting.")
        return

    # ---- fit models ----
    aic_const, rss_const, p_const = fit_model(model_constant, theta, width, p0=[np.mean(width)])
    aic_lin,   rss_lin,   p_lin   = fit_model(model_linear,   theta, width, p0=[np.mean(width), 0])
    aic_lay,   rss_lay,   p_lay   = fit_model(model_layered,  theta, width, p0=[np.mean(width)*0.8,
                                                                                np.mean(width),
                                                                                np.mean(width)*1.2])

    print(f"constant     AIC={aic_const:8.2f}   RSS={rss_const:8.2f}   params={p_const}")
    print(f"linear       AIC={aic_lin:8.2f}   RSS={rss_lin:8.2f}   params={p_lin}")
    print(f"layered(3)   AIC={aic_lay:8.2f}   RSS={rss_lay:8.2f}   params={p_lay}")

    # best model by AIC
    models = [
        ("constant", aic_const),
        ("linear",   aic_lin),
        ("layered",  aic_lay)
    ]
    best = min(models, key=lambda x: x[1])[0]

    print("\nbest model:", best)

    # ---- ΔAIC significance for layered vs linear ----
    delta_aic = aic_lin - aic_lay

    print(f"\nΔAIC = AIC_linear - AIC_layered = {delta_aic:.3f}")

    # Monte Carlo: shuffle widths
    delta_null = np.zeros(N_MC)
    for i in range(N_MC):
        w_shuff = np.random.permutation(width)
        aic_lay_s, _, _ = fit_model(model_layered, theta, w_shuff, p0=[np.mean(w_shuff),
                                                                      np.mean(w_shuff),
                                                                      np.mean(w_shuff)])
        aic_lin_s,  _, _ = fit_model(model_linear, theta, w_shuff, p0=[np.mean(w_shuff), 0])
        delta_null[i] = aic_lin_s - aic_lay_s

    p_value = np.mean(delta_null >= delta_aic)

    print(f"MC p_value = P(ΔAIC_null >= ΔAIC_real) = {p_value:.4f}")

    # scientific verdict
    print("\n--------------- scientific verdict ---------------")

    if p_value < 0.01:
        print(
            "strong evidence that a layered width model outperforms a linear trend "
            "for this subset. width-layering appears significant here."
        )
    elif p_value < 0.05:
        print(
            "moderate evidence for a layered width structure relative to a simple "
            "linear trend."
        )
    else:
        print(
            "no significant preference for layered width structure in this subset. "
            "width variation with angle is consistent with a linear or constant trend."
        )
    print("---------------------------------------------------")


# ===============================================================
# main
# ===============================================================

def main():
    print("\n===================================================")
    print(" FRB WIDTH LAYERING — REDSHIFT SPLIT TEST")
    print("===================================================\n")

    df = pd.read_csv(CATALOG_FILE)
    df = df.dropna(subset=["width", "ra", "dec", "z_est"])

    theta = compute_theta_to_axis(df["ra"], df["dec"], UNIFIED_L, UNIFIED_B)
    width = df["width"].values

    # split at median z
    z_med = np.median(df["z_est"].values)
    df_low  = df[df["z_est"] <= z_med]
    df_high = df[df["z_est"] >  z_med]

    theta_low  = compute_theta_to_axis(df_low["ra"], df_low["dec"], UNIFIED_L, UNIFIED_B)
    width_low  = df_low["width"].values

    theta_high = compute_theta_to_axis(df_high["ra"], df_high["dec"], UNIFIED_L, UNIFIED_B)
    width_high = df_high["width"].values

    print(f"total FRBs: {len(df)}")
    print(f"low-z sample:  {len(df_low)}")
    print(f"high-z sample: {len(df_high)}")

    # run tests
    analyse_subset("ALL",     theta,       width)
    analyse_subset("LOW-Z",   theta_low,   width_low)
    analyse_subset("HIGH-Z",  theta_high,  width_high)

    print("\nanalysis complete.")
    print("===================================================\n")


if __name__ == "__main__":
    main()
