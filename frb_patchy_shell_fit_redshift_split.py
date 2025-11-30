import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import sph_harm
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Utility: convert RA/Dec to theta, phi relative to unified axis
# ============================================================

def radec_to_unitvec(ra, dec):
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.vstack([x, y, z]).T

def gal_to_unitvec(l, b):
    l_rad = np.deg2rad(l)
    b_rad = np.deg2rad(b)
    x = np.cos(b_rad) * np.cos(l_rad)
    y = np.cos(b_rad) * np.sin(l_rad)
    z = np.sin(b_rad)
    return np.array([x, y, z])

def rotate_into_axis_frame(xyz, axis_vec):
    """
    Rotate vectors so axis_vec becomes the +z direction.
    """
    axis_vec = axis_vec / np.linalg.norm(axis_vec)
    z = np.array([0,0,1.0])

    v = np.cross(axis_vec, z)
    s = np.linalg.norm(v)
    c = np.dot(axis_vec, z)

    if s < 1e-12:
        return xyz

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + vx @ vx * ((1 - c)/(s**2))
    return xyz @ R.T

def xyz_to_thetaphi(xyz):
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    theta = np.arccos(z)
    phi   = np.arctan2(y, x) % (2*np.pi)
    return theta, phi

# ============================================================
# Model definitions
# ============================================================

def smooth_shell_model(theta, A, theta_break):
    return 1.0 + A * np.exp(-(theta/theta_break)**2)

def cone_model(theta, A, theta0):
    return 1.0 + A * (theta < theta0).astype(float)

def patchy_shell(theta, phi, A, B, theta_break):
    Y10 = sph_harm(0,1,phi,theta).real
    Y20 = sph_harm(0,2,phi,theta).real
    return 1.0 + A*np.exp(-(theta/theta_break)**2) + B*(Y10 + Y20)

def layered_shell(theta, A, B, theta0_deg):
    theta0 = np.deg2rad(theta0_deg)
    return np.where(theta < theta0, 1.0 + A, 1.0 + B)

def dip_quad(theta, phi, a, b):
    Y10 = sph_harm(0,1,phi,theta).real
    Y20 = sph_harm(0,2,phi,theta).real
    return 1.0 + a*Y10 + b*Y20

# ============================================================
# Fit wrapper
# ============================================================

def fit_model(func, xdata, ydata, p0):
    try:
        popt, _ = curve_fit(func, xdata, ydata, p0=p0, maxfev=30000)
        pred = func(xdata, *popt) if not isinstance(xdata, tuple) else func(*xdata, *popt)
        rss = np.sum((ydata - pred)**2)
        k = len(popt)
        aic = 2*k + len(ydata)*np.log(rss/len(ydata) + 1e-12)
        return aic, rss, popt
    except:
        return np.inf, np.inf, None

# ============================================================
# Runner for a given subset
# ============================================================

def run_shell_fit(df, label):

    axis_l = 159.85
    axis_b = -0.51
    axis_vec = gal_to_unitvec(axis_l, axis_b)

    xyz = radec_to_unitvec(df["ra"].values, df["dec"].values)
    xyz_rot = rotate_into_axis_frame(xyz, axis_vec)
    theta, phi = xyz_to_thetaphi(xyz_rot)

    data = np.ones_like(theta)

    models = []

    # Smooth shell
    aic, rss, popt = fit_model(
        lambda th, A, tb: smooth_shell_model(th, A, tb),
        theta, data, p0=[0.5, np.deg2rad(25)]
    )
    models.append(("smooth_shell", aic, rss, popt))

    # Cone
    aic, rss, popt = fit_model(
        lambda th, A, t0: cone_model(th, A, t0),
        theta, data, p0=[0.5, np.deg2rad(30)]
    )
    models.append(("cone", aic, rss, popt))

    # Patchy shell
    aic, rss, popt = fit_model(
        lambda th, ph, A, B, tb: patchy_shell(th, ph, A, B, tb),
        (theta, phi), data, p0=[0.3, 0.3, np.deg2rad(25)]
    )
    models.append(("patchy_shell", aic, rss, popt))

    # Layered shell
    aic, rss, popt = fit_model(
        lambda th, A, B, t0: layered_shell(th, A, B, t0),
        theta, data, p0=[0.2, 0.4, 25.0]
    )
    models.append(("layered_shell", aic, rss, popt))

    # Dipole+quadrupole
    aic, rss, popt = fit_model(
        lambda th, ph, a, b: dip_quad(th, ph, a, b),
        (theta, phi), data, p0=[1.0, 0.5]
    )
    models.append(("dipole+quadrupole", aic, rss, popt))

    print("\n=======================================================")
    print(f" FRB PATCHY-SHELL MODEL — {label}")
    print("=======================================================\n")

    for name, aic, rss, popt in models:
        print(f"{name:20s}  AIC={aic:8.2f}   RSS={rss:8.2f}   params={popt}")

    best = min(models, key=lambda x: x[1])
    print("\n-------------------------------------------------------")
    print(f" BEST MODEL ({label}): {best[0]}")
    print("-------------------------------------------------------\n")


# ============================================================
# MAIN
# ============================================================

def main():

    df = pd.read_csv("frbs.csv")
    df = df.dropna(subset=["z_est"])

    # split into low-z and high-z
    z_sorted = df["z_est"].values
    z_med = np.median(z_sorted)

    df_low  = df[df["z_est"] <= z_med].copy()
    df_high = df[df["z_est"] >  z_med].copy()

    print("Loaded FRBs with redshift:", len(df))
    print(f"Low-z sample:  {len(df_low)} events")
    print(f"High-z sample: {len(df_high)} events")

    run_shell_fit(df_low,  "LOW-Z HALF")
    run_shell_fit(df_high, "HIGH-Z HALF")

    print("Analysis complete.\n")


if __name__ == "__main__":
    main()
