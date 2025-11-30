import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =====================================================
# unified axis (galactic)
# =====================================================
UNIFIED_L = 159.85
UNIFIED_B = -0.51


# =====================================================
# stable rotation: RA/Dec → galactic → axis frame
# =====================================================
def rotate_to_axis_from_gal(l_deg, b_deg, L0=UNIFIED_L, B0=UNIFIED_B):
    """
    stable rotation avoiding skyoffset_frame.
    compute theta, phi relative to axis sitting at (L0,B0).
    """
    l = np.deg2rad(l_deg)
    b = np.deg2rad(b_deg)
    L0 = np.deg2rad(L0)
    B0 = np.deg2rad(B0)

    # spherical law of cosines for angle from the axis
    cos_th = (
        np.sin(b) * np.sin(B0) + 
        np.cos(b) * np.cos(B0) * np.cos(l - L0)
    )
    cos_th = np.clip(cos_th, -1, 1)
    theta = np.rad2deg(np.arccos(cos_th))

    # projected phi
    y = np.cos(b) * np.sin(l - L0)
    x = (np.sin(b) - np.sin(B0)*cos_th) / (np.cos(B0)*np.sin(np.deg2rad(theta)) + 1e-9)
    phi = np.rad2deg(np.arctan2(y, x)) % 360

    return theta, phi


# =====================================================
# 3D model definitions
# =====================================================

def cone_model(theta, A, C):
    """simple cone falloff"""
    return A * (1 - np.cos(np.deg2rad(theta))) + C

def sphere_shell_model(theta, R, W, A, C):
    """thin spherical shell at radius R with width W"""
    return A * np.exp(-0.5*((theta - R)/W)**2) + C

def layered_model(theta, R1, R2, A1, A2, A3):
    """two shells or layers"""
    out = np.zeros_like(theta)
    out += A1 * np.exp(-0.5*((theta - R1)/8)**2)
    out += A2 * np.exp(-0.5*((theta - R2)/8)**2)
    out += A3
    return out


# =====================================================
# main
# =====================================================
print("=====================================================")
print("FRB FULL 3D SHAPE MODEL — stable version")
print("=====================================================")

# load the correct catalog
df = pd.read_csv("frbs.csv")

# determine which columns are present
if "l" in df.columns and "b" in df.columns:
    l = df["l"].values
    b = df["b"].values
else:
    # must convert RA/Dec → galactic
    c = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg, frame="icrs")
    g = c.galactic
    l = g.l.deg
    b = g.b.deg

# rotate into axis frame
theta, phi = rotate_to_axis_from_gal(l, b)
theta = np.array(theta)

# radial histogram (phi averaged)
bins = np.linspace(0, 140, 50)
values, centers = np.histogram(theta, bins=bins)
theta_centers = 0.5*(bins[1:] + bins[:-1])
y = values.astype(float)


# =====================================================
# model fits
# =====================================================

def fit_model(func, p0):
    try:
        popt, _ = curve_fit(func, theta_centers, y, p0=p0, maxfev=40000)
        pred = func(theta_centers, *popt)
        rss = np.sum((y - pred)**2)
        k = len(popt)
        n = len(y)
        aic = n*np.log(rss/n) + 2*k
        bic = n*np.log(rss/n) + k*np.log(n)
        return popt, rss, aic, bic
    except:
        return None, np.inf, np.inf, np.inf


# cone
p_cone, rss_cone, aic_cone, bic_cone = fit_model(cone_model, [10, 10])

# sphere shell
p_shell, rss_shell, aic_shell, bic_shell = fit_model(sphere_shell_model, [30, 8, 5, 5])

# layered
p_layer, rss_layer, aic_layer, bic_layer = fit_model(layered_model, [10, 30, 5, 5, 5])

# pick best
models = {
    "cone": aic_cone,
    "sphere_shell": aic_shell,
    "layered": aic_layer,
}
best = min(models, key=models.get)

print("-----------------------------------------------------")
print(f"cone         AIC = {aic_cone:8.2f}  RSS = {rss_cone:8.2f}")
print(f"sphere_shell AIC = {aic_shell:8.2f}  RSS = {rss_shell:8.2f}")
print(f"layered      AIC = {aic_layer:8.2f}  RSS = {rss_layer:8.2f}")
print("-----------------------------------------------------")
print(f"BEST MODEL: {best}")
print("=====================================================")

# plot
plt.figure(figsize=(8,5))
plt.plot(theta_centers, y, "ko", label="data")
if best=="cone":
    plt.plot(theta_centers, cone_model(theta_centers, *p_cone), "r-", label="cone")
if best=="sphere_shell":
    plt.plot(theta_centers, sphere_shell_model(theta_centers, *p_shell), "b-", label="sphere shell")
if best=="layered":
    plt.plot(theta_centers, layered_model(theta_centers, *p_layer), "g-", label="layered")
plt.xlabel("theta (deg)")
plt.ylabel("counts")
plt.legend()
plt.tight_layout()
plt.savefig("frb_full_3d_shape_model.png")
print("saved: frb_full_3d_shape_model.png")
print("=====================================================")
