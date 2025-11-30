import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def angle_from_axis(ra, dec, l0, b0):
    """Angular distance between each FRB and the unified axis."""
    frb = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs').galactic
    axis = SkyCoord(l=l0*u.deg, b=b0*u.deg, frame='galactic')
    return frb.separation(axis).deg

def layer_model(theta, cuts, values):
    """Piecewise-constant layered model."""
    result = np.zeros_like(theta)
    for i, t in enumerate(theta):
        if t < cuts[0]:
            result[i] = values[0]
        elif t < cuts[1]:
            result[i] = values[1]
        else:
            result[i] = values[2]
    return result

def rss(model, data):
    return np.sum((model - data)**2)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("frbs.csv")

ra = df["ra"].values
dec = df["dec"].values
width = df["width"].values

# unified axis from your previous best-fit result
L0 = 159.85
B0 = -0.51

theta = angle_from_axis(ra, dec, L0, B0)

print("="*70)
print("FRB WIDTH LAYER SIGNIFICANCE TEST")
print("20,000 Monte Carlo permutations")
print("="*70)
print(f"Total FRBs: {len(width)}")
print(f"Theta range: {theta.min():.2f}° – {theta.max():.2f}°")
print()

# ============================================================
# FIT REAL DATA (linear vs 3-layer)
# ============================================================

# ----- linear model -----
def linear_model(theta, a, b):
    return a + b * theta

def linear_res(params):
    a, b = params
    return rss(linear_model(theta, a, b), width)

lin_opt = minimize(linear_res, [np.mean(width), 0], method="Nelder-Mead")
real_linear_rss = lin_opt.fun
real_linear_k = 2

# ----- 3-layer model -----
def three_layer_res(params):
    cut1, cut2 = sorted(params[:2])
    v1, v2, v3 = params[2:]
    pred = layer_model(theta, [cut1, cut2], [v1, v2, v3])
    return rss(pred, width)

three_init = [15.0, 30.0, np.mean(width), np.mean(width), np.mean(width)]
three_bounds = [(0,140),(0,140),(0,None),(0,None),(0,None)]

three_opt = minimize(three_layer_res, three_init, bounds=three_bounds)
real_three_rss = three_opt.fun
real_three_k = 5

# AIC values
def AIC(rss, k, n):
    return 2*k + n*np.log(rss/n)

n = len(width)
real_AIC_linear = AIC(real_linear_rss, real_linear_k, n)
real_AIC_three = AIC(real_three_rss, real_three_k, n)

real_delta_AIC = real_AIC_linear - real_AIC_three

print("SECTION 1 — REAL DATA FIT")
print("-"*60)
print(f"Linear RSS       = {real_linear_rss:.4f}")
print(f"3-layer RSS      = {real_three_rss:.4f}")
print(f"Linear AIC       = {real_AIC_linear:.2f}")
print(f"3-layer AIC      = {real_AIC_three:.2f}")
print(f"ΔAIC (linear - 3) = {real_delta_AIC:.2f}")
print("")

# ============================================================
# MONTE CARLO PERMUTATION TEST
# ============================================================

Nsim = 20000
delta_AIC_null = np.zeros(Nsim)

print("SECTION 2 — MONTE CARLO NULL (20,000 perms)")
print("-"*60)

for i in range(Nsim):
    perm = np.random.permutation(width)

    # linear on permuted data
    def perm_lin_res(params):
        a, b = params
        return rss(a + b*theta, perm)

    lin_opt_p = minimize(perm_lin_res, [np.mean(perm), 0], method="Nelder-Mead")
    rss_lin_p = lin_opt_p.fun

    # 3-layer on permuted data
    def perm_three_res(params):
        cut1, cut2 = sorted(params[:2])
        v1, v2, v3 = params[2:]
        pred = layer_model(theta, [cut1, cut2], [v1, v2, v3])
        return rss(pred, perm)

    three_opt_p = minimize(perm_three_res, three_init, bounds=three_bounds)
    rss_three_p = three_opt_p.fun

    # AIC
    AIC_lin_p = AIC(rss_lin_p, real_linear_k, n)
    AIC_three_p = AIC(rss_three_p, real_three_k, n)

    delta_AIC_null[i] = AIC_lin_p - AIC_three_p

    if (i+1) % 2000 == 0:
        print(f"Completed {i+1}/{Nsim}")

# p-value: fraction of null ΔAIC exceeding real ΔAIC
p_value = np.mean(delta_AIC_null >= real_delta_AIC)

print("")
print("SECTION 3 — SIGNIFICANCE")
print("-"*60)
print(f"Real ΔAIC = {real_delta_AIC:.2f}")
print(f"Mean ΔAIC(null) = {np.mean(delta_AIC_null):.2f}")
print(f"95% ΔAIC(null) = {np.percentile(delta_AIC_null,95):.2f}")
print(f"p-value = {p_value:.4f}")
print("")

# ============================================================
# PLOT DISTRIBUTION
# ============================================================

plt.figure(figsize=(10,6))
plt.hist(delta_AIC_null, bins=80, alpha=0.7, label="null permutations")
plt.axvline(real_delta_AIC, color="red", linewidth=2, label=f"real ΔAIC = {real_delta_AIC:.2f}")
plt.xlabel("ΔAIC (linear - 3-layer)")
plt.ylabel("count")
plt.title("Null distribution of ΔAIC vs real value")
plt.legend()
plt.tight_layout()
plt.savefig("width_layer_significance.png")

print("SECTION 4 — FIGURE")
print("-"*60)
print("saved: width_layer_significance.png")
print("")
print("analysis complete.")
print("="*70)
