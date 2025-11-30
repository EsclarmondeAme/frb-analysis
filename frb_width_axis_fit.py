import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize

print("=" * 69)
print("FRB WIDTH–AXIS MODEL FITTING")
print("testing linear, quadratic, and layered models for width(θ)")
print("=" * 69)

# ------------------------------------------------------------
# 1. unified axis
# ------------------------------------------------------------
unified_l = 159.85
unified_b = -0.51
unified_axis = SkyCoord(l=unified_l*u.deg, b=unified_b*u.deg, frame="galactic")

print("\nSECTION 1 — UNIFIED AXIS")
print("------------------------------------------------------------")
print(f"axis: l = {unified_l:.2f}°,  b = {unified_b:.2f}°")

# ------------------------------------------------------------
# 2. load FRBs
# ------------------------------------------------------------
print("\nSECTION 2 — DATASET")
print("------------------------------------------------------------")

frb = pd.read_csv("frbs.csv")
coords = SkyCoord(ra=frb["ra"].values*u.deg,
                  dec=frb["dec"].values*u.deg,
                  frame="icrs").galactic
theta = coords.separation(unified_axis).deg
width = frb["width"].values.astype(float)

frb["theta"] = theta

print(f"loaded {len(frb)} FRBs")
print(f"θ range: {theta.min():.2f}° – {theta.max():.2f}°")

# ------------------------------------------------------------
# 3. define models
# ------------------------------------------------------------

def linear_model(theta, a, b):
    return a + b*theta

def quadratic_model(theta, a, b, c):
    return a + b*theta + c*theta**2

def layer_model(theta, cuts, values):
    """step function model: cuts = [cut1, cut2...], values = [v1, v2, ...]"""
    out = np.zeros_like(theta)
    bins = np.concatenate(([-np.inf], cuts, [np.inf]))
    for i in range(len(values)):
        mask = (theta >= bins[i]) & (theta < bins[i+1])
        out[mask] = values[i]
    return out

# ------------------------------------------------------------
# 4. fit models
# ------------------------------------------------------------
print("\nSECTION 3 — MODEL FITTING")
print("------------------------------------------------------------")

# ----- linear -----
def lin_res(params):
    a, b = params
    pred = linear_model(theta, a, b)
    return np.sum((width - pred)**2)

lin_opt = minimize(lin_res, [np.mean(width), 0.0])
lin_params = lin_opt.x
lin_rss = lin_opt.fun
lin_k = 2

print(f"linear rss = {lin_rss:.3f}, params = {lin_params}")

# ----- quadratic -----
def quad_res(params):
    a, b, c = params
    pred = quadratic_model(theta, a, b, c)
    return np.sum((width - pred)**2)

quad_opt = minimize(quad_res, [np.mean(width), 0.0, 0.0])
quad_params = quad_opt.x
quad_rss = quad_opt.fun
quad_k = 3

print(f"quadratic rss = {quad_rss:.3f}, params = {quad_params}")

# ----- 2-layer -----
def two_layer_res(params):
    cut = params[0]
    v1, v2 = params[1], params[2]
    pred = layer_model(theta, [cut], [v1, v2])
    return np.sum((width - pred)**2)

two_opt = minimize(two_layer_res, [30.0, np.mean(width), np.mean(width)], bounds=[(0,140),(0,None),(0,None)])
two_cut, v1, v2 = two_opt.x
two_rss = two_opt.fun
two_k = 3

print(f"2-layer rss = {two_rss:.3f}, cut = {two_cut:.2f}, values = {v1:.4f}, {v2:.4f}")

# ----- 3-layer -----
def three_layer_res(params):
    cut1, cut2 = sorted(params[:2])
    v1, v2, v3 = params[2], params[3], params[4]
    pred = layer_model(theta, [cut1, cut2], [v1, v2, v3])
    return np.sum((width - pred)**2)

three_init = [
    15.0,               # initial cut1
    30.0,               # initial cut2
    np.mean(width),     # v1
    np.mean(width),     # v2
    np.mean(width)      # v3
]

three_bounds = [
    (0, 140),   # cut1
    (0, 140),   # cut2
    (0, None),  # v1
    (0, None),  # v2
    (0, None)   # v3
]

three_opt = minimize(three_layer_res, three_init, bounds=three_bounds)

cut1, cut2 = sorted(three_opt.x[:2])
v1, v2, v3 = three_opt.x[2:]
three_rss = three_opt.fun
three_k = 5

print(f"3-layer rss = {three_rss:.3f}, cuts = {cut1:.2f}, {cut2:.2f}, values = {v1:.4f}, {v2:.4f}, {v3:.4f}")



# ------------------------------------------------------------
# 5. AIC / BIC comparison
# ------------------------------------------------------------
print("\nSECTION 4 — MODEL COMPARISON (AIC/BIC)")
print("------------------------------------------------------------")

n = len(width)

def AIC(rss, k): return n*np.log(rss/n) + 2*k
def BIC(rss, k): return n*np.log(rss/n) + k*np.log(n)

models = {
    "linear"  : (lin_rss, lin_k),
    "quadratic": (quad_rss, quad_k),
    "2-layer" : (two_rss, two_k),
    "3-layer" : (three_rss, three_k)
}

for name, (rss, k) in models.items():
    print(f"{name:10s} AIC = {AIC(rss,k):.2f},  BIC = {BIC(rss,k):.2f},  rss = {rss:.2f}")

best_aic = min(models.items(), key=lambda x: AIC(x[1][0], x[1][1]))[0]
best_bic = min(models.items(), key=lambda x: BIC(x[1][0], x[1][1]))[0]

print(f"\nbest model by AIC: {best_aic}")
print(f"best model by BIC: {best_bic}")

# ------------------------------------------------------------
# 6. Figure
# ------------------------------------------------------------
print("\nSECTION 5 — FIGURE")
print("------------------------------------------------------------")

plt.figure(figsize=(10,6))
plt.scatter(theta, width, s=10, alpha=0.5, label="FRB widths")

grid = np.linspace(theta.min(), theta.max(), 300)

# plot each model
plt.plot(grid, linear_model(grid, *lin_params), label="linear", lw=2)
plt.plot(grid, quadratic_model(grid, *quad_params), label="quadratic", lw=2)

plt.plot(grid, layer_model(grid, [two_cut], [v1, v2]), label="2-layer", lw=2)
plt.plot(grid, layer_model(grid, [cut1, cut2], [v1, v2, v3]), label="3-layer", lw=2)

plt.xlabel("angular distance from unified axis θ (deg)")
plt.ylabel("FRB width (ms)")
plt.title("FRB Width vs Axis Distance — Model Comparison")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("width_axis_fit.png", dpi=200)
print("saved: width_axis_fit.png")

# ------------------------------------------------------------
# 7. verdict
# ------------------------------------------------------------
print("\nSECTION 6 — VERDICT")
print("------------------------------------------------------------")

print(f"best AIC model: {best_aic}")
print(f"best BIC model: {best_bic}")

if best_aic == "3-layer" or best_bic == "3-layer":
    print("\n→ evidence supports a layered structure in width(θ)")
elif best_aic == "2-layer" or best_bic == "2-layer":
    print("\n→ weak evidence for a 2-layer structure")
else:
    print("\n→ width variation is smooth (linear or quadratic), not layered")

print("\nanalysis complete")
print("="*69)
