import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# unified axis (galactic)
UNIFIED_L = 159.85
UNIFIED_B = -0.51

THETA_MIN = 20.0
THETA_MAX = 60.0

# -----------------------------------------------------
# coordinate transform to unified-axis frame
# -----------------------------------------------------
def compute_axis_frame(ra_deg, dec_deg, l0=UNIFIED_L, b0=UNIFIED_B):
    ra = np.asarray(ra_deg, float)
    dec = np.asarray(dec_deg, float)

    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    gal = coords.galactic

    l = np.radians(gal.l.value)
    b = np.radians(gal.b.value)
    l0 = np.radians(l0)
    b0 = np.radians(b0)

    cos_theta = np.sin(b0) * np.sin(b) + np.cos(b0) * np.cos(b) * np.cos(l - l0)
    theta = np.arccos(np.clip(cos_theta, -1, 1))

    x = np.cos(b) * np.cos(l - l0)
    y = np.cos(b) * np.sin(l - l0)
    phi = np.arctan2(y, x)

    return np.degrees(theta), phi

# -----------------------------------------------------
# ellipsoid radius model
# -----------------------------------------------------
def R_ellipsoid(theta_deg, phi, a, b, c, phi0):
    """
    ellipsoid in unified-axis frame:
    
    x = sinθ cosφ rotated by phi0,
    y = sinθ sinφ rotated by phi0,
    z = cosθ

    radius = 1 / sqrt((x/a)^2 + (y/b)^2 + (z/c)^2)

    converted back to degrees.
    """
    theta = np.radians(theta_deg)
    phi_rot = phi - phi0

    x = np.sin(theta) * np.cos(phi_rot)
    y = np.sin(theta) * np.sin(phi_rot)
    z = np.cos(theta)

    denom = (x/a)**2 + (y/b)**2 + (z/c)**2
    R = 1.0 / np.sqrt(denom)

    return np.degrees(np.arccos(z / R))

# -----------------------------------------------------
# likelihood: gaussian shell around ellipsoid
# -----------------------------------------------------
def neg_loglike(params, theta, phi, use_full):
    if use_full:
        a, b, c, phi0, sigma = params
    else:
        a, b, sigma = params
        c = a
        phi0 = 0.0

    sigma = max(sigma, 1e-3)

    R = R_ellipsoid(theta, phi, a, b, c, phi0)
    resid = theta - R

    return 0.5 * np.sum((resid / sigma)**2 + np.log(2*np.pi*sigma**2))

def AIC(nll, k):
    return 2*k + 2*nll

# -----------------------------------------------------
# monte-carlo significance (shuffle phi)
# -----------------------------------------------------
def mc_test(theta, phi, res_null, k_null, res_full, k_full, nsim=2000):
    nll_null = res_null.fun
    nll_full = res_full.fun
    delta_real = AIC(nll_null, k_null) - AIC(nll_full, k_full)

    deltas = []

    for _ in range(nsim):
        phi_shuf = np.random.permutation(phi)
        res_mc = minimize(
            neg_loglike, 
            x0=res_full.x,
            args=(theta, phi_shuf, True),
            bounds=[(0.2,5),(0.2,5),(0.2,5),(-np.pi,np.pi),(1,40)]
        )
        if res_mc.success:
            delta = AIC(nll_null, k_null) - AIC(res_mc.fun, k_full)
            deltas.append(delta)

    deltas = np.array(deltas)
    if len(deltas)==0:
        return 0.0, delta_real

    p = np.mean(deltas >= delta_real)
    return p, delta_real

# -----------------------------------------------------
# main
# -----------------------------------------------------
def main():
    print("===================================================")
    print("3D ELLIPSOIDAL SHELL FIT AROUND UNIFIED AXIS")
    print("===================================================")

    df = pd.read_csv("frbs.csv")
    df = df.dropna(subset=["ra","dec","z_est"])

    theta, phi = compute_axis_frame(df["ra"], df["dec"])
    mask = (theta>=THETA_MIN) & (theta<=THETA_MAX)

    theta_s = theta[mask]
    phi_s = phi[mask]

    print(f"using {len(theta_s)} frbs in shell {THETA_MIN}–{THETA_MAX} deg")

    # -------------------------------------------------
    # null model: axisymmetric ellipsoid (a=b=c)
    # -------------------------------------------------
    a0 = 1.0
    sigma0 = 10.0
    res_null = minimize(
        neg_loglike,
        x0=[1.0, 1.0, sigma0],
        args=(theta_s, phi_s, False),
        bounds=[(0.2,5),(0.2,5),(1,40)]
    )
    k_null = 3

    # -------------------------------------------------
    # full ellipsoid (triaxial)
    # -------------------------------------------------
    res_full = minimize(
        neg_loglike,
        x0=[1.0, 1.2, 0.8, 0.0, 8.0],
        args=(theta_s, phi_s, True),
        bounds=[(0.2,5),(0.2,5),(0.2,5),(-np.pi,np.pi),(1,40)]
    )
    k_full = 5

    # results
    a_null,b_null,sig_null = res_null.x
    a_f,b_f,c_f,phi0_f,sig_f = res_full.x

    print("---------------------------------------------------")
    print("AXISYMMETRIC ELLIPSOID (a=b=c) FIT")
    print("---------------------------------------------------")
    print(f"a=b=c ≈ {a_null:.3f}")
    print(f"sigma = {sig_null:.2f} deg")
    print(f"AIC = {AIC(res_null.fun,k_null):.2f}")
    print()

    print("---------------------------------------------------")
    print("FULL TRIAXIAL ELLIPSOID FIT")
    print("---------------------------------------------------")
    print(f"a = {a_f:.3f}")
    print(f"b = {b_f:.3f}")
    print(f"c = {c_f:.3f}")
    print(f"rotation φ0 = {np.degrees(phi0_f):.2f} deg")
    print(f"sigma = {sig_f:.2f} deg")
    print(f"AIC = {AIC(res_full.fun,k_full):.2f}")
    print()

    # significance
    print("running monte-carlo for ellipsoid significance...")
    p, delta = mc_test(theta_s, phi_s, res_null, k_null, res_full, k_full)
    print("---------------------------------------------------")
    print(f"ΔAIC (null → full) = {delta:.2f}")
    print(f"MC p-value = {p:.4g}")
    print("---------------------------------------------------")

    # visualization
    phi_plot = np.linspace(-np.pi, np.pi, 400)
    R_full = R_ellipsoid(theta_s, phi_s, a_f,b_f,c_f,phi0_f)  # scatter reference
    R_axis = R_ellipsoid(np.degrees(np.arccos(np.cos(np.radians(40)))), phi_plot, a_null,b_null,a_null,0)
    R_tri = R_ellipsoid(np.repeat(40,len(phi_plot)), phi_plot, a_f,b_f,c_f,phi0_f)

    plt.figure(figsize=(7,4))
    plt.scatter(phi_s,theta_s,s=5,alpha=0.3,label="FRBs")
    plt.plot(phi_plot,R_tri,label="triaxial shell")
    plt.xlabel("phi (rad)")
    plt.ylabel("theta (deg)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig("frb_ellipsoidal_shell_fit.png",dpi=200)

    print("saved plot: frb_ellipsoidal_shell_fit.png")
    print("analysis complete.")

if __name__ == "__main__":
    main()
