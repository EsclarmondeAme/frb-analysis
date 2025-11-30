import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import sph_harm
import warnings
warnings.filterwarnings("ignore")

import astropy.units as u
from astropy.coordinates import SkyCoord
from numpy.random import default_rng

######################################################
# unified FRB axis from CHIME results
######################################################
UNIFIED_L = 159.85
UNIFIED_B = -0.51

######################################################
# angular distance (vectorized)
######################################################
def angdist(ra1, dec1, ra2, dec2):
    ra1 = np.deg2rad(ra1)
    dec1 = np.deg2rad(dec1)
    ra2 = np.deg2rad(ra2)
    dec2 = np.deg2rad(dec2)

    return np.rad2deg(
        np.arccos(
            np.sin(dec1)*np.sin(dec2) +
            np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2)
        )
    )

######################################################
# rotate sky into unified-axis reference frame
######################################################
def rotate_to_axis(ra, dec, L0=UNIFIED_L, B0=UNIFIED_B):
    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    pole = SkyCoord(l=L0*u.deg, b=B0*u.deg, frame="galactic").icrs

    off = c.skyoffset_frame(pole)
    lon = off.lon.wrap_at(180*u.deg).deg
    lat = off.lat.deg

    theta = 90 - lat
    phi = lon % 360
    return theta, phi

######################################################
# multipole extraction ℓ ≤ 10
######################################################
def compute_multipoles(theta, phi, lmax=10):
    theta_r = np.deg2rad(theta)
    phi_r = np.deg2rad(phi)

    a_lm = {}
    for ell in range(lmax+1):
        for m in range(-ell, ell+1):
            Y = sph_harm(m, ell, phi_r, theta_r)
            a_lm[(ell,m)] = np.sum(Y).real

    Cl = {}
    for ell in range(lmax+1):
        Cl[ell] = sum(a_lm[(ell,m)]**2 for m in range(-ell, ell+1))

    return a_lm, Cl

######################################################
# broken-power radial model
######################################################
def broken_power(theta, t_break, A1, alpha1, A2, alpha2):
    out = np.zeros_like(theta, dtype=float)
    mask = theta < t_break
    out[mask]  = A1 * (theta[mask] / t_break)**alpha1
    out[~mask] = A2 * (theta[~mask] / t_break)**alpha2
    return out

######################################################
# φ-anisotropy test
######################################################
def phi_anisotropy(theta, phi, tmax=60):
    sel = theta <= tmax
    phi_sel = phi[sel]
    nbins = 12
    H, _ = np.histogram(phi_sel, bins=np.linspace(0,360,nbins+1))
    E = np.mean(H)
    return np.sum((H - E)**2 / (E + 1e-9))

######################################################
# convert p → −log10(p) (log-likelihood score)
######################################################
def loglike_from_p(p):
    if p <= 1e-300:
        return 50
    return -np.log10(p)

######################################################
# main pipeline
######################################################
def main():
    print("=======================================================")
    print("ASKAP CROSS-DATASET REPLICATION — UNIFIED COSMIC TEST")
    print("=======================================================\n")

    ######################################################
    # load ASKAP CSV
    ######################################################
    df = pd.read_csv("askap_frbs.csv")
    print(f"loaded ASKAP FRBs: {len(df)}\n")

    RA = df["ra"].values
    DEC = df["dec"].values

    # rotate ASKAP sky to unified-axis frame
    theta, phi = rotate_to_axis(RA, DEC)
    print(f"theta range: {theta.min():.2f}° – {theta.max():.2f}°\n")

    rng = default_rng(0)

    ######################################################
    # 1. axis alignment test
    ######################################################
    X = np.cos(np.deg2rad(theta))
    amp_real = np.mean(X)

    null_amp = []
    for _ in range(5000):
        sh = rng.permutation(theta)
        null_amp.append(np.mean(np.cos(np.deg2rad(sh))))
    null_amp = np.array(null_amp)

    p_axis = np.mean(null_amp >= amp_real)

    print("axis alignment")
    print(f"  amplitude      = {amp_real:.4f}")
    print(f"  p-value        = {p_axis:.4e}\n")

    ######################################################
    # 2. multipole test (quadrupole)
    ######################################################
    _, Cl = compute_multipoles(theta, phi, lmax=2)
    C2_real = Cl[2]

    null_C2 = []
    for _ in range(2000):
        sh = rng.permutation(theta)
        _, Cl_sh = compute_multipoles(sh, phi, lmax=2)
        null_C2.append(Cl_sh[2])
    null_C2 = np.array(null_C2)

    p_quad = np.mean(null_C2 >= C2_real)

    print("quadrupole test")
    print(f"  C2             = {C2_real:.4f}")
    print(f"  p-value        = {p_quad:.4e}\n")

    ######################################################
    # 3. radial break test (broken-power)
    ######################################################
    bins = np.linspace(0,140,30)
    H, _ = np.histogram(theta, bins=bins)
    tmid = 0.5*(bins[:-1] + bins[1:])
    p0 = [25, 1.0, 0.1, 1.0, 0.0]

    popt, _ = curve_fit(broken_power, tmid, H, p0=p0, maxfev=20000)
    Hfit = broken_power(tmid, *popt)
    RSS = np.sum((H - Hfit)**2)
    AIC = len(H)*np.log(RSS/len(H)) + 2*5

    null_AIC = []
    for _ in range(2000):
        sh = rng.permutation(theta)
        Hn,_ = np.histogram(sh, bins=bins)
        try:
            pn,_ = curve_fit(broken_power, tmid, Hn, p0=p0, maxfev=20000)
            Hfn = broken_power(tmid, *pn)
            RSSn = np.sum((Hn - Hfn)**2)
            AICn = len(H)*np.log(RSSn/len(H)) + 2*5
            null_AIC.append(AICn)
        except:
            pass

    null_AIC = np.array(null_AIC)
    p_break = np.mean(null_AIC <= AIC)

    print("radial break test")
    print(f"  AIC(broken)    = {AIC:.3f}")
    print(f"  p-value        = {p_break:.4e}\n")

    ######################################################
    # 4. φ-anisotropy
    ######################################################
    chi2_real = phi_anisotropy(theta, phi)
    chi2_null = []
    for _ in range(2000):
        sh = rng.permutation(phi)
        chi2_null.append(phi_anisotropy(theta, sh))
    chi2_null = np.array(chi2_null)

    p_phi = np.mean(chi2_null >= chi2_real)

    print("phi-anisotropy")
    print(f"  χ²             = {chi2_real:.2f}")
    print(f"  p-value        = {p_phi:.4e}\n")

    ######################################################
    # combined log-likelihood
    ######################################################
    LL = (
        loglike_from_p(p_axis) +
        loglike_from_p(p_quad) +
        loglike_from_p(p_break) +
        loglike_from_p(p_phi)
    )

    print("=======================================================")
    print("UNIFIED COSMIC LIKELIHOOD (ASKAP replication)")
    print("=======================================================")
    print(f"Total −log10(p_combined) = {LL:.3f}")
    print("=======================================================\n")

    print("analysis complete.\n")

if __name__ == "__main__":
    main()
