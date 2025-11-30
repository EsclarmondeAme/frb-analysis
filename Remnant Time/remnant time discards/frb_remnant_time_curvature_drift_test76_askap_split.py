import numpy as np
import sys, os
from math import radians, sin, cos, sqrt
from astropy.io import fits
import math

# --------------------------------------------------
# configuration
# --------------------------------------------------
ASKAP_FITS_DIR = r"data/positions"          # your actual directory
MATCH_TOL = 0.2                              # degrees
N_MC = 2000
# unified-axis direction (galactic coords approx)
AXIS_GAL_XYZ = np.array([0.15988173, -0.87645602, 0.45492345])

# --------------------------------------------------
def sph_to_xyz(ra, dec):
    ra = np.radians(ra)
    dec = np.radians(dec)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T

def angsep(ra1, dec1, ra2_arr, dec2_arr):
    """compute angular separation between one FRB and many ASKAP pointings"""
    ra1 = radians(ra1)
    dec1 = radians(dec1)

    ra2 = np.radians(ra2_arr)
    dec2 = np.radians(dec2_arr)

    cosang = (np.sin(dec1)*np.sin(dec2) +
              np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2))

    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


# --------------------------------------------------
# load unified CSV
# --------------------------------------------------
def load_unified(path):
    RA, Dec = [], []
    with open(path,"r") as f:
        next(f)  # skip header
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            RA.append(float(parts[3]))
            Dec.append(float(parts[4]))
    return np.array(RA), np.array(Dec)

# --------------------------------------------------
# load ASKAP pointing centers from FITS
# --------------------------------------------------
def load_askap_fits():
    ras = []
    decs = []
    files = [f for f in os.listdir(ASKAP_FITS_DIR) if f.lower().endswith(".fits")]
    for fn in files:
        try:
            h = fits.getheader(os.path.join(ASKAP_FITS_DIR, fn))
            ras.append(float(h["CRVAL1"]))
            decs.append(float(h["CRVAL2"]))
        except:
            pass
    return np.array(ras), np.array(decs)


# --------------------------------------------------
# classify ASKAP vs non-ASKAP
# --------------------------------------------------
def classify_askap(RA, Dec, ask_ra, ask_dec, tol=1.0):
    """return boolean array: True if FRB is within tol deg of any ASKAP tile"""
    flags = np.zeros(len(RA), dtype=bool)
    for i in range(len(RA)):
        d = angsep(RA[i], Dec[i], ask_ra, ask_dec)
        if np.min(d) <= tol:
            flags[i] = True
    return flags

# --------------------------------------------------
# compute curvature–drift statistic
# --------------------------------------------------
def curvature_drift_stat(xyz, axis):
    dots = xyz @ axis
    pos = np.where(dots >= 0)[0]
    neg = np.where(dots < 0)[0]

    if len(pos)==0 or len(neg)==0:
        return np.nan, np.nan, np.nan

    # curvature proxy = local second-moment of neighborhood
    # drift proxy      = mean angular displacement around axis
    # (simple but consistent with A/B versions)
    # --------------------------------------------------
    vpos = xyz[pos]
    vneg = xyz[neg]

    # curvature proxy ~ mean(1 - dot product)
    c_pos = 1 - np.mean(vpos @ vpos.T)
    c_neg = 1 - np.mean(vneg @ vneg.T)

    # drift proxy ~ mean(angular shift relative to axis)
    ang_pos = np.degrees(np.arccos(np.clip((vpos @ axis), -1, 1)))
    ang_neg = np.degrees(np.arccos(np.clip((vneg @ axis), -1, 1)))

    d_pos = np.mean(ang_pos)
    d_neg = np.mean(ang_neg)

    D = (c_pos - c_neg) + (d_pos - d_neg)
    return c_pos-c_neg, d_pos-d_neg, D

# --------------------------------------------------
# Monte Carlo null (rotate axis)
# --------------------------------------------------
def random_axis():
    v = np.random.normal(size=3)
    return v / np.linalg.norm(v)

def mc_null(xyz, Sreal):
    null_S = []
    for _ in range(N_MC):
        ax = random_axis()
        _, _, D = curvature_drift_stat(xyz, ax)
        null_S.append(D)
    null_S = np.array(null_S)
    mean = np.nanmean(null_S)
    std  = np.nanstd(null_S)

    if std==0 or np.isnan(std):
        p = 1.0
    else:
        # two-sided
        z = abs(Sreal - mean) / std
        p = 2 * (1 - 0.5*(1 + math.erf(z / math.sqrt(2))))

        p = max(min(p,1.0),0.0)

    return mean, std, p

# --------------------------------------------------
def main(path):
    print("===============================================")
    print("FRB REMNANT-TIME CURVATURE–DRIFT TEST (76C — ASKAP split)")
    print("===============================================")

    RA, Dec = load_unified(path)
    ask_ra, ask_dec = load_askap_fits()

    print(f"[info] loaded {len(ask_ra)} ASKAP pointing centers")

    X = sph_to_xyz(RA, Dec)

    is_A = classify_askap(RA, Dec, ask_ra, ask_dec, tol=MATCH_TOL)

    nA = np.sum(is_A)
    nN = len(RA) - nA

    print(f"ASKAP count     = {nA}")
    print(f"non-ASKAP count = {nN}")

    # subsets
    X_A = X[is_A]
    X_N = X[~is_A]

    # compute real stats
    print("------------------------------------------------")
    if nA > 1:
        cp, dp, SA = curvature_drift_stat(X_A, AXIS_GAL_XYZ)
    else:
        cp = dp = SA = np.nan

    cn, dn, SN = curvature_drift_stat(X_N, AXIS_GAL_XYZ)

    # Monte Carlo
    if nA > 1:
        meanA, stdA, pA = mc_null(X_A, SA)
    else:
        meanA = stdA = np.nan
        pA = 1.0

    meanN, stdN, pN = mc_null(X_N, SN)

    print(f"ASKAP:    D_real={SA}, null_mean={meanA}, null_std={stdA}, p={pA}")
    print(f"nonASKAP: D_real={SN}, null_mean={meanN}, null_std={stdN}, p={pN}")

    print("------------------------------------------------")
    print("interpretation:")
    print("  low p  -> curvature–drift asymmetry in subset")
    print("  high p -> subset consistent with isotropy")
    print("===============================================")
    print("test 76C complete.")
    print("===============================================")


if __name__ == "__main__":
    main(sys.argv[1])
