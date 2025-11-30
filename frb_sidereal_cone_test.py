import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import chisquare

# ============================================================
# sidereal phase utilities
# ============================================================

def sidereal_phase(times):
    gmst = times.sidereal_time("mean", 0 * u.deg).hour
    return (gmst / 24.0) % 1.0


def safe_hist(phases, bins=24):
    H, edges = np.histogram(phases, bins=bins, range=(0, 1))
    expected = np.ones_like(H) * (len(phases) / bins)
    return H, expected, edges

def safe_chisq(H, E):
    if np.any(E == 0):
        E = np.where(E == 0, 1e-9, E)
    if H.sum() == 0:
        return np.nan, np.nan
    return chisquare(f_obs=H, f_exp=E)

# ============================================================
# load data
# ============================================================

frb = pd.read_csv("frbs.csv")
nu  = pd.read_csv("neutrinos.csv")

frb = frb.dropna(subset=["ra", "dec", "utc"]).copy()
nu  = nu.dropna(subset=["ra", "dec", "utc"]).copy()

# ============================================================
# strict astropy-compatible cleaner
# ============================================================

def fix_time_column(series):
    clean = []
    for t in series.astype(str):
        t = t.strip()
        if t.lower() == "nan" or t == "":
            clean.append(np.nan)
            continue

        # replace first space with T
        if " " in t:
            t = t.replace(" ", "T", 1)

        # remove trailing Z
        if t.endswith("Z"):
            t = t[:-1]

        # remove timezone offsets (+00:00)
        if "+" in t:
            t = t.split("+")[0]

        # ensure format YYYY-MM-DDTHH:MM:SS
        # if no seconds, add :00
        if t.count(":") == 1:
            t = t + ":00"

        clean.append(t)
    return pd.Series(clean)

frb["utc_clean"] = fix_time_column(frb["utc"])
nu["utc_clean"]  = fix_time_column(nu["utc"])

frb = frb.dropna(subset=["utc_clean"])
nu  = nu.dropna(subset=["utc_clean"])

print("============================================================")
print("frb sidereal cone-axis modulation test")
print("============================================================")

# ============================================================
# parse using astropy
# ============================================================

frb_times = Time(frb["utc_clean"].tolist(), format="isot", scale="utc")
nu_times  = Time(nu["utc_clean"].tolist(),  format="isot", scale="utc")

frb_phase = sidereal_phase(frb_times)
nu_phase  = sidereal_phase(nu_times)

print(f"FRBs loaded and parsed: {len(frb_phase)}")
print(f"neutrinos parsed:       {len(nu_phase)}")

# ============================================================
# histograms + χ² tests
# ============================================================

bins = 24

H_frb, E_frb, _ = safe_hist(frb_phase, bins)
H_nu,  E_nu,  _ = safe_hist(nu_phase, bins)
H_all = H_frb + H_nu
E_all = E_frb + E_nu

chi_frb, p_frb = safe_chisq(H_frb, E_frb)
chi_nu,  p_nu  = safe_chisq(H_nu, E_nu)
chi_all, p_all = safe_chisq(H_all, E_all)

print("------------------------------------------------------------")
print("chi-square tests for uniformity:")
print(f"FRB:      χ²={chi_frb:.3f},   p={p_frb:.3f}")
print(f"neutrino: χ²={chi_nu:.3f},   p={p_nu:.3f}")
print(f"combined: χ²={chi_all:.3f},  p={p_all:.3f}")
print("------------------------------------------------------------")

def interpretation(p):
    if np.isnan(p):
        return "not enough statistics"
    if p < 0.01:
        return "strong deviation from isotropy"
    if p < 0.05:
        return "mild deviation"
    return "consistent with isotropy"

print("interpretation:")
print(f" • FRBs:      {interpretation(p_frb)}")
print(f" • neutrinos: {interpretation(p_nu)}")
print(f" • combined:  {interpretation(p_all)}")
print("============================================================")
