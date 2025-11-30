#!/usr/bin/env python3
import pandas as pd
import numpy as np
from math import radians, sin, cos, acos

# ============================================================
# background coincidence rate test
# robust version that auto-detects time column names
# and handles multiple datetime formats cleanly.
# ============================================================

def ang_sep(ra1, dec1, ra2, dec2):
    r1, d1 = radians(ra1), radians(dec1)
    r2, d2 = radians(ra2), radians(dec2)
    return np.degrees(
        acos(
            sin(d1)*sin(d2) +
            cos(d1)*cos(d2)*cos(r1 - r2)
        )
    )

# ------------------------------------------------------------
# load the real data
# ------------------------------------------------------------
frb = pd.read_csv("frbs.csv")
nu  = pd.read_csv("neutrinos_clean.csv")

# ------------------------------------------------------------
# auto-detect time column
# ------------------------------------------------------------
def find_time_column(df, df_name):
    candidates = ["event_time", "time", "timestamp", "utc", "date"]
    for c in candidates:
        if c in df.columns:
            print(f"[info] using '{c}' as time column for {df_name}")
            return c
    raise ValueError(f"{df_name} missing time column")

frb_time_col = find_time_column(frb, "FRBs")
nu_time_col  = find_time_column(nu,  "neutrinos")

# ------------------------------------------------------------
# clean missing values
# ------------------------------------------------------------
frb = frb.dropna(subset=["ra", "dec", frb_time_col])
nu  = nu.dropna(subset=["ra", "dec", nu_time_col])

# ------------------------------------------------------------
# robust datetime parsing
# ------------------------------------------------------------
def parse_dt(series):
    # works with ISO8601, mixed formats, and timezone offsets
    return pd.to_datetime(series, utc=True, format="ISO8601")

frb["t"] = parse_dt(frb[frb_time_col])
nu["t"]  = parse_dt(nu[nu_time_col])

# RA/DEC arrays
frb_ra  = frb["ra"].to_numpy()
frb_dec = frb["dec"].to_numpy()
frb_t   = frb["t"].astype(np.int64).to_numpy()   # nanoseconds

nu_ra   = nu["ra"].to_numpy()
nu_dec  = nu["dec"].to_numpy()
nu_t    = nu["t"].astype(np.int64).to_numpy()

# ------------------------------------------------------------
# coincidence window
# ------------------------------------------------------------
T_hours   = 6
θ_deg     = 6
T_seconds = T_hours * 3600

# ------------------------------------------------------------
# observed coincidences
# ------------------------------------------------------------
obs = 0
for i in range(len(frb)):
    for j in range(len(nu)):
        dt = abs((frb_t[i] - nu_t[j]) * 1e-9)  # seconds
        if dt <= T_seconds:
            sep = ang_sep(frb_ra[i], frb_dec[i], nu_ra[j], nu_dec[j])
            if sep <= θ_deg:
                obs += 1

# ------------------------------------------------------------
# monte carlo background
# ------------------------------------------------------------
N_mc = 20000
rng = np.random.default_rng()

nuTmin  = nu_t.min()
nuTmax  = nu_t.max()

def random_isotropic(size):
    ra  = rng.uniform(0, 360, size)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, size)))
    return ra, dec

mc_counts = []

for _ in range(N_mc):
    # randomize neutrino times uniformly in the same range
    nu_fake_t = rng.uniform(nuTmin, nuTmax, len(nu_t))

    # random sky locations
    nu_fake_ra, nu_fake_dec = random_isotropic(len(nu_t))

    c = 0
    for i in range(len(frb_t)):
        dt = abs((frb_t[i] - nu_fake_t) * 1e-9)
        idx = np.where(dt <= T_seconds)[0]
        if len(idx) > 0:
            for j in idx:
                sep = ang_sep(frb_ra[i], frb_dec[i],
                              nu_fake_ra[j], nu_fake_dec[j])
                if sep <= θ_deg:
                    c += 1

    mc_counts.append(c)

mc_counts = np.array(mc_counts)
bg_mean   = mc_counts.mean()
bg_std    = mc_counts.std()

# ------------------------------------------------------------
# output
# ------------------------------------------------------------
print("============================================================")
print("background FRB–neutrino coincidence test")
print("============================================================")
print(f"window:  Δt ≤ {T_hours} h,  Δθ ≤ {θ_deg} deg")
print("------------------------------------------------------------")
print(f"observed coincidences: {obs}")
print(f"expected background:   {bg_mean:.3f} ± {bg_std:.3f}")
print("------------------------------------------------------------")

if bg_std > 0:
    z = (obs - bg_mean)/bg_std
    print(f"significance z-score:  {z:.2f} σ")
else:
    print("significance z-score:  undefined")

print("============================================================")
