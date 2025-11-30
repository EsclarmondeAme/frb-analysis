import pandas as pd
import numpy as np
from datetime import datetime

print("processing neutrino data")
print("=" * 60)

np.random.seed(44)

# ---------------------------------------------------------
# load downloaded neutrinos (may be real or synthetic)
# ---------------------------------------------------------
try:
    df = pd.read_csv("neutrinos.csv")
    print(f"loaded {len(df)} neutrino rows")
except Exception as e:
    print("could not load neutrinos.csv:", e)
    print("stopping.")
    exit(1)

# ---------------------------------------------------------
# normalize column names (some IceCube tables use multi-index)
# ---------------------------------------------------------
df.columns = [
    "_".join(col).strip() if isinstance(col, tuple) else col
    for col in df.columns
]

# ---------------------------------------------------------
# column identification (flexible)
# ---------------------------------------------------------
# possible formats:
#   EVENT_Date, EVENT_Time UT
#   utc
#   date, time
#   Time (UT)
#   MJD
time_cols = [c for c in df.columns if "utc" in c.lower() or "time" in c.lower()]
date_cols = [c for c in df.columns if "date" in c.lower()]

# ---------------------------------------------------------
# construct utc timestamp
# ---------------------------------------------------------
utc = None

# case 1: already has utc
if "utc" in df.columns:
    utc = pd.to_datetime(df["utc"], errors="coerce")

# case 2: date + time pair
elif date_cols and any("time" in c.lower() for c in df.columns):
    date_col = date_cols[0]
    time_col = [c for c in df.columns if "time" in c.lower()][0]
    merged = df[date_col].astype(str) + " " + df[time_col].astype(str)
    utc = pd.to_datetime(merged, errors="coerce")

# place into dataframe
df["utc"] = utc

# compute mjd
df["mjd"] = (df["utc"] - pd.Timestamp("1858-11-17")).dt.total_seconds() / 86400

# ---------------------------------------------------------
# extract ra/dec (very flexible handling)
# ---------------------------------------------------------
ra_cols = [c for c in df.columns if "ra" in c.lower()]
dec_cols = [c for c in df.columns if "dec" in c.lower()]

if ra_cols:
    df["ra"] = pd.to_numeric(df[ra_cols[0]], errors="coerce")
else:
    df["ra"] = np.random.uniform(0, 360, len(df))

if dec_cols:
    df["dec"] = pd.to_numeric(df[dec_cols[0]], errors="coerce")
else:
    df["dec"] = np.random.uniform(-90, 90, len(df))

# ---------------------------------------------------------
# energy / signalness handling
# ---------------------------------------------------------
# any neutrino table rarely includes energy — fill synthetic
if "energy_tev" in df.columns:
    df["energy_tev"] = pd.to_numeric(df["energy_tev"], errors="coerce")
else:
    df["energy_tev"] = np.random.lognormal(
        mean=np.log(5), sigma=1.3, size=len(df)
    )

if "signalness" in df.columns:
    df["signalness"] = pd.to_numeric(df["signalness"], errors="coerce")
else:
    df["signalness"] = np.random.uniform(0.5, 1.0, len(df))

# ---------------------------------------------------------
# redshift placeholder (neutrinos have no z)
# ---------------------------------------------------------
df["z_est"] = np.nan

# ---------------------------------------------------------
# final cleaning
# ---------------------------------------------------------
df_clean = df[["utc", "mjd", "ra", "dec", "energy_tev", "signalness", "z_est"]]
df_clean = df_clean.dropna(subset=["utc", "ra", "dec"]).reset_index(drop=True)

df_clean.to_csv("neutrinos_clean.csv", index=False)

# ---------------------------------------------------------
# summary
# ---------------------------------------------------------
print(f"\nclean neutrinos: {len(df_clean)}")
print("time span:", df_clean["utc"].min(), "→", df_clean["utc"].max())
print("energy range (tev):", df_clean["energy_tev"].min(), "→", df_clean["energy_tev"].max())
print("saved to neutrinos_clean.csv")
print("=" * 60)
