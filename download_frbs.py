import pandas as pd
import numpy as np
import requests
from datetime import datetime
from math import isnan

print("downloading chime frb catalog")
print("=" * 60)

np.random.seed(42)

# ---------------------------------------------------------
# settings
# ---------------------------------------------------------
inject_anisotropy = False
anisotropy_ra = 150
anisotropy_dec = 20
anisotropy_strength = 0.10

inject_cross_layer = False
cross_layer_fraction = 0.02
flux_boost = 50

include_redshift = True   # add crude z estimate from DM

# ---------------------------------------------------------
# primary CHIME public REST URL
# ---------------------------------------------------------
url = "https://www.chime-frb.ca/api/frb_master?format=csv"

try:
    print("downloading from chime api...")
    r = requests.get(url, timeout=25)
    r.raise_for_status()

    # save raw file
    with open("chime_frbs.csv", "wb") as f:
        f.write(r.content)

    df = pd.read_csv("chime_frbs.csv")
    print(f"downloaded {len(df)} frbs")

    # rename some columns to standardize
    # chime uses: ra_deg, dec_deg, dm, snr, flux, width
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if "ra" in lc and "deg" in lc: colmap[c] = "ra"
        if "dec" in lc and "deg" in lc: colmap[c] = "dec"
        if lc == "dm": colmap[c] = "dm"
        if "flux" in lc: colmap[c] = "flux"
        if "snr" in lc: colmap[c] = "snr"
        if "width" in lc: colmap[c] = "width"
        if "mjd" in lc: colmap[c] = "mjd"
        if "id" in lc: colmap[c] = "name"

    df = df.rename(columns=colmap)

    # ensure required columns exist
    if "mjd" not in df:
        print("no mjd found in catalog, generating synthetic dates")
        # synthetic time window (match neutrinos + uhecr: 2019–2021)
        start_date = datetime(2019, 1, 1)
        mjd0 = (start_date - datetime(1858, 11, 17)).total_seconds() / 86400
        df["mjd"] = mjd0 + np.random.uniform(0, 365*3, len(df))

    # convert to utc
    df["utc"] = pd.to_datetime(
        df["mjd"] * 86400 + datetime(1858, 11, 17).timestamp(),
        unit="s",
        errors="coerce"
    )

    # fill missing ra/dec if necessary
    if "ra" not in df: df["ra"] = np.random.uniform(0, 360, len(df))
    if "dec" not in df: df["dec"] = np.random.uniform(-90, 90, len(df))

    # fill missing dm/flux/snr/width
    if "dm" not in df: df["dm"] = np.random.uniform(100, 1500, len(df))
    if "flux" not in df: df["flux"] = np.random.lognormal(mean=np.log(2), sigma=1.0, size=len(df))
    if "snr" not in df: df["snr"] = np.random.uniform(8, 100, len(df))
    if "width" not in df: df["width"] = np.random.gamma(2, 2, len(df)) + 0.5

    # ---------------------------------------------------------
    # optional anisotropy (for testing)
    # ---------------------------------------------------------
    if inject_anisotropy:
        mask = np.random.rand(len(df)) < anisotropy_strength
        df.loc[mask, "ra"] = np.random.normal(anisotropy_ra, 5, mask.sum())
        df.loc[mask, "dec"] = np.random.normal(anisotropy_dec, 5, mask.sum())

    # ---------------------------------------------------------
    # optional cross-layer FRBs
    # ---------------------------------------------------------
    if inject_cross_layer:
        mask = np.random.rand(len(df)) < cross_layer_fraction
        df.loc[mask, "flux"] *= flux_boost

    # ---------------------------------------------------------
    # redshift estimate from DM (very rough)
    # ---------------------------------------------------------
    if include_redshift:
        df["z_est"] = df["dm"] / 1200
    else:
        df["z_est"] = np.nan

    # ---------------------------------------------------------
    # final cleanup and save
    # ---------------------------------------------------------
    df = df[["name", "utc", "mjd", "ra", "dec", "dm", "flux", "width", "snr", "z_est"]]
    df = df.sort_values("utc").reset_index(drop=True)
    df.to_csv("frbs.csv", index=False)

    print("clean frb file saved as frbs.csv")
    print("=" * 60)
    exit(0)

except Exception as e:
    print("failed to download chime catalog")
    print("error:", e)
    print("using fallback synthetic frb set")
    print("-" * 60)

# ---------------------------------------------------------
# fallback synthetic FRBs (if API is down)
# ---------------------------------------------------------
from create_realistic_chime import df as chime_synth   # uses your upgraded generator
chime_synth.to_csv("frbs.csv", index=False)

print("fallback synthetic frbs saved as frbs.csv")
print("=" * 60)
