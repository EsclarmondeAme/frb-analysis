import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

print("downloading frb catalog (v2)")
print("=" * 60)

np.random.seed(42)

# ---------------------------------------------------------
# options
# ---------------------------------------------------------
inject_anisotropy = False
anisotropy_ra = 140
anisotropy_dec = 35
anisotropy_strength = 0.15

inject_cross_layer = False
cross_layer_fraction = 0.02
flux_boost = 50

include_redshift = True

# unify time window to 2019–2021 (matches neutrinos and uhecr)
start_date = datetime(2019, 1, 1)
duration_days = 365 * 3

# ---------------------------------------------------------
# try CHIME FRB download (recommended main source)
# ---------------------------------------------------------
try:
    print("trying CHIME FRB catalog...")
    chime_url = "https://www.chime-frb.ca/catalog?page=all&format=csv"
    frbs = pd.read_csv(chime_url)
    print(f"downloaded {len(frbs)} real FRBs from CHIME")

    # CHIME schema fields
    # mjd, ra, dec, width, snr, flux, dm
    # some rows may have NaN or missing mjd

    # clean up column names
    frbs.columns = [c.lower() for c in frbs.columns]

    # ensure mjd exists (CHIME uses 'mjd')
    if "mjd" not in frbs:
        raise RuntimeError("CHIME catalog missing 'mjd' column")

    # convert mjd → utc
    mjd_epoch = pd.Timestamp("1858-11-17", tz="UTC").timestamp()
    frbs["utc"] = pd.to_datetime(
        frbs["mjd"].astype(float) * 86400.0 + mjd_epoch,
        unit="s",
        utc=True
    )

    # fill required fields
    required = ["ra", "dec", "dm", "snr", "flux", "width"]
    for r in required:
        if r not in frbs:
            frbs[r] = np.nan

    # CHIME has a clean name column ("tns_name"), use fallback if needed
    if "tns_name" in frbs:
        frbs["name"] = frbs["tns_name"]
    else:
        frbs["name"] = frbs.get("frb_name", frbs.index.astype(str))

except Exception as e:
    print("CHIME FRB download failed:", e)
    print("falling back to FRBCAT…")

    try:
        # ----------------------------------------------
        # attempt FRBCAT (backup)
        # ----------------------------------------------
        print("trying FRBCAT…")
        url = "http://frbcat.org/static/download/FRBCAT_table.csv"
        frbs = pd.read_csv(url)
        print(f"downloaded {len(frbs)} FRBs from FRBCAT")

        # cleanup and synthetic timestamp injection
        colmap = {}
        for c in frbs.columns:
            lc = c.lower()
            if "ra" in lc: colmap[c] = "ra"
            if "dec" in lc: colmap[c] = "dec"
            if "dm" in lc: colmap[c] = "dm"
            if "flux" in lc: colmap[c] = "flux"
            if "snr" in lc: colmap[c] = "snr"
            if "width" in lc: colmap[c] = "width"
            if "name" in lc: colmap[c] = "name"

        frbs = frbs.rename(columns=colmap)

        # synthetic timestamps (FRBCAT lacks real times)
        frbs["utc"] = [
            (start_date + timedelta(days=float(np.random.uniform(0, duration_days))))
            for _ in range(len(frbs))
        ]

        frbs["mjd"] = (frbs["utc"] - datetime(1858, 11, 17)).dt.total_seconds() / 86400

        for r in ["flux", "snr", "width", "dm", "ra", "dec"]:
            if r not in frbs:
                frbs[r] = np.nan

    except Exception as e2:
        print("FRBCAT failed:", e2)
        print("using synthetic CHIME simulator instead")
        from create_realistic_chime import df as frbs


# ---------------------------------------------------------
# now apply optional physics upgrades
# ---------------------------------------------------------

# anisotropy
if inject_anisotropy:
    mask = np.random.rand(len(frbs)) < anisotropy_strength
    frbs.loc[mask, "ra"] = np.random.normal(anisotropy_ra, 5, mask.sum())
    frbs.loc[mask, "dec"] = np.random.normal(anisotropy_dec, 5, mask.sum())

# cross-layer synthetic bursts
if inject_cross_layer:
    mask = np.random.rand(len(frbs)) < cross_layer_fraction
    frbs.loc[mask, "flux"] *= flux_boost

# redshift
if include_redshift:
    if "dm" in frbs:
        frbs["z_est"] = frbs["dm"] / 1200
    else:
        frbs["z_est"] = np.nan
else:
    frbs["z_est"] = np.nan

# ensure the required columns exist
must = ["name","utc","mjd","ra","dec","dm","flux","width","snr","z_est"]
for m in must:
    if m not in frbs:
        frbs[m] = np.nan

# clean format + sort
frbs = frbs[must].sort_values("utc").reset_index(drop=True)
frbs.to_csv("frbs.csv", index=False)

print(f"\nfinal frb count: {len(frbs)}")
print("saved as frbs.csv")
print("=" * 60)
