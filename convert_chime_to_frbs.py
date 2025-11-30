# convert_chime_to_frbs.py
import pandas as pd
from datetime import datetime, timedelta

def mjd_to_utc(mjd):
    # mjd 0 corresponds to 1858-11-17 00:00:00
    base = datetime(1858, 11, 17)
    return base + timedelta(days=float(mjd))

def main():
    df = pd.read_csv("chime_frb_catalog1.csv")

    ra = df["ra"]
    dec = df["dec"]
    dm = df["dm_fitb"]
    snr = df["snr_fitb"]
    width = df["bc_width"]
    fluence = df["fluence"]

    # CHIME has MJD column "mjd_utc" — check
    if "mjd_utc" in df.columns:
        mjd = df["mjd_utc"]
    else:
        # fallback: extract date from name FRByyyymmddA
        mjd_list = []
        for name in df["tns_name"]:
            date = name[3:11]  # yyyymmdd
            dt = datetime.strptime(date, "%Y%m%d")
            # convert to MJD
            mjd0 = datetime(1858, 11, 17)
            mjd_list.append((dt - mjd0).days)
        mjd = pd.Series(mjd_list)

    # convert to UTC ISO strings
    utc = [mjd_to_utc(x).strftime("%Y-%m-%d %H:%M:%S") for x in mjd]

    # approximate redshift from DM
    z_est = dm / 1000.0

    out = pd.DataFrame({
        "name": df["tns_name"],
        "utc": utc,
        "mjd": mjd,
        "ra": ra,
        "dec": dec,
        "dm": dm,
        "snr": snr,
        "width": width,
        "fluence": fluence,
        "z_est": z_est
    })

    out.to_csv("frbs.csv", index=False)
    print("saved frbs.csv with", len(out), "FRBs")

if __name__ == "__main__":
    main()
