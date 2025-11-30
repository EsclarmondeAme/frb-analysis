import requests
import pandas as pd
import numpy as np
import re
from datetime import datetime, timezone

print("processing neutrino alerts (real icecube gold/bronze)")
print("=" * 60)

np.random.seed(44)

# -------------------------------------------------------------------
# helper: extract first floating point number from a table cell
# -------------------------------------------------------------------
float_re = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

def extract_float(cell):
    """
    take raw html cell content, extract first float-like token.
    return float or raise ValueError if none found.
    """
    if cell is None:
        raise ValueError("empty cell")
    m = float_re.search(str(cell))
    if not m:
        raise ValueError(f"no float in cell: {cell!r}")
    return float(m.group(0))


# ============================================================
# download and parse the amon icecube gold/bronze event table
# ============================================================
try:
    url = "https://gcn.gsfc.nasa.gov/amon_icecube_gold_bronze_events.html"
    print(f"downloading: {url}")

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    html = resp.text

    # parse table rows and cells
    tr_pat = re.compile(r"<tr[^>]*>(.*?)</tr>", re.I | re.S)
    td_pat = re.compile(r"<td[^>]*>(.*?)</td>", re.I | re.S)

    rows = []

    for tr in tr_pat.findall(html):
        tds = td_pat.findall(tr)

        # skip header or malformed rows
        # we expect at least:
        # 0: RunNum_EventNum (with link)
        # 1: Rev
        # 2: Date  (yy/mm/dd)
        # 3: Time  (hh:mm:ss.ss)
        # 4: NoticeType (GOLD/BRONZE)
        # 5: RA [deg]
        # 6: Dec [deg]
        # ...
        if len(tds) < 11:
            continue

        run_evt = re.sub(r"<.*?>", "", tds[0]).strip()
        rev     = re.sub(r"<.*?>", "", tds[1]).strip()
        date_s  = re.sub(r"<.*?>", "", tds[2]).strip()
        time_s  = re.sub(r"<.*?>", "", tds[3]).strip()
        ntype   = re.sub(r"<.*?>", "", tds[4]).strip()

        # basic sanity: must look like a real row
        if not run_evt or not date_s or not time_s:
            continue

        # parse date/time → utc datetime
        # date is in yy/mm/dd format (as documented on the page)
        try:
            dt_date = datetime.strptime(date_s, "%y/%m/%d")
        except ValueError:
            # if format somehow different, skip this row
            continue

        # time is hh:mm:ss.ss
        try:
            dt_time = datetime.strptime(time_s, "%H:%M:%S.%f").time()
        except ValueError:
            try:
                dt_time = datetime.strptime(time_s, "%H:%M:%S").time()
            except ValueError:
                continue

        utc = datetime(
            year=dt_date.year,
            month=dt_date.month,
            day=dt_date.day,
            hour=dt_time.hour,
            minute=dt_time.minute,
            second=dt_time.second,
            microsecond=dt_time.microsecond,
            tzinfo=timezone.utc,
        )

        # ra, dec, energy, signalness from numeric columns
        try:
            ra  = extract_float(tds[5])
            dec = extract_float(tds[6])
            energy = extract_float(tds[9])      # "Energy" column
            signal = extract_float(tds[10])     # "Signalness" column
        except ValueError:
            # if any key numeric field is missing, skip row
            continue

        event_id = f"{run_evt}_rev{rev}"

        rows.append(
            (
                event_id,
                utc,
                ra,
                dec,
                energy,
                signal,
                ntype,
            )
        )

    if not rows:
        raise RuntimeError("html parsed but produced zero usable rows")

    # build dataframe in the schema your pipeline expects
    df = pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "utc",
            "ra",
            "dec",
            "energy_tev",
            "signalness",
            "notice_type",
        ],
    )

    # sort by time
    df["utc"] = pd.to_datetime(df["utc"], utc=True)
    df = df.sort_values("utc").reset_index(drop=True)

    # add mjd and z_est placeholder
    mjd0 = datetime(1858, 11, 17, tzinfo=timezone.utc)
    df["mjd"] = (df["utc"] - mjd0).dt.total_seconds() / 86400.0
    df["z_est"] = np.nan

    # save
    df.to_csv("neutrinos_clean.csv", index=False)

    print(f"loaded {len(df)} real IceCube gold/bronze alerts")
    print("saved → neutrinos_clean.csv")
    print("============================================================")

except Exception as e:
    print("FATAL: failed to process real IceCube alerts")
    print("error:", e)
    print("============================================================")
    print("no synthetic fallback allowed — stopping.")
    raise SystemExit(1)
