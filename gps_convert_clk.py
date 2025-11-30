import gzip
import pandas as pd
import numpy as np
import re

# input file — CHANGE THIS if your filename is different
INPUT_FILE = "COD0MGXFIN_20240360000_01D_30S_CLK.CLK.gz"
OUTPUT_CSV = "gps_clock_drift_real.csv"

# regex to match clock entries
clk_pattern = re.compile(
    r"^\s*([\d]{4}\s+[\d]{3}\s+[\d]{2}\s+[\d]{2}\s+[\d]{2}\s+[\d]{2}\.\d+)\s+SAT\s+(\S+)\s+(\S+)"
)

rows = []

with gzip.open(INPUT_FILE, "rt", errors="ignore") as f:
    for line in f:
        m = clk_pattern.match(line)
        if m:
            time_str = m.group(1)
            sat = m.group(2)
            clk = float(m.group(3))

            # gps time format → pandas timestamp
            y, doy, h, mi, s, frac = time_str.split()
            doy = int(doy)
            y = int(y)
            h = int(h)
            mi = int(mi)
            s = int(float(s))

            # convert DOY to date
            date = pd.Timestamp(y, 1, 1) + pd.Timedelta(days=doy - 1) \
                   + pd.Timedelta(hours=h, minutes=mi, seconds=s)

            rows.append([date, sat, clk])

# dataframe
df = pd.DataFrame(rows, columns=["timestamp", "satellite", "clock_offset"])

# save
df.to_csv(OUTPUT_CSV, index=False)

print("done — extracted", len(df), "clock records into", OUTPUT_CSV)
