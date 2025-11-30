import gzip
import csv
from datetime import datetime

INPUT = "COD0MGXFIN_20240360000_01D_30S_CLK.CLK.gz"
OUTPUT = "gps_clock_drift_real.csv"

def parse_line(line):
    """
    Example line:
    AR BRUX00BEL 2024 02 05 00 00 00.000000  2    0.2032E-06  0.3894E-10
    station_id yyyy mm dd hh mm ss.frac sys clock_bias clock_drift
    """
    parts = line.split()
    if len(parts) < 10:
        return None

    try:
        station = parts[1]
        yyyy = int(parts[2])
        mm   = int(parts[3])
        dd   = int(parts[4])
        hh   = int(parts[5])
        minu = int(parts[6])
        ss   = float(parts[7])
        drift = float(parts[-1])   # last field = clock_drift (fractional frequency)
    except:
        return None

    # timestamp in Python datetime
    t = datetime(yyyy, mm, dd, hh, minu, int(ss))

    return station, t, drift


rows = []

with gzip.open(INPUT, "rt", errors="ignore") as f:
    header_passed = False
    for line in f:
        if "END OF HEADER" in line:
            header_passed = True
            continue
        if not header_passed:
            continue

        line_strip = line.strip()
        if not line_strip:
            continue

        parsed = parse_line(line_strip)
        if parsed:
            rows.append(parsed)

# compute reference time (first timestamp)
if not rows:
    print("no rows parsed!")
    exit()

t0 = min(t for (_, t, _) in rows)

# save CSV with time_days AND frac_freq (alias for drift)
with open(OUTPUT, "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["station", "timestamp", "clock_drift", "frac_freq", "time_days"])
    for station, t, drift in rows:
        delta = (t - t0).total_seconds() / 86400.0  # days
        frac_freq = drift  # drift is already Δf/f
        w.writerow([station, t.isoformat(), drift, frac_freq, delta])

print(f"done — extracted {len(rows)} real clock drift records")
print(f"saved to {OUTPUT}")
