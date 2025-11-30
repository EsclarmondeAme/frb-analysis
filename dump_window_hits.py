import numpy as np
import pandas as pd
from datetime import timedelta
from scan_coincidence_windows import angular_sep_deg, pair_score

print("============================================================")
print("dumping the 2 wide-window frb–neutrino candidates")
print("============================================================")

frbs = pd.read_csv("frbs.csv")
neus = pd.read_csv("neutrinos_clean.csv")

frbs["utc"] = pd.to_datetime(frbs["utc"], utc=True, errors="coerce")
neus["utc"] = pd.to_datetime(neus["utc"], utc=True, errors="coerce")

frbs = frbs.dropna(subset=["utc"])
neus = neus.dropna(subset=["utc"])

overlap_start = max(frbs["utc"].min(), neus["utc"].min())
overlap_end   = min(frbs["utc"].max(), neus["utc"].max())

frbs = frbs[(frbs["utc"]>=overlap_start)&(frbs["utc"]<=overlap_end)]
neus = neus[(neus["utc"]>=overlap_start)&(neus["utc"]<=overlap_end)]

# wide windows where matches appeared
T_list = [21600.0, 86400.0]   # 6h, 1 day
th_list = [10.0, 20.0, 40.0]

hits = []

for _, frb in frbs.iterrows():
    for _, nu in neus.iterrows():
        dt_s = abs((frb["utc"] - nu["utc"]).total_seconds())
        ang_deg = angular_sep_deg(frb["ra"], frb["dec"], nu["ra"], nu["dec"])

        for T in T_list:
            for th in th_list:
                if dt_s <= T and ang_deg <= th:
                    score = pair_score(frb, nu, dt_s, ang_deg, T, th)
                    hits.append({
                        "frb_name": frb.get("name", frb.get("frb_name", "unknown")),
                        "frb_time": frb["utc"],
                        "nu_id": nu.get("event_id", "unknown"),
                        "nu_time": nu["utc"],
                        "dt_s": dt_s,
                        "ang_deg": ang_deg,
                        "score": score,
                        "T_window": T,
                        "ang_window": th
                    })

df = pd.DataFrame(hits)

if len(df)==0:
    print("no candidates (should not happen)")
else:
    # sort by score descending
    df = df.sort_values("score", ascending=False)
    df.to_csv("candidate_pairs.csv", index=False)
    print(df.head(10))
    print("saved full list to candidate_pairs.csv")
