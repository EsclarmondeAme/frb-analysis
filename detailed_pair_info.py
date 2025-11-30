import pandas as pd
from datetime import timedelta
from scan_coincidence_windows import angular_sep_deg

frbs = pd.read_csv("frbs.csv")
neus = pd.read_csv("neutrinos_clean.csv")

frbs["utc"] = pd.to_datetime(frbs["utc"], utc=True, errors="coerce")
neus["utc"] = pd.to_datetime(neus["utc"], utc=True, errors="coerce")

frb = frbs[frbs["name"]=="FRB20190630C"].iloc[0]

nu_ids = ["132768_5390846_rev0", "132768_5390846_rev1"]
for nu_id in nu_ids:
    nu = neus[neus["event_id"]==nu_id].iloc[0]
    dt = (nu["utc"] - frb["utc"]).total_seconds()
    ang = angular_sep_deg(frb["ra"], frb["dec"], nu["ra"], nu["dec"])
    
    print("====================================================")
    print(f"neutrino id: {nu_id}")
    print("----------------------------------------------------")
    print(f"FRB time:      {frb['utc']}")
    print(f"neutrino time: {nu['utc']}")
    print(f"Δt (seconds):  {dt}")
    print(f"Δt (hours):    {dt/3600.0:.3f}")
    print(f"angular sep:   {ang:.3f} deg")
    print(f"neutrino energy: {nu.get('reco_energy', 'N/A')}")
    print(f"neutrino signalness: {nu.get('signalness', 'N/A')}")
    print("====================================================")
