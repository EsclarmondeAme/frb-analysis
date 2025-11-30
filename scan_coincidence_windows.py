import numpy as np
import pandas as pd
from datetime import timedelta

print("============================================================")
print("window scan: frb + neutrino coincidences (real data)")
print("============================================================")

# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------
def angular_sep_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """great-circle separation in degrees"""
    ra1 = np.radians(ra1_deg)
    dec1 = np.radians(dec1_deg)
    ra2 = np.radians(ra2_deg)
    dec2 = np.radians(dec2_deg)

    s = (
        np.sin(dec1) * np.sin(dec2) +
        np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    # numerical safety
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def time_coherence_weight(dt_s, T_window):
    """
    time weight: 1 at dt=0, falls as exp[-(dt/T)^2].
    T_window is the coincidence cutoff for that scan setting.
    """
    x = dt_s / max(T_window, 1.0)
    return float(np.exp(-x * x))


def angular_coherence_weight(theta_deg, theta_window):
    """
    angular weight: 1 at 0°, falls as exp[-(theta/theta_window)^2].
    theta_window is the coincidence cutoff for that scan setting.
    """
    x = theta_deg / max(theta_window, 1e-3)
    return float(np.exp(-x * x))


def pair_score(frb, nu, dt_s, ang_deg, T_window, theta_window):
    """
    physical-ish coincidence score:
      base ~ log10(flux) + 0.3*z_est + 0.5*signalness
      times time + angular coherence weights.
    """
    # try to get reasonable columns, but don't crash if missing
    frb_flux = None
    for key in ["fluence", "flux", "flux_jy", "flux_mjy"]:
        if key in frb.index:
            frb_flux = frb[key]
            break

    frb_z = frb["z_est"] if "z_est" in frb.index else np.nan
    nu_sig = nu["signalness"] if "signalness" in nu.index else np.nan

    base = 0.0

    if frb_flux is not None and np.isfinite(frb_flux):
        base += np.log10(1.0 + max(float(frb_flux), 0.0))

    if np.isfinite(frb_z):
        base += 0.3 * max(float(frb_z), 0.0)

    if np.isfinite(nu_sig):
        base += 0.5 * float(nu_sig)

    if base <= 0.0:
        base = 0.1  # tiny floor so weights still matter

    w_t = time_coherence_weight(dt_s, T_window)
    w_ang = angular_coherence_weight(ang_deg, theta_window)

    return base * w_t * w_ang


# ---------------------------------------------------------
# load data
# ---------------------------------------------------------
frb_file = "frbs.csv"
nu_file = "neutrinos_clean.csv"

frbs = pd.read_csv(frb_file)
neus = pd.read_csv(nu_file)

if "utc" not in frbs.columns or "utc" not in neus.columns:
    raise RuntimeError("both frbs.csv and neutrinos_clean.csv must have a 'utc' column")

frbs["utc"] = pd.to_datetime(frbs["utc"], utc=True, errors="coerce")
neus["utc"] = pd.to_datetime(neus["utc"], utc=True, errors="coerce")

frbs = frbs.dropna(subset=["utc"])
neus = neus.dropna(subset=["utc"])

print(f"loaded {len(frbs)} FRBs")
print(f"loaded {len(neus)} neutrinos")

# basic overlap cut so we don't waste time
overlap_start = max(frbs["utc"].min(), neus["utc"].min())
overlap_end   = min(frbs["utc"].max(), neus["utc"].max())

frbs_win = frbs[(frbs["utc"] >= overlap_start) & (frbs["utc"] <= overlap_end)].copy()
neus_win = neus[(neus["utc"] >= overlap_start) & (neus["utc"] <= overlap_end)].copy()

print("")
print("time overlap range:")
print("  ", overlap_start, "→", overlap_end)
print(f"FRBs in range:      {len(frbs_win)}")
print(f"neutrinos in range: {len(neus_win)}")
print("------------------------------------------------------------")

# require RA/Dec columns
for df, name in [(frbs_win, "FRBs"), (neus_win, "neutrinos")]:
    for col in ["ra", "dec"]:
        if col not in df.columns:
            raise RuntimeError(f"{name} table missing column '{col}'")

# ---------------------------------------------------------
# define windows to scan
# ---------------------------------------------------------
time_windows = [
    30.0,          # 30 s
    300.0,         # 5 min
    1800.0,        # 30 min
    3600.0,        # 1 hour
    21600.0,       # 6 hours
    86400.0        # 1 day
]

angle_windows = [
    5.0,           # 5°
    10.0,          # 10°
    20.0,          # 20°
    40.0           # 40°
]

# results dict: (T, theta) -> {count, max_score}
results = {}
for T in time_windows:
    for th in angle_windows:
        results[(T, th)] = {"count": 0, "max_score": 0.0}

# ---------------------------------------------------------
# brute-force scan
# ---------------------------------------------------------
print("scanning windows...")
pairs_checked = 0

frb_times = frbs_win["utc"].to_numpy()
frb_ra = frbs_win["ra"].to_numpy(dtype=float)
frb_dec = frbs_win["dec"].to_numpy(dtype=float)

nu_times = neus_win["utc"].to_numpy()
nu_ra = neus_win["ra"].to_numpy(dtype=float)
nu_dec = neus_win["dec"].to_numpy(dtype=float)

for i in range(len(frbs_win)):
    frb = frbs_win.iloc[i]
    t_f = frb_times[i]
    ra_f = frb_ra[i]
    dec_f = frb_dec[i]

    for j in range(len(neus_win)):
        nu = neus_win.iloc[j]
        t_n = nu_times[j]
        ra_n = nu_ra[j]
        dec_n = nu_dec[j]

        dt_s = abs((t_f - t_n).total_seconds())
        ang_deg = angular_sep_deg(ra_f, dec_f, ra_n, dec_n)

        pairs_checked += 1

        for T in time_windows:
            if dt_s > T:
                continue
            for th in angle_windows:
                if ang_deg > th:
                    continue

                key = (T, th)
                score = pair_score(frb, nu, dt_s, ang_deg, T, th)

                results[key]["count"] += 1
                if score > results[key]["max_score"]:
                    results[key]["max_score"] = score

print("")
print(f"total pairs checked: {pairs_checked}")
print("============================================================")
print("coincidences by window (FRB + neutrino)")
print("T_window [s]   ang_window [deg]   count   max_score")
print("------------------------------------------------------------")

for T in time_windows:
    for th in angle_windows:
        r = results[(T, th)]
        print(f"{T:10.0f}       {th:10.1f}      {r['count']:5d}   {r['max_score']:10.3f}")

print("============================================================")
print("if ALL counts are 0, even for 1-day and 40° windows,")
print("then with current data there is no obvious FRB–ν excess.")
print("if some windows have a few matches, we can inspect them next.")
print("============================================================")
