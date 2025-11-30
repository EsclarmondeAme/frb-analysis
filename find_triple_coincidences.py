import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

print("=" * 60)
print("triple coincidence search (frb + neutrino + uhecr)")
print("=" * 60)

# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------
def angular_distance(ra1, dec1, ra2, dec2):
    r1, d1 = radians(ra1), radians(dec1)
    r2, d2 = radians(ra2), radians(dec2)
    a = sin((d2 - d1) / 2) ** 2 + cos(d1) * cos(d2) * sin((r2 - r1) / 2) ** 2
    return 2 * atan2(sqrt(a), sqrt(1 - a)) * (180 / np.pi)

# ---------------------------------------------------------
# load data
# ---------------------------------------------------------
print("\nloading data...")
frbs = pd.read_csv("frbs.csv")
neutrinos = pd.read_csv("neutrinos_clean.csv")
uhecr = pd.read_csv("uhecr.csv")

frbs["utc"] = pd.to_datetime(frbs["utc"], format="mixed", utc=True)

neutrinos["utc"] = pd.to_datetime(neutrinos["utc"], format="mixed", utc=True)

uhecr["utc"] = pd.to_datetime(uhecr["utc"], format="mixed", utc=True)


print(f"{len(frbs)} frbs")
print(f"{len(neutrinos)} neutrinos")
print(f"{len(uhecr)} uhecr")

# overlapping time window for all three
start = max(frbs["utc"].min(), neutrinos["utc"].min(), uhecr["utc"].min())
end = min(frbs["utc"].max(), neutrinos["utc"].max(), uhecr["utc"].max())

frb_ov = frbs[(frbs["utc"] >= start) & (frbs["utc"] <= end)]
nu_ov = neutrinos[(neutrinos["utc"] >= start) & (neutrinos["utc"] <= end)]
cr_ov = uhecr[(uhecr["utc"] >= start) & (uhecr["utc"] <= end)]

print("\noverlap window:")
print(start, "→", end)
print(f"frbs: {len(frb_ov)}")
print(f"neutrinos: {len(nu_ov)}")
print(f"uhecr: {len(cr_ov)}")

if len(frb_ov) == 0 or len(nu_ov) == 0 or len(cr_ov) == 0:
    print("\nnot enough overlapping data for triple search")
    print("done")
    print("=" * 60)
    raise SystemExit

# ---------------------------------------------------------
# parameters
# ---------------------------------------------------------
# FRB–neutrino coincidence
base_fn_window = 60.0  # seconds (will be scaled by signalness)
max_ang_fn = 5.0       # degrees

# neutrino–UHECR time lag
base_cr_delay_days = 3.0   # typical allowed lag for ~50 EeV
cr_min_delay_days = 0.05   # 1.2 hours
cr_max_delay_days = 5.0    # upper cap

max_ang_nc = 8.0      # degrees (larger due to magnetic deflection)

background_flux = frb_ov["flux"].mean() if "flux" in frb_ov else 1.0
mean_cr_energy = cr_ov["energy_eev"].mean() if "energy_eev" in cr_ov else 50.0

# ---------------------------------------------------------
# triple search
# ---------------------------------------------------------
triples = []
total_duration_sec = (end - start).total_seconds()

print("\nsearching for triple coincidences...")

for _, frb in frb_ov.iterrows():
    frb_flux = frb.get("flux", 1.0)
    frb_z = frb.get("z_est", np.nan)

    for _, nu in nu_ov.iterrows():
        # dynamic FRB–neutrino window based on signalness
        nu_sig = nu.get("signalness", 0.5)
        energy_factor = max(0.2, min(3.0, 1.0 / (nu_sig + 0.1)))
        fn_window = base_fn_window * energy_factor

        dt_fn = abs((nu["utc"] - frb["utc"]).total_seconds())
        if dt_fn > fn_window:
            continue

        ang_fn = angular_distance(frb["ra"], frb["dec"], nu["ra"], nu["dec"])
        if ang_fn > max_ang_fn:
            continue

        # now search for UHECRs that lag behind the neutrino
        for _, cr in cr_ov.iterrows():
            if cr["utc"] <= nu["utc"]:
                continue

            e_eev = cr.get("energy_eev", mean_cr_energy)

            # energy-dependent allowed delay
            # higher energy → shorter delay window
            delay_scale = (50.0 / max(10.0, e_eev)) ** 0.5
            max_delay_days = base_cr_delay_days * delay_scale
            max_delay_days = max(cr_min_delay_days, min(cr_max_delay_days, max_delay_days))
            max_delay_sec = max_delay_days * 86400.0

            dt_nc = (cr["utc"] - nu["utc"]).total_seconds()
            if dt_nc < 0 or dt_nc > max_delay_sec:
                continue

            # angular separation between neutrino and uhecr
            ang_nc = angular_distance(nu["ra"], nu["dec"], cr["ra"], cr["dec"])
            if ang_nc > max_ang_nc:
                continue

            # optional: also check FRB–UHECR angle
            ang_fc = angular_distance(frb["ra"], frb["dec"], cr["ra"], cr["dec"])

            # --- new physics-based FRB–neutrino coherence score ---
            dt_s = dt_fn
            ang_fn_deg = ang_fn
            nu_E_tev = nu.get("energy_tev", np.nan)

            # weights from your cone model (same as pair version)
            w_E   = coupling_efficiency(nu_E_tev) if np.isfinite(nu_E_tev) else 1.0
            w_t   = time_coherence_weight(dt_s)
            w_ang = angular_coherence_weight(ang_fn_deg)

            base = 0.0

            if np.isfinite(frb_flux):
                base += np.log10(1.0 + max(frb_flux, 0.0))

            if np.isfinite(frb_z):
                base += 0.3 * max(frb_z, 0.0)

            if np.isfinite(nu_sig):
                base += 0.5 * nu_sig

            pair_score = base * w_E * w_t * w_ang

            # --- UHECR components ---
            cr_weight = e_eev / mean_cr_energy
            lag_penalty = np.exp(-dt_nc / max_delay_sec)

            # --- final triple score ---
            triple_score = pair_score * cr_weight * lag_penalty


            triples.append({
                "frb_name": frb["name"],
                "frb_time": frb["utc"],
                "nu_id": nu["event_id"],
                "nu_time": nu["utc"],
                "cr_name": cr["name"],
                "cr_time": cr["utc"],
                "frb_nu_dt_s": dt_fn,
                "nu_cr_dt_s": dt_nc,
                "fn_window_s": fn_window,
                "max_nc_delay_s": max_delay_sec,
                "ang_frb_nu_deg": ang_fn,
                "ang_nu_cr_deg": ang_nc,
                "ang_frb_cr_deg": ang_fc,
                "frb_flux": frb_flux,
                "frb_z": frb_z,
                "nu_signalness": nu_sig,
                "cr_energy_eev": e_eev,
                "triple_score": triple_score,
            })

print("\n" + "=" * 60)
print(f"found {len(triples)} triple coincidences")
print("=" * 60)

if len(triples) == 0:
    print("\nno triple coincidences found")
    print("this does not rule out cross-layer events")
    print("most astrophysical triple coincidences are extremely rare")
else:
    df = pd.DataFrame(triples)
    df.to_csv("triple_coincidences_enhanced.csv", index=False)

    print("\nstrongest triple candidates:")
    print(df.sort_values("triple_score", ascending=False).head(10).to_string(index=False))

    # very rough background expectation
    days = total_duration_sec / 86400.0
    avg_max_delay_sec = df["max_nc_delay_s"].mean()
    ang_frac_fn = (max_ang_fn / 180.0) ** 2
    ang_frac_nc = (max_ang_nc / 180.0) ** 2

    expected_random = (
        len(frb_ov)
        * len(nu_ov)
        * len(cr_ov)
        * (2 * base_fn_window / total_duration_sec)
        * (avg_max_delay_sec / total_duration_sec)
        * ang_frac_fn
        * ang_frac_nc
    )

    print("\napprox expected random triples:", round(expected_random, 4))
    print("observed triples:", len(df))

    if len(df) > expected_random * 3:
        print("\npossible excess beyond simple random background")

    # redshift diagnostic for triples (if frb z_est present)
    if "z_est" in frb_ov.columns:
        all_z = frb_ov["z_est"].replace([np.inf, -np.inf], np.nan).dropna()
        coinc_z = df["frb_z"].replace([np.inf, -np.inf], np.nan).dropna()

        if len(all_z) > 5 and len(coinc_z) > 0:
            print("\nredshift diagnostics (triple coincidences):")
            print(f"  all frbs:    mean z = {all_z.mean():.3f}, std = {all_z.std():.3f}")
            print(f"  triple frbs: mean z = {coinc_z.mean():.3f}, std = {coinc_z.std():.3f}")

            z_for_corr = df["frb_z"].fillna(all_z.mean())
            scores = df["triple_score"]
            corr = np.corrcoef(z_for_corr, scores)[0, 1]
            print(f"  corr(triple_score, z) ≈ {corr:.3f}")

print("\ndone")
print("=" * 60)
